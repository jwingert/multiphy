import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

from multiphy import runners
from multiphy.backends import deepunitmatch
from multiphy.contract.inputs import RawWaveformBundle
from multiphy.run import config
from multiphy.backends import BackendUnavailable, load_result, result_from_pairs
from multiphy.backends import unitmatch as unitmatch_backend
from multiphy.contract.backend import Availability


def _patch_unitmatch(monkeypatch, fn):
    monkeypatch.setattr(unitmatch_backend.UnitMatchBackend, 'available', lambda self: Availability(True, 'mock', None))
    monkeypatch.setattr(unitmatch_backend, 'run_unitmatch', fn)


class Dataset:
    def __init__(self, ids=('a', 'b', 'c', 'd'), source_groups=None):
        sessions = dict(a='train0', b='train1', c='test0', d='test1')
        self.units = pd.DataFrame({'cellid': ids, 'session_id': [sessions[u] for u in ids], 'su_type': 'SU',
            'recording_group': ['original_train' if u in 'ab' else 'original_test' for u in ids]})
        self.labels = pd.DataFrame({'cellid': ids, 'partition': ['train' if u in 'ab' else 'test' for u in ids],
            'source_unit_id': 'source0',
            'source_session': ['original_train' if u in 'ab' else 'original_test' for u in ids],
            'well': ['well1' if u in 'ab' else 'well2' for u in ids]})
        self.pairs = pd.DataFrame(columns=['cellid_a', 'cellid_b', 'match_score'])
        self.metadata = dict(dataset_id='example', block_id='well2', cache_built='fixture')
        self.source_groups = source_groups or dict(train0='well1', train1='well1', test0='well2', test1='well2')

    def subset(self, ids):
        return Dataset(tuple(ids), self.source_groups)

    def waveform_bundle(self):
        return RawWaveformBundle(tuple(self.units.cellid),
            np.zeros((len(self.units), 82, 4, 2)), tuple(self.units.session_id),
            {s: np.zeros((4, 2)) for s in self.units.session_id},
            provenance={'source_groups': self.source_groups})


def test_cache_paths_reject_traversal_and_symlink_escape(tmp_path):
    root = tmp_path / 'cache'
    root.mkdir()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (root / 'escape').symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match='stay under'):
        config.resolve_cache_path('escape/result', root=root)
    with pytest.raises(ValueError, match='directory name'):
        config.new_run_path('../repo', root=root)
    first, second = config.new_run_path('dataset', root=root), config.new_run_path('dataset', root=root)
    assert first != second
    assert first.parent == root / 'benchmarks/dataset/runs'
    assert not first.exists()


def test_runtime_temporary_files_stay_under_cache_and_restore_environment(tmp_path):
    old = os.environ.get('TMPDIR')
    old_temp = tempfile.tempdir
    old_bytecode = sys.dont_write_bytecode
    with runners._cache_runtime(tmp_path) as runtime:
        assert runtime.is_relative_to(tmp_path)
        with tempfile.TemporaryDirectory() as child:
            assert Path(child).is_relative_to(runtime)
        assert os.environ['TORCH_HOME'].startswith(str(runtime))
        assert sys.dont_write_bytecode
    assert not runtime.exists()
    assert tempfile.tempdir == old_temp
    assert sys.dont_write_bytecode == old_bytecode
    assert os.environ.get('TMPDIR') == old


def test_run_backend_defaults_to_test_partition_and_saves_portable_result(tmp_path, monkeypatch):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'manifest.json').write_text('{"fixture": "mock dataset"}')
    monkeypatch.setattr(runners, '_load_dataset', lambda _: Dataset())
    seen = {}
    def run(units, pairs, **kwargs):
        seen.update(kwargs)
        assert units.cellid.tolist() == ['c', 'd']
        evidence = pd.DataFrame({'cellid_a': ['c'], 'cellid_b': ['d'], 'match_score': [.99]})
        return result_from_pairs('unitmatch', 'mock-only', units, evidence,
            provenance=kwargs['provenance'], run_id=kwargs['run_id'])
    _patch_unitmatch(monkeypatch, run)
    output = runners.run_backend_dataset(tmp_path / 'input', 'unitmatch', root=tmp_path)
    assert output.parent == tmp_path / 'benchmarks/example/runs'
    assert seen['native_threshold'] is None
    assert seen['tracking'] == 'native_intermediate'
    assert seen['provenance']['partition'] == 'test'
    result = load_result(output / 'result.json')
    assert result.tracks == (('c', 'd'),)
    assert result.provenance.get('cache_built') == 'fixture'
    assert len(result.provenance.get('input_manifest_sha256')) == 64
    assert json.loads((output / 'run_manifest.json').read_text())['status'] == 'completed'
    assert json.loads((output / 'evaluation.json').read_text())['combined']['true_positive_pairs'] == 1
    assert '## Data lineage' in (output / 'METHODS.md').read_text()


def test_failed_backend_leaves_failure_manifest_without_fake_results(tmp_path, monkeypatch):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'manifest.json').write_text('{"fixture": "mock dataset"}')
    monkeypatch.setattr(runners, '_load_dataset', lambda _: Dataset())
    def unavailable(*args, **kwargs):
        raise BackendUnavailable('official package missing')
    _patch_unitmatch(monkeypatch, unavailable)
    output = tmp_path / 'failed'
    with pytest.raises(BackendUnavailable):
        runners.run_backend_dataset(tmp_path / 'input', 'unitmatch', output_dir=output, root=tmp_path)
    assert not (output / 'result.json').exists()
    assert json.loads((output / 'run_manifest.json').read_text())['status'] == 'failed'
    with pytest.raises(FileExistsError):
        runners.run_backend_dataset(tmp_path / 'input', 'unitmatch', output_dir=output, root=tmp_path)


def test_matching_never_pools_independent_original_recordings(tmp_path, monkeypatch):
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'manifest.json').write_text('{"fixture": "mock dataset"}')
    monkeypatch.setattr(runners, '_load_dataset', lambda _: Dataset())
    groups = []
    def run(units, pairs, **kwargs):
        assert units.recording_group.nunique() == 1
        groups.append(units.recording_group.iloc[0])
        a, b = units.cellid.tolist()
        return result_from_pairs('unitmatch', 'mock-only', units,
            pd.DataFrame({'cellid_a': [a], 'cellid_b': [b], 'match_score': [.99]}),
            provenance=kwargs['provenance'], run_id=kwargs['run_id'])
    _patch_unitmatch(monkeypatch, run)
    output = runners.run_backend_dataset(tmp_path / 'input', 'unitmatch', partition='all', root=tmp_path)
    assert set(groups) == {'original_test', 'original_train'}
    result = load_result(output / 'result.json')
    assert set(result.tracks) == {('a', 'b'), ('c', 'd')}
    assert result.provenance.get('independent_groups')
    assert len(result.provenance.get('group_runs')) == 2
    assert result.provenance.get('group') is None
    evaluation = json.loads((output / 'evaluation.json').read_text())
    assert evaluation['combined']['n_unknown_predicted_pairs'] == 0
    assert len(evaluation['groups']) == 2


def test_prepare_selects_only_training_sources_and_does_not_train(tmp_path, monkeypatch):
    monkeypatch.setattr(runners, '_load_dataset', lambda _: Dataset())
    seen = {}
    def prepare(units, bundle, **kwargs):
        seen.update(kwargs)
        assert units.cellid.tolist() == ['a', 'b']
        assert bundle.cellids == ('a', 'b')
        return kwargs['output_dir']
    monkeypatch.setattr(deepunitmatch, 'prepare_deepunitmatch_training', prepare)
    monkeypatch.setattr(deepunitmatch, 'train_deepunitmatch',
                        lambda *a, **k: pytest.fail('preparation must not launch training'))
    output = runners.prepare_deep_dataset(tmp_path / 'input', source_checkout=tmp_path / 'source', root=tmp_path)
    assert output.parent == tmp_path / 'benchmarks/example/training'
    assert seen['training_groups'] == {'well1'}
    assert seen['evaluation_groups'] == {'well2'}
    assert seen['autoencoder_epochs'] == 300
    assert seen['contrastive_epochs'] == 50


def test_training_rejects_source_group_leakage_across_temporal_pieces(tmp_path, monkeypatch):
    groups = dict(train0='well1', train1='well1', test0='well1', test1='well1')
    monkeypatch.setattr(runners, '_load_dataset', lambda _: Dataset(source_groups=groups))
    with pytest.raises(ValueError, match='share original source groups'):
        runners.prepare_deep_dataset(tmp_path / 'input', source_checkout=tmp_path, root=tmp_path)
    assert not (tmp_path / 'benchmarks').exists()


def test_training_refuses_prepared_directory_outside_cache(tmp_path):
    with pytest.raises(ValueError, match='stay under'):
        runners.train_prepared(tmp_path / 'code/training', root=tmp_path / 'cache')


def test_parser_exposes_explicit_actions_and_faithful_native_defaults():
    parser = runners.build_parser()
    args = parser.parse_args(['run-backend', '--input', '/cache/data', '--backend', 'unitmatch'])
    assert args.partition == 'test'
    assert args.native_threshold is None
    assert args.threshold == .9
    args = parser.parse_args(['prepare-deep', '--input', '/cache/data', '--source-checkout', '/source'])
    assert args.command == 'prepare-deep'
    assert args.autoencoder_epochs == 300
    assert args.contrastive_epochs == 50


def test_runner_reads_real_persisted_benchmark_contract(tmp_path, monkeypatch):
    from multiphy.benchmark import BenchmarkDataset, save_dataset
    fixture = Dataset()
    units = fixture.units.assign(siteid=fixture.units.session_id,
        recording_group=['original_train', 'original_train', 'original_test', 'original_test'])
    labels = fixture.labels.assign(source_unit_id='source0',
        source_session=['original_train', 'original_train', 'original_test', 'original_test'],
        well=['animal:well1', 'animal:well1', 'animal:well2', 'animal:well2'])
    features = {u: dict(waveform_halves=np.zeros((82, 4, 2)), fingerprints={},
        spike_samples=np.array([100, 200, 300, 400]), start_sample=0, sample_rate=30000,
        half_waveform_spike_samples=[np.array([100, 200]), np.array([300, 400])])
        for u in units.cellid}
    dataset = BenchmarkDataset(units, fixture.pairs, features, labels,
        {s: np.zeros((4, 2)) for s in units.session_id},
        dict(fixture.metadata, sources=[{'raw_source': 'synthetic integration fixture'}]))
    input_path = tmp_path / 'benchmarks/example/input'
    save_dataset(dataset, input_path, cache_root=tmp_path)
    def run(units, pairs, **kwargs):
        assert kwargs['waveform_bundle'].cellids == ('c', 'd')
        assert set(kwargs['waveform_bundle'].provenance['source_groups'].values()) == {'animal:well2'}
        return result_from_pairs('unitmatch', 'mock-only', units,
            pd.DataFrame({'cellid_a': ['c'], 'cellid_b': ['d'], 'match_score': [.99]}),
            provenance=kwargs['provenance'], run_id=kwargs['run_id'])
    _patch_unitmatch(monkeypatch, run)
    output = runners.run_backend_dataset(input_path, 'unitmatch', root=tmp_path)
    assert load_result(output / 'result.json').tracks == (('c', 'd'),)
def test_one_off_extraction_dependency_requires_completed_dataset(tmp_path, monkeypatch):
    closed = []
    monkeypatch.setattr(runners.os, 'pidfd_open', lambda pid: 99, raising=False)
    monkeypatch.setattr(runners.os, 'close', closed.append)
    readiness = iter([([], [], []), ([99], [], [])])
    monkeypatch.setattr(runners.select, 'select', lambda *a: next(readiness))
    with pytest.raises(RuntimeError, match='without a complete'):
        runners.wait_for_extraction(123456, tmp_path)
    assert closed == [99]
    def already_finished(pid):
        raise ProcessLookupError
    monkeypatch.setattr(runners.os, 'pidfd_open', already_finished)
    (tmp_path / 'manifest.json').write_text('{}')
    runners.wait_for_extraction(123456, tmp_path)
    with pytest.raises(ValueError, match='separate'):
        runners.wait_for_extraction(runners.os.getpid(), tmp_path)


def test_dependency_wait_without_pidfd_rejects_pid_reuse(tmp_path, monkeypatch):
    monkeypatch.delattr(runners.os, 'pidfd_open', raising=False)
    monkeypatch.setattr(runners.sys, 'platform', 'linux')
    identities = iter(['start1', 'start1', 'start2'])
    monkeypatch.setattr(runners, '_process_identity', lambda pid: next(identities))
    sleeps = []
    monkeypatch.setattr(runners.time, 'sleep', sleeps.append)
    (tmp_path / 'manifest.json').write_text('{}')
    runners.wait_for_extraction(123456, tmp_path)
    assert sleeps == [10]
def test_run_backend_dataset_refuses_an_undeclared_option(tmp_path, monkeypatch):
    from multiphy.contract.errors import ContractViolation
    (tmp_path / 'input').mkdir()
    (tmp_path / 'input' / 'manifest.json').write_text('{"fixture": "mock dataset"}')
    monkeypatch.setattr(runners, '_load_dataset', lambda _: Dataset())
    _patch_unitmatch(monkeypatch, lambda *a, **k: pytest.fail('a typo must fail before the backend runs'))
    with pytest.raises(ContractViolation, match='treshold'):
        runners.run_backend_dataset(tmp_path / 'input', 'unitmatch', root=tmp_path, treshold=.8)
    assert not list(tmp_path.rglob('run_manifest.json'))


def test_legacy_cli_forwards_only_declared_or_explicit_shared_flags(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(runners, 'run_backend_dataset', lambda *a, **k: seen.append(k) or tmp_path)
    runners.main(['run-backend', '--input', str(tmp_path), '--backend', 'dant'])
    assert not {'tracking', 'threshold', 'device', 'contradiction_threshold', 'ambiguity_margin'} & set(seen[-1])
    runners.main(['run-backend', '--input', str(tmp_path), '--backend', 'unitmatch'])
    assert seen[-1]['tracking'] == 'native_intermediate' and 'device' not in seen[-1]
    runners.main(['run-backend', '--input', str(tmp_path), '--backend', 'unitmatch', '--device', 'cuda'])
    assert seen[-1]['device'] == 'cuda'  # explicit, so run_backend_dataset refuses it rather than dropping it
