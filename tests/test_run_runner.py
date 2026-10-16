import json
import numpy as np
import pandas as pd
import pytest

from multiphy.contract import registry
from multiphy.contract.backend import Availability, BackendSpec, Option, Provides
from multiphy.contract.errors import MissingInputs, ContractViolation
from multiphy.contract.inputs import Input, RawWaveformBundle
from multiphy.contract.provenance import Provenance
from multiphy.contract.results import MatchResult, ScoreBlock, ScoreKind, TrackKind
from multiphy.run import runner, run_dir


class Dataset:
    """Duck-typed benchmark: two recording groups, labels present."""
    def __init__(self, ids=('a-1', 'a-2', 'b-1', 'b-2')):
        self.units = pd.DataFrame({'cellid': ids, 'session_id': [i.split('-')[0] + i[-1] for i in ids], 'su_type': 'SU',
                                   'recording_group': [i[0] for i in ids]})
        self.labels = pd.DataFrame({'cellid': ids, 'partition': 'test', 'source_unit_id': 's', 'well': [i[0] for i in ids],
                                    'source_session': [i[0] for i in ids]})
        self.pairs = pd.DataFrame(columns=['cellid_a', 'cellid_b', 'match_score'])
        self.features = {i: {} for i in ids}
        self.metadata = dict(dataset_id='example')

    def subset(self, ids):
        d = Dataset(tuple(ids)); return d

    def waveform_bundle(self):
        return RawWaveformBundle(tuple(self.units.cellid), np.zeros((len(self.units), 8, 4, 2)), tuple(self.units.session_id),
                                 {s: np.zeros((4, 2)) for s in self.units.session_id}, provenance={'raw_source': 't', 'split_method': 'h'})


class FakeBackend:
    spec = BackendSpec('fake', '1', frozenset({Input.UNITS, Input.FINGERPRINT}), frozenset(),
                       Provides(frozenset({ScoreKind.NONE}), frozenset({TrackKind.COMMON})), (Option('k', 'int', 1),))
    calls = []

    def available(self):
        return Availability(True, '1', None)

    def run(self, inp, options, ctx):
        FakeBackend.calls.append((inp.group, options['k']))
        u = inp.units
        return MatchResult('fake', '1', ctx.run_id, u, pd.DataFrame(columns=['cellid_a', 'cellid_b', 'match_score', 'selected', 'decision']),
                           (), tuple(u.cellid), (), ScoreBlock(ScoreKind.NONE, None), TrackKind.COMMON, 'mutual_margin_best_first',
                           ctx.provenance)


@pytest.fixture
def fake():
    registry.clear_for_tests(); registry.register(FakeBackend()); FakeBackend.calls.clear()
    yield
    registry.clear_for_tests()


def test_missing_inputs_raised_before_run_dir_exists(fake, tmp_path, monkeypatch):
    with pytest.raises(MissingInputs, match='fingerprint'):
        runner.run(Dataset(), 'fake', {}, group_column='recording_group', root=tmp_path)
    assert not list((tmp_path).rglob('run_manifest.json'))


def test_runs_each_group_merges_and_writes_run_dir(fake, tmp_path):
    ds = Dataset()
    out = runner.run(ds, 'fake', {'k': 3}, group_column='recording_group', root=tmp_path,
                     extra_inputs={'fingerprint': np.zeros((2, 2)), 'fingerprint_bin_keys': ('x', 'y')})
    assert sorted(FakeBackend.calls) == [('a', 3), ('b', 3)]
    manifest = json.loads((out / 'run_manifest.json').read_text())
    assert manifest['status'] == 'completed' and manifest['backend'] == 'fake'
    r = run_dir.load_result(out / 'result.json')
    assert r.run_id == out.name and set(r.unmatched) == set(ds.units.cellid)
    assert (out / 'evaluation.json').is_file() and (out / 'METHODS.md').is_file()
    assert r.provenance.get('group_runs') and set(r.provenance.get('group_runs')) == {'a', 'b'}


def test_second_group_failure_leaves_failed_manifest_and_no_result(fake, tmp_path, monkeypatch):
    original = FakeBackend.run
    def flaky(self, inp, options, ctx):
        if inp.group == 'b':
            raise RuntimeError('boom')
        return original(self, inp, options, ctx)
    monkeypatch.setattr(FakeBackend, 'run', flaky)
    with pytest.raises(RuntimeError, match='boom'):
        runner.run(Dataset(), 'fake', {}, group_column='recording_group', root=tmp_path,
                   extra_inputs={'fingerprint': np.zeros((2, 2)), 'fingerprint_bin_keys': ('x', 'y')})
    manifests = list(tmp_path.rglob('run_manifest.json'))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())
    assert m['status'] == 'failed' and 'boom' in m['error']
    assert not (manifests[0].parent / 'result.json').exists()


def test_bad_option_fails_before_data(fake, tmp_path):
    with pytest.raises(ContractViolation, match="'k'"):
        runner.run(Dataset(), 'fake', {'k': 'x'}, group_column='recording_group', root=tmp_path)


def test_real_day_same_date_track_rejected(fake, tmp_path, monkeypatch):
    ds = Dataset(('a-1', 'a-2'))
    ds.units['well'] = 'w'; ds.units['session_date'] = ['2026-01-01', '2026-01-01']
    del ds.labels
    def same_day(self, inp, options, ctx):
        u = inp.units
        return MatchResult('fake', '1', ctx.run_id, u, pd.DataFrame(columns=['cellid_a', 'cellid_b', 'match_score', 'selected', 'decision']),
                           (tuple(u.cellid),), (), (), ScoreBlock(ScoreKind.NONE, None), TrackKind.COMMON, 'x', ctx.provenance)
    monkeypatch.setattr(FakeBackend, 'run', same_day)
    with pytest.raises(ContractViolation, match='one actual day'):
        runner.run(ds, 'fake', {}, group_column='well', root=tmp_path,
                   extra_inputs={'fingerprint': np.zeros((2, 2)), 'fingerprint_bin_keys': ('x', 'y')})


def test_one_bin_archive_fingerprint_is_withheld_not_fatal(fake, tmp_path, monkeypatch):
    original = Dataset.waveform_bundle
    def one_bin(self):
        b = original(self)
        return RawWaveformBundle(b.cellids, b.waveforms, b.session_ids, b.channel_positions, provenance=b.provenance,
                                 peth=np.zeros((len(b.cellids), 1)), peth_bin_keys=('only',))
    monkeypatch.setattr(Dataset, 'waveform_bundle', one_bin)
    inp = runner.build_input(Dataset().subset(['a-1', 'a-2']), 'a', provenance=Provenance())
    assert Input.FINGERPRINT not in inp.available() and 'at least two' in inp.provenance.get('fingerprint_withheld')
    with pytest.raises(MissingInputs, match='fingerprint'):
        runner.run(Dataset(), 'fake', {}, group_column='recording_group', root=tmp_path)


def test_cli_list_and_run(fake, tmp_path, capsys):
    from multiphy.run.__main__ import main
    assert main(['list-backends']) == 0
    assert 'fake' in capsys.readouterr().out
class FakeUnitsBackend(FakeBackend):
    spec = BackendSpec('fake_units', '1', frozenset({Input.UNITS}), frozenset(),
                       Provides(frozenset({ScoreKind.NONE}), frozenset({TrackKind.COMMON})), (Option('k', 'int', 1),))

    def run(self, inp, options, ctx):
        result = FakeBackend.run(self, inp, options, ctx)
        result.backend_id = self.spec.id
        return result


def test_cli_benchmark_defaults_to_test_partition(fake, tmp_path, monkeypatch):
    from multiphy.run import __main__ as cli
    registry.register(FakeUnitsBackend())
    monkeypatch.setattr(cli, '_load_dataset', lambda path: Dataset())
    assert cli.main(['run-backend', '--input', str(tmp_path), '--backend', 'fake_units', '--root', str(tmp_path)]) == 0
    manifests = list(tmp_path.rglob('run_manifest.json'))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())
    assert m['partition'] == 'test' and m['status'] == 'completed'
class FakeWaveformBackend(FakeBackend):
    spec = BackendSpec('fake_waveforms', '1', frozenset({Input.UNITS, Input.WAVEFORM_HALVES}), frozenset(),
                       Provides(frozenset({ScoreKind.NONE}), frozenset({TrackKind.COMMON})), (Option('k', 'int', 1),))


def _misaligned(self):
    raise ValueError('bins misaligned')


def test_bundle_error_is_named_when_waveforms_are_missing(fake, tmp_path, monkeypatch):
    registry.register(FakeWaveformBackend())
    monkeypatch.setattr(Dataset, 'waveform_bundle', _misaligned)
    with pytest.raises(MissingInputs, match='waveform bundle could not be built: bins misaligned'):
        runner.run(Dataset(), 'fake_waveforms', {}, group_column='recording_group', root=tmp_path)
    assert not list(tmp_path.rglob('run_manifest.json'))


def test_bundle_error_is_recorded_for_backends_that_do_not_need_waveforms(fake, monkeypatch):
    monkeypatch.setattr(Dataset, 'waveform_bundle', _misaligned)
    inp = runner.build_input(Dataset().subset(['a-1', 'a-2']), 'a', provenance=Provenance())
    assert Input.WAVEFORM_HALVES not in inp.available()
    assert inp.provenance.get('waveform_bundle_error') == 'bins misaligned'
    monkeypatch.undo()
    clean = runner.build_input(Dataset().subset(['a-1', 'a-2']), 'a', provenance=Provenance())
    assert clean.provenance.get('waveform_bundle_error') is None


def test_missing_inputs_without_a_bundle_error_keep_the_plain_message(fake, tmp_path):
    with pytest.raises(MissingInputs) as info:
        runner.run(Dataset(), 'fake', {}, group_column='recording_group', root=tmp_path)
    assert 'waveform bundle' not in str(info.value)
def test_run_cli_refuses_model_flag(tmp_path, capsys):
    from multiphy.run.__main__ import main
    with pytest.raises(SystemExit) as exc:
        main(['run-backend', '--input', str(tmp_path), '--backend', 'unitmatch', '--model', 'x.json'])
    assert exc.value.code == 2
    assert 'unrecognized arguments: --model x.json' in capsys.readouterr().err


@pytest.mark.parametrize('bundle', [True, False])
def test_build_input_refuses_unknown_extra_inputs(tmp_path, monkeypatch, bundle):
    if not bundle:
        monkeypatch.delattr(Dataset, 'waveform_bundle')
    with pytest.raises(ValueError, match='fitted_model'):
        runner.build_input(Dataset().subset(['a-1', 'a-2']), 'a', provenance=Provenance(),
                           extra_inputs={'fitted_model': str(tmp_path / 'model.json')})


def test_merge_keeps_every_groups_calibration_reference(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a stray relative write lands in tmp_path, never the repo
    def calibrated(group, reference):
        u = Dataset().units
        u = u[u.recording_group == group].reset_index(drop=True)
        return MatchResult('fake', '1', f'run_{group}', u,
                           pd.DataFrame(columns=['cellid_a', 'cellid_b', 'match_score', 'selected', 'decision']),
                           (), tuple(u.cellid), (), ScoreBlock(ScoreKind.CALIBRATED_PROBABILITY, 'match_score',
                                                               calibration_reference=reference),
                           TrackKind.COMMON, 'mutual_margin_best_first', Provenance.from_json({'group': group}))
    merged = runner.merge_results([calibrated('a', "a: trained on ['a0'], calibrated on held-out ['a1']"),
                                   calibrated('b', "b: trained on ['b0'], calibrated on held-out ['b1']")],
                                  'fake', 'out', Provenance())
    assert merged.score.calibration_reference == (
        "a: trained on ['a0'], calibrated on held-out ['a1']; b: trained on ['b0'], calibrated on held-out ['b1']")
    assert merged.score.kind is ScoreKind.CALIBRATED_PROBABILITY and merged.score.column == 'match_score'


@pytest.mark.parametrize('bundle', [True, False])
def test_build_input_adds_n_spikes(monkeypatch, bundle):
    if not bundle:
        monkeypatch.delattr(Dataset, 'waveform_bundle')
    subset = Dataset().subset(['a-1', 'a-2'])
    subset.features['a-1'] = {'n_spikes': 1770}
    inp = runner.build_input(subset, 'a', provenance=Provenance())
    got = inp.units.set_index('cellid').n_spikes
    assert got['a-1'] == 1770.0
    assert np.isnan(got['a-2'])
    assert 'n_spikes' not in subset.units
