# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify safe workflow resume and failed-attempt preservation | date: 2026-09-18]
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from multiphy.benchmark_workflow import Workflow, run_workflow


def test_resume_verifies_completed_artifacts_and_rejects_changed_config(tmp_path):
    flow = Workflow(tmp_path / 'flow', {'seed': 0}, root=tmp_path)
    def create(destination):
        destination.mkdir()
        artifact = destination / 'result.json'
        artifact.write_text('{"verified":true}')
        return str(destination), [artifact]
    first = flow.stage('test', create)
    flow.close()
    resumed = Workflow(tmp_path / 'flow', {'seed': 0}, root=tmp_path)
    def unexpected(_):
        raise AssertionError('Completed stage must not rerun')
    assert resumed.stage('test', unexpected) == first
    resumed.close()
    with pytest.raises(ValueError, match='configuration'):
        Workflow(tmp_path / 'flow', {'seed': 1}, root=tmp_path)
    (Path(first) / 'result.json').write_text('changed')
    with Workflow(tmp_path / 'flow', {'seed': 0}, root=tmp_path) as resumed:
        with pytest.raises(ValueError, match='changed'):
            resumed.stage('test', unexpected)


def test_failed_stage_uses_new_attempt_directory_on_resume(tmp_path):
    flow = Workflow(tmp_path / 'flow', {}, root=tmp_path)
    def failed(destination):
        destination.mkdir()
        (destination/'partial').write_text('diagnostic')
        raise RuntimeError('upstream failed')
    with pytest.raises(RuntimeError, match='upstream'):
        flow.stage('model', failed)
    assert flow.state['status'] == 'failed'
    def success(destination):
        assert destination.name == 'model_002'
        destination.mkdir()
        f = destination / 'model'; f.write_text('done')
        return str(f), [f]
    flow.stage('model', success)
    assert (flow.output/'model_001'/'partial').read_text() == 'diagnostic'
    assert flow.state['stages']['model']['status'] == 'completed'
    assert flow.state['stages']['model']['history'][0]['status'] == 'failed'
    flow.close()


def test_lifetime_lock_blocks_another_process_and_releases_on_context_exit(tmp_path):
    output = tmp_path / 'flow'
    program = ('import fcntl,sys; f=open(sys.argv[1], "a+"); '
               'fcntl.flock(f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)')
    with Workflow(output, {}, root=tmp_path):
        with pytest.raises(RuntimeError, match='Another process'):
            Workflow(output, {}, root=tmp_path)
        child = subprocess.run([sys.executable, '-c', program, str(output / 'workflow.lock')],
                               capture_output=True, text=True)
        assert child.returncode != 0 and 'BlockingIOError' in child.stderr
    subprocess.run([sys.executable, '-c', program, str(output / 'workflow.lock')], check=True)


@pytest.mark.parametrize('status', ['running', 'uncertain', 'unknown'])
def test_owner_disappearance_does_not_restart_unverified_stage(tmp_path, status):
    output = tmp_path / 'flow'
    with Workflow(output, {}, root=tmp_path) as flow:
        flow.state['stages']['deep_train'] = dict(status=status, attempt=1, destination='preserved')
        flow.save()
    before = (output / 'workflow.json').read_bytes()
    with pytest.raises(RuntimeError, match='uncertain outcomes'):
        Workflow(output, {}, root=tmp_path)
    assert (output / 'workflow.json').read_bytes() == before


def test_interrupt_marks_uncertain_and_prohibits_same_owner_retry(tmp_path):
    with Workflow(tmp_path / 'flow', {}, root=tmp_path) as flow:
        def interrupted(destination):
            destination.mkdir()
            raise KeyboardInterrupt
        with pytest.raises(KeyboardInterrupt):
            flow.stage('deep_train', interrupted)
        assert flow.state['stages']['deep_train']['status'] == 'uncertain'
        with pytest.raises(RuntimeError, match='uncertain'):
            flow.stage('deep_train', lambda _: pytest.fail('must not restart'))


def mock_pipeline(monkeypatch, tmp_path, fail_training=False, bad_partition=False):
    from multiphy import benchmark, benchmark_collection, runners  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: benchmark_local is gone (Task 2) | date: 2026-09-27]
    paths = [tmp_path / name for name in ('training', 'calibration', 'test', 'missing')]
    calls, collections = [], {}
    for i, path in enumerate(paths):
        path.mkdir()
        if i >= 2:
            (path / 'manifest.json').write_text('{}')
    def wait(pid, path):
        calls.append(('wait', pid, Path(path)))
        (Path(path) / 'manifest.json').write_text('{}')
    def load(path):
        assert (Path(path) / 'manifest.json').is_file()
        if str(path) in collections:
            return SimpleNamespace(collection=str(path))
        index = paths.index(Path(path))
        partition = ('train', 'calibration', 'test', 'test')[index]
        if bad_partition and index == 1:
            partition = 'train'
        return SimpleNamespace(labels=pd.DataFrame({'partition': [partition]}))
    def collection(inputs, destination, **kwargs):
        assert list(map(Path, inputs))[:2] == paths[:2]
        assert Path(inputs[2]) in paths[2:]
        destination.mkdir()
        (destination / 'manifest.json').write_text('{}')
        collections[str(destination)] = list(map(Path, inputs))
        calls.append(('collection', destination, kwargs.get('pair_options')))
        return destination
    def prepare(path, output_dir, **kwargs):
        assert collections[str(path)] == [paths[0], paths[1], paths[2]]
        assert kwargs['autoencoder_epochs'] == 300 and kwargs['contrastive_epochs'] == 50
        output_dir.mkdir()
        (output_dir / 'training_waveforms.npz').write_bytes(b'synthetic placeholder')
        (output_dir / 'training_manifest.json').write_text('{"status":"prepared_not_started"}')
        calls.append(('prepare', output_dir))
        return output_dir
    failed = False
    def train(path, **kwargs):
        nonlocal failed
        path = Path(path)
        assert json.loads((path / 'training_manifest.json').read_text())['status'] == 'prepared_not_started'
        calls.append(('train', path))
        if fail_training and not failed:
            failed = True
            (path / 'training_manifest.json').write_text('{"status":"failed"}')
            raise RuntimeError('synthetic upstream training failure')
        checkpoint = path / 'checkpoint'
        checkpoint.write_text('synthetic model')
        recipe = dict(status='completed', checkpoint=str(checkpoint), probe_type='NP1010')
        (path / 'training_manifest.json').write_text(json.dumps(recipe))
        return recipe
    def match(path, backend, output_dir, partition, checkpoint=None, checkpoint_probe=None, **kwargs):
        assert partition == 'test'
        if backend == 'deepunitmatch':
            assert checkpoint_probe == 'NP1010' and Path(checkpoint).is_file()
        elif backend == 'unitmatch':
            assert kwargs['tracking'] in ('native_intermediate', 'common_constrained')
        else:
            assert backend == 'dant' and 'clustering' in kwargs['settings']
        output_dir.mkdir()
        for name in ('run_manifest.json', 'result.json', 'evaluation.json'):
            (output_dir / name).write_text('{}')
        calls.append(('match', Path(path), output_dir, backend))
        return output_dir
    monkeypatch.setattr(runners, 'wait_for_extraction', wait)
    monkeypatch.setattr(benchmark, 'load_dataset', load)
    monkeypatch.setattr(benchmark_collection, 'create_collection', collection)
    monkeypatch.setattr(runners, 'prepare_deep_dataset', prepare)
    monkeypatch.setattr(runners, 'train_prepared', train)
    monkeypatch.setattr(runners, 'run_backend_dataset', match)
    kwargs = dict(training=paths[0], calibration=paths[1], test=paths[2], missing_test=paths[3],
        output=tmp_path / 'workflow', source_checkout=tmp_path / 'official',
        training_pid=123, calibration_pid=456, root=tmp_path)
    return calls, kwargs


def test_mocked_complete_workflow_waits_checks_partitions_and_reuses_all_stages(monkeypatch, tmp_path):
    calls, kwargs = mock_pipeline(monkeypatch, tmp_path)
    output = run_workflow(**kwargs)
    state = json.loads((output / 'workflow.json').read_text())
    assert state['status'] == 'completed'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no local_* stages or preflight (Task 2) | date: 2026-09-27]
    assert len(state['stages']) == 15
    assert [c[0] for c in calls].count('wait') == 2
    assert not any(name.startswith('local_') for name in state['stages'])
    # [AGENT EDIT END]
    assert [c[0] for c in calls].count('prepare') == 1
    assert [c[0] for c in calls].count('train') == 1
    assert [c[0] for c in calls].count('match') == 12
    count = len(calls)
    assert run_workflow(**kwargs) == output
    assert len(calls) == count  # No stage reruns.  [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no evidence preflight (Task 2) | date: 2026-09-27]


def test_training_failure_resume_restages_fresh_data_and_preserves_failed_attempt(monkeypatch, tmp_path):
    calls, kwargs = mock_pipeline(monkeypatch, tmp_path, fail_training=True)
    with pytest.raises(RuntimeError, match='synthetic upstream'):
        run_workflow(**kwargs)
    failed = kwargs['output'] / 'deep_train_001'
    before = (failed / 'training_manifest.json').read_bytes()
    output = run_workflow(**kwargs)
    assert (failed / 'training_manifest.json').read_bytes() == before
    state = json.loads((output / 'workflow.json').read_text())
    assert state['stages']['deep_train']['attempt'] == 2
    assert state['stages']['deep_train']['history'][0]['status'] == 'failed'
    assert [c[0] for c in calls].count('prepare') == 2  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no local stages (Task 2) | date: 2026-09-27]
    assert [c[0] for c in calls].count('train') == 2
    assert Path(state['stages']['deep_train']['value']['checkpoint']).parent.name == 'deep_train_002'


def test_wrong_calibration_partition_stops_before_any_fit(monkeypatch, tmp_path):
    calls, kwargs = mock_pipeline(monkeypatch, tmp_path, bad_partition=True)
    with pytest.raises(ValueError, match='Expected only calibration'):
        run_workflow(**kwargs)
    assert all(c[0] == 'wait' for c in calls)


def test_explicit_repeat_override_and_existing_checkpoint_skip_new_training(monkeypatch, tmp_path):
    calls, kwargs = mock_pipeline(monkeypatch, tmp_path)
    checkpoint = tmp_path / 'existing-model'
    checkpoint.write_text('verified by native inference boundary')
    run_workflow(**kwargs, min_stimulus_repeats=1,
                 checkpoint=checkpoint, checkpoint_probe='NP1010')
    assert all(c[2] == {'min_repeats': 1} for c in calls if c[0] == 'collection')
    assert not any(c[0] in ('prepare', 'train') for c in calls)
    assert [c[0] for c in calls].count('match') == 12
# [AGENT EDIT END]
