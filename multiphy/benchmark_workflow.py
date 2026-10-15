# [AGENT EDIT START | agent: codex | user: wingertj | reason: complete reproducible held-out local and DeepUnitMatch comparison after verified extraction | date: 2026-09-18]
# [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no local ablations (Task 2) | date: 2026-09-27]
"""One-off dependency workflow: collections, published matchers, official training.

This explicitly starts real training. Outputs are cache-only and completed
stages are hash-verified before resume. Failed attempts remain inspectable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import socket
import uuid

from .paths import resolve_cache_path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Workflow:
    def __init__(self, output, config, root=None):
        self.output = resolve_cache_path(output, root=root)
        self.output.mkdir(parents=True, exist_ok=True)
        self._lock = (self.output / 'workflow.lock').open('a+')
        try:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._lock.close()
            self._lock = None
            raise RuntimeError(f'Another process owns this workflow: {self.output}') from exc
        self.path = self.output / 'workflow.json'
        try:
            if self.path.exists():
                self.state = json.loads(self.path.read_text())
                if self.state['config'] != config:
                    raise ValueError('Resume configuration differs from the recorded workflow.')
                uncertain = [name for name, stage in self.state['stages'].items()
                             if stage.get('status') not in ('completed', 'failed')]
                if uncertain:
                    raise RuntimeError('Previous workflow stages have uncertain outcomes: '
                        f'{uncertain}. Verify that their processes have stopped and inspect their '
                        'artifacts before explicitly resolving the recorded stage state; automatic restart refused.')
            else:
                self.state = dict(config=config, stages={}, status='running')
            self.state['owner'] = dict(pid=os.getpid(), host=socket.gethostname(),
                launch_id=uuid.uuid4().hex, acquired_at=datetime.now(timezone.utc).isoformat())
            self.save()
        except BaseException:
            self.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if getattr(self, '_lock', None) is not None:
            fcntl.flock(self._lock.fileno(), fcntl.LOCK_UN)
            self._lock.close()
            self._lock = None

    def __del__(self):
        self.close()

    def save(self):
        if self._lock is None:
            raise RuntimeError('Workflow state cannot change without its lifetime lock')
        self.state['updated_at'] = datetime.now(timezone.utc).isoformat()
        temporary = self.path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.state, indent=2, allow_nan=False))
        temporary.replace(self.path)

    def stage(self, name, action):
        if self._lock is None:
            raise RuntimeError('Workflow stages require an active lifetime lock')
        previous = self.state['stages'].get(name, {})
        if previous.get('status') == 'completed':
            for path, digest in previous['sha256'].items():
                if _digest(path) != digest:
                    raise ValueError(f'Completed workflow artifact changed: {path}')
            print(f'Reusing verified stage: {name}', flush=True)
            return previous['value']
        if previous and previous.get('status') != 'failed':
            raise RuntimeError(f'Stage {name} has an uncertain outcome; automatic retry refused')
        attempt = int(previous.get('attempt', 0)) + 1
        destination = self.output / f'{name}_{attempt:03d}'
        if destination.exists():
            raise FileExistsError(f'Attempt destination already exists; inspect it before resume: {destination}')
        record = dict(status='running', attempt=attempt, destination=str(destination))
        if previous:
            record['history'] = previous.get('history', []) + [
                {key: value for key, value in previous.items() if key != 'history'}]
        self.state['status'] = 'running'
        self.state['stages'][name] = record
        self.save()
        print(f'Starting stage: {name}', flush=True)
        try:
            value, files = action(destination)
            record.update(status='completed', value=value,
                          sha256={str(Path(f).resolve()): _digest(f) for f in files})
            self.save()
            return value
        except BaseException as exc:
            # Interruptions may leave an upstream child alive. Do not infer
            # completion or make a new training attempt until explicitly verified.
            outcome = 'failed' if isinstance(exc, Exception) else 'uncertain'
            record.update(status=outcome, error=f'{type(exc).__name__}: {exc}')
            self.state['status'] = outcome
            self.save()
            raise


def run_workflow(training, calibration, test, missing_test, output, *, source_checkout,
                 training_pid=None, calibration_pid=None, autoencoder_epochs=300,
                 contrastive_epochs=50, seed=0, cpu_threads=4, min_stimulus_repeats=None,
                 checkpoint=None, checkpoint_probe=None, root=None):
    if min_stimulus_repeats is not None and (isinstance(min_stimulus_repeats, bool)
            or not isinstance(min_stimulus_repeats, int) or min_stimulus_repeats < 1):
        raise ValueError('min_stimulus_repeats must be a positive integer')
    if bool(checkpoint) != bool(checkpoint_probe):
        raise ValueError('Supply both an existing checkpoint and its training probe identifier')
    checkpoint = str(Path(checkpoint).resolve()) if checkpoint else None
    paths = [str(Path(p).resolve()) for p in (training, calibration, test, missing_test)]
    config = dict(training=paths[0], calibration=paths[1], test=paths[2], missing_test=paths[3],
        source_checkout=str(Path(source_checkout).resolve()), autoencoder_epochs=autoencoder_epochs,
        contrastive_epochs=contrastive_epochs, seed=seed, cpu_threads=cpu_threads,
        local_threshold=.9, ambiguity_margin=.02, contradiction_threshold=.1,
        min_stimulus_repeats=min_stimulus_repeats, checkpoint=checkpoint,
        checkpoint_probe=checkpoint_probe,
        checkpoint_sha256=_digest(checkpoint) if checkpoint else None)
    with Workflow(output, config, root=root) as flow:
        return _run_workflow_locked(flow, paths, source_checkout=source_checkout,
            training_pid=training_pid, calibration_pid=calibration_pid, root=root)


def _run_workflow_locked(flow, paths, *, source_checkout, training_pid, calibration_pid, root):
    from .benchmark import load_dataset
    from .benchmark_collection import create_collection
    from .dant_settings import default_dant_settings
    from .runners import wait_for_extraction, prepare_deep_dataset, train_prepared, run_backend_dataset
    training, calibration, test, missing_test = paths
    config = flow.state['config']
    (flow.output / 'METHODS.md').write_text(f'''# Held-out matching workflow — methods notes

- **Output folder:** `{flow.output}`
- **Artifacts:** `workflow.json`, referenced collection/run/training folders
- **Generated by:** `nems_lbhb/unit_match/benchmark_workflow.py:run_workflow()` · **Updated:** {datetime.now(timezone.utc).date()}

## Goal

Fit an official DeepUnitMatch network on independent training wells. Evaluate complete and missing-counterpart test populations using fixed settings; no test-set threshold tuning occurs.

## Data lineage

- Collections ← `benchmark_collection.create_collection()`: hash-pinned existing train/calibration/test archives, no duplicated waveform storage. Each source retains its original partition.
- Network/checkpoints ← `runners.prepare_deep_dataset()/train_prepared()`: only training observations enter official two-stage fitting; preprocessing, source revision, sample rate, geometry and final checkpoint hashes appear in the training manifest.
- DeepUnitMatch results ← `runners.run_backend_dataset()`: explicit final trained checkpoint, native tracking, test observations only. Per-stage METHODS describe transformations and limitations.
- Published comparisons ← the same runner on the exact collections: UnitMatch native and common constrained tracking; DANT waveform/spike-timing and stimulus-fingerprint presets. Every result can be imported in the same curator.
- `workflow.json` ← `Workflow.stage()`: stage status, owner identity, attempts and artifact hashes. A lifetime interprocess lock prevents concurrent launches. Completed outputs are verified on resume; failed attempts are retained. Interrupted or abandoned running stages require explicit outcome verification before retry. Each retry prepares and trains into a fresh destination.

## Panels

No figures. Evaluation files record precision, recovery, erroneous identity links and unmatched decisions. Training and calibration labels never cross into the test feature inputs.

## Caveats

Reference identities depend on the original sort. Small training/calibration populations and same-recording temporal splits do not establish daily reinsertion accuracy. Published network augmentation and other upstream assumptions are explicitly recorded in its METHODS.

Minimum stimulus repeats override: {config['min_stimulus_repeats']}; None preserves archived candidate evidence. An explicit override rebuilds candidates from saved measured features without changing waveform archives. Repeat counts and reliability remain available; no missing sound response is invented. Existing checkpoint: {config['checkpoint'] or 'train a new model'}; an existing model is hash-pinned and its source/probe/preprocessing compatibility is checked again during inference.
''')
    dependencies = [(training_pid, training), (calibration_pid, calibration)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(wait_for_extraction, pid, path) for pid, path in dependencies
                   if pid is not None and not (Path(path) / 'manifest.json').exists()]
        for future in futures:
            future.result()
    for path, partition in zip(paths, ('train', 'calibration', 'test', 'test')):
        ds = load_dataset(path)
        if set(ds.labels.partition) != {partition}:
            raise ValueError(f'Expected only {partition} observations in {path}')
    hashes = {p: _digest(Path(p) / 'manifest.json') for p in paths}
    if flow.state.get('input_hashes', hashes) != hashes:
        raise ValueError('Workflow source manifests changed since the previous attempt.')
    flow.state['input_hashes'] = hashes
    flow.save()
    collections = {}
    for label, source_test in (('complete', test), ('missing', missing_test)):
        def make(destination, label=label, source_test=source_test):
            result = create_collection([training, calibration, source_test], destination,
                dataset_id=f'{flow.output.name}_{label}', root=root,
                pair_options=({'min_repeats': config['min_stimulus_repeats']}
                              if config['min_stimulus_repeats'] is not None else None))
            return str(result), [result / 'manifest.json']
        collections[label] = flow.stage(f'collection_{label}', make)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no local fixed-model stages or their evidence preflight (Task 2) | date: 2026-09-27]
    for label, path in collections.items():
    # [AGENT EDIT END]
        presets = [('unitmatch_native', 'unitmatch', {'tracking': 'native_intermediate'}),
                   ('unitmatch_constrained', 'unitmatch', {'tracking': 'common_constrained'}),
                   ('dant_waveform', 'dant', {'settings': default_dant_settings('waveform_acg')}),
                   ('dant_stimulus', 'dant', {'settings': default_dant_settings('waveform_acg_stimulus')})]
        for name, backend, options in presets:
            def published(destination, path=path, backend=backend, options=options):
                result = run_backend_dataset(path, backend, output_dir=destination,
                                             partition='test', root=root, **options)
                return str(result), [result / f for f in ('run_manifest.json', 'result.json', 'evaluation.json')]
            flow.stage(f'{name}_{label}', published)
    def train(destination):
        if config['checkpoint']:
            return dict(checkpoint=config['checkpoint'], probe=config['checkpoint_probe']), [config['checkpoint']]
        result = prepare_deep_dataset(collections['complete'], source_checkout=source_checkout,
            output_dir=destination, autoencoder_epochs=config['autoencoder_epochs'],
            contrastive_epochs=config['contrastive_epochs'], seed=config['seed'],
            cpu_threads=config['cpu_threads'], root=root)
        recipe = train_prepared(result, root=root)
        if recipe.get('status') != 'completed':
            raise RuntimeError('DeepUnitMatch training did not report completed status')
        return dict(checkpoint=recipe['checkpoint'], probe=recipe['probe_type']), [
            Path(result) / 'training_waveforms.npz', Path(result) / 'training_manifest.json', recipe['checkpoint']]
    model = flow.stage('deep_train', train)
    for label, path in collections.items():
        for policy in ('native_intermediate', 'common_constrained'):
            def match(destination, path=path, policy=policy):
                result = run_backend_dataset(path, 'deepunitmatch', output_dir=destination,
                    checkpoint=model['checkpoint'], checkpoint_probe=model['probe'],
                    tracking=policy, partition='test', root=root)
                return str(result), [result / f for f in ('run_manifest.json', 'result.json', 'evaluation.json')]
            flow.stage(f'deep_{label}_{policy}', match)
    flow.state['status'] = 'completed'
    flow.save()
    print(f'Workflow completed: {flow.output}', flush=True)
    return flow.output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for field in ('training', 'calibration', 'test', 'missing-test', 'output', 'source-checkout'):
        parser.add_argument('--' + field, required=True)
    for field in ('training-pid', 'calibration-pid'):
        parser.add_argument('--' + field, type=int)
    parser.add_argument('--autoencoder-epochs', type=int, default=300)
    parser.add_argument('--contrastive-epochs', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--min-stimulus-repeats', type=int,
                        help='Explicitly rebuild pair evidence from saved features at this repeat minimum')
    parser.add_argument('--checkpoint', help='Reuse an existing trained network instead of training again')
    parser.add_argument('--checkpoint-probe', help='Actual probe identifier used to train that checkpoint')
    run_workflow(**vars(parser.parse_args(argv)))


if __name__ == '__main__':
    main()
# [AGENT EDIT END]
