# [AGENT EDIT START | agent: codex | user: wingertj | reason: executable optional matching and training runners with cache-only output | date: 2026-09-17]
"""Execute persisted benchmark inputs without writing data into the code tree.

Examples (all output defaults to /auto/data/nems_db/unit_match):
    python -m nems_lbhb.unit_match.runners run-backend --input DATASET --backend unitmatch
    python -m nems_lbhb.unit_match.runners prepare-deep --input DATASET --source-checkout CHECKOUT
    python -m nems_lbhb.unit_match.runners train-deep --prepared PREPARED_DIRECTORY

Imports of official numerical implementations remain lazy. These commands never
install dependencies. Only the explicit train-deep command starts training.
"""
import argparse
import hashlib
import os
from pathlib import Path
import select
import sys
import time

from . import paths


def _load_dataset(path):
    from .benchmark import load_dataset
    return load_dataset(path)


# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: run-backend is a thin wrapper over run.runner; helpers moved to run/ and kept here as aliases | date: 2026-09-22]
from .run.run_dir import read_json_object as _json_object, write_json as _write_json  # noqa: E402
from .run.runner import _select_partition, cache_runtime as _cache_runtime  # noqa: E402


# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: pass every option on so a typo fails validation; drop only None and consumed input names (whole-branch review fix 6) | date: 2026-09-23]
INPUT_OPTIONS = frozenset({'checkpoint', 'checkpoint_probe'})  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no model input option (Task 2) | date: 2026-09-27]


def backend_options(backend, options):
    """Options for ``runner.run``: ``None`` values and input-carrying names the backend does not
    declare (already turned into inputs) are dropped; anything else is passed on, so a name the
    backend does not take raises ContractViolation instead of vanishing."""
    from .contract import registry
    declared = {o.name for o in registry.get(backend).spec.options}
    return {k: v for k, v in options.items() if v is not None and (k in declared or k not in INPUT_OPTIONS)}


def cli_shared_flags(backend, values, defaults):
    """The old CLIs share one flag set across backends: forward a flag the backend declares, or
    one the user set away from its default (which the backend then refuses if it does not take it)."""
    from .contract import registry
    declared = {o.name for o in registry.get(backend).spec.options}
    return {k: v for k, v in values.items() if k in declared or v != defaults.get(k)}
# [AGENT EDIT END]


def run_backend_dataset(input_path, backend, *, partition='test', output_dir=None, root=None, **options):
    """Run one registered backend on a persisted benchmark through ``run.runner.run``.

    ``options`` are the backend's declared options (see ``--help`` of
    ``python -m nems_lbhb.unit_match.run run-backend --backend ID``); ``None`` values are
    dropped and a name the backend does not declare raises ContractViolation.
    ``checkpoint``/``checkpoint_probe`` also become the input's CheckpointRef, which is what
    DeepUnitMatch reads.
    """
    from .run import runner
    from .contract.inputs import CheckpointRef
    dataset = _load_dataset(input_path)
    dataset.metadata['input_path'] = str(Path(input_path).expanduser().resolve())
    dataset.metadata['input_manifest_sha256'] = hashlib.sha256((Path(input_path) / 'manifest.json').read_bytes()).hexdigest()
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: undeclared options are refused, not dropped (whole-branch review fix 6) | date: 2026-09-23]
    extra = {}
    if options.get('checkpoint'):
        extra['checkpoint'] = CheckpointRef(str(options['checkpoint']), options.get('checkpoint_probe'))
    # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no fitted-model forwarding (Task 2) | date: 2026-09-27]
    kept = backend_options(backend, options)
    # [AGENT EDIT END]
    return runner.run(dataset, backend, kept, group_column='recording_group', partition=partition,
                      output_dir=output_dir, root=root, extra_inputs=extra)
# [AGENT EDIT END]


def prepare_deep_dataset(input_path, *, source_checkout, output_dir=None,
                         autoencoder_epochs=300, contrastive_epochs=50,
                         seed=0, cpu_threads=4, preprocessing=None, root=None):
    """Select training units only; reserve other source wells for later evaluation."""
    dataset = _load_dataset(input_path)
    training = _select_partition(dataset, 'train')
    bundle = training.waveform_bundle()
    full_bundle = dataset.waveform_bundle()
    groups = full_bundle.provenance.get('source_groups', {})
    session_by_id = dict(zip(full_bundle.cellids, full_bundle.session_ids))
    eval_ids = dataset.labels.loc[dataset.labels.partition != 'train', 'cellid'].tolist()
    if not eval_ids:
        raise ValueError('DeepUnitMatch training requires actual held-out units from other source groups')
    try:
        training_groups = {groups[s] for s in bundle.session_ids}
        evaluation_groups = {groups[session_by_id[u]] for u in eval_ids}
    except KeyError as exc:
        raise ValueError('Benchmark source_groups must identify the original well for every session') from exc
    if training_groups & evaluation_groups:
        raise ValueError('Training and evaluation share original source groups; temporal pieces cannot cross holdouts')
    output = (paths.new_run_path(dataset.metadata['dataset_id'], kind='training', root=root)
              if output_dir is None else paths.resolve_cache_path(output_dir, root=root))
    with _cache_runtime(root):
        from .external_backends import prepare_deepunitmatch_training
        return prepare_deepunitmatch_training(training.units, bundle,
            source_checkout=source_checkout, output_dir=output,
            training_groups=training_groups, evaluation_groups=evaluation_groups,
            autoencoder_epochs=autoencoder_epochs, contrastive_epochs=contrastive_epochs,
            seed=seed, cpu_threads=cpu_threads, preprocessing=preprocessing)


def train_prepared(prepared_dir, *, python_executable=None, root=None):
    """Only this explicit command executes a previously prepared training run."""
    prepared = paths.resolve_cache_path(prepared_dir, root=root)
    if not (prepared / 'training_manifest.json').is_file():
        raise ValueError('Prepared directory has no training_manifest.json')
    with _cache_runtime(root):
        from .external_backends import train_deepunitmatch
        return train_deepunitmatch(prepared, python_executable=python_executable)


def _process_identity(pid):
    try:
        fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    except FileNotFoundError:
        return None
    return None if fields[0] == 'Z' else fields[19]


def wait_for_extraction(pid, input_path):
    """Chain a single method invocation to an already-running Linux extraction.

    A pidfd (or Linux process start time) follows the exact process, avoiding PID reuse. This is a dependency
    within a one-off analysis run, not a recurring task. Missing output after
    process exit is a failure; there is no retry or fallback dataset.
    """
    if not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise ValueError('Supply the PID of the separate running extraction process.')
    print(f'Waiting for extraction process {pid}; expected dataset: {input_path}', flush=True)
    if hasattr(os, 'pidfd_open'):
        try:
            descriptor = os.pidfd_open(pid)
        except ProcessLookupError:
            descriptor = None
        if descriptor is not None:
            try:
                while not select.select([descriptor], [], [], 60)[0]:
                    pass
            finally:
                os.close(descriptor)
    elif sys.platform.startswith('linux'):
        identity = _process_identity(pid)
        while identity is not None and _process_identity(pid) == identity:
            time.sleep(10)
    else:
        raise RuntimeError('Waiting for an existing extraction requires Linux process identity support.')
    if not (Path(input_path) / 'manifest.json').is_file():
        raise RuntimeError('Extraction process ended without a complete benchmark dataset.')


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run-backend', help='Run a genuine optional matching implementation')
    run.add_argument('--input', required=True, type=Path)
    run.add_argument('--backend', required=True, choices=('unitmatch', 'dant', 'deepunitmatch'))
    run.add_argument('--partition', choices=('test', 'train', 'calibration', 'all'), default='test')
    run.add_argument('--wait-for-pid', type=int, help='Run once after this existing extraction process exits (Linux)')
    run.add_argument('--output', type=Path, help='New directory inside the matching cache root')
    run.add_argument('--settings', type=Path, help='Complete DANT settings as strict JSON')
    run.add_argument('--params', type=Path, help='Optional UnitMatch/DeepUnitMatch parameter JSON')
    run.add_argument('--checkpoint', type=Path)
    run.add_argument('--checkpoint-probe')
    run.add_argument('--device', default='cpu')
    run.add_argument('--tracking', default='native_intermediate', choices=(
        'native_liberal', 'native_intermediate', 'native_conservative', 'common_constrained'))
    run.add_argument('--native-threshold', type=float, help='Override official native tracking threshold (default 0.5)')
    run.add_argument('--threshold', type=float, default=.9, help='Common constrained tracker only')
    run.add_argument('--contradiction-threshold', type=float, default=.1)
    run.add_argument('--ambiguity-margin', type=float, default=.02)
    prepare = commands.add_parser('prepare-deep', help='Stage training-partition inputs; do not train')
    prepare.add_argument('--input', required=True, type=Path)
    prepare.add_argument('--source-checkout', required=True, type=Path)
    prepare.add_argument('--output', type=Path, help='New training directory inside the matching cache root')
    prepare.add_argument('--autoencoder-epochs', type=int, default=300)
    prepare.add_argument('--contrastive-epochs', type=int, default=50)
    prepare.add_argument('--seed', type=int, default=0)
    prepare.add_argument('--cpu-threads', type=int, default=4)
    prepare.add_argument('--preprocessing', type=Path, help='Official preprocessing parameter JSON')
    train = commands.add_parser('train-deep', help='Execute a prepared official two-stage training run')
    train.add_argument('--prepared', type=Path, required=True)
    train.add_argument('--python-executable', help='Existing interpreter with official dependencies installed')
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == 'run-backend':
            if args.wait_for_pid is not None:
                # Fail before a long dependency wait if the requested official
                # implementation cannot import in this interpreter.
                from .external_backends import _module
                _module({'unitmatch': 'UnitMatchPy.overlord', 'dant': 'pyDANT.Runner',
                         'deepunitmatch': 'UnitMatchPy.overlord'}[args.backend])
                wait_for_extraction(args.wait_for_pid, args.input)
            # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: shared flags reach only backends that take them, unless set explicitly (whole-branch review fix 6) | date: 2026-09-23]
            shared = ('settings', 'params', 'checkpoint', 'checkpoint_probe', 'device', 'tracking', 'native_threshold',
                      'threshold', 'contradiction_threshold', 'ambiguity_margin')
            defaults = vars(parser.parse_args(['run-backend', '--input', str(args.input), '--backend', args.backend]))
            options = cli_shared_flags(args.backend, {k: getattr(args, k) for k in shared}, defaults)
            for key in ('settings', 'params'):
                if key in options:
                    options[key] = _json_object(options[key])
            output = run_backend_dataset(args.input, args.backend, partition=args.partition,
                output_dir=args.output, **options)
            # [AGENT EDIT END]
            print(f'Matching result: {output / "result.json"}')
        elif args.command == 'prepare-deep':
            output = prepare_deep_dataset(args.input, source_checkout=args.source_checkout,
                output_dir=args.output, autoencoder_epochs=args.autoencoder_epochs,
                contrastive_epochs=args.contrastive_epochs, seed=args.seed, cpu_threads=args.cpu_threads,
                preprocessing=_json_object(args.preprocessing))
            print(f'Training prepared; not started: {output}')
        else:
            result = train_prepared(args.prepared, python_executable=args.python_executable)
            print(f'Training completed. Checkpoint: {result["checkpoint"]}')
    except Exception as exc:
        parser.exit(2, f'{type(exc).__name__}: {exc}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
# [AGENT EDIT END]
