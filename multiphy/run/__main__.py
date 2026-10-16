# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: one command line generated from backend option schemas | date: 2026-09-22]
"""``python -m multiphy.run run-backend --input DIR --backend ID [options]``.

Backend options are generated from the backend's spec, so ``--help`` after ``--backend ID``
lists exactly what that backend takes. ``list-backends`` prints ids, labels and availability.
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
import sys

from ..contract import registry
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no fitted-model safeguard (Task 2) | date: 2026-09-27]
from ..contract.inputs import CheckpointRef
# [AGENT EDIT END]
from . import options as opt, runner
# [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no model JSON to read (Task 2) | date: 2026-09-27]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every --input archive is a temporal-split benchmark; multiday archives are retired (Plan B Task 8) | date: 2026-09-28]
def _load_dataset(path):
    """The benchmark dataset at ``path``, with the sha256 of its manifest.json in its metadata."""
    from ..benchmark import load_dataset
    manifest_path = Path(path) / 'manifest.json'
    dataset = load_dataset(path)
    dataset.metadata['input_manifest_sha256'] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    return dataset
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run-backend --project runs a selection of project days (Plan B Task 2) | date: 2026-09-28]
def _run_project(args, spec, parser) -> int:
    from ..project.folder import Project
    from ..project.selection import Selection, default_days
    from .project_run import run_selection

    path = Path(args.project).expanduser().resolve()
    project = Project.open(path.parent, path.name)
    # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - no --root refusal; Do 7 refuses only --partition, --group-column and --no-evaluate, and run_selection uses the project's root (Plan B Task 2) | date: 2026-09-28]
    days = (args.day,) if args.day is not None else tuple(args.days) if args.days is not None else default_days(project)
    output = run_selection(Selection(project, days), args.backend, opt.from_namespace(spec, args),
                           output_dir=Path(args.output) if args.output else None)
    print(f'Matching result: {output / "result.json"}')
    return 0
# [AGENT EDIT END]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ['list-backends']:
        for spec in registry.list_specs():
            a = registry.get(spec.id).available()
            print(f'{spec.id:14s} {runner.backend_label(spec.id):60s} {"available" if a.ok else "unavailable: " + str(a.reason)}')
        return 0
    head = argparse.ArgumentParser(add_help=False)
    head.add_argument('--backend', required=True, choices=[s.id for s in registry.list_specs()])
    known, _ = head.parse_known_args([a for a in argv if a != 'run-backend'])
    spec = registry.get(known.backend).spec
    parser = argparse.ArgumentParser(prog='python -m multiphy.run', description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('run-backend')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run-backend takes an archive (--input) or project days (--project) (Plan B Task 2) | date: 2026-09-28]
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--input', help='an archived dataset folder containing manifest.json')
    source.add_argument('--project', metavar='DIR', help='a project folder; runs on its extracted days')
    days = p.add_mutually_exclusive_group()
    days.add_argument('--day', metavar='DAY', help="with --project: match this day's two temporal halves")
    days.add_argument('--days', metavar='DAY', nargs='+', help='with --project: match these days across days')
    p.add_argument('--backend', required=True)
    # [AGENT EDIT END]
    p.add_argument('--partition', choices=('train', 'calibration', 'test', 'all'))
    p.add_argument('--group-column'); p.add_argument('--output'); p.add_argument('--root')
    p.add_argument('--no-evaluate', action='store_true')
    opt.add_arguments(p, spec)
    args = parser.parse_args(argv)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project run goes through run_selection; archive-only flags are refused (Plan B Task 2) | date: 2026-09-28]
    if args.project is not None:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a supplied option is refused even when empty; --root is not an archive-only flag (Plan B Task 2 fix round 1) | date: 2026-09-28]
        archive_flags = [flag for flag, given in (('--partition', args.partition is not None),
                                                  ('--group-column', args.group_column is not None),
                                                  ('--no-evaluate', args.no_evaluate)) if given]
        if archive_flags:
            parser.error(f'{", ".join(archive_flags)} apply to --input archives only; a project run sets its '
                         'partition, group column and evaluation from the project')
        return _run_project(args, spec, parser)
        # [AGENT EDIT END]
    if args.day is not None or args.days is not None:
        parser.error('--day and --days apply to --project only')
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an --input archive is a benchmark: the test partition and recording_group by default (Plan B Task 8) | date: 2026-09-28]
    dataset = _load_dataset(args.input)
    options = opt.validate(spec, opt.from_namespace(spec, args))
    if args.partition is None:
        args.partition = 'test'
    dataset.metadata.setdefault('input_path', str(Path(args.input).expanduser().resolve()))
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no --model; no fitted-model scope (Task 2) | date: 2026-09-27]
    extra = {}
    # [AGENT EDIT END]
    if options.get('checkpoint'):
        extra['checkpoint'] = CheckpointRef(options['checkpoint'], options.get('checkpoint_probe'))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: recording_group is the only default group column (Plan B Task 8) | date: 2026-09-28]
    output = runner.run(dataset, args.backend, options, group_column=args.group_column or 'recording_group',
                        partition=args.partition, output_dir=args.output, root=args.root,
                        evaluate=not args.no_evaluate, extra_inputs=extra)
    # [AGENT EDIT END]
    print(f'Matching result: {output / "result.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
# [AGENT EDIT END]
