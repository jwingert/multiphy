# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher's command line: init, extract and status without writing Python (multiphy project folder Task 15) | date: 2026-09-25]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the module docstring names run (Plan B Task 3) | date: 2026-09-28]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the plugin commands (Plan C final review fix round 1) | date: 2026-10-02]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names export (run-curation Task 3) | date: 2026-10-04]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the cancelled outcome and exit 130 (launcher-gui Task 2) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names gui (launcher-gui Task 5) | date: 2026-10-07]
"""``python -m nems_lbhb.unit_match.launch [--root DIR] {init,extract,status,run,export,gui,<plugin command>} …``.

Plugin commands, such as ``lbhb-init``, come from the modules named in ``<root>/multiphy.json``
and from ``multiphy.plugins`` entry points; each takes the project first, then its own options.

``init`` adds days to a project, ``--from`` a folder of day folders or one ``--day``, and prints
each note. ``extract`` extracts the named days, by default every day not yet extracted, and
prints one line per day; the raw pass's own progress lines go to stderr. ``status`` prints one
line per day, then the across-days line. ``run`` runs one registered backend on a project's
extracted days, by default all of them, and prints ``<run folder> · <backend> · <days> · excluded
<n>``; the backend's own options follow ``--backend ID``. ``export`` writes the accepted tracks of
the across-day run ``--run RUN`` to a new CSV, by default under ``curation/runs/<run>/exports/``,
and prints ``<path> · <n> tracks · <n> units``. ``gui [PROJECT]`` opens the launcher window
(``window.run_window``), which runs every other command as a child process of this module; it
writes no ``launcher.log`` line (D22). ``--root`` defaults to ``run.config.cache_root()``.

An error, a command line the parser refuses included, prints ``error: <message>`` to stderr and
returns 1. ``--help`` returns 0. When the command line parses and ``<project>/project.json`` exists
after the command, ``<project>/logs/launcher.log`` gains one line: the UTC time, the command with
the arguments given after it, and ``ok``, ``error: <message>`` or ``cancelled``. A refused command
line runs nothing and writes no line (D12). SIGINT during a command prints ``cancelled`` on stderr
and returns 130, even in a process started with SIGINT ignored (D26, D75).
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
from __future__ import annotations

import argparse
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 1: the commands' own progress lines go to stderr | date: 2026-09-25]
from contextlib import redirect_stdout
# [AGENT EDIT END]
from datetime import datetime, timezone
from pathlib import Path
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: main restores the default SIGINT handler (launcher-gui Task 2) | date: 2026-10-07]
import signal
# [AGENT EDIT END]
import sys
from typing import Optional

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run command takes a registered backend and its options (Plan B Task 3) | date: 2026-09-28]
from ..contract import registry
from ..extract.pipeline import TIERS
from ..project.folder import Project
from ..run import options as opt
from ..run.config import cache_root
from .commands import extract, init_day, init_from, run, status, status_lines
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch export PROJECT --run RUN (run-curation Task 3) | date: 2026-10-04]
from . import commands
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher loads the root's plugins before parsing (Plan C Task 1) | date: 2026-10-01]
from .plugins import load_plugins
# [AGENT EDIT END]

PROG = 'python -m nems_lbhb.unit_match.launch'
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework fix round 1: --events and --raw belong to the --day arm; init_from takes neither, a --from day folder holds its own | date: 2026-09-25]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: init gains --probe TYPE (Plan B Task 3) | date: 2026-09-28]
INIT_USAGE = ('%(prog)s [-h] PROJECT (--from DIR | --day DAY --sort DIR --date YYYY-MM-DD '
              '[--events CSV] [--raw FILE]) [--uv-per-bit X] [--probe TYPE]')
# [AGENT EDIT END]
# [AGENT EDIT END]
OK = 'ok'
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: SIGINT during a command is the cancelled outcome (launcher-gui Task 2) | date: 2026-10-07]
CANCELLED = 'cancelled'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework (D12): one parser; a refused command line prints error: <message>, returns 1 and is not logged | date: 2026-09-25]
class UsageError(Exception):
    """A command line the parser refuses; the message is argparse's."""


class _Parser(argparse.ArgumentParser):
    """An ``ArgumentParser`` whose ``error`` raises ``UsageError`` in place of printing usage and exiting 2.
    Its subparsers are ``_Parser`` too."""

    def error(self, message):
        raise UsageError(message)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run adds the options of the backend named after --backend (Plan B Task 3) | date: 2026-09-28]
def _backend_spec(argv: list):
    """The spec of the backend named after ``--backend`` in ``argv``; ``None`` when none is named."""
    backend_id = None
    for index, arg in enumerate(argv):
        if arg == '--backend' and index + 1 < len(argv):
            backend_id = argv[index + 1]
        elif arg.startswith('--backend='):
            backend_id = arg.split('=', 1)[1]
    if backend_id is None:
        return None
    ids = [spec.id for spec in registry.list_specs()]
    if backend_id not in ids:
        raise UsageError(f'unknown backend {backend_id}; known: {", ".join(sorted(ids))}')
    return registry.get(backend_id).spec
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: --root is read before parsing so the root's plugins can add commands (Plan C Task 1) | date: 2026-10-01]
def _leading_root(argv: list) -> tuple:
    """``(root, command_index)``: the ``--root DIR`` / ``--root=DIR`` value given before the command
    word (``None`` when none is), and the index of that word (``len(argv)`` when there is none)."""
    root, index = None, 0
    while index < len(argv):
        if argv[index] == '--root' and index + 1 < len(argv):
            root, index = argv[index + 1], index + 2
        elif argv[index].startswith('--root='):
            root, index = argv[index].split('=', 1)[1], index + 1
        else:
            break
    return root, index
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: build_parser builds the CLI's parser so the launcher window reads its forms off it (launcher-gui Task 3) | date: 2026-10-07]
def build_parser(plugin_commands: Optional[dict] = None, backend_spec=None) -> argparse.ArgumentParser:
    """The launcher's parser: the built-in commands, then each plugin command. ``plugin_commands`` maps
    a plugin command's name to ``(configure, handler, help)``; ``run`` gains ``backend_spec``'s
    options when it is not ``None``."""
    plugin_commands = {} if plugin_commands is None else plugin_commands
# [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the description names run (Plan B Task 3) | date: 2026-09-28]
    parser = _Parser(prog=PROG, description='Add days to a project folder, extract and check them, and run matchers on them.',
                     allow_abbrev=False)
    # [AGENT EDIT END]  # only --root DIR or --root=DIR can precede the command
    parser.add_argument('--root', metavar='DIR', help='lab root holding the projects · default run.config.cache_root()')
    commands = parser.add_subparsers(dest='command', required=True)

    init = commands.add_parser('init', usage=INIT_USAGE, help='add days · creates a new project')
    init.add_argument('project', metavar='PROJECT')
    days = init.add_mutually_exclusive_group(required=True)
    days.add_argument('--from', dest='from_dir', metavar='DIR',
                      help='day folders · sort/, date.txt, optional events.csv and one raw.* file')
    days.add_argument('--day', metavar='DAY', help='one day · with --sort and --date')
    init.add_argument('--sort', metavar='DIR', help='phy folder · with --day')
    init.add_argument('--date', metavar='YYYY-MM-DD', help='day date · with --day')
    init.add_argument('--events', metavar='CSV', help='events CSV · with --day')
    init.add_argument('--raw', metavar='FILE', help='raw voltage file · with --day · needs --uv-per-bit')
    init.add_argument('--uv-per-bit', type=float, metavar='X', help='microvolts per bit of the raw files')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: init gains --probe TYPE (Plan B Task 3) | date: 2026-09-28]
    init.add_argument('--probe', metavar='TYPE', help='probe type · recorded once per project')
    # [AGENT EDIT END]

    ext = commands.add_parser('extract', help='extract days · default every day not yet extracted')
    ext.add_argument('project', metavar='PROJECT')
    ext.add_argument('--day', dest='days', action='extend', nargs='+', metavar='DAY',
                     help='days to extract · repeatable')
    ext.add_argument('--tier', choices=TIERS, help='default raw with a raw file, else templates')
    ext.add_argument('--scratch', metavar='DIR', help='absolute folder for voltage snippets · default _runtime/')

    check = commands.add_parser('status', help="each day's sources, features and runs")
    check.add_argument('project', metavar='PROJECT')
    check.add_argument('--verify', action='store_true', help='re-hash every recorded file')

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run PROJECT --backend ID (Plan B Task 3) | date: 2026-09-28]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run refuses abbreviated flags; _backend_spec reads only the full --backend (Follow-up Task 3) | date: 2026-09-29]
    run_parser = commands.add_parser('run', help='run a registered matcher · default every extracted day',
                                     allow_abbrev=False)
    # [AGENT EDIT END]
    run_parser.add_argument('project', metavar='PROJECT')
    run_parser.add_argument('--backend', required=True, metavar='ID', help='a registered backend id')
    chosen = run_parser.add_mutually_exclusive_group()
    chosen.add_argument('--day', metavar='DAY', help='one day · matches its two temporal halves')
    chosen.add_argument('--days', nargs='+', metavar='DAY', help='two or more days · matched across days')
    run_parser.add_argument('--python', metavar='EXE', help='run the backend in this interpreter')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: build_parser takes the backend spec; _parse reads it from argv (launcher-gui Task 3) | date: 2026-10-07]
    if backend_spec is not None:
        opt.add_arguments(run_parser, backend_spec)
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch export PROJECT --run RUN [--out PATH] (run-curation Task 3) | date: 2026-10-04]
    export_parser = commands.add_parser('export', help="an across-day run's accepted tracks as a CSV",
                                        allow_abbrev=False)
    export_parser.add_argument('project', metavar='PROJECT')
    export_parser.add_argument('--run', required=True, metavar='RUN', help="a run folder in the project's runs/")
    export_parser.add_argument('--out', metavar='PATH',
                               help='a new CSV · default curation/runs/<run>/exports/accepted_tracks_<stamp>.csv')
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch gui [PROJECT] opens the launcher window (launcher-gui Task 5) | date: 2026-10-07]
    gui = commands.add_parser('gui', help='the launcher window', allow_abbrev=False)
    gui.add_argument('project', metavar='PROJECT', nargs='?',
                     help='project shown first; a new name fills the init form')
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each plugin command takes the project first, then its own options (Plan C Task 1) | date: 2026-10-01]
    for name, (configure, _handler, help_text) in plugin_commands.items():
        plugin_parser = commands.add_parser(name, help=help_text, allow_abbrev=False)
        plugin_parser.add_argument('project', metavar='PROJECT')
        configure(plugin_parser)
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _parse builds its parser with build_parser (launcher-gui Task 3) | date: 2026-10-07]
    return parser


def _parse(argv: list, plugin_commands: Optional[dict] = None) -> argparse.Namespace:
    """The parsed command; ``UsageError`` for a command line it refuses, ``SystemExit(0)`` after ``--help``.
    ``plugin_commands`` maps a plugin command's name to ``(configure, handler, help)``."""
    # --backend is read only when the command word is run, so a plugin command may take its own --backend
    command_index = _leading_root(argv)[1]
    is_run = command_index < len(argv) and argv[command_index] == 'run'
    spec = _backend_spec(argv) if is_run else None
    parser = build_parser(plugin_commands, spec)
    # [AGENT EDIT END]
    args = parser.parse_args(argv)
    if args.command == 'init':
        if args.from_dir is not None:
            for flag, value in (('--sort', args.sort), ('--date', args.date), ('--events', args.events),
                                ('--raw', args.raw)):
                if value is not None:  # a --from day folder holds its own sort, date, events and raw
                    raise UsageError(f'argument {flag}: not allowed with argument --from')
        else:
            missing = [flag for flag, value in (('--sort', args.sort), ('--date', args.date)) if value is None]
            if missing:
                raise UsageError(f'the following arguments are required with --day: {", ".join(missing)}')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run PROJECT --backend ID (Plan B Task 3) | date: 2026-09-28]
    if args.command == 'run':
        if args.days is not None and len(args.days) < 2:
            raise UsageError('--days needs two or more days; use --day for one')
        args.options = opt.from_namespace(spec, args)
    # [AGENT EDIT END]
    return args


def _after_command(argv: list) -> list:
    """The arguments given after the command word of a parsed ``argv``; only ``--root DIR`` and
    ``--root=DIR`` can precede that word."""
    index = 0
    while argv[index] == '--root' or argv[index].startswith('--root='):
        index += 2 if argv[index] == '--root' else 1
    return argv[index + 1:]
# [AGENT EDIT END]


def _extract_line(day: str, result: dict) -> str:
    """One day's ``extract`` line; a templates-tier day has no halves id and shows ``—``."""
    halves_id, full_id = result['features']['halves'], result['features']['full']
    return f'{day} · {result["tier"]} · halves {(halves_id or "")[:12] or "—"} · full {full_id[:12]}'


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a plugin command runs its own handler (Plan C Task 1) | date: 2026-10-01]
def _run(args: argparse.Namespace, root: Path, plugin_commands: Optional[dict] = None) -> list:
    """Run the parsed command under ``root``; return the lines it prints."""
    if plugin_commands and args.command in plugin_commands:
        return list(plugin_commands[args.command][1](args, root))
# [AGENT EDIT END]
    if args.command == 'init':
        if args.from_dir is not None:
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: init passes --probe (Plan B Task 3) | date: 2026-09-28]
            result = init_from(root, args.project, Path(args.from_dir), uv_per_bit=args.uv_per_bit,
                               probe=args.probe)
        else:
            result = init_day(root, args.project, day=args.day, sort=args.sort, date=args.date,
                              events=args.events, raw=args.raw, uv_per_bit=args.uv_per_bit, probe=args.probe)
            # [AGENT EDIT END]
        return list(result['notes'])
    if args.command == 'extract':
        scratch_root = None if args.scratch is None else Path(args.scratch)
        result = extract(root, args.project, days=args.days, tier=args.tier, scratch_root=scratch_root)
        return [_extract_line(day, day_result) for day, day_result in result['days'].items()]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run prints one line per run (Plan B Task 3) | date: 2026-09-28]
    if args.command == 'run':
        days = [args.day] if args.day is not None else args.days
        result = run(root, args.project, backend=args.backend, days=days, options=args.options, python=args.python)
        return [f'{result["run"]} · {result["backend"]} · {",".join(result["days"])} · excluded {result["excluded"]}']
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch export prints the written path and its counts (run-curation Task 3) | date: 2026-10-04]
    if args.command == 'export':
        result = commands.export(root, args.project, run=args.run, out=None if args.out is None else Path(args.out))
        return [f'{result["path"]} · {result["tracks"]} tracks · {result["units"]} units']
    # [AGENT EDIT END]
    return status_lines(status(root, args.project, verify=args.verify))


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework (D12): the line needs a parsed command line and project.json; progress lines go to stderr | date: 2026-09-25]
def _log(root: Path, project: str, line: str) -> None:
    """Append ``line`` to the project's ``logs/launcher.log`` when its folder holds a ``project.json``.
    A project that has lost its ``logs/`` gets it back."""
    try:
        opened = Project.open(root, project)
    except (FileNotFoundError, ValueError):  # no project.json, or PROJECT names no folder under the root
        return
    (opened.path / 'logs').mkdir(exist_ok=True)  # Project.create made it before project.json
    opened.log(line)


def main(argv: Optional[list] = None) -> int:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: SIGINT is KeyboardInterrupt even when inherited ignored; a cancelled command returns 130 (launcher-gui Task 2) | date: 2026-10-07]
    """Run one launcher command; return 0 on success or after ``--help``, 130 after SIGINT during the
    command, 1 after an error."""
    argv = list(sys.argv[1:] if argv is None else argv)
    signal.signal(signal.SIGINT, signal.default_int_handler)  # D26: a child started with SIGINT ignored
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the root's plugins load before parsing; a plugin that fails stops the launcher (Plan C Task 1) | date: 2026-10-01]
    try:
        api = load_plugins(cache_root(_leading_root(argv)[0]))
    except ValueError as exc:
        print(f'error: {exc}', file=sys.stderr)
        return 1
    plugin_commands = api.commands
    try:
        args = _parse(argv, plugin_commands)
    # [AGENT EDIT END]
    except SystemExit as exc:  # argparse has printed the help
        return 0 if exc.code is None else int(exc.code)
    except UsageError as exc:  # the command never ran, so nothing is logged
        print(f'error: {exc}', file=sys.stderr)
        return 1
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: gui opens the window; window is imported only here and gui writes no log line (D22) (launcher-gui Task 5) | date: 2026-10-07]
    if args.command == 'gui':
        try:
            root = cache_root(args.root)
            from .window import run_window
            return run_window(root, args.project, api)
        except Exception as exc:
            print(f'error: {exc}', file=sys.stderr)
            return 1
    # [AGENT EDIT END]
    root = None
    outcome = OK
    try:
        root = cache_root(args.root)
        with redirect_stdout(sys.stderr):  # the commands' own progress lines; stdout holds only their result
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _run gets the plugins' commands (Plan C Task 1) | date: 2026-10-01]
            lines = _run(args, root, plugin_commands)
            # [AGENT EDIT END]
        for line in lines:
            print(line)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: SIGINT during the command is logged as cancelled (launcher-gui Task 2) | date: 2026-10-07]
    except KeyboardInterrupt:
        print('cancelled', file=sys.stderr)
        outcome = CANCELLED
    # [AGENT EDIT END]
    except Exception as exc:
        print(f'error: {exc}', file=sys.stderr)
        outcome = f'error: {exc}'
    if root is not None:
        utc_iso = datetime.now(timezone.utc).isoformat()
        try:
            _log(root, args.project, f'{utc_iso} · {args.command} {" ".join(_after_command(argv))} · {outcome}')
        except Exception as exc:  # the outcome is printed; only its log line is missing
            print(f'error: {exc}', file=sys.stderr)
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a cancelled command still returns 130 when its line cannot be written (launcher-gui Task 2, D75) | date: 2026-10-07]
            return 130 if outcome == CANCELLED else 1
            # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a cancelled command returns 130 (launcher-gui Task 2) | date: 2026-10-07]
    if outcome == CANCELLED:
        return 130
    # [AGENT EDIT END]
    return 0 if outcome == OK else 1
# [AGENT EDIT END]


if __name__ == '__main__':
    sys.exit(main())
# [AGENT EDIT END]
