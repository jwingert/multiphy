# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher's command line: exit codes, printed lines and launcher.log (multiphy project folder Task 15) | date: 2026-09-25]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the module docstring names run (Plan B Task 3) | date: 2026-09-28]
"""The launcher's command line, ``python -m nems_lbhb.unit_match.launch [--root DIR] {init,extract,status,run} …``.

``main`` returns 0 on success; any error, a refused command line included, prints
``error: <message>`` to stderr and returns 1. When the command line parses and
``<project>/project.json`` exists after the command, ``<project>/logs/launcher.log`` gains one
line: the UTC time, the command with its arguments, and ``ok`` or ``error: <message>``. A
refused command line runs nothing and writes no line (D12).
"""
# [AGENT EDIT END]
from __future__ import annotations

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 1: log stamps are bounded by the call | date: 2026-09-25]
from datetime import datetime, timedelta, timezone
# [AGENT EDIT END]
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

import pytest

from nems_lbhb.unit_match.launch.__main__ import main
from nems_lbhb.unit_match.project.folder import Project
from nems_lbhb.unit_match.sources.base import clear_readers_for_tests

from .lab_blocker import run_without_lab
from .synthetic_day import write_day

REPO = Path(__file__).resolve().parents[2]
# The brief's values, written out rather than imported from the code under test.
RAW_FOUND = 'raw file found at dat_path; pass --uv-per-bit to use it'
D1_LINE = 'D1 · 2026-01-01 · raw · sources ok · features ok · runs 0'
ACROSS_LINE = 'across days · runs 0'
EXISTS = 'Day D1 already exists; a day is added once. Put a changed day in a new project.'


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


@pytest.fixture
def from_dir(tmp_path):
    """``DIR``: ``DIR/D1/`` links ``sort`` and ``raw.bin`` to a synthetic day, copies ``events.csv``, dates it."""
    sd = write_day(tmp_path / 'in', 'D1')
    folder = tmp_path / 'from' / 'D1'
    folder.mkdir(parents=True)
    (folder / 'sort').symlink_to(sd.sort_dir)
    (folder / 'raw.bin').symlink_to(sd.raw_path)
    shutil.copyfile(sd.events_path, folder / 'events.csv')
    (folder / 'date.txt').write_text('2026-01-01\n')
    return tmp_path / 'from'


def _log_lines(root: Path, project: str = 'P') -> list:
    return (root / project / 'logs' / 'launcher.log').read_text(encoding='utf-8').splitlines()


def _stamp(line: str) -> datetime:
    """The UTC time that starts a log line; it must be timezone-aware and at offset zero."""
    stamp = datetime.fromisoformat(line.split(' · ', 1)[0])
    assert stamp.utcoffset() == timedelta(0), line
    return stamp


def test_cli_writes_launcher_log(tmp_path, from_dir, capsys):
    root = tmp_path / 'lab'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 1: the log stamp is taken during the call | date: 2026-09-25]
    started = datetime.now(timezone.utc)
    # [AGENT EDIT END]
    assert main(['--root', str(root), 'init', 'P', '--from', str(from_dir), '--uv-per-bit', '0.195']) == 0
    lines = _log_lines(root)
    assert re.search(r' · init P --from .* · ok$', lines[-1])
    # The arguments after the command, as given; the global --root is not among them.
    assert lines == [f'{lines[0].split(" · ", 1)[0]} · init P --from {from_dir} --uv-per-bit 0.195 · ok']
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 1: the log stamp is taken during the call | date: 2026-09-25]
    assert started <= _stamp(lines[0]) <= datetime.now(timezone.utc)
    # [AGENT EDIT END]
    out = capsys.readouterr()
    assert (out.out, out.err) == ('', '')  # init prints its notes, and D1 has none

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 1: raw extract prints only its day line on stdout | date: 2026-09-25]
    # extract prints one line per day with the recorded ids and nothing else on stdout: the raw
    # pass's own progress lines go to stderr. status prints status_lines.
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 2: --scratch reaches the extraction | date: 2026-09-25]
    scratch = tmp_path / 'scratch'
    assert main([f'--root={root}', 'extract', 'P', '--scratch', str(scratch)]) == 0
    features = Project.open(root, 'P').read()['days']['D1']['features']
    out = capsys.readouterr()
    assert out.out == f"D1 · raw · halves {features['halves'][:12]} · full {features['full'][:12]}\n"
    assert f'Temporary extraction working directory: {scratch.resolve()}/extract-' in out.err
    assert 'D1 · ' not in out.err
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    assert Project.open(root, 'P').read()['days']['D1']['files']['raw']['sha256'] is None
    assert main(['--root', str(root), 'status', 'P', '--verify']) == 0
    assert capsys.readouterr().out == f'{D1_LINE}\n{ACROSS_LINE}\n'
    # --verify reached status: it recorded the raw file's first sha256.
    assert Project.open(root, 'P').read()['days']['D1']['files']['raw']['sha256_from'] == 'verify'
    lines = _log_lines(root)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 fix round 2: --scratch reaches the extraction | date: 2026-09-25]
    assert [line.split(' · ', 1)[1] for line in lines[1:]] == [f'extract P --scratch {scratch} · ok',
                                                               'status P --verify · ok']
    # [AGENT EDIT END]
    assert _stamp(lines[0]) <= _stamp(lines[1]) <= _stamp(lines[2])

    # One --day without --uv-per-bit: the note is printed, and params.py's dat_path is not recorded.
    sort, events, raw = (str(from_dir / 'D1' / name) for name in ('sort', 'events.csv', 'raw.bin'))
    assert main(['--root', str(root), 'init', 'Q', '--day', 'D1', '--sort', sort, '--date', '2026-01-01']) == 0
    assert capsys.readouterr().out == f'{RAW_FOUND}\n'
    assert 'raw' not in Project.open(root, 'Q').read()['days']['D1']['files']
    # One --day with every option; --tier templates on a day with raw has no halves, printed as —.
    assert main(['--root', str(root), 'init', 'T', '--day', 'D1', '--sort', sort, '--date', '2026-01-01',
                 '--events', events, '--raw', raw, '--uv-per-bit', '0.195']) == 0
    assert capsys.readouterr().out == ''
    record = Project.open(root, 'T').read()['days']['D1']
    assert (record['files']['events']['given'], record['files']['raw']['given']) == (events, raw)
    assert record['reader_options']['uv_per_bit'] == 0.195
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework fix round 1: extract --day extracts only the named day | date: 2026-09-25]
    # A second day that extract --day D1 leaves unextracted.
    assert main(['--root', str(root), 'init', 'T', '--day', 'D2', '--sort', sort, '--date', '2026-01-02']) == 0
    assert capsys.readouterr().out == f'{RAW_FOUND}\n'
    assert main(['--root', str(root), 'extract', 'T', '--day', 'D1', '--tier', 'templates']) == 0
    days = Project.open(root, 'T').read()['days']
    full = days['D1']['features']['full']
    assert capsys.readouterr().out == f'D1 · templates · halves — · full {full[:12]}\n'
    assert days['D2']['features']['full'] is None
    assert [line.split(' · ', 1)[1] for line in _log_lines(root, 'T')] == [
        f'init T --day D1 --sort {sort} --date 2026-01-01 --events {events} --raw {raw} --uv-per-bit 0.195 · ok',
        f'init T --day D2 --sort {sort} --date 2026-01-02 · ok',
        'extract T --day D1 --tier templates · ok']
    # [AGENT EDIT END]


def test_cli_error_exit(tmp_path, from_dir, capsys, monkeypatch):
    root = tmp_path / 'lab'
    # The default root is a test folder, so a command line with no usable --root cannot reach the lab's.
    default_root = tmp_path / 'default'
    (default_root / 'P').mkdir(parents=True)
    monkeypatch.setenv('MULTIPHY_CACHE_ROOT', str(default_root))
    missing = tmp_path / 'missing'
    assert main(['--root', str(root), 'init', 'P', '--from', str(missing)]) == 1
    out = capsys.readouterr()
    assert out.err.startswith('error: ')
    assert out.err == f'error: {missing} is not a folder.\n' and out.out == ''
    assert not (root / 'P').exists()  # no project folder, so no log

    # With project.json there after the command, the error is logged in place of ok.
    assert main(['--root', str(root), 'init', 'P', '--from', str(from_dir), '--uv-per-bit', '0.195']) == 0
    assert main(['--root', str(root), 'init', 'P', '--from', str(from_dir), '--uv-per-bit', '0.195']) == 1
    assert capsys.readouterr().err == f'error: {EXISTS}\n'
    assert _log_lines(root)[-1].endswith(f' · init P --from {from_dir} --uv-per-bit 0.195 · error: {EXISTS}')
    assert main(['--root', str(root), 'extract', 'P', '--day', 'D9']) == 1
    assert capsys.readouterr().err == 'error: Day D9 is not in project P.\n'
    assert _log_lines(root)[-1].endswith(' · extract P --day D9 · error: Day D9 is not in project P.')

    # A project the failed command created is logged too: add_day refuses the name after the project exists.
    sort = str(from_dir / 'D1' / 'sort')
    refused = 'Day name _D1 must match [A-Za-z0-9][A-Za-z0-9_.-]*.'
    assert main(['--root', str(root), 'init', 'R', '--day', '_D1', '--sort', sort, '--date', '2026-01-01']) == 1
    assert capsys.readouterr().err == f'error: {refused}\n'
    assert Project.open(root, 'R').read()['days'] == {}
    assert _log_lines(root, 'R')[-1].endswith(f' · init R --day _D1 --sort {sort} --date 2026-01-01 · error: {refused}')

    # A project that is not there stays absent; its error is printed, not logged.
    assert main(['--root', str(root), 'status', 'S']) == 1
    assert capsys.readouterr().err.startswith('error: No project S under ')
    assert not (root / 'S').exists()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework (D12): the line needs project.json; a refused command line is printed, never logged | date: 2026-09-25]
    # A folder without project.json is no project: its error is printed and nothing is written in it.
    (root / 'M').mkdir()
    assert main(['--root', str(root), 'status', 'M']) == 1
    assert tuple(capsys.readouterr()) == ('', f'error: No project M under {root.resolve()}.\n')
    assert list((root / 'M').iterdir()) == []
    # A project that has lost its logs/ still gets its line: project.json is there.
    shutil.rmtree(root / 'R' / 'logs')
    assert main(['--root', str(root), 'status', 'R']) == 0
    assert capsys.readouterr().out == f'{ACROSS_LINE}\n'
    assert [line.split(' · ', 1)[1] for line in _log_lines(root, 'R')] == ['status R · ok']

    # An OSError is an error too: an unreadable date.txt stops init before it writes.
    unreadable = tmp_path / 'unreadable' / 'D2'
    unreadable.mkdir(parents=True)
    (unreadable / 'sort').symlink_to(from_dir / 'D1' / 'sort')
    (unreadable / 'date.txt').write_text('2026-01-02\n')
    (unreadable / 'date.txt').chmod(0)
    try:
        if not os.access(unreadable / 'date.txt', os.R_OK):  # root reads it anyway
            assert main(['--root', str(root), 'init', 'P', '--from', str(unreadable.parent)]) == 1
            err = capsys.readouterr().err
            assert err.startswith('error: [Errno 13] Permission denied: ') and err.count('\n') == 1
            assert _log_lines(root)[-1].endswith(f' · init P --from {unreadable.parent} · {err.strip()}')
    finally:
        (unreadable / 'date.txt').chmod(0o644)

    # A command line the parser refuses prints error: <message> and returns 1. It runs nothing
    # and writes no log line, even in the initialized project it names.
    log = root / 'P' / 'logs' / 'launcher.log'
    before = log.read_bytes()
    assert main(['--root', str(root), 'status', 'P', '--verify=true']) == 1
    out = capsys.readouterr()
    assert out.err.startswith('error: ')
    assert log.read_bytes() == before
    assert tuple(out) == ('', "error: argument --verify: ignored explicit argument 'true'\n")
    for argv, message in (
            (['--root', str(root), 'init', 'P', '--from', str(from_dir), '--events', 'e.csv'],
             'argument --events: not allowed with argument --from'),
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework fix round 1: --raw with --from is refused like --events | date: 2026-09-25]
            (['--root', str(root), 'init', 'P', '--from', str(from_dir), '--raw', 'r.bin'],
             'argument --raw: not allowed with argument --from'),
            # [AGENT EDIT END]
            (['--root', str(root), 'init', 'P', '--day', 'D2', '--sort', sort],
             'the following arguments are required with --day: --date'),
            (['--root', str(root), 'init', 'P'], 'one of the arguments --from --day is required'),
            (['--root', str(root), 'init', 'P', '--from', str(from_dir), '--uv-per-bit', 'x'],
             "argument --uv-per-bit: invalid float value: 'x'"),
            (['--root', str(root), 'extract', 'P', '--tier', 'voltage'],
             "argument --tier: invalid choice: 'voltage' (choose from 'raw', 'templates')"),
            (['--root', str(root), 'extract', 'P', '--tier'], 'argument --tier: expected one argument'),
            (['--root', str(root), 'extract', 'P', '--day'], 'argument --day: expected at least one argument'),
            (['--root', str(root), 'status', 'P', '--bogus'], 'unrecognized arguments: --bogus'),
            (['--root', str(root), '--bogus', 'status', 'P'], 'unrecognized arguments: --bogus'),
            (['--root', str(root)], 'the following arguments are required: command'),
            (['--root', str(root), 'status'], 'the following arguments are required: PROJECT'),
            (['--root', '--bogus', 'status', 'P'], 'argument --root: expected one argument'),
    ):
        assert main(argv) == 1, argv
        assert tuple(capsys.readouterr()) == ('', f'error: {message}\n'), argv
        assert log.read_bytes() == before, argv
    assert sorted(Project.open(root, 'P').read()['days']) == ['D1']
    assert list((default_root / 'P').iterdir()) == []  # a --root without its value names no root
    assert list((root / 'M').iterdir()) == []
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework fix round 1: a held project.lock is an error that is printed and logged | date: 2026-09-25]
    # A held project.lock refuses status --verify's record; ProjectLocked is an error like any other.
    lock = Project.open(root, 'P').path / 'project.lock'
    lock.write_text('otherhost 1')
    locked = f'project.json is locked by otherhost 1; remove {lock} if that process is gone.'
    assert main(['--root', str(root), 'status', 'P', '--verify']) == 1
    assert tuple(capsys.readouterr()) == ('', f'error: {locked}\n')
    assert _log_lines(root)[-1].endswith(f' · status P --verify · error: {locked}')
    assert lock.read_text() == 'otherhost 1'
    lock.unlink()
    # [AGENT EDIT END]


def test_cli_help(tmp_path, capsys):
    env = dict(os.environ, PYTHONPATH=str(REPO))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: --help reads multiphy.json from a tmp root, never the lab default (Plan C Task 1 fix round 1) | date: 2026-10-01]
    help_root = str(tmp_path / 'help')
    proc = subprocess.run([sys.executable, '-m', 'nems_lbhb.unit_match.launch', '--root', help_root, '--help'],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    # [AGENT EDIT END]
    assert proc.returncode == 0, proc.stderr
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch gains gui (launcher-gui Task 5) | date: 2026-10-07]
    assert '{init,extract,status,run,export,gui}' in proc.stdout
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 15 rework fix round 1: init's usage puts --events and --raw in the --day arm; the entry point runs real commands | date: 2026-09-25]
    # init's usage: --events and --raw go with --day, since a --from day folder holds its own.
    root = tmp_path / 'lab'
    assert main(['--root', str(root), 'init', '--help']) == 0
    usage = capsys.readouterr().out.split('\n\n', 1)[0]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: init gains --probe TYPE (Plan B Task 3) | date: 2026-09-28]
    assert ' '.join(usage.split()) == (
        'usage: python -m nems_lbhb.unit_match.launch init [-h] PROJECT '
        '(--from DIR | --day DAY --sort DIR --date YYYY-MM-DD [--events CSV] [--raw FILE]) [--uv-per-bit X] '
        '[--probe TYPE]')
    # [AGENT EDIT END]
    assert not root.exists()

    # The entry point runs the command and exits with main's code: 0 after it ran, 1 after an error.
    Project.ensure(root, 'P')
    command = [sys.executable, '-m', 'nems_lbhb.unit_match.launch', '--root', str(root), 'status']
    proc = subprocess.run(command + ['P'], cwd=REPO, env=env, capture_output=True, encoding='utf-8', timeout=120)
    assert (proc.returncode, proc.stdout) == (0, 'across days · runs 0\n'), proc.stderr
    assert [line.split(' · ', 1)[1] for line in _log_lines(root)] == ['status P · ok']
    proc = subprocess.run(command + ['S'], cwd=REPO, env=env, capture_output=True, encoding='utf-8', timeout=120)
    assert (proc.returncode, proc.stdout) == (1, '')
    assert proc.stderr.splitlines()[-1] == f'error: No project S under {root.resolve()}.'
    # [AGENT EDIT END]

    # The command line imports no lab module, and main returns 0 for --help.
    proc = run_without_lab(
        'from nems_lbhb.unit_match.launch.__main__ import main\n'
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: --help reads multiphy.json from a tmp root, never the lab default (Plan C Task 1 fix round 1) | date: 2026-10-01]
        f"code = main(['--root', {help_root!r}, '--help'])\n"
        # [AGENT EDIT END]
        "print('code', code)\n"
        "print('lab modules', sorted(m for m in sys.modules if m.startswith(BLOCKED)))\n"
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: --help imports no Qt; the window is imported only for gui (launcher-gui Task 5) | date: 2026-10-07]
        "print('qt modules', sorted(m for m in sys.modules if m.split('.')[0] in ('PySide6', 'PyQt5')))\n", REPO)
        # [AGENT EDIT END]
    assert proc.returncode == 0, proc.stderr
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch gains gui (launcher-gui Task 5) | date: 2026-10-07]
    assert '{init,extract,status,run,export,gui}' in proc.stdout
    # [AGENT EDIT END]
    assert 'code 0' in proc.stdout and 'lab modules []' in proc.stdout
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: --help imports no Qt (launcher-gui Task 5) | date: 2026-10-07]
    assert 'qt modules []' in proc.stdout
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run refuses abbreviated flags (Follow-up Task 3) | date: 2026-09-29]
def test_run_refuses_abbreviated_backend(tmp_path, capsys):
    code = main(['--root', str(tmp_path / 'lab'), 'run', 'P', '--back', 'functional_likelihood'])
    err = capsys.readouterr().err
    assert code == 1
    assert err.endswith('error: the following arguments are required: --backend\n'), err
    assert 'Traceback' not in err
    assert not (tmp_path / 'lab').exists()


def test_run_refuses_abbreviated_backend_option(tmp_path, capsys):
    code = main(['--root', str(tmp_path / 'lab'), 'run', 'P', '--backend', 'functional_likelihood',
                 '--regist', 'none'])
    err = capsys.readouterr().err
    assert code == 1
    assert err.endswith('error: unrecognized arguments: --regist none\n'), err
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch export and the export name a plugin cannot take (run-curation Task 3) | date: 2026-10-04]
def test_plugin_cannot_register_export():
    from nems_lbhb.unit_match.launch import plugins
    with pytest.raises(ValueError, match='plugin command export is already defined'):
        plugins.PluginApi().add_command('export', lambda parser: None, lambda args, root: [])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: gui is a builtin command a plugin cannot take (launcher-gui Task 5) | date: 2026-10-07]
    with pytest.raises(ValueError, match='plugin command gui is already defined'):
        plugins.PluginApi().add_command('gui', lambda parser: None, lambda args, root: [])
    # [AGENT EDIT END]


def test_launch_export_prints_path(tmp_path, capsys):
    from nems_lbhb.unit_match.project.selection import Selection
    from nems_lbhb.unit_match.run.project_run import run_selection
    from .synthetic_day import extracted_project

    path = extracted_project(tmp_path)
    root, project = path.parent, path.name
    name = run_selection(Selection(Project.open(root, project), ('D1', 'D2')), 'functional_likelihood').name
    capsys.readouterr()
    assert main(['--root', str(root), 'export', project, '--run', name]) == 0
    out = capsys.readouterr().out
    assert out.rstrip('\n').endswith(' · 0 tracks · 0 units'), out
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run-curation Task 3 fix round 2: the readback of a .gz-named --out reads the plain CSV the writer wrote | date: 2026-10-04]
    gz = tmp_path / 'accepted.csv.gz'
    assert main(['--root', str(root), 'export', project, '--run', name, '--out', str(gz)]) == 0
    assert capsys.readouterr().out.rstrip('\n').endswith(' · 0 tracks · 0 units')
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: SIGINT during a command logs cancelled and exits 130 (launcher-gui Task 2) | date: 2026-10-07]
def _interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def test_cli_cancelled_outcome(tmp_path, from_dir, capsys, monkeypatch):
    root = tmp_path / 'root'
    assert main(['--root', str(root), 'init', 'P', '--from', str(from_dir), '--uv-per-bit', '0.195']) == 0
    capsys.readouterr()
    monkeypatch.setattr('nems_lbhb.unit_match.launch.__main__.extract', _interrupt)
    assert main(['--root', str(root), 'extract', 'P']) == 130
    err = capsys.readouterr().err
    assert err.endswith('cancelled\n'), err
    assert _log_lines(root)[-1].endswith(' · extract P · cancelled')


def test_cli_cancelled_log_failure_keeps_130(tmp_path, from_dir, capsys, monkeypatch):
    root = tmp_path / 'root'
    assert main(['--root', str(root), 'init', 'P', '--from', str(from_dir), '--uv-per-bit', '0.195']) == 0
    capsys.readouterr()
    monkeypatch.setattr('nems_lbhb.unit_match.launch.__main__.extract', _interrupt)

    def _read_only(*args, **kwargs):
        raise OSError('read-only')

    monkeypatch.setattr('nems_lbhb.unit_match.launch.__main__._log', _read_only)
    assert main(['--root', str(root), 'extract', 'P']) == 130
    lines = capsys.readouterr().err.splitlines()
    assert 'cancelled' in lines and 'error: read-only' in lines, lines


@pytest.mark.parametrize('inherited', ('default', 'ignored'))
def test_cli_sigint_real_process(tmp_path, inherited):
    import signal
    root = tmp_path / 'root'
    root.mkdir()
    (root / 'multiphy.json').write_text('{"plugins": ["tests.unit_match.wait_plugin"]}', encoding='utf-8')
    Project.ensure(root, 'P')
    preexec = (lambda: signal.signal(signal.SIGINT, signal.SIG_IGN)) if inherited == 'ignored' else None
    proc = subprocess.Popen([sys.executable, '-m', 'nems_lbhb.unit_match.launch', '--root', str(root), 'wait', 'P'],
                            cwd=REPO, env=dict(os.environ, PYTHONPATH=str(REPO), PYTHONDONTWRITEBYTECODE='1'),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, preexec_fn=preexec)
    try:
        for line in proc.stderr:
            if line.rstrip('\n') == 'waiting':
                break
        else:
            raise AssertionError(f'no waiting line; exit {proc.wait(timeout=30)}')
        proc.send_signal(signal.SIGINT)
        assert proc.wait(timeout=30) == 130
        rest = proc.stderr.read()
        assert 'cancelled' in rest, rest
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        proc.stdout.close()
        proc.stderr.close()
    assert _log_lines(root)[-1].endswith(' · wait P · cancelled')
# [AGENT EDIT END]
