# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run runs any registered matcher on project days, init --probe records the probe (Plan B Task 3) | date: 2026-09-28]
"""``launch run`` and ``init --probe``: one command runs any registered backend on extracted project days."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import typing
from typing import Optional

import pandas as pd
import pytest

from multiphy.contract import registry
from multiphy.launch import commands
from multiphy.launch.__main__ import main
from multiphy.project.folder import RUNS_DIR, Project
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refusal test relabels a day's units through Selection.dataset (event-optional Task 14) | date: 2026-10-10]
from multiphy.project.selection import Selection
# [AGENT EDIT END]
from multiphy.run import subprocess as sp

from .synthetic_day import extracted_project, write_day

FL = 'functional_likelihood'


def _open(project_dir: Path) -> Project:
    return Project.open(project_dir.parent, project_dir.name)


def _runs(project: Project) -> set:
    folder = project.path / RUNS_DIR
    return set(folder.iterdir()) if folder.is_dir() else set()


@pytest.fixture(scope='module')
def weak(tmp_path_factory) -> Project:
    """D1 and D2 extracted from raw; D1 holds the weak unit, which the selection excludes."""
    return _open(extracted_project(tmp_path_factory.mktemp('weak'), weak_units=('D1',)))


@pytest.fixture(scope='module')
def clean(tmp_path_factory) -> Project:
    """D1 and D2 extracted from raw; no test here writes a run into it."""
    return _open(extracted_project(tmp_path_factory.mktemp('clean')))


@pytest.fixture
def unavailable(monkeypatch):
    """functional_likelihood reports itself unavailable; returns the recorders of the lab interpreter and subprocess."""
    backend = registry.get(FL)
    monkeypatch.setattr(backend, 'available', lambda: SimpleNamespace(ok=False, reason='missing package'))
    calls = SimpleNamespace(interpreter=[], run=[], found='/lab/python')

    def interpreter(*args, **kwargs):
        calls.interpreter.append((args, kwargs))
        return calls.found

    def run(command, **kwargs):
        calls.run.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, '', '')

    monkeypatch.setattr(commands, 'verified_interpreter', interpreter)
    monkeypatch.setattr(commands.subprocess, 'run', run)
    # '/lab/python' is not a file here; the recorded command only has to name it.
    monkeypatch.setattr(sp, '_program', lambda program: Path(program))
    return calls


def test_launch_run_defaults_to_all_extracted_days(weak):
    result = commands.run(weak.root, 'P', backend=FL)
    assert result['days'] == ['D1', 'D2']
    assert result['excluded'] == 1
    assert (result['project'], result['backend']) == ('P', FL)
    assert result['run'].parent == weak.path / RUNS_DIR
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the run matched both days: the saved selection names D1 and D2 and the weak unit D1:4, and the result holds the good clusters 0-2 of both days (Plan B Task 3) | date: 2026-09-28]
    saved = json.loads((result['run'] / 'selection.json').read_text())
    assert [d['day'] for d in saved['days']] == ['D1', 'D2']
    assert saved['mode'] == 'across_days'
    assert [(e['cellid'], e['day']) for e in saved['excluded']] == [('D1:4', 'D1')]
    matched = json.loads((result['run'] / 'result.json').read_text())['result']
    assert matched['provenance']['days'] == ['D1', 'D2']
    units = {u for track in matched['tracks'] for u in track} | set(matched['unmatched'])
    assert units == {f'{day}:{cluster}' for day in ('D1', 'D2') for cluster in (0, 1, 2)}
    # [AGENT EDIT END]


def test_launch_run_needs_two_days(tmp_path):
    project = _open(extracted_project(tmp_path, ('D1',)))
    with pytest.raises(ValueError, match='needs 2 or more') as caught:
        commands.run(project.root, 'P', backend=FL)
    assert str(caught.value) == ('launch run needs 2 or more extracted days by default; P has 1; '
                                 'pass --day for one day')
    assert not (project.path / RUNS_DIR).exists()


def test_launch_run_child_failure_keeps_selection(tmp_path):
    project = _open(extracted_project(tmp_path))
    with pytest.raises(RuntimeError, match='0 recording groups can be matched') as caught:
        commands.run(project.root, 'P', backend='pair_logistic', python=sys.executable)
    (output,) = _runs(project)
    assert str(caught.value).startswith('pair_logistic failed: ')
    assert str(caught.value).endswith(f' · run folder {output}')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the failed run keeps a full selection and its 9 cross-day pairs of the good clusters 0-2 (Plan B Task 3) | date: 2026-09-28]
    saved = json.loads((output / 'selection.json').read_text())
    assert [d['day'] for d in saved['days']] == ['D1', 'D2']
    assert (saved['mode'], saved['excluded']) == ('across_days', [])
    pairs = pd.read_csv(output / 'pairs.csv')
    assert sorted(zip(pairs.cellid_a, pairs.cellid_b)) == [(f'D1:{a}', f'D2:{b}') for a in range(3) for b in range(3)]
    # [AGENT EDIT END]
    assert json.loads((output / 'run_manifest.json').read_text())['status'] == 'failed'


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the recorded command names the module, project, days, backend, output and supplied option, and keeps the existing PYTHONPATH (Plan B Task 3) | date: 2026-09-28]
def test_unavailable_backend_runs_in_lab_interpreter(clean, unavailable, monkeypatch):
    monkeypatch.setenv('PYTHONPATH', '/existing/path')
    result = commands.run(clean.root, 'P', backend=FL, options={'seed': 7})
    assert unavailable.interpreter == [((), {})]
    ((command, kwargs),) = unavailable.run
    assert command[:17] == ['/lab/python', '-u', '-m', 'multiphy.run', 'run-backend',
                            '--project', str(clean.path), '--days', 'D1', 'D2', '--backend', FL,
                            '--output', str(result['run']), '--min-repeats', '5', '--min-spikes']
    assert command[command.index('--seed') + 1] == '7'
    repo = Path(commands.__file__).resolve().parents[2]
    assert (kwargs['cwd'], kwargs['capture_output'], kwargs['text']) == (repo, True, True)
    assert kwargs['env']['PYTHONPATH'] == f'{repo}:/existing/path'
    assert kwargs['env']['PYTHONDONTWRITEBYTECODE'] == '1'
# [AGENT EDIT END]
    assert not _runs(clean)


def test_unavailable_backend_without_interpreter_refused(clean, unavailable):
    unavailable.found = None
    with pytest.raises(ValueError, match='--python') as caught:
        commands.run(clean.root, 'P', backend=FL)
    assert str(caught.value) == (f'{FL} is unavailable here (missing package) and no verified interpreter '
                                 'exists; pass --python EXE')
    assert unavailable.run == []
    assert not (clean.path / RUNS_DIR).exists()


def test_init_probe_recorded_and_conflict_refused(tmp_path):
    root = tmp_path / 'lab'
    d1 = write_day(tmp_path / 'in', 'D1')
    d2 = write_day(tmp_path / 'in', 'D2', seed=1, date='2026-01-02')
    commands.init_day(root, 'P', day='D1', sort=str(d1.sort_dir), date=d1.date, probe='NP1')
    project = Project.open(root, 'P')
    assert project.read()['identity']['probe'] == 'NP1'
    before = (project.path / 'project.json').read_bytes()
    with pytest.raises(ValueError) as caught:
        commands.init_day(root, 'P', day='D2', sort=str(d2.sort_dir), date=d2.date, probe='NP2')
    assert str(caught.value) == 'project P probe is NP1; --probe says NP2'
    assert not project.day_dir('D2').exists()
    assert (project.path / 'project.json').read_bytes() == before
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the same probe is accepted (Plan B Task 3) | date: 2026-09-28]
    commands.init_day(root, 'P', day='D2', sort=str(d2.sort_dir), date=d2.date, probe='NP1')
    assert project.read()['identity']['probe'] == 'NP1'
    assert project.day_dir('D2').exists()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the accepted D2 is recorded in project.json (Follow-up Task 2 fix round 1) | date: 2026-09-29]
    days = project.read()['days']
    assert days['D2']['date'] == d2.date
    assert days['D2']['date_source'] == 'user'
    # [AGENT EDIT END]
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: init claims the probe before it adds the day, so a failed day leaves the probe and a rerun completes it (Follow-up Task 2) | date: 2026-09-29]
def test_init_probe_claimed_before_day(tmp_path, monkeypatch):
    root = tmp_path / 'lab'
    d1 = write_day(tmp_path / 'in', 'D1')

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the probe is claimed before add_day starts (Follow-up Task 2 fix round 1) | date: 2026-09-29]
    def raising(opened, day, **kwargs):
        manifest = opened.read()
        assert manifest['identity']['probe'] == 'NP1'
        assert 'D1' not in manifest['days']
        entered.append(day)
        raise OSError('disk full')

    entered = []
    # [AGENT EDIT END]
    monkeypatch.setattr(commands, 'add_day', raising)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - add_day was entered once and its own error surfaced (Follow-up Task 2 fix round 1) | date: 2026-09-29]
    with pytest.raises(OSError, match='disk full'):
        commands.init_day(root, 'P', day='D1', sort=str(d1.sort_dir), date=d1.date, probe='NP1')
    assert entered == ['D1']
    # [AGENT EDIT END]
    manifest = Project.open(root, 'P').read()
    assert manifest['identity']['probe'] == 'NP1'
    assert 'D1' not in manifest['days']
    monkeypatch.undo()
    commands.init_day(root, 'P', day='D1', sort=str(d1.sort_dir), date=d1.date, probe='NP1')
    manifest = Project.open(root, 'P').read()
    assert 'D1' in manifest['days']
    assert manifest['identity']['probe'] == 'NP1'


def test_init_probe_conflict_checked_under_lock(tmp_path, monkeypatch):
    root = tmp_path / 'lab'
    d1 = write_day(tmp_path / 'in', 'D1')
    Project.create(root, 'P')
    real_update = Project.update
    calls = []
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the refusals raised under the lock are collected (Follow-up Task 2 fix round 1) | date: 2026-09-29]
    refused = []
    # [AGENT EDIT END]

    def racing_update(self, change):
        calls.append(change)
        if len(calls) == 1:  # another process claims NP2 between the unlocked read and the lock
            def claim(manifest: dict) -> None:
                manifest['identity']['probe'] = 'NP2'
            real_update(self, claim)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the caller's change raises the conflict while project.lock is held (Follow-up Task 2 fix round 1) | date: 2026-09-29]
        def checked(manifest: dict) -> None:
            assert (self.path / 'project.lock').exists()
            with pytest.raises(ValueError) as inside:
                change(manifest)
            refused.append(str(inside.value))
            raise inside.value

        return real_update(self, checked)
        # [AGENT EDIT END]

    monkeypatch.setattr(Project, 'update', racing_update)
    with pytest.raises(ValueError) as caught:
        commands.init_day(root, 'P', day='D1', sort=str(d1.sort_dir), date=d1.date, probe='NP1')
    assert str(caught.value) == 'project P probe is NP2; --probe says NP1'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the refusal came from inside the locked update (Follow-up Task 2 fix round 1) | date: 2026-09-29]
    assert refused == ['project P probe is NP2; --probe says NP1']
    # [AGENT EDIT END]
    project = Project.open(root, 'P')
    manifest = project.read()
    assert manifest['identity']['probe'] == 'NP2'
    assert 'D1' not in manifest['days']
    assert not project.day_dir('D1').exists()
# [AGENT EDIT END]


def test_cli_run_prints_run_line(weak, capsys):
    before = _runs(weak)
    assert main(['--root', str(weak.root), 'run', 'P', '--backend', FL]) == 0
    (run,) = _runs(weak) - before
    assert capsys.readouterr().out == f'{run} · {FL} · D1,D2 · excluded 1\n'


def test_cli_run_days_needs_two(weak, capsys):
    before = _runs(weak)
    assert main(['--root', str(weak.root), 'run', 'P', '--backend', FL, '--days', 'D1']) == 1
    assert capsys.readouterr().err.rstrip('\n').endswith('error: --days needs two or more days; use --day for one')
    assert _runs(weak) == before


def test_cli_run_unknown_backend(weak, capsys):
    assert main(['--root', str(weak.root), 'run', 'P', '--backend', 'nope']) == 1
    err = capsys.readouterr().err
    assert 'error: unknown backend nope; known: ' in err
    assert FL in err


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run prints each evidence fallback as a warning (event-optional Task 4) | date: 2026-10-09]
def test_launch_run_prints_evidence_warnings(tmp_path, monkeypatch, capsys):
    project = _open(extracted_project(tmp_path, conditions={'D2': ('C', 'D')}))
    monkeypatch.setattr(commands, 'run_selection', lambda *args, **kwargs: None)
    commands.run(project.root, 'P', backend=FL)
    assert 'warning · no events · shared conditions per day pair: D1+D2 0' in capsys.readouterr().err.splitlines()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run refuses a day without a single unit before it prints any evidence warning (event-optional Task 14) | date: 2026-10-10]
def test_launch_run_refuses_before_warnings(tmp_path, monkeypatch, capsys):
    project = _open(extracted_project(tmp_path, conditions={'D2': ('C', 'D')}))
    dataset = Selection.dataset

    def relabelled(self):
        loaded = dataset(self)
        loaded.units.loc[loaded.units.session_id == 'D2', 'su_type'] = 'MU'
        return loaded

    monkeypatch.setattr(Selection, 'dataset', relabelled)
    calls = []
    monkeypatch.setattr(commands, 'run_selection', lambda *args, **kwargs: calls.append((args, kwargs)))
    capsys.readouterr()
    with pytest.raises(ValueError) as caught:
        commands.run(project.root, 'P', backend=FL)
    assert str(caught.value) == 'days without a single unit: D2; run without them'
    assert not [line for line in capsys.readouterr().err.splitlines() if line.startswith('warning · ')]
    assert calls == []
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run warns about proposed same-day links in the saved result (native-same-day Task 6) | date: 2026-10-11]
@pytest.mark.parametrize('n', (0, 1, 2))
def test_launch_run_counts_proposed_same_day_links(tmp_path, monkeypatch, capsys, n):
    project = _open(extracted_project(tmp_path))

    def fake(selection, backend, options, output_dir):
        output_dir.mkdir(parents=True)  # project_run_path creates nothing
        (output_dir / 'result.json').write_text('{}')

    same_day = [('D1-a', 'D1-b'), ('D1-a', 'D1-c')][:n]
    rows = same_day + [('D1-a', 'D2-a')]
    pairs = pd.DataFrame({'cellid_a': [a for a, _ in rows], 'cellid_b': [b for _, b in rows],
                          'decision': ['proposed_link'] * len(rows)})
    loaded = SimpleNamespace(units=pd.DataFrame({'cellid': ['D1-a', 'D1-b', 'D1-c', 'D2-a'],
                                                 'session_id': ['D1', 'D1', 'D1', 'D2']}),
                             pairs=pairs)
    monkeypatch.setattr(commands, 'run_selection', fake)
    monkeypatch.setattr(commands, 'load_result', lambda path: loaded)
    capsys.readouterr()
    commands.run(project.root, 'P', backend=FL)
    lines = [line for line in capsys.readouterr().err.splitlines() if 'proposed same-day' in line]
    expected = {0: [],
                1: ['warning · 1 proposed same-day link · review in the viewer'],
                2: ['warning · 2 proposed same-day links · review in the viewer']}
    assert lines == expected[n]
# [AGENT EDIT END]


def test_launch_run_signatures():
    for function in (commands.run, commands.init_day, commands.init_from):
        hints = typing.get_type_hints(function)
        assert hints['return'] is dict, function.__name__
    for function in (commands.init_day, commands.init_from):
        assert typing.get_type_hints(function)['probe'] == Optional[str], function.__name__
    hints = typing.get_type_hints(commands.run)
    assert hints['days'] == Optional[list]
    assert hints['options'] == Optional[dict]
    assert hints['python'] == Optional[str]
# [AGENT EDIT END]
