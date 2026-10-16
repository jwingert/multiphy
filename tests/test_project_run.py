"""A project run records its selection and pairs before the backend starts, then runs through ``runner.run``."""
from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from pathlib import Path
import shutil
import sys
import typing
from typing import Optional

import pandas as pd
import pytest

from multiphy.contract import registry
from multiphy.contract.backend import BackendSpec
from multiphy.launch import commands
from multiphy.project.folder import RUNS_DIR, Project
from multiphy.project.selection import Selection
from multiphy.project.selection import HALVES_SESSIONS
from multiphy.run import run_dir, runner
from multiphy.run import subprocess as sp
from multiphy.run.__main__ import main
from multiphy.run.project_run import (check_real_day_run, check_single_unit_days, project_run_path,
                                                  run_selection, selection_runs)
from multiphy.run.subprocess import build_selection_command

from .synthetic_day import extracted_project

FL = 'functional_likelihood'


def _open(project_dir: Path) -> Project:
    return Project.open(project_dir.parent, project_dir.name)


def _manifest(folder: Path) -> dict:
    return json.loads((folder / 'run_manifest.json').read_text())


@pytest.fixture(scope='module')
def project(tmp_path_factory) -> Project:
    """D1 and D2 extracted from raw, with the default conditions A and B on both days; only the CLI test runs into its runs/."""
    return _open(extracted_project(tmp_path_factory.mktemp('project')))


def test_across_days_run_writes_selection_and_pairs(tmp_path):
    project = _open(extracted_project(tmp_path))
    selection = Selection(project, ('D1', 'D2'))
    output = run_selection(selection, FL)
    assert output.parent == project.path / RUNS_DIR
    saved = (output / 'selection.json').read_bytes()
    assert saved == selection.record_bytes()
    assert len(pd.read_csv(output / 'pairs.csv')) == 9
    assert (output / 'availability.json').is_file()
    assert _manifest(output)['input_manifest_sha256'] == hashlib.sha256(saved).hexdigest()
    assert _manifest(output)['status'] == 'completed'


def test_same_day_run_evaluates_all_partition(tmp_path):
    project = _open(extracted_project(tmp_path, ('D1',)))
    output = run_selection(Selection(project, ('D1',)), FL)
    assert output.parent == project.day_dir('D1') / RUNS_DIR
    assert _manifest(output)['partition'] == 'all'
    # the synthetic day has 3 good clusters (0-2); each is split into 2 temporal halves: 6 units,
    # 3 x 3 cross-half pairs, 3 of them the same source cluster
    combined = json.loads((output / 'evaluation.json').read_text())['combined']
    assert (combined['n_units'], combined['n_known_pairs'], combined['n_known_positive_pairs']) == (6, 9, 3)
    assert combined['true_positive_pairs'] + combined['false_negative_pairs'] == 3
    assert combined['false_positive_pairs'] + combined['true_negative_pairs'] == 6
    assert combined['evaluable_source_identities'] == 3


def test_no_shared_condition_allowed(tmp_path):
    project = _open(extracted_project(tmp_path, ('D1', 'D2', 'D3'), conditions={'D3': ('C', 'D')}))
    selection = Selection(project, ('D1', 'D2', 'D3'))
    assert check_real_day_run(selection, registry.get(FL).spec) is None
    assert selection.evidence().reasons == ('no events · shared conditions per day pair: D1+D2 2 · D1+D3 0 · D2+D3 0',)


def test_pairwise_overlap_allowed_without_events(tmp_path):
    conditions = {'D1': ('A', 'B'), 'D2': ('B', 'C'), 'D3': ('A', 'C')}
    project = _open(extracted_project(tmp_path, ('D1', 'D2', 'D3'), conditions=conditions))
    selection = Selection(project, ('D1', 'D2', 'D3'))
    assert check_real_day_run(selection, registry.get(FL).spec) is None
    assert selection.evidence().reasons == ('no events · shared conditions per day pair: D1+D2 1 · D1+D3 1 · D2+D3 1',)


# 3 good units a day give at most 6 same-day neighbour pairs, so these runs lower min_pairs (D53)
SMALL = {'min_spikes': 2, 'n_splits': 2, 'min_pairs': 1}


def _saved_evidence(output: Path) -> dict:
    """The one recording group's ``run_evidence`` saved in the run's ``result.json``."""
    result = run_dir.load_result(output / 'result.json')
    (group,) = result.provenance.get('group_runs').values()
    (evidence,) = group['provenance']['run_evidence'].values()
    return evidence


def test_mixed_project_runs_without_events(tmp_path):
    project = _open(extracted_project(tmp_path, no_event_days=('D2',)))
    output = run_selection(Selection(project, ('D1', 'D2')), FL, dict(SMALL))
    assert _manifest(output)['status'] == 'completed'
    evidence = _saved_evidence(output)
    assert evidence['events'] is False and evidence['spont'] == 'whole_recording'
    assert evidence['reasons'] == ['no events · shared conditions per day pair: D1+D2 0',
                                   'spont · whole recording · days without baseline windows: D2']


def test_launch_warnings_equal_saved_reasons(tmp_path, capsys):
    project = _open(extracted_project(tmp_path, no_event_days=('D2',)))
    capsys.readouterr()
    out = commands.run(project.root, 'P', backend=FL, days=['D1', 'D2'], options=dict(SMALL))
    prefix = 'warning · '
    warnings = [line[len(prefix):] for line in capsys.readouterr().err.splitlines() if line.startswith(prefix)]
    reasons = _saved_evidence(out['run'])['reasons']
    assert len(reasons) == 2 and warnings == reasons


def test_no_shared_condition_runs_without_events(tmp_path):
    project = _open(extracted_project(tmp_path, ('D1', 'D2', 'D3'), conditions={'D3': ('C', 'D')}))
    output = run_selection(Selection(project, ('D1', 'D2', 'D3')), FL, dict(SMALL))
    assert _saved_evidence(output) == {
        'events': False, 'spont': 'baseline', 'shared': {'D1+D2': 2, 'D1+D3': 0, 'D2+D3': 0},
        'reasons': ['no events · shared conditions per day pair: D1+D2 2 · D1+D3 0 · D2+D3 0']}


def test_pairwise_overlap_runs_without_events(tmp_path):
    conditions = {'D1': ('A', 'B'), 'D2': ('B', 'C'), 'D3': ('A', 'C')}
    project = _open(extracted_project(tmp_path, ('D1', 'D2', 'D3'), conditions=conditions))
    output = run_selection(Selection(project, ('D1', 'D2', 'D3')), FL, dict(SMALL))
    evidence = _saved_evidence(output)
    assert evidence['events'] is False
    assert evidence['shared'] == {'D1+D2': 1, 'D1+D3': 1, 'D2+D3': 1}


def test_benchmark_baseline_refused_across_days(project):
    spec = dataclasses.replace(registry.get(FL).spec, id='fake', real_days=False)
    with pytest.raises(ValueError) as caught:
        check_real_day_run(Selection(project, ('D1', 'D2')), spec)
    assert str(caught.value) == 'fake is a benchmark baseline (real_days False); it cannot run across days'
    assert check_real_day_run(Selection(project, ('D1',)), spec) is None


def _without_single_units(monkeypatch, day: str) -> None:
    """``Selection.dataset`` labels every unit of ``day`` multi-unit."""
    dataset = Selection.dataset

    def relabelled(self):
        loaded = dataset(self)
        loaded.units.loc[loaded.units.session_id == day, 'su_type'] = 'MU'
        return loaded

    monkeypatch.setattr(Selection, 'dataset', relabelled)


def test_day_without_single_unit_refused(tmp_path, monkeypatch):
    project = _open(extracted_project(tmp_path))
    _without_single_units(monkeypatch, 'D2')
    with pytest.raises(ValueError) as caught:
        run_selection(Selection(project, ('D1', 'D2')), FL)
    assert str(caught.value) == 'days without a single unit: D2; run without them'
    assert not list((project.path / RUNS_DIR).glob('*'))


def test_same_day_half_without_single_unit(project):
    selection = Selection(project, ('D1',))
    first, second = HALVES_SESSIONS

    def units(*rows):
        return pd.DataFrame([dict(cellid=f'u{i}', session_id=session, su_type=kind)
                             for i, (session, kind) in enumerate(rows)])

    with pytest.raises(ValueError) as caught:
        check_single_unit_days(selection, units((first, 'SU'), (second, 'MU')))
    assert str(caught.value) == f'day D1 half {second} has no single unit'
    with pytest.raises(ValueError) as caught:
        check_single_unit_days(selection, units((first, 'MU'), (second, 'MU')))
    assert str(caught.value) == 'day D1 has no single unit'
    assert check_single_unit_days(selection, units((first, 'SU'), (second, 'SU'))) is None


def test_check_single_unit_days_signature():
    assert typing.get_type_hints(check_single_unit_days) == {'selection': Selection, 'units': pd.DataFrame,
                                                             'return': type(None)}


def test_files_written_before_manifest_survive_failure(tmp_path, monkeypatch):
    project = _open(extracted_project(tmp_path))
    selection = Selection(project, ('D1', 'D2'))
    expected = (selection.record_bytes(), selection.dataset().pairs.to_csv(index=False).encode())
    seen = []
    write_json = runner.write_json

    def contents(folder):
        return tuple((folder / name).read_bytes() if (folder / name).is_file() else None
                     for name in ('selection.json', 'pairs.csv'))

    def spy(path, *args, **kwargs):
        if not seen:
            seen.append(contents(Path(path).parent))
        return write_json(path, *args, **kwargs)

    monkeypatch.setattr(runner, 'write_json', spy)
    with pytest.raises(ValueError, match='0 recording groups can be matched'):
        run_selection(selection, 'pair_logistic')
    assert seen == [expected]
    (output,) = list((project.path / RUNS_DIR).iterdir())
    assert contents(output) == expected
    assert _manifest(output)['status'] == 'failed'


def test_reserved_file_names_refused(project, tmp_path):
    selection = Selection(project, ('D1', 'D2'))
    dataset, timing = selection.dataset(), selection.timing()
    output = tmp_path / 'out'
    for name in ('result.json', 'a/b', '', '.', '..', 'a\0b'):
        with pytest.raises(ValueError, match='reserved'):
            runner.run(dataset, FL, group_column='recording_group', output_dir=output, root=project.root,
                       evaluate=False, extra_inputs={'timing': timing}, files={name: b''})
        assert not output.exists()


def test_build_command_forwards_root(tmp_path):
    src = tmp_path / 'ds'
    src.mkdir()
    (src / 'manifest.json').write_text('{}')
    _, argv = sp.build_command(src, FL, {}, output=tmp_path / 'o', root=tmp_path)
    assert argv[-2:] == ['--root', str(tmp_path)]
    _, argv = sp.build_command(src, FL, {}, output=tmp_path / 'o', root=None)
    assert '--root' not in argv


def test_build_selection_command_argv():
    # Do 6 appends every validated option; these are functional_likelihood's defaults.
    defaults = ['--min-repeats', '5', '--min-spikes', '200', '--n-splits', '20', '--seed', '0', '--radius-um', '80.0',
                '--min-pairs', '30', '--registration', 'matches', '--feature-stream', 'combined', '--threshold', '0.9',
                '--contradiction-threshold', '0.1', '--ambiguity-margin', '0.02']
    program, argv = build_selection_command(Path('/x/P'), ('D1',), FL, {}, output=Path('/x/o'))
    assert program == str(Path(sys.executable).absolute())
    assert argv == ['-u', '-m', 'multiphy.run', 'run-backend', '--project', '/x/P', '--day', 'D1',
                    '--backend', 'functional_likelihood', '--output', '/x/o'] + defaults
    _, argv = build_selection_command(Path('/x/P'), ('D1', 'D2'), FL, {}, output=Path('/x/o'))
    assert argv == ['-u', '-m', 'multiphy.run', 'run-backend', '--project', '/x/P',
                    '--days', 'D1', 'D2', '--backend', 'functional_likelihood', '--output', '/x/o'] + defaults


def test_run_cli_project_mode_writes_run(project, capsys):
    o = str(project.root / 'cli_run')
    assert main(['run-backend', '--project', str(project.path), '--days', 'D1', 'D2', '--backend', FL,
                 '--output', o]) == 0
    assert capsys.readouterr().out.strip().splitlines()[-1] == f'Matching result: {o}/result.json'
    saved = (Path(o) / 'selection.json').read_bytes()
    assert saved == Selection(project, ('D1', 'D2')).record_bytes()
    result = run_dir.load_result(Path(o) / 'result.json')
    assert result.backend_id == FL and result.run_id == 'cli_run'
    manifest = _manifest(Path(o))
    assert manifest['status'] == 'completed'
    assert manifest['input_manifest_sha256'] == hashlib.sha256(saved).hexdigest()
    # a relative output is under the root, as runner.run resolves it
    assert run_selection(Selection(project, ('D1',)), FL, output_dir=Path('rel_run')) == project.root / 'rel_run'


def test_run_cli_refuses_archive_flags_with_project(project):
    output = project.root / 'refused'
    for flags in (['--project', 'P', '--partition', 'all'], ['--project', 'P', '--group-column', 'well'],
                  ['--project', 'P', '--group-column', ''], ['--project', 'P', '--no-evaluate'],
                  ['--input', 'I', '--project', 'P'], ['--project', 'P', '--day', 'D1', '--days', 'D1', 'D2']):
        flags = [str(project.path) if a == 'P' else a for a in flags]
        with pytest.raises(SystemExit) as caught:
            main(['run-backend', *flags, '--backend', FL, '--output', str(output)])
        assert caught.value.code == 2, flags
        assert not output.exists()


def test_selection_runs_lists_this_selection_only(tmp_path):
    project = _open(extracted_project(tmp_path, ('D1', 'D2', 'D3')))
    output = run_selection(Selection(project, ('D1', 'D2')), FL)
    other = run_selection(Selection(project, ('D1', 'D3')), FL)  # another selection, same P/runs
    assert other.parent == output.parent
    # the same dataset_id recorded under another selection hash: a stale record of D1+D2
    stale = output.parent / f'{output.name}_stale'
    shutil.copytree(output, stale)
    manifest = _manifest(stale)
    manifest['input_manifest_sha256'] = '0' * 64
    (stale / 'run_manifest.json').write_text(json.dumps(manifest))
    entries = selection_runs(Selection(project, ('D1', 'D2')))
    assert [e.folder for e in entries] == [output]
    assert [e.folder for e in selection_runs(Selection(project, ('D1', 'D3')))] == [other]
    assert selection_runs(Selection(project, ('D1',))) == []


def test_project_run_signatures():
    hints = typing.get_type_hints
    assert hints(check_real_day_run) == {'selection': Selection, 'spec': BackendSpec, 'return': type(None)}
    assert hints(project_run_path) == {'selection': Selection, 'backend_id': str, 'return': Path}
    assert hints(run_selection) == {'selection': Selection, 'backend_id': str, 'options': Optional[dict],
                                    'output_dir': Optional[Path], 'return': Path}
    assert hints(selection_runs) == {'selection': Selection, 'return': list}
    assert hints(build_selection_command) == {'project_path': Path, 'days': tuple, 'backend_id': str,
                                              'options': dict, 'output': Path, 'interpreter': Optional[str],
                                              'return': tuple}
    assert hints(sp.build_command)['return'] is tuple
    files = inspect.signature(runner.run).parameters['files']
    assert files.kind is inspect.Parameter.KEYWORD_ONLY and files.default is None
