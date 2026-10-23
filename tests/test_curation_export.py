"""Accepted tracks leave a run as one CSV row per member; an export never overwrites and never leaves a partial file."""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from multiphy import curation
from multiphy.clusters import cluster_id
from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection

from .synthetic_day import extracted_project

FL = 'functional_likelihood'
TRACKS = [['D1:0', 'D2:0', 'D3:0'], ['D1:2', 'D2:2']]
UNMATCHED = ['D1:1']
UNITS = [m for t in TRACKS for m in t] + UNMATCHED
SITES = {m: m.split(':')[0] for m in UNITS}
T3 = cluster_id(TRACKS[0])
SINGLE = cluster_id(['D1:1'])


def _state(*actions):
    rows = [curation.make_row(op, cluster_id=tid) for op, tid in actions]
    return curation.replay(TRACKS, UNMATCHED, rows)


def test_accepted_rows_hand_counted():
    rows = curation.accepted_rows(_state(('accept', T3), ('accept', SINGLE)), SITES)
    assert len(rows) == 4
    by_cell = rows.set_index('cellid')
    for cellid in ('D1:0', 'D2:0', 'D3:0'):
        assert (by_cell.loc[cellid, 'n_units'], by_cell.loc[cellid, 'n_days']) == (3, 3)
    assert (by_cell.loc['D1:1', 'n_units'], by_cell.loc['D1:1', 'n_days']) == (1, 1)
    assert 'D1:2' not in by_cell.index and 'D2:2' not in by_cell.index
    assert set(rows.review_status) == {'accepted'}
    assert set(rows.backend_track_id) == set(rows.backend_id) == set(rows.backend_run_id) == {''}


def test_accepted_rows_preserve_backend_assignments_after_merge():
    result = SimpleNamespace(tracks=TRACKS, backend_id='example_backend', run_id='original_run')
    merged = cluster_id(TRACKS[0] + TRACKS[1])
    state = curation.replay(TRACKS, UNMATCHED, [
        curation.make_row('merge', parent_ids=[cluster_id(track) for track in TRACKS]),
        curation.make_row('accept', cluster_id=merged),
    ])
    rows = curation.accepted_rows(state, SITES, result=result)
    assert set(rows.track_id) == {merged}
    assert set(rows.review_status) == {'accepted'}
    assert set(rows.backend_id) == {'example_backend'}
    assert set(rows.backend_run_id) == {'original_run'}
    by_cell = rows.set_index('cellid')
    assert set(by_cell.index) == set(TRACKS[0] + TRACKS[1])
    for track in TRACKS:
        assert set(by_cell.loc[track, 'backend_track_id']) == {cluster_id(track)}
    assert result.tracks == TRACKS


def test_manually_linked_unmatched_units_export_after_acceptance(tmp_path):
    units = ['D1:1', 'D2:1']
    sites = dict(zip(units, ('D1', 'D2')))
    result = SimpleNamespace(tracks=[], unmatched=units)
    merged = cluster_id(units)
    link = curation.make_row('merge', parent_ids=[cluster_id([unit]) for unit in units])
    linked = curation.replay([], units, [link])
    assert curation.accepted_rows(linked, sites, result=result).empty
    accepted = curation.replay([], units, [link, curation.make_row('accept', cluster_id=merged)])
    out = curation.write_export(accepted, sites, tmp_path, result=result)
    rows = pd.read_csv(out, keep_default_na=False)
    assert set(rows.cellid) == set(units)
    assert set(rows.track_id) == {merged}
    assert set(rows.backend_track_id) == set(rows.backend_id) == set(rows.backend_run_id) == {''}
    assert set(rows.review_status) == {'accepted'}


def test_editing_accepted_membership_requires_approval_again():
    state = curation.replay(TRACKS, UNMATCHED, [
        curation.make_row('accept', cluster_id=T3),
        curation.make_row('remove_member', cluster_id=T3, members=['D1:0']),
    ])
    assert curation.accepted_rows(state, SITES, result=SimpleNamespace(tracks=TRACKS)).empty


def test_reopened_track_not_exported():
    assert len(curation.accepted_rows(_state(('accept', T3), ('reopen', T3)), SITES)) == 0


def test_export_refuses_existing_out(tmp_path):
    out = tmp_path / 'out.csv'
    out.write_text('keep')
    with pytest.raises(FileExistsError):
        curation.write_export(_state(('accept', T3)), SITES, tmp_path, out)
    assert out.read_text() == 'keep'


def test_export_failure_leaves_no_file(tmp_path, monkeypatch):
    out = tmp_path / 'exports' / 'out.csv'

    def fail(*args, **kwargs):
        raise OSError('disk full')

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail)
    with pytest.raises(OSError):
        curation.write_export(_state(('accept', T3)), SITES, tmp_path, out)
    assert not out.exists()


def test_export_cleanup_failure_is_raised(tmp_path, monkeypatch):
    out = tmp_path / 'exports' / 'out.csv'

    def fail_write(*args, **kwargs):
        raise OSError('disk full')

    def fail_remove(path):
        raise PermissionError(13, 'read-only', str(path))

    monkeypatch.setattr(pd.DataFrame, 'to_csv', fail_write)
    monkeypatch.setattr(curation.os, 'remove', fail_remove)
    with pytest.raises(PermissionError) as raised:
        curation.write_export(_state(('accept', T3)), SITES, tmp_path, out)
    assert raised.value.filename == str(out)
    assert str(raised.value.__context__) == 'disk full'


def test_export_missing_site_creates_nothing(tmp_path):
    out = tmp_path / 'exports' / 'out.csv'
    sites = {k: v for k, v in SITES.items() if k != 'D1:0'}
    with pytest.raises(KeyError):
        curation.write_export(_state(('accept', T3)), sites, tmp_path, out)
    assert not out.exists()


@pytest.fixture(scope='module')
def project(tmp_path_factory) -> Project:
    """D1 and D2 extracted; the tests add their own runs."""
    path = extracted_project(tmp_path_factory.mktemp('export'))
    return Project.open(path.parent, path.name)


def test_export_run_default_path(project):
    name = run_selection(Selection(project, ('D1', 'D2')), FL).name
    out = curation.export_run(project.path, name)
    assert out.parent == project.path / 'curation' / 'runs' / name / 'exports'
    assert re.fullmatch(r'accepted_tracks_.{15}\.csv', out.name), out.name
    assert out.read_text().splitlines() == [
        'track_id,cellid,siteid,n_units,n_days,accepted_at,curator,'
        'backend_track_id,review_status,backend_id,backend_run_id']


def test_export_run_uses_original_result_provenance(project):
    from multiphy import backends

    run = run_selection(Selection(project, ('D1', 'D2')), FL)
    result = backends.load_result(run / 'result.json')
    members = result.tracks[0] if result.tracks else [result.unmatched[0]]
    tid = cluster_id(members)
    controller = curation.Curation.open(project.path / 'curation' / 'runs' / run.name, result)
    try:
        controller.do('accept', cluster_id=tid)
    finally:
        controller.close()

    rows = pd.read_csv(curation.export_run(project.path, run.name), keep_default_na=False)
    assert set(rows.cellid) == set(members)
    assert set(rows.track_id) == {tid}
    assert set(rows.backend_track_id) == {tid if result.tracks else ''}
    assert set(rows.backend_id) == {result.backend_id}
    assert set(rows.backend_run_id) == {result.run_id}


def test_export_run_refuses_one_day_run(project):
    name = run_selection(Selection(project, ('D1',)), FL).name
    with pytest.raises(ValueError, match='not a completed run'):
        curation.export_run(project.path, name)


def test_export_run_refuses_one_day_run_in_project_runs(project):
    from multiphy.project.folder import RUNS_DIR

    run_selection(Selection(project, ('D1',)), FL, output_dir=project.path / RUNS_DIR / 'one_day')
    with pytest.raises(ValueError, match='not an across-day run'):
        curation.export_run(project.path, 'one_day')
    assert not (project.path / 'curation' / 'runs' / 'one_day' / 'exports').exists()


def test_export_run_refuses_unknown_name(project):
    with pytest.raises(ValueError, match='not a completed run'):
        curation.export_run(project.path, 'no_such_run')


def test_export_run_refuses_path_like_name(project):
    with pytest.raises(ValueError, match='run folder name'):
        curation.export_run(project.path, '../x')
