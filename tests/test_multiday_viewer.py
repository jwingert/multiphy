# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the days viewer opens across-days project runs; multiday-archive tests dropped (Plan B Task 5) | date: 2026-09-28]
"""The days viewer opens a completed across-days project run and curates it in the project."""
import copy
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: dataclasses.replace swaps the shown result (viewer-split Task 8) | date: 2026-10-07]
import dataclasses
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: attempt 4 - rewrite a run manifest's status (Plan B Task 5) | date: 2026-09-28]
import json
# [AGENT EDIT END]
import os
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the settings pin test matches the settings folder name (day-table-keys Task 6) | date: 2026-10-14]
import re
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - copy a result out of its run folder (Plan B Task 5) | date: 2026-09-28]
import shutil
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the lock names its host (run-curation Task 7) | date: 2026-10-05]
import socket
# [AGENT EDIT END]
from pathlib import Path

import pytest

from multiphy import backends
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - the GUI imports are required, never skipped (Plan B Task 5) | date: 2026-09-28]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the window tests open runs through viewer.app; Qt constants come from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
from multiphy.viewer import app as viewer_app
from multiphy.viewer.qt import QtCore
# [AGENT EDIT END]
from PySide6 import QtWidgets
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a close event for the layout-failure test (run-curation Task 7 fix round 1) | date: 2026-10-05]
from PySide6 import QtGui as QtGui
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the load and validate tests read the across-days mode (viewer-split Task 3) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open_run and main of the one viewer; App and MODES for the empty window (viewer-split Task 8) | date: 2026-10-07]
from multiphy.viewer.app import App, main, open_run
from multiphy.viewer.modes import MODES
# [AGENT EDIT END]
from multiphy.viewer.modes import across_days
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the header text is patched on the Qt-free display module (viewer-split Task 8) | date: 2026-10-07]
from multiphy.viewer import display
# [AGENT EDIT END]
from multiphy.viewer.modes.across_days import validate_days_result
# [AGENT EDIT END]
from multiphy.project.folder import RUNS_DIR, Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection

from tests.synthetic_day import extracted_project
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the key-pressing tests read their shortcuts from a settings file under tmp_path (day-table-keys Task 4) | date: 2026-10-14]
from tests.theme_fixtures import theme_env  # noqa: F401
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every test reads settings from a fresh folder under the pytest base temp; the pin test reads the settings file name through theme (day-table-keys Task 6) | date: 2026-10-14]
from multiphy.viewer import theme
from tests.settings_fixtures import tmp_settings  # noqa: F401
# [AGENT EDIT END]

FL = 'functional_likelihood'
UNITS = {'D1:0': 'D1', 'D1:1': 'D1', 'D1:2': 'D1', 'D2:0': 'D2', 'D2:1': 'D2', 'D2:2': 'D2'}


@pytest.fixture(scope='module')
def project(tmp_path_factory):
    """D1 (with a weak unit, excluded) and D2; a functional_likelihood run of D1+D2, a failed
    pair_logistic run of D1+D2 and a functional_likelihood run of D1."""
    folder = extracted_project(tmp_path_factory.mktemp('project'), weak_units=('D1',))
    project = Project.open(folder.parent, folder.name)
    days = Selection(project, ('D1', 'D2'))
    days_run = run_selection(days, FL)
    with pytest.raises(ValueError):
        run_selection(days, 'pair_logistic')
    (failed_run,) = [p for p in (project.path / RUNS_DIR).iterdir() if '_pair_logistic_' in p.name]
    day_run = run_selection(Selection(project, ('D1',)), FL)
    return dict(project=project, days=days, days_run=days_run, failed_run=failed_run, day_run=day_run)


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - PySide6 is required, never skipped (Plan B Task 5) | date: 2026-09-28]
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    # [AGENT EDIT END]


def _result(run: Path):
    return backends.load_result(run / 'result.json')


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the across-days mode loads the run into RunData (viewer-split Task 3) | date: 2026-10-06]
def test_display_cache_meta_for_selection(project):
    dataset = project['days'].dataset()
    data = across_days.load(project['days'], dataset, _result(project['days_run']), {})
    assert [(d['siteid_a'], d['siteid_b'], d['days_apart']) for d in data.day_pairs()] == [('D1', 'D2', 1)]
    assert data.min_repeats() == 4
    assert data.day_name('D1') == '2026-01-01 · D1'
    assert data.title.endswith(' · days')
# [AGENT EDIT END]


def test_validate_days_result_refuses_other_selection(project):
    dataset = project['days'].dataset()
    result = _result(project['days_run'])
    assert validate_days_result(dataset, result) is result
    other = copy.deepcopy(result)
    other.provenance = other.provenance.merged(input_manifest_sha256='0' * 64)
    with pytest.raises(ValueError, match='result belongs to another selection'):
        validate_days_result(dataset, other)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: days-viewer tests that write curation start and end with no project curation folder (run-curation Task 5) | date: 2026-10-05]
@pytest.fixture
def clean_curation(project):
    folder = project['project'].path / 'curation'
    shutil.rmtree(folder, ignore_errors=True)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the days run is curated by one controller in the project's curation/runs/<run> (run-curation Task 5) | date: 2026-10-05]
def test_open_run_uses_project_curation(qt, project, clean_curation):
    window = open_run(project['days_run'])
    run_curation = project['project'].path / 'curation' / 'runs' / project['days_run'].name
    assert window.curation.curation_dir == run_curation
    assert (run_curation / 'history.lock').exists()
    # [AGENT EDIT END]
    assert window.state.sites == UNITS
    window.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - a same-day track and a changed listed run are refused (Plan B Task 5) | date: 2026-09-28]
def test_validate_days_result_refuses_same_day_track(project):
    dataset = project['days'].dataset()
    other = copy.deepcopy(_result(project['days_run']))
    members = sorted({u for t in other.tracks for u in t} | set(other.unmatched))
    other.tracks = [('D1:0', 'D1:1')]
    other.unmatched = tuple(u for u in members if u not in ('D1:0', 'D1:1'))
    with pytest.raises(ValueError, match='^a track holds two units from one day$'):
        validate_days_result(dataset, other)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - database guard (Plan B Task 11) | date: 2026-09-28]
def test_run_list_refuses_changed_selection(qt, tmp_path):
# [AGENT EDIT END]
    folder = extracted_project(tmp_path, weak_units=('D1',))
    days = Selection(Project.open(folder.parent, folder.name), ('D1', 'D2'))
    shown, changed = run_selection(days, FL), run_selection(days, FL)
    window = open_run(shown)
    key = (window.match_result.backend_id, window.match_result.run_id)
    selection_file = changed / 'selection.json'
    selection_file.chmod(0o644)
    selection_file.write_bytes(selection_file.read_bytes() + b' ')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - the Refresh button handler lists the runs (viewer-split Task 8) | date: 2026-10-07]
    window.refresh_btn.click()
    # [AGENT EDIT END]
    index = next(i for i, e in enumerate(window._run_items) if e is not None and e.result_path.parent == changed)
    window._run_chosen(index)
    assert window._shown_key == key
    assert (changed / 'result.json') in window._rejected_runs
    assert 'no longer matches project' in window.statusBar().currentMessage()
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - Import refuses a changed run and a result outside a run folder, and opens an unchanged run (Plan B Task 5) | date: 2026-09-28]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - database guard (Plan B Task 11) | date: 2026-09-28]
def test_import_refuses_changed_selection(qt, tmp_path):
# [AGENT EDIT END]
    folder = extracted_project(tmp_path / 'p', weak_units=('D1',))
    days = Selection(Project.open(folder.parent, folder.name), ('D1', 'D2'))
    shown, other, changed = run_selection(days, FL), run_selection(days, FL), run_selection(days, FL)
    window = open_run(shown)
    key = (window.match_result.backend_id, window.match_result.run_id)
    selection_file = changed / 'selection.json'
    selection_file.chmod(0o644)
    selection_file.write_bytes(selection_file.read_bytes() + b' ')
    with pytest.raises(ValueError, match='no longer matches project'):
        window.import_match_result(changed / 'result.json')
    loose = tmp_path / 'loose'
    loose.mkdir()
    shutil.copy(other / 'result.json', loose / 'result.json')
    with pytest.raises(FileNotFoundError, match='not a project run'):
        window.import_match_result(loose / 'result.json')
    assert window._shown_key == key
    assert all(e.result_path.parent != changed for e in window._imported_runs)
    imported = window.import_match_result(other / 'result.json')
    assert window._shown_key == (imported.backend_id, imported.run_id) != key
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: attempt 4 - Import refuses a run folder whose manifest is not completed, even when its result.json is valid (Plan B Task 5) | date: 2026-09-28]
def test_import_refuses_unfinished_run(qt, tmp_path):
    folder = extracted_project(tmp_path / 'p', weak_units=('D1',))
    days = Selection(Project.open(folder.parent, folder.name), ('D1', 'D2'))
    shown, unfinished = run_selection(days, FL), run_selection(days, FL)
    window = open_run(shown)
    key = (window.match_result.backend_id, window.match_result.run_id)
    manifest = unfinished / 'run_manifest.json'
    manifest.chmod(0o644)
    manifest.write_text(json.dumps(dict(json.loads(manifest.read_text()), status='failed')))
    assert backends.load_result(unfinished / 'result.json').run_id != key[1]
    with pytest.raises(ValueError, match='is not a completed run'):
        window.import_match_result(unfinished / 'result.json')
    assert window._shown_key == key
    assert all(e.result_path.parent != unfinished for e in window._imported_runs)
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - opening a run writes history.lock, so this test cleans the project's curation too (run-curation Task 5) | date: 2026-10-05]
def test_dated_labels_show_date_and_day(qt, project, clean_curation):
# [AGENT EDIT END]
    window = open_run(project['days_run'])
    panel = next(p for p in window.plots if p.panel_id == 'waveform')
    window.state.set_selection(['D1:0'])
    assert panel.label_for('D1:0') == '2026-01-01 · D1'
    window.close()


def test_open_run_refuses_failed_run(qt, project):
    with pytest.raises(ValueError, match='is not a completed run'):
        open_run(project['failed_run'])


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open_run picks the mode the selection records: an across-days run curates, a same-day run is inspect-only (viewer-split Task 8) | date: 2026-10-07]
def test_open_run_picks_mode_from_selection(qt, project, clean_curation):
    window = open_run(project['days_run'])
    assert window.mode.id == 'across_days'
    assert window.curation is not None
    window.close()
    window = open_run(project['day_run'])
    assert window.mode.id == 'same_day'
    assert window.curation is None
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Run… button is gone, so its disabled-state test is removed (run-curation Task 5, D25) | date: 2026-10-05]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - the error names --run, --input is gone, and --run DIR opens that run (Plan B Task 5) | date: 2026-09-28]
def test_main_requires_run(qt, project, capsys, monkeypatch):
    with pytest.raises(SystemExit) as caught:
        main([])
    assert caught.value.code == 2
    assert 'the following arguments are required: --run' in capsys.readouterr().err
    with pytest.raises(SystemExit) as caught:
        main(['--input', str(project['days_run'])])
    assert caught.value.code == 2
    assert '--run' in capsys.readouterr().err
    opened = []
    class Window:
        def show(self):
            opened.append('shown')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: main opens through viewer.app.open_run (viewer-split Task 8) | date: 2026-10-07]
    monkeypatch.setattr(viewer_app, 'open_run', lambda run: opened.append(run) or Window())
    # [AGENT EDIT END]
    monkeypatch.setattr(QtWidgets.QApplication, 'exec', lambda self: 0)
    assert main(['--run', str(project['days_run'])]) == 0
    assert opened == [project['days_run'], 'shown']
# [AGENT EDIT END]


def test_gui_has_no_multiday_archive_opener():
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the App window has no archive opener (viewer-split Task 8) | date: 2026-10-07]
    assert hasattr(App, 'open_multiday_archive') is False
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the curator reads and writes through the run's controller (run-curation Task 5) | date: 2026-10-05]
TRACK = ('D1:0', 'D2:0')


@pytest.fixture(scope='module')
def tracked_run(project):
    """A copy of the days run whose result proposes D1:0+D2:0 as one track, with that pair as its candidate
    (the synthetic functional_likelihood run proposes no track and no pair)."""
    from multiphy.run.run_dir import save_result
    source = project['days_run']
    run = source.parent / (source.name + '_tracked')
    shutil.copytree(source, run)
    result = copy.deepcopy(_result(source))
    result.tracks = (TRACK,)
    result.unmatched = tuple(u for u in result.unmatched if u not in TRACK)
    result.pairs = result.pairs.iloc[:0].copy()
    result.pairs.loc[0] = {c: None for c in result.pairs.columns}
    result.pairs.loc[0, ['cellid_a', 'cellid_b']] = list(TRACK)
    for column in ('p_global', 'p_stim', 'p_phys', 'match_score'):
        if column in result.pairs:
            result.pairs[column] = [.9]
    result.pairs['selected'] = [True]
    result.pairs['decision'] = ['selected']
    (run / 'result.json').chmod(0o644)
    (run / 'result.json').unlink()
    save_result(result, run / 'result.json')
    return run


def _rows(panel):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    return {panel.table.item(i, 0).data(QtCore.Qt.UserRole): panel.table.item(i, 4).text()
            for i in range(panel.table.rowCount())}
    # [AGENT EDIT END]


def _select(panel, track_id):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    row = next(i for i in range(panel.table.rowCount())
               if panel.table.item(i, 0).data(QtCore.Qt.UserRole) == track_id)
    # [AGENT EDIT END]
    panel.table.setCurrentCell(row, 0)
    panel.table.selectRow(row)


def test_reject_shows_singletons(qt, tracked_run, clean_curation):
    from multiphy.clusters import cluster_id
    window = open_run(tracked_run)
    panel = window.tracked
    track_id, track = next((t, v) for t, v in window.track_state().tracks.items() if len(v['members']) >= 2)
    panel.hide_done.setChecked(True)
    _select(panel, track_id)
    panel.reject()
    rows = _rows(panel)
    assert track_id not in rows
    assert all(rows[cluster_id([m])] == 'unmatched' for m in track['members'])
    shown = len(rows)
    panel.hide_done.setChecked(False)
    rows = _rows(panel)
    assert len(rows) == shown + 1
    assert rows[track_id] == 'rejected'
    window.close()


def test_reject_disabled_on_singleton(qt, tracked_run, clean_curation):
    from multiphy.clusters import cluster_id
    window = open_run(tracked_run)
    panel = window.tracked
    _select(panel, cluster_id(TRACK))
    assert panel.buttons['Reject'].isEnabled() is True
    _select(panel, cluster_id(['D1:1']))
    assert panel.buttons['Reject'].isEnabled() is False
    window.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: E accepts the pair; the keys come from a settings file under tmp_path; tag wraps the whole test (day-table-keys Task 4) | date: 2026-10-14]
def test_verdict_appends_history_row(qt, tracked_run, clean_curation, theme_env):
    import csv
    from PySide6.QtTest import QTest
    window = open_run(tracked_run)
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window, 2000)
    assert window.state.row is not None
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    QTest.keyClick(window, QtCore.Qt.Key_E)
    # [AGENT EDIT END]
    history = clean_curation / 'runs' / tracked_run.name / 'history.csv'
    with open(history, newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 1
    assert (rows[0]['operation'], rows[0]['verdict']) == ('verdict', 'accept')
    window.close()
# [AGENT EDIT END]


def test_skipped_tag_lists_op(qt, project, clean_curation):
    import csv
    from multiphy import curation
    folder = clean_curation / 'runs' / project['days_run'].name
    folder.mkdir(parents=True)
    row = curation.make_row('split', cluster_id='uc_000000000000', members=['D1:0'])
    with open(folder / 'history.csv', 'w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=curation.COLUMNS, lineterminator='\n')
        writer.writeheader()
        writer.writerow(row)
    window = open_run(project['days_run'])
    tag = window.tracked.skipped_tag
    assert not tag.isHidden()
    assert tag.text() == 'skipped 1'
    assert row['op_id'] in tag.toolTip()
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - a failed write that is not a refusal (ENOSPC) still disables every mutation control (run-curation Task 5) | date: 2026-10-05]
def test_failed_write_disables_controls(qt, tracked_run, clean_curation, monkeypatch):
    import errno
    from multiphy import curation
    from multiphy.clusters import cluster_id
    window = open_run(tracked_run)
    panel = window.tracked
    _select(panel, cluster_id(TRACK))
    assert panel.buttons['Accept'].isEnabled() is True

    def full(*args):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(curation.os, 'replace', full)
    with pytest.raises(OSError):
        panel.accept()
    assert window.curation.read_only is True
    assert not any(b.isEnabled() for b in panel.buttons.values())
    assert not any(b.isEnabled() for b in window.evidence.verdict_buttons)
    monkeypatch.undo()
    window.close()


def test_failed_verdict_write_disables_controls(qt, tracked_run, clean_curation, monkeypatch):
    import errno
    from multiphy import curation
    window = open_run(tracked_run)
    assert window.state.row is not None
    assert all(b.isEnabled() for b in window.evidence.verdict_buttons)

    def full(*args):
        raise OSError(errno.ENOSPC, 'No space left on device')
    monkeypatch.setattr(curation.os, 'replace', full)
    with pytest.raises(OSError):
        window.set_verdict('accept')
    assert not any(b.isEnabled() for b in window.evidence.verdict_buttons)
    assert not any(s.isEnabled() for s in window.verdict_shortcuts)
    assert not window.tracked.buttons['Accept'].isEnabled()
    monkeypatch.undo()
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - a result that fails display preparation leaves the shown run's controller in place (run-curation Task 5) | date: 2026-10-05]
def test_failed_result_load_keeps_controller(qt, tracked_run, clean_curation):
    window = open_run(tracked_run)
    shown, controller = window.match_result, window.curation
    bad = copy.deepcopy(shown)
    bad.provenance = bad.provenance.merged(extra={'threshold': None})
    bad.validate()
    other = clean_curation / 'runs' / 'other_run'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the bad result reaches set_data inside the shown run data (viewer-split Task 8) | date: 2026-10-07]
    with pytest.raises(TypeError):
        window.set_data(dataclasses.replace(window.state.data, result=bad), curation_dir=other)
    # [AGENT EDIT END]
    assert window.curation is controller
    assert window.curation.curation_dir == clean_curation / 'runs' / tracked_run.name
    assert window.match_result is shown
    assert not other.exists()
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a result whose header text fails leaves the shown run, controller and lock in place (run-curation Task 7 review 3, D37) | date: 2026-10-05]
def test_failed_header_text_keeps_old_run(qt, tracked_run, clean_curation, monkeypatch):
    window = open_run(tracked_run)
    shown, controller = window.match_result, window.curation
    lock = clean_curation / 'runs' / tracked_run.name / 'history.lock'
    other = clean_curation / 'runs' / 'other_run'

    def broken(result):
        raise AttributeError('no calibrated model')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the header text comes from display.result_summary; the shown run data is set again (viewer-split Task 8) | date: 2026-10-07]
    monkeypatch.setattr(display, 'result_summary', broken)
    with pytest.raises(AttributeError):
        window.set_data(window.state.data, curation_dir=other)
    # [AGENT EDIT END]
    assert window.curation is controller
    assert window.match_result is shown
    assert lock.exists()
    assert not other.exists()
    monkeypatch.undo()
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - moved from test_curation.py: a new window has no run, so the inspect-only tag shows and every mutation control is disabled (run-curation Task 5) | date: 2026-10-05]
def test_new_window_starts_inspect_only(qt, tmp_path, monkeypatch):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a new across-days App with its layout file under tmp_path (viewer-split Task 8) | date: 2026-10-07]
    window = App(MODES['across_days'], layout_file=tmp_path / 'layout.json')
    # [AGENT EDIT END]
    assert window.curation is None
    assert not window.tracked.inspect_tag.isHidden()
    assert not any(b.isEnabled() for b in window.tracked.buttons.values())
    assert not any(b.isEnabled() for b in window.evidence.verdict_buttons)
    assert not any(s.isEnabled() for s in window.verdict_shortcuts)
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Ctrl+Z / Ctrl+Shift+Z undo and redo, and the Reopen button (run-curation Task 6) | date: 2026-10-05]
def _status_of(panel, track_id):
    hit = panel._rows[panel._rows.cluster_id == track_id]
    return hit.iloc[0].status


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the keys come from a settings file under tmp_path; tag wraps the whole test (day-table-keys Task 4) | date: 2026-10-14]
def test_ctrl_z_undoes_and_ctrl_shift_z_redoes(qt, tracked_run, clean_curation, theme_env):
    import csv
    from PySide6.QtTest import QTest
    from multiphy.clusters import cluster_id
    window = open_run(tracked_run)
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window, 2000)
    panel = window.tracked
    track_id = cluster_id(TRACK)
    _select(panel, track_id)
    panel.accept()
    assert _status_of(panel, track_id) == 'accepted'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    QTest.keyClick(window, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier)
    # [AGENT EDIT END]
    assert _status_of(panel, track_id) == 'proposed'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    QTest.keyClick(window, QtCore.Qt.Key_Z, QtCore.Qt.ControlModifier | QtCore.Qt.ShiftModifier)
    # [AGENT EDIT END]
    assert _status_of(panel, track_id) == 'accepted'
    history = clean_curation / 'runs' / tracked_run.name / 'history.csv'
    with open(history, newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    window.close()
# [AGENT EDIT END]


def test_reopen_button_enabled_by_status(qt, tracked_run, clean_curation):
    from multiphy.clusters import cluster_id
    window = open_run(tracked_run)
    panel = window.tracked
    track_id = cluster_id(TRACK)
    _select(panel, track_id)
    reopen = panel.buttons['Reopen']
    assert reopen.toolTip() == 'back to undecided'
    assert reopen.isEnabled() is False
    panel.accept()
    _select(panel, track_id)
    assert reopen.isEnabled() is True
    reopen.click()
    assert _status_of(panel, track_id) == 'proposed'
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: read-only second viewer, Take over, lock release on close and a failed run switch (run-curation Task 7) | date: 2026-10-05]
def test_second_window_is_read_only(qt, tracked_run, clean_curation):
    from multiphy.clusters import cluster_id
    first = open_run(tracked_run)
    second = open_run(tracked_run)
    panel = second.tracked
    _select(panel, cluster_id(TRACK))
    assert second.curation.read_only is True
    assert not panel.lock_tag.isHidden()
    assert panel.lock_tag.text() == 'read-only · locked'
    lock = clean_curation / 'runs' / tracked_run.name / 'history.lock'
    assert panel.lock_tag.toolTip() == f'{socket.gethostname()} pid {os.getpid()} · {lock}'
    assert panel.buttons['Accept'].isEnabled() is False
    assert panel.take_over_btn.isHidden()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: read-only disables the undo and redo shortcuts (run-curation Task 7 fix round 1) | date: 2026-10-05]
    assert [s.isEnabled() for s in second.undo_shortcuts] == [False, False]
    assert [s.isEnabled() for s in first.undo_shortcuts] == [True, True]
    # [AGENT EDIT END]
    second.close()
    first.close()


def test_take_over_remote_lock(qt, tracked_run, clean_curation):
    from multiphy.clusters import cluster_id
    folder = clean_curation / 'runs' / tracked_run.name
    folder.mkdir(parents=True)
    lock = folder / 'history.lock'
    lock.write_text(json.dumps({'pid': 4242, 'host': 'elsewhere', 'token': 'f00d', 'at': '2026-10-05T00:00:00'}))
    window = open_run(tracked_run)
    panel = window.tracked
    _select(panel, cluster_id(TRACK))
    assert panel.buttons['Accept'].isEnabled() is False
    assert not panel.take_over_btn.isHidden()
    panel.take_over_btn.click()
    assert json.loads(lock.read_text())['host'] == 'elsewhere'
    assert panel.take_over_btn.isHidden()
    assert not panel.confirm_take_over_btn.isHidden()
    assert panel.confirm_take_over_btn.toolTip() == 'the other viewer stops saving'
    assert not panel.cancel_take_over_btn.isHidden()
    panel.cancel_take_over_btn.click()
    assert not panel.take_over_btn.isHidden()
    assert panel.confirm_take_over_btn.isHidden()
    panel.take_over_btn.click()
    panel.confirm_take_over_btn.click()
    _select(panel, cluster_id(TRACK))
    assert panel.buttons['Accept'].isEnabled() is True
    assert json.loads(lock.read_text())['host'] == socket.gethostname()
    assert panel.lock_tag.isHidden()
    assert panel.confirm_take_over_btn.isHidden() and panel.cancel_take_over_btn.isHidden()
    window.close()


def test_read_only_disables_reopen(qt, tracked_run, clean_curation):
    from multiphy.clusters import cluster_id
    track_id = cluster_id(TRACK)
    first = open_run(tracked_run)
    _select(first.tracked, track_id)
    first.tracked.accept()
    second = open_run(tracked_run)
    _select(second.tracked, track_id)
    assert _status_of(second.tracked, track_id) == 'accepted'
    assert second.tracked.buttons['Reopen'].isEnabled() is False
    second.close()
    first.close()


def test_close_releases_lock(qt, tracked_run, clean_curation):
    window = open_run(tracked_run)
    lock = clean_curation / 'runs' / tracked_run.name / 'history.lock'
    assert lock.exists()
    window.close()
    assert not lock.exists()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed layout save still releases the lock (run-curation Task 7 fix round 1) | date: 2026-10-05]
    window = open_run(tracked_run)
    assert lock.exists()

    def refuse_layout():
        raise OSError('disk full')
    window._save_layout = refuse_layout
    with pytest.raises(OSError):
        window.closeEvent(QtGui.QCloseEvent())
    assert window.curation is None
    assert not lock.exists()
    # [AGENT EDIT END]


def test_failed_run_switch_keeps_old_run(qt, tracked_run, clean_curation, monkeypatch):
    from multiphy import curation
    window = open_run(tracked_run)
    controller = window.curation
    lock = clean_curation / 'runs' / tracked_run.name / 'history.lock'

    def refuse(*args, **kwargs):
        raise OSError('cannot open the history')
    monkeypatch.setattr(curation.Curation, 'open', refuse)
    with pytest.raises(OSError):
        window.import_match_result(tracked_run / 'result.json')
    assert window.curation is controller
    assert lock.exists()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed controller open shows in the status bar (run-curation Task 7 fix round 2, D37) | date: 2026-10-05]
    assert 'cannot open the history' in window.statusBar().currentMessage()
    # [AGENT EDIT END]
    monkeypatch.undo()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed close of the old controller closes the new one and keeps the old (run-curation Task 7 fix round 1) | date: 2026-10-05]
    opened = []
    real_open = curation.Curation.open

    def record(curation_dir, result):
        opened.append(real_open(clean_curation / 'runs' / 'next_run', result))
        return opened[-1]

    def refuse_close():
        raise OSError('cannot remove the lock')
    monkeypatch.setattr(curation.Curation, 'open', record)
    monkeypatch.setattr(controller, 'close', refuse_close)
    with pytest.raises(OSError):
        window.import_match_result(tracked_run / 'result.json')
    assert window.curation is controller
    assert len(opened) == 1
    assert not (clean_curation / 'runs' / 'next_run' / 'history.lock').exists()
    assert lock.exists()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed old-controller close shows in the status bar (run-curation Task 7 fix round 2, D37) | date: 2026-10-05]
    assert 'cannot remove the lock' in window.statusBar().currentMessage()
    # [AGENT EDIT END]
    monkeypatch.undo()
    # [AGENT EDIT END]
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-day merge is allowed and flagged; Export writes the accepted tracks (run-curation Task 8) | date: 2026-10-05]
def _select_rows(panel, track_ids):
    model = panel.table.selectionModel()
    model.clearSelection()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    for i in range(panel.table.rowCount()):
        if panel.table.item(i, 0).data(QtCore.Qt.UserRole) in track_ids:
            model.select(panel.table.model().index(i, 0),
                         QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    # [AGENT EDIT END]


def test_same_day_merge_flagged(qt, tracked_run, clean_curation):
    window = open_run(tracked_run)
    panel = window.tracked
    tracks = window.track_state().tracks
    holding = [tid for tid, t in tracks.items() if {'D1:0', 'D1:1'} & set(t['members'])]
    assert len(holding) == 2
    _select_rows(panel, holding)
    assert sorted(panel.selected_ids()) == sorted(holding)
    panel.buttons['Merge'].click()
    joined = [tid for tid, t in window.track_state().tracks.items() if {'D1:0', 'D1:1'} <= set(t['members'])]
    assert len(joined) == 1
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt constants from viewer.qt (viewer-split Task 8) | date: 2026-10-07]
    row = next(i for i in range(panel.table.rowCount())
               if panel.table.item(i, 0).data(QtCore.Qt.UserRole) == joined[0])
    # [AGENT EDIT END]
    assert '2 units on one day' in panel.table.item(row, 2).text()
    window.close()


def test_export_button_writes_file(qt, tracked_run, clean_curation):
    import csv
    from multiphy.clusters import cluster_id
    window = open_run(tracked_run)
    panel = window.tracked
    track_id = cluster_id(TRACK)
    _select(panel, track_id)
    panel.accept()
    assert window.export_btn.isEnabled() is True
    assert window.export_btn.toolTip() == 'accepted tracks → csv'
    window.export_btn.click()
    exports = clean_curation / 'runs' / tracked_run.name / 'exports'
    files = list(exports.iterdir())
    assert len(files) == 1
    assert window.statusBar().currentMessage() == str(files[0])
    with open(files[0], newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(r['cellid'] for r in rows) == sorted(TRACK)
    assert all(r['track_id'] == track_id and r['n_units'] == str(len(TRACK)) for r in rows)
    window.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a read-only viewer's Export replays the history on disk (run-curation Task 8) | date: 2026-10-05]
def test_read_only_export_replays_history(qt, tracked_run, clean_curation):
    import csv
    from multiphy.clusters import cluster_id
    track_id = cluster_id(TRACK)
    first = open_run(tracked_run)
    second = open_run(tracked_run)
    assert second.curation.read_only is True
    _select(first.tracked, track_id)
    first.tracked.accept()
    assert second.export_btn.isEnabled() is True
    second.export_btn.click()
    exports = clean_curation / 'runs' / tracked_run.name / 'exports'
    files = list(exports.iterdir())
    assert len(files) == 1
    assert second.statusBar().currentMessage() == str(files[0])
    with open(files[0], newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert sorted(r['cellid'] for r in rows) == sorted(TRACK)
    assert all(r['track_id'] == track_id and r['n_units'] == str(len(TRACK)) for r in rows)
    second.close()
    first.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pins tmp_settings: with no fixture requested, the settings file lies in a fresh settings folder under the pytest base temp, never ~/.config/multiphy or an after folder (day-table-keys Task 6) | date: 2026-10-14]
def test_settings_live_in_a_fresh_tmp_folder():
    p = Path(theme.settings().fileName())
    assert p.name == 'multiphy.ini'
    assert p.parent.name == 'multiphy'
    assert re.fullmatch(r'settings\d+', p.parent.parent.name) is not None
# [AGENT EDIT END]
