"""The ``App`` window over ``DictData`` runs and the fake mode; every file it writes lives under ``tmp_path``."""
import json
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from PySide6 import QtWidgets

from multiphy.viewer import display
from multiphy.viewer.app import App

from .viewer_fakes import FAKE_MODE, DictData
from multiphy import clusters as cl
from multiphy.viewer.qt import QtCore

from .viewer_fakes import day_table_data
from .viewer_fakes import many_days_data
from PySide6.QtTest import QTest

from multiphy.viewer import keys

from .theme_fixtures import theme_env  # noqa: F401
from multiphy.viewer import theme

import concurrent.futures
import copy
import dataclasses
import shutil
import typing

from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection
from multiphy.viewer import app as viewer_app

import importlib.util

from multiphy.project.folder import Project

from .synthetic_day import extracted_project

UNITS = ['a-01', 'b-01', 'c-01', 'd-01']


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _data():
    pairs = pd.DataFrame({'cellid_a': UNITS[:3], 'cellid_b': UNITS[1:], 'p_global': [.99, .98, .97]})
    return DictData(UNITS, {}, {}, {u: u.split('-')[0] for u in UNITS}, pairs)


def _dock_layout(window):
    """Each dock's area, floating and hidden flags, and the docks tabbed with it."""
    return {name: (window.dockWidgetArea(d), d.isFloating(), d.isHidden(),
                   sorted(t.objectName() for t in window.tabifiedDockWidgets(d)))
            for name, d in window._docks.items()}


def test_failed_prep_keeps_old_run(qt, tmp_path, monkeypatch):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    shown = _data()
    window.set_data(shown, curation_dir=tmp_path / 'a')
    controller = window.curation
    assert controller is not None and not controller.read_only
    other = tmp_path / 'other'

    def broken(result):
        raise RuntimeError('no summary')
    monkeypatch.setattr(display, 'result_summary', broken)
    with pytest.raises(RuntimeError):
        window.set_data(_data(), curation_dir=other)
    assert window.state.data is shown
    assert window.curation is controller
    assert not controller.closed
    assert json.loads((tmp_path / 'a' / 'history.lock').read_text())['token'] == controller.token
    assert not (other / 'history.lock').exists()
    monkeypatch.undo()
    window.close()


def test_inspect_only_without_curation_dir(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(_data())
    assert window.curation is None
    assert not window.tracked.buttons['Accept'].isEnabled()
    window.close()


def test_layout_file_written_on_close(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.close()
    assert 'state' in json.loads((tmp_path / 'l.json').read_text())
    assert not (tmp_path / 'l.json.tmp').exists()


def test_wrong_shape_layout_opens_default(qt, tmp_path):
    reference = App(FAKE_MODE, layout_file=tmp_path / 'missing.json')
    (tmp_path / 'l.json').write_text('[]')
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    assert _dock_layout(window) == _dock_layout(reference)
    assert bytes(window.saveState()) == bytes(reference.saveState())
    for w in (window, reference):
        w._save_layout = lambda: None
        w.close()


def test_drift_dock_beside_probe(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    dock = window._docks['drift']
    assert window.tabifiedDockWidgets(dock) == []
    # every plot dock shares the probe's area: the first split joins the plot grid to the browser's
    assert window.dockWidgetArea(dock) == window.dockWidgetArea(window._docks['probe'])
    assert not dock.isHidden()
    window._save_layout = lambda: None
    window.close()


def test_layout_without_drift_opens_default(qt, tmp_path):
    reference = App(FAKE_MODE, layout_file=tmp_path / 'missing.json')
    first = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    first._docks['drift'].hide()
    first._save_layout()
    saved = json.loads((tmp_path / 'l.json').read_text())
    saved['docks'].remove('drift')
    assert saved['state'] != bytes(reference.saveState().toBase64()).decode()
    (tmp_path / 'l.json').write_text(json.dumps(saved))
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    assert _dock_layout(window) == _dock_layout(reference)
    assert not window._docks['drift'].isHidden()
    for w in (first, window, reference):
        w._save_layout = lambda: None
        w.close()


def test_layout_with_spiking_dock_opens_default(qt, tmp_path):
    first = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    first.show()
    first._docks['drift'].hide()
    first._docks['clusters'].raise_()
    qt.processEvents()
    first._save_layout()
    first._save_layout = lambda: None
    first.close()
    saved = json.loads((tmp_path / 'l.json').read_text())
    saved['docks'] = sorted([d for d in saved['docks'] if d not in ('isi', 'acg')] + ['spiking'])
    (tmp_path / 'l.json').write_text(json.dumps(saved))
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.show()
    qt.processEvents()
    assert window._docks['drift'].isHidden() is False
    assert _front_tab(window) == 'Browse'
    assert sorted(t.objectName() for t in window.tabifiedDockWidgets(window._docks['isi'])) == ['um_acg_dock']
    window._save_layout = lambda: None
    window.close()

def test_failed_render_closes_new_controller(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(_data(), curation_dir=tmp_path / 'a')
    failing = _data()

    def broken_panel():
        if window.state.data is failing:
            raise RuntimeError('panel failed')
    window.state.changed.connect(broken_panel)
    with pytest.raises(RuntimeError, match='panel failed'):
        window.set_data(failing, curation_dir=tmp_path / 'b')
    assert window.curation is None
    assert not (tmp_path / 'b' / 'history.lock').exists()
    assert not window.tracked.buttons['Merge'].isEnabled()
    window.state.changed.disconnect(broken_panel)
    window._save_layout = lambda: None
    window.close()


def test_failed_render_with_failed_close_detaches_controller(qt, tmp_path, monkeypatch):
    from multiphy import curation
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(_data(), curation_dir=tmp_path / 'a')
    failing = _data()
    real_close = curation.Curation.close

    def close(self):
        if (tmp_path / 'b') in Path(self._lock_path).parents:
            raise OSError('lock release failed')
        real_close(self)
    monkeypatch.setattr(curation.Curation, 'close', close)

    def broken_panel():
        if window.state.data is failing:
            raise RuntimeError('panel failed')
    window.state.changed.connect(broken_panel)
    with pytest.raises(RuntimeError, match='panel failed') as info:
        window.set_data(failing, curation_dir=tmp_path / 'b')
    assert isinstance(info.value.__cause__, OSError)
    assert window.curation is None
    assert not window.tracked.buttons['Merge'].isEnabled()
    assert window.tracked.inspect_tag.isVisibleTo(window)
    window.state.changed.disconnect(broken_panel)
    window._save_layout = lambda: None
    window.close()


@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    """A D1+D2+D3 run, a second D1+D2+D3 run and a D1+D2 run of one project."""
    path = extracted_project(tmp_path_factory.mktemp('snap'), days=('D1', 'D2', 'D3'))
    out = {'project': Project.open(path.parent, path.name)}
    out['across_days'] = run_selection(Selection(out['project'], ('D1', 'D2', 'D3')), 'functional_likelihood')
    out['second'] = run_selection(Selection(out['project'], ('D1', 'D2', 'D3')), 'functional_likelihood')
    out['two_days'] = run_selection(Selection(out['project'], ('D1', 'D2')), 'functional_likelihood')
    return out


@pytest.fixture
def clean_curation(runs):
    folder = runs['project'].path / 'curation'
    shutil.rmtree(folder, ignore_errors=True)
    yield folder
    shutil.rmtree(folder, ignore_errors=True)


def _close(window):
    window.close()
    QtWidgets.QApplication.processEvents()


def test_open_run_refuses_incomplete_run(qt, runs):
    run = runs['second'].parent / (runs['second'].name + '_no_record')
    shutil.copytree(runs['second'], run)
    run.chmod(0o755)
    (run / 'run_manifest.json').unlink()
    with pytest.raises(ValueError, match='is not a completed run'):
        viewer_app.open_run(run)


def test_failed_result_load_keeps_controller(qt, runs, clean_curation):
    window = viewer_app.open_run(runs['across_days'])
    shown, controller = window.state.data, window.curation
    lock = clean_curation / 'runs' / runs['across_days'].name / 'history.lock'
    assert controller is not None and not controller.read_only
    bad = copy.deepcopy(shown.result)
    bad.provenance = bad.provenance.merged(input_manifest_sha256='0' * 64)
    with pytest.raises(ValueError, match='another selection'):
        window.open_result(bad, runs['second'])
    assert window.curation is controller and not controller.closed
    assert window.state.data is shown
    assert json.loads(lock.read_text())['token'] == controller.token
    assert not (clean_curation / 'runs' / runs['second'].name).exists()
    _close(window)


def test_import_refuses_other_selection(qt, runs, clean_curation):
    window = viewer_app.open_run(runs['across_days'])
    shown = window.state.data
    with pytest.raises(ValueError, match='another selection'):
        window.import_match_result(runs['two_days'] / 'result.json')
    assert window.state.data is shown
    _close(window)


def test_failed_run_list_keeps_shown_run(qt, runs, clean_curation, monkeypatch):
    window = viewer_app.open_run(runs['across_days'])
    shown, controller = window.state.data, window.curation
    lock = clean_curation / 'runs' / runs['across_days'].name / 'history.lock'

    def boom(selection):
        raise ValueError('boom')
    monkeypatch.setattr(viewer_app, 'selection_runs', boom)
    with pytest.raises(ValueError, match='boom'):
        window.import_match_result(runs['second'] / 'result.json')
    assert window.state.data is shown
    assert window.curation is controller and not controller.closed
    assert json.loads(lock.read_text())['token'] == controller.token
    assert not (clean_curation / 'runs' / runs['second'].name / 'history.lock').exists()
    monkeypatch.undo()
    _close(window)


def test_failed_first_open_releases_lock(qt, runs, clean_curation, monkeypatch):
    def boom(selection):
        raise ValueError('boom')
    monkeypatch.setattr(viewer_app, 'selection_runs', boom)
    with pytest.raises(ValueError, match='boom'):
        viewer_app.open_run(runs['across_days'])
    QtWidgets.QApplication.processEvents()
    assert not (clean_curation / 'runs' / runs['across_days'].name / 'history.lock').exists()


def test_prepare_run_builds_no_window(qt, runs, clean_curation, monkeypatch):
    def no_app(*args, **kwargs):
        raise AssertionError('App built')
    monkeypatch.setattr(viewer_app, 'App', no_app)
    messages = []

    def record(mode, context, message):
        messages.append(message)
    before = list(QtWidgets.QApplication.topLevelWidgets())
    previous = QtCore.qInstallMessageHandler(record)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            prepared = pool.submit(viewer_app.prepare_run, runs['across_days']).result(timeout=120)
    finally:
        QtCore.qInstallMessageHandler(previous)
    assert prepared.mode.id == 'across_days'
    assert prepared.run_dir == runs['across_days']
    assert prepared.curation_dir == clean_curation / 'runs' / runs['across_days'].name
    assert (clean_curation / 'runs' / runs['across_days'].name / 'history.lock').exists() is False
    assert messages == []
    assert [w for w in QtWidgets.QApplication.topLevelWidgets() if w not in before] == []


def test_open_prepared_reuses_the_read(qt, runs, clean_curation, monkeypatch):
    prepared = viewer_app.prepare_run(runs['across_days'])
    calls = []
    original = Selection.dataset

    def counted(self, *args, **kwargs):
        calls.append(self)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(Selection, 'dataset', counted)
    window = viewer_app.open_prepared(prepared)
    try:
        assert window.state.data is prepared.data
        assert (clean_curation / 'runs' / runs['across_days'].name / 'history.lock').exists() is True
        window.import_match_result(runs['second'] / 'result.json')
        assert len(calls) == 0
    finally:
        _close(window)


def test_prepared_run_signatures():
    assert typing.get_type_hints(viewer_app.prepare_run) == {'run_dir': Path,
                                                             'return': viewer_app.PreparedRun}
    assert typing.get_type_hints(viewer_app.open_prepared) == {'prepared': viewer_app.PreparedRun,
                                                               'return': viewer_app.App}
    assert [f.name for f in dataclasses.fields(viewer_app.PreparedRun)] == [
        'selection', 'mode', 'run_dir', 'layout_file', 'inputs', 'data', 'runs', 'curation_dir']


def test_main_requires_run(qt, capsys):
    with pytest.raises(SystemExit) as caught:
        viewer_app.main([])
    assert caught.value.code == 2
    with pytest.raises(SystemExit) as caught:
        viewer_app.main(['--run', 'x', '--backend', 'x'])
    assert caught.value.code == 2
    assert 'unrecognized arguments: --backend x' in capsys.readouterr().err


def test_run_box_reads_no_paths_after_switch(qt, runs, clean_curation, monkeypatch):
    window = viewer_app.open_run(runs['across_days'])
    original = window.set_data

    def switch_then_forbid(data, **kw):
        original(data, **kw)

        def forbidden(self, *a, **k):
            raise AssertionError(f'path read after set_data: {self}')
        monkeypatch.setattr(Path, 'resolve', forbidden)
    monkeypatch.setattr(window, 'set_data', switch_then_forbid)
    imported = window.import_match_result(runs['second'] / 'result.json')
    monkeypatch.undo()
    listed = [e.run_id for e in window._run_items if e is not None]
    assert imported.run_id in listed
    _close(window)


@pytest.mark.parametrize('flag', ['--r', '--ru'])
def test_main_rejects_abbreviated_run(qt, flag):
    with pytest.raises(SystemExit) as caught:
        viewer_app.main([flag, 'x'])
    assert caught.value.code == 2


def test_import_match_result_declares_match_result():
    import typing
    from multiphy.contract.results import MatchResult
    assert typing.get_type_hints(App.import_match_result)['return'] is MatchResult


def test_refresh_runs_reads_nothing(qt, runs, clean_curation, monkeypatch):
    window = viewer_app.open_run(runs['across_days'])
    listed = list(window._run_entries)
    items = list(window._run_items)

    def boom(selection):
        raise ValueError('boom')
    monkeypatch.setattr(viewer_app, 'selection_runs', boom)
    window.refresh_runs(listed)
    assert window._run_items == items
    window.refresh_btn.click()
    assert 'run listing failed · boom' in window.statusBar().currentMessage()
    assert window._run_entries == listed and window._run_items == items
    monkeypatch.undo()
    with pytest.raises(TypeError):
        window.refresh_runs()
    _close(window)


def test_old_viewer_modules_gone():
    for name in ('gui', 'multiday_viewer', 'benchmark_viewer'):
        assert importlib.util.find_spec('multiphy.' + name) is None, name


def _day_window(tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(day_table_data())
    return window


def test_open_run_leaves_day_table_unpicked(qt, tmp_path):
    window = _day_window(tmp_path)
    assert window.days.picked_unit is None
    assert window.days.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    _close(window)


def test_day_pick_selects_its_track_in_tracked_units(qt, tmp_path):
    window = _day_window(tmp_path)
    window.days.pick('B-1')
    t = window.tracked.table
    assert t.item(t.currentRow(), 0).data(QtCore.Qt.UserRole) == cl.cluster_id(['A-1', 'B-1'])
    assert window.state.selection == ['B-1']
    _close(window)


def test_tracked_click_picks_earliest_day_member(qt, tmp_path):
    window = _day_window(tmp_path)
    window.days.pick('A-3')
    t = window.tracked.table
    i = next(i for i in range(t.rowCount()) if t.item(i, 0).data(QtCore.Qt.UserRole) == cl.cluster_id(['A-1', 'B-1']))
    t.setCurrentCell(i, 0)
    assert window.days.picked_unit == 'A-1'
    _close(window)


def test_select_pair_finds_row_either_order(qt, tmp_path):
    window = _day_window(tmp_path)
    state = window.state
    window.select_pair('B-1', 'A-1')
    assert (state.row.cellid_a, state.row.cellid_b) == ('A-1', 'B-1')
    assert {state.day_pair['siteid_a'], state.day_pair['siteid_b']} == {'A', 'B'}
    window.select_pair('A-2', 'A-1')
    assert (state.row.cellid_a, state.row.cellid_b) == ('A-1', 'A-2')
    assert state.day_pair is None
    window.select_pair('A-4', 'B-2')
    assert state.row is None
    assert state.selection == ['A-4', 'B-2']
    assert window.evidence.status.text() == 'no pair evidence'
    assert window.header.text() == 'Selected units: A-4, B-2'
    _close(window)


def test_select_pair_keeps_selection(qt, theme_env):
    tmp_path = theme_env
    window = _day_window(tmp_path)
    state = window.state
    window.select_pair('B-1', 'A-1', selection=['A-1', 'B-1', 'A-3'])
    assert (state.row.cellid_a, state.row.cellid_b) == ('A-1', 'B-1')
    assert state.selection == ['A-1', 'B-1', 'A-3']
    window.select_pair('A-4', 'B-2', selection=['A-1', 'A-4', 'B-2'])
    assert state.row is None
    assert state.selection == ['A-1', 'A-4', 'B-2']
    assert window.evidence.status.text() == 'no pair evidence'
    _close(window)


def test_curation_keeps_string_selection(qt, theme_env):
    tmp_path = theme_env
    window = _day_window(tmp_path)
    window.set_data(day_table_data(), curation_dir=tmp_path / 'cur')
    days, state = window.days, window.state
    days.pick('A-1')
    days.move_day(1)
    days.step(1)
    assert days.strung == {'B': 'B-2'}
    days.accept()
    assert window.track_state().tracks[cl.cluster_id(['A-1', 'B-1'])]['status'] == 'accepted'
    assert state.selection == ['A-1', 'B-2']
    assert (state.row.cellid_a, state.row.cellid_b) == ('A-1', 'B-2')
    window._undo_redo('undo')
    assert window.track_state().tracks[cl.cluster_id(['A-1', 'B-1'])]['status'] == 'proposed'
    assert state.selection == ['A-1', 'B-2']
    days.clear_string()
    days.accept()
    assert state.selection == ['A-1', 'B-1']
    _close(window)


def test_undo_of_merge_keeps_string(qt, theme_env):
    tmp_path = theme_env
    window = _day_window(tmp_path)
    window.set_data(day_table_data(), curation_dir=tmp_path / 'cur')
    days, state = window.days, window.state
    days.pick('A-3')
    days.toggle('B-2')
    days.merge()
    assert cl.cluster_id(['A-3', 'B-2']) in window.track_state().tracks
    t = window.tracked.table
    assert t.item(t.currentRow(), 0).data(QtCore.Qt.UserRole) == cl.cluster_id(['A-3', 'B-2'])
    window._undo_redo('undo')
    assert cl.cluster_id(['A-3', 'B-2']) not in window.track_state().tracks
    assert days.picked_unit == 'A-3'
    assert days.strung == {'B': 'B-2'}
    assert state.selection == ['A-3', 'B-2']
    assert (state.row.cellid_a, state.row.cellid_b) == ('A-3', 'B-2')
    assert t.item(t.currentRow(), 0).data(QtCore.Qt.UserRole) == cl.cluster_id(['A-3'])
    _close(window)


def test_hidden_track_action_keeps_pick(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(day_table_data(), curation_dir=tmp_path / 'cur')
    window.tracked.hide_done.setChecked(True)
    window.days.pick('A-2')
    window.days.buttons['Accept'].click()
    assert window.days.picked_unit == 'A-2'
    assert window.tracked.table.currentRow() == -1
    assert window.tracked.members.rowCount() == 0
    assert window.tracked.hide_done.isChecked() is True
    assert window.state.selection == ['A-2']
    _close(window)

    window = App(FAKE_MODE, layout_file=tmp_path / 'l2.json')
    window.set_data(day_table_data(), curation_dir=tmp_path / 'cur2')
    window.tracked.only_review.setChecked(True)
    window.days.pick('B-1')
    window.days.buttons['Reject'].click()
    assert window.days.picked_unit == 'B-1'
    assert window.tracked.table.currentRow() == -1
    assert window.state.selection == ['B-1']
    _close(window)


def test_shown_track_action_follows_it(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.set_data(day_table_data(), curation_dir=tmp_path / 'cur')
    window.days.pick('A-2')
    window.days.buttons['Accept'].click()
    assert window.days.picked_unit == 'A-2'
    t = window.tracked.table
    assert t.item(t.currentRow(), 0).data(QtCore.Qt.UserRole) == cl.cluster_id(['A-2'])
    _close(window)


def test_drift_click_toggles_day_table(qt, theme_env):
    tmp_path = theme_env
    window = _day_window(tmp_path)
    drift = next(x for x in window.plots if x.panel_id == 'drift')
    drift.unit_clicked.emit('B-1')
    assert window.days.picked_unit == 'B-1'
    scrolled = []
    window.days.tables['A'].scrollToItem = lambda item, *args: scrolled.append(item.data(QtCore.Qt.UserRole))
    drift.unit_clicked.emit('A-2')
    assert window.days.selected == ['B-1', 'A-2']
    assert scrolled == ['A-2']
    assert sorted(window.state.selection) == ['A-2', 'B-1']
    assert window.days.picked_unit == 'B-1'
    drift.unit_clicked.emit('A-2')
    assert window.days.strung == {}
    assert window.state.selection == ['B-1']
    _close(window)


def _front_tab(window):
    # Some Qt builds retain obsolete tab bars after relayout; inspect the visible dock instead.
    (dock,) = [window._docks[name] for name in ('browse', 'clusters')
               if not window._docks[name].visibleRegion().isEmpty()]
    return dock.windowTitle()


def test_browse_in_front_by_default(qt, tmp_path):
    first = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    first.show()
    qt.processEvents()
    assert _front_tab(first) == 'Browse'
    first._docks['clusters'].raise_()
    qt.processEvents()
    assert _front_tab(first) == 'Tracked units'
    first._save_layout()
    first._save_layout = lambda: None
    first.close()
    second = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    second.show()
    qt.processEvents()
    assert _front_tab(second) == 'Tracked units'
    second._reset_layout()
    qt.processEvents()
    assert _front_tab(second) == 'Browse'
    second._save_layout = lambda: None
    second.close()


def test_isi_and_acg_docks_tabbed(qt, tmp_path):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.resize(1600, 1000)
    window.show()
    qt.processEvents()
    assert 'spiking' not in window._docks
    assert sorted(t.objectName() for t in window.tabifiedDockWidgets(window._docks['isi'])) == ['um_acg_dock']
    assert not window._docks['isi'].visibleRegion().isEmpty()
    assert window._docks['acg'].visibleRegion().isEmpty()
    assert window._docks['isi'].geometry().x() < window._docks['fingerprint'].geometry().x()
    assert window._docks['isi'].geometry().y() > window._docks['waveform'].geometry().y()
    window._save_layout = lambda: None
    window.close()


def _settle(qt):
    for _ in range(3):
        qt.processEvents()
        QtCore.QThread.msleep(20)


def _wholly_shown(days):
    """The sites whose day column lies wholly inside the day scroll's viewport, left to right."""
    viewport = days.scroll.viewport()
    shown = []
    for site, table in days.tables.items():
        column = table.parentWidget()
        left = column.mapTo(viewport, QtCore.QPoint(0, 0)).x()
        # Dock splitter sizing can round by one pixel on Windows.
        if left >= 0 and left + column.width() <= viewport.width() + 1:
            shown.append(site)
    return shown


def _drag_browse(window, width):
    """Drag the Browse dock to ``width``, as a user drags its edge."""
    browse = window._docks['browse']
    for _ in range(40):
        before = browse.width()
        window.resizeDocks([browse], [width], QtCore.Qt.Horizontal)
        window.layout().activate()
        if browse.width() in (width, before):
            return


def _fitted_window(qt, tmp_path, n_days):
    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    window.resize(2400, 1000)
    window.show()
    _settle(qt)
    window.set_data(many_days_data(n_days))
    _settle(qt)
    window._save_layout = lambda: None
    return window


def test_browse_fits_up_to_three_days_then_scrolls(qt, tmp_path):
    window = _fitted_window(qt, tmp_path, 3)
    days = window.days
    narrowest = days.same_table.width()
    assert _wholly_shown(days) == ['D00a', 'D01a', 'D02a']
    assert not days.scroll.horizontalScrollBar().isVisible()
    assert [t.width() for t in days.tables.values()] == [narrowest] * 3
    window.set_data(many_days_data(7))
    _settle(qt)
    assert _wholly_shown(days) == ['D00a', 'D01a', 'D02a']
    assert days.scroll.horizontalScrollBar().isVisible()
    assert [t.width() for t in days.tables.values()] == [narrowest] * 7
    for d in window._docks.values():
        if d.isVisible():
            assert d.width() >= d.minimumSizeHint().width(), d.objectName()
    window.close()


def test_narrower_browse_scrolls_and_reset_fits_again(qt, tmp_path):
    window = _fitted_window(qt, tmp_path, 3)
    days = window.days
    narrowest = days.same_table.width()
    _drag_browse(window, 560)
    _settle(qt)
    assert window._docks['browse'].width() == 560
    assert days.scroll.horizontalScrollBar().isVisible()
    assert len(_wholly_shown(days)) < 3
    assert [t.width() for t in days.tables.values()] == [narrowest] * 3
    window._reset_layout()
    # the offscreen screen is 800 x 800 and restoreGeometry clamps the window to it, so the window is widened
    # again before the deferred fit runs
    window.resize(2400, 1000)
    window.layout().activate()
    # the default layout alone is not fitted to the three days
    assert days.scroll.viewport().width() != days.fit_width()
    _settle(qt)
    assert abs(days.scroll.viewport().width() - days.fit_width()) <= 1
    assert _wholly_shown(days) == ['D00a', 'D01a', 'D02a']
    assert not days.scroll.horizontalScrollBar().isVisible()
    assert [t.width() for t in days.tables.values()] == [narrowest] * 3
    window.close()


def _activate(window):
    """Show ``window`` and wait until it is the active window, so its shortcuts fire."""
    window.show()
    window.activateWindow()
    QTest.qWaitForWindowActive(window, 2000)


def _press(window, key, modifier=QtCore.Qt.NoModifier):
    QTest.keyClick(window, key, modifier)


def test_wasd_keys_drive_day_table(qt, theme_env):
    window = _day_window(theme_env)
    days = window.days
    _activate(window)
    days.pick('A-1')
    _press(window, QtCore.Qt.Key_D)
    assert days.strung == {'B': 'B-1'}
    _press(window, QtCore.Qt.Key_S)
    assert days.strung == {'B': 'B-2'}
    _press(window, QtCore.Qt.Key_W)
    assert days.strung == {'B': 'B-1'}
    _press(window, QtCore.Qt.Key_X, QtCore.Qt.ShiftModifier)
    assert days.strung == {}
    _press(window, QtCore.Qt.Key_A)
    assert days.active_site == 'A'
    _press(window, QtCore.Qt.Key_D)
    assert days.strung == {'B': 'B-1'}
    _press(window, QtCore.Qt.Key_Escape)
    assert days.strung == {}
    _press(window, QtCore.Qt.Key_F)
    assert days.strung == {'B': 'B-1'}
    _press(window, QtCore.Qt.Key_Down)
    assert days.picked_unit == 'A-1'
    assert days.strung == {'B': 'B-1'}
    _close(window)


def test_verdict_key_saves_then_steps(qt, theme_env):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    days, state = window.days, window.state
    window.pair_review_action.setChecked(True)
    _activate(window)
    days.pick('A-1')
    _press(window, QtCore.Qt.Key_D)
    _press(window, QtCore.Qt.Key_E, QtCore.Qt.AltModifier)
    assert {k: v['verdict'] for k, v in window.track_state().verdicts.items()} == {('A-1', 'B-1'): 'accept'}
    assert days.strung == {'B': 'B-2'}
    assert (state.row.cellid_a, state.row.cellid_b) == ('A-1', 'B-2')
    _close(window)


def test_track_keys_accept_and_merge(qt, theme_env):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    days = window.days
    _activate(window)
    days.pick('A-1')
    _press(window, QtCore.Qt.Key_E)
    assert window.track_state().tracks[cl.cluster_id(['A-1', 'B-1'])]['status'] == 'accepted'
    days.pick('A-3')
    _press(window, QtCore.Qt.Key_D)
    assert days.strung == {'B': 'B-2'}
    _press(window, QtCore.Qt.Key_Q)
    track_id = cl.cluster_id(['A-3', 'B-2'])
    assert window.track_state().tracks[track_id]['status'] == 'proposed'
    _press(window, QtCore.Qt.Key_E)
    assert window.track_state().tracks[track_id]['status'] == 'accepted'
    assert window.track_state().verdicts == {}
    assert days.picked_unit == 'A-3'
    assert days.strung == {'B': 'B-2'}
    _press(window, QtCore.Qt.Key_X)
    assert window.track_state().tracks[cl.cluster_id(['B-2'])]['status'] == 'unmatched'
    assert window.track_state().tracks[cl.cluster_id(['A-3'])]['status'] == 'unmatched'
    days.pick('A-1')
    _press(window, QtCore.Qt.Key_R)
    assert window.track_state().tracks[cl.cluster_id(['A-1'])]['status'] == 'unmatched'
    assert window.track_state().tracks[cl.cluster_id(['B-1'])]['status'] == 'unmatched'
    assert window.track_state().verdicts == {}
    _close(window)


def test_letter_keys_type_in_notes(qt, theme_env):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    window.pair_review_action.setChecked(True)
    days = window.days
    days.pick('A-1')
    days.move_day(1)
    _activate(window)
    notes = window.evidence.notes
    notes.setFocus()
    QtWidgets.QApplication.processEvents()
    QTest.keyClicks(notes, 'ewsx')
    assert notes.toPlainText() == 'ewsx'
    assert window.track_state().verdicts == {}
    assert days.strung == {'B': 'B-1'}
    QTest.keyClick(notes, QtCore.Qt.Key_Escape)
    assert days.strung == {}
    _close(window)


def test_explicit_tracked_choice_clears_hidden_day_comparison(qt, theme_env):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    days, tracked = window.days, window.tracked
    days.pick('A-3')
    days.toggle('B-2')
    # The same row is already current because picking A-3 synchronized the list.
    row = tracked.table.currentRow()
    assert tracked.current_cluster()[0] == cl.cluster_id(['A-3'])
    tracked.table.cellClicked.emit(row, 0)
    assert window.state.selection == ['A-3']
    assert days.selected == ['A-3']
    assert days.strung == {} and days.extras == []
    days.merge()
    assert cl.cluster_id(['A-3', 'B-2']) not in window.track_state().tracks
    _close(window)


def test_tracked_keys_match_selected_rows_and_unlink_selected_member(qt, theme_env):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    _activate(window)
    window._docks['clusters'].raise_()
    _settle(qt)
    table = window.tracked.table
    ids = [cl.cluster_id(['A-3']), cl.cluster_id(['B-2'])]
    rows = [next(i for i in range(table.rowCount()) if table.item(i, 0).data(QtCore.Qt.UserRole) == tid)
            for tid in ids]
    table.setCurrentCell(rows[0], 0)
    table.selectRow(rows[0])
    table.selectionModel().select(table.model().index(rows[1], 0),
                                  QtCore.QItemSelectionModel.Select | QtCore.QItemSelectionModel.Rows)
    table.setFocus()
    _press(window, QtCore.Qt.Key_Q)
    merged = cl.cluster_id(['A-3', 'B-2'])
    assert merged in window.track_state().tracks
    _press(window, QtCore.Qt.Key_E)
    assert window.track_state().tracks[merged]['status'] == 'accepted'
    members = window.tracked.members
    row = next(i for i in range(members.rowCount()) if members.item(i, 0).data(QtCore.Qt.UserRole) == 'B-2')
    members.setCurrentCell(row, 0)
    members.selectRow(row)
    members.setFocus()
    _press(window, QtCore.Qt.Key_X)
    assert window.track_state().tracks[cl.cluster_id(['B-2'])]['status'] == 'unmatched'
    assert window.track_state().tracks[cl.cluster_id(['A-3'])]['status'] == 'unmatched'
    _close(window)


def test_rejected_tracked_row_cannot_accept_hidden_singleton_with_key(qt, theme_env):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    window.days.pick('A-1')
    window.days.reject()
    _activate(window)
    window._docks['clusters'].raise_()
    _settle(qt)
    table = window.tracked.table
    rejected = cl.cluster_id(['A-1', 'B-1'])
    row = next(i for i in range(table.rowCount()) if table.item(i, 0).data(QtCore.Qt.UserRole) == rejected)
    table.setCurrentCell(row, 0)
    table.setFocus()
    assert window.tracked.current_status() == 'rejected'
    assert not window.tracked.buttons['Accept'].isEnabled()
    _press(window, QtCore.Qt.Key_E)
    assert not any(track['status'] == 'accepted' for track in window.track_state().tracks.values())
    # The front Tracked tab remains the action context when focus moves into a plot.
    window.evidence.setFocus()
    _press(window, QtCore.Qt.Key_E)
    assert not any(track['status'] == 'accepted' for track in window.track_state().tracks.values())
    _close(window)


@pytest.mark.parametrize('key', [QtCore.Qt.Key_E, QtCore.Qt.Key_R])
def test_tracked_decision_key_does_not_advance_when_decided_track_is_hidden(qt, theme_env, key):
    window = _day_window(theme_env)
    window.set_data(day_table_data(), curation_dir=theme_env / 'cur')
    window.tracked.hide_done.setChecked(True)
    _activate(window)
    window._docks['clusters'].raise_()
    _settle(qt)
    table = window.tracked.table
    track_id = cl.cluster_id(['A-1', 'B-1'])
    row = next(i for i in range(table.rowCount()) if table.item(i, 0).data(QtCore.Qt.UserRole) == track_id)
    table.setCurrentCell(row, 0)
    table.selectRow(row)
    table.setFocus()
    _press(window, key)
    assert table.rowCount() > 0  # other groups are available, but none becomes current
    assert table.currentRow() == -1
    assert window.tracked.current_cluster() == (None, [])
    assert window.tracked.members.rowCount() == 0
    assert window.state.selection == [] and window.state.row is None
    assert window.days.picked_unit == 'A-1'
    assert not window.tracked.buttons['Accept'].isEnabled()
    for site, unit in [('A', 'A-1'), ('B', 'B-1')]:
        day_table = window.days.tables[site]
        at = window.days.column_ids(site).index(unit)
        assert day_table.item(at, 0).data(theme.TONE_ROLE) == ('accept' if key == QtCore.Qt.Key_E else None)
        review_tip = 'accepted for export' if key == QtCore.Qt.Key_E else 'unreviewed'
        assert review_tip in day_table.item(at, 4).toolTip()
    history = theme_env / 'cur' / 'history.csv'
    saved = history.read_bytes()
    _press(window, QtCore.Qt.Key_E)
    _press(window, QtCore.Qt.Key_R)
    assert history.read_bytes() == saved  # repeated decisions cannot affect the next group
    _close(window)


def test_pair_review_is_opt_in_and_preserves_saved_verdicts(qt, theme_env):
    window = _day_window(theme_env)
    data = day_table_data()
    window.set_data(data, curation_dir=theme_env / 'cur')
    window.days.pick('A-1')
    window.days.move_day(1)
    _activate(window)
    assert not window.pair_review_enabled
    assert window.evidence.pair_review.isHidden()
    assert window._docks['chains'].isHidden()
    assert not window._docks['chains'].toggleViewAction().isVisible()
    assert not any(s.isEnabled() for s in window.verdict_shortcuts)
    _press(window, QtCore.Qt.Key_E, QtCore.Qt.AltModifier)
    window.set_verdict('accept')
    assert window.track_state().verdicts == {}
    window.pair_review_action.setChecked(True)
    assert not window.evidence.pair_review.isHidden()
    assert not window._docks['chains'].isHidden()
    assert all(s.isEnabled() for s in window.verdict_shortcuts)
    window.evidence.notes.setPlainText('legacy pair annotation')
    window.set_verdict('accept')
    verdicts = window.track_state().verdicts.copy()
    assert verdicts[('A-1', 'B-1')]['notes'] == 'legacy pair annotation'
    _close(window)
    reopened = _day_window(theme_env)
    reopened.set_data(data, curation_dir=theme_env / 'cur')
    assert reopened.track_state().verdicts == verdicts
    assert not reopened.pair_review_enabled
    assert reopened._docks['chains'].isHidden()  # saved visible layout cannot reopen advanced controls
    assert not reopened._docks['chains'].toggleViewAction().isEnabled()
    reopened._reset_layout()
    assert reopened._docks['chains'].isHidden()
    _close(reopened)


def test_pair_button_labels_follow_custom_bindings(qt, theme_env):
    window = _day_window(theme_env)
    assert [b.text() for b in window.evidence.verdict_buttons] == [
        'Pair same (Alt+E)', 'Pair different (Alt+R)', 'Pair unsure (Alt+U)']
    keys.save_binding('accept', 'Alt+Y')
    keys.save_binding('reject', '')
    assert [b.text() for b in window.evidence.verdict_buttons] == [
        'Pair same (Alt+Y)', 'Pair different', 'Pair unsure (Alt+U)']
    _close(window)


def test_inspect_only_keys_move_only(qt, theme_env):
    window = _day_window(theme_env)
    days = window.days
    _activate(window)
    days.pick('A-1')
    _press(window, QtCore.Qt.Key_D)
    assert days.strung == {'B': 'B-1'}
    assert [s.isEnabled() for s in window.day_shortcuts] == [False] * 4
    assert window.shortcuts['right'].isEnabled() is True
    _press(window, QtCore.Qt.Key_Q)
    assert days.status.text() == ''
    _close(window)


def test_rebound_key_applies_live(qt, theme_env):
    window = _day_window(theme_env)
    days = window.days
    keys.save_binding('right', 'L')
    assert window.shortcuts['right'].key().toString() == 'L'
    _activate(window)
    days.pick('A-1')
    _press(window, QtCore.Qt.Key_L)
    assert days.strung == {'B': 'B-1'}
    _press(window, QtCore.Qt.Key_Escape)
    _press(window, QtCore.Qt.Key_D)
    assert days.strung == {}
    _close(window)


def test_settings_live_in_a_fresh_tmp_folder():
    p = Path(theme.settings().fileName())
    assert p.name == 'multiphy.ini'
    assert p.parent.name == 'multiphy'
    assert re.fullmatch(r'settings\d+', p.parent.parent.name) is not None
