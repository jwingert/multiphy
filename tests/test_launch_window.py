"""The launcher window's lists, its open button and ``set_root``; every file lives under ``tmp_path``."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import typing
from typing import Optional

import pytest
from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtTest import QTest

from multiphy.launch import commands
from multiphy.launch import window as launch_window
from multiphy.launch.__main__ import main
from multiphy.launch.plugins import PluginApi
from multiphy.launch.window import LauncherWindow, day_rows, list_projects, run_groups, run_window
from multiphy.project.folder import Project
from multiphy.run import catalog
from multiphy.viewer import app as viewer_app

from .launch_window_fixtures import held_reads, open_and_wait, qt, runs, wait_until  # noqa: F401 (fixtures by name)
from .synthetic_day import extracted_project, write_day


def _window(root: Path, project: Optional[str] = None) -> LauncherWindow:
    return LauncherWindow(root, project, plugins=PluginApi(), interpreter_lookup=lambda: None)


def _last_log(window: LauncherWindow) -> str:
    return window.log_pane.toPlainText().splitlines()[-1]


def _days(window: LauncherWindow) -> list:
    return [window.days_list.item(i).text() for i in range(window.days_list.count())]


def _top(window: LauncherWindow) -> list:
    return [window.run_tree.topLevelItem(i) for i in range(window.run_tree.topLevelItemCount())]


def _close(widget) -> None:
    widget.close()
    QtWidgets.QApplication.processEvents()


def test_list_projects(tmp_path):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    Project.ensure(root, 'R')
    (root / 'Q').mkdir()
    (root / 'x.txt').write_text('x')
    assert list_projects(root) == ['P', 'R']
    assert list_projects(tmp_path / 'missing') == []


def test_day_rows_tags(tmp_path):
    path = extracted_project(tmp_path, tiers={'D2': 'templates'})
    root = path.parent
    d3 = write_day(tmp_path / 'in', 'D3', seed=2, date='2026-01-03')
    commands.init_day(root, 'P', day='D3', sort=str(d3.sort_dir), date=d3.date, events=str(d3.events_path),
                      raw=str(d3.raw_path), uv_per_bit=d3.uv_per_bit)
    assert day_rows(Project.open(root, 'P')) == [('D1', 'raw'), ('D2', 'templates'), ('D3', 'none')]


def test_run_tree_groups(qt, runs, tmp_path):
    fresh = tmp_path / 'lab'
    Project.ensure(fresh, 'Q')
    assert run_groups(Project.open(fresh, 'Q')) == []
    window = _window(fresh, 'Q')
    assert window.project_ok is True
    assert not any(line.startswith('error:') for line in window.log_pane.toPlainText().splitlines())
    _close(window)

    window = _window(runs['root'], 'P')
    top = _top(window)
    assert [item.text(0) for item in top] == ['across days', 'D1']
    entry = catalog.read_run(runs['across'])
    across = top[0]
    assert across.childCount() == 1  # P/runs/unfinished is not listed
    child = across.child(0)
    assert child.text(0) == catalog.run_labels([entry])[0]
    assert child.toolTip(0) == catalog.run_tooltip(entry)
    assert [g for g, _ in run_groups(Project.open(runs['root'], 'P'))] == ['across days', 'D1']
    assert all(e.folder.name != 'unfinished' for _, entries in run_groups(Project.open(runs['root'], 'P'))
               for e in entries)
    _close(window)


def test_open_button_follows_selection(qt, runs):
    window = _window(runs['root'], 'P')
    across, day = _top(window)
    window.run_tree.setCurrentItem(across.child(0))
    assert (window.open_button.text(), window.open_button.isEnabled()) == ('Curate…', True)
    window.run_tree.setCurrentItem(day.child(0))
    assert (window.open_button.text(), window.open_button.isEnabled()) == ('Inspect…', True)
    window.run_tree.setCurrentItem(across)
    assert window.open_button.isEnabled() is False
    _close(window)


def test_open_run_by_kind(qt, runs, monkeypatch):
    window = _window(runs['root'], 'P')
    across, day = _top(window)
    try:
        window.run_tree.setCurrentItem(across.child(0))
        open_and_wait(window)
        assert window.viewers[-1].mode.id == 'across_days'
        assert window.viewers[-1].curation is not None
        window.run_tree.setCurrentItem(day.child(0))
        open_and_wait(window)
        assert window.viewers[-1].mode.id == 'same_day'
        assert window.viewers[-1].curation is None
        assert len(window.viewers) == 2
        monkeypatch.setitem(sys.modules, 'multiphy.viewer.app', None)
        window.open_run()
        assert _last_log(window).startswith('error:')
        assert window.opening() is False
        assert (window.open_button.text(), window.open_button.isEnabled()) == ('Inspect…', True)
        assert len(window.viewers) == 2
    finally:
        for viewer in window.viewers:
            _close(viewer)
        _close(window)


def _close_all(window: LauncherWindow) -> None:
    for viewer in window.viewers:
        _close(viewer)
    _close(window)


def _opening_text(window: LauncherWindow, label: str) -> None:
    text = window.open_button.text()
    assert text[0] in '◐◓◑◒', text
    assert text[1:] == ' opening · ' + label
    assert window.open_button.isEnabled() is False


def test_open_locks_button_while_reading(qt, runs, held_reads, monkeypatch):
    window = _window(runs['root'], 'P')
    try:
        window.resize(1280, 860)
        window.show()
        item = _top(window)[0].child(0)
        label = item.text(0)
        window.run_tree.setCurrentItem(item)
        states = []
        original = viewer_app.open_prepared

        def recording(prepared):
            states.append(window.open_button.isEnabled())
            return original(prepared)

        monkeypatch.setattr(viewer_app, 'open_prepared', recording)
        window.open_run()
        button = window.open_button
        assert (button.text(), button.toolTip(), button.isEnabled()) == (
            '◐ opening · ' + label, '◐ opening · ' + label, False)
        window.open_run()
        assert wait_until(lambda: window.open_button.text() == '◓ opening · ' + label, 5)
        assert window.viewers == []
        held_reads.release()
        assert wait_until(lambda: not window.opening(), 120)
        assert [(p.resolve(), main) for p, main in held_reads.calls] == [(runs['across'].resolve(), False)]
        assert states == [False]
        assert len(window.viewers) == 1
        assert window.viewers[0].mode.id == 'across_days'
        assert window.viewers[0].isVisible() is True
        assert (window.open_button.text(), window.open_button.isEnabled()) == ('Curate…', True)
    finally:
        _close_all(window)


def test_open_lock_survives_redraws(qt, runs, held_reads):
    window = _window(runs['root'], 'P')
    try:
        window.resize(1280, 860)
        window.show()
        item = _top(window)[0].child(0)
        label = item.text(0)
        window.run_tree.setCurrentItem(item)
        window.open_run()
        window.reload()
        _opening_text(window, label)
        window.run_tree.setCurrentItem(_top(window)[1].child(0))  # a day's run
        _opening_text(window, label)
        held_reads.release()
        assert wait_until(lambda: not window.opening(), 120)
        assert (window.open_button.text(), window.open_button.isEnabled()) == ('Inspect…', True)
        assert window.viewers[0].mode.id == 'across_days'
    finally:
        _close_all(window)


def _post_click(widget: QtWidgets.QWidget) -> None:
    centre = QtCore.QPointF(widget.rect().center())
    world = QtCore.QPointF(widget.mapToGlobal(widget.rect().center()))
    for kind, buttons in ((QtCore.QEvent.MouseButtonPress, QtCore.Qt.LeftButton),
                          (QtCore.QEvent.MouseButtonRelease, QtCore.Qt.NoButton)):
        event = QtGui.QMouseEvent(kind, centre, world, QtCore.Qt.LeftButton, buttons, QtCore.Qt.NoModifier)
        QtCore.QCoreApplication.postEvent(widget, event)


def test_clicks_during_window_build_are_dropped(qt, runs, held_reads, monkeypatch):
    window = _window(runs['root'], 'P')
    try:
        window.resize(1280, 860)
        window.show()
        window.run_tree.setCurrentItem(_top(window)[0].child(0))
        held_reads.release()
        original = viewer_app.open_prepared

        def clicking(prepared):
            _post_click(window.open_button)
            _post_click(window.open_button)
            return original(prepared)

        monkeypatch.setattr(viewer_app, 'open_prepared', clicking)
        open_and_wait(window)
        QTest.qWait(200)
        assert len(held_reads.calls) == 1
        assert len(window.viewers) == 1
        assert window.opening() is False
        assert (window.open_button.text(), window.open_button.isEnabled()) == ('Curate…', True)
    finally:
        _close_all(window)


@pytest.mark.parametrize('stage', ('prepare_run', 'open_prepared', 'show'))
def test_failed_open_unlocks(qt, runs, monkeypatch, stage):
    def boom(*_args, **_kwargs):
        raise ValueError('boom')

    if stage == 'show':
        monkeypatch.setattr(viewer_app.App, 'show', boom)
    else:
        monkeypatch.setattr(viewer_app, stage, boom)
    window = _window(runs['root'], 'P')
    try:
        window.run_tree.setCurrentItem(_top(window)[0].child(0))
        open_and_wait(window)
        assert _last_log(window) == 'error: boom'
        assert (window.open_button.text(), window.open_button.isEnabled()) == ('Curate…', True)
        assert window.viewers == []
        lock = runs['root'] / 'P' / 'curation' / 'runs' / runs['across'].name / 'history.lock'
        assert lock.exists() is False
    finally:
        _close_all(window)


def test_opening_text_elides_long_run_label(qt, runs, held_reads):
    long = 'Functional likelihood · 10-15 · 20261015T194032Z_functional_likelihood_f2bd29f1'
    window = _window(runs['root'], 'P')
    try:
        window.resize(1280, 711)
        window.show()
        QTest.qWait(50)
        item = _top(window)[0].child(0)
        item.setText(0, long)
        window.run_tree.setCurrentItem(item)
        column = window.open_button.parentWidget().width()
        height = window.open_button.height()
        window.open_run()
        QTest.qWait(50)
        assert '…' in window.open_button.text()
        assert window.open_button.toolTip()[1:] == ' opening · ' + long
        assert window.open_button.parentWidget().width() == column
        assert window.open_button.width() == column
        assert window.open_button.height() == height
        held_reads.release()
        assert wait_until(lambda: not window.opening(), 120)
        assert window.open_button.text() == 'Curate…'
        assert window.open_button.parentWidget().width() == column
    finally:
        _close_all(window)


def test_reload_error_clears_lists(qt, tmp_path):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    d1 = write_day(tmp_path / 'in', 'D1')
    commands.init_day(root, 'P', day='D1', sort=str(d1.sort_dir), date=d1.date, events=str(d1.events_path),
                      raw=str(d1.raw_path), uv_per_bit=d1.uv_per_bit)
    window = _window(root, 'P')
    assert _days(window) == ['D1 · none']
    (root / 'P' / 'project.json').write_text('{')
    window.reload()
    assert _last_log(window).startswith('error:')
    assert window.days_list.count() == 0
    assert window.run_tree.topLevelItemCount() == 0
    assert window.project_ok is False
    _close(window)


def test_bad_plugin_root_refused(qt, runs, tmp_path, monkeypatch):
    window = _window(runs['root'], 'P')
    root_before, forms_before = window.root, window.forms

    def refused(root: Path, starts: str, exact: bool = False) -> None:
        assert window.set_root(root) is False
        assert window.root == root_before
        assert window.forms is forms_before
        line = _last_log(window)
        assert line == starts if exact else line.startswith(starts), line

    missing = tmp_path / 'missing_plugin'
    missing.mkdir()
    (missing / 'multiphy.json').write_text(json.dumps({'plugins': ['tests.no_such_plugin']}))
    refused(missing, 'error: plugin')

    (tmp_path / 'bad_configure_plugin.py').write_text(
        'def _configure(parser):\n'
        "    raise RuntimeError('configure failed')\n\n\n"
        'def register(api):\n'
        "    api.add_command('boom', _configure, lambda args, root: [], 'boom')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    configure = tmp_path / 'bad_configure'
    configure.mkdir()
    (configure / 'multiphy.json').write_text(json.dumps({'plugins': ['bad_configure_plugin']}))
    refused(configure, 'error: configure failed', exact=True)

    unreadable = tmp_path / 'unreadable'
    unreadable.mkdir()
    config = unreadable / 'multiphy.json'
    config.write_text(json.dumps({'plugins': []}))
    config.chmod(0o000)
    try:
        refused(unreadable, 'error:')
    finally:
        config.chmod(0o644)
    _close(window)


def test_gui_writes_no_log_line(tmp_path, monkeypatch):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    log = root / 'P' / 'logs' / 'launcher.log'
    before = log.read_bytes() if log.exists() else None
    recorded = []

    def fake_run_window(root, project, plugins):
        recorded.append((root, project))
        return 0
    monkeypatch.setattr(launch_window, 'run_window', fake_run_window)
    assert main(['--root', str(root), 'gui', 'P']) == 0
    assert recorded == [(root.resolve(), 'P')]
    after = log.read_bytes() if log.exists() else None
    assert after == before


def test_gui_project_not_in_root(qt, tmp_path):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    window = _window(root, 'NEW')
    assert window.project_combo.currentIndex() == -1
    assert window.new_project == 'NEW'
    _close(window)
    window = _window(root, str(root / 'P'))
    assert window.project_combo.currentText() == 'P'
    assert window.new_project is None
    _close(window)
    window = _window(root, str(root / 'NEW'))
    assert window.new_project == 'NEW'
    _close(window)


def test_root_label_elided(qt, runs, tmp_path):
    root = tmp_path / ('r' * 150)
    Project.ensure(root, 'P')
    window = _window(root)
    window.resize(600, 500)
    window.show()
    QTest.qWait(50)
    full = str(root.resolve())
    assert window.root_label.toolTip() == full
    assert '…' in window.root_label.text()
    assert len(window.root_label.text()) < len(full)
    assert window.root_action.text() == 'Root…'
    assert window.set_root(runs['root']) is True
    assert window.root_label.toolTip() == str(runs['root'].resolve())
    _close(window)


def test_window_interpreter_reads_its_root(qt, tmp_path, monkeypatch):
    first, second = tmp_path / 'A', tmp_path / 'B'
    first.mkdir()
    second.mkdir()
    seen = []

    def spy(*, root, **_kwargs):
        seen.append(root)
        return '/x/python'

    monkeypatch.setattr(launch_window.dependency_setup, 'verified_interpreter', spy)
    window = LauncherWindow(first, plugins=PluginApi(), interpreter_lookup=None)
    window.form_widgets['run']['python'].setText('')
    assert window._run_interpreter() == '/x/python'
    assert seen == [first]
    assert window._run_interpreter() == '/x/python'
    assert seen == [first]
    assert window.set_root(second) is True
    assert window._run_interpreter() == '/x/python'
    assert seen == [first, second]
    _close(window)


def test_window_signatures():
    hints = typing.get_type_hints
    assert hints(list_projects) == {'root': Path, 'return': list}
    assert hints(day_rows) == {'project': Project, 'return': list}
    assert hints(run_groups) == {'project': Project, 'return': list}
    assert hints(LauncherWindow.reload) == {'return': type(None)}
    assert hints(LauncherWindow.set_root) == {'path': Path, 'return': bool}
    assert hints(LauncherWindow.open_run) == {'return': type(None)}
    assert hints(LauncherWindow.opening) == {'return': bool}
    assert hints(open_and_wait) == {'timeout_s': float, 'return': type(None)}
    assert hints(LauncherWindow.append_log) == {'line': str, 'return': type(None)}
    assert hints(run_window) == {'root': Path, 'project': Optional[str], 'plugins': PluginApi, 'return': int}
    assert hints(wait_until) == {'timeout_s': float, 'return': bool}


def test_opens_at_open_size_within_the_screen(qt, monkeypatch, tmp_path):
    Project.ensure(tmp_path / 'lab', 'P')
    area = QtWidgets.QApplication.primaryScreen().availableGeometry()
    window = _window(tmp_path / 'lab')
    want = launch_window.OPEN_SIZE
    share = launch_window.SCREEN_SHARE
    assert (window.width(), window.height()) == (min(want[0], int(area.width() * share)),
                                                 min(want[1], int(area.height() * share)))
    # the offscreen screen (800 x 800) caps 1280 x 860
    assert window.width() < want[0]
    _close(window)
    monkeypatch.setattr(launch_window, 'OPEN_SIZE', (500, 400))
    window = _window(tmp_path / 'lab')
    assert (window.width(), window.height()) == (500, 400)
    _close(window)


def test_panes_have_an_edge_gaps_and_spaced_rows(qt, tmp_path):
    Project.ensure(tmp_path / 'lab', 'P')
    window = _window(tmp_path / 'lab')
    body = window.centralWidget()
    m = body.contentsMargins()
    assert (m.left(), m.top(), m.right(), m.bottom()) == (launch_window.EDGE,) * 4
    assert body.handleWidth() == launch_window.GAP
    assert body.widget(0).handleWidth() == launch_window.GAP
    forms = window.tabs.findChildren(QtWidgets.QFormLayout)
    assert forms
    for f in forms:
        assert f.verticalSpacing() == launch_window.FORM_ROWS
        assert f.horizontalSpacing() == launch_window.FORM_ROWS + 4
    _close(window)
