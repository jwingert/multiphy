"""The launcher window's Cancel, kill grace, queue list and close bar, then real CLI children end to
end; every file lives under ``tmp_path``, and every window gets ``interpreter_lookup=lambda: None``."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import typing

import pytest
from PySide6 import QtCore, QtWidgets
from PySide6.QtTest import QTest

from multiphy.launch import window as window_module
from multiphy.launch.plugins import PluginApi, load_plugins
from multiphy.launch.window import ACROSS_DAYS, CommandProcess, LauncherWindow
from multiphy.project.folder import Project

from .launch_window_fixtures import FakeProcess, held_reads, open_and_wait, qt, runs, wait_until  # noqa: F401 (fixtures by name)
from .synthetic_day import write_day


def _window(root: Path, project: typing.Optional[str] = None, *, plugins=None, factory=FakeProcess) -> LauncherWindow:
    window = LauncherWindow(root, project, plugins=plugins if plugins is not None else PluginApi(),
                            process_factory=factory, interpreter_lookup=lambda: None)
    window.show()
    return window


def _made():
    made = []

    def factory(parent):
        process = FakeProcess(parent)
        made.append(process)
        return process
    return made, factory


def _log(window: LauncherWindow) -> list:
    return window.log_pane.toPlainText().splitlines()


def _close(widget) -> None:
    widget.close()
    QtWidgets.QApplication.processEvents()


def _check(window: LauncherWindow, *days: str) -> None:
    for i in range(window.days_list.count()):
        item = window.days_list.item(i)
        item.setCheckState(QtCore.Qt.Checked if item.data(QtCore.Qt.UserRole) in days else QtCore.Qt.Unchecked)


def _backend(window: LauncherWindow, backend_id: str) -> None:
    window.backend_combo.setCurrentIndex(window.backend_combo.findData(backend_id))


def _queue_texts(window: LauncherWindow) -> list:
    return [window.queue_list.item(i).text() for i in range(window.queue_list.count())]


def _remove_button(window: LauncherWindow, row: int) -> QtWidgets.QToolButton:
    widget = window.queue_list.itemWidget(window.queue_list.item(row))
    button = widget.findChild(QtWidgets.QToolButton)
    assert button.text() == '×' and button.toolTip() == 'remove'
    return button


def _days(window: LauncherWindow) -> list:
    return [window.days_list.item(i).text() for i in range(window.days_list.count())]


def test_cancel_interrupts_then_kills(qt, runs, monkeypatch):
    monkeypatch.setattr(window_module, 'KILL_AFTER_MS', 50)
    window = _window(runs['root'], 'P')
    window.run_command('status')
    assert window.cancel_button.isEnabled() is False
    window.process.started.emit()
    assert window.cancel_button.isEnabled() is True
    QTest.mouseClick(window.cancel_button, QtCore.Qt.LeftButton)
    QTest.mouseClick(window.cancel_button, QtCore.Qt.LeftButton)
    assert window.process.interrupts == 1
    assert window.process.kills == 0
    QTest.qWait(300)
    assert window.process.kills == 1
    window.process.ended.emit(-1)
    assert _log(window)[-1] == 'killed'
    _close(window)


def test_cancel_no_kill_after_exit(qt, runs, monkeypatch):
    monkeypatch.setattr(window_module, 'KILL_AFTER_MS', 50)
    window = _window(runs['root'], 'P')
    window.run_command('status')
    process = window.process
    process.started.emit()
    window.cancel_button.click()
    process.ended.emit(130)
    QTest.qWait(300)
    assert process.kills == 0
    assert _log(window)[-1] == 'exit 130'
    _close(window)


def test_close_confirm(qt, runs):
    window = _window(runs['root'], 'P')
    _check(window, 'D1', 'D2')
    window.run_command('run')
    process = window.process
    process.started.emit()
    window.close()
    assert window.isVisible() is True
    assert window.close_bar.isHidden() is False
    assert window.close_label.text() == 'run running · cancel and close?'
    window.close_no.click()
    assert window.close_bar.isHidden() is True
    assert window.isVisible() is True
    window.close()
    window.close_yes.click()
    assert process.interrupts == 1
    process.ended.emit(130)
    assert window.isVisible() is False


def test_close_bar_hides_when_command_ends(qt, runs):
    window = _window(runs['root'], 'P')
    window.run_command('status')
    window.process.started.emit()
    window.close()
    assert window.close_bar.isHidden() is False
    assert window.close_label.text() == 'status running · cancel and close?'
    window.run_command('status')
    window.run_command('status')
    assert window.close_label.text() == 'status running · 2 waiting · cancel and close?'
    _remove_button(window, 0).click()
    assert window.close_label.text() == 'status running · 1 waiting · cancel and close?'
    window.clear_button.click()
    assert window.close_label.text() == 'status running · cancel and close?'
    assert window.close_bar.isHidden() is False
    window.process.ended.emit(0)
    assert window.close_bar.isHidden() is True
    assert window.isVisible() is True
    assert _log(window)[-1] == 'exit 0'
    _close(window)


def test_close_with_waiting_drops_queue(qt, runs):
    made, factory = _made()
    window = _window(runs['root'], 'P', factory=factory)
    _check(window, 'D1', 'D2')
    window.run_command('run')
    made[0].started.emit()
    window.run_command('status')
    window.run_command('status')
    window.close()
    assert window.close_label.text() == 'run running · 2 waiting · cancel and close?'
    window.close_no.click()
    assert len(window.queue) == 2
    window.close()
    window.close_yes.click()
    assert window.queue == []
    assert window.queue_box.isHidden()
    assert made[0].interrupts == 1
    assert not any(button.isEnabled() for button in window.run_buttons.values())
    window.run_command('status')
    assert window.queue == []
    assert len(made) == 1
    made[0].ended.emit(130)
    assert len(made) == 1
    assert window.isVisible() is False


def test_queue_list_remove_and_clear(qt, runs):
    made, factory = _made()
    window = _window(runs['root'], 'P', factory=factory)
    _check(window, 'D1', 'D2')
    _backend(window, 'functional_likelihood')
    window.run_command('status')
    window.run_command('run')
    window.run_command('status')
    assert not window.queue_box.isHidden()
    assert _queue_texts(window) == ['launch run P --backend functional_likelihood --days D1 D2', 'launch status P']
    _remove_button(window, 0).click()
    assert [j['command'] for j in window.queue] == ['status']
    assert _queue_texts(window) == ['launch status P']
    window.run_command('run')
    window.clear_button.click()
    assert window.queue == []
    assert window.queue_list.count() == 0
    assert window.queue_box.isHidden()
    assert made[0].interrupts == 0
    made[0].ended.emit(0)
    assert len(made) == 1
    assert window.root_action.isEnabled() is True
    assert typing.get_type_hints(LauncherWindow.remove_queued) == {'index': int, 'return': type(None)}
    assert typing.get_type_hints(LauncherWindow.clear_queue) == {'return': type(None)}
    _close(window)


def test_cancel_before_started_waits(qt, runs, monkeypatch):
    monkeypatch.setattr(window_module, 'KILL_AFTER_MS', 50)
    made, factory = _made()
    window = _window(runs['root'], 'P', factory=factory)
    window.run_command('status')
    window.close()
    window.close_yes.click()
    assert made[0].interrupts == 0
    assert window.cancel_button.isEnabled() is False
    made[0].started.emit()
    assert made[0].interrupts == 1
    assert window.cancel_button.isEnabled() is False
    made[0].ended.emit(130)
    assert window.isVisible() is False

    made, factory = _made()
    window = _window(runs['root'], 'P', factory=factory)
    window.run_command('status')
    window.close()
    window.close_yes.click()
    made[0].ended.emit(-1)
    QTest.qWait(300)
    assert made[0].interrupts == 0
    assert made[0].kills == 0
    assert window.isVisible() is False


def test_close_idle_leaves_viewers(qt, runs):
    window = _window(runs['root'], 'P')
    window.run_tree.setCurrentItem(window._find_run(runs['across']))
    open_and_wait(window)
    viewer = window.viewers[-1]
    try:
        window.close()
        assert window.isHidden()
        assert viewer.isVisible() is True
    finally:
        _close(viewer)


def _finish(window: LauncherWindow, held) -> None:
    """Release ``held``, wait for the open to end, close every viewer the window opened, end a running
    fake command and close the window."""
    held.release()
    wait_until(lambda: not window.opening(), 120)
    for viewer in window.viewers:
        _close(viewer)
    if window.running():
        window.process.ended.emit(0)
    _close(window)


def _open_across(window: LauncherWindow, runs: dict) -> str:
    """Select the across run, open it and return its run list label."""
    item = window._find_run(runs['across'])
    label = item.text(0)
    window.run_tree.setCurrentItem(item)
    window.open_run()
    return label


def test_close_while_opening_waits_for_viewer(qt, runs, held_reads):
    window = _window(runs['root'], 'P')
    try:
        label = _open_across(window, runs)
        window.close()
        assert window.isVisible() is True
        assert _log(window)[-1] == f'closing · once {label} opens'
        assert window.close_bar.isHidden() is True
        window.close()
        assert _log(window).count(f'closing · once {label} opens') == 1
        held_reads.release()
        assert wait_until(window.isHidden, 120) is True
        assert len(window.viewers) == 1
        assert window.viewers[0].isVisible() is True
    finally:
        _finish(window, held_reads)


def test_close_while_opening_failure_keeps_window(qt, runs, held_reads):
    held_reads.error = ValueError('boom')
    window = _window(runs['root'], 'P')
    try:
        label = _open_across(window, runs)
        window.close()
        held_reads.release()
        assert wait_until(lambda: not window.opening(), 120)
        assert window.isVisible() is True
        assert _log(window)[-2:] == [f'closing · once {label} opens', 'error: boom']
        assert window.viewers == []
        held_reads.error = None
        open_and_wait(window)
        assert window.isVisible() is True
        assert len(window.viewers) == 1
        window.close()
        assert window.isHidden() is True
    finally:
        _finish(window, held_reads)


def test_close_bar_yes_while_opening_waits(qt, runs, held_reads):
    window = _window(runs['root'], 'P')
    try:
        window.run_command('status')
        window.process.started.emit()
        label = _open_across(window, runs)
        window.close()
        assert window.close_bar.isHidden() is False
        assert window.close_label.text() == 'status running · cancel and close?'
        assert not any(line.startswith('closing · once') for line in _log(window))
        window.close_yes.click()
        window.process.ended.emit(130)
        assert window.isVisible() is True
        assert _log(window)[-2:] == ['exit 130', f'closing · once {label} opens']
        held_reads.release()
        assert wait_until(window.isHidden, 120) is True
        assert window.viewers[0].isVisible() is True
    finally:
        _finish(window, held_reads)


def test_close_bar_yes_then_failed_open_stays_usable(qt, runs, held_reads):
    held_reads.error = ValueError('boom')
    window = _window(runs['root'], 'P')
    try:
        window.run_command('status')
        window.process.started.emit()
        label = _open_across(window, runs)
        window.close()
        window.close_yes.click()
        window.process.ended.emit(130)
        held_reads.release()
        assert wait_until(lambda: not window.opening(), 120)
        assert window.isVisible() is True
        assert _log(window)[-3:] == ['exit 130', f'closing · once {label} opens', 'error: boom']
        window.run_command('status')
        assert window.running() is True
        window.process.ended.emit(0)
        window.close()
        assert window.isHidden() is True
    finally:
        _finish(window, held_reads)


def test_close_while_opening_then_close_bar_yes_failed_open_keeps_yes(qt, runs, held_reads):
    held_reads.error = ValueError('boom')
    window = _window(runs['root'], 'P')
    try:
        label = _open_across(window, runs)
        window.close()
        assert _log(window)[-1] == f'closing · once {label} opens'
        window.run_command('status')
        window.process.started.emit()
        window.close()
        assert window.close_bar.isHidden() is False
        window.close_yes.click()
        held_reads.release()
        assert wait_until(lambda: not window.opening(), 120)
        assert window.isVisible() is True
        assert _log(window)[-1] == 'error: boom'
        assert window.running() is True
        window.run_command('status')
        assert window.queue == []
        window.process.ended.emit(130)
        assert window.isHidden() is True
    finally:
        _finish(window, held_reads)


def test_real_cancel_logs_cancelled(qt, tmp_path):
    root = tmp_path / 'lab'
    root.mkdir()
    (root / 'multiphy.json').write_text(json.dumps({'plugins': ['tests.wait_plugin']}), encoding='utf-8')
    Project.ensure(root, 'P')
    window = _window(root, 'P', plugins=load_plugins(root), factory=CommandProcess)
    try:
        window.run_command('wait')
        assert wait_until(lambda: 'waiting' in _log(window), 60)
        assert wait_until(window.cancel_button.isEnabled, 10)
        window.cancel_button.click()
        assert wait_until(lambda: _log(window)[-1:] == ['exit 130'], 30)
        lines = (root / 'P' / 'logs' / 'launcher.log').read_text(encoding='utf-8').splitlines()
        assert lines[-1].endswith(' · wait P · cancelled')
    finally:
        if window.running():
            window.process.kill()
            wait_until(lambda: not window.running(), 10)
        _close(window)


@pytest.fixture
def two_day_from(tmp_path) -> Path:
    """``DIR`` with ``D1`` (seed 0, 2026-01-01) and ``D2`` (seed 1, 2026-01-02), each laid out as
    ``test_launch_cli.py``'s ``from_dir`` lays out one day."""
    for seed, day, date in ((0, 'D1', '2026-01-01'), (1, 'D2', '2026-01-02')):
        sd = write_day(tmp_path / 'in', day, seed=seed, date=date)
        folder = tmp_path / 'from' / day
        folder.mkdir(parents=True)
        (folder / 'sort').symlink_to(sd.sort_dir)
        (folder / 'raw.bin').symlink_to(sd.raw_path)
        shutil.copyfile(sd.events_path, folder / 'events.csv')
        (folder / 'date.txt').write_text(f'{date}\n')
    return tmp_path / 'from'


def test_real_process_pipeline(qt, tmp_path, two_day_from):
    root = tmp_path / 'lab'
    root.mkdir()
    window = _window(root, factory=CommandProcess)
    viewer = None
    try:
        init = window.form_widgets['init']
        init['project'].setText('P')
        init['from_dir'].setText(str(two_day_from))
        init['uv_per_bit'].setText('0.195')
        window.run_command('init')
        assert wait_until(lambda: not window.running(), 600)
        assert window.project_combo.currentText() == 'P'
        assert _days(window) == ['D1 · none', 'D2 · none']

        _check(window)
        window.run_command('extract')
        assert wait_until(lambda: not window.running(), 600)
        assert _days(window) == ['D1 · raw', 'D2 · raw']

        _check(window, 'D1', 'D2')
        _backend(window, 'functional_likelihood')
        window.run_command('run')
        assert wait_until(lambda: not window.running(), 600)
        current = window.run_tree.currentItem()
        assert current is not None and current.parent() is not None
        assert current.parent().text(0) == ACROSS_DAYS
        folder = current.data(0, QtCore.Qt.UserRole).folder
        assert folder.parent == root / 'P' / 'runs'

        open_and_wait(window)
        viewer = window.viewers[-1]
        assert viewer.mode.id == 'across_days'
        assert viewer.curation is not None
        assert _log(window).count('exit 0') == 3, _log(window)
        lines = (root / 'P' / 'logs' / 'launcher.log').read_text(encoding='utf-8').splitlines()
        assert all(line.endswith('· ok') for line in lines[-3:]), lines[-3:]
    finally:
        if viewer is not None:
            _close(viewer)
        _close(window)
