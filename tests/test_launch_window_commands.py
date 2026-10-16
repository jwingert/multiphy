"""The launcher window's command tabs, ``CommandProcess`` and ``child_command``; every file lives under
``tmp_path``, and every window gets a ``FakeProcess`` and ``interpreter_lookup=lambda: None``."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys
import typing

import pytest
from PySide6 import QtCore, QtWidgets

from multiphy.contract import registry
from multiphy.contract.backend import Availability
from multiphy.launch import commands
from multiphy.launch import forms
from multiphy.launch.plugins import PluginApi, load_plugins
from multiphy.launch.window import REPO, CommandProcess, LauncherWindow, child_command
from multiphy.project.folder import Project
from multiphy.project.selection import Selection

from .launch_window_fixtures import FakeProcess, qt, runs, wait_until  # noqa: F401 (fixtures by name)
from .synthetic_day import extracted_project, write_day


def _window(root: Path, project: typing.Optional[str] = None, *, plugins=None, factory=FakeProcess) -> LauncherWindow:
    return LauncherWindow(root, project, plugins=plugins if plugins is not None else PluginApi(),
                          process_factory=factory, interpreter_lookup=lambda: None)


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


def _item(window: LauncherWindow, backend_id: str):
    return window.backend_combo.model().item(window.backend_combo.findData(backend_id))


def _select(window: LauncherWindow, folder: Path) -> None:
    item = window._find_run(folder)
    assert item is not None
    window.run_tree.setCurrentItem(item)


def _current_folder(window: LauncherWindow) -> Path:
    return window.run_tree.currentItem().data(0, QtCore.Qt.UserRole).folder


def _label(window: LauncherWindow, name: str, dest: str) -> QtWidgets.QLabel:
    widget = window.form_widgets[name][dest]
    page = window.tabs.widget(list(window.run_buttons).index(name))
    for layout in page.findChildren(QtWidgets.QFormLayout):
        label = layout.labelForField(widget)
        if label is not None:
            return label
    raise AssertionError(f'no label for {name}.{dest}')


def _set_project(window: LauncherWindow, name: str) -> None:
    window.project_combo.setCurrentIndex(window.project_combo.findText(name))


def _plugin_root(path: Path, module: str) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / 'multiphy.json').write_text(json.dumps({'plugins': [module]}))
    return path


@pytest.fixture
def initialised(tmp_path) -> Path:
    """A root with project ``P`` holding one initialised-only day ``D1``."""
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    d1 = write_day(tmp_path / 'in', 'D1')
    commands.init_day(root, 'P', day='D1', sort=str(d1.sort_dir), date=d1.date, events=str(d1.events_path),
                      raw=str(d1.raw_path), uv_per_bit=d1.uv_per_bit)
    return root


def test_run_tab_argv_and_header(qt, runs):
    root = runs['root']
    window = _window(root, 'P')
    _check(window, 'D1', 'D2')
    _backend(window, 'functional_likelihood')
    window.form_widgets['run']['threshold'].setText('0.8')
    window.run_command('run')
    program, arguments, env, cwd = window.process.calls[0]
    assert program == sys.executable
    assert arguments == ['-m', 'multiphy.launch', '--root', str(root), 'run', 'P', '--backend',
                         'functional_likelihood', '--days', 'D1', 'D2', '--threshold', '0.8']
    assert env['PYTHONPATH'].startswith(str(REPO))
    assert env['PYTHONIOENCODING'] == 'utf-8'
    assert cwd == str(REPO)
    assert _log(window)[-1] == '$ launch run P --backend functional_likelihood --days D1 D2 --threshold 0.8'
    _close(window)


def test_success_selects_printed_run(qt, runs):
    window = _window(runs['root'], 'P')
    _check(window, 'D1', 'D2')
    window.run_command('run')
    window.process.stdout_line.emit(f"{runs['across']} · functional_likelihood · D1,D2 · excluded 0")
    window.process.ended.emit(0)
    assert _current_folder(window).resolve() == Path(runs['across']).resolve()
    assert _log(window)[-1] == 'exit 0'
    assert window.viewers == []
    _close(window)


def test_unlisted_run_line_keeps_selection(qt, runs):
    window = _window(runs['root'], 'P')
    _select(window, runs['same'])
    _check(window, 'D1', 'D2')
    window.run_command('run')
    window.process.stdout_line.emit('/nowhere · functional_likelihood · D1,D2 · excluded 0')
    window.process.ended.emit(0)
    assert _current_folder(window).resolve() == Path(runs['same']).resolve()
    assert _log(window)[-2:] == ['exit 0', 'error: run folder not listed: /nowhere']
    _close(window)


def test_failure_keeps_selection(qt, runs):
    window = _window(runs['root'], 'P')
    _select(window, runs['same'])
    window.run_command('status')
    window.process.stderr_line.emit('error: boom')
    window.process.ended.emit(1)
    assert _current_folder(window).resolve() == Path(runs['same']).resolve()
    assert _log(window)[-2:] == ['error: boom', 'exit 1']
    _close(window)


def test_only_root_locked_while_busy(qt, runs):
    window = _window(runs['root'], 'P')
    _check(window, 'D1', 'D2')
    _select(window, runs['across'])
    before = {name: b.isEnabled() for name, b in window.run_buttons.items()}
    window.run_command('status')
    assert window.running() is True
    assert window.root_action.isEnabled() is False
    assert window.project_combo.isEnabled() is True
    assert window.open_button.isEnabled() is True
    assert {name: b.isEnabled() for name, b in window.run_buttons.items()} == before
    window.process.ended.emit(0)
    assert window.running() is False
    assert window.root_action.isEnabled() is True
    assert {name: b.isEnabled() for name, b in window.run_buttons.items()} == before
    _close(window)


def test_run_while_busy_queues(qt, runs):
    made = []

    def factory(parent):
        process = FakeProcess(parent)
        made.append(process)
        return process
    window = _window(runs['root'], 'P', factory=factory)
    _check(window, 'D1', 'D2')
    window.run_command('run')
    window.run_command('status')
    assert len(made) == 1
    assert len(window.queue) == 1
    made[0].ended.emit(1)
    assert len(made) == 2
    assert made[1].calls[0][1][-2:] == ['status', 'P']
    assert window.queue == []
    assert _log(window)[-2:] == ['exit 1', '$ launch status P']
    assert window.root_action.isEnabled() is False
    made[1].ended.emit(0)
    assert window.root_action.isEnabled() is True
    assert len(made) == 2
    _close(window)


def test_end_follows_only_unmoved_project(qt, tmp_path):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    Project.ensure(root, 'R')
    window = _window(root, 'P')
    window.form_widgets['init']['project'].setText('NEW')
    window.run_command('init')
    _set_project(window, 'R')
    Project.ensure(root, 'NEW')
    window.process.ended.emit(0)
    assert window.project_combo.currentText() == 'R'

    _set_project(window, 'P')
    window.form_widgets['init']['project'].setText(str(root / 'NEW2'))
    window.run_command('init')
    assert window.process.calls[0][1][-2:] == ['init', str(root / 'NEW2')]
    Project.ensure(root, 'NEW2')
    window.process.ended.emit(0)
    assert window.project_combo.currentText() == 'NEW2'
    _close(window)


def test_python_field_survives_backend_change(qt, runs, monkeypatch):
    record = []

    def refusal(project, days, backend_id, *, interpreter):
        record.append((backend_id, interpreter))
        return ''
    monkeypatch.setattr(forms, 'backend_refusal', refusal)
    window = _window(runs['root'], 'P')
    _check(window, 'D1', 'D2')
    window.form_widgets['run']['python'].setText('/lab/python')
    w = window.form_widgets['run']['python']
    record.clear()
    _backend(window, 'dant')
    assert window.form_widgets['run']['python'] is w
    assert w.text() == '/lab/python'
    assert 'preset' in window.form_widgets['run']
    assert 'threshold' not in window.form_widgets['run']
    assert ('functional_likelihood', '/lab/python') in record
    _close(window)


def test_extract_takes_unextracted_checked_days(qt, runs):
    window = _window(runs['root'], 'P')
    _check(window, 'D1')
    assert window.run_buttons['extract'].isEnabled() is False
    _check(window, 'D1', 'D3')
    assert window.run_buttons['extract'].isEnabled() is True
    window.run_command('extract')
    assert window.submitted['argv'] == ['extract', 'P', '--day', 'D3']
    window.process.ended.emit(0)
    _check(window)
    assert window.run_buttons['extract'].isEnabled() is True
    window.run_command('extract')
    assert window.submitted['argv'] == ['extract', 'P', '--day', 'D3']
    window.process.ended.emit(0)
    _close(window)


def test_queued_extract_keeps_days_from_run(qt, tmp_path):
    path = extracted_project(tmp_path, days=('D1',))
    root = path.parent

    def init(day: str, seed: int, date: str) -> None:
        written = write_day(tmp_path / 'in', day, seed=seed, date=date)
        commands.init_day(root, 'P', day=day, sort=str(written.sort_dir), date=written.date,
                          events=str(written.events_path), raw=str(written.raw_path),
                          uv_per_bit=written.uv_per_bit)
    init('D2', 1, '2026-01-02')
    made = []

    def factory(parent):
        process = FakeProcess(parent)
        made.append(process)
        return process
    window = _window(root, 'P', factory=factory)
    _check(window)
    window.run_command('status')
    window.run_command('extract')
    assert len(made) == 1
    assert len(window.queue) == 1
    init('D3', 2, '2026-01-03')
    made[0].ended.emit(0)
    assert len(made) == 2
    assert made[1].calls[0][1][-4:] == ['extract', 'P', '--day', 'D2']
    made[1].ended.emit(0)
    _close(window)


def test_run_needs_extracted_raw_days(qt, runs, tmp_path):
    window = _window(runs['root'], 'P')
    _check(window, 'D3')
    assert window.run_buttons['run'].isEnabled() is False
    _check(window, 'D1', 'D2')
    assert window.run_buttons['run'].isEnabled() is True
    _close(window)
    path = extracted_project(tmp_path, tiers={'D2': 'templates'})
    window = _window(path.parent, 'P')
    _check(window, 'D1', 'D2')
    assert window.run_buttons['run'].isEnabled() is False
    _close(window)


def test_backend_greyed_with_reason(qt, tmp_path, monkeypatch):
    path = extracted_project(tmp_path, conditions={'D2': ('C', 'D')})
    monkeypatch.setattr(registry.get('dant'), 'available', lambda: Availability(False, None, 'missing package'))
    window = _window(path.parent, 'P')
    _check(window, 'D1', 'D2')
    item = _item(window, 'dant')
    assert item.isEnabled() is False
    assert item.toolTip() == 'unavailable here: missing package; no verified interpreter'
    assert _item(window, 'functional_likelihood').isEnabled() is True
    _check(window, 'D2')
    assert _item(window, 'functional_likelihood').isEnabled() is True

    def broken(*args, **kwargs):
        raise OSError('gone')
    monkeypatch.setattr(forms, 'backend_refusal', broken)
    _check(window, 'D1', 'D2')
    model = window.backend_combo.model()
    for i in range(window.backend_combo.count()):
        assert model.item(i).isEnabled() is False
        assert model.item(i).toolTip() == 'error: gone'
    _close(window)
    monkeypatch.undo()
    calls = []

    def lookup():
        calls.append(1)
        raise ValueError('bad env')
    window = LauncherWindow(path.parent, 'P', plugins=PluginApi(), process_factory=FakeProcess,
                            interpreter_lookup=lookup)
    _check(window, 'D1', 'D2')
    _check(window, 'D2')
    _check(window, 'D1', 'D2')
    model = window.backend_combo.model()
    for i in range(window.backend_combo.count()):
        assert model.item(i).isEnabled() is False
        assert model.item(i).toolTip() == 'error: bad env'
    assert len(calls) == 1
    _close(window)


def test_evidence_tags_on_mixed_days(qt, tmp_path):
    path = extracted_project(tmp_path, no_event_days=('D2',))
    window = _window(path.parent, 'P')
    _check(window, 'D1', 'D2')
    _backend(window, 'functional_likelihood')
    reasons = Selection(Project.open(path.parent, 'P'), ('D1', 'D2')).evidence().reasons
    assert window.evidence_tags_shown() == [
        ('no events', 'no events · shared conditions per day pair: D1+D2 0'),
        ('spont · whole recording', 'spont · whole recording · days without baseline windows: D2')]
    assert [tooltip for _text, tooltip in window.evidence_tags_shown()] == list(reasons)
    assert window.run_buttons['run'].isEnabled() is True
    _close(window)


def test_stimulus_choice_greyed_without_events(qt, tmp_path):
    path = extracted_project(tmp_path, conditions={'D2': ('C', 'D')})
    window = _window(path.parent, 'P')
    _check(window, 'D1', 'D2')
    _backend(window, 'functional_likelihood')
    combo = window.form_widgets['run']['feature_stream']
    model = combo.model()
    stimulus = combo.findText('stimulus')
    assert model.item(stimulus).isEnabled() is False
    assert model.item(stimulus).toolTip() == 'no events'
    assert all(model.item(i).isEnabled() and model.item(i).toolTip() == ''
               for i in range(combo.count()) if i != stimulus)
    assert window.run_buttons['run'].isEnabled() is True
    combo.setCurrentIndex(stimulus)
    assert window._ready('run') is False
    assert window.run_buttons['run'].isEnabled() is False
    _check(window, 'D1')
    assert model.item(stimulus).isEnabled() is True
    assert model.item(stimulus).toolTip() == ''
    assert window.run_buttons['run'].isEnabled() is True
    _close(window)


def test_min_repeats_change_refreshes_tags(qt, tmp_path):
    path = extracted_project(tmp_path)
    window = _window(path.parent, 'P')
    _check(window, 'D1', 'D2')
    _backend(window, 'functional_likelihood')
    assert window.evidence_tags_shown() == []
    min_repeats = window.form_widgets['run']['min_repeats']
    min_repeats.setText('6')
    assert window.evidence_tags_shown() == [('no events', 'no events · shared conditions per day pair: D1+D2 0')]
    assert window.run_buttons['run'].isEnabled() is True
    min_repeats.setText('0')
    assert window.evidence_tags_shown() == [('error', 'min_repeats must be an int >= 2, not 0')]
    assert window.run_buttons['run'].isEnabled() is False
    min_repeats.setText('1')
    assert window.evidence_tags_shown() == [('error', 'min_repeats must be an int >= 2, not 1')]
    assert window.run_buttons['run'].isEnabled() is False
    min_repeats.setText('2.5')
    assert window.evidence_tags_shown() == [('error', "min_repeats must be an int >= 2, not '2.5'")]
    assert window.run_buttons['run'].isEnabled() is False
    _check(window)
    assert window.evidence_tags_shown() == [('error', "min_repeats must be an int >= 2, not '2.5'")]
    min_repeats.setText('0')
    assert window.evidence_tags_shown() == [('error', 'min_repeats must be an int >= 2, not 0')]
    assert window.run_buttons['run'].isEnabled() is False
    _check(window, 'D1', 'D2')
    min_repeats.setText('')
    assert window.evidence_tags_shown() == []
    assert window.run_buttons['run'].isEnabled() is True
    _close(window)


def test_required_and_unsupported_fields(qt, runs, tmp_path, monkeypatch):
    window = _window(runs['root'], 'P')
    assert _label(window, 'export', 'run').text() == 'run *'
    assert window.run_buttons['export'].isEnabled() is False
    window.form_widgets['export']['run'].setText('x')
    assert window.run_buttons['export'].isEnabled() is True
    _close(window)

    (tmp_path / 'cli_only_plugin.py').write_text(
        'def _configure(parser):\n'
        "    parser.add_argument('--tag', action='append', help='a tag')\n\n\n"
        'def register(api):\n'
        "    api.add_command('tagged', _configure, lambda args, root: [], 'tagged')\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    root = _plugin_root(tmp_path / 'lab', 'cli_only_plugin')
    Project.ensure(root, 'P')
    window = _window(root, 'P', plugins=load_plugins(root))
    widget = window.form_widgets['tagged']['tag']
    assert isinstance(widget, QtWidgets.QLabel)
    assert widget.text() == 'CLI only · --tag'
    assert widget.toolTip() == '_AppendAction has no form field; run it from the command line'
    assert window.run_buttons['tagged'].isEnabled() is False
    _close(window)


def test_unreadable_project_disables_commands(qt, initialised):
    window = _window(initialised, 'P')
    assert window.run_buttons['extract'].isEnabled() is True
    (initialised / 'P' / 'project.json').write_text('{')
    window.reload()
    for name in ('extract', 'run', 'status', 'export'):
        assert window.run_buttons[name].isEnabled() is False, name
    assert window.run_buttons['init'].isEnabled() is True
    _close(window)


def test_project_field_prefill_and_new_project(qt, tmp_path):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    window = _window(root, 'P')
    assert window.form_widgets['init']['project'].text() == 'P'
    _close(window)

    plugin_root = _plugin_root(tmp_path / 'plugin_lab', 'tests.example_plugin')
    Project.ensure(plugin_root, 'P')
    window = _window(plugin_root, 'NEW', plugins=load_plugins(plugin_root))
    assert window.form_widgets['init']['project'].text() == 'NEW'
    assert window.form_widgets['hello']['project'].text() == 'NEW'
    window.run_command('init')
    assert window.process.calls[0][1][-2:] == ['init', 'NEW']
    window.form_widgets['init']['project'].setText('OTHER')
    Project.ensure(plugin_root, 'NEW')
    window.process.ended.emit(0)
    assert window.project_combo.currentText() == 'NEW'
    _close(window)


def test_labels_hover_placeholders(qt, runs, tmp_path):
    window = _window(runs['root'], 'P')
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == [
        'Initialization', 'Extract features', 'Run matching', 'Export tracks', 'Status']
    assert window.tabs.tabToolTip(0) == 'launch init'
    label = _label(window, 'init', 'uv_per_bit')
    assert label.text() == 'uv per bit'
    assert label.toolTip() == 'microvolts per bit of the raw files'
    _backend(window, 'functional_likelihood')
    assert window.form_widgets['run']['threshold'].placeholderText() == '0.9'
    registration = window.form_widgets['run']['registration']
    assert registration.currentIndex() == -1
    assert registration.placeholderText() == 'matches'
    _close(window)

    root = _plugin_root(tmp_path / 'plugin_lab', 'tests.example_plugin')
    window = _window(root, plugins=load_plugins(root))
    assert window.tabs.tabText(5) == 'hello'
    assert window.tabs.tabToolTip(5) == 'launch hello'
    _close(window)


def test_command_process_lines_and_failed_start(qt, tmp_path):
    def collect(process):
        out, err, codes = [], [], []
        process.stdout_line.connect(out.append)
        process.stderr_line.connect(err.append)
        process.ended.connect(codes.append)
        return out, err, codes

    process = CommandProcess()
    out, err, codes = collect(process)
    process.start(sys.executable, ['-c', "import sys; sys.stdout.buffer.write(b'a \\xc2\\xb7 b\\nbad \\xff\\npartial'); "
                                         "sys.stderr.write('e\\n')"], dict(os.environ), str(tmp_path))
    assert wait_until(lambda: codes, 30.0)
    assert out == ['a · b', 'bad �', 'partial']
    assert err == ['e']
    assert codes == [0]
    assert process.running() is False

    process = CommandProcess()
    out, err, codes = collect(process)
    process.start(str(tmp_path / 'no_such_python'), [], dict(os.environ), str(tmp_path))
    assert wait_until(lambda: codes, 10.0)
    assert len(err) == 1 and err[0].startswith('error:')
    assert codes == [-1]
    assert process.running() is False


def test_command_signatures():
    hints = typing.get_type_hints
    assert hints(CommandProcess.start) == {'program': str, 'arguments': list, 'env': dict, 'cwd': str,
                                           'return': type(None)}
    assert hints(CommandProcess.interrupt) == {'return': type(None)}
    assert hints(CommandProcess.kill) == {'return': type(None)}
    assert hints(CommandProcess.running) == {'return': bool}
    assert hints(child_command) == {'root': Path, 'argv': list, 'return': tuple}
    assert hints(LauncherWindow.run_command) == {'command': str, 'return': type(None)}
    assert hints(LauncherWindow.running) == {'return': bool}
