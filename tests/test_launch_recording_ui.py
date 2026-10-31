"""Recording-entry workflows use temporary paths and never start a child process."""
from pathlib import Path

import pytest
from PySide6 import QtWidgets

from multiphy.launch.plugins import PluginApi
from multiphy.launch.window import LauncherWindow
from multiphy.project.folder import Project

from .launch_window_fixtures import FakeProcess, qt, wait_until  # noqa: F401 (fixture by name)


def _sort(path: Path, raw=None) -> Path:
    """Only filenames and params are inspected by the form; no array data is needed."""
    path.mkdir(parents=True, exist_ok=True)
    params = 'sample_rate = 30000\nn_channels_dat = 4\n'
    if raw is not None:
        params += f'dat_path = {str(raw)!r}\n'
    (path / 'params.py').write_text(params, encoding='utf-8')
    for name in ('spike_times.npy', 'spike_clusters.npy', 'channel_positions.npy', 'cluster_group.tsv'):
        (path / name).touch()
    return path


def _fill(window: LauncherWindow, **values) -> None:
    for dest, value in values.items():
        window.form_widgets['init'][dest].setText(str(value))
    assert wait_until(lambda: not window._init_validation_timer.isActive(), 2)


def _manual(window: LauncherWindow, tmp_path: Path) -> None:
    _fill(window, day='baseline', date='2026-02-01', sort=_sort(tmp_path / 'sort'))


@pytest.fixture
def window(qt, tmp_path):
    root = tmp_path / 'lab'
    Project.ensure(root, 'P')
    widget = LauncherWindow(root, 'P', plugins=PluginApi(), process_factory=FakeProcess,
                            interpreter_lookup=lambda: None)
    yield widget
    widget.clear_queue()
    if widget.running():
        widget.process.ended.emit(1)
    widget.close()
    QtWidgets.QApplication.processEvents()


def test_recording_examples_are_not_values_and_missing_details_explain_disabled_button(window):
    fields = window.form_widgets['init']
    assert window._init_mode() == 'manual'
    assert window.run_buttons['init'].text() == 'Add recording'
    assert fields['day'].placeholderText() == 'e.g. baseline or D01'
    assert fields['date'].placeholderText() == 'YYYY-MM-DD'
    assert fields['uv_per_bit'].placeholderText() == 'From acquisition metadata'
    assert all(not fields[name].text() for name in ('day', 'date', 'sort', 'uv_per_bit'))
    assert 'structure.oebin' in fields['uv_per_bit'].toolTip()
    assert 'bit_volts' in fields['uv_per_bit'].toolTip()
    assert 'units' in fields['uv_per_bit'].toolTip()
    assert not window.run_buttons['init'].isEnabled()
    assert 'recording name' in window.init_status_label.text().lower()
    window.run_command('init')
    assert window.process is None
    assert not window._init_error_labels['day'].isHidden()


def test_invalid_date_has_inline_explanation_and_correction_enables_add(window, tmp_path):
    _manual(window, tmp_path)
    assert window.run_buttons['init'].isEnabled()
    _fill(window, date='2026-02-30')
    assert not window.run_buttons['init'].isEnabled()
    assert not window._init_error_labels['date'].isHidden()
    assert 'valid recording date' in window._init_error_labels['date'].text()
    _fill(window, date='2026-02-28')
    assert window.run_buttons['init'].isEnabled()
    assert window._init_error_labels['date'].isHidden()


def test_modes_keep_drafts_and_only_submit_the_selected_mode(window, tmp_path):
    _manual(window, tmp_path)
    incoming = tmp_path / 'incoming'
    _sort(incoming / 'D02' / 'sort')
    (incoming / 'D02' / 'date.txt').write_text('2026-02-02', encoding='utf-8')
    _fill(window, from_dir=incoming)
    assert 'from_dir' not in window._values('init')
    assert window._init_rows['from_dir'][1].isHidden()
    window.init_import_button.click()
    assert window._init_mode() == 'import'
    assert window.run_buttons['init'].text() == 'Import recordings'
    assert window._init_rows['day'][1].isHidden()
    assert not window._init_rows['from_dir'][1].isHidden()
    assert 'day' not in window._values('init')
    assert window.run_buttons['init'].isEnabled()
    window.run_command('init')
    arguments = window.process.calls[0][1]
    assert arguments[4:] == ['init', 'P', '--from', str(incoming)]
    window.process.ended.emit(1)
    window.init_manual_button.click()
    assert window.form_widgets['init']['day'].text() == 'baseline'
    assert window.form_widgets['init']['date'].text() == '2026-02-01'
    assert window.form_widgets['init']['from_dir'].text() == str(incoming)


def test_raw_detection_preview_distinguishes_included_from_excluded_voltage(window, tmp_path):
    raw = tmp_path / 'continuous.dat'
    raw.write_bytes(b'\x00\x00' * 4)
    _fill(window, day='D01', date='2026-02-01', sort=_sort(tmp_path / 'sort', raw))
    assert window.run_buttons['init'].isEnabled()
    assert str(raw) in window.init_raw_label.text()
    assert 'raw voltage will be excluded' in window.init_raw_label.text()
    _fill(window, uv_per_bit='0.2')
    assert window.run_buttons['init'].isEnabled()
    assert 'raw voltage will be included' in window.init_raw_label.text()
    assert window.form_widgets['init']['raw'].text() == ''
    _fill(window, uv_per_bit='', raw=raw)
    assert not window.run_buttons['init'].isEnabled()
    assert not window._init_error_labels['uv_per_bit'].isHidden()
    assert 'voltage scale' in window._init_error_labels['uv_per_bit'].text()


@pytest.mark.parametrize('exit_code', [0, 1])
def test_finished_manual_recording_clears_only_on_success(window, tmp_path, exit_code):
    _manual(window, tmp_path)
    _fill(window, probe='Probe A', uv_per_bit='0.2')
    submitted = window._values('init')
    window.run_command('init')
    window.process.ended.emit(exit_code)
    values = window._values('init')
    if exit_code:
        assert values == submitted
        assert window.run_buttons['init'].isEnabled()
    else:
        assert values['project'] == 'P'
        assert values['probe'] == 'Probe A'
        assert all(values[name] == '' for name in ('day', 'date', 'sort', 'events', 'raw', 'uv_per_bit'))
        assert not window.run_buttons['init'].isEnabled()


def test_success_preserves_next_recording_draft_entered_while_running(window, tmp_path):
    _manual(window, tmp_path)
    window.run_command('init')
    _fill(window, day='next_recording', date='2026-02-02')
    next_draft = window._values('init')
    window.process.ended.emit(0)
    assert window._values('init') == next_draft
    assert window.run_buttons['init'].isEnabled()


def test_success_preserves_draft_after_switching_modes_while_running(window, tmp_path):
    _manual(window, tmp_path)
    window.run_command('init')
    window.init_import_button.click()
    _fill(window, from_dir=tmp_path / 'incoming')
    window.process.ended.emit(0)
    assert window._init_mode() == 'import'
    assert window.form_widgets['init']['from_dir'].text() == str(tmp_path / 'incoming')
    window.init_manual_button.click()
    assert window.form_widgets['init']['day'].text() == 'baseline'


def test_success_preserves_hidden_import_draft_when_returned_to_manual_mode(window, tmp_path):
    _manual(window, tmp_path)
    window.run_command('init')
    window.init_import_button.click()
    _fill(window, from_dir=tmp_path / 'incoming')
    window.init_manual_button.click()
    window.process.ended.emit(0)
    assert window.form_widgets['init']['day'].text() == ''
    window.init_import_button.click()
    assert window.form_widgets['init']['from_dir'].text() == str(tmp_path / 'incoming')


def test_successful_import_clears_source_and_scale_but_keeps_project_probe(window, tmp_path):
    incoming = tmp_path / 'incoming'
    _sort(incoming / 'D02' / 'sort')
    (incoming / 'D02' / 'date.txt').write_text('2026-02-02', encoding='utf-8')
    window.init_import_button.click()
    _fill(window, from_dir=incoming, uv_per_bit='0.2', probe='Probe A')
    window.run_command('init')
    window.process.ended.emit(0)
    assert window._init_mode() == 'import'
    assert window._values('init') == {'project': 'P', 'probe': 'Probe A', 'uv_per_bit': '', 'from_dir': ''}
    assert not window.run_buttons['init'].isEnabled()


def test_queued_recording_uses_submitted_values_and_preserves_later_draft(window, tmp_path):
    window.run_command('status')
    _manual(window, tmp_path)
    window.run_command('init')
    assert len(window.queue) == 1
    _fill(window, day='next_recording', date='2026-02-02')
    next_draft = window._values('init')
    window.process.ended.emit(0)
    assert window.submitted['command'] == 'init'
    arguments = window.process.calls[0][1]
    assert arguments[arguments.index('--day') + 1] == 'baseline'
    assert arguments[arguments.index('--date') + 1] == '2026-02-01'
    window.process.ended.emit(0)
    assert window._values('init') == next_draft


def test_submit_rechecks_paths_changed_outside_the_form(window, tmp_path):
    _manual(window, tmp_path)
    assert window.run_buttons['init'].isEnabled()
    (tmp_path / 'sort' / 'params.py').unlink()
    window.run_command('init')
    assert window.process is None
    assert 'params.py' in window._init_error_labels['sort'].text()
    assert not window._init_error_labels['sort'].isHidden()
    assert not window.run_buttons['init'].isEnabled()


def test_blur_rechecks_recording_files_fixed_outside_the_form(window, tmp_path):
    _manual(window, tmp_path)
    params = tmp_path / 'sort' / 'params.py'
    params.unlink()
    window.form_widgets['init']['sort'].editingFinished.emit()
    assert not window.run_buttons['init'].isEnabled()
    assert 'params.py' in window._init_error_labels['sort'].text()
    _sort(tmp_path / 'sort')
    window.form_widgets['init']['sort'].editingFinished.emit()
    assert window.run_buttons['init'].isEnabled()
    assert window._init_error_labels['sort'].isHidden()


def test_root_change_rebuilds_recording_form_for_new_project(window, tmp_path):
    _manual(window, tmp_path)
    window.init_import_button.click()
    second_root = tmp_path / 'second_lab'
    Project.ensure(second_root, 'R')
    assert window.set_root(second_root)
    assert window.form_widgets['init']['project'].text() == 'R'
    assert window._init_mode() == 'manual'
    assert window.run_buttons['init'].text() == 'Add recording'
    assert not window.run_buttons['init'].isEnabled()
    _manual(window, tmp_path)
    window.run_command('init')
    assert window.process.calls[0][1][2:6] == ['--root', str(second_root), 'init', 'R']
