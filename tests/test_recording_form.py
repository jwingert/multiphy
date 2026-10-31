"""Metadata-only feedback for the launcher's manual and folder-import forms."""
from pathlib import Path

import numpy as np
import pytest

from multiphy.launch.recording_form import FIELD_COPY, init_values, validate_init
from multiphy.project.folder import Project
from multiphy.project.manifest import new_day


def _sort(path, params="sample_rate = 30000\nn_channels_dat = 2\n"):
    path.mkdir(parents=True)
    (path / 'params.py').write_text(params, encoding='utf-8')
    for name in ('spike_times.npy', 'spike_clusters.npy', 'channel_positions.npy', 'cluster_group.tsv'):
        (path / name).touch()
    return path


def _values(tmp_path, **changes):
    values = {'project': 'mouse01', 'day': 'D01', 'date': '2026-10-01',
              'sort': str(_sort(tmp_path / 'sort'))}
    values.update(changes)
    return values


def _record(project, name='existing', date='2026-09-01'):
    project.update(lambda manifest: manifest['days'].update({
        name: new_day(date, 'user', {'sort': 'phy@1'}, {}, {})}))


def _import_day(folder, name, date='2026-10-01', raw=False):
    child = folder / name
    _sort(child / 'sort')
    (child / 'date.txt').write_text(date, encoding='utf-8')
    if raw:
        (child / 'raw.dat').touch()
    return child


def test_form_copy_explains_identity_date_and_scale():
    assert FIELD_COPY['day']['label'] == 'Recording name'
    assert FIELD_COPY['date']['placeholder'] == 'YYYY-MM-DD'
    assert FIELD_COPY['uv_per_bit']['placeholder'] == 'From acquisition metadata'
    help_text = FIELD_COPY['uv_per_bit']['help']
    assert 'structure.oebin' in help_text and 'bit_volts' in help_text and 'units' in help_text
    assert '1,000,000' in help_text
    assert '0.195' not in help_text


def test_mode_filter_discards_stale_hidden_arguments():
    values = dict(project=' P ', day='D01', date='2026-10-01', sort='sort', from_dir='import',
                  probe='NP1', uv_per_bit='0.2', raw='raw.dat')
    manual = init_values(values)
    assert manual['project'] == 'P' and 'from_dir' not in manual
    imported = init_values(values, 'import')
    assert imported == {'project': 'P', 'probe': 'NP1', 'uv_per_bit': '0.2', 'from_dir': 'import'}


def test_valid_sorting_only_does_not_read_arrays_or_create_project(tmp_path, monkeypatch):
    values = _values(tmp_path)
    root = tmp_path / 'root'
    monkeypatch.setattr(np, 'load', lambda *args, **kwargs: pytest.fail('Loaded an array during validation'))
    result = validate_init(root, values)
    assert result.ready and result.message == ''
    assert 'sorting data only' in result.raw_message
    assert not root.exists()


@pytest.mark.parametrize('date', ['2026-1-01', '20261001', '2026-02-30', 'not a date', ''])
def test_recording_date_requires_real_iso_date(tmp_path, date):
    result = validate_init(tmp_path / 'root', _values(tmp_path, date=date))
    assert not result.ready and 'date' in result.errors


def test_duplicate_name_and_date_have_distinct_feedback(tmp_path):
    root = tmp_path / 'root'
    project = Project.create(root, 'mouse01')
    _record(project, 'D01', '2026-10-01')
    before = (project.path / 'project.json').read_bytes()
    result = validate_init(root, _values(tmp_path))
    assert 'already exists' in result.errors['day']
    assert 'one recording per date' in result.errors['date']
    assert (project.path / 'project.json').read_bytes() == before


@pytest.mark.parametrize('name', ['day 1', '../other', '', '.hidden'])
def test_invalid_recording_names(tmp_path, name):
    assert 'day' in validate_init(tmp_path / 'root', _values(tmp_path, day=name)).errors


def test_missing_source_fields_report_inline_errors(tmp_path):
    values = _values(tmp_path)
    (Path(values['sort']) / 'params.py').unlink()
    values.update(events=str(tmp_path / 'missing.csv'), raw=str(tmp_path / 'missing.dat'))
    result = validate_init(tmp_path / 'root', values)
    assert {'sort', 'events', 'raw', 'uv_per_bit'} <= set(result.errors)
    assert 'params.py' in result.errors['sort']
    assert result.raw_level == 'error'


def test_sort_folder_checks_required_file_existence_only(tmp_path):
    values = _values(tmp_path)
    (Path(values['sort']) / 'spike_times.npy').unlink()
    result = validate_init(tmp_path / 'root', values)
    assert 'spike_times.npy' in result.errors['sort']


def test_raw_detection_warns_without_scale_and_includes_with_scale(tmp_path):
    values = _values(tmp_path)
    sort = Path(values['sort'])
    (sort / 'params.py').write_text("sample_rate = 30000\nn_channels_dat = 2\ndat_path = '../continuous.dat'\n", encoding='utf-8')
    (tmp_path / 'continuous.dat').touch()
    result = validate_init(tmp_path / 'root', values)
    assert result.ready and result.raw_level == 'warning'
    assert 'raw voltage will be excluded' in result.raw_message
    assert Path(result.detected_raw).resolve() == tmp_path / 'continuous.dat'
    assert values.get('raw') is None
    values['uv_per_bit'] = '0.25'
    result = validate_init(tmp_path / 'root', values)
    assert result.ready and result.raw_level == 'info'
    assert 'raw voltage will be included' in result.raw_message


def test_explicit_raw_requires_scale_and_params_layout(tmp_path):
    values = _values(tmp_path, raw=str(tmp_path / 'voltage.dat'))
    (tmp_path / 'voltage.dat').touch()
    assert 'uv_per_bit' in validate_init(tmp_path / 'root', values).errors
    values['uv_per_bit'] = '0.25'
    assert validate_init(tmp_path / 'root', values).ready
    (Path(values['sort']) / 'params.py').write_text('sample_rate = 30000\n', encoding='utf-8')
    assert 'n_channels_dat' in validate_init(tmp_path / 'root', values).errors['sort']


@pytest.mark.parametrize('scale', ['0', '-1', 'NaN', 'inf', '-inf', '0.25 mV'])
def test_voltage_scale_must_be_positive_finite_number(tmp_path, scale):
    result = validate_init(tmp_path / 'root', _values(tmp_path, uv_per_bit=scale))
    assert 'uv_per_bit' in result.errors


def test_missing_detected_raw_can_be_relocated_or_omitted(tmp_path):
    values = _values(tmp_path)
    sort = Path(values['sort'])
    (sort / 'params.py').write_text("sample_rate = 30000\nn_channels_dat = 2\ndat_path = '../missing.dat'\n", encoding='utf-8')
    result = validate_init(tmp_path / 'root', values)
    assert result.ready and result.raw_level == 'warning'
    assert 'missing raw file' in result.raw_message
    relocated = tmp_path / 'relocated.dat'
    relocated.touch()
    values.update(raw=str(relocated), uv_per_bit='0.25')
    result = validate_init(tmp_path / 'root', values)
    assert result.ready and str(relocated) in result.raw_message


def test_params_are_never_executed(tmp_path):
    values = _values(tmp_path)
    sentinel = tmp_path / 'executed.txt'
    (Path(values['sort']) / 'params.py').write_text(
        f"__import__('pathlib').Path({str(sentinel)!r}).touch()\n", encoding='utf-8')
    result = validate_init(tmp_path / 'root', values)
    assert 'literal assignments' in result.errors['sort']
    assert not sentinel.exists()


def test_probe_mismatch_and_uninitialized_project_folder(tmp_path):
    root = tmp_path / 'root'
    values = _values(tmp_path, probe='NP2')
    project = Project.create(root, 'mouse01')
    project.update(lambda manifest: manifest['identity'].update(probe='NP1'))
    assert 'probe' in validate_init(root, values).errors
    values['probe'] = ''
    assert validate_init(root, values).ready
    (root / 'not_a_project').mkdir()
    values['project'] = 'not_a_project'
    assert 'project.json' in validate_init(root, values).errors['project']


def test_import_preview_counts_sorting_only_and_preserves_sources(tmp_path):
    source = tmp_path / 'import'
    _import_day(source, 'baseline', '2026-10-01')
    _import_day(source, 'followup', '2026-10-02')
    root = tmp_path / 'root'
    result = validate_init(root, {'project': 'P', 'from_dir': str(source)}, 'import')
    assert result.ready and '2 recordings ready' in result.raw_message
    assert '2 will use sorting data only' in result.raw_message
    assert not root.exists()


def test_import_rejects_duplicate_dates_before_any_write(tmp_path):
    source = tmp_path / 'import'
    _import_day(source, 'baseline')
    _import_day(source, 'followup')
    root = tmp_path / 'root'
    result = validate_init(root, {'project': 'P', 'from_dir': str(source)}, 'import')
    assert 'one recording per date' in result.errors['from_dir']
    assert not root.exists()


def test_import_requires_scale_for_explicit_raw(tmp_path):
    source = tmp_path / 'import'
    _import_day(source, 'D01', raw=True)
    values = {'project': 'P', 'from_dir': str(source)}
    assert 'uv_per_bit' in validate_init(tmp_path / 'root', values, 'import').errors
    values['uv_per_bit'] = '0.25'
    result = validate_init(tmp_path / 'root', values, 'import')
    assert result.ready and 'included for every recording' in result.raw_message


def test_import_reports_missing_date_and_multiple_raw_files(tmp_path):
    source = tmp_path / 'import'
    child = _import_day(source, 'D01', raw=True)
    values = {'project': 'P', 'from_dir': str(source), 'uv_per_bit': '0.25'}
    (child / 'raw.bin').touch()
    assert 'more than one raw.*' in validate_init(tmp_path / 'root', values, 'import').errors['from_dir']
    (child / 'date.txt').unlink()
    assert 'missing date.txt' in validate_init(tmp_path / 'root', values, 'import').errors['from_dir']
