"""The templates tier: a day's full day from its sort's unwhitened templates, with no raw voltage.

``extract_day`` runs the templates tier by default on a day without a raw file, and on request
on a day with one. It verifies and normalizes as the raw tier does, reads no voltage, writes no
halves and records ``tier`` ``'templates'`` with ``features['halves']`` ``None``. A sort
without templates, the raw tier without a raw file and a second extraction are refused.
"""
from __future__ import annotations

import datetime
import hashlib
from pathlib import Path

import numpy as np
import pytest

from multiphy.extract import pipeline
from multiphy.extract.full_day import load_full_day
from multiphy.extract.pipeline import extract_day
from multiphy.extract.templates import template_full_rows
from multiphy.project.days import add_day
from multiphy.project.folder import Project
from multiphy.sources.base import clear_readers_for_tests
from multiphy.sources.normalize import read_source

from .synthetic_day import write_day

NO_RAW = {'sort': 'phy', 'events': 'events_csv'}
WITH_RAW = {'sort': 'phy', 'events': 'events_csv', 'raw': 'binary'}
# The brief's values, written out rather than imported from the code under test.
OPTIONS_LITERAL = {'before': 41, 'after': 41, 'max_spikes': 100, 'min_spikes': 20, 'seed': 0}
TOO_FEW = 'Too few spikes in the analysis window.'
NO_TEMPLATES = 'Day D1 has no templates; the templates tier needs a sort with templates.'
ALREADY = 'Day D1 is already extracted; features are written once. Put it in a new project to extract it again.'
ANALYSIS_WINDOW = (0, 597000)  # [0, stop of the last trial) in the synthetic events.csv
TABLE_COLUMNS = ['cellid', 'session_id', 'siteid', 'well', 'recording_group', 'session_date',
                 'su_type', 'eligible', 'exclusion_reason']


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _project(tmp_path, *, raw: bool, **day_kwargs):
    """``(sd, project)``: the synthetic day D1 added to project P, with the ``binary`` reader only when ``raw``."""
    sd = write_day(tmp_path / 'in', 'D1', **day_kwargs)
    project = Project.create(tmp_path / 'lab', 'P', group='G')
    options = {'day': 'D1', 'sort_dir': str(sd.sort_dir), 'events_path': str(sd.events_path),
               'raw_path': str(sd.raw_path), 'uv_per_bit': 0.195}
    add_day(project, 'D1', readers=dict(WITH_RAW if raw else NO_RAW), reader_options=options,
            date=sd.date, date_source='user')
    return sd, project


def _extract(project, tmp_path, **kwargs) -> dict:
    return extract_day(project, 'D1', scratch_root=tmp_path / 'scratch', **kwargs)


def _no_voltage(monkeypatch) -> None:
    """Opening the raw pass or building a halves archive fails the test: the templates tier reads no voltage."""
    def refuse(*args, **kwargs):
        raise AssertionError('the templates tier read voltage')
    monkeypatch.setattr(pipeline.extract_benchmark, 'open_recording_sources', refuse)
    monkeypatch.setattr(pipeline, 'build_dataset', refuse)
    monkeypatch.setattr(pipeline, 'save_dataset', refuse)


def _listing(folder: Path) -> list:
    return sorted(path.name for path in folder.iterdir())


def _tree(root: Path) -> dict:
    """Relative path -> bytes of every file under ``root``."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def _log_steps(features: Path) -> list:
    """The steps of ``features/extract.log``, each line checked to be ``'<UTC ISO time> · <step>'``."""
    steps = []
    for line in (features / 'extract.log').read_text(encoding='utf-8').splitlines():
        stamp, sep, step = line.partition(' · ')
        assert sep and step
        assert datetime.datetime.fromisoformat(stamp).utcoffset() == datetime.timedelta(0)
        steps.append(step)
    return steps


def _inside(spikes: np.ndarray) -> np.ndarray:
    start, stop = ANALYSIS_WINDOW
    return spikes[(spikes >= start) & (spikes < stop)]


def test_templates_tier_writes_no_halves(tmp_path, monkeypatch):
    sd, project = _project(tmp_path, raw=False)
    files_before = project.read()['days']['D1']['files']
    _no_voltage(monkeypatch)
    result = _extract(project, tmp_path)

    day_dir = project.day_dir('D1')
    features = day_dir / 'features'
    full_id, source_id = _sha256(features / 'full' / 'manifest.json'), _sha256(day_dir / 'source' / 'extraction_manifest.json')
    assert result == {'day': 'D1', 'tier': 'templates', 'source_id': source_id,
                      'features': {'halves': None, 'full': full_id}}
    written = project.read()
    record = written['days']['D1']
    assert record['tier'] == 'templates'
    assert record['features']['halves'] is None
    assert record['features'] == {'halves': None, 'full': full_id}
    assert record['source_id'] == source_id
    assert record['raw_params'] is None  # a day without a raw file has no raw parameters
    assert record['files'] == files_before
    assert written['identity']['geometry_sha256'] is not None

    # Only full/ and the log: no halves archive.
    assert not (features / 'halves').exists()
    assert _listing(features) == ['extract.log', 'full']
    assert _listing(day_dir) == ['features', 'source']

    full = load_full_day(features / 'full')
    assert full['metadata'] == {'project': 'P', 'day': 'D1', 'tier': 'templates', 'halves_id': None,
                                'source_id': source_id, 'halves_rule': None, 'halves_boundary': None,
                                'extraction_options': OPTIONS_LITERAL}
    table = full['table']
    assert table.columns.tolist() == TABLE_COLUMNS
    assert table['cellid'].tolist() == sd.units
    assert table['eligible'].tolist() == [True, True, True]
    assert table['exclusion_reason'].tolist() == ['', '', '']
    assert set(table['session_id']) == set(table['siteid']) == {'D1'}
    assert set(table['well']) == set(table['recording_group']) == {'G'}
    assert set(table['session_date']) == {sd.date}
    assert set(table['su_type']) == {'SU'}
    np.testing.assert_array_equal(full['geometry']['D1'], sd.geometry)

    assert sorted(full['features']) == sd.units
    for unit in sd.units:
        feature = full['features'][unit]
        assert feature['tier'] == 'templates'
        assert 'waveform_halves' not in feature
        assert feature['eligible'] is True and feature['su_type'] == 'SU'
        assert (feature['start_sample'], feature['stop_sample']) == ANALYSIS_WINDOW
        assert feature['sample_rate'] == 30000.0
        assert feature['fingerprint_bin_seconds'] == 0.05
        np.testing.assert_array_equal(feature['spike_samples'], _inside(sd.spikes[unit]))
        assert feature['n_spikes'] == len(_inside(sd.spikes[unit]))
        assert feature['trial_counts'] == {'A': 5, 'B': 5}
        assert 'template' in feature['preprocessing']

    # METHODS.md names the published folder and describes templates, not pooled halves.
    methods = (features / 'full' / 'METHODS.md').read_text(encoding='utf-8')
    assert f'- **Output folder:** `{features / "full"}`' in methods.splitlines()
    assert 'template' in methods and 'pooled from temporal halves' not in methods

    steps = _log_steps(features)
    assert steps[1:4] == ['day D1 recorded, not yet extracted', 'tier templates', f'verified {len(files_before)} files']
    assert f'full {full_id}' in steps and not any(step.startswith('halves') for step in steps)
    assert steps[-2:] == ['published features/', 'recording in project.json; project.json says whether it was recorded']

    # A unit with fewer than min_spikes spikes in the analysis window gets an ineligible row and no features.
    source = read_source(day_dir / 'source')
    n_first = len(_inside(sd.spikes['D1:0']))
    kept = template_full_rows(source, day='D1', group='G', session_date=sd.date, min_spikes=n_first)
    assert kept['table']['eligible'].tolist()[0] is True and 'D1:0' in kept['features']
    dropped = template_full_rows(source, day='D1', group='G', session_date=sd.date, min_spikes=n_first + 1)
    assert dropped['table'].set_index('cellid').loc['D1:0', 'exclusion_reason'] == TOO_FEW
    assert not dropped['table'].set_index('cellid').loc['D1:0', 'eligible']
    assert 'D1:0' not in dropped['features']
    none = template_full_rows(source, day='D1', group='G', session_date=sd.date, min_spikes=201)
    assert none['table']['eligible'].tolist() == [False, False, False]
    assert none['table']['exclusion_reason'].tolist() == [TOO_FEW] * 3
    assert none['features'] == {}


def test_forced_templates_tier(tmp_path, monkeypatch):
    sd, project = _project(tmp_path, raw=True)
    _no_voltage(monkeypatch)
    result = _extract(project, tmp_path, tier='templates')

    features = project.day_dir('D1') / 'features'
    assert not (features / 'halves').exists()
    assert _listing(features) == ['extract.log', 'full']
    assert result['tier'] == 'templates' and result['features']['halves'] is None
    record = project.read()['days']['D1']
    assert record['tier'] == 'templates'
    assert record['features'] == {'halves': None, 'full': _sha256(features / 'full' / 'manifest.json')}
    # The raw file's parameters, as the normalized source holds them; its bytes were never read.
    assert record['raw_params'] == {'sample_rate': 30000.0, 'n_channels': 8, 'dtype': '<i2',
                                    'uv_per_bit': 0.195, 'gzip': False}
    assert record['files']['raw']['sha256'] is None
    full = load_full_day(features / 'full')
    assert full['metadata']['tier'] == 'templates' and full['metadata']['halves_id'] is None
    assert full['table']['eligible'].tolist() == [True, True, True]
    assert 'tier templates' in _log_steps(features)


def test_template_waveform_unwhitened(tmp_path):
    sd, project = _project(tmp_path, raw=False)
    _extract(project, tmp_path)
    full = load_full_day(project.day_dir('D1') / 'features' / 'full')

    assert np.allclose(full['features']['D1:0']['waveform'], sd.templates[0], atol=1e-4) is True
    # The sort stores whitened templates; only the unwhitened one matches the ground truth.
    assert not np.allclose(np.load(sd.sort_dir / 'templates.npy')[0], sd.templates[0], atol=1e-4)
    for unit in sd.units:
        waveform = full['features'][unit]['waveform']
        assert waveform.dtype == np.float32
        np.testing.assert_allclose(waveform, sd.templates[int(unit.split(':')[1])], atol=1e-4)


def test_no_templates_refused(tmp_path):
    sd, project = _project(tmp_path, raw=False, with_templates=False)
    before = (project.path / 'project.json').read_bytes()

    with pytest.raises(ValueError, match='has no templates') as raised:
        _extract(project, tmp_path)
    assert str(raised.value) == NO_TEMPLATES
    # Verified and normalized as the raw tier does, then refused: no features, nothing recorded, lock released.
    assert _listing(project.day_dir('D1')) == ['source']
    assert read_source(project.day_dir('D1') / 'source')['templates'] is None
    assert (project.path / 'project.json').read_bytes() == before


def test_raw_tier_without_raw_refused(tmp_path):
    sd, project = _project(tmp_path, raw=False)
    before = (project.path / 'project.json').read_bytes()

    with pytest.raises(ValueError, match='has no raw file') as raised:
        _extract(project, tmp_path, tier='raw')
    assert str(raised.value) == 'Day D1 has no raw file.'
    with pytest.raises(ValueError, match='Unknown tier bogus.'):
        _extract(project, tmp_path, tier='bogus')
    # The tier is checked before anything is written, and the lock is released.
    assert _listing(project.day_dir('D1')) == []
    assert (project.path / 'project.json').read_bytes() == before


def test_raw_after_templates_refused(tmp_path):
    sd, project = _project(tmp_path, raw=True)
    _extract(project, tmp_path, tier='templates')
    day_dir = project.day_dir('D1')
    tree, before = _tree(day_dir), (project.path / 'project.json').read_bytes()

    with pytest.raises(FileExistsError, match='already extracted') as raised:
        _extract(project, tmp_path, tier='raw')
    assert str(raised.value) == ALREADY
    with pytest.raises(FileExistsError, match='already extracted'):
        _extract(project, tmp_path)
    assert _tree(day_dir) == tree
    assert (project.path / 'project.json').read_bytes() == before
    assert project.read()['days']['D1']['tier'] == 'templates'
