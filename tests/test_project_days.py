# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: adding a day records its file identities, and source/ holds the normalized reader output (multiphy project folder Task 8; Task 10, D11 written once: a day is added once, so the replace and archive tests go, as do the source lock, aside-folder and pinned-read scenarios; write_source replaces a source/ a stopped extraction left) | date: 2026-09-25]
"""Adding a day records its files; ``source/`` holds the normalized reader output and round-trips.

A day is added once: a second ``add_day`` for it is refused and changes nothing.
``write_source`` builds its stage first; only a complete stage replaces a ``source/`` a stopped
extraction left, and a failure deletes the stage.
"""
from __future__ import annotations

import copy
import dataclasses
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import stat

import numpy as np
import pytest

from multiphy.project.days import add_day
from multiphy.project.folder import Project
from multiphy.project.identity import FileIdentity, sha256_file
from multiphy.sources.base import clear_readers_for_tests, read_day, register_reader
from multiphy.sources import normalize
from multiphy.sources.normalize import geometry_sha256, read_source, source_matches, write_source

from .synthetic_day import write_day, write_legacy_inputs

READERS = {'sort': 'phy', 'events': 'events_csv', 'raw': 'binary'}
SORT_FILES = {'params': 'params.py', 'spike_times': 'spike_times.npy', 'spike_clusters': 'spike_clusters.npy',
              'channel_positions': 'channel_positions.npy', 'cluster_labels': 'cluster_group.tsv',
              'templates': 'templates.npy', 'whitening_mat_inv': 'whitening_mat_inv.npy',
              'spike_templates': 'spike_templates.npy'}
FEATURES = {'halves/manifest.json': b'{}', 'halves/waveforms.npy': b'halves payload',
            'full/manifest.json': b'{"full": 1}', 'full/rows.npz': b'full payload'}
# The brief's refusal, written out here rather than imported from the code under test.
ADDED_ONCE = 'Day D1 already exists; a day is added once. Put a changed day in a new project.'
# The brief's preprocessing text, written out here rather than imported from the code under test.
REQUIRED_PREPROCESSING = ('Acquired samples scaled by uv_per_bit; shared_unwhitened_v1 profile at its defaults: '
                          'high-pass, median reference, unwhitened; native spike timestamps.')
# SharedWaveformPreprocessor(sample_rate, n_channels).metadata() at its signature defaults (highpass_hz=150.,
# filter_order=3, padding_seconds=.05, reference_channels=None: every channel) for the synthetic day's 8 channels.
DEFAULT_WAVEFORM_METADATA = {
    'profile': 'shared_unwhitened_v1', 'highpass_hz': 150.0, 'filter_order': 3, 'padding_seconds': 0.05,
    'reference_channels': [0, 1, 2, 3, 4, 5, 6, 7],
    'filter': 'Butterworth SOS forward/backward with odd endpoint extension',
    'reference': 'samplewise median after filtering', 'whitening': False, 'baseline_subtraction': False,
    'amplitude_normalization': False, 'alignment': 'native exported spike timestamps; no per-spike realignment'}


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


@pytest.fixture
def sd(tmp_path):
    return write_day(tmp_path / 'in', 'D1')


@pytest.fixture
def project(tmp_path):
    return Project.create(tmp_path / 'lab', 'P')


@pytest.fixture
def options(sd):
    return {'day': 'D1', 'sort_dir': str(sd.sort_dir), 'events_path': str(sd.events_path),
            'raw_path': str(sd.raw_path), 'uv_per_bit': 0.195}


def _add(project, sd, options, **kwargs):
    kwargs.setdefault('date', sd.date)
    return add_day(project, 'D1', readers=dict(READERS), reader_options=dict(options), date_source='user', **kwargs)


def _day_source(record):
    """The day as extraction reads it: the recorded files only."""
    files: dict = {}
    for key, identity in record['files'].items():
        files.setdefault(identity['role'], {})[key] = Path(identity['resolved'])
    return read_day(record['readers'], files, record['reader_options'])


def _write(project, record, day_source, target):
    identities = {key: FileIdentity.from_dict(identity) for key, identity in record['files'].items()}
    return write_source(day_source, target, day='D1', project=project.name, readers=record['readers'],
                        identities=identities)


def _normalize(project, record, target):
    """Read the day as extraction will and write it to ``target``; return the source_id."""
    return _write(project, record, _day_source(record), target)


def _tree(root: Path) -> dict:
    """Relative path -> bytes of every file under ``root``."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def _put(root: Path, files: dict) -> None:
    for relative, data in files.items():
        (root / relative).parent.mkdir(parents=True, exist_ok=True)
        (root / relative).write_bytes(data)


def _listing(folder: Path) -> list:
    return sorted(path.name for path in folder.iterdir())


class _FileFreeReader:
    """Serves sort and events and names no file; counts how often it is asked for its files."""
    id = 'file_free'
    version = 1
    roles = frozenset({'sort', 'events'})

    def __init__(self):
        self.calls = 0

    def files(self, options):
        self.calls += 1
        return {}

    def read(self, files, options, day=None):
        raise AssertionError('add_day reads no data')


def test_add_day_records_identities(project, sd, options, tmp_path):
    record = _add(project, sd, options)
    files = record['files']
    assert files['raw']['sha256'] is None
    assert files['params']['sha256'] == sha256_file(sd.sort_dir / 'params.py')
    assert record['readers']['sort'] == 'phy@1'

    assert record['readers'] == {'sort': 'phy@1', 'events': 'events_csv@1', 'raw': 'binary@1'}
    expected = {key: sd.sort_dir / name for key, name in SORT_FILES.items()}
    expected['events'] = sd.events_path
    assert set(files) == set(expected) | {'raw'}
    for key, path in expected.items():  # every file but the raw voltage is hashed
        assert files[key]['resolved'] == str(path)
        assert files[key]['sha256'] == sha256_file(path) and files[key]['sha256_from'] == 'init'
        assert files[key]['size_bytes'] == path.stat().st_size and files[key]['mtime_ns'] == path.stat().st_mtime_ns
        assert files[key]['role'] == ('events' if key == 'events' else 'sort')
    assert files['raw']['role'] == 'raw' and files['raw']['resolved'] == str(sd.raw_path)
    assert files['raw']['sha256_from'] is None and files['raw']['size_bytes'] == sd.raw_path.stat().st_size
    assert files['raw']['mtime_ns'] == sd.raw_path.stat().st_mtime_ns
    assert record['features'] == {'halves': None, 'full': None} and record['tier'] is None
    assert project.read()['days']['D1'] == record
    assert project.day_dir('D1').is_dir()

    # A reader serving two roles is asked for its files once, even when it names none.
    reader = _FileFreeReader()
    register_reader(reader)
    other = Project.create(tmp_path / 'file-free', 'P')
    free = add_day(other, 'D1', readers={'sort': 'file_free', 'events': 'file_free'}, reader_options={},
                   date=sd.date, date_source='user')
    assert reader.calls == 1
    assert free['files'] == {} and free['readers'] == {'sort': 'file_free@1', 'events': 'file_free@1'}


def test_add_day_refuses_existing(project, sd, options, tmp_path):
    first = _add(project, sd, options)
    day_dir = project.day_dir('D1')
    _put(day_dir / 'features', FEATURES)
    before = (project.path / 'project.json').read_bytes()
    with pytest.raises(ValueError, match='already exists') as raised:
        _add(project, sd, options)
    assert str(raised.value) == ADDED_ONCE
    assert sd.date != '2001-02-03'
    with pytest.raises(ValueError, match='already exists'):
        _add(project, sd, options, date='2001-02-03')  # another valid date: the day name alone is refused
    # Replacing a day is gone: the parameter no longer exists.
    with pytest.raises(TypeError, match='replace_day'):
        _add(project, sd, options, replace_day=True)
    # The day is refused before its readers are asked for files: an input gone missing, or a reader
    # that does not exist, still gets the write-once refusal.
    sd.events_path.unlink()
    with pytest.raises(ValueError, match='already exists') as raised:
        _add(project, sd, options)
    assert str(raised.value) == ADDED_ONCE
    with pytest.raises(ValueError, match='already exists'):
        add_day(project, 'D1', readers={'sort': 'no_such_reader'}, reader_options={}, date=sd.date,
                date_source='user')
    assert (project.path / 'project.json').read_bytes() == before
    assert project.read()['days'] == {'D1': first}
    assert _tree(day_dir / 'features') == FEATURES
    assert _listing(day_dir) == ['features']
    assert not (project.path / 'project.lock').exists()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: final review fix round 1: a day folder that cannot be made leaves no day in project.json, so a rerun adds the day | date: 2026-09-25]
def test_day_folder_made_before_the_record(project, sd, options, monkeypatch):
    """The day folder is made before ``project.json`` records the day: a failed ``mkdir`` records nothing."""
    day_dir = project.day_dir('D1')
    before = (project.path / 'project.json').read_bytes()
    real_mkdir = Path.mkdir

    def mkdir(self, *args, **kwargs):
        if self == day_dir:
            raise PermissionError(errno.EACCES, 'Permission denied', str(self))
        return real_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, 'mkdir', mkdir)
    with pytest.raises(PermissionError):
        _add(project, sd, options)
    assert (project.path / 'project.json').read_bytes() == before
    assert not day_dir.exists()
    assert not (project.path / 'project.lock').exists()

    monkeypatch.undo()  # the cause is gone: the same call now adds the day
    record = _add(project, sd, options)
    assert project.read()['days'] == {'D1': record}
    assert day_dir.is_dir()
# [AGENT EDIT END]


def test_write_source_roundtrip(project, sd, options, tmp_path):
    record = _add(project, sd, options)
    target = tmp_path / 'out' / 'source'
    target.parent.mkdir()
    os.chmod(target.parent, 0o770)  # group-writable, unlike the umask default: source/ must get these bits
    day_source = _day_source(record)
    earlier = dataclasses.replace(day_source, templates=None,
                                  spikes={unit: samples + 1 for unit, samples in day_source.spikes.items()})
    earlier_id = _write(project, record, earlier, target)
    first = read_source(target)
    assert first['templates'] is None and not (target / 'templates.npz').exists()
    assert all(np.array_equal(first['spikes'][unit], sd.spikes[unit] + 1) for unit in sd.units)

    source_id = _write(project, record, day_source, target)  # the second write replaces the first
    source = read_source(target)
    assert source['spikes'].keys() == sd.spikes.keys()
    for unit, samples in sd.spikes.items():
        assert np.array_equal(source['spikes'][unit], samples)
    entry = source['entry']
    assert 'partition' not in entry and 'well' not in entry and 'windows' not in entry

    assert source_id != earlier_id and 'templates_file' not in first['entry']
    assert _listing(target.parent) == ['source']  # no stage or old copy left, and no lock file
    assert stat.S_IMODE(target.stat().st_mode) == 0o770  # the parent's bits, not mkdtemp's 0700
    manifest_bytes = (target / 'extraction_manifest.json').read_bytes()
    assert source_id == source['source_id'] == hashlib.sha256(manifest_bytes).hexdigest()
    assert manifest_bytes.decode('utf-8').endswith('}\n')
    assert all(source['spikes'][unit].dtype == np.int64 for unit in sd.units)
    assert np.array_equal(source['geometry'], sd.geometry)
    assert sorted(source['templates']) == sd.units
    for unit in sd.units:
        np.testing.assert_allclose(source['templates'][unit], sd.templates[int(unit.split(':')[1])], atol=1e-3)
    assert entry['source_session'] == entry['day'] == 'D1' and entry['project'] == 'P'
    assert entry['readers'] == record['readers']
    assert entry['raw_voltage'] == str(sd.raw_path)
    assert (entry['dtype'], entry['voltage_scale']) == ('<i2', 0.195)
    assert (entry['spikes_file'], entry['geometry_file'], entry['templates_file']) == (
        'spikes.npz', 'geometry.npy', 'templates.npz')
    assert entry['single_unit_ids'] == sd.units
    assert (entry['sample_rate'], entry['n_samples'], entry['analysis_window']) == (30000.0, 600000, [0, 597000])
    assert entry['preprocessing'] == REQUIRED_PREPROCESSING == normalize.PREPROCESSING
    assert entry['waveform_preprocessing'] == DEFAULT_WAVEFORM_METADATA
    assert entry['clock_verification']['sample_index_origin'] == 'zero_based_voltage_row'
    assert len(entry['trials']) == 10 and entry['trials'][0] == [0, 57000]
    assert entry['stimuli'] == sd.stimuli
    assert entry['baseline_windows'] == sd.baselines.tolist()
    assert entry['provenance']['reader'] == 'phy@1'
    assert entry['provenance']['files'] == record['files']
    methods = (target / 'METHODS.md').read_text(encoding='utf-8')
    assert REQUIRED_PREPROCESSING in methods and 'phy@1' in methods and 'binary@1' in methods
    assert 'params (sort)' in methods


def test_write_source_replaces_leftover(project, sd, options, tmp_path, monkeypatch):
    record = _add(project, sd, options)
    target = tmp_path / 'out' / 'source'
    _put(target, {'stray.txt': b'left by a stopped extraction'})

    source_id = _normalize(project, record, target)
    assert not (target / 'stray.txt').exists()
    source = read_source(target)
    assert source['spikes'].keys() == sd.spikes.keys()
    for unit, samples in sd.spikes.items():
        assert np.array_equal(source['spikes'][unit], samples)
    assert not any(path.name.startswith('.source-') for path in target.parent.iterdir())
    assert _listing(target.parent) == ['source']
    assert source['source_id'] == source_id
    written = _tree(target)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every new source holds extras.npz (Plan C Task 5 fix round 1) | date: 2026-10-01]
    assert sorted(written) == ['METHODS.md', 'extraction_manifest.json', 'extras.npz', 'geometry.npy', 'spikes.npz',
                               'templates.npz']
    # [AGENT EDIT END]

    # The stage is complete before source/ is touched: a failure while staging leaves source/ as it was.
    def disk_full(*args, **kwargs):
        raise OSError(errno.ENOSPC, 'No space left on device')
    earlier = dataclasses.replace(_day_source(record), spikes={unit: samples + 1 for unit, samples in sd.spikes.items()})
    with monkeypatch.context() as m:
        m.setattr(np, 'savez_compressed', disk_full)
        with pytest.raises(OSError, match='No space left'):
            _write(project, record, earlier, target)
    assert _tree(target) == written
    assert _listing(target.parent) == ['source']

    # A manifest that cannot be written (NaN) is refused before any stage is made.
    invalid = dataclasses.replace(earlier, clock=dict(earlier.clock, drift=float('nan')))
    with pytest.raises(ValueError, match='JSON compliant'):
        _write(project, record, invalid, target)
    assert _tree(target) == written
    assert _listing(target.parent) == ['source']

    # A failed rename after the old source/ is deleted: the stage is deleted too and the error re-raised.
    real_rename = os.rename

    def refuse_rename(source, destination):
        if Path(destination) == target:
            raise OSError(errno.EIO, 'Input/output error')
        return real_rename(source, destination)
    with monkeypatch.context() as m:
        m.setattr(os, 'rename', refuse_rename)
        with pytest.raises(OSError, match='Input/output error'):
            _write(project, record, earlier, target)
    assert _listing(target.parent) == []

    # The next write puts a whole source in place.
    assert _normalize(project, record, target) == source_id
    assert _tree(target) == written
    assert _listing(target.parent) == ['source']


def test_excluded_tail_counts(project, sd, options, tmp_path):
    record = _add(project, sd, options)
    _normalize(project, record, tmp_path / 'source')
    entry = read_source(tmp_path / 'source')['entry']
    assert entry['excluded_tail_spike_counts']['D1:0'] == int((sd.spikes['D1:0'] >= 597000).sum())

    expected = {unit: int((sd.spikes[unit] >= 597000).sum()) for unit in sd.units}
    assert entry['excluded_tail_spike_counts'] == expected
    assert sum(expected.values()) > 0  # the seed puts some spikes in the tail, so the count is not trivially zero

    # At the stop exactly: a spike at the stop is excluded, one sample before it is not.
    assert entry['analysis_window'][1] == 597000
    day_source = _day_source(record)
    boundary = dataclasses.replace(day_source, spikes={**day_source.spikes,
                                                       'D1:1': np.array([596999, 597000, 597001], dtype=np.int64)})
    _write(project, record, boundary, tmp_path / 'boundary')
    counts = read_source(tmp_path / 'boundary')['entry']['excluded_tail_spike_counts']
    assert counts['D1:1'] == 2

    # Another stop: the count follows the analysis window.
    early = dataclasses.replace(day_source, analysis_window=(0, 500000))
    _write(project, record, early, tmp_path / 'early')
    counts = read_source(tmp_path / 'early')['entry']['excluded_tail_spike_counts']
    assert counts == {unit: int((sd.spikes[unit] >= 500000).sum()) for unit in sd.units} != expected


def test_read_source_schema1(sd, tmp_path):
    legacy = tmp_path / 'legacy'
    write_legacy_inputs(sd, legacy, dataset_id='legacy', well='G')
    source = read_source(legacy)
    entry = source['entry']
    assert 'partition' not in entry
    assert 'well' not in entry

    original = json.loads((legacy / 'extraction_manifest.json').read_text(encoding='utf-8'))['recordings'][0]
    assert original['partition'] == 'test' and original['well'] == 'G'
    assert entry == {key: value for key, value in original.items() if key not in ('partition', 'well')}
    assert source['spikes'].keys() == sd.spikes.keys()
    for unit, samples in sd.spikes.items():
        assert np.array_equal(source['spikes'][unit], samples)
    assert np.array_equal(source['geometry'], sd.geometry)
    assert source['templates'] is None
    assert source['source_id'] == hashlib.sha256((legacy / 'extraction_manifest.json').read_bytes()).hexdigest()

    # The files are opened by the names the manifest gives: another relative name, and an absolute path.
    renamed = tmp_path / 'renamed'
    write_legacy_inputs(sd, renamed, dataset_id='legacy', well='G')
    elsewhere = tmp_path / 'elsewhere'
    elsewhere.mkdir()
    (renamed / 'spikes.npz').rename(renamed / 'units.npz')
    (renamed / 'geometry.npy').rename(elsewhere / 'probe.npy')
    manifest = json.loads((renamed / 'extraction_manifest.json').read_text(encoding='utf-8'))
    manifest['recordings'][0].update(spikes_file='units.npz', geometry_file=str(elsewhere / 'probe.npy'))
    (renamed / 'extraction_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    moved = read_source(renamed)
    assert (moved['entry']['spikes_file'], moved['entry']['geometry_file']) == ('units.npz', str(elsewhere / 'probe.npy'))
    assert all(np.array_equal(moved['spikes'][unit], sd.spikes[unit]) for unit in sd.units)
    assert np.array_equal(moved['geometry'], sd.geometry)


def test_source_matches(project, sd, options, tmp_path):
    record = _add(project, sd, options)
    target = tmp_path / 'source'
    _normalize(project, record, target)
    assert source_matches(target, record['files']) is True
    changed = copy.deepcopy(record['files'])
    changed['events']['mtime_ns'] += 1
    assert source_matches(target, changed) is False

    for field, value in (('resolved', str(sd.events_path) + '.moved'), ('size_bytes', 1), ('mtime_ns', 1)):
        changed = copy.deepcopy(record['files'])
        changed['events'][field] = value  # each matched field on its own
        assert source_matches(target, changed) is False, field
    for key in ('params', 'raw'):  # other files than events, each matched field on its own
        for field in ('resolved', 'size_bytes', 'mtime_ns'):
            changed = copy.deepcopy(record['files'])
            changed[key][field] = changed[key][field] + ('.moved' if field == 'resolved' else 1)
            assert source_matches(target, changed) is False, (key, field)
    changed = copy.deepcopy(record['files'])
    changed['events']['sha256'] = '0' * 64  # not a matched field
    assert source_matches(target, changed) is True
    assert source_matches(tmp_path / 'nowhere', record['files']) is False
    fewer = {key: value for key, value in record['files'].items() if key != 'events'}
    assert source_matches(target, fewer) is False
    identities = {key: FileIdentity.from_dict(value) for key, value in record['files'].items()}
    assert source_matches(target, identities) is True


def test_geometry_sha256_stable(project, sd, options, tmp_path):
    first = geometry_sha256(sd.geometry)
    second = geometry_sha256(sd.geometry.copy())
    assert first == second
    record = _add(project, sd, options)
    _normalize(project, record, tmp_path / 'source')
    written = hashlib.sha256((tmp_path / 'source' / 'geometry.npy').read_bytes()).hexdigest()
    assert first == written and second == written

    assert geometry_sha256(read_source(tmp_path / 'source')['geometry']) == first
    assert geometry_sha256(sd.geometry.astype(np.int64)) == first  # hashed as float

    # Fractional positions, hashed here without the module's helper: float64 .npy bytes.
    fractional = sd.geometry + np.array([0.125, 0.3])
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(fractional, dtype=float))
    expected = hashlib.sha256(buffer.getvalue()).hexdigest()
    assert geometry_sha256(fractional) == geometry_sha256(fractional.tolist()) == expected
    _write(project, record, dataclasses.replace(_day_source(record), geometry=fractional), tmp_path / 'fractional')
    assert hashlib.sha256((tmp_path / 'fractional' / 'geometry.npy').read_bytes()).hexdigest() == expected
    assert np.array_equal(read_source(tmp_path / 'fractional')['geometry'], fractional)

    # More channels than the synthetic day's eight: every row is hashed.
    wide = np.column_stack([np.tile([0.0, 32.0], 12), np.repeat(np.arange(12) * 20.0, 2)])
    assert wide.shape == (24, 2)
    buffer = io.BytesIO()
    np.save(buffer, wide.astype(float))
    assert geometry_sha256(wide) == hashlib.sha256(buffer.getvalue()).hexdigest() != geometry_sha256(wide[:8])
# [AGENT EDIT END]
