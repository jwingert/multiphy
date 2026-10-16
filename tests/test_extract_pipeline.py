"""``extract_day`` for the raw tier: one pipeline writes each day's halves and full day, once.

The halves and full-day features equal the old path's (``extract_manifest`` plus
``reuse_day_summary``) for every unit and half. Extraction refuses a held lock, a changed
file, a foreign geometry and a day already extracted. A failure before the stage is published
leaves nothing; ``features/`` a stopped extraction left unrecorded is replaced by the next one.
"""
from __future__ import annotations

import datetime
import errno
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import stat

import numpy as np
import pandas as pd
import pytest

from multiphy.benchmark import load_dataset
from multiphy.extract import pipeline
from multiphy.extract.full_day import load_full_day
from multiphy.extract.pipeline import EXTRACTION_OPTIONS, ExtractLocked, extract_day
from multiphy.extract_benchmark import extract_manifest
from multiphy.multiday_reuse import reuse_day_summary
from multiphy.project.days import add_day
from multiphy.project.folder import Project, ProjectLocked
from multiphy.project.identity import SourceChanged, SourceMissing
from multiphy.sources.base import clear_readers_for_tests

from .full_day_oracle import assert_pooled_from_halves
from .lab_blocker import run_without_lab
from .synthetic_day import OPTIONS, write_day, write_legacy_inputs

REPO = Path(__file__).resolve().parents[1]
READERS = {'sort': 'phy', 'events': 'events_csv', 'raw': 'binary'}
# The brief's values, written out rather than imported from the code under test.
OPTIONS_LITERAL = {'before': 41, 'after': 41, 'max_spikes': 100, 'min_spikes': 20, 'seed': 0}
WEAK_REASON = 'Too few spikes in temporal split.'
ALREADY = 'Day D1 is already extracted; features are written once. Put it in a new project to extract it again.'
# The last two steps of extract.log, both written in the stage before its rename: the rename to
# features/, then the record in project.json that follows it. project.json alone says whether that record was written.
PUBLISHED = 'published features/'
PUBLISHED_OVER_LEFTOVER = 'published features/ in place of an unrecorded features/'
RECORDING = 'recording in project.json; project.json says whether it was recorded'


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _options(sd, **changes) -> dict:
    options = {'day': 'D1', 'sort_dir': str(sd.sort_dir), 'events_path': str(sd.events_path),
               'raw_path': str(sd.raw_path), 'uv_per_bit': 0.195}
    options.update(changes)
    return options


def _project(tmp_path, sd, lab='lab') -> Project:
    project = Project.create(tmp_path / lab, 'P', group='G')
    add_day(project, 'D1', readers=dict(READERS), reader_options=_options(sd), date=sd.date, date_source='user')
    return project


def _extract(project, tmp_path, **kwargs) -> dict:
    return extract_day(project, 'D1', scratch_root=tmp_path / 'scratch', **kwargs)


def _listing(folder: Path) -> list:
    return sorted(path.name for path in folder.iterdir())


def _stages(day_dir: Path) -> list:
    return [name for name in _listing(day_dir) if name.startswith('.features-')]


def _tree(root: Path) -> dict:
    """Relative path -> bytes of every file under ``root``."""
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in sorted(root.rglob('*')) if path.is_file()}


def _project_json(project) -> bytes:
    return (project.path / 'project.json').read_bytes()


def _old_path(sd, tmp_path):
    """The old extraction path on the same day: the schema 1 halves archive and its manifest."""
    manifest = write_legacy_inputs(sd, tmp_path / 'legacy', dataset_id='legacy', well='G')
    archive = extract_manifest(manifest, root=tmp_path / 'cache', scratch_root=tmp_path / 'scratch')
    return manifest, archive, json.loads(manifest.read_text())['recordings'][0]


def _halves_by_unit(dataset) -> dict:
    """(source_unit_id, split_index) -> the halves feature dict without ``preprocessing``."""
    rows = {}
    for label in dataset.labels.itertuples():
        feature = dict(dataset.features[label.cellid])
        feature.pop('preprocessing')
        key = (label.source_unit_id, int(feature['split_index']))
        assert key not in rows
        rows[key] = feature
    return rows


def _without_preprocessing(feature: dict) -> dict:
    feature = dict(feature)
    feature.pop('preprocessing')
    return feature


def _geometry_sha256(geometry) -> str:
    """The sha256 of the ``.npy`` bytes of ``geometry`` as float, computed here independently."""
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(geometry, dtype=float))
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def _unrecorded(project) -> bool:
    return project.read()['days']['D1']['features'] == {'halves': None, 'full': None}


def _fail_log_writes(patch, when) -> None:
    """Opening an ``extract.log`` to write raises ``OSError`` (EIO) while ``when()`` is true.

    ``Path.write_text`` opens through ``Path.open``, so both are covered.
    """
    real_open = Path.open

    def open_for_write(self, mode='r', *args, **kwargs):
        if self.name == 'extract.log' and set(mode) & set('wax+') and when():
            raise OSError(errno.EIO, 'Input/output error')
        return real_open(self, mode, *args, **kwargs)
    patch.setattr(Path, 'open', open_for_write)


def _log_steps(features: Path) -> list:
    """The steps of ``features/extract.log``, each line checked to be ``'<UTC ISO time> · <step>'``."""
    steps = []
    for line in (features / 'extract.log').read_text(encoding='utf-8').splitlines():
        stamp, sep, step = line.partition(' · ')
        assert sep and step
        assert datetime.datetime.fromisoformat(stamp).utcoffset() == datetime.timedelta(0)
        steps.append(step)
    return steps


def test_extract_day_matches_old_path(tmp_path):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)
    result = _extract(project, tmp_path)

    day_dir = project.day_dir('D1')
    features = day_dir / 'features'
    new_halves = load_dataset(features / 'halves')
    new_full = load_full_day(features / 'full')

    manifest, archive, spec = _old_path(sd, tmp_path)
    old_halves = load_dataset(archive)
    summary = reuse_day_summary(archive, manifest, spec, session_date=sd.date,
                                analysis_window=tuple(spec['analysis_window']), extraction_options=OPTIONS)

    # The halves: one feature dict per (source unit, half), equal but for the preprocessing text.
    old_rows, new_rows = _halves_by_unit(old_halves), _halves_by_unit(new_halves)
    assert sorted(new_rows) == sorted(old_rows) == [(unit, split) for unit in sd.units for split in (0, 1)]
    np.testing.assert_equal(new_rows, old_rows)
    assert new_halves.exclusions == old_halves.exclusions

    # The full day: one feature dict per unit; the old id is 'D1-' + unit, the new id is the unit.
    assert sorted(new_full['features']) == sd.units
    assert sorted(summary['features']) == ['D1-' + unit for unit in sd.units]
    for unit in sd.units:
        np.testing.assert_equal(_without_preprocessing(new_full['features'][unit]),
                                _without_preprocessing(summary['features']['D1-' + unit]))
        cellids = old_halves.labels.loc[old_halves.labels.source_unit_id == unit, 'cellid']
        assert_pooled_from_halves(new_full['features'][unit], [old_halves.features[c] for c in cellids],
                                  sd.spikes[unit], spec, tuple(spec['analysis_window']))
    table = new_full['table'].set_index('cellid')
    assert table['eligible'].tolist() == [True, True, True]
    assert table['exclusion_reason'].tolist() == ['', '', '']
    assert set(table['well']) == set(table['recording_group']) == {'G'}
    assert set(table['session_date']) == {sd.date}

    # What extract_day returns and records.
    halves_id, full_id = _sha256(features / 'halves' / 'manifest.json'), _sha256(features / 'full' / 'manifest.json')
    source_id = _sha256(day_dir / 'source' / 'extraction_manifest.json')
    assert result == {'day': 'D1', 'tier': 'raw', 'source_id': source_id,
                      'features': {'halves': halves_id, 'full': full_id}}
    written = project.read()
    record = written['days']['D1']
    assert record['tier'] == 'raw'
    assert record['source_id'] == source_id
    assert record['features'] == {'halves': halves_id, 'full': full_id}
    assert record['raw_params'] == {'sample_rate': 30000.0, 'n_channels': 8, 'dtype': '<i2', 'uv_per_bit': 0.195,
                                    'gzip': False}
    assert written['identity']['geometry_sha256'] == _geometry_sha256(sd.geometry)

    # The halves archive carries no partition, and its metadata names the day and its source.
    assert 'partition' not in new_halves.labels
    metadata = new_halves.metadata
    assert EXTRACTION_OPTIONS == OPTIONS_LITERAL
    assert metadata['extraction_options'] == OPTIONS_LITERAL
    assert metadata['dataset_id'] == 'P/D1'
    assert {key: metadata[key] for key in ('extraction_manifest_sha256', 'waveform_storage', 'project', 'day',
                                           'halves_rule', 'halves_boundary')} == {
        'extraction_manifest_sha256': source_id, 'waveform_storage': 'snippets', 'project': 'P', 'day': 'D1',
        'halves_rule': 'trial_start', 'halves_boundary': 300000}
    assert [source['well'] for source in metadata['sources']] == ['G']
    assert new_full['metadata'] == {'project': 'P', 'day': 'D1', 'tier': 'raw', 'halves_id': halves_id,
                                    'source_id': source_id, 'halves_rule': 'trial_start', 'halves_boundary': 300000,
                                    'extraction_options': OPTIONS_LITERAL}

    # The log has one timestamped line per step, the publish and the record included; the lock and the stage are gone.
    assert _log_steps(features) == [
        f'extract P/D1 on {socket.gethostname()} as process {os.getpid()}',
        'day D1 recorded, not yet extracted',
        'tier raw',
        f"verified {len(record['files'])} files",
        f'normalized source {source_id}',
        f'geometry {_geometry_sha256(sd.geometry)}',
        'halves rule trial_start, boundary 300000',
        f'halves {halves_id}',
        f'full {full_id}',
        PUBLISHED,
        RECORDING]
    assert _listing(day_dir) == ['features', 'source']
    assert _listing(features) == ['extract.log', 'full', 'halves']
    # The published folders are as readable as the day folder (mkdtemp alone would make them 0700).
    mode = stat.S_IMODE(day_dir.stat().st_mode)
    for folder in (features, features / 'halves', features / 'full'):
        assert stat.S_IMODE(folder.stat().st_mode) == mode
    # Each METHODS.md names the folder it was published in, not the stage it was written in.
    for part in ('halves', 'full'):
        methods = (features / part / 'METHODS.md').read_text(encoding='utf-8')
        assert f'- **Output folder:** `{features / part}`\n' in methods
        assert '.features-' not in methods

    # The pipeline imports no lab module, and nothing from project/days.py.
    proc = run_without_lab('import sys\nimport multiphy.extract.pipeline\n'
                           "print('multiphy.project.days' in sys.modules)\n", REPO)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip().endswith('False')


def test_weak_half_marks_unit_ineligible(tmp_path):
    sd = write_day(tmp_path / 'in', 'D1', weak_unit=True)
    project = _project(tmp_path, sd)
    _extract(project, tmp_path)

    features = project.day_dir('D1') / 'features'
    table = pd.read_csv(features / 'full' / 'units.csv', dtype=str, keep_default_na=False).set_index('cellid')
    assert sorted(table.index) == ['D1:0', 'D1:1', 'D1:2', 'D1:4']
    assert table.loc['D1:4', 'eligible'] == 'False'
    assert table.loc['D1:4', 'exclusion_reason'] == WEAK_REASON
    assert (table['eligible'] == 'True').sum() == 3
    assert sorted(load_full_day(features / 'full')['features']) == ['D1:0', 'D1:1', 'D1:2']

    # The weak unit's excluded half, and every other half, equal the old path's.
    new_halves = load_dataset(features / 'halves')
    _, archive, _ = _old_path(sd, tmp_path)
    old_halves = load_dataset(archive)
    assert new_halves.exclusions == old_halves.exclusions == [
        {'source_session': 'D1', 'source_unit_id': 'D1:4', 'split_index': 1, 'reason': WEAK_REASON}]
    np.testing.assert_equal(_halves_by_unit(new_halves), _halves_by_unit(old_halves))


def test_changed_events_refused(tmp_path):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)
    before = _project_json(project)

    path = sd.events_path
    status = path.stat()
    data = bytearray(path.read_bytes())
    index = data.index(b',A\n') + 1  # the condition of the first stimulus row
    data[index] = ord('C')
    path.write_bytes(bytes(data))
    os.utime(path, ns=(status.st_atime_ns, status.st_mtime_ns))
    assert (path.stat().st_size, path.stat().st_mtime_ns) == (status.st_size, status.st_mtime_ns)

    with pytest.raises(SourceChanged, match='events source changed'):
        _extract(project, tmp_path)
    day_dir = project.day_dir('D1')
    assert not (day_dir / 'source').exists()
    assert not (day_dir / 'features').exists()
    assert _listing(day_dir) == []  # and the lock is released
    assert not (tmp_path / 'scratch').exists()  # refused before any voltage was read
    assert _project_json(project) == before

    path.unlink()
    with pytest.raises(SourceMissing, match='events source missing'):
        _extract(project, tmp_path)
    assert _listing(day_dir) == []
    assert _project_json(project) == before


def test_reextract_refused(tmp_path, monkeypatch):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)
    day_dir = project.day_dir('D1')
    _extract(project, tmp_path)
    project_json = _project_json(project)
    full_manifest = (day_dir / 'features' / 'full' / 'manifest.json').read_bytes()
    tree = _tree(day_dir)

    with pytest.raises(FileExistsError, match='already extracted') as raised:
        _extract(project, tmp_path)
    assert str(raised.value) == ALREADY
    assert _project_json(project) == project_json
    assert (day_dir / 'features' / 'full' / 'manifest.json').read_bytes() == full_manifest
    assert _tree(day_dir) == tree
    assert _listing(day_dir) == ['features', 'source']  # no stage, and the lock is released

    # Features another process recorded while this extraction ran are found at the publish step:
    # refused, and the recorded features/ is not deleted.
    other = _project(tmp_path, sd, lab='lab-publish')
    other_dir = other.day_dir('D1')
    real_save = pipeline.save_full_day
    theirs = {'full/manifest.json': b'{"theirs": 1}'}

    def recorded_meanwhile(*args, **kwargs):
        full_id = real_save(*args, **kwargs)
        for relative, data in theirs.items():
            (other_dir / 'features' / relative).parent.mkdir(parents=True)
            (other_dir / 'features' / relative).write_bytes(data)

        def record(manifest):
            manifest['days']['D1']['features'] = {'halves': 'h' * 64, 'full': hashlib.sha256(b'{"theirs": 1}').hexdigest()}
        other.update(record)
        return full_id
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'save_full_day', recorded_meanwhile)
        with pytest.raises(FileExistsError, match='already extracted'):
            _extract(other, tmp_path)
    assert _tree(other_dir / 'features') == theirs
    assert _stages(other_dir) == []
    assert other.read()['days']['D1']['tier'] is None  # nothing of this extraction was recorded

    # Recorded between the publish and the update: the update refuses, and project.json keeps the other record.
    late = _project(tmp_path, sd, lab='lab-update')
    real_update = late.update

    def update_after_another(change):
        real_update(lambda manifest: manifest['days']['D1']['features'].update(halves='h' * 64, full='f' * 64))
        return real_update(change)
    with monkeypatch.context() as patch:
        patch.setattr(late, 'update', update_after_another)
        with pytest.raises(FileExistsError, match='already extracted'):
            _extract(late, tmp_path)
    record = late.read()['days']['D1']
    assert record['features'] == {'halves': 'h' * 64, 'full': 'f' * 64} and record['tier'] is None
    assert not (late.day_dir('D1') / '.extract.lock').exists()


def test_unrecorded_features_replaced(tmp_path, monkeypatch):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)
    day_dir = project.day_dir('D1')
    before = _project_json(project)
    (day_dir / 'source').mkdir()  # a source/ a stopped extraction left, without its manifest
    (day_dir / 'source' / 'stray.txt').write_text('left by a stopped extraction')

    # An unrecorded features/ is deleted at the publish step, before the rename: when the rename
    # then fails, the stage goes too and no features/ is left. project.json is unchanged.
    (day_dir / 'features').mkdir()
    (day_dir / 'features' / 'stray.txt').write_text('left by a stopped extraction')
    real_rename = os.rename

    def refuse_rename(source, destination):
        if Path(destination) == day_dir / 'features':
            raise OSError(errno.EIO, 'Input/output error')
        return real_rename(source, destination)
    with monkeypatch.context() as patch:
        patch.setattr(os, 'rename', refuse_rename)
        with pytest.raises(OSError, match='Input/output error'):
            _extract(project, tmp_path)
    assert _listing(day_dir) == ['source']  # no features/, no stage, and the lock is released
    assert _project_json(project) == before

    # The rerun publishes and records its own features.
    (day_dir / 'features').mkdir()
    (day_dir / 'features' / 'stray.txt').write_text('left by a stopped extraction')
    result = _extract(project, tmp_path)
    assert not (day_dir / 'features' / 'stray.txt').exists()
    assert not (day_dir / 'source' / 'stray.txt').exists()
    full_id = project.read()['days']['D1']['features']['full']
    assert _sha256(day_dir / 'features' / 'full' / 'manifest.json') == full_id == result['features']['full']
    assert _sha256(day_dir / 'features' / 'halves' / 'manifest.json') == result['features']['halves']
    assert _sha256(day_dir / 'source' / 'extraction_manifest.json') == result['source_id']
    assert _listing(day_dir) == ['features', 'source']


def test_lock_held(tmp_path):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)
    before = _project_json(project)
    lock = project.day_dir('D1') / '.extract.lock'
    lock.write_text('other 7')

    with pytest.raises(ExtractLocked, match='other 7') as raised:
        _extract(project, tmp_path)
    assert str(raised.value) == f'Day D1 is being extracted by other 7; remove {lock} if that process is gone.'
    assert lock.read_text() == 'other 7'
    assert _listing(project.day_dir('D1')) == ['.extract.lock']
    assert _project_json(project) == before

    lock.unlink()
    with pytest.raises(ValueError, match='Day D2 is not in project P.'):
        extract_day(project, 'D2', scratch_root=tmp_path / 'scratch')
    assert not project.day_dir('D2').exists()

    # The tier is checked under the lock, which is then released; nothing is written.
    with pytest.raises(ValueError, match='Unknown tier bogus.'):
        _extract(project, tmp_path, tier='bogus')
    add_day(project, 'D2', readers={'sort': 'phy', 'events': 'events_csv'},
            reader_options=_options(sd, day='D2'), date='2026-01-02', date_source='user')
    with pytest.raises(ValueError, match='Day D2 has no raw file.'):
        extract_day(project, 'D2', tier='raw', scratch_root=tmp_path / 'scratch')
    for day in ('D1', 'D2'):
        assert _listing(project.day_dir(day)) == []
    assert project.read()['days']['D1'] == json.loads(before)['days']['D1']


def test_geometry_mismatch_refused(tmp_path, monkeypatch):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)

    def foreign_geometry(manifest):
        manifest['identity']['geometry_sha256'] = 'x' * 64
    project.update(foreign_geometry)
    before = _project_json(project)

    with pytest.raises(ValueError, match='differs from the project geometry') as raised:
        _extract(project, tmp_path)
    assert str(raised.value) == 'Geometry of D1 differs from the project geometry.'
    day_dir = project.day_dir('D1')
    assert not (day_dir / 'features').exists()
    assert _stages(day_dir) == []
    assert not (day_dir / '.extract.lock').exists()
    assert not (tmp_path / 'scratch').exists()  # refused before any voltage was read
    assert _project_json(project) == before
    assert _listing(day_dir) == ['source']

    # A project geometry recorded by another day while this one was extracting is checked at the update.
    # The features were already published by then: they stay, unrecorded, for the next extraction to replace.
    def no_geometry(manifest):
        manifest['identity']['geometry_sha256'] = None
    project.update(no_geometry)
    real_save = pipeline.save_full_day

    def other_day_then_save(*args, **kwargs):
        def other_geometry(manifest):
            manifest['identity']['geometry_sha256'] = 'y' * 64
        project.update(other_geometry)
        return real_save(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'save_full_day', other_day_then_save)
        with pytest.raises(ValueError, match='differs from the project geometry'):
            _extract(project, tmp_path)
    written = project.read()
    assert written['identity']['geometry_sha256'] == 'y' * 64
    assert written['days']['D1'] == json.loads(before)['days']['D1']
    assert (day_dir / 'features' / 'full' / 'manifest.json').is_file()
    assert _stages(day_dir) == []
    (day_dir / 'features' / 'stale.txt').write_text('published, never recorded')

    project.update(no_geometry)
    result = _extract(project, tmp_path)
    assert not (day_dir / 'features' / 'stale.txt').exists()
    assert _sha256(day_dir / 'features' / 'full' / 'manifest.json') == result['features']['full']
    assert project.read()['days']['D1']['features'] == result['features']
    assert project.read()['identity']['geometry_sha256'] == _geometry_sha256(sd.geometry)


def test_reader_refusal_writes_nothing(tmp_path):
    sd = write_day(tmp_path / 'in', 'D1')
    sd.events_path.write_text('kind,start,stop,trial,condition\ntrial,0,500,1,\ntrial,400,800,2,\n')
    project = _project(tmp_path, sd)  # recorded as it now is, so it verifies
    before = _project_json(project)
    with pytest.raises(ValueError, match='out of order or overlap'):
        _extract(project, tmp_path)
    assert _listing(project.day_dir('D1')) == []
    assert _project_json(project) == before


def test_failed_write_leaves_nothing(tmp_path, monkeypatch):
    sd = write_day(tmp_path / 'in', 'D1')
    project = _project(tmp_path, sd)
    day_dir = project.day_dir('D1')
    before = _project_json(project)

    def fail(*args, **kwargs):
        raise RuntimeError('full day not written')
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'save_full_day', fail)
        with pytest.raises(RuntimeError, match='full day not written'):
            _extract(project, tmp_path)
    assert not (day_dir / 'features').exists()
    assert _stages(day_dir) == []
    assert project.read()['days']['D1']['features'] == {'halves': None, 'full': None}
    assert _project_json(project) == before
    assert _listing(day_dir) == ['source']  # the normalized source stays, and the lock is released

    # The retry reuses that source: it matches the recorded files, so the readers are not run again.
    normalized = (day_dir / 'source').stat().st_ino

    def no_read(*args, **kwargs):
        raise AssertionError('a matching source was read again')

    # A log that cannot be written: it is written in the stage before the rename, so the stage is
    # deleted and nothing is published or recorded.
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'read_day', no_read)
        _fail_log_writes(patch, lambda: True)
        with pytest.raises(OSError, match='Input/output error'):
            _extract(project, tmp_path)
    assert _listing(day_dir) == ['source']
    assert _project_json(project) == before

    # A project.lock still held when the wait runs out: ProjectLocked propagates and the lock stays.
    # The features were published before the update, so they stay, unrecorded.
    project_lock = project.path / 'project.lock'
    project_lock.write_text('other 8')
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'read_day', no_read)
        patch.setattr(pipeline, 'PROJECT_LOCK_WAIT', 0.0)
        with pytest.raises(ProjectLocked, match='other 8'):
            _extract(project, tmp_path)
    assert project_lock.read_text() == 'other 8'  # never broken
    assert (day_dir / 'source').stat().st_ino == normalized
    assert (day_dir / 'features' / 'full' / 'manifest.json').is_file()
    assert _unrecorded(project) and _project_json(project) == before
    assert _stages(day_dir) == [] and not (day_dir / '.extract.lock').exists()
    # Its log names the record that followed the publish; project.json says the record was not written.
    assert _log_steps(day_dir / 'features')[-2:] == [PUBLISHED, RECORDING]

    # A project.lock another writer releases while the update waits: the extraction is recorded, and
    # the unrecorded features/ the last attempt left is replaced. Nothing is written once the record
    # is: a log write that would fail from then on is never made, and extract_day returns.
    (day_dir / 'features' / 'stale.txt').write_text('published, never recorded')
    waits: list = []
    at_record: list = []
    real_update_waiting = pipeline._update_waiting

    def release(seconds):
        waits.append(seconds)
        if len(waits) == 2:
            project_lock.unlink()

    def record_then_snapshot(*args, **kwargs):
        written = real_update_waiting(*args, **kwargs)
        at_record.append(_tree(day_dir / 'features'))
        return written
    with monkeypatch.context() as patch:
        patch.setattr(pipeline, 'read_day', no_read)
        patch.setattr(pipeline, 'sleep', release)  # the default wait
        patch.setattr(pipeline, '_update_waiting', record_then_snapshot)
        _fail_log_writes(patch, lambda: bool(at_record))
        result = _extract(project, tmp_path)
    assert len(at_record) == 1 and _tree(day_dir / 'features') == at_record[0]
    assert waits == [0.5, 0.5] == [pipeline.PROJECT_LOCK_POLL] * 2
    assert pipeline.PROJECT_LOCK_WAIT == 60.0
    assert not project_lock.exists()
    assert project.read()['days']['D1']['features'] == result['features']
    assert _sha256(day_dir / 'features' / 'full' / 'manifest.json') == result['features']['full']
    assert not (day_dir / 'features' / 'stale.txt').exists()
    assert (day_dir / 'source').stat().st_ino == normalized
    steps = _log_steps(day_dir / 'features')
    assert f"reused source {result['source_id']}" in steps
    assert steps[-2:] == [PUBLISHED_OVER_LEFTOVER, RECORDING]
    assert _listing(day_dir) == ['features', 'source']


def test_day_without_events_extracts(tmp_path):
    sd = write_day(tmp_path / 'in', 'D1')
    project = Project.create(tmp_path / 'lab', 'P', group='G')
    options = {key: value for key, value in _options(sd).items() if key != 'events_path'}
    add_day(project, 'D1', readers={'sort': 'phy', 'raw': 'binary'}, reader_options=options, date=sd.date,
            date_source='user')
    result = _extract(project, tmp_path)

    features = project.day_dir('D1') / 'features'
    assert project.read()['days']['D1']['features'] == result['features']
    table = pd.read_csv(features / 'full' / 'units.csv', dtype=str, keep_default_na=False)
    assert sorted(table.cellid) == ['D1:0', 'D1:1', 'D1:2']
    assert (table['eligible'] == 'True').all()
    full = load_full_day(features / 'full')['features']
    assert sorted(full) == ['D1:0', 'D1:1', 'D1:2']
    assert all(row['fingerprints'] == {} for row in full.values())
