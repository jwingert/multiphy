"""One pipeline turns a project day into its features, once: ``extract_day``.

``extract_day`` holds the day's ``.extract.lock`` and, in order: refuses a day already
extracted; checks the tier; verifies every recorded file against its identity; normalizes the
day into ``source/`` (unless ``source/`` was normalized from these files); checks the day's
geometry against the project's. The raw tier then cuts the analysis window into two halves;
extracts the halves from raw voltage into the benchmark archive format (``features/halves/``,
no partition); pools each unit's halves into its full-day row (``features/full/``). The
templates tier instead builds the full day from the sort's templates and the complete spike
trains (``templates.py``): it reads no voltage, writes no halves and records
``features['halves']`` ``None``. The tier is ``'raw'`` when the day records a raw file, else
``'templates'``; ``'templates'`` may also be forced on a day with raw. The features are built
in a hidden stage in the day folder, renamed to ``features/`` and then recorded in
``project.json``.
``features/extract.log`` has one timestamped line per step. It is written in the stage before the
rename, so nothing is written after the record: its last line names the record in
``project.json`` that follows the publish, and ``project.json`` alone says whether that record
was written.

A day's ``source/`` and ``features/`` are written once: a day with recorded features is
refused, and a changed day goes in a new project. A failure before the rename deletes the stage
and leaves ``project.json`` and recorded features as they were. An unrecorded ``features/`` (a
stopped extraction left it) is deleted at the publish step just before the rename, so a failed
rename leaves no ``features/``. A failure after the rename leaves ``features/`` in place,
unrecorded. Either way the next ``extract_day`` publishes its own.
The raw pass hashes a ``.gz`` raw file's stored bytes as it reads them, at no extra read. A
``sha256`` recorded when extraction starts that differs refuses the day before anything is
published. The record step checks the digest again, because a ``sha256`` may be recorded
meanwhile: a mismatch there raises ``SourceChanged`` after the rename, so ``features/`` is left
in place, unrecorded, like any failure after the rename, and ``project.json`` stays
authoritative. Otherwise the record of the features also stores the digest, with
``sha256_from`` ``'extract'``. A flat raw file is never hashed here.
Extraction options are fixed: ``EXTRACTION_OPTIONS``. No module here imports a lab module, and
nothing here imports ``project/days.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import shutil
import socket
import stat
import tempfile
from time import monotonic, sleep
from typing import Optional

import numpy as np

from .. import extract_benchmark
from ..benchmark import build_dataset, save_dataset
from ..project.folder import Project, ProjectLocked
from ..project.identity import FileIdentity, SourceChanged, verify
from ..sources.base import read_day
from ..sources.normalize import geometry_sha256, read_source, source_matches, write_source
from .full_day import full_day_rows, save_full_day
from .raw_pass import raw_segments
from .split import halves
from .templates import template_full_rows, templates_methods

EXTRACTION_OPTIONS = {'before': 41, 'after': 41, 'max_spikes': 100, 'min_spikes': 20, 'seed': 0}
WAVEFORM_STORAGE = 'snippets'
TIERS = ('raw', 'templates')
LOCK_FILE = '.extract.lock'
SOURCE_DIR = 'source'
FEATURES_DIR = 'features'
LOG_FILE = 'extract.log'
PROJECT_FILE = 'project.json'
# Every other project.json writer holds project.lock for one short update. Recording the features
# polls for it every PROJECT_LOCK_POLL seconds, for up to PROJECT_LOCK_WAIT, and never breaks it.
PROJECT_LOCK_WAIT = 60.0
PROJECT_LOCK_POLL = 0.5
METHODS_FILE = 'METHODS.md'
_OUTPUT_LINE = b'- **Output folder:** '


class ExtractLocked(RuntimeError):
    """``.extract.lock`` exists in the day folder: another process is extracting the day, or one died doing so."""


def _take_lock(lock: Path, day: str) -> None:
    """Create ``lock`` exclusively, holding ``'<host> <pid>'``; a held lock raises ``ExtractLocked``."""
    try:
        handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            holder = lock.read_text(encoding='utf-8').strip() or 'a process that has not written its name yet'
        except OSError:
            holder = 'an unreadable lock'
        raise ExtractLocked(f'Day {day} is being extracted by {holder}; remove {lock} if that process is gone.') from None
    try:
        try:
            os.write(handle, f'{socket.gethostname()} {os.getpid()}'.encode('utf-8'))
        finally:
            os.close(handle)
    except BaseException:
        lock.unlink(missing_ok=True)  # our own lock, never another holder's
        raise


def _not_extracted(manifest: dict, project: Project, day: str) -> dict:
    """``day``'s record in ``manifest``; a day not in it, or one with recorded features, is refused."""
    record = manifest['days'].get(day)
    if record is None:
        raise ValueError(f'Day {day} is not in project {project.name}.')
    if record['features']['full'] is not None:
        raise FileExistsError(f'Day {day} is already extracted; features are written once. Put it in a new '
                              'project to extract it again.')
    return record


def _check_geometry(identity: dict, geometry_hash: str, day: str) -> None:
    recorded = identity.get('geometry_sha256')
    if recorded is not None and recorded != geometry_hash:
        raise ValueError(f'Geometry of {day} differs from the project geometry.')


def _publish_methods(folder: Path, published: Path) -> None:
    """Point ``folder/METHODS.md``'s output-folder line at ``published``, where ``folder`` is renamed to.

    ``METHODS.md`` is not among the files a manifest hashes, so the halves and full ids stay.
    """
    path = folder / METHODS_FILE
    lines = path.read_bytes().split(b'\n')
    found = [index for index, line in enumerate(lines) if line.startswith(_OUTPUT_LINE)]
    if len(found) != 1:
        raise ValueError(f'{path} names its output folder {len(found)} times, not once.')
    lines[found[0]] = _OUTPUT_LINE + f'`{published}`'.encode('utf-8')
    path.write_bytes(b'\n'.join(lines))


def _raw_params(entry: dict, geometry: np.ndarray) -> dict:
    """The raw file's parameters as the normalized source holds them; ``gzip`` is true when every segment is ``.gz``."""
    paths = [path for path, _ in raw_segments(entry['raw_voltage'], int(entry['n_samples']))]
    return {'sample_rate': float(entry['sample_rate']), 'n_channels': int(np.asarray(geometry).shape[0]),
            'dtype': entry['dtype'], 'uv_per_bit': float(entry['voltage_scale']),
            'gzip': all(path.suffix == '.gz' for path in paths)}


def _raw_key(files: dict, raw_voltage: str) -> str:
    """The key of the raw-role file in ``files`` that ``raw_voltage``, one path, resolves to."""
    resolved = os.path.realpath(raw_voltage)
    keys = [key for key, file in files.items()
            if file['role'] == 'raw' and os.path.realpath(file['resolved']) == resolved]
    if len(keys) != 1:
        raise ValueError(f'{len(keys)} recorded raw-role files are {resolved}, not one.')
    return keys[0]


def _checked_raw_sha256(raw_file: dict, digest: Optional[str]) -> str:
    """``digest``, the raw pass's sha256 of the ``.gz`` ``raw_file``; a different recorded ``sha256`` raises ``SourceChanged``."""
    if digest is None:  # the readers only seek forward, so a .gz pass always yields a digest
        raise RuntimeError(f"The raw pass did not hash {raw_file['resolved']}.")
    recorded = raw_file['sha256']
    if recorded is not None and recorded != digest:
        raise SourceChanged(f"raw source changed: {raw_file['resolved']}")
    return digest


def _share_mode(path: Path, parent: Path) -> None:
    """Give ``path`` its parent's permission bits: ``mkdtemp`` makes folders 0700, and queue jobs run as another user."""
    os.chmod(path, stat.S_IMODE(os.stat(parent).st_mode))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def extract_day(project: Project, day: str, *, tier: Optional[str] = None,
                scratch_root: Optional[Path] = None) -> dict:
    """Extract ``day``'s features once and record them in ``project.json``; return ``{'day', 'tier', 'source_id', 'features'}``.

    ``tier`` ``None`` is ``'raw'`` when a ``raw`` file is recorded, else ``'templates'``;
    ``'raw'`` needs a recorded ``raw`` file, and ``'templates'`` may be forced on a day with one.
    The templates tier needs a sort with templates, reads no voltage and writes no halves.
    ``scratch_root`` holds the temporary voltage snippets (default: ``_runtime/`` under the
    lab root). A held ``.extract.lock`` (``ExtractLocked``), a day already extracted
    (``FileExistsError``), a changed or missing file (``SourceChanged``, ``SourceMissing``) and a
    day geometry other than the project's (``ValueError``) are refused before any features are
    written. A ``.gz`` raw file is hashed during the raw pass. A ``sha256`` recorded when
    extraction starts that differs raises ``SourceChanged`` before the features are published;
    one recorded later that differs raises ``SourceChanged`` at the record, after the publish,
    and leaves ``features/`` in place, unrecorded (``project.json`` stays authoritative).
    Otherwise the digest is recorded with ``sha256_from`` ``'extract'``.
    """
    if day not in project.read()['days']:  # before the lock: an unknown day has no folder to hold it
        raise ValueError(f'Day {day} is not in project {project.name}.')
    day_dir = project.day_dir(day)
    lock = day_dir / LOCK_FILE
    _take_lock(lock, day)
    try:
        return _extract_locked(project, day, day_dir, tier=tier, scratch_root=scratch_root)
    finally:
        lock.unlink(missing_ok=True)


def _extract_locked(project: Project, day: str, day_dir: Path, *, tier: Optional[str],
                    scratch_root: Optional[Path]) -> dict:
    source_dir = day_dir / SOURCE_DIR
    features_dir = day_dir / FEATURES_DIR
    log_lines: list = []

    def log(step: str) -> None:
        log_lines.append(f'{datetime.now(timezone.utc).isoformat()} · {step}')

    log(f'extract {project.name}/{day} on {socket.gethostname()} as process {os.getpid()}')

    # 1. The day is in the project and not yet extracted.
    manifest = project.read()
    record = _not_extracted(manifest, project, day)
    identity = manifest['identity']
    files = record['files']
    log(f'day {day} recorded, not yet extracted')

    # 2. Tier.
    has_raw = any(file['role'] == 'raw' for file in files.values())
    if tier is None:
        tier = 'raw' if has_raw else 'templates'
    if tier not in TIERS:
        raise ValueError(f'Unknown tier {tier}.')
    if tier == 'raw' and not has_raw:
        raise ValueError(f'Day {day} has no raw file.')
    log(f'tier {tier}')

    # 3. Every recorded file is still the file the day was recorded from; only the raw file is not re-hashed.
    for key in sorted(files):
        verify(FileIdentity.from_dict(files[key]), rehash=files[key]['role'] != 'raw')
    log(f'verified {len(files)} files')

    # 4. Normalize, unless source/ was normalized from these files; write_source replaces a source/
    # that a stopped extraction left.
    if source_matches(source_dir, files):
        normalized = 'reused'
    else:
        by_role: dict = {}
        for key, file in files.items():
            by_role.setdefault(file['role'], {})[key] = Path(file['resolved'])
        day_source = read_day(record['readers'], by_role, record['reader_options'])
        write_source(day_source, source_dir, day=day, project=project.name, readers=record['readers'],
                     identities=files)
        normalized = 'normalized'
    source = read_source(source_dir)
    entry = source['entry']
    source_id = source['source_id']
    log(f'{normalized} source {source_id}')
    if tier == 'templates' and source['templates'] is None:
        raise ValueError(f'Day {day} has no templates; the templates tier needs a sort with templates.')

    # 5. Geometry.
    geometry_hash = geometry_sha256(source['geometry'])
    _check_geometry(identity, geometry_hash, day)
    log(f'geometry {geometry_hash}')

    group = identity['group']
    halves_id = raw_sha256 = None  # the templates tier has no halves and hashes no raw file
    if tier == 'raw':
        # 6. The halves.
        analysis_window = tuple(entry['analysis_window'])
        h = halves(np.asarray(entry['trials'], dtype=np.int64).reshape(-1, 2), analysis_window)
        log(f"halves rule {h['rule']}, boundary {h['boundary']}")

        # 7. The schema 1 manifest the benchmark builders read, in memory only.
        spec = dict(entry, source_session=day, well=group, partition=None, windows=h['windows'].tolist(),
                    spikes_file=str(source_dir / 'spikes.npz'), geometry_file=str(source_dir / 'geometry.npy'))
        extraction = {'schema_version': 1, 'dataset_id': f'{project.name}/{day}', 'recordings': [spec]}

    gz_raw = (tier == 'raw' and isinstance(entry['raw_voltage'], str)
              and Path(entry['raw_voltage']).suffix == '.gz')
    raw_key = _raw_key(files, entry['raw_voltage']) if gz_raw else None

    stage = Path(tempfile.mkdtemp(prefix='.features-', dir=day_dir))
    try:
        _share_mode(stage, day_dir)
        if tier == 'raw':
            # 8. The halves, from raw voltage.
            with extract_benchmark.open_recording_sources(extraction, source_dir / 'extraction_manifest.json',
                                                          options=dict(EXTRACTION_OPTIONS), root=project.root,
                                                          scratch_root=scratch_root,
                                                          waveform_storage=WAVEFORM_STORAGE, hash_raw=True) as sources:
                dataset = build_dataset(sources, dataset_id=f'{project.name}/{day}',
                                        extraction_options=dict(EXTRACTION_OPTIONS))
                raw_sha256 = sources[0].read_samples.raw_sha256
            if gz_raw:
                raw_sha256 = _checked_raw_sha256(files[raw_key], raw_sha256)
            dataset.metadata.update(extraction_manifest_sha256=source_id, waveform_storage=WAVEFORM_STORAGE,
                                    project=project.name, day=day, halves_rule=h['rule'],
                                    halves_boundary=h['boundary'])
            save_dataset(dataset, stage / 'halves', cache_root=project.root)
            _share_mode(stage / 'halves', day_dir)
            _publish_methods(stage / 'halves', features_dir / 'halves')
            halves_id = _sha256(stage / 'halves' / 'manifest.json')
            log(f'halves {halves_id}')

            # 9. The full day, pooled from the halves.
            full = full_day_rows(dataset, source['spikes'], entry, day=day, group=group, session_date=record['date'],
                                 analysis_window=analysis_window, windows=h['windows'],
                                 max_spikes=EXTRACTION_OPTIONS['max_spikes'])
            full['extras'] = dict(source['extras'])
            full_id = save_full_day(full, stage / 'full', metadata={
                'project': project.name, 'day': day, 'tier': tier, 'halves_id': halves_id, 'source_id': source_id,
                'halves_rule': h['rule'], 'halves_boundary': h['boundary'],
                'extraction_options': dict(EXTRACTION_OPTIONS)})
            _share_mode(stage / 'full', day_dir)
            _publish_methods(stage / 'full', features_dir / 'full')
            log(f'full {full_id}')
        else:
            # 8. No halves: the templates tier reads no voltage.
            log('no halves: the templates tier reads no voltage')

            # 9. The full day, from the sort's templates and the complete spike trains.
            full = template_full_rows(source, day=day, group=group, session_date=record['date'],
                                      min_spikes=EXTRACTION_OPTIONS['min_spikes'])
            full['extras'] = dict(source['extras'])
            full_id = save_full_day(full, stage / 'full', metadata={
                'project': project.name, 'day': day, 'tier': tier, 'halves_id': None, 'source_id': source_id,
                'halves_rule': None, 'halves_boundary': None, 'extraction_options': dict(EXTRACTION_OPTIONS)})
            # METHODS.md is not among the files the manifest hashes; this one describes templates, not halves.
            (stage / 'full' / METHODS_FILE).write_text(templates_methods(stage / 'full', full), encoding='utf-8')
            _share_mode(stage / 'full', day_dir)
            _publish_methods(stage / 'full', features_dir / 'full')
            log(f'full {full_id}')

        # 10. Publish. Recorded features are never replaced; an unrecorded features/ is what a
        # stopped extraction left, and this one's stage takes its place. The log is written into the
        # stage, so its publish line reaches features/ only through the rename.
        _not_extracted(project.read(), project, day)
        leftover = os.path.lexists(features_dir)
        log(f'published {FEATURES_DIR}/' + (f' in place of an unrecorded {FEATURES_DIR}/' if leftover else ''))
        # Step 11's line as well: the log is finished before the record, so no write can fail after
        # project.json records the features. Whether it did is project.json's to say, not the log's.
        log(f'recording in {PROJECT_FILE}; {PROJECT_FILE} says whether it was recorded')
        (stage / LOG_FILE).write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
        if leftover:
            shutil.rmtree(features_dir)
        os.rename(stage, features_dir)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)  # gone already once it became features/
        raise

    # 11. Record the features. A failure from here on leaves features/ in place, unrecorded; once the
    # record is written, nothing else is.
    features = {'halves': halves_id, 'full': full_id}
    raw_params = _raw_params(entry, source['geometry']) if entry.get('raw_voltage') else None

    def change(current: dict) -> None:
        day_record = _not_extracted(current, project, day)
        _check_geometry(current['identity'], geometry_hash, day)  # another day may have recorded one meanwhile
        day_record.update(tier=tier, source_id=source_id, features=dict(features), raw_params=raw_params)
        if current['identity'].get('geometry_sha256') is None:
            current['identity']['geometry_sha256'] = geometry_hash
        if gz_raw:
            raw_file = day_record['files'][raw_key]
            _checked_raw_sha256(raw_file, raw_sha256)
            raw_file.update(sha256=raw_sha256, sha256_from='extract')

    _update_waiting(project, change)
    return {'day': day, 'tier': tier, 'source_id': source_id, 'features': dict(features)}


def _update_waiting(project: Project, change) -> dict:
    """``project.update(change)``, waiting for a held ``project.lock``; ``ProjectLocked`` after ``PROJECT_LOCK_WAIT``.

    ``Project.update`` raises ``ProjectLocked`` when it cannot create the lock, before ``change``
    runs, so trying again is safe.
    """
    deadline = monotonic() + PROJECT_LOCK_WAIT
    while True:
        try:
            return project.update(change)
        except ProjectLocked:
            if monotonic() >= deadline:
                raise
        sleep(PROJECT_LOCK_POLL)
