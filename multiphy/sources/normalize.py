"""A day's normalized ``source/`` folder: what its readers found, written once, read by extraction.

``write_source`` turns a ``DaySource`` into ``spikes.npz``, ``geometry.npy``, ``templates.npz``
(when the sort has templates), ``METHODS.md`` and ``extraction_manifest.json`` schema 2, the
schema 1 recording entry without ``partition``, ``well`` and ``windows``, plus the day, the
project, the readers, the trials, the spikes excluded after the analysis stop and each file's
identity. The folder is built as a stage beside ``source/`` and renamed into place once whole;
a ``source/`` a stopped extraction left is deleted first. A day's ``source/`` is written once
(a changed day goes in a new project), and its only writer is ``extract_day``, which holds the
day's ``.extract.lock``. ``read_source`` reads schema 2 and schema 1 folders. The sha256 of
``extraction_manifest.json`` is the day's ``source_id``. No voltage is read or written here.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile

import numpy as np

from ..waveform_preprocessing import SharedWaveformPreprocessor
from .base import DaySource

PREPROCESSING = ('Acquired samples scaled by uv_per_bit; shared_unwhitened_v1 profile at its defaults: high-pass, '
                 'median reference, unwhitened; native spike timestamps.')
SCHEMA_VERSION = 2
KIND = 'multiphy_day_source'
MANIFEST_FILE = 'extraction_manifest.json'
SPIKES_FILE = 'spikes.npz'
GEOMETRY_FILE = 'geometry.npy'
TEMPLATES_FILE = 'templates.npz'
EXTRAS_FILE = 'extras.npz'
METHODS_FILE = 'METHODS.md'
_ROLE_ORDER = ('sort', 'events', 'raw')
_MATCHED_FIELDS = ('resolved', 'size_bytes', 'mtime_ns')


def _npy_bytes(geometry) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.asarray(geometry, dtype=float))
    return buffer.getvalue()


def geometry_sha256(geometry: np.ndarray) -> str:
    """The sha256 of the ``.npy`` bytes of ``geometry`` as float: the hash of a normalized ``geometry.npy``."""
    return hashlib.sha256(_npy_bytes(geometry)).hexdigest()


def _identity_dict(identity) -> dict:
    """A ``FileIdentity`` or its ``to_dict()`` form, as that dict."""
    return identity.to_dict() if hasattr(identity, 'to_dict') else dict(identity)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def _windows(windows) -> list:
    return np.asarray(windows, dtype=np.int64).reshape(-1, 2).tolist()


def _entry(day_source: DaySource, *, day: str, project: str, readers: dict, identities: dict) -> dict:
    """The schema 2 recording entry for ``day_source``."""
    spikes = {unit: np.asarray(day_source.spikes[unit], dtype=np.int64) for unit in sorted(day_source.spikes)}
    raw = day_source.raw
    start, stop = (int(value) for value in day_source.analysis_window)
    n_channels = int(np.asarray(day_source.geometry).shape[0])
    raw_voltage = None
    if raw:
        raw_voltage = (raw['path'] if 'path' in raw
                       else [[segment['path'], segment['n_samples']] for segment in raw['segments']])
    entry = {
        'source_session': day,
        'raw_voltage': raw_voltage,
        'spikes_file': SPIKES_FILE,
        'geometry_file': GEOMETRY_FILE,
        'single_unit_ids': sorted(spikes),
        'sample_rate': float(day_source.sample_rate),
        'n_samples': int(day_source.n_samples),
        'analysis_window': [start, stop],
        'dtype': raw['dtype'] if raw else None,
        'voltage_scale': raw['uv_per_bit'] if raw else None,
        'waveform_preprocessing': SharedWaveformPreprocessor(day_source.sample_rate, n_channels).metadata(),
        'preprocessing': PREPROCESSING,
        'clock_verification': dict(day_source.clock),
        'stimuli': [dict(stimulus) for stimulus in day_source.stimuli],
        'baseline_windows': _windows(day_source.baseline_windows),
        'day': day,
        'project': project,
        'readers': dict(readers),
        'trials': _windows(day_source.trials),
        'excluded_tail_spike_counts': {unit: int((samples >= stop).sum()) for unit, samples in spikes.items()},
    }
    if day_source.templates is not None:
        entry['templates_file'] = TEMPLATES_FILE
    entry['extras_file'] = EXTRAS_FILE
    entry['provenance'] = dict(day_source.provenance,
                               files={key: _identity_dict(identities[key]) for key in sorted(identities)})
    return entry


def _methods(entry: dict, identities: dict) -> str:
    """One paragraph: the readers, each file's role and the preprocessing extraction applies."""
    readers = entry['readers']
    roles = [role for role in _ROLE_ORDER if role in readers] + sorted(set(readers) - set(_ROLE_ORDER))
    reader_text = ', '.join(f'{role} {readers[role]}' for role in roles) or 'none'
    file_text = ', '.join(f'{key} ({_identity_dict(identities[key])["role"]})' for key in identities) or 'none'
    held = 'spikes.npz (unit id to int64 spike samples), geometry.npy (channel x, y in µm)'
    if 'templates_file' in entry:
        held += ", templates.npz (unit id to the sort's unwhitened template)"
    if 'extras_file' in entry:
        held += ', extras.npz (per-unit arrays from the readers, such as tone tuning; no backend reads them)'
    start, stop = entry['analysis_window']
    return (f"Normalized source of day {entry['day']} in project {entry['project']}: the output of its readers, "
            f'with no voltage. Readers by role: {reader_text}. Files read, as key (role): {file_text}; each '
            f"file's identity is in the provenance of extraction_manifest.json (schema {SCHEMA_VERSION}). "
            f"This folder holds {held}. Spike times are 0-based rows of the raw clock "
            f"({entry['clock_verification'].get('method')}); the analysis window is [{start}, {stop}) of "
            f"{entry['n_samples']} samples at {entry['sample_rate']:g} Hz, and "
            f'{sum(entry["excluded_tail_spike_counts"].values())} spikes at or after its stop are excluded. '
            f'Extraction preprocessing: {PREPROCESSING}\n')


def write_source(day_source: DaySource, target: Path, *, day: str, project: str, readers: dict,
                 identities: dict) -> str:
    """Write ``day_source`` to the folder ``target`` as a whole; return the ``source_id``.

    ``readers`` maps role to ``id@version``; ``identities`` maps file key to its ``FileIdentity``
    (or that identity's ``to_dict()``). The folder is built as a stage beside ``target``. Once
    the stage is complete, a folder already at ``target`` (one a stopped extraction left) is
    deleted and the stage is renamed to ``target``. Any failure deletes the stage and re-raises.
    The caller holds the day's ``.extract.lock``, so no other write runs on ``target``.
    """
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    entry = _entry(day_source, day=day, project=project, readers=readers, identities=identities)
    manifest = {'schema_version': SCHEMA_VERSION, 'kind': KIND, 'recordings': [entry]}
    manifest_bytes = (json.dumps(manifest, indent=2, allow_nan=False, default=_json_default) + '\n').encode('utf-8')

    stage = Path(tempfile.mkdtemp(prefix='.source-', dir=target.parent))
    try:
        # mkdtemp makes the stage 0700; give it the parent's bits so other lab users and queue
        # jobs running as another user can read source/.
        os.chmod(stage, stat.S_IMODE(os.stat(target.parent).st_mode))
        np.savez_compressed(stage / SPIKES_FILE, **{unit: np.asarray(day_source.spikes[unit], dtype=np.int64)
                                                    for unit in entry['single_unit_ids']})
        (stage / GEOMETRY_FILE).write_bytes(_npy_bytes(day_source.geometry))
        if day_source.templates is not None:
            np.savez_compressed(stage / TEMPLATES_FILE, **{unit: np.asarray(day_source.templates[unit])
                                                           for unit in sorted(day_source.templates)})
        np.savez_compressed(stage / EXTRAS_FILE, **{name: np.asarray(value)
                                                    for name, value in (day_source.extras or {}).items()})
        (stage / METHODS_FILE).write_text(_methods(entry, identities), encoding='utf-8')
        (stage / MANIFEST_FILE).write_bytes(manifest_bytes)
        if os.path.lexists(target):
            shutil.rmtree(target)
        os.rename(stage, target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return hashlib.sha256(manifest_bytes).hexdigest()


def _input_path(value, source_dir: Path) -> Path:
    """A manifest file path: absolute as written, else relative to the source folder."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else source_dir / path


def _only_recording(manifest: dict, path: Path) -> dict:
    version = manifest.get('schema_version') if isinstance(manifest, dict) else None
    if version not in (1, SCHEMA_VERSION):
        raise ValueError(f'{path}: schema_version must be 1 or {SCHEMA_VERSION}, not {version!r}.')
    if version == SCHEMA_VERSION and manifest.get('kind') != KIND:
        raise ValueError(f'{path}: kind must be {KIND}, not {manifest.get("kind")!r}.')
    recordings = manifest.get('recordings')
    if not isinstance(recordings, list) or len(recordings) != 1:
        count = len(recordings) if isinstance(recordings, list) else 0
        raise ValueError(f'{path}: a day source holds exactly one recording, not {count}.')
    return recordings[0]


def _read_entry(source_dir: Path) -> tuple:
    """``(entry, manifest_bytes)`` of ``source_dir``; a schema 1 entry loses its ``partition`` and ``well``."""
    path = source_dir / MANIFEST_FILE
    manifest_bytes = path.read_bytes()
    manifest = json.loads(manifest_bytes)
    entry = dict(_only_recording(manifest, path))
    if manifest['schema_version'] == 1:
        entry.pop('partition', None)
        entry.pop('well', None)
    return entry, manifest_bytes


def read_source_entry(source_dir: Path) -> dict:
    """The recording entry ``read_source`` returns, read from ``extraction_manifest.json`` alone."""
    return _read_entry(Path(source_dir))[0]


def read_source(source_dir: Path) -> dict:
    """``{'entry', 'spikes', 'geometry', 'templates', 'extras', 'source_id'}`` of a schema 2 or schema 1 folder.

    ``extras`` holds the arrays of the folder's extras file, ``{}`` when it names none.

    A schema 1 entry loses its ``partition`` and ``well``. ``spikes`` holds the entry's
    ``single_unit_ids``; ``templates`` is ``None`` when the entry names no templates file. Each
    file opens by its path: a relative path under ``source_dir``, an absolute one as written (a
    schema 1 folder may point outside itself). The ``source_id`` is the sha256 of
    ``extraction_manifest.json``.
    """
    source_dir = Path(source_dir)
    entry, manifest_bytes = _read_entry(source_dir)

    spikes_path = _input_path(entry['spikes_file'], source_dir)
    with np.load(spikes_path, allow_pickle=False) as saved:
        missing = [unit for unit in entry['single_unit_ids'] if unit not in saved.files]
        if missing:
            raise ValueError(f'{spikes_path} has no spikes for unit {missing[0]}.')
        spikes = {unit: saved[unit].copy() for unit in entry['single_unit_ids']}
    geometry = np.load(_input_path(entry['geometry_file'], source_dir), allow_pickle=False)
    templates = None
    if entry.get('templates_file'):
        with np.load(_input_path(entry['templates_file'], source_dir), allow_pickle=False) as saved:
            templates = {unit: saved[unit].copy() for unit in saved.files}
    extras = {}
    if entry.get('extras_file'):
        with np.load(_input_path(entry['extras_file'], source_dir), allow_pickle=False) as saved:
            extras = {name: saved[name].copy() for name in saved.files}
    return {'entry': entry, 'spikes': spikes, 'geometry': geometry, 'templates': templates, 'extras': extras,
            'source_id': hashlib.sha256(manifest_bytes).hexdigest()}


def source_matches(source_dir: Path, files: dict) -> bool:
    """True when ``source_dir`` was normalized from exactly ``files``: same keys, and each file's
    ``resolved``, ``size_bytes`` and ``mtime_ns`` equal. ``files`` maps key to an identity or its dict."""
    path = Path(source_dir) / MANIFEST_FILE
    try:
        manifest = json.loads(path.read_bytes())
        recorded = _only_recording(manifest, path)['provenance']['files']
    except (OSError, ValueError, KeyError, TypeError):
        return False
    if not isinstance(recorded, dict) or set(recorded) != set(files):
        return False
    for key, identity in files.items():
        current, stored = _identity_dict(identity), recorded[key]
        if not isinstance(stored, dict) or any(stored.get(f) != current.get(f) for f in _MATCHED_FIELDS):
            return False
    return True
