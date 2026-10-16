"""The ``project.json`` schema: a project's identity and its days.

Each day records its date, where the date came from, the readers and reader options that
produced it, the identities of its files, and the ids of its extracted features.
``validate_manifest`` runs on every read and every write, so an invalid file is never
written and never read back silently.
"""
from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
import stat
import tempfile

SCHEMA_VERSION = 1
KIND = 'multiphy_project'
ROLES = ('sort', 'events', 'raw')
DATE_SOURCES = ('reader', 'user')
TIERS = (None, 'raw', 'templates')


def _utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def new_manifest(name: str, *, group: str) -> dict:
    """An empty project: identity fields other than ``group`` are filled in later."""
    return {
        'schema_version': SCHEMA_VERSION,
        'kind': KIND,
        'name': name,
        'created_at': _utc_now(),
        'identity': {'subject': None, 'implant': None, 'group': group, 'probe': None, 'geometry_sha256': None},
        'days': {},
    }


def new_day(date: str, date_source: str, readers: dict, reader_options: dict, files: dict) -> dict:
    """A day record before extraction: no raw parameters, tier, source or features yet."""
    return {
        'date': date,
        'date_source': date_source,
        'added_at': _utc_now(),
        'readers': dict(readers),
        'reader_options': dict(reader_options),
        'files': dict(files),
        'raw_params': None,
        'tier': None,
        'source_id': None,
        'features': {'halves': None, 'full': None},
    }


def validate_manifest(manifest: dict) -> dict:
    """Raise ``ValueError`` naming the first invalid field; return ``manifest`` unchanged."""
    if not isinstance(manifest, dict):
        raise ValueError(f'project.json must hold a JSON object, not {type(manifest).__name__}.')
    if manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError(f'schema_version must be {SCHEMA_VERSION}, not {manifest.get("schema_version")!r}.')
    if manifest.get('kind') != KIND:
        raise ValueError(f'kind must be {KIND}, not {manifest.get("kind")!r}.')
    days = manifest.get('days')
    if not isinstance(days, dict):
        raise ValueError(f'days must be an object of day name to day record, not {type(days).__name__}.')
    first_on = {}
    for day in sorted(days):
        record = days[day]
        if not isinstance(record, dict):
            raise ValueError(f'days.{day} must be an object, not {type(record).__name__}.')
        if record.get('date_source') not in DATE_SOURCES:
            raise ValueError(f'days.{day}.date_source must be reader or user, not {record.get("date_source")!r}.')
        if record.get('tier') not in TIERS:
            raise ValueError(f'days.{day}.tier must be None, raw or templates, not {record.get("tier")!r}.')
        try:
            date = datetime.date.fromisoformat(record.get('date'))
        except (TypeError, ValueError):
            raise ValueError(f'days.{day}.date must be YYYY-MM-DD, not {record.get("date")!r}.') from None
        files = record.get('files')
        if not isinstance(files, dict):
            raise ValueError(f'days.{day}.files must be an object, not {type(files).__name__}.')
        for key in sorted(files):
            role = files[key].get('role') if isinstance(files[key], dict) else None
            if role not in ROLES:
                raise ValueError(f'days.{day}.files.{key}.role must be one of {", ".join(ROLES)}, not {role!r}.')
        if date in first_on:
            raise ValueError(f'Days {first_on[date]} and {day} share the date {date.isoformat()}.')
        first_on[date] = day
    return manifest


def read_manifest(path) -> dict:
    with Path(path).open(encoding='utf-8') as stream:
        return validate_manifest(json.load(stream))


def write_manifest(path, manifest: dict) -> None:
    """Validate, then replace ``path`` atomically; a failed write leaves ``path`` and no temp file."""
    path = Path(path)
    validate_manifest(manifest)
    text = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix='.project-')
    try:
        with os.fdopen(handle, 'w', encoding='utf-8') as stream:
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        # mkstemp makes the file 0600; give it the project folder's read/write bits so other
        # lab users (and queue jobs running as another user) can read project.json.
        os.chmod(temporary, stat.S_IMODE(os.stat(path.parent).st_mode) & 0o666)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise
