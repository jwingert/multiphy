"""Adding a day to a project.

``add_day`` asks each of the day's readers which files it will open, records each file's
identity (every file but the raw voltage is hashed) and writes the day record into
``project.json``. It reads no data: normalizing into ``source/`` happens at extraction. A day
is added once: its ``source/`` and ``features/`` are written once, and a changed day goes in a
new project.
"""
from __future__ import annotations

from ..sources.base import file_role, get_reader
from .folder import NAME_PATTERN, Project
from .identity import record
from .manifest import ROLES, new_day


def _day_readers(readers: dict) -> dict:
    """Role -> reader, in role order; refuses an unknown role, a reader that does not serve its role and a day without a sort reader."""
    unknown = sorted(set(readers) - set(ROLES))
    if unknown:
        raise ValueError(f'Unknown role {unknown[0]}; the roles are {", ".join(ROLES)}.')
    if 'sort' not in readers:
        raise ValueError('A day needs a sort reader.')
    found = {}
    for role in ROLES:
        if role not in readers:
            continue
        reader = get_reader(readers[role])
        if role not in reader.roles:
            raise ValueError(f'Reader {reader.id} does not serve the {role} role.')
        found[role] = reader
    return found


def _record_files(by_role: dict, reader_options: dict) -> dict:
    """File key -> ``FileIdentity`` for every file each distinct reader will open."""
    owner: dict = {}
    paths: dict = {}
    listed: set = set()
    for reader in by_role.values():
        if reader.id in listed:
            continue  # one reader serving several roles lists its files once, even when it names none
        listed.add(reader.id)
        for key, path in reader.files(reader_options).items():
            if key in owner:
                raise ValueError(f'Readers {owner[key].id} and {reader.id} both name the file key {key}.')
            owner[key] = reader
            paths[key] = path
    roles = {key: file_role(owner[key], key) for key in paths}
    return {key: record(roles[key], str(path), hash_contents=roles[key] != 'raw')
            for key, path in paths.items()}


def _refuse_existing(manifest: dict, day: str) -> None:
    if day in manifest['days']:
        raise ValueError(f'Day {day} already exists; a day is added once. Put a changed day in a new project.')


def add_day(project: Project, day: str, *, readers: dict, reader_options: dict, date: str, date_source: str) -> dict:
    """Make ``day``'s folder, then record its readers and file identities in ``project.json``; return the day record.

    ``readers`` maps role to reader id. A day already in ``project.json`` is refused: a day is
    added once, and a changed day goes in a new project. The folder is made under
    ``project.lock`` before the record is written, so a folder that cannot be made records
    nothing and the same call can be run again; a failed record leaves at most an empty,
    unrecorded day folder, which the next ``add_day`` reuses.
    """
    if not NAME_PATTERN.match(day):
        raise ValueError(f'Day name {day} must match [A-Za-z0-9][A-Za-z0-9_.-]*.')
    # A day already added is refused before its readers are asked for files, so a missing input or
    # an unknown reader cannot hide the refusal; change() checks again under project.lock.
    _refuse_existing(project.read(), day)
    by_role = _day_readers(readers)
    identities = _record_files(by_role, reader_options)
    day_record = new_day(date, date_source,
                         {role: f'{reader.id}@{reader.version}' for role, reader in by_role.items()},
                         reader_options, {key: identity.to_dict() for key, identity in identities.items()})

    def change(manifest: dict) -> None:
        _refuse_existing(manifest, day)
        project.day_dir(day).mkdir(exist_ok=True)
        manifest['days'][day] = day_record

    project.update(change)
    return day_record
