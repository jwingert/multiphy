# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: reader protocol, DaySource, reader registry and read_day (multiphy project folder Task 6; Task 7: built-ins add phy and binary; Task 7 fix round 1: DaySource.channel_map) | date: 2026-09-25]
"""The reader protocol, ``DaySource``, the reader registry and ``read_day``.

A reader turns one day's original files into a ``DaySource``; it never reads voltage. Each
reader serves one or more of the roles ``sort``, ``events`` and ``raw``. ``read_day`` runs a
day's readers in the order sort, raw, events: the sort reader starts the day, and each later
reader returns a new ``DaySource`` built from the one before.

Readers are found by id. The built-in readers in ``_BUILTIN_MODULES`` register on the first
registry use. Other packages declare ``[project.entry-points."multiphy.sources"]
name = "pkg.module:factory"``, where ``factory()`` returns a reader. Entry points load once,
on the first listing or the first unknown id; one that fails to load is recorded in
``READER_ERRORS`` and skipped. This module imports no lab module.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import importlib.metadata
from typing import Optional, Protocol

import numpy as np

from ..project.manifest import ROLES

ENTRY_POINT_GROUP = 'multiphy.sources'
_BUILTIN_MODULES = ('nems_lbhb.unit_match.sources.events_csv', 'nems_lbhb.unit_match.sources.phy',
                    'nems_lbhb.unit_match.sources.binary')
_RUN_ORDER = ('sort', 'raw', 'events')  # the sort reader starts the day; events may extend its sample count

_READERS: dict = {}
READER_ERRORS: dict = {}  # entry-point name -> why it could not be loaded
_BUILTINS_LOADED = False
_ENTRY_POINTS_LOADED = False


def _empty_windows() -> np.ndarray:
    return np.zeros((0, 2), dtype=np.int64)


@dataclass(frozen=True)
class DaySource:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: raw may hold segments in place of one path (Plan C Task 2) | date: 2026-10-01]
    """One day as its readers found it; every time is a 0-based sample on the raw clock.

    ``analysis_window`` is ``(start, stop)``, stop exclusive. ``spikes`` maps unit id to sorted
    int64 samples and ``su_type`` maps unit id to ``'SU'``. ``geometry`` is ``(n_channels, 2)``
    in µm. ``clock`` holds ``sample_index_origin`` (``'zero_based_voltage_row'``), ``method``
    and ``n_samples_from`` (``'last_spike'``, ``'raw_size'``, ``'events'`` or ``'clock'``).
    ``templates`` maps unit id to a ``(n_template_samples, n_channels)`` array, or is ``None``.
    ``trials`` and ``baseline_windows`` are ``(n, 2)`` int64 ``[start, stop)`` windows;
    ``stimuli`` holds ``{'stimulus', 'start', 'stop', 'trial'}`` dicts. ``raw`` holds the
    ``binary`` reader's parameters, or is ``None``. ``extras`` holds optional per-unit arrays
    that no backend reads. ``channel_map`` is the int64 raw-file column of each geometry row,
    or ``None`` when the sort names none; ``binary`` accepts only the identity map.
    ``raw`` may hold ``'segments'``, a list of ``{'path': str, 'n_samples': int}`` in clock
    order, in place of ``'path'``: the recordings of one day on one clock, read as one file.
    """
    # [AGENT EDIT END]
    sample_rate: float
    n_samples: int
    analysis_window: tuple
    spikes: dict
    su_type: dict
    geometry: np.ndarray
    clock: dict
    provenance: dict
    templates: Optional[dict] = None
    trials: np.ndarray = field(default_factory=_empty_windows)
    stimuli: tuple = ()
    baseline_windows: np.ndarray = field(default_factory=_empty_windows)
    raw: Optional[dict] = None
    extras: dict = field(default_factory=dict)
    channel_map: Optional[np.ndarray] = None


class SourceReader(Protocol):
    """Turns one day's files into a ``DaySource``. A reader never reads voltage."""
    id: str
    version: int
    roles: frozenset  # a subset of ROLES

    def files(self, options: dict) -> dict:
        """File key -> ``Path`` of every file ``read`` will open."""

    def read(self, files: dict, options: dict, day: Optional[DaySource] = None) -> DaySource:
        """Read ``files``; a reader after the sort reader builds on ``day``."""


def _add(reader) -> None:
    if reader.id in _READERS:
        raise ValueError(f'Reader {reader.id} is already registered.')
    _READERS[reader.id] = reader


def _load_builtins() -> None:
    """Import and register the built-in readers once per reset; never swallows an ImportError."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    for name in _BUILTIN_MODULES:
        reader = importlib.import_module(name).READER
        if reader.id not in _READERS:  # present only when an earlier load stopped part way
            _add(reader)
    _BUILTINS_LOADED = True


def register_reader(reader) -> None:
    """Register ``reader`` under ``reader.id``; an id already taken, built-in or not, is refused."""
    _load_builtins()
    _add(reader)


def _entry_points():
    eps = importlib.metadata.entry_points()
    return eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, 'select') else eps.get(ENTRY_POINT_GROUP, ())


def load_reader_entry_points() -> None:
    """Register every ``multiphy.sources`` entry point once; a failing one is recorded, not raised."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    _load_builtins()
    for ep in _entry_points():
        try:
            reader = ep.load()()
            if reader.id not in _READERS:
                _add(reader)
        except Exception as exc:  # one broken plugin must not hide the others or the built-ins
            READER_ERRORS[ep.name] = f'{type(exc).__name__}: {exc}'
    _ENTRY_POINTS_LOADED = True


def _strip_version(reader_id: str) -> str:
    head, sep, tail = reader_id.rpartition('@')
    return head if sep and tail.isdigit() else reader_id


def get_reader(reader_id: str):
    """The reader registered as ``reader_id``; an ``@version`` suffix is ignored."""
    rid = _strip_version(reader_id)
    _load_builtins()
    if rid not in _READERS:
        load_reader_entry_points()
    if rid not in _READERS:
        raise KeyError(f'Unknown reader {reader_id!r}; known: {sorted(_READERS)}')
    return _READERS[rid]


def reader_ids() -> tuple:
    """Every registered reader id, built-ins and entry points included, sorted."""
    _load_builtins()
    load_reader_entry_points()
    return tuple(sorted(_READERS))


def clear_readers_for_tests() -> None:
    """Empty the registry; the next lookup loads the built-ins and entry points again."""
    global _BUILTINS_LOADED, _ENTRY_POINTS_LOADED
    _READERS.clear()
    READER_ERRORS.clear()
    _BUILTINS_LOADED = False
    _ENTRY_POINTS_LOADED = False


def file_role(reader, key: str) -> str:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: key raw or raw_<k> is a raw-role file (Plan C Task 2) | date: 2026-10-01]
    """The role a file of ``reader`` is recorded under.

    ``'raw'`` for key ``'raw'`` or any key starting ``'raw_'`` (one raw segment each), else the
    reader's first role. A raw-role file is never hashed when its day is added.
    """
    if key == 'raw' or key.startswith('raw_'):
        return 'raw'
    # [AGENT EDIT END]
    for role in ROLES:
        if role in reader.roles:
            return role
    raise ValueError(f'Reader {reader.id} serves none of the roles {", ".join(ROLES)}.')


def read_day(readers: dict, files: dict, options: dict) -> DaySource:
    """Run a day's readers in the order sort, raw, events and return the finished ``DaySource``.

    ``readers`` maps role -> reader id (``@version`` allowed); ``files`` maps role ->
    ``{key: Path}``. A reader serving several roles runs once, at its first role, with those
    roles' files merged. The sort reader gets ``day=None``; each later reader gets the result
    so far.
    """
    unknown = sorted(set(readers) - set(ROLES))
    if unknown:
        raise ValueError(f'Unknown role {unknown[0]}; the roles are {", ".join(ROLES)}.')
    if 'sort' not in readers:
        raise ValueError('A day needs a sort reader.')
    roles_of: dict = {}
    for role in _RUN_ORDER:
        if role not in readers:
            continue
        reader = get_reader(readers[role])
        if role not in reader.roles:
            raise ValueError(f'Reader {reader.id} does not serve the {role} role.')
        roles_of.setdefault(reader.id, (reader, []))[1].append(role)
    day = None
    for reader, roles in roles_of.values():
        merged: dict = {}
        for role in roles:
            merged.update(files.get(role, {}))
        result = reader.read(merged, options, day)
        if not isinstance(result, DaySource):
            raise TypeError(f'Reader {reader.id} returned {type(result).__name__}, not a DaySource.')
        day = result
    return day
# [AGENT EDIT END]
