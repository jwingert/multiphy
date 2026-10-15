# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: shared upstream-package helpers for the published-method adapters (Task 9) | date: 2026-09-22]
"""Helpers shared by the UnitMatch, DANT and DeepUnitMatch adapters.

Moved from ``external_backends.py``: package import with a BackendUnavailable on failure,
installed version, source hashing, raw-waveform bundle preparation and the isolated
DeepUnitMatch import. ``availability`` probes a package with ``find_spec`` only, so listing
backends never imports an upstream numerical stack.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: imports for recording upstream calls during one call (drift-view Task 5) | date: 2026-10-08]
import contextlib
import threading
from dataclasses import dataclass
from typing import Iterator, Optional
# [AGENT EDIT END]
import importlib
import importlib.metadata
import importlib.util
import sys

import numpy as np

from ..contract.backend import Availability
from ..contract.errors import BackendUnavailable
from ..contract.inputs import RawWaveformBundle
from ..contract.results import single_units

UNITMATCH_SOURCE = 'https://github.com/EnnyvanBeest/UnitMatch'
UNITMATCH_REVISION = '95b6fed5db84d12f67487b92d07f0a2664a6122f'
DANT_SOURCE = 'https://github.com/jiumao2/pyDANT'
DANT_REVISION = '741240511e6a4ed9315b2c80d71ab5cfe4c588eb'
DEEPUNITMATCH_PAPER_REVISION = '18be056f6b7aceaecd15c2a8af560b2f05693656'


def module(name):
    try:
        return importlib.import_module(name)
    except (ImportError, ModuleNotFoundError) as exc:
        raise BackendUnavailable(f'{name} or its dependencies are unavailable: {exc}') from exc


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return 'source-checkout-unversioned'


def source_info(module):
    path = Path(module.__file__).resolve()
    return {'module': module.__name__, 'path': str(path),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def prepare_bundle(units, bundle):
    if not isinstance(bundle, RawWaveformBundle):
        raise BackendUnavailable('Prepare a RawWaveformBundle with independent temporal-half '
                                 'raw multichannel waveforms; cached sorting templates are insufficient.')
    eligible, excluded = single_units(units)
    ids = tuple(map(str, bundle.cellids))
    if len(ids) != len(set(ids)) or len(bundle.session_ids) != len(ids):
        raise ValueError('Waveform cell IDs must be unique with one session ID per unit')
    waves = np.asarray(bundle.waveforms, dtype=float)
    if waves.ndim != 4 or waves.shape[0] != len(ids) or waves.shape[3] != 2:
        raise ValueError('Raw waveforms require shape (unit, sample, channel, 2 temporal halves)')
    if not bundle.provenance.get('raw_source') or not bundle.provenance.get('split_method'):
        raise ValueError('Raw waveform provenance requires raw_source and split_method')
    if not np.isfinite(waves).all():
        raise ValueError('Raw waveform halves must be finite; missing halves cannot be duplicated')
    if not set(eligible.cellid) <= set(ids):
        raise ValueError('Every eligible single unit requires an explicitly mapped waveform')
    index = {u: i for i, u in enumerate(ids)}
    # [AGENT EDIT START | agent: codex | user: wingertj | reason: chronological native grouping for real recording dates while preserving benchmark order | date: 2026-09-18]
    if 'session_date' in eligible:
        from datetime import date
        if eligible.session_date.isna().any() or eligible.groupby('session_id').session_date.nunique().gt(1).any():
            raise ValueError('Each source session must have one explicit recording date')
        for value in eligible.session_date:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError('Recording dates must use YYYY-MM-DD format')
        ordering = ['session_date', 'session_id', 'cellid']
    else:
        ordering = ['session_id', 'cellid']
    eligible = eligible.sort_values(ordering).reset_index(drop=True)
    # [AGENT EDIT END]
    order = np.array([index[u] for u in eligible.cellid], dtype=int)
    sessions = list(dict.fromkeys(eligible.session_id))
    if len(sessions) < 2:
        raise ValueError('External matching requires at least two nonempty single-unit sessions')
    for row, i in zip(eligible.itertuples(), order):
        if str(bundle.session_ids[i]) != str(row.session_id):
            raise ValueError(f'Waveform session disagrees with unit metadata: {row.cellid}')
    geometry = []
    for session in sessions:
        if session not in bundle.channel_positions:
            raise ValueError(f'Missing measured channel geometry for session {session}')
        g = np.asarray(bundle.channel_positions[session], dtype=float)
        if g.shape != (waves.shape[2], 2) or not np.isfinite(g).all():
            raise ValueError('Channel coordinates must be finite (channel, 2), in micrometers')
        geometry.append(g.copy())
    sid = np.array([sessions.index(s) for s in eligible.session_id], dtype=int)
    switch = np.r_[0, np.cumsum(np.bincount(sid))]
    return eligible, excluded, order, waves[order].copy(), geometry, sid, switch


def deep_module(name):
    # Upstream imports a top-level `utils` package. Refuse to collide with a
    # different application's utils rather than patching global module aliases.
    existing = sys.modules.get('utils')
    if existing is not None and 'DeepUnitMatch' not in str(getattr(existing, '__path__', '')):
        raise BackendUnavailable('DeepUnitMatch needs an isolated process: another utils package is loaded')
    spec = importlib.util.find_spec('DeepUnitMatch')
    if spec is None:
        raise BackendUnavailable('The official DeepUnitMatch source/package is unavailable')
    root = str(next(iter(spec.submodule_search_locations)))
    if root not in sys.path:
        sys.path.insert(0, root)
    return module('DeepUnitMatch.' + name)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: record named upstream functions during one call and restore them after it, also after an error (drift-view Task 5) | date: 2026-10-08]
@dataclass(frozen=True)
class Call:
    """One completed call of a recorded upstream function; ``parent`` is the recorded caller."""
    name: str
    args: tuple
    kwargs: dict
    result: object
    parent: Optional[str]


_RECORDING_LOCK = threading.RLock()


@contextlib.contextmanager
def recording(module, *names: str) -> Iterator[list]:
    """Swap each named function of ``module`` for a recorder while the block runs.

    Yields the list of ``Call`` in completion order (a callee before its caller); a call that
    raises is not appended. Every original object is put back when the block ends, also after
    an error. A name that is not a callable attribute raises ``BackendUnavailable`` before
    anything is swapped.
    """
    for name in names:
        if not callable(getattr(module, name, None)):
            raise BackendUnavailable(f'{module.__name__}.{name} is missing; drift offsets cannot be captured')
    with _RECORDING_LOCK:
        originals = {name: getattr(module, name) for name in names}
        calls, stack = [], []

        def recorder(name, original):
            def record(*args, **kwargs):
                parent = stack[-1] if stack else None
                stack.append(name)
                try:
                    result = original(*args, **kwargs)
                finally:
                    stack.pop()
                calls.append(Call(name, args, kwargs, result, parent))
                return result
            return record

        try:
            for name, original in originals.items():
                setattr(module, name, recorder(name, original))
            yield calls
        finally:
            for name, original in originals.items():
                setattr(module, name, original)
# [AGENT EDIT END]


def availability(package, *, extra_reason=None):
    """Availability without importing the package's numerical stack."""
    try:
        found = importlib.util.find_spec(package) is not None
    except (ValueError, ImportError):
        found = False
    if not found:
        return Availability(False, None, f'{package} is not importable in this interpreter')
    if extra_reason:
        return Availability(False, version(package), extra_reason)
    return Availability(True, version(package), None)
# [AGENT EDIT END]
