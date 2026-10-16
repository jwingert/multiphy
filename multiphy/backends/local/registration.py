"""Match-driven depth registration: one rigid depth offset per day, measured from confident matches.

Offsets follow ``ev.register_depths``: ``offsets[b] - offsets[a]`` is added to day ``b``'s depths
to put them in day ``a``'s frame, and the offsets' mean is pinned at 0. ``profile_offsets`` seeds
them from where activity sits on the shank (identity-free). ``match_offsets`` then refines them
from the units themselves: within a depth window around the current offset, mutual best pairs
scoring above a null threshold are matches, and the median depth difference of a day pair's
matches is its measurement. A day pair with too few matches falls back to its profile
measurement, or to none, and the diagnostics name which source each pair used.
``functional_offsets`` runs the whole chain with the functional ensemble's global score.
"""
from __future__ import annotations

import itertools
from typing import Callable, Sequence

import numpy as np

from . import evidence as ev
from .examples import pair_properties

METHODS = ('none', 'profile', 'matches')


def _sessions(obs_by_session: dict) -> list:
    sessions = list(obs_by_session)
    if not sessions:
        raise ValueError('0 sessions to register; at least 1 is needed. Pass every day of the project')
    return sessions


def _usable(measurement) -> bool:
    """A measurement ``ev.register_depths`` actually solves from: finite value, finite positive weight."""
    return (measurement is not None and bool(np.isfinite(measurement[2])) and bool(np.isfinite(measurement[3]))
            and measurement[3] > 0)


def _groups(sessions: list, measurements: list) -> list:
    """Sessions linked by usable measurements, each group in ``sessions`` order.

    ``ev.register_depths`` pins each group's mean at 0 on its own, so offsets are comparable only
    within a group; more than one group means some days are unregistered relative to the rest.
    """
    parent = {s: s for s in sessions}

    def find(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for a, b, _, _ in (m for m in measurements if _usable(m)):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    groups: dict = {}
    for s in sessions:
        groups.setdefault(find(s), []).append(s)
    return list(groups.values())


def profile_offsets(obs_by_session: dict, *, max_shift_um: float = 400.0) -> tuple:
    """``(offsets, measurements)`` from the depth-amplitude profile of every session pair.

    Each measurement is ``(a, b, m, max(cc, 0.0))`` from ``ev.estimate_offset`` on the units'
    ``depth`` and ``amp``; the offsets are their ``ev.register_depths`` solution.
    """
    sessions = _sessions(obs_by_session)
    max_shift = int(round(max_shift_um / ev.UM_PER_CHAN))
    measurements = []
    for a, b in itertools.combinations(sessions, 2):
        units_a, units_b = obs_by_session[a].values(), obs_by_session[b].values()
        m, cc = ev.estimate_offset([u['depth'] for u in units_a], [u['amp'] for u in units_a],
                                   [u['depth'] for u in units_b], [u['amp'] for u in units_b], max_shift=max_shift)
        measurements.append((a, b, m, max(cc, 0.0)))
    return ev.register_depths(measurements, days=sessions)[0], measurements


def _seed_lookup(seed_measurements: list) -> dict:
    lookup = {}
    for a, b, m, w in seed_measurements:
        lookup[(a, b)] = (a, b, float(m), float(w))
        lookup.setdefault((b, a), (b, a, -float(m), float(w)))
    return lookup


def _mutual_best(scores: np.ndarray, null_threshold: float) -> list:
    """(row, column) pairs that are each other's best and score above ``null_threshold``."""
    if not scores.size:
        return []
    best_col = np.argmax(scores, axis=1)
    best_row = np.argmax(scores, axis=0)
    return [(i, j) for i, j in enumerate(best_col)
            if best_row[j] == i and np.isfinite(scores[i, j]) and scores[i, j] > null_threshold]


def match_offsets(obs_by_session: dict, score_pair: Callable[..., float], null_threshold: float,
                  seed_offsets: dict, seed_measurements: list, *, windows_um: tuple = (400.0, 80.0),
                  tol_um: float = 5.0, max_iter: int = 5, min_matches: int = 3, aligned: bool = False) -> tuple:
    """``(offsets, diagnostics)`` refined from mutual best matches, starting from ``seed_offsets``.

    For each window in ``windows_um`` (coarse to fine), up to ``max_iter`` iterations: the
    candidates of a session pair ``(a, b)`` are unit pairs with
    ``abs(depth_a - (depth_b + offsets[b] - offsets[a])) <= window``; a match is a mutual best
    candidate pair with ``score_pair(cellid_a, cellid_b) > null_threshold``. With at least
    ``min_matches`` matches the pair's measurement is ``(a, b, median(depth_a - depth_b),
    n_matches)``; otherwise its ``seed_measurements`` entry (source ``'profile'``) or nothing
    (source ``'none'``). A window ends when no offset moved by ``tol_um`` or more.

    A unit pair is scored once. With ``aligned`` it is scored once per relative offset tried
    instead: ``score_pair(cellid_a, cellid_b, rel)`` with ``rel = offsets[b] - offsets[a]``, the
    shift that puts day ``b``'s depths in day ``a``'s frame, cached by ``round(rel, 3)``.
    """
    if not windows_um or max_iter < 1:
        raise ValueError(f'match_offsets needs at least one window and one iteration; got windows_um={windows_um!r}, max_iter={max_iter}')
    sessions = _sessions(obs_by_session)
    missing = [s for s in sessions if s not in seed_offsets]
    if missing:
        raise ValueError(f'{len(missing)} sessions have no seed offset: {missing}; give seed_offsets every session')
    if min_matches < 1:
        raise ValueError(f'min_matches is {min_matches}; at least 1 match must back a measurement')
    if not np.isfinite(null_threshold):
        raise ValueError(f'null_threshold is {null_threshold}; it must be finite. Recompute it as a quantile of the '
                         'finite different-unit reference scores')
    seeds = _seed_lookup(seed_measurements)
    offsets = {s: float(seed_offsets[s]) for s in sessions}
    cache: dict = {}
    cells = {s: list(obs_by_session[s]) for s in sessions}
    depth = {s: np.array([obs_by_session[s][c]['depth'] for c in cells[s]], dtype=float) for s in sessions}
    iterations, converged, pairs = 0, False, []
    for window in windows_um:
        converged = False
        for _ in range(max_iter):
            iterations += 1
            measurements, pairs = [], []
            for a, b in itertools.combinations(sessions, 2):
                rel = offsets[b] - offsets[a]
                near = np.abs(depth[a][:, None] - (depth[b][None, :] + rel)) <= window
                scores = np.full(near.shape, -np.inf)
                for i, j in zip(*np.nonzero(near)):
                    if aligned:
                        key = (cells[a][i], cells[b][j], round(rel, 3))
                        if key not in cache:
                            cache[key] = float(score_pair(cells[a][i], cells[b][j], rel))
                    else:
                        key = (cells[a][i], cells[b][j])
                        if key not in cache:
                            cache[key] = float(score_pair(*key))
                    scores[i, j] = cache[key]
                bad = [(cells[a][i], cells[b][j]) for i, j in zip(*np.nonzero(near & ~np.isfinite(scores)))]
                if bad:
                    raise ValueError(f'{len(bad)} non-finite pair scores between {a} and {b}, e.g. {bad[:3]}; '
                                     'score_pair must return a finite score for every candidate pair. Fix the scorer')
                matches = _mutual_best(scores, null_threshold)
                if len(matches) >= min_matches:
                    m = float(np.median([depth[a][i] - depth[b][j] for i, j in matches]))
                    measurement, source = (a, b, m, float(len(matches))), 'matches'
                elif _usable(seeds.get((a, b))):
                    measurement, source = seeds[(a, b)], 'profile'
                else:
                    measurement, source = None, 'none'
                if measurement is not None:
                    measurements.append(measurement)
                pairs.append(dict(session_a=a, session_b=b, source=source, n_matches=len(matches),
                                  offset_um=np.nan if measurement is None else measurement[2]))
            solved = ev.register_depths(measurements, days=sessions)[0]
            moved = max(abs(solved[s] - offsets[s]) for s in sessions)
            offsets = {s: float(solved[s]) for s in sessions}
            if moved < tol_um:
                converged = True
                break
    diagnostics = dict(pairs=pairs, iterations=iterations, converged=converged, null_threshold=float(null_threshold),
                       groups=_groups(sessions, measurements))
    return offsets, diagnostics


def functional_offsets(obs_by_session: dict, ensemble: ev.Ensemble, diff_ref: dict, keys: Sequence[str], *,
                       method: str = 'matches', null_quantile: float = 0.99, aligned: bool = False) -> tuple:
    """``(offsets, diagnostics)`` for ``method`` ``'none'``, ``'profile'`` or ``'matches'``.

    The null threshold is the ``null_quantile`` of the ensemble's finite ``s_global`` scores on
    the different-unit references ``diff_ref``; a pair's score is the ``s_global`` of its
    ``pair_properties`` on split 0. With ``aligned``, ``match_offsets`` scores each pair at every
    relative offset it tries, passed to ``pair_properties`` as ``offset_um``.
    """
    if method not in METHODS:
        raise ValueError(f'registration method {method!r} is not one of {METHODS}')
    sessions = _sessions(obs_by_session)
    reference = np.asarray(ensemble.reference_scores(diff_ref)['s_global'], dtype=float)
    reference = reference[np.isfinite(reference)]
    if not len(reference):
        raise ValueError('0 finite different-unit reference scores; the null threshold needs labeled different '
                         'examples. Refit the ensemble on labeled_examples of this project')
    null = float(np.quantile(reference, null_quantile))
    observations = {c: o for s in sessions for c, o in obs_by_session[s].items()}

    def score_pair(a: str, b: str, rel: float = 0.0) -> float:
        # unaligned, match_offsets passes only the two cellids: rel stays pair_properties' own default 0.0
        return float(np.asarray(ensemble.score(pair_properties(observations[a], observations[b], keys, 0,
                                                               offset_um=rel))['s_global']).ravel()[0])

    if method == 'none':
        offsets = {s: 0.0 for s in sessions}
        pairs = [dict(session_a=a, session_b=b, source='none', n_matches=0, offset_um=np.nan)
                 for a, b in itertools.combinations(sessions, 2)]
        diagnostics = dict(pairs=pairs, iterations=0, converged=True, null_threshold=null,
                           groups=[[s] for s in sessions])
    else:
        seed_offsets, seed_measurements = profile_offsets(obs_by_session)
        if method == 'profile':
            offsets = seed_offsets
            pairs = [dict(session_a=a, session_b=b, source='profile', n_matches=0, offset_um=m)
                     if _usable((a, b, m, w)) else
                     dict(session_a=a, session_b=b, source='none', n_matches=0, offset_um=np.nan)
                     for a, b, m, w in seed_measurements]
            diagnostics = dict(pairs=pairs, iterations=0, converged=True, null_threshold=null,
                               groups=_groups(sessions, seed_measurements))
        else:
            offsets, diagnostics = match_offsets(obs_by_session, score_pair, null, seed_offsets, seed_measurements,
                                                 aligned=aligned)
    diagnostics.update(method=method, null_quantile=null_quantile)
    return offsets, diagnostics
