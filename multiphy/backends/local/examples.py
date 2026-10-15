# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: labeled same/different examples pooled over every project day (functional-likelihood Task 5) | date: 2026-09-27]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: module docstring names the local correlations on the threshold neighbourhoods (threshold neighbourhood) | date: 2026-10-04]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: module docstring names the peak-to-peak correlations footprint_r and footprint_r_local (event-optional Task 1) | date: 2026-10-09]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: module docstring names the same-unit waveform jitter (event-optional Task 7) | date: 2026-10-10]
"""Labeled examples from every project day: one same/different pile per project.

A same example is a unit against itself: half 0 of a trial split against half 1 of the same
split, its waveform properties read with half 1 shifted by one seeded draw in
``±WAVEFORM_JITTER_UM`` µm (same-day halves sit on the same channels). A different example is
two units recorded on the same day within ``RADIUS_UM`` of each other, which are certainly
different neurons. Examples are pooled over every day of the
project and every trial split; a project with fewer than ``MIN_NEIGHBOUR_PAIRS`` distinct
neighbour pairs is refused rather than fitted on a thin pile. ``align_channels`` puts one
day's waveform on another day's channels, so waveform properties compare like with like
across a depth offset.

``waveform_r`` and ``footprint_r`` compare every channel, unchanged. ``waveform_r_local`` and
``footprint_r_local`` compare only the union of both units' threshold channel neighbourhoods
(``neighbourhood.unit_neighbourhood``, ``neighbourhood.pair_channels``) in ``x``'s frame: ``waveform_r_local`` on the
``neighbourhood.spike_window`` samples, ``footprint_r_local`` on the peak-to-peak over the
whole snippet.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
from __future__ import annotations

import itertools
from typing import Sequence

import numpy as np
import pandas as pd

from . import evidence as ev
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair channel union and spike window for the local properties (local-neighbourhood Task 2) | date: 2026-10-02]
from .neighbourhood import pair_channels, spike_window
# [AGENT EDIT END]

RADIUS_UM = 80.0
MIN_NEIGHBOUR_PAIRS = 30
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: same-unit rows read half 1 shifted by up to this many µm (event-optional Task 7) | date: 2026-10-10]
WAVEFORM_JITTER_UM = 20.0
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: PROPERTIES gains the local-neighbourhood waveform and template correlations (local-neighbourhood Task 2) | date: 2026-10-02]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the peak-to-peak correlations are named footprint_r and footprint_r_local (event-optional Task 1) | date: 2026-10-09]
PROPERTIES = ('fingerprint_r', 'd_log_spont', 'd_log_rate_session', 'd_log_rate_shared', 'isi_cosine', 'acg_cosine',
              'd_spike_width_ms', 'waveform_r', 'footprint_r', 'waveform_r_local', 'footprint_r_local')
# [AGENT EDIT END]
# [AGENT EDIT END]


def align_channels(values: np.ndarray, geometry_from: np.ndarray, geometry_to: np.ndarray,
                   offset_um: float) -> np.ndarray:
    """``values`` (sample, channel) on ``geometry_from``, interpolated onto ``geometry_to``'s channels.

    ``offset_um`` is added to ``geometry_from`` depths to put them in ``geometry_to``'s frame,
    so channel ``(x, y)`` of ``geometry_to`` reads column ``x`` of ``geometry_from`` at depth
    ``y - offset_um``. A depth outside that column's range is NaN.
    """
    values = np.asarray(values, dtype=float)
    geometry_from = np.asarray(geometry_from, dtype=float)
    geometry_to = np.asarray(geometry_to, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(geometry_from):
        raise ValueError(f'values of shape {values.shape} do not follow the {len(geometry_from)} source channels')
    aligned = np.full((values.shape[0], len(geometry_to)), np.nan)
    for x in np.unique(geometry_to[:, 0]):
        source = np.flatnonzero(geometry_from[:, 0] == x)
        if not len(source):
            raise ValueError('channel columns differ')
        source = source[np.argsort(geometry_from[source, 1], kind='stable')]
        depths = geometry_from[source, 1]
        target = np.flatnonzero(geometry_to[:, 0] == x)
        at = geometry_to[target, 1] - offset_um
        column = values[:, source]
        if np.isfinite(column).all():
            # np.interp of each source channel's unit vector gives the interpolation weights, so one
            # product interpolates every sample; at a channel's own depth the weights are exactly 1 and 0
            weights = np.stack([np.interp(at, depths, unit) for unit in np.eye(len(source))], axis=1)
            aligned[:, target] = column @ weights.T
            aligned[:, target[(at < depths[0]) | (at > depths[-1])]] = np.nan
        else:
            for sample in range(values.shape[0]):
                aligned[sample, target] = np.interp(at, depths, column[sample], left=np.nan, right=np.nan)
    return aligned


def _fingerprint(half: dict, keys: Sequence[str], split: int) -> np.ndarray:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no shared condition gives an empty fingerprint, not an error (event-optional Task 5) | date: 2026-10-09]
    if not len(keys):
        return np.array([])
    # [AGENT EDIT END]
    missing = [k for k in keys if k not in half['fingerprints']]
    if missing:
        raise ValueError(f'conditions {missing} are not measured on split {split}; pass keys every unit measures')
    return np.concatenate([half['fingerprints'][k] for k in keys])


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_properties docstring names the local correlations on the threshold neighbourhoods (threshold neighbourhood) | date: 2026-10-04]
def pair_properties(x: dict, y: dict, keys: Sequence[str], split: int, *, offset_um: float = 0.0) -> dict:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring says empty keys leave the event properties NaN (event-optional Task 5) | date: 2026-10-09]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the peak-to-peak correlations footprint_r and footprint_r_local (event-optional Task 1) | date: 2026-10-09]
    """``PROPERTIES`` of half 0 of ``x`` against half 1 of ``y`` on trial split ``split``.

    Empty ``keys`` (a run without events) leave ``fingerprint_r`` and ``d_log_rate_shared`` NaN.
    ``offset_um`` puts ``y``'s depths in ``x``'s frame for the waveform properties; these are
    NaN unless both units carry waveform halves. ``waveform_r`` and ``footprint_r`` read every
    channel. ``waveform_r_local`` and ``footprint_r_local`` read only the union of both units'
    threshold channel neighbourhoods (``pair_channels``) in ``x``'s frame: ``waveform_r_local`` on the
    ``spike_window`` samples only, ``footprint_r_local`` on the peak-to-peak over the whole
    snippet. Fewer than 5 finite values leave a local property NaN.
    """
    # [AGENT EDIT END]
    # [AGENT EDIT END]
# [AGENT EDIT END]
    hx, hy = x['splits'][split][0], y['splits'][split][1]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: empty keys give NaN event properties (event-optional Task 5) | date: 2026-10-09]
    events = bool(len(keys))
    properties = dict(
        fingerprint_r=ev.pearson(_fingerprint(hx, keys, split), _fingerprint(hy, keys, split)) if events else np.nan,
        d_log_spont=ev.dlog(hx['spont'], hy['spont']),
        d_log_rate_session=ev.dlog(hx['rate'], hy['rate']),
        d_log_rate_shared=ev.dlog(np.mean([hx['per_stim'][k] for k in keys]),
                                  np.mean([hy['per_stim'][k] for k in keys])) if events else np.nan,
        # [AGENT EDIT END]
        isi_cosine=ev.cosine(hx['isi'], hy['isi']),
        acg_cosine=ev.cosine(hx['acg'], hy['acg']),
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: local properties start NaN like the other waveform properties (local-neighbourhood Task 2) | date: 2026-10-02]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the footprint properties start NaN under their footprint names (event-optional Task 1) | date: 2026-10-09]
        d_spike_width_ms=np.nan, waveform_r=np.nan, footprint_r=np.nan, waveform_r_local=np.nan,
        footprint_r_local=np.nan)
        # [AGENT EDIT END]
        # [AGENT EDIT END]
    if x.get('waveform_halves') is not None and y.get('waveform_halves') is not None:
        wx = np.asarray(x['waveform_halves'], dtype=float)[..., 0]
        wy = align_channels(np.asarray(y['waveform_halves'], dtype=float)[..., 1], y['geometry'], x['geometry'],
                            offset_um)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the all-channel peak-to-peak correlation is named footprint_r (event-optional Task 1) | date: 2026-10-09]
        properties.update(waveform_r=ev.pearson(wx, wy),
                          footprint_r=ev.pearson(np.ptp(wx, axis=0), np.ptp(wy, axis=0)),
                          d_spike_width_ms=abs(x['sw_halves'][0] - y['sw_halves'][1]))
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: waveform and template correlations on the pair's threshold channel union (threshold neighbourhood) | date: 2026-10-04]
        channels = pair_channels(x['neighbourhood'], y['neighbourhood'], x['geometry'], y['geometry'], offset_um)
        window = spike_window(wx.shape[0])
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the peak-channel peak-to-peak correlation is named footprint_r_local (event-optional Task 1) | date: 2026-10-09]
        properties.update(
            waveform_r_local=ev.pearson(wx[np.ix_(window, channels)], wy[np.ix_(window, channels)]),
            footprint_r_local=ev.pearson(np.ptp(wx[:, channels], axis=0), np.ptp(wy[:, channels], axis=0)))
        # [AGENT EDIT END]
        # [AGENT EDIT END]
    return {p: properties[p] for p in PROPERTIES}


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: labeled_examples takes jitter_um and seed; its docstring names the same-unit jitter (event-optional Task 7) | date: 2026-10-10]
def labeled_examples(observations: dict, session_of: dict, keys: Sequence[str], *, radius_um: float = RADIUS_UM,
                     min_pairs: int = MIN_NEIGHBOUR_PAIRS, jitter_um: float = WAVEFORM_JITTER_UM,
                     seed: int = 0) -> pd.DataFrame:
    """Same and different examples on every split, pooled over every day of the project.

    ``observations`` maps cellid -> ``unit_observation``; ``session_of`` maps cellid -> its
    session. Same rows: every unit against itself, each row's waveform properties read with
    half 1 shifted by its own uniform draw in ``±jitter_um`` µm (``np.random.default_rng(seed)``,
    in row order), since same-day halves sit on the same channels. Different rows: every
    distinct unordered same-session pair within ``radius_um`` of depth, unshifted. Fewer than
    ``min_pairs`` such pairs raises before any property is computed.
    """
# [AGENT EDIT END]
    missing = sorted(c for c in observations if c not in session_of)
    if missing:
        raise ValueError(f'{len(missing)} units have no session: {missing[:5]}; give session_of every cellid')
    cellids = sorted(observations)
    days = sorted({session_of[c] for c in cellids})
    pairs = [(u, v) for u, v in itertools.combinations(cellids, 2)
             if session_of[u] == session_of[v]
             and abs(observations[u]['depth'] - observations[v]['depth']) <= radius_um]
    n = len(pairs)
    if n < min_pairs:
        raise ValueError(f'{n} distinct same-day neighbour pairs within {radius_um:g} µm across days {days}; '
                         f'at least {min_pairs} are needed. Rerun with a wider radius.')
    # each day keeps its own trial splits; the units of one day share them, so a pair (always
    # same-day) is compared on its day's splits
    n_splits = {}
    for c in cellids:
        n_splits.setdefault(session_of[c], set()).add(len(observations[c]['splits']))
    uneven = {d: sorted(n) for d, n in n_splits.items() if len(n) != 1}
    if uneven:
        raise ValueError(f'units of one day carry different numbers of trial splits {uneven}; '
                         'observe every unit of a day on that day\'s splits')
    n_splits = {d: n.pop() for d, n in n_splits.items()}
    # a day with no splits would count toward the minimum yet add no row; refuse it
    empty = sorted(d for d, n in n_splits.items() if n == 0)
    if empty:
        raise ValueError(f'days {empty} carry no trial splits (0 per unit); observe every unit on at least '
                         'one split, or leave those days out')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one jitter draw per same-unit row, drawn before the rows are built; different-unit rows at offset 0.0 (event-optional Task 7) | date: 2026-10-10]
    n_same_rows = sum(n_splits[session_of[c]] for c in cellids)
    draws = iter(np.random.default_rng(seed).uniform(-jitter_um, jitter_um, size=n_same_rows))
    rows = []
    for same, compared in ((True, [(c, c) for c in cellids]), (False, pairs)):
        for u, v in compared:
            for split in range(n_splits[session_of[u]]):
                offset_um = next(draws) if same else 0.0
                rows.append(dict(cellid_a=u, cellid_b=v, session=session_of[u], split=split, same_unit=same,
                                 **pair_properties(observations[u], observations[v], keys, split,
                                                   offset_um=offset_um)))
    # [AGENT EDIT END]
    return pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'session', 'split', 'same_unit', *PROPERTIES])


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fit_ensemble fits the given streams; the references hold ALL_PROPERTIES plus every stream property (event-optional Task 6) | date: 2026-10-10]
def fit_ensemble(examples: pd.DataFrame, streams: dict) -> tuple:
    """``(ev.Ensemble(same_ref, diff_ref, streams), same_ref, diff_ref)`` from the labeled examples.

    ``streams`` is a stream dict (``ev.run_streams``, or ``ev.STREAMS``). The same and different
    references hold ``ev.ALL_PROPERTIES`` plus every property of ``streams``.
    """
    same = examples.same_unit.astype(bool)
    properties = ev._reference_properties(streams)
    same_ref = {p: examples.loc[same, p].tolist() for p in properties}
    diff_ref = {p: examples.loc[~same, p].tolist() for p in properties}
    return ev.Ensemble(same_ref, diff_ref, streams), same_ref, diff_ref
# [AGENT EDIT END]
# [AGENT EDIT END]
