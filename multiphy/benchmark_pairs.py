"""Build ordinary candidate evidence from independently recomputed pseudo-units.

This module never reads source-unit labels, learns probabilities, or chooses a
unit's best match. Labels belong in the separate benchmark ground-truth table.
Each recording group is paired independently because within-sort temporal
identity does not establish identity between original recordings.
"""
import itertools

import numpy as np
import pandas as pd

from . import evidence as ev
from .backends import single_units


def _waveform_summary(row, geometry):
    waveform = np.asarray(row.get('waveform'), dtype=float)
    geometry = np.asarray(geometry, dtype=float)
    if waveform.ndim != 2 or not np.isfinite(waveform).all():
        raise ValueError('Each eligible pseudo-unit requires a freshly extracted finite (sample, channel) waveform')
    if geometry.shape != (waveform.shape[1], 2) or not np.isfinite(geometry).all():
        raise ValueError('Measured geometry must be finite (channel, 2) and align with the waveform channels')
    fs = float(row.get('sample_rate', np.nan))
    if not np.isfinite(fs) or fs <= 0:
        raise ValueError('Waveform sample_rate must be finite and positive')
    half_r = np.nan
    if 'waveform_halves' in row:
        halves = np.asarray(row['waveform_halves'], dtype=float)
        if halves.shape != waveform.shape + (2,) or not np.isfinite(halves).all():
            raise ValueError('Independent waveform halves must match (sample, channel, 2)')
        half_r = ev.pearson(halves[..., 0], halves[..., 1])
    ptp = np.ptp(waveform, axis=0)
    if not len(ptp) or not np.isfinite(ptp).all() or ptp.max() <= 0:
        raise ValueError('A zero-amplitude waveform has no measurable spatial location')
    peak = int(np.argmax(ptp))
    weights = np.where(ptp >= .25 * ptp[peak], ptp, 0.)
    centroid = np.sum(geometry * weights[:, None], axis=0) / weights.sum()
    peak_wave = waveform[:, peak]
    width = np.nan
    if len(peak_wave) >= 5:
        trough = int(np.argmin(peak_wave))
        if trough < len(peak_wave) - 2:
            positive = trough + int(np.argmax(peak_wave[trough:]))
            if positive > trough:
                width = 1000 * (positive - trough) / fs
    # Carry only measured features. In particular source_unit/source_session are
    # deliberately not copied out of the extraction result.
    return {'depth': float(centroid[1]), 'x_um': float(centroid[0]),
            'ptp': ptp, 'amp': float(ptp[peak]), 'sw_ms': width,
            'spont': row.get('spont', np.nan), 'rate_session': row.get('rate_session', np.nan),
            'isi': row.get('isi'), 'acg': row.get('acg'),
            'waveform': waveform, 'sample_rate': fs, 'waveform_half_reliability': half_r}


def _reliability(row, conditions):
    halves = row.get('fingerprint_halves') or {}
    counts = row.get('half_trial_counts') or {}
    left, right = [], []
    for condition in conditions:
        if condition not in halves or condition not in counts:
            return np.nan
        half = np.asarray(halves[condition], dtype=float)
        count = np.asarray(counts[condition], dtype=float)
        full = np.asarray(row['fingerprints'][condition], dtype=float)
        if half.shape != (len(full), 2) or count.shape != (2,):
            raise ValueError('fingerprint_halves must have shape (bin, 2) and half_trial_counts shape (2,)')
        if not np.isfinite(count).all() or (count < 1).any() or not np.equal(count, np.floor(count)).all():
            return np.nan
        left.extend(half[:, 0])
        right.extend(half[:, 1])
    return ev.spearman_brown(ev.pearson(left, right)) if left else np.nan


def _aligned_fingerprints(a, z, min_repeats, min_reliability):
    first, second = a.get('fingerprints') or {}, z.get('fingerprints') or {}
    na, nz = a.get('trial_counts') or {}, z.get('trial_counts') or {}
    conditions = []
    for condition in sorted(set(first) & set(second)):
        counts = [na.get(condition, np.nan), nz.get(condition, np.nan)]
        if any(not np.isfinite(n) or int(n) != n or n < min_repeats for n in counts):
            continue
        x, y = np.asarray(first[condition], dtype=float), np.asarray(second[condition], dtype=float)
        if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
            raise ValueError(f'Condition {condition!r} has incompatible fingerprint bin shapes')
        if x.size:
            conditions.append(condition)
    if conditions:
        step_a, step_z = a.get('fingerprint_bin_seconds', np.nan), z.get('fingerprint_bin_seconds', np.nan)
        if not np.isfinite(step_a) or not np.isfinite(step_z) or step_a <= 0 or not np.isclose(step_a, step_z, rtol=1e-9, atol=0):
            raise ValueError('Shared fingerprint bins must have the same measured positive duration')
    reliability_a, reliability_z = _reliability(a, conditions), _reliability(z, conditions)
    x = np.concatenate([first[c] for c in conditions]) if conditions else None
    y = np.concatenate([second[c] for c in conditions]) if conditions else None
    reason = 'measured' if conditions else 'no_shared_conditions_with_sufficient_repeats'
    if min_reliability is not None and any(not np.isfinite(r) or r < min_reliability
                                         for r in (reliability_a, reliability_z)):
        x, y = None, None
        reason = 'missing_or_low_split_half_reliability'
    return x, y, conditions, reliability_a, reliability_z, reason


def build_benchmark_pairs(units, feature_rows, channel_positions, *,
                          min_repeats=4, min_reliability=None, max_distance_um=None):
    """Return unscored ordinary pairs within each caller-specified recording group.

    ``units`` supplies opaque ``cellid``, pseudo-session ``session_id``, explicit
    ``su_type``, and ``recording_group``. ``feature_rows`` maps cellid to an
    eligible extraction row from ``temporal_split_unit_features``; source-unit
    metadata in those rows is ignored. ``channel_positions`` maps session IDs to
    measured channel-order (x, depth) coordinates in micrometers.

    All cross-pseudo-session pairs are considered. An optional Euclidean spatial
    gate uses fresh waveform locations, never known identity. Fingerprints join
    exact condition IDs and equal-duration bins, requiring min_repeats in both
    units. Reliability stays NaN without independently measured fingerprint
    halves; an explicit min_reliability masks unavailable/weak fingerprint
    evidence but does not remove the whole candidate or its physiology.

    This builder preserves observed repeat counts; it cannot equalize sampling
    from means alone. It does not estimate drift, calibrate scores, or assert
    that a within-sort benchmark reproduces daily probe reinsertion.
    """
    if not isinstance(min_repeats, (int, np.integer)) or min_repeats < 1:
        raise ValueError('min_repeats must be a positive integer')
    if min_reliability is not None and (not np.isfinite(min_reliability) or not -1 <= min_reliability <= 1):
        raise ValueError('min_reliability must be finite in [-1, 1] or None')
    if max_distance_um is not None and (not np.isfinite(max_distance_um) or max_distance_um < 0):
        raise ValueError('max_distance_um must be finite and nonnegative or None')
    if not {'session_id', 'recording_group'} <= set(units.columns):
        raise ValueError('Explicit pseudo-session IDs and recording_group are required')
    for col in ('session_id', 'recording_group'):
        if units[col].isna().any() or units[col].astype(str).str.strip().eq('').any():
            raise ValueError(f'{col} cannot be missing')
    eligible, _ = single_units(units)
    if (eligible.groupby('session_id').recording_group.nunique() > 1).any():
        raise ValueError('Each pseudo-session must belong to exactly one original recording group')
    summary = {}
    for unit in eligible.itertuples():
        if unit.cellid not in feature_rows or not feature_rows[unit.cellid].get('eligible', False):
            raise ValueError(f'{unit.cellid} is missing eligible independently extracted features')
        if unit.session_id not in channel_positions:
            raise ValueError(f'Missing measured channel geometry for {unit.session_id}')
        summary[unit.cellid] = _waveform_summary(feature_rows[unit.cellid], channel_positions[unit.session_id])
    rows = []
    for group_id, group in eligible.groupby('recording_group', sort=True):
        for a, z in itertools.combinations(group.sort_values(['session_id', 'cellid']).itertuples(), 2):
            if a.session_id == z.session_id:
                continue
            ua, uz = dict(summary[a.cellid]), dict(summary[z.cellid])
            ga, gz = np.asarray(channel_positions[a.session_id]), np.asarray(channel_positions[z.session_id])
            if not np.array_equal(ga, gz):
                raise ValueError('Within-recording temporal comparisons require matching measured channel order and geometry')
            if ua['sample_rate'] != uz['sample_rate'] or ua['waveform'].shape != uz['waveform'].shape:
                raise ValueError('Within-recording waveform samples must share sampling rate and window shape')
            distance = float(np.hypot(ua['depth'] - uz['depth'], ua['x_um'] - uz['x_um']))
            if max_distance_um is not None and distance > max_distance_um:
                continue
            fa, fz = feature_rows[a.cellid], feature_rows[z.cellid]
            x, y, conditions, ra, rz, status = _aligned_fingerprints(fa, fz, min_repeats, min_reliability)
            ua.update(psth=x, reliability=ra,
                      per_stim={c: float(np.nanmean(fa['fingerprints'][c])) for c in conditions})
            uz.update(psth=y, reliability=rz,
                      per_stim={c: float(np.nanmean(fz['fingerprints'][c])) for c in conditions})
            pair = ev.pair_evidence(ua, uz, shared_stims=conditions)
            pair.update(cellid_a=a.cellid, cellid_b=z.cellid,
                        siteid_a=a.session_id, siteid_b=z.session_id,
                        recording_group=group_id, distance_um=distance,
                        waveform_r=ev.pearson(ua['waveform'], uz['waveform']),
                        waveform_halves_r_a=ua['waveform_half_reliability'],
                        waveform_halves_r_b=uz['waveform_half_reliability'],
                        fingerprint_status=status, shared_conditions=tuple(conditions),
                        n_shared_conditions=len(conditions),
                        n_shared_bins=sum(len(fa['fingerprints'][c]) for c in conditions),
                        trial_counts_a=tuple(int(fa['trial_counts'][c]) for c in conditions),
                        trial_counts_b=tuple(int(fz['trial_counts'][c]) for c in conditions))
            rows.append(pair)
    if rows:
        return pd.DataFrame(rows)
    # Keep the numerical classifier columns even when all candidates are gated.
    return pd.DataFrame(columns=['cellid_a', 'cellid_b', 'siteid_a', 'siteid_b',
                                 'recording_group', *ev.ALL_PROPERTIES, 'waveform_r',
                                 'depth_delta_um', 'distance_um'])
