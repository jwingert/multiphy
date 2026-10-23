"""Full-day rows pooled from a day's two halves, and the hash-verified ``full/`` folder.

A unit's full-day row comes from its two halves rows: the waveform mean and standard error
are pooled over the snippets each half already selected, and spike statistics and
exact-condition fingerprints are recomputed from the complete spike train. No voltage is
read. A unit with fewer than two halves rows gets an ineligible table row carrying the reason
of its first excluded half, and no features.

``save_full_day`` writes ``units.csv``, ``features.npz``, ``features.json``, ``extras.npz``,
``METHODS.md`` and ``manifest.json``; the sha256 of ``manifest.json`` is the full id.
``load_full_day`` checks every recorded digest before it reads the data. ``_pool_waveforms``
and ``_spike_features`` moved here verbatim from ``multiday_reuse.py``, which imports them
back. This module imports no lab module.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from ..benchmark import _pack, _unpack
from .unit_features import _checked_sample_spikes, _isi_acg

KIND = 'multiphy_full_day'
DATA_FILES = ('units.csv', 'features.npz', 'features.json', 'extras.npz')
TABLE_COLUMNS = ('cellid', 'session_id', 'siteid', 'well', 'recording_group', 'session_date',
                 'su_type', 'eligible', 'exclusion_reason')


def _pool_waveforms(rows):
    """Pool retained snippet moments, accounting for between-part mean shifts."""
    counts = np.asarray([r['n_waveform_spikes'] for r in rows], dtype=np.int64)
    means = np.stack([r['waveform'] for r in rows]).astype(float)
    errors = np.stack([r['waveform_standard_error'] for r in rows]).astype(float)
    if ((counts < 2).any() or not np.isfinite(means).all()
            or means.shape != errors.shape or not np.isfinite(errors).all() or (errors < 0).any()):
        raise ValueError('Reuse needs finite raw mean/SEM moments with at least two selected spikes per part')
    n = int(counts.sum())
    weights = counts[:, None, None]
    mean = (means * weights).sum(axis=0) / n
    m2 = (errors ** 2 * weights * (weights - 1) + weights * (means - mean) ** 2).sum(axis=0)
    return mean.astype(np.float32), np.sqrt(m2 / (n - 1) / n).astype(np.float32)


def _spike_features(spikes, bounds, sample_rate, spec, bin_seconds):
    """Recompute full spike statistics and exact-condition trial means, no voltage."""
    start, stop = bounds
    selected = spikes[(spikes >= start) & (spikes < stop)]
    isi, acg = _isi_acg(selected / sample_rate)
    baseline = np.asarray(spec.get('baseline_windows', []), dtype=np.int64).reshape(-1, 2)
    clipped = np.column_stack((np.maximum(baseline[:, 0], start), np.minimum(baseline[:, 1], stop)))
    clipped = clipped[clipped[:, 1] > clipped[:, 0]]
    exposure = int(np.diff(clipped, axis=1).sum())
    count = sum(int(np.searchsorted(selected, b) - np.searchsorted(selected, a)) for a, b in clipped)
    result = dict(spike_samples=selected, n_spikes=len(selected), isi=isi, acg=acg,
        rate_session=len(selected) / ((stop - start) / sample_rate),
        spont=count / (exposure / sample_rate) if exposure else np.nan,
        baseline_spike_count=count, baseline_exposure_samples=exposure,
        fingerprints={}, trial_counts={}, fingerprint_halves={}, half_trial_counts={})
    width = int(round(bin_seconds * sample_rate))
    if width < 1:
        raise ValueError('Fingerprint bin duration must span an acquisition sample')
    trials = pd.DataFrame(spec.get('stimuli', []), columns=['stimulus', 'start', 'stop'])
    trials = trials[(trials.start >= start) & (trials.stop <= stop)]
    for condition, group in trials.groupby('stimulus', sort=True):
        durations = (group.stop - group.start).to_numpy(dtype=np.int64)
        if not (durations == durations[0]).all():
            raise ValueError('Exact stimulus conditions must have identical durations')
        n_bins = int(durations[0] // width)
        if not n_bins:
            continue
        edges = np.arange(n_bins + 1) * width
        trial_counts = []
        for trial in group.sort_values('start').itertuples():
            values = selected[(selected >= trial.start) & (selected < trial.start + edges[-1])]
            trial_counts.append(np.histogram(values - trial.start, bins=edges)[0])
        trial_rates = np.asarray(trial_counts) / (width / sample_rate)
        counts = np.sum(trial_counts, axis=0)
        result['fingerprints'][str(condition)] = (counts / len(group) / (width / sample_rate)).astype(np.float32)
        result['trial_counts'][str(condition)] = len(group)
        mid = len(group) // 2
        result['half_trial_counts'][str(condition)] = (mid, len(group) - mid)
        if mid:
            result['fingerprint_halves'][str(condition)] = np.column_stack(
                (trial_rates[:mid].mean(axis=0), trial_rates[mid:].mean(axis=0))).astype(np.float32)
    return result


def pool_day_row(rows: list, spikes: np.ndarray, spec: dict, *, analysis_window: tuple,
                 windows: np.ndarray, max_spikes: int) -> dict:
    """One unit's full-day feature row from its two halves rows (sorted by start); no voltage.

    ``spikes`` is the unit's complete sorted spike train, ``windows`` the ``(2, 2)`` bounds of
    the halves and ``max_spikes`` the per-half waveform limit, recorded as
    ``fresh_full_mean_limit``. Raises ``ValueError`` when the halves' fingerprints disagree
    with ``spec``'s stimuli and the spike train.
    """
    selections = [np.asarray(r['waveform_spike_samples']).copy() for r in rows]
    mean, error = _pool_waveforms(rows)
    bins = rows[0]['fingerprint_bin_seconds']
    if any(r['fingerprint_bin_seconds'] != bins for r in rows):
        raise ValueError('Archived fingerprint bins differ across source parts')
    summaries = [_spike_features(spikes, bounds, spec['sample_rate'], spec, bins) for bounds in windows]
    for row, summary in zip(rows, summaries):
        if (row['trial_counts'] != summary['trial_counts'] or set(row['fingerprints']) != set(summary['fingerprints'])
                or any(not np.array_equal(row['fingerprints'][c], summary['fingerprints'][c]) for c in summary['fingerprints'])):
            raise ValueError('Archived fingerprints disagree with original stimulus intervals or complete spikes')
    feature = _spike_features(spikes, analysis_window, spec['sample_rate'], spec, bins)
    feature.update(start_sample=int(analysis_window[0]), stop_sample=int(analysis_window[1]),
        sample_rate=float(spec['sample_rate']), su_type='SU', preprocessing=spec['preprocessing'], eligible=True,
        waveform=mean, waveform_standard_error=error, waveform_spike_samples=np.concatenate(selections),
        n_waveform_spikes=sum(len(s) for s in selections),
        n_eligible_waveform_spikes=sum(r['n_eligible_waveform_spikes'] for r in rows),
        waveform_halves=np.stack([r['waveform'] for r in rows], axis=-1),
        half_waveform_spike_samples=selections, waveform_half_windows=windows.astype(int).tolist(),
        fingerprint_bin_seconds=bins)
    feature['summary_reuse'] = dict(full_mean_sampling='Count-weighted union of separately sampled outer-window snippets',
        half_windows=windows.astype(int).tolist(), half_spike_counts=[len(s) for s in selections],
        full_mean_selected_count=feature['n_waveform_spikes'], fresh_full_mean_limit=max_spikes,
        fingerprint_half_definition='Each condition chronological trials split at floor(n_trials/2), matching fresh extraction; independent of waveform outer windows',
        moment_precision='Pooled from float32 stored means and standard errors; floating-point rounding remains')
    return feature


def full_day_rows(dataset, spikes: dict, spec: dict, *, day: str, group: str, session_date: str,
                  analysis_window: tuple, windows: np.ndarray, max_spikes: int) -> dict:
    """Full-day table rows and features for every unit in ``spec['single_unit_ids']``.

    ``dataset`` is the day's halves archive (a ``BenchmarkDataset``) and ``spikes`` maps unit
    id to its complete spike train. A unit with two halves rows gets an eligible row and a
    ``features`` entry from ``pool_day_row``. A unit with fewer gets an ineligible row whose
    ``exclusion_reason`` is the reason recorded for its lowest excluded ``split_index``, and
    no features. Returns ``{'table', 'features', 'geometry'}``: ``table`` has
    ``TABLE_COLUMNS``, ``features`` is keyed by unit id and ``geometry`` is ``{day: geometry}``.
    ``windows`` must be two contiguous halves covering ``analysis_window``, and a pooled
    unit's halves rows must have been extracted on ``windows`` from the same spike train as
    ``spikes[unit]`` inside ``analysis_window``; otherwise ``ValueError``.
    """
    geometries = [np.asarray(g) for g in dataset.geometry.values()]
    if not geometries or any(not np.array_equal(g, geometries[0]) for g in geometries[1:]):
        raise ValueError('The halves pseudo-sessions need one shared geometry.')
    windows = np.asarray(windows)
    if (windows.shape != (2, 2) or not np.isfinite(windows).all()
            or not np.equal(windows, np.floor(windows)).all() or (windows[:, 1] <= windows[:, 0]).any()
            or windows[0, 1] != windows[1, 0]
            or [int(windows[0, 0]), int(windows[1, 1])] != [int(v) for v in analysis_window]):
        raise ValueError('The halves windows must be two contiguous parts covering the analysis window.')
    labels = dataset.labels
    table, features = [], {}
    for unit in spec['single_unit_ids']:
        unit = str(unit)
        cellids = labels.loc[labels.source_unit_id == unit, 'cellid']
        rows = sorted([dataset.features[c] for c in cellids], key=lambda row: row['start_sample'])
        if len(rows) > 2:
            raise ValueError(f'Unit {unit} has {len(rows)} halves rows; a day has two halves.')
        reason = ''
        if len(rows) == 2:
            train = _checked_sample_spikes(spikes[unit])
            if any([row['start_sample'], row['stop_sample']] != bounds.tolist()
                   for row, bounds in zip(rows, windows)):
                raise ValueError(f'Unit {unit}: the halves rows were extracted on other windows.')
            inside = train[(train >= analysis_window[0]) & (train < analysis_window[1])]
            if not np.array_equal(np.concatenate([row['spike_samples'] for row in rows]), inside):
                raise ValueError(f'Unit {unit}: the halves rows do not hold its complete spike train.')
            features[unit] = pool_day_row(rows, train, spec,
                                          analysis_window=analysis_window, windows=windows,
                                          max_spikes=max_spikes)
        else:
            excluded = [e for e in dataset.exclusions if str(e['source_unit_id']) == unit]
            if not excluded:
                raise ValueError(f'Unit {unit} has {len(rows)} halves rows and no recorded exclusion.')
            reason = min(excluded, key=lambda e: int(e['split_index']))['reason']
        table.append(dict(cellid=unit, session_id=day, siteid=day, well=group, recording_group=group,
                          session_date=session_date, su_type='SU', eligible=len(rows) == 2,
                          exclusion_reason=reason))
    return {'table': pd.DataFrame(table, columns=list(TABLE_COLUMNS)), 'features': features,
            'geometry': {day: geometries[0].copy()}}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _methods(target: Path, full: dict, metadata: dict) -> str:
    table = full['table']
    n_eligible = int(table['eligible'].sum()) if len(table) else 0
    return f'''# Full-day unit rows pooled from temporal halves — methods notes

- **Output folder:** `{target}`
- **Artifacts:** `units.csv`, `features.npz`, `features.json`, `extras.npz`, `manifest.json`
- **Generated by:** `multiphy/extract/full_day.py:full_day_rows()/save_full_day()` · **Env:** `{sys.executable}` · **Updated:** `{datetime.now(timezone.utc).date()}`
- **Units:** {len(table)} declared single units · {n_eligible} eligible

---

## Goal

One row per declared single unit over the whole analysis window of one recording day, built from the day's two temporal halves without reading raw voltage a second time.

## Data lineage

- `units.csv` ← `full_day.py:full_day_rows()`; one row per unit in the source's single-unit list. `eligible` is `True` when both halves passed the halves extraction. Otherwise `exclusion_reason` holds the reason recorded for the unit's first excluded half, and the unit has no features.
- `features.npz` and `features.json` ← `full_day.py:pool_day_row()`. The waveform mean and standard error are pooled, count-weighted and with the between-half mean shift, over the snippets each half selected. Spike count, rate, baseline rate, interval histogram, autocorrelogram and exact-condition stimulus fingerprints are recomputed from the complete spike train over the analysis window. Each feature's `summary_reuse` records the half windows and snippet counts.
- `extras.npz` ← optional per-unit arrays carried from the source; no matching backend reads them.
- `manifest.json` ← the sha256 of the four data files and the extraction metadata (halves rule and boundary, when recorded); its own sha256 is the full-day id.

## Caveats

The full-day waveform mean pools two separately sampled half selections, so it can use up to twice the per-half snippet limit; it is not a fresh full-day sample. Rounding from the float32 half moments remains.
'''


def save_full_day(full: dict, target: Path, *, metadata: dict) -> str:
    """Write ``full`` into the new folder ``target`` and return the full id.

    Writes ``units.csv``, ``features.npz`` and ``features.json`` (features, geometry and
    ``metadata``), ``extras.npz`` (``full.get('extras', {})``), ``METHODS.md`` and
    ``manifest.json``, which records the sha256 of the four data files. The full id is the
    sha256 of ``manifest.json``. The folder is staged beside ``target`` and renamed into
    place; an existing ``target`` raises ``FileExistsError``.
    """
    target = Path(target)
    if target.exists():
        raise FileExistsError(f'Full day already exists: {target}')
    extras = dict(full.get('extras', {}))
    if any(np.asarray(value).dtype.hasobject for value in extras.values()):
        raise ValueError('Full-day extras must be plain arrays, not objects.')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.full-day-', dir=target.parent))
    try:
        full['table'].to_csv(stage / 'units.csv', index=False)
        arrays = {}
        tree = _pack(dict(features=full['features'], geometry=full['geometry'], metadata=metadata), arrays)
        np.savez_compressed(stage / 'features.npz', **arrays)
        (stage / 'features.json').write_text(json.dumps(tree, allow_nan=False, indent=2))
        np.savez_compressed(stage / 'extras.npz', **extras)
        (stage / 'METHODS.md').write_text(_methods(target, full, metadata), encoding='utf-8')
        manifest = {'schema_version': 1, 'kind': KIND, 'metadata': metadata,
                    'sha256': {name: _sha256(stage / name) for name in DATA_FILES}}
        (stage / 'manifest.json').write_text(json.dumps(manifest, allow_nan=False, indent=2))
        full_id = _sha256(stage / 'manifest.json')
        if sys.platform == 'win32':
            # Windows rename refuses an existing destination, including an
            # empty directory, so publish directly without a placeholder.
            stage.rename(target)
        else:
            # Reserve the destination before replacing our empty directory.
            target.mkdir()
            stage.replace(target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return full_id


def load_full_day(path: Path) -> dict:
    """Read a full-day folder after checking every recorded digest.

    Returns ``{'table', 'features', 'geometry', 'metadata', 'extras'}``. A changed data file
    raises ``ValueError('Full-day content hash mismatch: <name>')``.
    """
    path = Path(path)
    manifest = json.loads((path / 'manifest.json').read_text())
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != KIND
            or set(manifest.get('sha256', {})) != set(DATA_FILES)):
        raise ValueError(f'Unsupported or incomplete full-day manifest: {path}')
    for name, digest in manifest['sha256'].items():
        if _sha256(path / name) != digest:
            raise ValueError(f'Full-day content hash mismatch: {name}')
    table = pd.read_csv(path / 'units.csv', dtype=str, keep_default_na=False)
    table['eligible'] = table['eligible'].eq('True')
    with np.load(path / 'features.npz', allow_pickle=False) as arrays:
        contents = _unpack(json.loads((path / 'features.json').read_text()), arrays)
    if contents['metadata'] != manifest['metadata']:
        raise ValueError('Full-day metadata disagree with the manifest.')
    with np.load(path / 'extras.npz', allow_pickle=False) as saved:
        extras = {name: saved[name] for name in saved.files}
    return {'table': table, 'features': contents['features'], 'geometry': contents['geometry'],
            'metadata': contents['metadata'], 'extras': extras}
