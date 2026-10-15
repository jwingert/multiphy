# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: ISI_BINS, ACG_BINS, _isi_acg and the temporal extraction functions moved verbatim from features.py, which imports them back (multiphy project folder Task 11) | date: 2026-09-25]
"""Per-unit spike statistics and waveform averages in disjoint temporal pseudo-sessions.

``_isi_acg`` gives the inter-spike-interval histogram and autocorrelogram on ``ISI_BINS`` and
``ACG_BINS``. ``temporal_split_unit_features`` recomputes a unit's evidence in each sample
window: rates, interval statistics, exact-condition fingerprints, and waveform averages read
from voltage through ``average_spike_waveforms``. ``waveform_sample_selection`` plans which
snippets an average reads.

Units differ. ``_isi_acg`` takes spike times in seconds: ``ISI_BINS`` span 1 ms to 0.5 s,
and ``ACG_BINS`` span 0 to 50 ms in 1 ms steps. Distinct sample indices given to it drop every
interval, so the histogram is all zero and the autocorrelogram all NaN. The temporal
extraction functions (``temporal_sample_windows`` through ``temporal_split_unit_features``)
take acquisition sample indices; ``temporal_split_unit_features`` divides by
``sample_rate`` before it calls ``_isi_acg``.

This code moved verbatim from ``features.py``, which imports every name back. The module
imports only the standard library, numpy and pandas.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ISI_BINS = np.logspace(np.log10(1e-3), np.log10(0.5), 41)
ACG_BINS = np.arange(0.0, 0.051, 0.001)


def _isi_acg(st):
    if len(st) < 20:
        return np.full(len(ISI_BINS) - 1, np.nan), np.full(len(ACG_BINS) - 1, np.nan)
    st = np.sort(st)
    d = np.diff(st)
    h, _ = np.histogram(d[(d > 0) & (d < 0.5)], bins=ISI_BINS)
    h = h / (h.sum() + 1e-12)
    lags, j = [], 0
    for i in range(len(st)):
        while j < len(st) and st[j] <= st[i] + ACG_BINS[-1]:
            j += 1
        seg = st[i + 1:j] - st[i]
        if len(seg):
            lags.append(seg)
    if not lags:
        return h, np.full(len(ACG_BINS) - 1, np.nan)
    a, _ = np.histogram(np.concatenate(lags), bins=ACG_BINS)
    return h, a / (a.sum() + 1e-12)


# [AGENT EDIT START | agent: codex | user: wingertj | reason: recompute disjoint temporal single-unit benchmark features from voltage and spike times | date: 2026-09-17]
def temporal_sample_windows(start, stop, n_parts=4):
    """Disjoint half-open acquisition-sample windows, never alternate spikes.

    Use the same windows for every unit in a recording. Source cluster identity
    remains the benchmark reference, not an input feature to a matcher.
    """
    if int(start) != start or int(stop) != stop or int(n_parts) != n_parts:
        raise ValueError('Sample boundaries and n_parts must be integers.')
    start, stop, n_parts = int(start), int(stop), int(n_parts)
    if start < 0 or n_parts < 2 or stop - start < n_parts:
        raise ValueError('Need a nonnegative start and at least one sample per part.')
    edges = start + (np.arange(n_parts + 1, dtype=np.int64) * (stop - start)) // n_parts
    return np.column_stack((edges[:-1], edges[1:]))


def _checked_sample_spikes(spike_samples):
    x = np.asarray(spike_samples)
    if x.ndim != 1 or not np.isfinite(x).all() or (x < 0).any():
        raise ValueError('Spike samples must be a finite, nonnegative vector.')
    if not np.equal(x, np.floor(x)).all():
        raise ValueError('Spike sample indices must be integers, not seconds.')
    x = np.sort(x.astype(np.int64))
    if len(x) > 1 and (np.diff(x) == 0).any():
        raise ValueError('Duplicate spike sample indices must be resolved upstream.')
    return x


def waveform_sample_selection(spike_samples, window, before=41, after=41,
                              max_spikes=1000, seed=0):
    """Plan the exact raw snippets used for an average, without reading voltage."""
    spikes = _checked_sample_spikes(spike_samples)
    start, stop = window
    for value in (start, stop, before, after, max_spikes):
        if not np.isfinite(value) or int(value) != value:
            raise ValueError('Waveform sample boundaries and limits must be integers.')
    if start < 0 or stop <= start or before < 1 or after < 1 or max_spikes < 2:
        raise ValueError('Invalid waveform window or spike limit.')
    eligible = spikes[(spikes - before >= start) & (spikes + after <= stop)]
    if len(eligible) < 2:
        raise ValueError('At least two complete waveform snippets are required.')
    selected = (np.sort(np.random.default_rng(seed).choice(eligible, int(max_spikes), replace=False))
                if len(eligible) > max_spikes else eligible)
    return selected, len(eligible)


def average_spike_waveforms(spike_samples, read_samples, window, n_channels,
                            before=41, after=41, max_spikes=1000, seed=0):
    """Read fresh voltage snippets and average them; never split a template.

    ``read_samples(start, stop)`` must return a (sample, channel) array of
    consistently preprocessed voltage in the SAME acquisition-sample clock as
    the spikes. Filtering, reference subtraction, scaling, channel order, and
    any motion correction are the caller's responsibility and must be recorded
    in provenance. A read-only memmap can be supplied through a slicing lambda.
    No snippet may cross the requested temporal window. Subsampling here applies
    only to waveform averaging, never to firing-rate/interval measurements.
    """
    spikes = _checked_sample_spikes(spike_samples)
    start, stop = window
    for value in (start, stop, n_channels, before, after, max_spikes):
        if not np.isfinite(value) or int(value) != value:
            raise ValueError('Waveform dimensions and sample boundaries must be integers.')
    start, stop, n_channels = int(start), int(stop), int(n_channels)
    before, after, max_spikes = int(before), int(after), int(max_spikes)
    if start < 0 or stop <= start or n_channels < 1 or before < 1 or after < 1 or max_spikes < 2:
        raise ValueError('Invalid waveform window, channel count, or spike limit.')
    selected, n_eligible = waveform_sample_selection(spikes, (start, stop), before, after, max_spikes, seed)
    shape = (before + after, n_channels)
    # [AGENT EDIT START | agent: codex | user: wingertj | reason: consume only exact verified streaming selections without changing ordinary waveform readers | date: 2026-09-18]
    summary_provider = getattr(read_samples, 'waveform_summary', None)
    if callable(summary_provider):
        summary = summary_provider(selected, before=before, after=after, n_channels=n_channels)
        if summary is not None:
            for key in ('waveform', 'waveform_standard_error'):
                value = np.asarray(summary[key])
                if value.shape != shape or value.dtype != np.float32 or not np.isfinite(value).all():
                    raise ValueError('Summary reader returned an invalid waveform summary.')
            if (summary['waveform_standard_error'] < 0).any():
                raise ValueError('Summary reader returned negative waveform standard errors.')
            return dict(waveform=summary['waveform'].copy(),
                        waveform_standard_error=summary['waveform_standard_error'].copy(),
                        waveform_spike_samples=selected, n_waveform_spikes=len(selected),
                        n_eligible_waveform_spikes=n_eligible)
    # [AGENT EDIT END]
    mean = np.zeros(shape, dtype=float)
    m2 = np.zeros(shape, dtype=float)
    for count, spike in enumerate(selected, 1):
        voltage = np.asarray(read_samples(int(spike - before), int(spike + after)), dtype=float)
        if voltage.shape != shape or not np.isfinite(voltage).all():
            raise ValueError(f'Voltage reader returned invalid snippet at sample {spike}: {voltage.shape}.')
        delta = voltage - mean
        mean += delta / count
        m2 += delta * (voltage - mean)
    return dict(waveform=mean.astype(np.float32),
                waveform_standard_error=np.sqrt(m2 / (len(selected) - 1) / len(selected)).astype(np.float32),
                waveform_spike_samples=selected, n_waveform_spikes=len(selected),
                n_eligible_waveform_spikes=n_eligible)


def temporal_split_unit_features(spike_samples, read_samples, windows, *,
                                 sample_rate, n_channels, source_unit,
                                 source_session, su_type, preprocessing,
                                 stimuli=None, baseline_windows=None,
                                 bin_seconds=0.05, before=41, after=41,
                                 max_spikes=1000, min_spikes=20, seed=0):
    """Recompute single-unit evidence in disjoint temporal pseudo-sessions.

    All times are acquisition sample indices, including ``stimuli`` rows
    (``stimulus``, ``start``, ``stop``) and ``baseline_windows``. Stimulus labels
    must already identify the full acoustic condition, including channel/ear.
    Only fully contained stimulus trials are used; trial counts are returned.
    Each pseudo-session also has independently recomputed early/late waveform
    averages in ``waveform_halves`` (sample, channel, 2), suitable for published
    waveform matchers after their documented preprocessing.

    This is a feature-extraction primitive, not evidence of across-day accuracy.
    Train/evaluation separation must keep all parts of a source unit (preferably
    all units from a recording or well) together. Insufficient splits are reported
    explicitly and never padded with a copied waveform.
    """
    if str(su_type).strip().upper() not in ('SU', 'SINGLE', 'SINGLE UNIT', 'SINGLE_UNIT'):
        raise ValueError('Temporal identity benchmarks require an explicit single-unit label.')
    if not isinstance(preprocessing, str) or not preprocessing.strip():
        raise ValueError('Describe voltage preprocessing and spike-to-voltage clock alignment.')
    if any(not isinstance(value, str) or not value.strip()
           for value in (source_unit, source_session)):
        raise ValueError('Nonempty source_unit and source_session strings are required.')
    if not np.isfinite(sample_rate) or sample_rate <= 0 or not np.isfinite(bin_seconds) or bin_seconds <= 0:
        raise ValueError('Sample rate and fingerprint bin duration must be positive.')
    if int(min_spikes) != min_spikes or min_spikes < 2:
        raise ValueError('min_spikes must be an integer at least two.')
    spikes = _checked_sample_spikes(spike_samples)
    win = np.asarray(windows)
    if win.ndim != 2 or win.shape[1] != 2 or not len(win):
        raise ValueError('windows must contain start/stop pairs.')
    if not np.isfinite(win).all() or not np.equal(win, np.floor(win)).all():
        raise ValueError('Window boundaries must be finite integer sample indices.')
    win = win.astype(np.int64)
    if (win[:, 0] < 0).any() or (win[:, 1] <= win[:, 0]).any() or (win[1:, 0] < win[:-1, 1]).any():
        raise ValueError('Temporal windows must be ordered, disjoint, and nonempty.')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an empty stimuli list (a day without events) reads as no stimuli, as full_day.py does; pd.DataFrame([]) has no columns | date: 2026-10-08]
    if stimuli is None or len(stimuli) == 0:
        trials = pd.DataFrame(columns=['stimulus', 'start', 'stop'])
    else:
        trials = pd.DataFrame(stimuli)
    # [AGENT EDIT END]
    if not {'stimulus', 'start', 'stop'}.issubset(trials.columns):
        raise ValueError('Stimuli require stimulus, start, and stop columns.')
    bounds = trials[['start', 'stop']].to_numpy(dtype=float)
    if len(bounds) and (not np.isfinite(bounds).all() or not np.equal(bounds, np.floor(bounds)).all()
                        or (bounds[:, 0] < 0).any() or (bounds[:, 1] <= bounds[:, 0]).any()):
        raise ValueError('Stimulus boundaries must be valid acquisition sample indices.')
    if trials.stimulus.isna().any():
        raise ValueError('Every stimulus trial requires a condition label.')
    for stimulus, group in trials.groupby('stimulus', sort=False):
        durations = (group.stop - group.start).to_numpy(dtype=np.int64)
        if not (durations == durations[0]).all():
            raise ValueError(f'Stimulus {stimulus!r} has inconsistent durations across splits; distinguish conditions upstream.')
    baseline = np.asarray(baseline_windows if baseline_windows is not None else [], dtype=float).reshape(-1, 2)
    if len(baseline) and (not np.isfinite(baseline).all() or not np.equal(baseline, np.floor(baseline)).all()
                         or (baseline[:, 0] < 0).any() or (baseline[:, 1] <= baseline[:, 0]).any()
                         or (baseline[1:, 0] < baseline[:-1, 1]).any()):
        raise ValueError('Baseline windows must be ordered, disjoint sample intervals.')
    bin_samples = int(round(bin_seconds * sample_rate))
    if bin_samples < 1:
        raise ValueError('Fingerprint bins must span at least one acquisition sample.')
    results = []
    for split_index, (start, stop) in enumerate(win):
        st = spikes[(spikes >= start) & (spikes < stop)]
        row = dict(source_unit=str(source_unit), source_session=str(source_session),
                   split_index=split_index, start_sample=int(start), stop_sample=int(stop),
                   sample_rate=float(sample_rate), n_spikes=len(st), su_type='SU',
                   preprocessing=str(preprocessing), reference_kind='within_sort_source_cluster',
                   spike_samples=st, eligible=False)
        results.append(row)
        if len(st) < min_spikes:
            row['exclusion_reason'] = 'Too few spikes in temporal split.'
            continue
        row['rate_session'] = len(st) / ((stop - start) / sample_rate)
        row['isi'], row['acg'] = _isi_acg(st / sample_rate)
        base = np.column_stack((np.maximum(baseline[:, 0], start), np.minimum(baseline[:, 1], stop)))
        base = base[base[:, 1] > base[:, 0]]
        duration = float(np.diff(base, axis=1).sum() / sample_rate)
        n_baseline = sum(int(np.searchsorted(st, b, side='left') - np.searchsorted(st, a, side='left')) for a, b in base)
        row['spont'] = n_baseline / duration if duration else np.nan
        row['fingerprints'], row['trial_counts'] = {}, {}
        row['fingerprint_halves'], row['half_trial_counts'] = {}, {}
        selected_trials = trials[(trials.start >= start) & (trials.stop <= stop)]
        for stimulus, group in selected_trials.groupby('stimulus', sort=True):
            durations = (group.stop - group.start).to_numpy(dtype=np.int64)
            if not (durations == durations[0]).all():
                raise ValueError(f'Stimulus {stimulus!r} has inconsistent durations; distinguish conditions upstream.')
            n_bins = int(durations[0] // bin_samples)
            if not n_bins:
                continue
            trial_counts = []
            edges = np.arange(n_bins + 1) * bin_samples
            for trial in group.sort_values('start').itertuples():
                relative = st[(st >= trial.start) & (st < trial.start + edges[-1])] - trial.start
                trial_counts.append(np.histogram(relative, bins=edges)[0])
            trial_rates = np.asarray(trial_counts) / (bin_samples / sample_rate)
            counts = np.sum(trial_counts, axis=0)
            row['fingerprints'][str(stimulus)] = (counts / len(group) / (bin_samples / sample_rate)).astype(np.float32)
            row['trial_counts'][str(stimulus)] = len(group)
            trial_mid = len(group) // 2
            row['half_trial_counts'][str(stimulus)] = (trial_mid, len(group) - trial_mid)
            if trial_mid:
                row['fingerprint_halves'][str(stimulus)] = np.column_stack(
                    (trial_rates[:trial_mid].mean(axis=0), trial_rates[trial_mid:].mean(axis=0))).astype(np.float32)
        row['fingerprint_bin_seconds'] = bin_samples / sample_rate
        try:
            row.update(average_spike_waveforms(st, read_samples, (start, stop), n_channels,
                                              before, after, max_spikes, seed + split_index * 3))
            mid = (start + stop) // 2
            halves = [average_spike_waveforms(st, read_samples, half, n_channels,
                                              before, after, max_spikes, seed + split_index * 3 + h + 1)
                      for h, half in enumerate(((start, mid), (mid, stop)))]
        except ValueError as exc:
            row['exclusion_reason'] = str(exc)
            continue
        row['waveform_halves'] = np.stack([h['waveform'] for h in halves], axis=-1)
        row['half_waveform_spike_samples'] = [h['waveform_spike_samples'] for h in halves]
        row['eligible'] = True
    return results
# [AGENT EDIT END]
# [AGENT EDIT END]
