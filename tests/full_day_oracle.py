"""Check a full-day row against its two halves rows without running ``pool_day_row``.

``reuse_day_summary`` calls ``pool_day_row``, so comparing the two tests the pooling against
itself. This oracle derives the expected row another way:

- the pooled waveform mean and standard error from each half's selected-snippet count ``n``,
  mean ``m`` and standard error ``se``, through sums (``n * m``) and sums of squares
  (``(n - 1) * n * se**2 + n * m**2``), rather than through the between-half deviations
  ``pool_day_row`` uses;
- the spike count, rate and baseline rate from the complete spike train in the analysis
  window;
- each condition's fingerprint as the trial-count-weighted mean of the halves' fingerprints:
  the halves split at a trial start, so every trial lies in one half.
"""
from __future__ import annotations

import numpy as np
import pytest


def assert_pooled_from_halves(feature: dict, rows: list, spikes: np.ndarray, spec: dict,
                              analysis_window: tuple) -> None:
    """Raise ``AssertionError`` unless ``feature`` is the full-day row of the two halves ``rows``."""
    rows = sorted(rows, key=lambda row: row['start_sample'])
    assert len(rows) == 2

    # The waveform: sums and sums of squares of the selected snippets, recovered from each half.
    counts = np.array([row['n_waveform_spikes'] for row in rows], dtype=float)[:, None, None]
    means = np.stack([np.asarray(row['waveform'], dtype=float) for row in rows])
    errors = np.stack([np.asarray(row['waveform_standard_error'], dtype=float) for row in rows])
    n = counts.sum()
    mean = (counts * means).sum(axis=0) / n
    squares = ((counts - 1) * counts * errors ** 2 + counts * means ** 2).sum(axis=0)
    error = np.sqrt((squares - n * mean ** 2) / (n - 1) / n)
    assert np.abs(mean).max() > 10  # the synthetic templates peak at 80 µV or more
    np.testing.assert_allclose(feature['waveform'], mean, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(feature['waveform_standard_error'], error, rtol=1e-5, atol=1e-5)
    assert feature['n_waveform_spikes'] == int(n)
    np.testing.assert_array_equal(feature['waveform_spike_samples'],
                                  np.concatenate([row['waveform_spike_samples'] for row in rows]))
    np.testing.assert_array_equal(feature['waveform_halves'], np.stack([row['waveform'] for row in rows], axis=-1))

    # Spike statistics over the analysis window, from the complete train.
    start, stop = (int(v) for v in analysis_window)
    rate = float(spec['sample_rate'])
    train = np.asarray(spikes)
    inside = train[(train >= start) & (train < stop)]
    assert (feature['start_sample'], feature['stop_sample']) == (start, stop)
    np.testing.assert_array_equal(feature['spike_samples'], inside)
    assert feature['n_spikes'] == len(inside)
    assert feature['rate_session'] == pytest.approx(len(inside) / ((stop - start) / rate))
    windows = [(max(int(a), start), min(int(b), stop)) for a, b in spec['baseline_windows']]
    windows = [(a, b) for a, b in windows if b > a]
    exposure = sum(b - a for a, b in windows)
    count = sum(int(((inside >= a) & (inside < b)).sum()) for a, b in windows)
    assert exposure > 0 and count > 0
    assert feature['spont'] == pytest.approx(count / (exposure / rate))

    # Fingerprints: every trial lies in one half.
    conditions = set(rows[0]['fingerprints']) | set(rows[1]['fingerprints'])
    assert conditions and set(feature['fingerprints']) == conditions
    for condition in conditions:
        trials = [row['trial_counts'].get(condition, 0) for row in rows]
        assert feature['trial_counts'][condition] == sum(trials)
        expected = sum(k * np.asarray(row['fingerprints'][condition], dtype=float)
                       for k, row in zip(trials, rows) if k) / sum(trials)
        assert np.abs(expected).max() > 0
        np.testing.assert_allclose(feature['fingerprints'][condition], expected, rtol=1e-5, atol=1e-4)
