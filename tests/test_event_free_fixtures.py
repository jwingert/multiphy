"""A synthetic day or drift day without events keeps its spikes, units and waveforms."""
import numpy as np
import pandas as pd

from .synthetic_day import write_day
from .synthetic_drift import make_drift_days


def test_day_without_events_keeps_spikes(tmp_path):
    with_events = write_day(tmp_path, 'D1', seed=0)
    without = write_day(tmp_path / 'b', 'D1', seed=0, events=False)
    assert list(without.spikes) == list(with_events.spikes)
    for unit, samples in with_events.spikes.items():
        np.testing.assert_array_equal(without.spikes[unit], samples)
    assert without.events_path is None
    assert not (without.root / 'events.csv').exists()
    assert without.stimuli == []
    assert without.baselines.shape == (0, 2)


def test_drift_days_without_events_keep_units():
    with_events = make_drift_days(seed=0)
    without = make_drift_days(seed=0, events=False)
    pd.testing.assert_frame_equal(without.units, with_events.units)
    assert list(without.features) == list(with_events.features)
    for cellid, features in with_events.features.items():
        np.testing.assert_array_equal(without.features[cellid]['waveform_halves'], features['waveform_halves'])
    assert without.timing == {}
