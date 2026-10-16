# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify independent waveform recomputation and temporal feature boundaries | date: 2026-09-17]
import numpy as np
import pytest

from multiphy.features import (
    average_spike_waveforms, temporal_sample_windows, temporal_split_unit_features,
)


def test_temporal_windows_partition_recording_without_thinning():
    windows = temporal_sample_windows(7, 110, 4)
    assert windows[0, 0] == 7 and windows[-1, 1] == 110
    assert np.array_equal(windows[1:, 0], windows[:-1, 1])
    with pytest.raises(ValueError):
        temporal_sample_windows(0, 2, 4)


def test_waveform_reads_are_disjoint_and_recomputed_from_voltage():
    voltage = np.zeros((1000, 2))
    spikes = np.arange(10, 990, 10)
    voltage[spikes, 0] = np.where(spikes < 500, -4, -9)
    calls = []

    def reader(start, stop):
        calls.append((start, stop))
        return voltage[start:stop]

    parts = temporal_split_unit_features(
        spikes, reader, [[0, 500], [500, 1000]], sample_rate=1000,
        n_channels=2, source_unit='cell1', source_session='session1',
        su_type='SU', preprocessing='synthetic voltage; identical sample clock',
        before=2, after=3)
    assert all(p['eligible'] for p in parts)
    assert parts[0]['waveform'][2, 0] == -4
    assert parts[1]['waveform'][2, 0] == -9
    assert all(stop <= 500 or start >= 500 for start, stop in calls)
    assert not set(parts[0]['waveform_spike_samples']) & set(parts[1]['waveform_spike_samples'])
    for p in parts:
        assert p['waveform_halves'].shape == (5, 2, 2)
        assert not set(p['half_waveform_spike_samples'][0]) & set(p['half_waveform_spike_samples'][1])
        assert p['reference_kind'] == 'within_sort_source_cluster'


def test_waveform_subsampling_does_not_thin_rate_or_intervals():
    voltage = np.zeros((1000, 2))
    parts = temporal_split_unit_features(
        np.arange(10, 990, 10), lambda a, b: voltage[a:b], [[0, 1000]],
        sample_rate=1000, n_channels=2, source_unit='one', source_session='a',
        su_type='SU', preprocessing='synthetic', before=2, after=3, max_spikes=5)
    p = parts[0]
    assert p['n_spikes'] == 98
    assert p['rate_session'] == 98
    assert p['n_waveform_spikes'] == 5
    assert p['isi'].sum() == pytest.approx(1)


def test_fingerprints_preserve_conditions_and_drop_cross_boundary_trials():
    voltage = np.zeros((1000, 2))
    trials = [dict(stimulus='sound:left', start=0, stop=100),
              dict(stimulus='sound:left', start=200, stop=300),
              dict(stimulus='sound:right', start=300, stop=400),
              dict(stimulus='crossing', start=450, stop=550)]
    parts = temporal_split_unit_features(
        np.arange(5, 1000, 10), lambda a, b: voltage[a:b], [[0, 500], [500, 1000]],
        sample_rate=1000, n_channels=2, source_unit='one', source_session='a',
        su_type='SU', preprocessing='synthetic', stimuli=trials,
        baseline_windows=[[0, 100], [400, 600]], before=2, after=3)
    p = parts[0]
    assert p['trial_counts'] == {'sound:left': 2, 'sound:right': 1}
    assert np.array_equal(p['fingerprints']['sound:left'], [100, 100])
    assert p['spont'] == 100
    assert parts[1]['trial_counts'] == {}


@pytest.mark.parametrize('label', ['MU', 'unknown', '', 'noise'])
def test_non_single_units_are_rejected(label):
    with pytest.raises(ValueError, match='single-unit'):
        temporal_split_unit_features([], None, [[0, 100]], sample_rate=1000,
                                     n_channels=2, source_unit='one', source_session='a',
                                     su_type=label, preprocessing='synthetic')


def test_bad_reader_and_insufficient_halves_do_not_create_fake_waveforms():
    with pytest.raises(ValueError, match='invalid snippet'):
        average_spike_waveforms([10, 20], lambda a, b: np.zeros((1, 1)),
                                (0, 100), 2, before=2, after=3)
    voltage = np.zeros((1000, 2))
    parts = temporal_split_unit_features(
        np.arange(10, 400, 10), lambda a, b: voltage[a:b], [[0, 1000]],
        sample_rate=1000, n_channels=2, source_unit='one', source_session='a',
        su_type='SU', preprocessing='synthetic', before=2, after=3)
    assert not parts[0]['eligible']
    assert 'waveform_halves' not in parts[0]
    assert 'two complete waveform' in parts[0]['exclusion_reason']


def test_overlapping_windows_and_fractional_spike_samples_fail():
    args = dict(sample_rate=1000, n_channels=2, source_unit='one',
                source_session='a', su_type='SU', preprocessing='synthetic')
    with pytest.raises(ValueError, match='disjoint'):
        temporal_split_unit_features([], None, [[0, 60], [50, 100]], **args)
    with pytest.raises(ValueError, match='integers'):
        temporal_split_unit_features([0.5, 1.5], None, [[0, 100]], **args)
@pytest.mark.parametrize('field', ['source_unit', 'source_session', 'preprocessing'])
@pytest.mark.parametrize('value', [None, '', ' '])
def test_missing_reference_provenance_is_rejected(field, value):
    args = dict(sample_rate=1000, n_channels=2, source_unit='one',
                source_session='a', su_type='SU', preprocessing='synthetic')
    args[field] = value
    with pytest.raises(ValueError):
        temporal_split_unit_features([], None, [[0, 100]], **args)


def test_stimulus_duration_must_agree_across_temporal_parts():
    trials = [dict(stimulus='same', start=0, stop=100),
              dict(stimulus='same', start=500, stop=700)]
    with pytest.raises(ValueError, match='durations across splits'):
        temporal_split_unit_features([], None, [[0, 500], [500, 1000]],
                                     sample_rate=1000, n_channels=2,
                                     source_unit='one', source_session='a',
                                     su_type='SU', preprocessing='synthetic', stimuli=trials)
# [AGENT EDIT END]
