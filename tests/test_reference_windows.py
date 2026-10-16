"""Reference windows: a unit's halves come from its day's trials in event runs, else from 5 s chunks;
spont reads baselines only when every day has them; no-event groups skip by spikes only."""
import inspect
from pathlib import Path

import numpy as np
import pytest

from multiphy.backends.local.examples import pair_properties
from multiphy.backends.local.references import (
    chunk_windows, evidence_by_group, group_observations, reference_windows, require_waveform_halves,
    trial_splits, unit_observation)
from multiphy.contract.inputs import RunEvidence, TrialTiming
from multiphy.project.folder import Project
from multiphy.project.selection import Selection

from .synthetic_day import extracted_project
from .synthetic_drift import make_drift_days

FS = 30000.0
EVENTS = RunEvidence(events=True, spont='baseline', shared={}, reasons=())
NO_EVENTS = RunEvidence(events=False, spont='baseline', shared={}, reasons=('no events · test',))
WHOLE = RunEvidence(events=False, spont='whole_recording', shared={},
                    reasons=('no events · test', 'spont · whole recording · test'))


def _inputs(project_dir: Path, days: tuple = ('D1', 'D2')) -> tuple:
    """``(units, features, geometry, timing)`` of an extracted project's days, as a project run reads them."""
    selection = Selection(Project.open(project_dir.parent, project_dir.name), days)
    dataset = selection.dataset()
    return dataset.units, dataset.features, dataset.geometry, selection.timing()


@pytest.fixture(scope='module')
def shared(tmp_path_factory) -> tuple:
    return _inputs(extracted_project(tmp_path_factory.mktemp('shared')))


@pytest.fixture(scope='module')
def disjoint(tmp_path_factory) -> tuple:
    return _inputs(extracted_project(tmp_path_factory.mktemp('disjoint'), conditions={'D2': ('C', 'D')}))


@pytest.fixture(scope='module')
def mixed(tmp_path_factory) -> tuple:
    return _inputs(extracted_project(tmp_path_factory.mktemp('mixed'), no_event_days=('D2',)))


def _three_trial_timing() -> TrialTiming:
    """3 trials of 5 s at 30 kHz in [0, 600000): baseline the first 0.5 s, stimulus the next 2 s."""
    starts = (0, 200000, 400000)
    return TrialTiming(
        source_session='hand', sample_rate=FS, analysis_window=(0, 600000),
        trial_windows=np.asarray([[t, t + 150000] for t in starts], np.int64),
        stimuli=tuple({'stimulus': 'A', 'start': t + 15000, 'stop': t + 75000, 'trial': i + 1}
                      for i, t in enumerate(starts)),
        baseline_windows=np.asarray([[t, t + 15000] for t in starts], np.int64),
        provenance={'sample_index_origin': 'zero_based_voltage_row', 'method': 'hand'}).validate()


def test_chunk_windows_drop_leftover():
    assert chunk_windows(0, 597000, FS).shape == (3, 2)
    four = chunk_windows(0, 600000, FS)
    assert four.shape == (4, 2) and four[-1].tolist() == [450000, 600000]
    assert chunk_windows(0, 149999, FS).shape == (0, 2)


def test_reference_windows_three_modes():
    timing = _three_trial_timing()
    feature = {'start_sample': 0, 'stop_sample': 600000, 'sample_rate': FS}
    chunks = chunk_windows(0, 600000, FS)
    events = reference_windows(feature, timing, EVENTS)
    assert events.trial_windows.shape == (3, 2)
    assert np.array_equal(events.trial_windows, timing.trial_windows)
    assert events.stimuli == list(timing.stimuli)
    assert np.array_equal(events.baseline_windows, timing.baseline_windows)
    no_events = reference_windows(feature, timing, NO_EVENTS)
    assert no_events.trial_windows.shape == (4, 2) and np.array_equal(no_events.trial_windows, chunks)
    assert no_events.stimuli == []
    assert np.array_equal(no_events.baseline_windows, timing.baseline_windows)
    whole = reference_windows(feature, None, WHOLE)
    assert whole.trial_windows.shape == (4, 2) and np.array_equal(whole.trial_windows, chunks)
    assert whole.stimuli == []
    assert np.array_equal(whole.baseline_windows, chunks)


def test_whole_recording_spont_on_every_day(mixed):
    units, features, geometry, timing = mixed
    assert sorted(timing) == ['D1']
    (group,), _ = group_observations(units, features, geometry, timing, min_spikes=2, n_splits=2)
    assert group.keys == ()
    assert sorted(set(group.session_of.values())) == ['D1', 'D2']
    for cellid, observation in group.observations.items():
        spikes = np.asarray(features[cellid]['spike_samples'])
        if group.session_of[cellid] == 'D1':
            expected = ((spikes >= 0) & (spikes < 450000)).sum() / 15.0
        else:
            expected = ((spikes >= 0) & (spikes < 600000)).sum() / 20.0
        assert np.isclose(observation['spont'], expected)


def test_short_window_unit_skipped():
    days = make_drift_days(n_days=2, n_neurons=4, offsets_um=(0.0, 0.0), presence=1.0, seed=3, events=False)
    short = days.units.cellid.iloc[0]
    feature = days.features[short]
    spikes = np.asarray(feature['spike_samples'])
    features = dict(days.features)
    features[short] = dict(feature, stop_sample=int(8 * FS), spike_samples=spikes[spikes < int(8 * FS)])
    tail = days.units.cellid.iloc[1]
    tail_spikes = np.asarray(days.features[tail]['spike_samples'])
    assert ((tail_spikes >= int(10 * FS)) & (tail_spikes < int(14.9 * FS))).sum() > 1
    features[tail] = dict(days.features[tail], stop_sample=int(14.9 * FS),
                          spike_samples=tail_spikes[tail_spikes < int(14.9 * FS)])
    chunked = dict(days.features[tail], stop_sample=int(10 * FS), spike_samples=tail_spikes[tail_spikes < int(10 * FS)])
    (group,), skipped = group_observations(days.units, features, days.geometry, days.timing, min_spikes=2,
                                           n_splits=2)
    assert short not in group.observations
    assert any(entry.get('cellid') == short and 'fewer than 2 5 s chunks' in entry['reason'] for entry in skipped)
    direct = []
    for unit in (features[tail], chunked):
        windows = reference_windows(unit, None, WHOLE)
        splits = trial_splits(windows, unit['start_sample'], unit['stop_sample'], n_splits=2)
        direct.append(unit_observation(unit, days.geometry[group.session_of[tail]], windows, splits, min_spikes=2))
    for observation in (group.observations[tail], direct[0]):
        assert np.isclose(observation['rate_session'], direct[1]['rate_session'])
        assert np.allclose(observation['isi'], direct[1]['isi'], equal_nan=True)
        assert np.allclose(observation['acg'], direct[1]['acg'], equal_nan=True)


def test_event_group_keeps_reliability_gate(shared):
    units, features, geometry, timing = shared
    (group,), _ = group_observations(units, features, geometry, timing, min_spikes=2, min_reliability=-1.0)
    assert group.keys == ('A', 'B')


def test_no_event_group_skips_by_spikes_only(disjoint):
    units, features, geometry, timing = disjoint
    _, skipped = group_observations(units, features, geometry, timing)
    assert skipped
    for entry in skipped:
        assert (entry['reason'].endswith('spikes; this day is not matched')
                or entry['reason'].endswith('spikes; matching needs at least 2'))
        assert 'reliability' not in entry['reason']


def test_empty_keys_give_nan_event_properties():
    days = make_drift_days(n_days=1, n_neurons=3, offsets_um=(0.0,), presence=1.0, seed=3, events=False)
    session = days.sessions[0]
    observations = []
    for cellid in days.units.cellid[:2]:
        feature = days.features[cellid]
        windows = reference_windows(feature, None, WHOLE)
        splits = trial_splits(windows, feature['start_sample'], feature['stop_sample'], n_splits=2)
        observations.append(unit_observation(feature, days.geometry[session], windows, splits))
    properties = pair_properties(*observations, (), 0)
    assert np.isnan(properties['fingerprint_r']) and np.isnan(properties['d_log_rate_shared'])
    assert np.isfinite(properties['isi_cosine'])


def test_require_waveform_halves_names_backend():
    days = make_drift_days(n_days=1, n_neurons=3, offsets_um=(0.0,), presence=1.0, seed=3, events=False)
    features = dict(days.features)
    first = days.units.cellid.iloc[0]
    features[first] = dict(features[first], waveform_halves=None)
    with pytest.raises(ValueError) as caught:
        require_waveform_halves(days.units, features, 'functional_likelihood')
    assert str(caught.value).startswith('functional_likelihood needs waveform halves')


def test_reference_signatures():
    assert list(inspect.signature(reference_windows).parameters) == ['feature', 'timing', 'evidence']
    assert list(inspect.signature(evidence_by_group).parameters) == ['units', 'features', 'timing_by_session',
                                                                      'min_repeats']
    assert list(inspect.signature(chunk_windows).parameters) == ['start', 'stop', 'sample_rate', 'seconds']
