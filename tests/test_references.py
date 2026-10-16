"""Trial splits, per-split unit halves, ``TrialTiming.from_source_entry`` and synthetic drift days."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pytest

from multiphy.backends.local.references import (
    alternating_references, trial_splits, unit_halves, unit_observation)
from multiphy.contract.inputs import TrialTiming
from multiphy.project.days import add_day
from multiphy.project.folder import Project
from multiphy.project.identity import FileIdentity
from multiphy.sources.base import clear_readers_for_tests, read_day
from multiphy.sources.normalize import read_source, write_source

from .synthetic_day import write_day
from .synthetic_drift import make_drift_days


@pytest.fixture(scope='module')
def small_days():
    return make_drift_days(n_days=1, n_neurons=3, offsets_um=(0.0,), presence=1.0, seed=3)


def _unit(days):
    cellid = days.units.cellid.iloc[0]
    session = days.units.session_id.iloc[0]
    return days.features[cellid], days.geometry[session], days.timing[session]


def _entry(**changes):
    entry = {'source_session': 'day1', 'sample_rate': 1000.0, 'analysis_window': [0, 2000],
             'trials': [[0, 400], [500, 900], [1000, 1400], [1500, 1900]],
             'stimuli': [{'stimulus': 'A', 'start': i + 100, 'stop': i + 300, 'trial': j + 1}
                         for j, i in enumerate(range(0, 2000, 500))],
             'baseline_windows': [[i, i + 100] for i in range(0, 2000, 500)],
             'clock_verification': {'sample_index_origin': 'zero_based_voltage_row', 'method': 'test clock'},
             'day': 'day1', 'project': 'P'}
    entry.update(changes)
    return entry


FS = 1000.0
TRIAL, BASELINE, STIMULUS = 1000, 100, 500     # samples: trial length, baseline length, stimulus length


def _hand_timing(counts: dict) -> TrialTiming:
    """Trials of 1 s at 1 kHz, conditions round-robin: baseline [t, t+100), stimulus [t+100, t+600)."""
    order, remaining = [], dict(counts)
    while any(remaining.values()):
        for condition in sorted(remaining):
            if remaining[condition]:
                order.append(condition)
                remaining[condition] -= 1
    starts = [i * TRIAL for i in range(len(order))]
    return TrialTiming(
        source_session='hand', sample_rate=FS, analysis_window=(0, len(order) * TRIAL),
        trial_windows=np.asarray([[t, t + TRIAL] for t in starts]),
        stimuli=tuple({'stimulus': c, 'start': t + BASELINE, 'stop': t + BASELINE + STIMULUS, 'trial': i + 1}
                      for i, (c, t) in enumerate(zip(order, starts))),
        baseline_windows=np.asarray([[t, t + BASELINE] for t in starts]),
        provenance={'sample_index_origin': 'zero_based_voltage_row', 'method': 'hand'}).validate()


def _expected_splits(counts: dict, n_trials: int, n_splits: int, seed: int) -> list:
    """The brief's draw written out: split k uses default_rng([seed, k]) on trials, baselines, sorted conditions."""
    def draw(rng, n):
        perm = rng.permutation(n)
        half = np.ones(n, int)
        half[perm[:n // 2]] = 0
        return half
    expected = [(np.arange(n_trials) % 2, np.arange(n_trials) % 2, {c: np.arange(n) % 2 for c, n in counts.items()})]
    for k in range(1, n_splits):
        rng = np.random.default_rng([seed, k])
        trial, baseline = draw(rng, n_trials), draw(rng, n_trials)
        expected.append((trial, baseline, {c: draw(rng, counts[c]) for c in sorted(counts)}))
    return expected


def test_split_zero_is_alternating(small_days):
    feature, _, timing = _unit(small_days)
    start, stop = timing.analysis_window
    split = trial_splits(timing, start, stop)[0]
    halves = unit_halves(feature, timing, split)
    reference = alternating_references(feature, timing)
    assert reference['halves'] and len(halves) == 2
    for h, suffix in ((0, '_a'), (1, '_b')):
        half = halves[h]
        assert sorted(half['fingerprints']) == sorted(reference['halves'])
        for condition, pair in reference['halves'].items():
            assert np.allclose(half['fingerprints'][condition], pair[h])
        assert np.allclose(half['spont'], reference['spont' + suffix])
        assert np.allclose(half['isi'], reference['isi' + suffix])
        assert np.allclose(half['acg'], reference['acg' + suffix])
    # the two halves are different data, not one half returned twice
    assert not all(np.allclose(halves[0]['fingerprints'][c], halves[1]['fingerprints'][c])
                   for c in reference['halves'])


def test_split_zero_hand_counted():
    timing = _hand_timing({'A': 5})
    spikes = []
    for r, stimulus in enumerate(timing.stimuli):
        spikes += [stimulus['start'] + 10 + j for j in range((r + 1) ** 2)]                  # bin 0
        spikes.append(stimulus['start'] + 250)                                               # bin 2
        if r % 2 == 0:
            spikes += [stimulus['start'] - BASELINE + 5, stimulus['start'] - BASELINE + 7]   # baseline
    feature = {'start_sample': 0, 'stop_sample': timing.analysis_window[1], 'sample_rate': FS,
               'spike_samples': np.asarray(sorted(spikes), dtype=np.int64), 'fingerprint_bin_seconds': 0.1}
    halves = unit_halves(feature, timing, trial_splits(timing, 0, timing.analysis_window[1], n_splits=1)[0])
    # repeats 0, 2, 4 (half 0) carry 1, 9, 25 bin-0 spikes; repeats 1, 3 (half 1) carry 4, 16; bins are 0.1 s
    assert np.allclose(halves[0]['fingerprints']['A'], [35 / 3 / 0.1, 0, 10, 0, 0])
    assert np.allclose(halves[1]['fingerprints']['A'], [10 / 0.1, 0, 10, 0, 0])
    assert np.isclose(halves[0]['per_stim']['A'], (35 + 3) / (3 * 0.5))
    assert np.isclose(halves[1]['per_stim']['A'], (20 + 2) / (2 * 0.5))
    assert np.isclose(halves[0]['spont'], 2 / 0.1) and halves[1]['spont'] == 0.0
    assert halves[0]['n_spikes'] == 35 + 3 + 6 and halves[1]['n_spikes'] == 20 + 2
    assert np.isclose(halves[0]['rate'], 44 / 3.0) and np.isclose(halves[1]['rate'], 22 / 2.0)


def test_splits_are_seeded(small_days):
    timing = small_days.timing['day0']
    start, stop = timing.analysis_window

    def arrays(splits):
        return [a for s in splits for a in (s.trial_half, s.baseline_half, *(s.stimulus_half[c] for c in sorted(s.stimulus_half)))]

    first, again = arrays(trial_splits(timing, start, stop, seed=0)), arrays(trial_splits(timing, start, stop, seed=0))
    assert all(np.array_equal(a, b) for a, b in zip(first, again)) and len(first) == len(again)
    other = arrays(trial_splits(timing, start, stop, seed=1))
    assert any(not np.array_equal(a, b) for a, b in zip(first, other))
    # split k draws from default_rng([seed, k]), trials first
    n = len(timing.trial_windows)
    perm = np.random.default_rng([0, 3]).permutation(n)
    expected = np.ones(n, int)
    expected[perm[:n // 2]] = 0
    assert np.array_equal(trial_splits(timing, start, stop, seed=0)[3].trial_half, expected)


def test_splits_follow_seed_per_array():
    counts = {'A': 5, 'B': 7, 'C': 4}
    timing = _hand_timing(counts)
    for seed in (0, 1):
        splits = trial_splits(timing, *timing.analysis_window, n_splits=6, seed=seed)
        for split, (trial, baseline, stimulus) in zip(splits, _expected_splits(counts, 16, 6, seed)):
            assert np.array_equal(split.trial_half, trial)
            assert np.array_equal(split.baseline_half, baseline)
            assert set(split.stimulus_half) == set(stimulus)
            assert all(np.array_equal(split.stimulus_half[c], stimulus[c]) for c in counts)
    zero = trial_splits(timing, *timing.analysis_window, n_splits=6, seed=0)
    one = trial_splits(timing, *timing.analysis_window, n_splits=6, seed=1)
    # the seed moves each kind of array, not only the trials
    getters = [lambda s: s.trial_half, lambda s: s.baseline_half] + [lambda s, c=c: s.stimulus_half[c] for c in counts]
    for get in getters:
        assert any(not np.array_equal(get(a), get(b)) for a, b in zip(zero[1:], one[1:]))


def test_splits_are_stratified(small_days):
    timing = small_days.timing['day0']
    # expected counts come from the fixture's construction (4 conditions x 10 repeats, one baseline
    # per trial), cross-checked against the timing, never from the returned masks
    counts: dict = {}
    for stimulus in timing.stimuli:
        counts[stimulus['stimulus']] = counts.get(stimulus['stimulus'], 0) + 1
    assert counts == {f'sound_{c}': 10 for c in range(4)}
    assert len(timing.trial_windows) == 40 and len(timing.baseline_windows) == 40
    splits = trial_splits(timing, *timing.analysis_window)
    assert len(splits) == 20
    for split in splits[1:]:
        assert set(split.stimulus_half) == set(counts)
        for condition, n in counts.items():
            halves = np.asarray(split.stimulus_half[condition])
            assert len(halves) == n and set(halves.tolist()) <= {0, 1}
            assert (halves == 0).sum() == n // 2
        for halves in (split.trial_half, split.baseline_half):
            halves = np.asarray(halves)
            assert len(halves) == 40 and set(halves.tolist()) <= {0, 1}
            assert (halves == 0).sum() == 20


def test_splits_are_stratified_odd_counts():
    timing = _hand_timing({'A': 5, 'B': 6})          # 11 trials and baselines
    for split in trial_splits(timing, *timing.analysis_window, n_splits=10)[1:]:
        assert (np.asarray(split.stimulus_half['A']) == 0).sum() == 2
        assert (np.asarray(split.stimulus_half['B']) == 0).sum() == 3
        assert (np.asarray(split.trial_half) == 0).sum() == 5 and len(split.trial_half) == 11
        assert (np.asarray(split.baseline_half) == 0).sum() == 5 and len(split.baseline_half) == 11


def test_silent_unit_has_zero_rates(small_days):
    feature, _, timing = _unit(small_days)
    silent = dict(feature, spike_samples=np.array([], dtype=np.int64))
    split = trial_splits(timing, *timing.analysis_window, n_splits=2)[1]
    for half in unit_halves(silent, timing, split):
        assert half['n_spikes'] == 0 and half['rate'] == 0.0 and half['spont'] == 0.0
        assert half['fingerprints'] and all(np.all(p == 0) for p in half['fingerprints'].values())
        assert all(rate == 0.0 for rate in half['per_stim'].values())


def test_too_few_repeats_are_named(small_days):
    feature, _, timing = _unit(small_days)
    split = trial_splits(timing, *timing.analysis_window, n_splits=1)[0]
    for half in unit_halves(feature, timing, split, min_repeats=11):
        assert half['fingerprints'] == {} and half['per_stim'] == {}
        assert sorted(half['excluded']) == [f'sound_{c}' for c in range(4)]
        assert all('10 repeats' in r and 'min_repeats=11' in r for r in half['excluded'].values())


def test_from_source_entry():
    entry = _entry()
    timing = TrialTiming.from_source_entry(entry)
    assert timing.trial_windows.tolist() == entry['trials']
    assert timing.provenance['day'] == 'day1' and timing.provenance['project'] == 'P'
    assert timing.provenance['method'] == 'test clock'
    assert timing.source_session == 'day1' and timing.sample_rate == 1000.0 and timing.analysis_window == (0, 2000)
    assert list(timing.stimuli) == entry['stimuli'] and timing.baseline_windows.tolist() == entry['baseline_windows']
    assert timing.provenance['sample_index_origin'] == 'zero_based_voltage_row'


def test_from_source_entry_rejects_invalid_entries():
    with pytest.raises(ValueError, match='trial_windows must contain complete ordered disjoint'):
        TrialTiming.from_source_entry(_entry(trials=[[0, 400], [500, 900], [1000, 1400], [1500, 2100]]))
    with pytest.raises(ValueError, match='Verified acquisition-clock timing provenance is required'):
        TrialTiming.from_source_entry(_entry(clock_verification={'method': 'test clock'}))
    with pytest.raises(ValueError, match='outside its nominal TRIALSTART/TRIALSTOP'):
        TrialTiming.from_source_entry(_entry(baseline_windows=[[400, 450]]))


def test_from_source_entry_needs_trials():
    missing = _entry()
    del missing['trials']
    for entry in (missing, _entry(trials=[])):
        with pytest.raises(ValueError, match=re.escape(
                "day day1: timing needs trial events and this day's source has none; add an events reader")):
            TrialTiming.from_source_entry(entry)


def test_from_source_entry_on_a_phy_day(tmp_path):
    clear_readers_for_tests()
    try:
        sd = write_day(tmp_path / 'in', 'D1')
        project = Project.create(tmp_path / 'lab', 'P')
        readers = {'sort': 'phy', 'events': 'events_csv', 'raw': 'binary'}
        options = {'day': 'D1', 'sort_dir': str(sd.sort_dir), 'events_path': str(sd.events_path),
                   'raw_path': str(sd.raw_path), 'uv_per_bit': 0.195}
        record = add_day(project, 'D1', readers=dict(readers), reader_options=dict(options),
                         date_source='user', date=sd.date)
        files: dict = {}
        for key, identity in record['files'].items():
            files.setdefault(identity['role'], {})[key] = Path(identity['resolved'])
        day_source = read_day(record['readers'], files, record['reader_options'])
        write_source(day_source, tmp_path / 'source', day='D1', project=project.name, readers=record['readers'],
                     identities={key: FileIdentity.from_dict(i) for key, i in record['files'].items()})
        entry = read_source(tmp_path / 'source')['entry']
    finally:
        clear_readers_for_tests()
    with open(sd.events_path, newline='') as handle:
        expected = [[int(r['start']), int(r['stop'])] for r in csv.DictReader(handle) if r['kind'] == 'trial']
    timing = TrialTiming.from_source_entry(entry)
    assert timing.trial_windows.tolist() == expected
    assert timing.provenance['day'] == 'D1'
    with open(sd.events_path, newline='') as handle:
        rows = list(csv.DictReader(handle))
    assert timing.baseline_windows.tolist() == [[int(r['start']), int(r['stop'])] for r in rows if r['kind'] == 'baseline']
    assert [(s['stimulus'], s['start'], s['stop']) for s in timing.stimuli] == [
        (r['condition'], int(r['start']), int(r['stop'])) for r in rows if r['kind'] == 'stimulus']
    assert timing.sample_rate == 30000.0 and timing.source_session == entry['source_session']
    assert timing.provenance['sample_index_origin'] == 'zero_based_voltage_row'


def test_unit_observation_has_splits(small_days):
    feature, geometry, timing = _unit(small_days)
    splits = trial_splits(timing, *timing.analysis_window, n_splits=20)
    obs = unit_observation(feature, geometry, timing, splits)
    assert len(obs['splits']) == 20
    assert obs['waveform_halves'].shape == (41, 96, 2)
    assert len(obs['sw_halves']) == 2 and all(np.isfinite(obs['sw_halves']))
    spikes = np.asarray(feature['spike_samples'])
    trials = np.asarray(timing.trial_windows)
    for split, (half_0, half_1) in zip(splits, obs['splits']):
        for h, half in ((0, half_0), (1, half_1)):
            own = trials[np.asarray(split.trial_half) == h]
            assert half['n_spikes'] == sum(int(((spikes >= a) & (spikes < b)).sum()) for a, b in own)
    assert len({half_0['n_spikes'] for half_0, _ in obs['splits']}) > 1


def test_drift_days_truth_offsets():
    days = make_drift_days()
    assert days.truth_offsets['day2'] - days.truth_offsets['day0'] == 60.0
    assert np.allclose([days.truth_offsets[f'day{k}'] for k in range(3)], [-20 / 3, -140 / 3, 160 / 3])
