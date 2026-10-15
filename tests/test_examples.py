# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: labeled same/different examples pooled over every project day (functional-likelihood Task 5) | date: 2026-09-27]
"""Labeled examples from every project day: channel alignment, pair properties and the pile's minimum."""
from __future__ import annotations

import itertools
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the jitter tests read signatures, compare frames and patch the examples module (event-optional Task 7) | date: 2026-10-10]
import inspect
# [AGENT EDIT END]

import numpy as np
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the jitter tests read signatures, compare frames and patch the examples module (event-optional Task 7) | date: 2026-10-10]
import pandas as pd
# [AGENT EDIT END]
import pytest

from nems_lbhb.unit_match.backends.local import evidence as ev
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the jitter tests read signatures, compare frames and patch the examples module (event-optional Task 7) | date: 2026-10-10]
from nems_lbhb.unit_match.backends.local import examples
# [AGENT EDIT END]
from nems_lbhb.unit_match.backends.local.examples import (
    MIN_NEIGHBOUR_PAIRS, PROPERTIES, RADIUS_UM, align_channels, fit_ensemble, labeled_examples)

from .synthetic_drift import make_drift_days, observations_by_session, probe_geometry

KEYS = ('sound_0', 'sound_1', 'sound_2', 'sound_3')


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # every test runs in its own tmp_path, so a stray relative write never lands in the repo
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='module')
def drift_observations():
    days = make_drift_days(seed=0)
    by_session = observations_by_session(days)
    observations = {c: o for session in by_session.values() for c, o in session.items()}
    session_of = dict(zip(days.units.cellid, days.units.session_id))
    return observations, session_of


@pytest.fixture(scope='module')
def drift_examples(drift_observations):
    observations, session_of = drift_observations
    return labeled_examples(observations, session_of, KEYS)


def test_refuses_below_minimum_pairs():
    # properties are never computed: these observations carry nothing but a depth
    observations = {f'A-{i}': {'depth': 10.0 * i} for i in range(8)}
    observations.update({'B-0': {'depth': 0.0}, 'B-1': {'depth': 50.0}})
    session_of = {c: c.split('-')[0] for c in observations}
    with pytest.raises(ValueError) as error:
        labeled_examples(observations, session_of, KEYS)
    assert '29' in str(error.value)
    assert 'Rerun with a wider radius' in str(error.value)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refusal names the radius it counted with, following the argument (final review fix round 1) | date: 2026-09-28]
    assert 'within 80 µm' in str(error.value)
    with pytest.raises(ValueError) as narrower:
        labeled_examples(observations, session_of, KEYS, radius_um=60.0)
    assert '28 distinct' in str(narrower.value)
    assert 'within 60 µm' in str(narrower.value)
    # [AGENT EDIT END]


def test_example_counts(drift_observations, drift_examples):
    observations, session_of = drift_observations
    n_pairs = sum(1 for u, v in itertools.combinations(sorted(observations), 2)
                  if session_of[u] == session_of[v]
                  and abs(observations[u]['depth'] - observations[v]['depth']) <= 80.0)
    same = drift_examples[drift_examples.same_unit]
    different = drift_examples[~drift_examples.same_unit]
    assert RADIUS_UM == 80.0 and MIN_NEIGHBOUR_PAIRS == 30
    assert n_pairs >= MIN_NEIGHBOUR_PAIRS
    assert len(same) == len(observations) * 20
    assert len(different) == n_pairs * 20
    assert list(drift_examples.columns) == ['cellid_a', 'cellid_b', 'session', 'split', 'same_unit', *PROPERTIES]
    assert (same.cellid_a == same.cellid_b).all()
    assert (different.cellid_a.map(session_of) == different.session).all()
    assert (different.cellid_b.map(session_of) == different.session).all()
    assert sorted(set(drift_examples.split)) == list(range(20))


def test_days_keep_their_own_split_counts(drift_observations):
    # day A observed on 20 splits, day B on its first 10: each day's rows use that day's splits
    observations, session_of = drift_observations
    sessions = sorted(set(session_of.values()))
    short = sessions[-1]
    trimmed = {c: (dict(o, splits=o['splits'][:10]) if session_of[c] == short else o)
               for c, o in observations.items()}
    examples = labeled_examples(trimmed, session_of, KEYS)
    for session in sessions:
        rows = examples[examples.session == session]
        n_splits = 10 if session == short else 20
        assert sorted(set(rows.split)) == list(range(n_splits))
        n_units = sum(1 for c in observations if session_of[c] == session)
        assert (rows.same_unit.sum()) == n_units * n_splits


def test_refuses_split_counts_differing_within_a_day(drift_observations):
    observations, session_of = drift_observations
    first = sorted(observations)[0]
    trimmed = dict(observations)
    trimmed[first] = dict(observations[first], splits=observations[first]['splits'][:10])
    with pytest.raises(ValueError, match='trial splits'):
        labeled_examples(trimmed, session_of, KEYS)


def test_refuses_a_day_with_no_splits():
    # day A's 28 pairs would lift day B's 3 over the minimum, yet A has no split to compare on;
    # refused before any property is computed (B's placeholder halves are never read)
    observations = {f'A-{i}': {'depth': 10.0 * i, 'splits': []} for i in range(8)}
    observations.update({f'B-{i}': {'depth': 10.0 * i, 'splits': [None]} for i in range(3)})
    session_of = {c: c.split('-')[0] for c in observations}
    with pytest.raises(ValueError, match=r"days \['A'\] carry no trial splits"):
        labeled_examples(observations, session_of, KEYS)


def test_shift_channels_row_multiple_exact():
    geom_a = probe_geometry()
    geom_b = geom_a + [0.0, 40.0]
    v = np.random.default_rng(0).normal(size=(41, len(geom_a)))
    assert np.array_equal(align_channels(v, geom_b, geom_a, -40.0), v)


def test_align_outside_range_is_nan():
    geom_a = probe_geometry()
    geom_b = geom_a + [0.0, 20.0]
    v = np.random.default_rng(1).normal(size=(41, len(geom_a)))
    aligned = align_channels(v, geom_b, geom_a, 0.0)
    bottom = geom_a[:, 1] == 0.0
    assert bottom.sum() == 4
    assert np.isnan(aligned[:, bottom]).all()
    assert np.isfinite(aligned[:, ~bottom]).all()


def test_align_matches_per_sample_interp():
    geom_a = probe_geometry()
    geom_b = geom_a + [0.0, 20.0]
    v = np.random.default_rng(2).normal(size=(41, len(geom_a)))
    aligned = align_channels(v, geom_b, geom_a, -13.7)
    for x in np.unique(geom_a[:, 0]):
        column = np.flatnonzero(geom_b[:, 0] == x)
        target = np.flatnonzero(geom_a[:, 0] == x)
        for sample in (0, 20, 40):
            expected = np.interp(geom_a[target, 1] + 13.7, geom_b[column, 1], v[sample, column],
                                 left=np.nan, right=np.nan)
            np.testing.assert_allclose(aligned[sample, target], expected, rtol=0, atol=1e-12)


def test_align_refuses_other_columns():
    geom_a = probe_geometry()
    other = geom_a.copy()
    other[0, 0] = 99.0
    with pytest.raises(ValueError, match='channel columns differ'):
        align_channels(np.zeros((41, len(geom_a))), geom_a, other, 0.0)


def test_same_examples_compare_own_halves(drift_observations, drift_examples):
    observations, _ = drift_observations
    row = drift_examples[drift_examples.same_unit & (drift_examples.split == 7)].iloc[3]
    halves = observations[row.cellid_a]['splits'][7]
    expected = ev.pearson(np.concatenate([halves[0]['fingerprints'][k] for k in KEYS]),
                          np.concatenate([halves[1]['fingerprints'][k] for k in KEYS]))
    assert np.isfinite(expected)
    assert row.fingerprint_r == expected


def test_fit_ensemble_splits_by_label(drift_examples):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fit_ensemble takes the streams; these tests fit the fixed two (event-optional Task 6) | date: 2026-10-10]
    ensemble, same_ref, diff_ref = fit_ensemble(drift_examples, ev.STREAMS)
    # [AGENT EDIT END]
    assert isinstance(ensemble, ev.Ensemble)
    assert set(same_ref) == set(diff_ref) == set(ev.ALL_PROPERTIES)
    assert same_ref['fingerprint_r'] == list(drift_examples.fingerprint_r[drift_examples.same_unit])
    assert diff_ref['fingerprint_r'] == list(drift_examples.fingerprint_r[~drift_examples.same_unit])
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the peak-to-peak correlations are named footprint, in PROPERTIES and in each waveform_channels subset (event-optional Task 1) | date: 2026-10-09]
def test_properties_name_the_footprint():
    from nems_lbhb.unit_match.backends.local.pair_logistic import WAVEFORM_CHANNELS, model_properties
    assert PROPERTIES == ('fingerprint_r', 'd_log_spont', 'd_log_rate_session', 'd_log_rate_shared', 'isi_cosine',
                          'acg_cosine', 'd_spike_width_ms', 'waveform_r', 'footprint_r', 'waveform_r_local',
                          'footprint_r_local')
    functional = ('fingerprint_r', 'd_log_spont', 'd_log_rate_session', 'd_log_rate_shared', 'isi_cosine',
                  'acg_cosine', 'd_spike_width_ms')
    assert model_properties('peak_threshold') == (*functional, 'waveform_r_local', 'footprint_r_local')
    assert model_properties('full_array') == (*functional, 'waveform_r', 'footprint_r')
    for choice in WAVEFORM_CHANNELS:
        assert len(model_properties(choice)) == 9
        assert not [p for p in model_properties(choice) if p.startswith('template')]
# [AGENT EDIT END]



# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: same-unit rows read half 1 shifted by one seeded jitter draw each; different-unit rows stay unshifted (event-optional Task 7) | date: 2026-10-10]
WAVE_COLUMNS = ('waveform_r', 'footprint_r', 'waveform_r_local', 'footprint_r_local')


@pytest.fixture(scope='module')
def two_units():
    # one day, two neurons 45 µm apart (seed 3), each observed on 2 trial splits
    days = make_drift_days(n_days=1, n_neurons=2, offsets_um=(0.0,), presence=1.0, seed=3)
    observations = observations_by_session(days, n_splits=2)['day0']
    session_of = dict(zip(days.units.cellid, days.units.session_id))
    assert len(observations) == 2
    return observations, session_of


def test_same_rows_jittered_once_each(two_units, monkeypatch):
    observations, session_of = two_units
    real = examples.pair_properties
    calls = []

    def recording(x, y, keys, split, *, offset_um=0.0):
        calls.append((x is y, offset_um))
        return real(x, y, keys, split, offset_um=offset_um)

    monkeypatch.setattr(examples, 'pair_properties', recording)
    rows = labeled_examples(observations, session_of, KEYS, min_pairs=1)
    same = [offset for is_same, offset in calls if is_same]
    different = [offset for is_same, offset in calls if not is_same]
    assert len(same) == int(rows.same_unit.sum()) == 4
    assert len(different) == int((~rows.same_unit).sum()) == 2
    assert same[:2] == pytest.approx((5.478467, -9.208531), abs=1e-6)
    assert all(offset == 0.0 for offset in different)


def test_jitter_zero_and_seed(two_units):
    observations, session_of = two_units
    cellids = sorted(observations)
    # today's rows: same rows then different rows, each at offset 0.0
    compared = [(u, u, True) for u in cellids] + [(u, v, False) for u, v in itertools.combinations(cellids, 2)]
    today = pd.DataFrame([dict(cellid_a=u, cellid_b=v, session='day0', split=split, same_unit=same,
                               **examples.pair_properties(observations[u], observations[v], KEYS, split))
                          for u, v, same in compared for split in range(2)],
                         columns=['cellid_a', 'cellid_b', 'session', 'split', 'same_unit', *PROPERTIES])
    pd.testing.assert_frame_equal(labeled_examples(observations, session_of, KEYS, min_pairs=1, jitter_um=0.0),
                                  today)

    seed_0 = labeled_examples(observations, session_of, KEYS, min_pairs=1, seed=0)
    seed_1 = labeled_examples(observations, session_of, KEYS, min_pairs=1, seed=1)
    same = seed_0.same_unit.to_numpy()
    for column in WAVE_COLUMNS:
        assert not np.array_equal(seed_0.loc[same, column], seed_1.loc[same, column])
    pd.testing.assert_frame_equal(seed_0[~same], seed_1[~same])


def test_examples_signatures():
    parameters = inspect.signature(labeled_examples).parameters
    assert parameters['jitter_um'].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters['jitter_um'].default == 20.0
    assert parameters['seed'].kind is inspect.Parameter.KEYWORD_ONLY
    assert parameters['seed'].default == 0
# [AGENT EDIT END]
