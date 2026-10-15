# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: match-driven depth registration tests (functional-likelihood Task 6) | date: 2026-09-27]
"""Match-driven depth registration: profile seed, match refinement and the functional score."""
from __future__ import annotations

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the signature test reads both registration signatures (event-optional Task 8) | date: 2026-10-10]
import inspect
# [AGENT EDIT END]
import itertools

import numpy as np
import pytest

from nems_lbhb.unit_match.backends.local import evidence as ev
from nems_lbhb.unit_match.backends.local.examples import fit_ensemble, labeled_examples
from nems_lbhb.unit_match.backends.local.registration import functional_offsets, match_offsets, profile_offsets

from .synthetic_drift import make_drift_days, observations_by_session

KEYS = ('sound_0', 'sound_1', 'sound_2', 'sound_3')
NULL = 2.33


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # every test runs in its own tmp_path, so a stray relative write never lands in the repo
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='module')
def drift():
    days = make_drift_days(offsets_um=(0.0, 40.0, -60.0), seed=0)
    return days, observations_by_session(days)


def _same_neuron(days):
    neuron = dict(zip(days.truth.cellid, days.truth.neuron))
    return lambda a, b: 5.0 if neuron[a] == neuron[b] else 0.0


def _relative_errors(offsets, truth):
    return {(a, b): abs((offsets[b] - offsets[a]) - (truth[b] - truth[a]))
            for a, b in itertools.combinations(sorted(truth), 2)}


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - recovery tests pin the zero-mean convention | date: 2026-09-27]
def _assert_mean_zero(offsets):
    assert sum(offsets.values()) / len(offsets) == pytest.approx(0.0, abs=1e-9), offsets
# [AGENT EDIT END]


def test_profile_offsets_recover_shift(drift):
    days, obs = drift
    offsets, measurements = profile_offsets(obs)
    assert len(measurements) == 3
    errors = _relative_errors(offsets, days.truth_offsets)
    assert max(errors.values()) <= 20.0, errors
    _assert_mean_zero(offsets)  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 | date: 2026-09-27]


def test_match_offsets_recover_injected_shift(drift):
    days, obs = drift
    offsets, diagnostics = match_offsets(obs, _same_neuron(days), NULL, {s: 0.0 for s in obs}, [])
    errors = _relative_errors(offsets, days.truth_offsets)
    # the day1-day2 pair is 100 µm apart and day0-day2 60 µm: both must be recovered
    assert abs(days.truth_offsets['day2'] - days.truth_offsets['day0']) == pytest.approx(60.0)
    assert max(errors.values()) <= 10.0, errors
    _assert_mean_zero(offsets)  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 | date: 2026-09-27]
    assert {p['source'] for p in diagnostics['pairs']} == {'matches'}
    assert diagnostics['null_threshold'] == NULL


def test_failed_seed_recovered(drift):
    days, obs = drift
    seed = dict(days.truth_offsets)
    seed['day1'] += 300.0
    offsets, diagnostics = match_offsets(obs, _same_neuron(days), NULL, seed, [])
    errors = _relative_errors(offsets, days.truth_offsets)
    assert max(errors.values()) <= 10.0, errors
    _assert_mean_zero(offsets)  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 | date: 2026-09-27]
    assert diagnostics['converged'] is True


def test_no_matches_keeps_profile(drift):
    days, obs = drift
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - wrong starting offsets and hand-set measurements, so the result must be solved, not the seed returned | date: 2026-09-27]
    # z1 - z0 = 40, z2 - z0 = -60, z2 - z1 = -100 (consistent); mean 0 gives z0 = 20/3
    seed_measurements = [('day0', 'day1', 40.0, 1.0), ('day0', 'day2', -60.0, 1.0), ('day1', 'day2', -100.0, 1.0)]
    expected = {'day0': 20.0 / 3, 'day1': 20.0 / 3 + 40.0, 'day2': 20.0 / 3 - 60.0}
    seed_offsets = {'day0': 300.0, 'day1': -500.0, 'day2': 0.0}
    offsets, diagnostics = match_offsets(obs, lambda a, b: 0.0, NULL, seed_offsets, seed_measurements)
    assert offsets == pytest.approx(expected)
    assert [p['offset_um'] for p in diagnostics['pairs']] == [40.0, -60.0, -100.0]
    assert diagnostics['groups'] == [['day0', 'day1', 'day2']]
    # [AGENT EDIT END]
    assert len(diagnostics['pairs']) == 3
    assert all(p['source'] == 'profile' for p in diagnostics['pairs'])
    assert all(p['n_matches'] == 0 for p in diagnostics['pairs'])


def test_functional_offsets_recover_synthetic_drift(drift):
    days, obs = drift
    observations = {c: o for session in obs.values() for c, o in session.items()}
    session_of = dict(zip(days.units.cellid, days.units.session_id))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fit_ensemble takes the streams; these tests fit the fixed two (event-optional Task 6) | date: 2026-10-10]
    ensemble, _, diff_ref = fit_ensemble(labeled_examples(observations, session_of, KEYS), ev.STREAMS)
    # [AGENT EDIT END]
    offsets, diagnostics = functional_offsets(obs, ensemble, diff_ref, KEYS)
    errors = _relative_errors(offsets, days.truth_offsets)
    assert max(errors.values()) <= 10.0, errors
    _assert_mean_zero(offsets)  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 | date: 2026-09-27]
    assert diagnostics['method'] == 'matches' and diagnostics['null_quantile'] == 0.99
    assert {p['source'] for p in diagnostics['pairs']} == {'matches'}


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - unusable measurements, non-finite scores and a single session | date: 2026-09-27]
@pytest.mark.parametrize('measurement', [('day0', 'day1', -40.0, 0.0), ('day0', 'day1', np.nan, 1.0),
                                         ('day0', 'day1', -40.0, np.nan)])
def test_unusable_profile_measurement_is_no_source(drift, measurement):
    _, obs = drift
    two = {s: obs[s] for s in ('day0', 'day1')}
    offsets, diagnostics = match_offsets(two, lambda a, b: 0.0, NULL, {'day0': 0.0, 'day1': 0.0}, [measurement])
    assert offsets == {'day0': 0.0, 'day1': 0.0}
    [pair] = diagnostics['pairs']
    assert pair['source'] == 'none' and np.isnan(pair['offset_um'])
    assert diagnostics['groups'] == [['day0'], ['day1']]


def test_nonfinite_score_raises(drift):
    _, obs = drift
    with pytest.raises(ValueError, match='non-finite pair scores'):
        match_offsets(obs, lambda a, b: float('nan'), NULL, {s: 0.0 for s in obs}, [])


def test_single_session_is_zero(drift):
    _, obs = drift
    one = {'day0': obs['day0']}
    offsets, measurements = profile_offsets(one)
    assert offsets == {'day0': 0.0} and measurements == []
    offsets, diagnostics = match_offsets(one, lambda a, b: 5.0, NULL, {'day0': 12.0}, [])
    assert offsets == {'day0': 0.0}
    assert diagnostics['pairs'] == [] and diagnostics['groups'] == [['day0']]
    with pytest.raises(ValueError, match='0 sessions'):
        profile_offsets({})
# [AGENT EDIT END]


def test_functional_offsets_methods(drift):
    days, obs = drift
    observations = {c: o for session in obs.values() for c, o in session.items()}
    session_of = dict(zip(days.units.cellid, days.units.session_id))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fit_ensemble takes the streams; these tests fit the fixed two (event-optional Task 6) | date: 2026-10-10]
    ensemble, _, diff_ref = fit_ensemble(labeled_examples(observations, session_of, KEYS), ev.STREAMS)
    # [AGENT EDIT END]
    none, diagnostics = functional_offsets(obs, ensemble, diff_ref, KEYS, method='none')
    assert none == {s: 0.0 for s in obs}
    assert all(p['source'] == 'none' for p in diagnostics['pairs'])
    profile, _ = functional_offsets(obs, ensemble, diff_ref, KEYS, method='profile')
    assert profile == pytest.approx(profile_offsets(obs)[0])
    with pytest.raises(ValueError, match='method'):
        functional_offsets(obs, ensemble, diff_ref, KEYS, method='gate')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - guard the threshold, search bound, mutuality, seed, minimum support and quantile rules | date: 2026-09-27]
def _depth_obs(depths: dict) -> dict:
    """session -> cellid -> {'depth': d}: all ``match_offsets`` reads from an observation."""
    return {s: {c: {'depth': float(d)} for c, d in units.items()} for s, units in depths.items()}


@pytest.mark.parametrize('threshold', [np.nan, np.inf, -np.inf])
def test_nonfinite_null_threshold_raises(drift, threshold):
    days, obs = drift
    with pytest.raises(ValueError, match='null_threshold'):
        match_offsets(obs, _same_neuron(days), threshold, {s: 0.0 for s in obs}, [])


def test_profile_offsets_respect_max_shift(drift):
    _, obs = drift
    # the day1-day2 shift is 100 µm; a 20 µm search bound cannot report more than 20 µm
    _, measurements = profile_offsets(obs, max_shift_um=20.0)
    assert max(abs(m) for _, _, m, _ in measurements) <= 20.0, measurements
    _, unbounded = profile_offsets(obs)
    assert max(abs(m) for _, _, m, _ in unbounded) > 20.0, unbounded


def test_matches_are_mutual_best():
    # a1 and a2 both prefer b1, but b1 prefers a1: only (a1, b1) is a match
    obs = _depth_obs({'day0': {'a1': 100.0, 'a2': 200.0}, 'day1': {'b1': 100.0}})
    scores = {('a1', 'b1'): 5.0, ('a2', 'b1'): 4.0}
    offsets, diagnostics = match_offsets(obs, lambda a, b: scores[(a, b)], NULL, {'day0': 0.0, 'day1': 0.0}, [],
                                         min_matches=1)
    [pair] = diagnostics['pairs']
    assert pair['source'] == 'matches' and pair['n_matches'] == 1
    assert pair['offset_um'] == 0.0  # one-way best pairs would give median(0, 100) = 50
    assert offsets == {'day0': 0.0, 'day1': 0.0}


def test_seed_sets_candidate_window(drift):
    days, obs = drift
    # a 20 µm window finds the 100 µm day1-day2 pair's matches only around the seed's offset
    offsets, diagnostics = match_offsets(obs, _same_neuron(days), NULL, dict(days.truth_offsets), [],
                                         windows_um=(20.0,))
    errors = _relative_errors(offsets, days.truth_offsets)
    assert max(errors.values()) <= 10.0, errors
    assert {p['source'] for p in diagnostics['pairs']} == {'matches'}
    _, zero = match_offsets(obs, _same_neuron(days), NULL, {s: 0.0 for s in obs}, [], windows_um=(20.0,))
    assert {p['source'] for p in zero['pairs']} != {'matches'}


@pytest.mark.parametrize('n_true, source', [(1, 'profile'), (2, 'profile'), (3, 'matches')])
def test_min_matches_needs_three(n_true, source):
    cells = [f'u{k}' for k in range(3)]
    obs = _depth_obs({'day0': {f'a_{c}': 100.0 * k for k, c in enumerate(cells)},
                      'day1': {f'b_{c}': 100.0 * k + 30.0 for k, c in enumerate(cells)}})
    matched = set(cells[:n_true])
    score = lambda a, b: 5.0 if a[2:] == b[2:] and a[2:] in matched else 0.0  # noqa: E731
    _, diagnostics = match_offsets(obs, score, NULL, {'day0': 0.0, 'day1': 0.0}, [('day0', 'day1', -30.0, 1.0)])
    [pair] = diagnostics['pairs']
    assert pair['n_matches'] == n_true and pair['source'] == source
    assert pair['offset_um'] == -30.0


def test_functional_offsets_null_quantile(drift):
    days, obs = drift
    observations = {c: o for session in obs.values() for c, o in session.items()}
    session_of = dict(zip(days.units.cellid, days.units.session_id))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fit_ensemble takes the streams; these tests fit the fixed two (event-optional Task 6) | date: 2026-10-10]
    ensemble, _, diff_ref = fit_ensemble(labeled_examples(observations, session_of, KEYS), ev.STREAMS)
    # [AGENT EDIT END]
    reference = np.asarray(ensemble.reference_scores(diff_ref)['s_global'], dtype=float)
    expected = float(np.quantile(reference[np.isfinite(reference)], 0.9))
    for method in ('none', 'profile', 'matches'):
        _, diagnostics = functional_offsets(obs, ensemble, diff_ref, KEYS, method=method, null_quantile=0.9)
        assert diagnostics['null_threshold'] == expected and diagnostics['null_quantile'] == 0.9
    assert expected != float(np.quantile(reference[np.isfinite(reference)], 0.99))
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an empty offset schedule refuses instead of reaching an unbound name (Plan B Task 9) | date: 2026-09-28]
def test_match_offsets_refuses_empty_schedule():
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refusal comes before any work, so no drift days are built (Plan B Task 9 fix round 2) | date: 2026-09-29]
    obs = {'s0': {}, 's1': {}}
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='needs at least one window and one iteration'):
        match_offsets(obs, lambda a, b: 0.0, NULL, {s: 0.0 for s in obs}, [], windows_um=(), max_iter=0)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each empty-schedule condition refuses on its own (Plan B Task 9 fix round 1) | date: 2026-09-29]
    for windows_um, max_iter in [((), 3), ((50.0,), 0)]:
        expected = (f'match_offsets needs at least one window and one iteration; '
                    f'got windows_um={windows_um!r}, max_iter={max_iter}')
        with pytest.raises(ValueError) as err:
            match_offsets(obs, lambda a, b: 0.0, NULL, {s: 0.0 for s in obs}, [],
                          windows_um=windows_um, max_iter=max_iter)
        assert str(err.value) == expected
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a negative iteration count and an empty schedule refuse before any other check (Follow-up Task 5) | date: 2026-09-29]
def test_match_offsets_refuses_negative_iterations():
    with pytest.raises(ValueError) as err:
        match_offsets({'s0': {}, 's1': {}}, lambda a, b: 0.0, NULL, {'s0': 0.0, 's1': 0.0}, [],
                      windows_um=(50.0,), max_iter=-1)
    assert str(err.value) == ('match_offsets needs at least one window and one iteration; '
                              'got windows_um=(50.0,), max_iter=-1')


def test_match_offsets_schedule_guard_runs_first():
    # no sessions: the schedule error, not '0 sessions to register'
    with pytest.raises(ValueError) as err:
        match_offsets({}, lambda a, b: 0.0, NULL, {}, [], windows_um=(), max_iter=3)
    assert str(err.value).startswith('match_offsets needs at least one window')
    # no seed offsets: the schedule error, not '2 sessions have no seed offset'
    with pytest.raises(ValueError) as err:
        match_offsets({'s0': {}, 's1': {}}, lambda a, b: 0.0, NULL, {}, [], windows_um=(), max_iter=3)
    assert str(err.value).startswith('match_offsets needs at least one window')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: aligned registration scores each pair at every relative offset tried (event-optional Task 8) | date: 2026-10-10]
def test_aligned_scores_at_each_offset():
    obs = _depth_obs({'s1': {f'a{k}': d for k, d in enumerate([0.0, 100.0, 200.0, 300.0])},
                      's2': {f'b{k}': d for k, d in enumerate([-40.0, 60.0, 160.0, 260.0])}})
    seeds = {'s1': 0.0, 's2': 0.0}
    plain_calls, aligned_calls = [], []

    def plain(a, b):
        plain_calls.append((a, b))
        return 1.0 if a[1:] == b[1:] else 0.0

    def aligned(a, b, rel):
        aligned_calls.append((a, b, rel))
        return 1.0 if a[1:] == b[1:] else 0.0

    plain_offsets, _ = match_offsets(obs, plain, 0.5, seeds, [])
    assert len(plain_calls) == 16
    assert plain_offsets['s1'] == pytest.approx(-20.0, abs=1e-9)
    assert plain_offsets['s2'] == pytest.approx(20.0, abs=1e-9)
    aligned_offsets, _ = match_offsets(obs, aligned, 0.5, seeds, [], aligned=True)
    assert len(aligned_calls) == 32
    assert [rel for a, b, rel in aligned_calls if (a, b) == ('a0', 'b0')] == pytest.approx([0.0, 40.0], abs=1e-9)
    assert aligned_offsets['s1'] == pytest.approx(-20.0, abs=1e-9)
    assert aligned_offsets['s2'] == pytest.approx(20.0, abs=1e-9)


def test_registration_signatures():
    for function in (match_offsets, functional_offsets):
        parameter = inspect.signature(function).parameters['aligned']
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY, function.__name__
        assert parameter.default is False, function.__name__
# [AGENT EDIT END]
