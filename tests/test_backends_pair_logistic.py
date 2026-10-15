# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic backend - calibrated per-project classifier (functional-likelihood Task 8) | date: 2026-09-27]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: module preamble required of every new module (Task 8 fix round 1) | date: 2026-09-27]
"""Tests of the ``pair_logistic`` backend on synthetic drift days: refusal of the templates tier,
track recovery, registration, held-out-day calibration and its numerical probabilities."""
from __future__ import annotations

# [AGENT EDIT END]
import copy
import re

import numpy as np
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the chunk-provenance test joins two groups' unit tables (event-optional Task 10 fix round 1) | date: 2026-10-10]
import pandas as pd
# [AGENT EDIT END]
import pytest

import nems_lbhb.unit_match.backends  # noqa: F401  (registers the built-ins)
from nems_lbhb.unit_match.backends.local import pair_logistic
from nems_lbhb.unit_match.backends.local.pair_logistic import SCORE_COLUMN, run_pair_logistic
from nems_lbhb.unit_match.contract import registry
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the spec test reads Input (event-optional Task 10) | date: 2026-10-10]
from nems_lbhb.unit_match.contract.inputs import Input
# [AGENT EDIT END]
from nems_lbhb.unit_match.contract.results import ScoreKind
from nems_lbhb.unit_match.evaluation import evaluate_result

from .synthetic_drift import ground_truth, make_drift_days


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # every test runs in its own tmp_path, so a stray relative write never lands in the repo
    monkeypatch.chdir(tmp_path)


def _run(days, **options):
    return run_pair_logistic(days.units, days.features, days.geometry, days.timing, **options)


@pytest.fixture(scope='module')
def drift_run():
    days = make_drift_days(seed=0)
    return days, _run(days)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an absent feature record is refused like absent waveform halves; the whole message is pinned (Task 8 fix round 1) | date: 2026-09-27]
@pytest.mark.parametrize('absent', ['waveform_halves', 'feature_record'])
def test_pair_logistic_refuses_templates_tier(absent):
    days = make_drift_days(n_neurons=8, seed=2)
    features = copy.deepcopy(days.features)
    cellid = days.units.cellid.iloc[3]
    if absent == 'waveform_halves':
        del features[cellid]['waveform_halves']
    else:
        del features[cellid]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refusal is the shared require_waveform_halves message, which names no fallback backend (event-optional Task 10) | date: 2026-10-10]
    message = (f'pair_logistic needs waveform halves from raw voltage; 1 units have none, first {cellid} '
               '(templates tier). Re-extract from raw voltage.')
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match=f'^{re.escape(message)}$'):
        run_pair_logistic(days.units, features, days.geometry, days.timing)
# [AGENT EDIT END]


def test_pair_logistic_recovers_synthetic_tracks(drift_run):
    days, result = drift_run
    scores = evaluate_result(result, ground_truth(days), expected_partition=None)
    assert scores['pair_precision'] >= 0.9, scores
    assert scores['pair_recall'] >= 0.6, scores


def test_registry_holds_the_five_methods():
    registry.get('pair_logistic')  # built-ins loaded
    assert sorted(registry.registered_ids()) == ['dant', 'deepunitmatch', 'functional_likelihood', 'pair_logistic',
                                                 'unitmatch']


def test_train_and_calibration_days_disjoint(drift_run):
    days, result = drift_run
    assert days.sessions == ('day0', 'day1', 'day2')
    assert result.provenance.get('training_days') == ['day0', 'day2']
    assert result.provenance.get('calibration_days') == ['day1']
    model = result.provenance.get('calibrated_model')['synthetic']
    assert model['training_groups'] == ['day0', 'day2'] and model['calibration_groups'] == ['day1']


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: behavioral check that the held-out day reaches only the calibration, never the classifier (Task 8 fix round 1) | date: 2026-09-27]
def test_held_out_day_moves_only_the_calibration(drift_run, monkeypatch):
    days, baseline = drift_run
    real_examples = pair_logistic.labeled_examples

    def perturbed_held_out_day(*args, **kwargs):
        examples = real_examples(*args, **kwargs).copy()
        held_out = (examples.session == 'day1').to_numpy()
        rng = np.random.default_rng(7)
        for p in pair_logistic.PROPERTIES:
            values = examples[p].to_numpy(dtype=float)
            values[held_out] = values[held_out] * 1.5 + rng.normal(0, 1, held_out.sum()) * np.nanstd(values)
            examples[p] = values
        return examples

    monkeypatch.setattr(pair_logistic, 'labeled_examples', perturbed_held_out_day)
    moved = _run(days)
    before = baseline.provenance.get('calibrated_model')['synthetic']
    after = moved.provenance.get('calibrated_model')['synthetic']
    for key in ('center', 'scale', 'weights'):
        np.testing.assert_array_equal(after[key], before[key], err_msg=key)
    assert not np.allclose(after['calibration'], before['calibration'])
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: independent oracle of the fit: the classifier minimizes its objective on the training days only and the calibration minimizes its objective on the held-out day only, where pooling moves both measurably (Task 8 fix round 2) | date: 2026-09-27]
def _design(x, center, scale):
    missing = ~np.isfinite(x)
    return np.column_stack(((np.where(missing, center, x) - center) / scale, missing.astype(float)))


def _classifier_objective(w, design, y, l2):
    # penalized logistic loss on labels in {-1, +1}; the intercept is last and unpenalized
    z = np.clip(design @ w[:-1] + w[-1], -30, 30)
    return float(np.sum(np.log1p(np.exp(-y * z))) + l2 * np.sum(w[:-1] ** 2))


def _calibration_objective(ab, score, y, l2):
    z = ab[0] * score + ab[1]
    return float(np.sum(np.logaddexp(0., z) - y * z) + l2 * ab[0] ** 2)


def test_fit_uses_training_days_for_the_classifier_and_the_held_out_day_for_calibration(monkeypatch):
    from scipy.optimize import minimize
    days = make_drift_days(seed=0)
    captured = []
    real_examples = pair_logistic.labeled_examples

    def recording(*args, **kwargs):
        examples = real_examples(*args, **kwargs)
        captured.append(examples.copy())
        return examples

    monkeypatch.setattr(pair_logistic, 'labeled_examples', recording)
    model = _run(days).provenance.get('calibrated_model')['synthetic']
    assert len(captured) == 1
    examples, props, l2 = captured[0], model['properties'], model['l2']
    center, scale, weights = (np.asarray(model[k], dtype=float) for k in ('center', 'scale', 'weights'))
    slope_intercept = np.asarray(model['calibration'], dtype=float)
    rows = {'training': examples.session.isin(['day0', 'day2']).to_numpy(),
            'held_out': (examples.session == 'day1').to_numpy(),
            'pooled': np.ones(len(examples), dtype=bool)}
    assert rows['held_out'].sum() >= 20 and rows['training'].sum() >= 20

    def x_y(which):
        sub = examples[rows[which]]
        return sub[props].to_numpy(dtype=float), sub.same_unit.to_numpy(dtype=float)

    def standardization(x):
        med = np.array([np.median(c[np.isfinite(c)]) if np.isfinite(c).any() else 0. for c in x.T])
        sd = np.array([np.std(c[np.isfinite(c)]) if np.isfinite(c).any() else 1. for c in x.T])
        return med, np.where(sd < 1e-9, 1., sd)

    # the standardization is measured on the training days alone
    x_train, y_train = x_y('training')
    np.testing.assert_allclose(center, standardization(x_train)[0], rtol=1e-12)
    np.testing.assert_allclose(scale, standardization(x_train)[1], rtol=1e-12)
    x_pool, y_pool = x_y('pooled')
    assert not np.allclose(standardization(x_pool)[1], scale), 'pooling does not move the scale; the oracle cannot tell'

    # the classifier weights minimize the penalized loss of the training days, not of the pooled days
    def best(design, y):
        start = np.zeros(design.shape[1] + 1)
        return minimize(_classifier_objective, start, args=(design, y, l2), method='L-BFGS-B').fun

    train_design = _design(x_train, center, scale)
    train_min = best(train_design, y_train * 2 - 1)
    assert _classifier_objective(weights, train_design, y_train * 2 - 1, l2) <= train_min + 1e-4 * abs(train_min)
    pool_center, pool_scale = standardization(x_pool)
    pool_min = best(_design(x_pool, pool_center, pool_scale), y_pool * 2 - 1)
    pooled_weights = minimize(_classifier_objective, np.zeros(len(weights)),
                              args=(_design(x_pool, pool_center, pool_scale), y_pool * 2 - 1, l2),
                              method='L-BFGS-B').x
    assert pool_min < _classifier_objective(weights, _design(x_pool, pool_center, pool_scale), y_pool * 2 - 1, l2)
    assert not np.allclose(pooled_weights, weights, atol=1e-2)

    # the calibration minimizes the calibration loss of the held-out day's scores under the saved classifier
    def calibration_on(which):
        x, y = x_y(which)
        score = _design(x, center, scale) @ weights[:-1] + weights[-1]
        opt = minimize(_calibration_objective, [1., 0.], args=(score, y, l2), method='L-BFGS-B',
                       bounds=[(0., None), (None, None)])
        return opt.x, opt.fun, score, y

    held_ab, held_min, held_score, held_y = calibration_on('held_out')
    np.testing.assert_allclose(slope_intercept, held_ab, atol=1e-3)
    assert _calibration_objective(slope_intercept, held_score, held_y, l2) <= held_min + 1e-6 * abs(held_min)
    pooled_ab = calibration_on('pooled')[0]
    assert np.max(np.abs(pooled_ab - held_ab)) > 0.05, (pooled_ab, held_ab)
    assert not np.allclose(slope_intercept, pooled_ab, atol=1e-3)
# [AGENT EDIT END]


def test_pair_logistic_scores_are_calibrated(drift_run):
    _, result = drift_run
    assert result.score.kind.value == 'calibrated_probability'
    assert result.score.kind is ScoreKind.CALIBRATED_PROBABILITY
    assert 'day1' in result.score.calibration_reference
    assert result.score.calibration_reference == (
        "synthetic: trained on ['day0', 'day2'], calibrated on held-out ['day1']")


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: numerical oracle of the calibrated probability, independent of CalibratedPairModel (Task 8 fix round 1) | date: 2026-09-27]
def test_scores_equal_the_calibrated_logistic_of_candidate_properties(drift_run):
    _, result = drift_run
    model = result.provenance.get('calibrated_model')['synthetic']
    slope, intercept = model['calibration']
    assert abs(slope - 1) > 1e-3 or abs(intercept) > 1e-3, 'calibration is the identity; the oracle cannot tell'
    pairs = result.pairs
    assert len(pairs) > 0
    x = pairs[model['properties']].to_numpy(dtype=float)
    center, scale, weights = (np.asarray(model[k], dtype=float) for k in ('center', 'scale', 'weights'))
    missing = ~np.isfinite(x)
    design = np.column_stack(((np.where(missing, center, x) - center) / scale, missing.astype(float)))
    z = design @ weights[:-1] + weights[-1]
    expected = 1.0 / (1.0 + np.exp(-(slope * z + intercept)))
    expected[missing.all(axis=1)] = np.nan
    np.testing.assert_allclose(pairs[SCORE_COLUMN].to_numpy(dtype=float), expected, rtol=1e-10, atol=1e-12)
    uncalibrated = 1.0 / (1.0 + np.exp(-z))
    assert not np.allclose(pairs[SCORE_COLUMN].to_numpy(dtype=float), uncalibrated, equal_nan=True)
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a held-out day too thin to calibrate on is refused with the group, days, counts and fix (final review fix round 1) | date: 2026-09-28]
def test_thin_held_out_day_is_refused_with_counts():
    days = make_drift_days(seed=0)
    kept_day1 = days.units[days.units.session_id == 'day1'].cellid.iloc[0]
    units = days.units[(days.units.session_id != 'day1') | (days.units.cellid == kept_day1)].reset_index(drop=True)
    with pytest.raises(ValueError) as error:
        run_pair_logistic(units, days.features, days.geometry, days.timing)
    message = str(error.value)
    assert message.startswith("synthetic: the calibration part (days ['day1']) holds 20 same-unit and "
                              "0 different-unit examples"), message
    assert 'at least 20' in message
    assert 'Widen radius_um, add days, or use functional_likelihood.' in message
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic scores with the peak-threshold neighbourhood waveform properties by default; the retired unitmatch_d10 value is refused (threshold neighbourhood) | date: 2026-10-04]
_FUNCTIONAL_PROPERTIES = ('fingerprint_r', 'd_log_spont', 'd_log_rate_session', 'd_log_rate_shared', 'isi_cosine',
                          'acg_cosine', 'd_spike_width_ms')


def test_model_properties_follow_waveform_channels():
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the subsets name the peak-to-peak correlations footprint (event-optional Task 1) | date: 2026-10-09]
    assert pair_logistic.model_properties('peak_threshold') == (
        *_FUNCTIONAL_PROPERTIES, 'waveform_r_local', 'footprint_r_local')
    assert pair_logistic.model_properties('full_array') == (*_FUNCTIONAL_PROPERTIES, 'waveform_r', 'footprint_r')
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match="waveform_channels is 'unitmatch_d10'"):
        pair_logistic.model_properties('unitmatch_d10')


def test_spec_defaults_to_peak_threshold():
    option = next(o for o in pair_logistic.SPEC.options if o.name == 'waveform_channels')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: version 4: timing is optional (event-optional Task 10) | date: 2026-10-10]
    assert (pair_logistic.SPEC.version, option.default) == ('4', 'peak_threshold')
    # [AGENT EDIT END]


def test_run_records_waveform_channels(drift_run):
    _, result = drift_run
    assert result.provenance.get('waveform_channels') == 'peak_threshold'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the saved model properties name the peak-channel footprint correlation footprint_r_local (event-optional Task 1) | date: 2026-10-09]
    assert result.provenance.get('calibrated_model')['synthetic']['properties'] == [
        'fingerprint_r', 'd_log_spont', 'd_log_rate_session', 'd_log_rate_shared', 'isi_cosine', 'acg_cosine',
        'd_spike_width_ms', 'waveform_r_local', 'footprint_r_local']
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic runs without events and declares timing optional (event-optional Task 10) | date: 2026-10-10]
def _anchored_errors(offsets: dict, truth: dict, anchor: str = 'day0') -> dict:
    """Each day's offset error once both the run's offsets and the true ones put ``anchor`` at 0."""
    return {s: abs((offsets[s] - offsets[anchor]) - (truth[s] - truth[anchor])) for s in truth}


def test_pl_no_event_drift_recovered():
    days = make_drift_days(n_neurons=20, seed=1, events=False)
    assert days.timing == {}
    result = _run(days)
    errors = _anchored_errors(result.provenance.get('registration')['offsets_um'], days.truth_offsets)
    assert max(errors.values()) <= 1.0, errors
    scores = evaluate_result(result, ground_truth(days), expected_partition=None)
    assert scores['pair_recall'] >= 0.85, scores
    assert scores['pair_precision'] >= 0.95, scores
    # the run saves what it measured: no events, chunks, no timing read, no event or session-rate property
    provenance = result.provenance
    assert provenance.get('run_evidence')['synthetic']['events'] is False
    assert provenance.get('calibrated_model')['synthetic']['properties'] == [
        'd_log_spont', 'isi_cosine', 'acg_cosine', 'd_spike_width_ms', 'waveform_r_local', 'footprint_r_local']
    assert provenance.get('chunk_seconds') == 5.0
    assert (provenance.get('used_optional_inputs'), provenance.get('timing_provenance')) == ([], {})


def test_pl_spec_timing_optional():
    spec = pair_logistic.SPEC
    assert spec.version == '4'
    assert Input.TIMING in spec.optional
    assert Input.TIMING not in spec.requires
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a no-event group skipped before observation builds no chunks, so the run saves no chunk length (event-optional Task 10 fix round 1) | date: 2026-10-10]
def test_pl_chunk_seconds_only_when_a_group_observed_chunks():
    days = make_drift_days(seed=0)
    lone = make_drift_days(n_neurons=6, seed=3, events=False)
    lone_units = lone.units[lone.units.session_id == 'day0']
    rename = {c: f'lone-{c}' for c in lone_units.cellid}
    lone_units = lone_units.assign(cellid=lone_units.cellid.map(rename), session_id='lone0', recording_group='lone')
    units = pd.concat([days.units, lone_units], ignore_index=True)
    features = {**days.features, **{rename[c]: lone.features[c] for c in rename}}
    geometry = {**days.geometry, 'lone0': lone.geometry['day0']}
    provenance = run_pair_logistic(units, features, geometry, days.timing).provenance.to_json()
    evidence = provenance['run_evidence']
    assert (evidence['synthetic']['events'], evidence['synthetic']['spont']) == (True, 'baseline')
    assert evidence['lone']['events'] is False
    assert dict(group='lone', reason='1 session with single units; matching needs at least 2') in \
        provenance['skipped_groups']
    assert 'chunk_seconds' not in provenance
# [AGENT EDIT END]
