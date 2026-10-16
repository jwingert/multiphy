# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify held-out source-cluster evaluation and leakage guards | date: 2026-09-17]
import numpy as np
import pandas as pd
import pytest

# [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: results come from the track helper below (Task 3) | date: 2026-09-27]
from multiphy.evaluation import compare_results, evaluate_result, validate_partitions


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no registered backend tracks cached pair scores; fabricate results with the common tracker under a real id (Task 3) | date: 2026-09-27]
def track(units, pairs, **kwargs):
    from multiphy.backends import result_from_pairs
    return result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', **kwargs)
# [AGENT EDIT END]


def fixture_data():
    units = pd.DataFrame({'cellid': ['early-1', 'late-1', 'early-2', 'late-2', 'early-3'],
                          'su_type': ['SU'] * 5})
    labels = units[['cellid']].copy()
    labels['source_unit_id'] = [1, 1, 2, 2, 3]
    labels['source_session'] = 'original_recording'
    labels['well'] = 'Prince:3'
    labels['partition'] = 'test'
    return units, labels


def make_pairs(*rows):
    return pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'p_global'])


def test_correct_tracks_and_intentionally_unmatched_unit():
    units, labels = fixture_data()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units,
    # [AGENT EDIT END]
                         make_pairs(('early-1', 'late-1', .99), ('early-2', 'late-2', .98)))
    metrics = evaluate_result(result, labels)
    assert metrics['pair_precision'] == metrics['pair_recall'] == 1
    assert metrics['unmatched_precision'] == metrics['unmatched_recall'] == 1
    assert metrics['true_positive_pairs'] == 2
    assert metrics['n_known_pairs'] == 6
    assert metrics['score_reliability']['brier_score'] is None
    assert metrics['split_source_identities'] == metrics['merged_tracks'] == 0


def test_missing_candidates_count_as_false_negatives():
    units, labels = fixture_data()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs(('early-1', 'late-1', .99)))
    # [AGENT EDIT END]
    metrics = evaluate_result(result, labels)
    assert metrics['pair_recall'] == .5
    assert metrics['candidate_positive_coverage'] == .5
    assert metrics['false_negative_pairs'] == 1
    assert metrics['split_source_identities'] == 1
    assert metrics['unmatched_precision'] == pytest.approx(1 / 3)


def test_wrong_pair_reports_merge_and_source_identity_splits():
    units, labels = fixture_data()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs(('early-1', 'late-2', .99)))
    # [AGENT EDIT END]
    metrics = evaluate_result(result, labels)
    assert metrics['false_positive_pairs'] == 1
    assert metrics['merged_tracks'] == 1
    assert metrics['split_source_identities'] == 2
    assert metrics['pair_precision'] == 0


def test_cross_source_recording_identity_is_unknown_not_negative():
    units, labels = fixture_data()
    labels.loc[labels.cellid == 'late-2', 'source_session'] = 'another_recording'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs(('early-1', 'late-2', .99)))
    # [AGENT EDIT END]
    metrics = evaluate_result(result, labels)
    assert metrics['n_unknown_predicted_pairs'] == 1
    assert metrics['false_positive_pairs'] == 0
    assert metrics['pair_precision'] is None


def test_unknown_cross_recording_units_do_not_become_true_unmatched():
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'su_type': ['SU', 'SU']})
    labels = pd.DataFrame({'cellid': units.cellid, 'source_unit_id': [1, 1],
                           'source_session': ['original_A', 'original_B'],
                           'well': ['Prince:3', 'Prince:3'], 'partition': ['test', 'test']})
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs())
    # [AGENT EDIT END]
    metrics = evaluate_result(result, labels)
    assert metrics['true_unmatched_units'] == 0
    assert metrics['correctly_unmatched_units'] == 0
    assert metrics['unknown_unmatched_status_units'] == 2
    assert metrics['reported_unmatched_with_unknown_status'] == 2
    assert metrics['unmatched_precision'] is None
    assert metrics['unmatched_recall'] is None


def test_known_counterparts_still_make_unmatched_predictions_wrong_with_unknowns():
    units, labels = fixture_data()
    labels.loc[labels.cellid == 'early-3', 'source_session'] = 'another_recording'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs())
    # [AGENT EDIT END]
    metrics = evaluate_result(result, labels)
    assert metrics['unknown_unmatched_status_units'] == 1
    assert metrics['evaluable_reported_unmatched_units'] == 4
    assert metrics['unmatched_precision'] == 0
    assert metrics['unmatched_recall'] is None


def test_temporal_halves_cannot_cross_partitions():
    _, labels = fixture_data()
    labels.loc[0, 'partition'] = 'train'
    with pytest.raises(ValueError, match='Partition leakage'):
        validate_partitions(labels)
    with pytest.raises(ValueError, match='Partition leakage'):
        validate_partitions(labels, group_columns=('source_unit_id',))


def test_default_well_holdout_and_explicit_source_session_holdout():
    _, labels = fixture_data()
    labels.loc[0, ['source_session', 'partition']] = ['another_recording', 'train']
    with pytest.raises(ValueError, match='Partition leakage'):
        validate_partitions(labels)
    report = validate_partitions(labels, group_columns=('well', 'source_session'))
    assert report['n_source_recordings'] == 2


def test_evaluation_does_not_accept_training_partition():
    units, labels = fixture_data()
    labels['partition'] = 'train'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs())
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='only the'):
        evaluate_result(result, labels)


def test_every_unit_needs_unique_ground_truth():
    units, labels = fixture_data()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units, make_pairs())
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='Every evaluated'):
        evaluate_result(result, labels.iloc[:-1])
    with pytest.raises(ValueError, match='unique row'):
        evaluate_result(result, pd.concat([labels, labels.iloc[:1]]))


def test_compare_identical_and_disagreeing_assignments():
    units, labels = fixture_data()
    p = make_pairs(('early-1', 'late-1', .99), ('early-2', 'late-2', .98))
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    a = track(units, p)
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the legacy backend is gone; a second run at another threshold (Task 1) | date: 2026-09-27]
    z = track(units, p, threshold=.95, run_id='second')
    # [AGENT EDIT END]
    table, agreement = compare_results([a, z], labels)
    assert len(table) == 2
    assert agreement.iloc[0].assignment_jaccard == 1
    z.unmatched = tuple(units.cellid)
    z.tracks = ()
    table, agreement = compare_results([a, z], labels)
    assert agreement.iloc[0].assignment_jaccard == 0


def test_probability_reliability_uses_all_known_scored_candidates():
    units, labels = fixture_data()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units,
    # [AGENT EDIT END]
                         make_pairs(('early-1', 'late-1', .99), ('early-1', 'late-2', .01)))
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: the score kind and calibration reference live on result.score, not a score_kind column or provenance (Task 12) | date: 2026-09-22]
    from multiphy.contract.results import ScoreBlock, ScoreKind
    result.score = ScoreBlock(ScoreKind.CALIBRATED_PROBABILITY, 'match_score',
                              calibration_reference='separate calibration dataset fixture')
    # [AGENT EDIT END]
    reliability = evaluate_result(result, labels)['score_reliability']
    assert reliability['n_calibrated_probability_pairs'] == 2
    assert reliability['brier_score'] == pytest.approx(.0001)
    assert reliability['bins'][0]['observed_same_fraction'] == 0
    assert reliability['bins'][-1]['observed_same_fraction'] == 1


@pytest.mark.parametrize('unmeasured', [np.nan, np.inf, -np.inf])
def test_native_dant_similarity_counts_for_coverage_without_becoming_probability(unmeasured):
    units, labels = fixture_data()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered id; the current_local alias is gone (Task 1); fabricated under unitmatch via result_from_pairs (Task 3) | date: 2026-09-27]
    result = track(units,
    # [AGENT EDIT END]
        make_pairs(('early-1', 'late-1', .99), ('early-2', 'late-2', .98)))
    result.backend_id = 'dant'
    result.pairs['match_score'] = np.nan
    result.pairs['dant_similarity'] = [-8.5, unmeasured]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: DANT's native similarity is now declared on the score block, not inferred from backend_id (Task 12) | date: 2026-09-22]
    from multiphy.contract.results import ScoreBlock, ScoreKind
    result.score = ScoreBlock(ScoreKind.NONE, None, 'dant_similarity')
    # [AGENT EDIT END]
    metrics = evaluate_result(result, labels)
    assert metrics['candidate_positive_coverage'] == .5
    assert metrics['candidate_evidence_columns'] == ['match_score', 'dant_similarity']
    assert result.pairs.match_score.isna().all()
    assert metrics['score_reliability']['n_known_scored_pairs'] == 0
    assert metrics['score_reliability']['n_calibrated_probability_pairs'] == 0
    assert metrics['score_reliability']['brier_score'] is None
    assert metrics['pair_recall'] == 1
# [AGENT EDIT END]
