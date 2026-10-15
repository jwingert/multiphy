# [AGENT EDIT START | agent: codex | user: wingertj | reason: leakage-aware held-out evaluation of multiday matching backends | date: 2026-09-17]
"""Evaluate synthetic temporal-split benchmarks without giving labels to matchers.

Ground-truth rows identify the source cluster that generated each pseudo-unit.
They establish identity *within a source recording*, not between recordings.
Splitting an existing sorted cluster provides conditional reference identity;
it cannot establish that the original cluster was a perfectly isolated neuron
or reproduce all changes caused by probe reinsertion.

The matcher receives pseudo-unit IDs and features only. Keep this label table
out of its input bundle. Call ``validate_partitions`` on the full dataset before
training, tuning, or evaluating; evaluating a test-only table cannot prove that
unseen training data were independent.
"""
import itertools

import numpy as np
import pandas as pd


LABEL_COLUMNS = ('cellid', 'source_unit_id', 'source_session', 'well', 'partition')


def validate_partitions(records, *, group_columns=('well',), partition_column='partition'):
    """Reject source-recording leakage and enforce the requested holdout groups.

    The default holds out entire wells; use well identifiers namespaced by
    animal (e.g. ``Prince:3``). To allow training on other sessions of the same
    well, explicitly use ``group_columns=('well', 'source_session')``. Temporal
    splits of a source recording must never cross train/calibration/test sets.
    The returned counts describe coverage; they do not certify absent data.
    """
    required = {'source_session', 'well', partition_column, *group_columns}
    missing = required - set(records.columns)
    if missing:
        raise ValueError(f'Partition metadata missing columns: {sorted(missing)}')
    if records[list(required)].isna().any().any():
        raise ValueError('Partition grouping metadata cannot be missing')
    if any(records[c].astype(str).str.strip().eq('').any() for c in required):
        raise ValueError('Partition grouping metadata cannot be empty')
    if not group_columns:
        raise ValueError('At least one explicit holdout group is required')
    # Enforce source recordings independently of the requested coarser split.
    for cols in (('well', 'source_session'), tuple(group_columns)):
        counts = records.groupby(list(cols), dropna=False)[partition_column].nunique()
        if (counts > 1).any():
            raise ValueError(f'Partition leakage: groups {cols} cross partitions: {list(counts[counts > 1].index[:5])}')
    return {'group_columns': list(group_columns),
            'partition_counts': records[partition_column].value_counts().to_dict(),
            'n_source_recordings': len(records[['well', 'source_session']].drop_duplicates())}


def _labels(result, ground_truth, expected_partition, group_columns):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: ground truth without a partition column is accepted when no partition is expected (multiphy project folder Task 2) | date: 2026-09-25]
    partitioned = 'partition' in ground_truth.columns
    required = LABEL_COLUMNS if partitioned else tuple(c for c in LABEL_COLUMNS if c != 'partition')
    missing = set(required) - set(ground_truth.columns)
    if missing:
        raise ValueError(f'Ground truth missing columns: {sorted(missing)}')
    if not partitioned and expected_partition is not None:
        raise ValueError('Ground truth has no partition column; expected_partition must be None.')
    labels = ground_truth.copy()
    if labels[list(required)].isna().any().any():
        raise ValueError('Ground-truth identifiers cannot be missing')
    labels['cellid'] = labels.cellid.astype(str)
    if labels.cellid.duplicated().any():
        raise ValueError('Ground truth requires a unique row per pseudo-unit')
    if partitioned:
        partition_info = validate_partitions(labels, group_columns=group_columns)
    else:
        partition_info = {'group_columns': list(group_columns), 'partition_counts': {},
                          'n_source_recordings': len(labels[['well', 'source_session']].drop_duplicates())}
    # [AGENT EDIT END]
    ids = set(result.units.cellid)
    if not ids <= set(labels.cellid):
        raise ValueError('Every evaluated single unit needs source-cluster ground truth')
    labels = labels[labels.cellid.isin(ids)].copy()
    if expected_partition is not None and not labels.partition.eq(expected_partition).all():
        raise ValueError(f'Evaluation requires only the {expected_partition!r} partition')
    session_of = result.units.set_index('cellid').session_id.to_dict()
    # Known labels never extend across source recordings. Different original
    # clusters in one sort are conditional negatives, including sorter errors.
    truth = {}
    for _, group in labels.groupby(['well', 'source_session']):
        for a, z in itertools.combinations(group.itertuples(), 2):
            if session_of[a.cellid] != session_of[z.cellid]:
                truth[tuple(sorted((a.cellid, z.cellid)))] = a.source_unit_id == z.source_unit_id
    return labels, truth, partition_info


def _predicted_links(result):
    return {tuple(sorted(edge)) for track in result.tracks
            for edge in itertools.combinations(track, 2)}


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: score kind is now result-level (result.score.kind), not a per-row column (Task 7 ruling) | date: 2026-09-22]
def score_reliability(pairs, truth, *, n_bins=10, score_kind=None):
# [AGENT EDIT END]
    """Report observed identity fractions without relabeling scores as calibrated.

    Only cross-pseudo-session pairs with known source-recording identity enter
    this diagnostic. This is conditional on the method's measured candidate
    set; assignment recall separately counts known positives with no score.
    Brier score is reported only for explicitly calibrated probability rows.
    Related pairs are not independent, so no binomial confidence intervals are
    attached. An empty bin has an undefined identity fraction, not zero.
    """
    if not isinstance(n_bins, int) or n_bins < 1:
        raise ValueError('n_bins must be a positive integer')
    samples = []
    for r in pairs.itertuples():
        key = tuple(sorted((r.cellid_a, r.cellid_b)))
        if key in truth and pd.notna(r.match_score):
            # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: score kind is now result-level (result.score.kind), not a per-row column (Task 7 ruling) | date: 2026-09-22]
            samples.append((float(r.match_score), int(truth[key]), score_kind))
            # [AGENT EDIT END]
    rows = []
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        selected = [(s, y, k) for s, y, k in samples if lo <= s and (s < hi or (i == n_bins - 1 and s <= hi))]
        rows.append({'lower': lo, 'upper': hi, 'n_pairs': len(selected),
                     'mean_score': float(np.mean([s for s, _, _ in selected])) if selected else None,
                     'observed_same_fraction': float(np.mean([y for _, y, _ in selected])) if selected else None})
    probabilities = [(s, y) for s, y, kind in samples if kind == 'calibrated_probability']
    return {'n_known_scored_pairs': len(samples), 'bins': rows,
            'n_calibrated_probability_pairs': len(probabilities),
            'brier_score': float(np.mean([(s - y) ** 2 for s, y in probabilities])) if probabilities else None,
            'scope': 'Known source-recording pairs within the measured candidate set; pairs are dependent'}


def evaluate_result(result, ground_truth, *, expected_partition='test',
                    group_columns=('well',), n_bins=10):
    """Evaluate final track identity, explicit unmatched units, and score behavior.

    Every implied within-track pair counts as a prediction, including a chain's
    missing direct links. Positives absent from the candidate table count as
    missed matches. Comparisons across independent original recordings remain
    unknown and are counted separately, never treated as confirmed negatives.
    Candidate coverage includes finite native DANT similarity even when there
    is no bounded match score; probability diagnostics still use match_score only.
    """
    result.validate()
    labels, truth, partitions = _labels(result, ground_truth, expected_partition, group_columns)
    predicted = _predicted_links(result)
    known = set(truth)
    positive = {key for key, same in truth.items() if same}
    true_positive = len(predicted & positive)
    false_positive = len((predicted & known) - positive)
    false_negative = len(positive - predicted)
    true_negative = len(known - positive - predicted)
    has_counterpart = {u for edge in positive for u in edge}
    # Absence of a known positive is not evidence of absence when another
    # recording's identity is unknown. Score unmatched decisions only when a
    # positive counterpart is known, or all possible comparisons are labeled.
    known_degree = {u: 0 for u in result.units.cellid}
    for a, z in known:
        known_degree[a] += 1
        known_degree[z] += 1
    session_sizes = result.units.session_id.value_counts().to_dict()
    fully_observed = {r.cellid for r in result.units.itertuples()
                      if known_degree[r.cellid] == len(result.units) - session_sizes[r.session_id]}
    true_unmatched = fully_observed - has_counterpart
    determinate_unmatched = true_unmatched | has_counterpart
    unknown_unmatched = set(result.units.cellid) - determinate_unmatched
    reported_unmatched = set(result.unmatched)
    true_unmatched_reported = len(true_unmatched & reported_unmatched)
    assignment = {u: i for i, track in enumerate(result.tracks) for u in track}
    assignment.update({u: len(result.tracks) + i for i, u in enumerate(result.unmatched)})
    split_identities = 0
    evaluable_identities = 0
    for _, group in labels.groupby(['well', 'source_session', 'source_unit_id']):
        if any(tuple(sorted(edge)) in positive for edge in itertools.combinations(group.cellid, 2)):
            evaluable_identities += 1
            split_identities += int(len({assignment[u] for u in group.cellid}) > 1)
    merged_tracks = sum(any(tuple(sorted(edge)) in truth and not truth[tuple(sorted(edge))]
                            for edge in itertools.combinations(track, 2)) for track in result.tracks)
    evidence_columns = ['match_score']
    measured_mask = np.isfinite(pd.to_numeric(result.pairs.match_score, errors='coerce'))
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: a declared similarity column (DANT's native similarity) counts as measured evidence; no backend-id check (Task 12) | date: 2026-09-22]
    similarity = result.score.similarity_column
    if similarity and similarity in result.pairs:
        evidence_columns.append(similarity)
        measured_mask |= np.isfinite(pd.to_numeric(result.pairs[similarity], errors='coerce'))
    # [AGENT EDIT END]
    measured = {tuple(sorted((r.cellid_a, r.cellid_b)))
                for r in result.pairs.loc[measured_mask].itertuples()}
    return {'backend_id': result.backend_id, 'run_id': result.run_id,
            'n_units': len(result.units), 'n_tracks': len(result.tracks),
            'n_unmatched': len(result.unmatched), 'n_known_pairs': len(known),
            'n_known_positive_pairs': len(positive), 'n_unknown_predicted_pairs': len(predicted - known),
            'true_positive_pairs': true_positive, 'false_positive_pairs': false_positive,
            'false_negative_pairs': false_negative, 'true_negative_pairs': true_negative,
            'pair_precision': _ratio(true_positive, true_positive + false_positive),
            'pair_recall': _ratio(true_positive, len(positive)),
            'candidate_positive_coverage': _ratio(len(positive & measured), len(positive)),
            'candidate_evidence_columns': evidence_columns,
            'true_unmatched_units': len(true_unmatched),
            'unknown_unmatched_status_units': len(unknown_unmatched),
            'reported_unmatched_with_unknown_status': len(reported_unmatched & unknown_unmatched),
            'evaluable_reported_unmatched_units': len(reported_unmatched & determinate_unmatched),
            'correctly_unmatched_units': true_unmatched_reported,
            'unmatched_precision': _ratio(true_unmatched_reported, len(reported_unmatched & determinate_unmatched)),
            'unmatched_recall': _ratio(true_unmatched_reported, len(true_unmatched)),
            'merged_tracks': int(merged_tracks), 'split_source_identities': split_identities,
            'evaluable_source_identities': evaluable_identities,
            # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: score kind is now result-level (result.score.kind), not a per-row column (Task 7 ruling) | date: 2026-09-22]
            'score_reliability': score_reliability(result.pairs, truth, n_bins=n_bins,
                                                   score_kind=result.score.kind.value),
            # [AGENT EDIT END]
            'partitions': partitions,
            'ground_truth_scope': 'Identity conditional on original sorted source clusters; cross-recording identities unknown'}


def compare_results(results, ground_truth, **kwargs):
    """Return evaluation rows and pairwise assignment agreement for identical units."""
    results = list(results)
    if not results:
        return pd.DataFrame(), pd.DataFrame()
    eligible = set(results[0].units.cellid)
    if any(set(result.units.cellid) != eligible for result in results):
        raise ValueError('Backend comparisons require identical eligible single units')
    sessions = results[0].units.set_index('cellid').session_id.to_dict()
    if any(result.units.set_index('cellid').session_id.to_dict() != sessions for result in results):
        raise ValueError('Backend comparisons require identical session identities')
    rows = [evaluate_result(result, ground_truth, **kwargs) for result in results]
    agreement = []
    for a, z in itertools.combinations(results, 2):
        left, right = _predicted_links(a), _predicted_links(z)
        agreement.append({'run_id_a': a.run_id, 'run_id_b': z.run_id,
                          'backend_a': a.backend_id, 'backend_b': z.backend_id,
                          'shared_predicted_pairs': len(left & right),
                          'only_a_pairs': len(left - right), 'only_b_pairs': len(right - left),
                          'assignment_jaccard': _ratio(len(left & right), len(left | right))})
    return pd.DataFrame(rows), pd.DataFrame(agreement)
# [AGENT EDIT END]
