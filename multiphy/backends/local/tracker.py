# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: shared tracker moved out of backends.py behind the result contract | date: 2026-09-22]
"""Track formation shared by every backend that emits pair scores.

``assign`` is the deterministic best-first tracker with mutual candidate margins, one
unit per session and rejection of measured contradictions. ``result_from_pairs`` wraps
any [0, 1] pair scores into a MatchResult through it.
"""
from __future__ import annotations
import hashlib
import itertools
import json
import uuid

import numpy as np
import pandas as pd

from ...contract.results import Decision, MatchResult, ScoreBlock, ScoreKind, TrackKind, single_units
from ...contract.provenance import Provenance, json_safe, payload_digest


def score_column_for(pairs, feature_stream):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the waveform stream's posterior is a feature_stream choice (event-optional Task 9) | date: 2026-10-10]
    choices = {'combined': ('p_global', 'posterior'), 'stimulus': ('p_stim',),
               'physiology': ('p_phys',), 'waveform': ('p_wave',)}
    # [AGENT EDIT END]
    if feature_stream not in choices:
        raise ValueError(f'feature_stream must be one of {tuple(choices)}')
    for col in choices[feature_stream]:
        if col in pairs and pd.to_numeric(pairs[col], errors='coerce').notna().any():
            return col
    for col in choices[feature_stream]:
        if col in pairs:
            return col
    if pairs.empty:
        return choices[feature_stream][0]
    raise ValueError(f'No measured score for {feature_stream!r}; no silent stream substitution')


def prepare_pairs(units, pairs, score_column):
    pairs = pairs.copy()
    if not {'cellid_a', 'cellid_b'} <= set(pairs.columns):
        if not pairs.empty:
            raise ValueError('pairs requires cellid_a and cellid_b')
        pairs = pd.DataFrame(columns=['cellid_a', 'cellid_b'])
    if pairs[['cellid_a', 'cellid_b']].isna().any().any():
        raise ValueError('Pair endpoints cannot be missing')
    for col in ('cellid_a', 'cellid_b'):
        pairs[col] = pairs[col].astype(str)
    ids = set(units.cellid)
    pairs = pairs[pairs.cellid_a.isin(ids) & pairs.cellid_b.isin(ids)].copy().reset_index(drop=True)
    keys = [tuple(sorted((a, b))) for a, b in zip(pairs.cellid_a, pairs.cellid_b)]
    if len(keys) != len(set(keys)):
        raise ValueError('Repeated pair evidence; select one recording block before matching')
    pairs['match_score'] = pd.to_numeric(pairs.get(score_column, pd.Series(np.nan, index=pairs.index)), errors='coerce')
    pairs['selected'] = False
    pairs['decision'] = np.where(pairs.match_score.notna(), Decision.BELOW_THRESHOLD.value,
                                 Decision.MISSING_EVIDENCE.value)
    return pairs


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: connected components dropped: assign always enforces margins, session uniqueness and contradictions (Task 1) | date: 2026-09-27]
def assign(units, pairs, threshold, contradiction_threshold, ambiguity_margin) -> tuple:
# [AGENT EDIT END]
    sessions = units.set_index('cellid').session_id.to_dict()
    groups = {u: {u} for u in units.cellid}
    owner = {u: u for u in units.cellid}
    measured = {frozenset((r.cellid_a, r.cellid_b)): r.match_score
                for r in pairs.itertuples() if pd.notna(r.match_score)}
    candidates = []
    competitors = {}
    for i, r in pairs.iterrows():
        a, b = r.cellid_a, r.cellid_b
        if sessions[a] == sessions[b]:
            pairs.at[i, 'decision'] = Decision.SAME_SESSION.value
            continue
        if pd.isna(r.match_score) or r.match_score < threshold:
            continue
        candidates.append((i, a, b, float(r.match_score)))
    # Compare against all measured alternatives, even just below the threshold:
    # 0.901 versus 0.899 is ambiguous at a 0.90 operating threshold.
    for r in pairs.itertuples():
        a, b = r.cellid_a, r.cellid_b
        if pd.notna(r.match_score) and sessions[a] != sessions[b]:
            competitors.setdefault((a, sessions[b]), []).append((b, r.match_score))
            competitors.setdefault((b, sessions[a]), []).append((a, r.match_score))
    for i, a, b, score in sorted(candidates, key=lambda x: (-x[3], min(x[1:3]), max(x[1:3]))):
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: connected components dropped; competition always applies (Task 1) | date: 2026-09-27]
        if any(
            score - other_score < ambiguity_margin or other_score > score
            for u, v in ((a, b), (b, a))
            for other, other_score in competitors[(u, sessions[v])] if other != v
        ):
        # [AGENT EDIT END]
            pairs.at[i, 'decision'] = Decision.AMBIGUOUS_COMPETITION.value
            continue
        ga, gb = groups[owner[a]], groups[owner[b]]
        if ga is gb:
            pairs.at[i, 'decision'] = Decision.SUPPORTING_TRACK.value
            pairs.at[i, 'selected'] = True
            continue
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: connected components dropped; session uniqueness and contradictions always apply (Task 1) | date: 2026-09-27]
        if {sessions[u] for u in ga} & {sessions[u] for u in gb}:
            pairs.at[i, 'decision'] = Decision.SESSION_COLLISION.value
            continue
        if any(measured.get(frozenset((u, v)), np.inf) < contradiction_threshold
               for u in ga for v in gb):
            pairs.at[i, 'decision'] = Decision.MEASURED_CONTRADICTION.value
            continue
        # [AGENT EDIT END]
        keep, remove = owner[a], owner[b]
        ga.update(gb)
        for u in gb:
            owner[u] = keep
        del groups[remove]
        pairs.at[i, 'decision'] = Decision.SELECTED.value
        pairs.at[i, 'selected'] = True
    tracks = tuple(sorted(tuple(sorted(g)) for g in groups.values() if len(g) > 1))
    unmatched = tuple(sorted(next(iter(g)) for g in groups.values() if len(g) == 1))
    coverage = []
    for track in tracks:
        links = [measured.get(frozenset(edge)) for edge in itertools.combinations(track, 2)]
        coverage.append({'members': list(track), 'n_possible': len(links),
                         'n_measured': sum(x is not None for x in links),
                         'n_missing': sum(x is None for x in links)})
    return tracks, unmatched, coverage


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: connected-components keyword dropped: every result uses the common tracker (Task 1) | date: 2026-09-27]
def result_from_pairs(backend_id, version, units, pairs, *, score_column='match_score',
                      score_kind=ScoreKind.UNCALIBRATED, threshold=.90,
                      contradiction_threshold=.10, ambiguity_margin=.02,
                      provenance=None, run_id=None, calibration_reference=None) -> MatchResult:
# [AGENT EDIT END]
    """Wrap genuine pair outputs with a common, conservative track assignment.

    Values must be on a [0, 1] scale, which alone does not make them probabilities.
    A calibrated probability requires ``calibration_reference``.
    Track formation is deterministic best-first grouping with mutual candidate
    competition, not a globally optimal assignment or published-method tracker.
    """
    score_kind = ScoreKind(score_kind)
    if not 0 <= contradiction_threshold <= threshold <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError('Require 0 <= contradiction_threshold <= threshold <= 1 and margin in [0, 1]')
    eligible, excluded = single_units(units)
    if {'cellid_a', 'cellid_b'} <= set(pairs.columns):
        endpoints = set(pairs.cellid_a.dropna().astype(str)) | set(pairs.cellid_b.dropna().astype(str))
        if not endpoints <= set(units.cellid.astype(str)):
            raise ValueError('Pair evidence references unknown unit IDs')
    out = prepare_pairs(eligible, pairs, score_column)
    values = out.match_score.dropna().to_numpy()
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError('Scores must be finite in [0, 1] or missing')
    if (out.cellid_a == out.cellid_b).any():
        raise ValueError('A cross-session pair cannot compare a unit to itself')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: always the common tracker (Task 1) | date: 2026-09-27]
    tracks, unmatched, coverage = assign(eligible, out, threshold, contradiction_threshold,
                                         ambiguity_margin)
    metadata = dict(provenance or {})
    metadata.update(score_column=score_column, score_kind=score_kind.value, threshold=threshold,
                    contradiction_threshold=contradiction_threshold,
                    ambiguity_margin=ambiguity_margin, enforce_session_unique=True,
                    assignment='mutual_margin_best_first',
    # [AGENT EDIT END]
                    internal_evidence_coverage=coverage, n_excluded_non_su=len(excluded),
                    quality_policy='Explicit su_type=SU only; cache labels are not independently verified')
    if run_id is None:
        run_id = f'{backend_id}_{uuid.uuid4().hex}'
    metadata['input_pair_digest'] = payload_digest(json_safe(
        out[['cellid_a', 'cellid_b', 'match_score']].to_dict(orient='records')))
    metadata['run_config_digest'] = hashlib.sha256(json.dumps(
        {'backend_id': backend_id, 'version': version, 'threshold': threshold,
         'contradiction_threshold': contradiction_threshold, 'ambiguity_margin': ambiguity_margin,
         'score_column': score_column, 'score_kind': score_kind.value,
         'calibrated_model': metadata.get('calibrated_model'),
         'input_pair_digest': metadata['input_pair_digest']}, sort_keys=True).encode()).hexdigest()
    return MatchResult(backend_id, version, run_id, eligible, out, tracks, unmatched, excluded,
                       ScoreBlock(score_kind, score_column, calibration_reference=calibration_reference),
                       # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: always the common tracker (Task 1) | date: 2026-09-27]
                       TrackKind.COMMON, 'mutual_margin_best_first',
                       # [AGENT EDIT END]
                       Provenance.from_json(metadata)).validate()
# [AGENT EDIT END]
