# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: failing tests for MatchResult contract (Task 3) | date: 2026-09-22]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: base result from result_from_pairs (Task 3 Do 10) | date: 2026-09-27]
import dataclasses

import numpy as np
import pandas as pd
import pytest

from multiphy.backends import result_from_pairs
from multiphy.contract.results import (Decision, ScoreBlock, ScoreKind, TrackKind,
                                                   single_units)
# [AGENT EDIT END]


def units(*ids, su='SU'):
    return pd.DataFrame({'cellid': ids, 'session_id': [i.split('-')[0] for i in ids], 'su_type': [su] * len(ids)})


def pairs(*rows):
    df = pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'match_score', 'selected'])
    df['decision'] = np.where(df.selected, Decision.SELECTED.value, Decision.BELOW_THRESHOLD.value)
    return df


def result(**over):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the valid unitmatch base is built with result_from_pairs (Task 3 Do 10; review r3); each test's overrides inject its defect | date: 2026-09-27]
    scores = pd.DataFrame({'cellid_a': ['A-1', 'A-1'], 'cellid_b': ['B-1', 'C-1'], 'match_score': [.95, .1]})
    base = result_from_pairs('unitmatch', '1', units('A-1', 'B-1', 'C-1'), scores, run_id='r1')
    assert base.tracks == (('A-1', 'B-1'),) and base.unmatched == ('C-1',)
    return dataclasses.replace(base, **over)
    # [AGENT EDIT END]


def test_valid_result_passes_and_digest_is_stable():
    r = result().validate()
    assert r.digest() == result().digest()


def test_single_units_excludes_unknown_quality():
    u = units('A-1', 'B-1', 'C-1'); u.loc[2, 'su_type'] = None
    eligible, excluded = single_units(u)
    assert list(eligible.cellid) == ['A-1', 'B-1'] and excluded == ('C-1',)


# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: all-NaN scores with a score column are valid (every pair missing_evidence); Task 7 ruling | date: 2026-09-22]
def test_none_score_kind_requires_all_nan_scores():
    with pytest.raises(ValueError, match='match_score'):
        result(score=ScoreBlock(ScoreKind.NONE, None)).validate()
    p = pairs(('A-1', 'B-1', np.nan, True), ('A-1', 'C-1', np.nan, False))
    result(pairs=p).validate()
# [AGENT EDIT END]
    result(pairs=p, score=ScoreBlock(ScoreKind.NONE, None)).validate()


def test_calibrated_requires_reference():
    with pytest.raises(ValueError, match='calibration_reference'):
        result(score=ScoreBlock(ScoreKind.CALIBRATED_PROBABILITY, 'match_score')).validate()
    result(score=ScoreBlock(ScoreKind.CALIBRATED_PROBABILITY, 'match_score', calibration_reference='ref')).validate()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: connected components gone: a same-session track is refused for every track kind (Task 1) | date: 2026-09-27]
def test_track_kinds():
    assert {k.value for k in TrackKind} == {'native', 'common'}


def test_same_session_track_refused_for_every_kind():
    u = units('A-1', 'A-2', 'B-1')
    p = pairs(('A-1', 'A-2', .95, True), ('A-1', 'B-1', .1, False))
    with pytest.raises(ValueError, match='same-session'):
        result(units=u, pairs=p, tracks=(('A-1', 'A-2'),), unmatched=('B-1',)).validate()
    with pytest.raises(ValueError, match='same-session'):
        result(units=u, pairs=p, tracks=(('A-1', 'A-2'),), unmatched=('B-1',),
               track_kind=TrackKind.NATIVE, tracking_rule='native_intermediate').validate()
# [AGENT EDIT END]


def test_partition_rules_and_unknown_decision():
    with pytest.raises(ValueError, match='partition'):
        result(unmatched=()).validate()
    with pytest.raises(ValueError, match='Singleton'):
        result(tracks=(('A-1', 'B-1'), ('C-1',)), unmatched=()).validate()
    p = pairs(('A-1', 'B-1', .95, True), ('A-1', 'C-1', .1, False)); p.loc[0, 'decision'] = 'made_up'
    with pytest.raises(ValueError, match='decision'):
        result(pairs=p).validate()


def test_validate_against_spec_checks_emissions():
    from multiphy.contract.backend import BackendSpec, Provides
    from multiphy.contract.inputs import Input
    spec = BackendSpec('unitmatch', '1',  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: constrained is no longer an id (Task 3) | date: 2026-09-27]
                       frozenset({Input.UNITS}), frozenset(),
                       Provides(frozenset({ScoreKind.UNCALIBRATED}), frozenset({TrackKind.COMMON})), ())
    result().validate(spec)
    with pytest.raises(ValueError, match='track_kind'):
        result(track_kind=TrackKind.NATIVE, tracking_rule='native_intermediate').validate(spec)
    with pytest.raises(ValueError, match='similarity'):
        result(score=ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score', similarity_column='sim')).validate(spec)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: proposed_link is a pair decision (native-same-day Task 1) | date: 2026-10-11]
def test_proposed_link_is_a_decision():
    from multiphy.contract.results import DECISIONS
    assert Decision('proposed_link') is Decision.PROPOSED_LINK
    assert ('proposed_link' in DECISIONS) is True
# [AGENT EDIT END]
