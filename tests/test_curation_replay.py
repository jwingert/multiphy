"""Curation state is the run's result folded with the curator's history rows; undo and redo replay."""
from __future__ import annotations

import pytest

from multiphy import curation
from multiphy.clusters import cluster_id

TRACKS = [['D1:0', 'D2:0', 'D3:0']]
UNMATCHED = ['D1:1', 'D2:1']
T = cluster_id(['D1:0', 'D2:0', 'D3:0'])


def _replay(*rows):
    return curation.replay(TRACKS, UNMATCHED, list(rows))


def _held(state):
    return {k: v['status'] for k, v in state.tracks.items()}


def _undo(row):
    return curation.make_row('undo', target_op=row['op_id'])


def test_initial_state_from_result():
    s = _replay()
    assert _held(s) == {T: 'proposed', cluster_id(['D1:1']): 'unmatched', cluster_id(['D2:1']): 'unmatched'}
    assert s.rejected == {}


def test_reject_releases_singletons():
    s = _replay(curation.make_row('reject', cluster_id=T))
    assert len(s.tracks) == 5
    assert all(v['status'] == 'unmatched' and len(v['members']) == 1 for v in s.tracks.values())
    assert s.rejected == {T: ['D1:0', 'D2:0', 'D3:0']}


def test_undo_reject_restores_track():
    base = _replay()
    rej = curation.make_row('reject', cluster_id=T)
    s = _replay(rej, _undo(rej))
    assert s.tracks == base.tracks
    assert s.rejected == base.rejected
    assert s.verdicts == base.verdicts
    assert s.accepted_by == base.accepted_by
    assert s.undo_target is None
    assert s.redo_target == rej['op_id']


def test_new_action_clears_redo():
    rej = curation.make_row('reject', cluster_id=T)
    s = _replay(rej, _undo(rej), curation.make_row('accept', cluster_id=T))
    assert s.redo_target is None
    assert s.tracks[T]['status'] == 'accepted'


def test_redo_reapplies():
    rej = curation.make_row('reject', cluster_id=T)
    alone = _replay(rej)
    s = _replay(rej, _undo(rej), curation.make_row('redo', target_op=rej['op_id']))
    assert s.tracks == alone.tracks
    assert s.rejected == alone.rejected


def test_split_then_merge_holds_each_unit_once():
    split = curation.make_row('split', cluster_id=T, members=['D1:0'])
    merge = curation.make_row('merge', parent_ids=[cluster_id(['D1:0']), cluster_id(['D1:1'])])
    s = _replay(split, merge)
    assert _held(s) == {cluster_id(['D2:0', 'D3:0']): 'proposed',
                        cluster_id(['D1:0', 'D1:1']): 'proposed',
                        cluster_id(['D2:1']): 'unmatched'}
    units = [m for v in s.tracks.values() for m in v['members']]
    assert sorted(units) == ['D1:0', 'D1:1', 'D2:0', 'D2:1', 'D3:0']


def test_reject_singleton_refused():
    with pytest.raises(ValueError, match='a single unit cannot be rejected'):
        curation.apply(_replay(), curation.make_row('reject', cluster_id=cluster_id(['D1:1'])))


def test_reopen_rejected_restores():
    s = _replay(curation.make_row('reject', cluster_id=T), curation.make_row('reopen', cluster_id=T))
    assert s.tracks[T]['status'] == 'proposed'
    assert len(s.tracks) == 3


def test_reopen_rejected_refused_after_join():
    s = _replay(curation.make_row('reject', cluster_id=T),
                curation.make_row('merge', parent_ids=[cluster_id(['D1:0']), cluster_id(['D1:1'])]))
    with pytest.raises(ValueError, match='joined another track'):
        curation.apply(s, curation.make_row('reopen', cluster_id=T))


def test_replay_skips_inapplicable_action():
    split = curation.make_row('split', cluster_id='uc_000000000000', members=['D1:0'])
    s = _replay(split, curation.make_row('accept', cluster_id=T))
    assert split['op_id'] in [op for op, _ in s.skipped]
    assert s.tracks[T]['status'] == 'accepted'


def test_verdict_latest_wins_and_undo():
    first = curation.make_row('verdict', cellid_a='D2:0', cellid_b='D1:0', verdict='accept')
    second = curation.make_row('verdict', cellid_a='D2:0', cellid_b='D1:0', verdict='reject')
    s = _replay(first, second, _undo(second))
    assert s.verdicts[('D1:0', 'D2:0')]['verdict'] == 'accept'


def test_comma_cellid_roundtrips():
    tracks = [['a,1', 'b,1']]
    split = curation.make_row('split', cluster_id=cluster_id(['a,1', 'b,1']), members=['a,1'])
    s = curation.replay(tracks, [], [split])
    assert set(s.tracks) == {cluster_id(['a,1']), cluster_id(['b,1'])}
    assert curation.make_row('verdict', cellid_a='a,1', cellid_b='b,1', verdict='accept', notes='x\ny')['notes'] == 'x y'


def test_malformed_row_skipped():
    acc = curation.make_row('accept', cluster_id=T)
    bad = {c: '' for c in curation.COLUMNS}
    bad.update(op_id='op_frobnicate0', operation='frobnicate')
    s = _replay(acc, dict(acc), bad)
    assert len(s.skipped) == 2
    assert all(reason.startswith('malformed') for _, reason in s.skipped)
    assert s.tracks[T]['status'] == 'accepted'


def test_malformed_row_reserves_op_id():
    acc = curation.make_row('accept', cluster_id=T)
    bad = dict(acc, members='not json')
    s = _replay(bad, acc)
    assert [op for op, _ in s.skipped] == [acc['op_id'], acc['op_id']]
    assert all(reason.startswith('malformed') for _, reason in s.skipped)
    assert s.tracks[T]['status'] == 'proposed'
    assert s.undo_target is None


def test_verdict_unknown_unit_skipped():
    row = curation.make_row('verdict', cellid_a='Z9:0', cellid_b='D1:0', verdict='accept')
    s = _replay(row)
    assert row['op_id'] in [op for op, _ in s.skipped]
    assert s.verdicts == {}


def test_columns_are_fixed():
    literal = ('op_id', 'at', 'curator', 'operation', 'cluster_id', 'parent_ids', 'members',
               'target_op', 'cellid_a', 'cellid_b', 'verdict', 'notes')
    assert curation.COLUMNS == literal
    assert set(curation.make_row('accept', cluster_id=T)) == set(literal)


def test_rapid_actions_remain_distinct_when_clock_does_not_advance(monkeypatch):
    monkeypatch.setattr(curation.time, 'time_ns', lambda: 1234567890)
    single = cluster_id(['D1:1'])
    accepted = curation.make_row('accept', cluster_id=T)
    accepted_single = curation.make_row('accept', cluster_id=single)
    first = curation.make_row('verdict', cellid_a='D1:0', cellid_b='D2:0', verdict='accept')
    second = curation.make_row('verdict', cellid_a='D1:0', cellid_b='D2:0', verdict='reject')
    rows = [accepted, accepted_single, first, second, _undo(second), _undo(first)]

    assert len({row['op_id'] for row in rows}) == len(rows)
    state = _replay(*rows)
    assert state.skipped == []
    assert state.tracks[T]['status'] == state.tracks[single]['status'] == 'accepted'
    assert state.verdicts == {}
    assert state.redo_target == first['op_id']
    restored = _replay(*rows, curation.make_row('redo', target_op=first['op_id']))
    assert restored.skipped == []
    assert restored.verdicts[('D1:0', 'D2:0')]['verdict'] == 'accept'


def test_legacy_operation_ids_still_replay():
    accepted = curation.make_row('accept', cluster_id=T)
    accepted['op_id'] = 'op_0123456789ab'
    state = _replay(accepted, _undo(accepted))
    assert state.skipped == []
    assert state.tracks[T]['status'] == 'proposed'
    assert state.redo_target == accepted['op_id']
