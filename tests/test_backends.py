import pandas as pd
import pytest

from multiphy import backends as b


def units(*ids):
    return pd.DataFrame({'cellid': ids, 'su_type': ['SU'] * len(ids)})


def pairs(*edges):
    return pd.DataFrame(edges, columns=['cellid_a', 'cellid_b', 'p_global'])


def track(u, p, score_column='p_global', **kwargs):
    return b.result_from_pairs('unitmatch', '1', u, p, score_column=score_column, **kwargs)


@pytest.fixture
def pair_backend():
    """A candidate-pair backend registered for the run_backend adapter checks, then removed."""
    from multiphy.contract import registry
    from multiphy.contract.backend import COMMON_TRACKER_OPTIONS, Availability, BackendSpec, Provides
    from multiphy.contract.inputs import Input

    class PairBackend:
        spec = BackendSpec('pairs_test', '1', frozenset({Input.UNITS, Input.CANDIDATE_PAIRS}), frozenset(),
                           Provides(frozenset({b.ScoreKind.UNCALIBRATED}), frozenset({b.TrackKind.COMMON})),
                           COMMON_TRACKER_OPTIONS)

        def available(self):
            return Availability(True, '1', None)

        def run(self, inp, options, ctx):
            return b.result_from_pairs('pairs_test', '1', inp.units, inp.candidate_pairs, score_column='p_global',
                                       run_id=ctx.run_id, provenance=ctx.provenance.to_json(), **options)

    registry.clear_for_tests()
    registry.register(PairBackend())
    yield 'pairs_test'
    registry.clear_for_tests()


def test_single_units_and_unmatched_include_no_candidate_units():
    u = units('A-1', 'B-1', 'C-1', 'D-1', 'E-1')
    u.loc[3:, 'su_type'] = ['MU', None]
    result = track(u, pairs(('A-1', 'B-1', .99), ('A-1', 'D-1', .999)))
    assert result.tracks == (('A-1', 'B-1'),)
    assert result.unmatched == ('C-1',)
    assert result.excluded_units == ('D-1', 'E-1')
    assert len(result.pairs) == 1
    assert result.score.kind.value == 'uncalibrated_score'


def test_strong_measured_contradiction_blocks_transitive_join():
    u = units('A-1', 'B-1', 'C-1')
    p = pairs(('A-1', 'B-1', .99), ('B-1', 'C-1', .97), ('A-1', 'C-1', .01))
    result = track(u, p)
    assert result.tracks == (('A-1', 'B-1'),)
    assert result.unmatched == ('C-1',)
    assert result.pairs.iloc[1].decision == 'measured_contradiction'


def test_missing_evidence_is_not_a_rejection_and_is_reported():
    result = track(units('A-1', 'B-1', 'C-1'),
                           pairs(('A-1', 'B-1', .99), ('B-1', 'C-1', .98)))
    assert result.tracks == (('A-1', 'B-1', 'C-1'),)
    assert result.provenance.get('internal_evidence_coverage')[0]['n_missing'] == 1


def test_same_session_collision_cannot_enter_track():
    result = track(units('A-1', 'A-2', 'B-1', 'C-1'),
                           pairs(('A-1', 'B-1', .99), ('B-1', 'C-1', .98), ('C-1', 'A-2', .95)))
    assert result.tracks == (('A-1', 'B-1', 'C-1'),)
    assert result.unmatched == ('A-2',)
    assert result.pairs.iloc[2].decision == 'session_collision'


def test_ambiguous_alternatives_below_threshold_still_compete():
    result = track(units('A-1', 'B-1', 'B-2'),
                           pairs(('A-1', 'B-1', .901), ('A-1', 'B-2', .899)))
    assert not result.tracks
    assert result.pairs.iloc[0].decision == 'ambiguous_competition'


def test_clear_competition_selects_one_and_is_order_invariant():
    u = units('A-1', 'B-1', 'B-2')
    p = pairs(('A-1', 'B-1', .99), ('A-1', 'B-2', .93))
    result = track(u, p)
    assert result.tracks == (('A-1', 'B-1'),)
    assert result.tracks == track(u, p.iloc[::-1]).tracks


def test_explicit_session_ids_override_cellid_prefix():
    u = units('differentprefix-1', 'anotherprefix-1')
    u['session_id'] = 'same_recording'
    result = track(u, pairs(('differentprefix-1', 'anotherprefix-1', .99)))
    assert not result.tracks
    assert result.pairs.iloc[0].decision == 'same_session'


def test_stimulus_ablation_preserves_other_evidence_without_using_it():
    p = pairs(('A-1', 'B-1', .99))
    p['p_stim'] = .5
    p['fingerprint_r'] = .75
    from multiphy.backends.local.tracker import score_column_for
    result = track(units('A-1', 'B-1'), p, score_column=score_column_for(p, 'stimulus'),
                   provenance={'feature_stream': 'stimulus'})
    assert not result.tracks
    assert result.pairs.iloc[0].fingerprint_r == .75
    assert result.provenance.get('feature_stream') == 'stimulus'
    with pytest.raises(ValueError, match='No measured score'):
        score_column_for(p, 'physiology')


def test_empty_pairs_and_all_unknown_quality_are_valid():
    result = track(units('A-1'), pd.DataFrame())
    assert result.unmatched == ('A-1',)
    result = track(pd.DataFrame({'cellid': ['A-1']}), pd.DataFrame())
    assert not result.unmatched
    assert result.excluded_units == ('A-1',)
    result = track(units('A-1', 'B-1'),
                           pairs(('A-1', 'B-1', float('nan'))))
    assert result.unmatched == ('A-1', 'B-1')
    assert result.pairs.iloc[0].decision == 'missing_evidence'


def test_duplicate_units_and_pairs_rejected():
    with pytest.raises(ValueError, match='uniquely'):
        track(units('A-1', 'A-1'), pd.DataFrame())
    with pytest.raises(ValueError, match='Repeated pair'):
        track(units('A-1', 'B-1'),
                      pairs(('A-1', 'B-1', .99), ('B-1', 'A-1', .99)))


def test_external_methods_never_fake_local_results():
    with pytest.raises((KeyError, ValueError, b.BackendUnavailable)):
        b.run_backend('deepunitmatch', units('A-1'), pd.DataFrame())


def test_calibration_requires_provenance_and_run_identity_is_separate():
    u, p = units('A-1', 'B-1'), pairs(('A-1', 'B-1', .99))
    with pytest.raises(ValueError, match='calibration_reference'):
        b.result_from_pairs('external', '1', u, p, score_column='p_global', score_kind='calibrated_probability')
    first = track(u, p)
    second = track(u, p)
    assert first.run_id != second.run_id
    assert first.provenance.get('input_pair_digest') == second.provenance.get('input_pair_digest')


def test_result_validation_rejects_overlapping_members():
    result = track(units('A-1', 'B-1'), pairs(('A-1', 'B-1', .99)))
    result.unmatched = ('A-1',)
    with pytest.raises(ValueError, match='partition'):
        result.validate()


def test_invalid_score_range_is_not_clipped():
    with pytest.raises(ValueError, match='Scores must'):
        track(units('A-1', 'B-1'), pairs(('A-1', 'B-1', 2.0)))


def test_unknown_ids_and_false_calibration_are_rejected(pair_backend):
    with pytest.raises(ValueError, match='unknown unit IDs'):
        track(units('A-1'), pairs(('A-1', 'typo-1', .99)))
    with pytest.raises(ValueError, match='cannot be relabeled'):
        b.run_backend(pair_backend, units('A-1', 'B-1'), pairs(('A-1', 'B-1', .99)),
                      score_kind='calibrated_probability', provenance={'calibration_reference': 'claimed'})


def test_json_roundtrip_preserves_partition_flags_and_missing_scores(tmp_path):
    import json
    import numpy as np

    result = track(units('A-1', 'B-1', 'C-1'),
                           pairs(('A-1', 'B-1', .99), ('A-1', 'C-1', np.nan)),
                           provenance={'array_metadata': np.array([1., np.nan]), 'count': np.int64(3)})
    path = b.save_result(result, tmp_path / 'result.json')
    document = json.loads(path.read_text())
    assert document['result']['provenance']['array_metadata'] == [1., None]
    assert document['result']['pairs']['records'][1]['match_score'] is None
    restored = b.load_result(path)
    assert restored.tracks == result.tracks
    assert restored.unmatched == result.unmatched
    assert restored.run_id == result.run_id
    assert restored.pairs.selected.tolist() == [True, False]
    assert pd.isna(restored.pairs.iloc[1].match_score)
    with pytest.raises(FileExistsError):
        b.save_result(result, path)
    b.save_result(result, path, overwrite=True)
    assert not list(tmp_path.glob('*.tmp'))


def test_json_roundtrip_of_empty_result_tables(tmp_path):
    result = track(units('A-1'), pd.DataFrame())
    restored = b.load_result(b.save_result(result, tmp_path / 'empty.json'))
    assert restored.unmatched == ('A-1',)
    assert list(restored.pairs.columns) == list(result.pairs.columns)


def test_json_import_rejects_corruption_and_wrong_schema(tmp_path):
    import json

    result = track(units('A-1'), pd.DataFrame())
    path = b.save_result(result, tmp_path / 'result.json')
    document = json.loads(path.read_text())
    document['result']['unmatched'] = []
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='digest'):
        b.load_result(path)
    document['sha256'] = b._payload_digest(document['result'])
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='partition'):
        b.load_result(path)
    document['schema_version'] = 999
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='schema version'):
        b.load_result(path)


def test_run_backend_refuses_undeclared_arguments(tmp_path, pair_backend):
    with pytest.raises(ValueError, match='calibrated_model'):
        b.run_backend(pair_backend, units('A-1', 'B-1'), pairs(('A-1', 'B-1', .99)),
                      calibrated_model=str(tmp_path / 'model.json'))
