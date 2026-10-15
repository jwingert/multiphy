# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify leakage-safe frozen empirical assignment operating points | date: 2026-09-18]
import copy
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the score-kind refusal test replaces the score block (Follow-up Task 5) | date: 2026-09-29]
import dataclasses
# [AGENT EDIT END]
import json

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match.backends import MatchResult, save_result
from nems_lbhb.unit_match.benchmark import BenchmarkDataset, save_dataset
from nems_lbhb.unit_match import operating_points as op
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the score-kind refusal test builds a calibrated score block (Follow-up Task 5) | date: 2026-09-29]
from nems_lbhb.unit_match.contract.results import ScoreBlock, ScoreKind
# [AGENT EDIT END]


HASH = 'a' * 64


def dataset(n_sessions=2):
    rows, labels, features, geometry, pairs, sources = [], [], {}, {}, [], []
    for partition in ('train', 'calibration', 'test'):
        well, original = partition + '_well', partition + '_sort'
        sources.append(dict(well=well, source_session=original, partition=partition,
                            preprocessing='synthetic raw means', provenance={'probe_type': 'synthetic'}))
        for session in range(n_sessions):
            sid = f'{partition}{session}'
            geometry[sid] = np.array([[0., 0.], [0., 20.]])
            for unit in range(2):
                cid = f'{sid}-u{unit}'
                rows.append(dict(cellid=cid, session_id=sid, siteid=sid, su_type='SU', recording_group=original))
                labels.append(dict(cellid=cid, source_unit_id=f'source{unit}', source_session=original, well=well, partition=partition))
                features[cid] = dict(eligible=True, waveform_halves=np.ones((5, 2, 2)), sample_rate=1000., fingerprint_bin_seconds=.01,
                    start_sample=session*100, stop_sample=(session+1)*100)
        group = [r for r in rows if r['recording_group'] == original]
        for i, a in enumerate(group):
            for b in group[i+1:]:
                if a['session_id'] != b['session_id']:
                    pairs.append(dict(cellid_a=a['cellid'], cellid_b=b['cellid'], fingerprint_r=.8))
    return BenchmarkDataset(pd.DataFrame(rows), pd.DataFrame(pairs), features, pd.DataFrame(labels), geometry,
        dict(schema_version=1, dataset_id='full', block_id='full', runclass='BNT', cache_built='fixed',
             sources=sources, extraction_options={}, pair_options={})).validate()


def score_result(ds, partition='calibration', values=(.95, .2, .1, .8), backend='unitmatch', digest=HASH):
    ids = set(ds.labels.loc[ds.labels.partition == partition, 'cellid'])
    units = ds.units[ds.units.cellid.isin(ids)].copy().reset_index(drop=True)
    pairs = ds.pairs[ds.pairs.cellid_a.isin(ids) & ds.pairs.cellid_b.isin(ids)].copy().reset_index(drop=True)
    pairs['match_score'] = np.asarray(values, dtype=float)
    pairs['official_probability_ab'] = pairs.match_score
    pairs['official_probability_ba'] = pairs.match_score
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: the score kind lives on the result's score block, not a score_kind column (Task 12) | date: 2026-09-22]
    pairs['selected'], pairs['decision'] = False, 'scored'
    # [AGENT EDIT END]
    p = dict(ds.metadata, input_manifest_sha256=digest, benchmark_partition=partition,
        official_source='upstream-url', verified_api_revision='pinned',
        implementation={'module': 'native.score', 'sha256': 'b' * 64, 'path': '/old/install'},
        parameters={'n_units': len(units), 'n_sessions': units.session_id.nunique(), 'n_expected_matches': 4,
                    'channel_radius': 150, 'match_threshold': .5},
        pair_reduction='minimum of both directional probabilities',
        waveform_provenance=dict(ds.metadata, source_groups={s: partition + '_well' for s in units.session_id}))
    if backend == 'deepunitmatch':
        p.update(checkpoint_sha256='c' * 64, checkpoint_probe='synthetic', network_preprocessing={'crop': 5},
                 training={'training_groups': ['train_well']})
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: 12-field MatchResult: uncalibrated native scores, native tracks, typed Provenance (Task 12) | date: 2026-09-22]
    from nems_lbhb.unit_match.contract.provenance import Provenance
    from nems_lbhb.unit_match.contract.results import ScoreBlock, ScoreKind, TrackKind
    return MatchResult(backend, 'native1', partition + '_scores', units, pairs, (), tuple(units.cellid), (),
                       ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score'), TrackKind.NATIVE, 'native_intermediate',
                       Provenance.from_json(p)).validate()
    # [AGENT EDIT END]


def select(ds, scores=None, **kwargs):
    return op.select_operating_point(ds, scores or score_result(ds), input_manifest_sha256=HASH, target_fdr=0., **kwargs)


def test_selection_uses_calibration_only_and_apply_never_reselects(tmp_path, monkeypatch):
    ds = dataset()
    chosen = select(ds)
    assert chosen['selection']['threshold'] == .8
    changed = copy.deepcopy(ds)
    mask = changed.labels.partition == 'test'
    changed.labels.loc[mask, 'source_unit_id'] = ['t0', 't1', 't1', 't0']
    new = select(changed)
    assert chosen['selection']['calibration_curve'] == new['selection']['calibration_curve']
    assert chosen['selection']['threshold'] == new['selection']['threshold']
    monkeypatch.setattr(op, 'select_operating_point', lambda *a, **k: pytest.fail('Apply must not select a threshold'))
    result, metrics = op.apply_operating_point(ds, score_result(ds, 'test'), chosen, input_manifest_sha256=HASH)
    assert metrics['aggregate']['true_positive_pairs'] == 2
    assert metrics['aggregate']['false_positive_pairs'] == 0
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('threshold') == .8
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('operating_point_sha256') == chosen['sha256']
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('target_empirical_false_match_fraction') == 0.
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('calibration_selection')['true_positive_pairs'] == 2
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('calibration_source_groups') == ['calibration_well']
    # [AGENT EDIT END]


def test_score_one_missing_evidence_ties_and_no_qualified_point():
    ds = dataset()
    one = select(ds, score_result(ds, values=(1., np.nan, np.nan, np.nan)))
    assert one['selection']['threshold'] == 1.
    assert one['selection']['calibration_selection']['n_unmatched'] == 2
    tied = select(ds, score_result(ds, values=(.9, .9, .9, .9)))
    assert tied['selection']['status'] == 'no_qualified_operating_point'
    assert tied['selection']['calibration_curve'][0]['empirical_false_match_fraction'] is None
    missing = select(ds, score_result(ds, values=(np.nan,) * 4))
    assert missing['selection']['threshold'] is None and missing['selection']['calibration_curve'] == []
    wrong = select(ds, score_result(ds, values=(.1, .95, .94, .1)))
    assert wrong['selection']['status'] == 'no_qualified_operating_point'
    with pytest.raises(ValueError, match='No qualified'):
        op.apply_operating_point(ds, score_result(ds, 'test'), wrong, input_manifest_sha256=HASH)


def test_final_implied_track_pairs_determine_error_not_selected_edges():
    ds = dataset(n_sessions=3)
    scores = score_result(ds, values=np.full(12, np.nan))
    for a, b, value in [('calibration0-u0', 'calibration1-u0', .9), ('calibration1-u0', 'calibration2-u1', .8)]:
        idx = (scores.pairs.cellid_a == a) & (scores.pairs.cellid_b == b)
        scores.pairs.loc[idx, ['match_score', 'official_probability_ab', 'official_probability_ba']] = value
    chosen = op.select_operating_point(ds, scores, input_manifest_sha256=HASH, target_fdr=.8)
    at_eight = next(r for r in chosen['selection']['calibration_curve'] if r['threshold'] == .8)
    assert at_eight['assigned_pairs'] == 3 and at_eight['false_positive_pairs'] == 2
    assert at_eight['empirical_false_match_fraction'] == 2 / 3
    assert chosen['selection']['threshold'] == .9


def test_full_archive_leakage_population_and_unknown_pair_guards():
    ds = dataset()
    score = score_result(ds)
    leaked = copy.deepcopy(ds)
    leaked.labels.loc[leaked.labels.partition == 'test', 'well'] = 'calibration_well'
    with pytest.raises(ValueError, match='leakage'):
        select(leaked, score)
    score.units.loc[0, 'session_id'] = 'unknown_session'
    with pytest.raises(ValueError, match='population and session'):
        select(ds, score)
    score = score_result(ds)
    score.pairs.loc[0, 'cellid_b'] = 'unknown_unit'
    with pytest.raises(ValueError, match='endpoints'):
        select(ds, score)
    with pytest.raises(ValueError, match='native baseline'):
        select(ds, score_result(ds, backend='dant'))


@pytest.mark.parametrize('change', ['parameter', 'revision', 'implementation', 'reduction', 'archive', 'directional', 'waveform'])
def test_native_scoring_configuration_and_identity_mismatch_rejected(change):
    ds = dataset()
    chosen = select(ds)
    score = score_result(ds, 'test')
    if change == 'parameter':
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
        score.provenance.get('parameters')['channel_radius'] = 200
        # [AGENT EDIT END]
    elif change == 'revision':
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write through .merged (Task 12) | date: 2026-09-22]
        score.provenance = score.provenance.merged(verified_api_revision='different')
        # [AGENT EDIT END]
    elif change == 'implementation':
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
        score.provenance.get('implementation')['sha256'] = 'e' * 64
        # [AGENT EDIT END]
    elif change == 'reduction':
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write through .merged (Task 12) | date: 2026-09-22]
        score.provenance = score.provenance.merged(extra={'pair_reduction': 'mean'})
        # [AGENT EDIT END]
    elif change == 'archive':
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write through .merged (Task 12) | date: 2026-09-22]
        score.provenance = score.provenance.merged(input_manifest_sha256='e' * 64)
        # [AGENT EDIT END]
    elif change == 'waveform':
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
        score.provenance.get('waveform_provenance')['extraction_options'] = {'before': 999}
        # [AGENT EDIT END]
    else:
        score.pairs.loc[0, 'official_probability_ab'] = .1
    with pytest.raises(ValueError, match='differ|identity|reduction'):
        op.apply_operating_point(ds, score, chosen, input_manifest_sha256=HASH)


def test_population_fitted_native_parameters_and_install_locations_may_differ():
    ds = dataset()
    chosen = select(ds)
    score = score_result(ds, 'test')
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    score.provenance.get('parameters').update(n_units=999, n_sessions=9, n_expected_matches=57)
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    score.provenance.get('implementation')['path'] = '/durable/install'
    # [AGENT EDIT END]
    _, metrics = op.apply_operating_point(ds, score, chosen, input_manifest_sha256=HASH)
    assert metrics['aggregate']['true_positive_pairs'] == 2
    assert chosen['selection']['compatibility_excluded_population_parameters'] == list(op.POPULATION_PARAMETERS)


def test_deep_checkpoint_and_training_groups_are_part_of_frozen_contract():
    ds = dataset()
    scores = score_result(ds, backend='deepunitmatch')
    chosen = select(ds, scores)
    score = score_result(ds, 'test', backend='deepunitmatch')
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write through .merged (Task 12) | date: 2026-09-22]
    score.provenance = score.provenance.merged(extra={'checkpoint_sha256': 'd' * 64})
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='checkpoint configurations differ'):
        op.apply_operating_point(ds, score, chosen, input_manifest_sha256=HASH)
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    score.provenance.get('training')['training_groups'] = ['test_well']
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='verified training wells'):
        op.scoring_signature(ds, score)


def test_artifacts_are_hashed_immutable_cache_only_and_preserve_native_results(tmp_path):
    ds = dataset()
    archive = save_dataset(ds, 'data', cache_root=tmp_path)
    digest = op._file_hash(archive / 'manifest.json')
    calibration, test = tmp_path / 'calibration.json', tmp_path / 'test.json'
    save_result(score_result(ds, digest=digest), calibration)
    save_result(score_result(ds, 'test', digest=digest), test)
    empty = tmp_path / 'missing-scores.json'
    save_result(score_result(ds, values=(np.nan,) * 4, digest=digest), empty)
    empty_output = op.run_select(archive, empty, target_fdr=0., output_dir='no-scores', root=tmp_path)
    assert json.loads((empty_output / 'selection.json').read_text())['selection']['status'] == 'no_qualified_operating_point'
    old = test.read_bytes()
    output = op.run_select(archive, calibration, target_fdr=0., output_dir='selected', root=tmp_path)
    applied = op.run_apply(archive, test, output / 'selection.json', output_dir='applied', root=tmp_path)
    assert test.read_bytes() == old
    from nems_lbhb.unit_match.comparison_report import inspect_run
    inspected = inspect_run(applied)
    assert inspected
    saved_result = op.load_result(applied / 'result.json')
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert saved_result.provenance.get('selection_file_sha256') == op._file_hash(output / 'selection.json')
    # [AGENT EDIT END]
    assert all((applied / p).exists() for p in ('result.json', 'evaluation.json', 'METHODS.md', 'report.md'))
    assert 'not a newly prospective' in (applied / 'report.md').read_text()
    with pytest.raises(FileExistsError):
        op.run_select(archive, calibration, target_fdr=0., output_dir='selected', root=tmp_path)
    with pytest.raises(ValueError, match='stay under'):
        op.run_select(archive, calibration, target_fdr=0., output_dir=tmp_path.parent / 'escape', root=tmp_path)
    document = json.loads((output / 'selection.json').read_text())
    document['selection']['threshold'] = .1
    with pytest.raises(ValueError, match='content hash'):
        op.apply_operating_point(ds, score_result(ds, 'test', digest=digest), document, input_manifest_sha256=digest)
def test_explicit_parent_allows_only_test_omissions_without_reselection(monkeypatch):
    ds = dataset()
    chosen = select(ds)
    view = ds.subset(set(ds.units.cellid) - {'test1-u1'})
    view.metadata = dict(view.metadata, dataset_id='missing', block_id='missing', cache_built='new')
    score = score_result(ds, 'test')
    ids = set(view.units.cellid)
    score.units = score.units[score.units.cellid.isin(ids)].copy()
    score.pairs = score.pairs[score.pairs.cellid_a.isin(ids) & score.pairs.cellid_b.isin(ids)].copy()
    score.unmatched = tuple(score.units.cellid)
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write through .merged (typed dataset_id and manifest hash) (Task 12) | date: 2026-09-22]
    score.provenance = score.provenance.merged(dataset_id='missing', input_manifest_sha256='d'*64,
                                               extra=dict(block_id='missing', cache_built='new'))
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    score.provenance.get('waveform_provenance').update(dataset_id='missing', block_id='missing', cache_built='new')
    # [AGENT EDIT END]
    with pytest.raises(ValueError, match='explicit selection parent'):
        op.apply_operating_point(view, score, chosen, input_manifest_sha256='d'*64)
    monkeypatch.setattr(op, 'select_operating_point', lambda *a, **k: pytest.fail('Missing view must not reselect'))
    result, metrics = op.apply_operating_point(view, score, chosen, input_manifest_sha256='d'*64,
        selection_dataset=ds, selection_manifest_sha256=HASH)
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('threshold') == .8
    # [AGENT EDIT END]
    assert metrics['aggregate']['true_positive_pairs'] == 1
    assert metrics['aggregate']['true_unmatched_units'] == metrics['aggregate']['correctly_unmatched_units'] == 1
    changed = copy.deepcopy(view)
    changed.features['train0-u0']['waveform_halves'][0, 0, 0] = 2.
    with pytest.raises(ValueError, match='retained measured features'):
        op.apply_operating_point(changed, score, chosen, input_manifest_sha256='d'*64,
            selection_dataset=ds, selection_manifest_sha256=HASH)
    changed = ds.subset(set(ds.units.cellid) - {'calibration1-u1', 'test1-u1'})
    with pytest.raises(ValueError, match='omit test observations only'):
        op._validate_missing_view(changed, ds, HASH, chosen['selection'])


def test_preprocessor_storage_is_not_a_transform_but_unknown_fields_are_retained():
    ds = dataset()
    for source in ds.metadata['sources']:
        source['provenance']['waveform_preprocessing'] = dict(profile='same', highpass_hz=150, storage='cached snippets')
    chosen = select(ds)
    changed = copy.deepcopy(ds)
    changed.metadata['sources'][-1]['provenance']['waveform_preprocessing']['storage'] = 'streaming summaries'
    _, evaluation = op.apply_operating_point(changed, score_result(changed, 'test'), chosen, input_manifest_sha256=HASH)
    assert evaluation['aggregate']['true_positive_pairs'] == 2
    changed.metadata['sources'][-1]['provenance']['waveform_preprocessing']['future_filter_option'] = True
    with pytest.raises(ValueError, match='configurations differ'):
        op.apply_operating_point(changed, score_result(changed, 'test'), chosen, input_manifest_sha256=HASH)
def test_deleting_a_competitor_cannot_create_a_calibration_operating_point():
    ds = dataset()
    native = score_result(ds, values=(.95, .94, .1, .8))
    assert select(ds, native)['selection']['status'] == 'no_qualified_operating_point'
    native.pairs = native.pairs.drop(index=1)
    with pytest.raises(ValueError, match='pair coverage'):
        select(ds, native)
    # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no frozen local scores to delete from (Task 2) | date: 2026-09-27]


def test_nested_real_collection_namespaces_share_frozen_cutoff(tmp_path):
    from nems_lbhb.unit_match.benchmark_collection import create_collection
    base = dataset()
    sessions = base.units.set_index('cellid').session_id
    for suffix in ('a', 'b'):
        base.pairs['siteid_' + suffix] = base.pairs['cellid_' + suffix].map(sessions)
    leaves = {}
    for partition in ('train', 'calibration', 'test'):
        ds = base.subset(base.labels.loc[base.labels.partition == partition, 'cellid'])
        ds.metadata = dict(ds.metadata, dataset_id=partition, block_id=partition,
            sources=[s for s in ds.metadata['sources'] if s['partition'] == partition])
        leaves[partition] = save_dataset(ds, partition, cache_root=tmp_path)
    full_path = create_collection(list(leaves.values()), 'full-collection', dataset_id='full-collection', root=tmp_path)
    missing_leaf = create_collection([leaves['test']], 'missing-test', dataset_id='missing-test',
        omissions={str(leaves['test']): ['test1-u1']}, root=tmp_path)
    view_path = create_collection([leaves['train'], leaves['calibration'], missing_leaf],
        'missing-collection', dataset_id='missing-collection', root=tmp_path)
    full, full_hash = op._load_archive(full_path)
    view, view_hash = op._load_archive(view_path)
    chosen = op.select_operating_point(full, score_result(full, digest=full_hash),
        input_manifest_sha256=full_hash, target_fdr=0.)
    assert not set(view.labels.loc[view.labels.partition == 'test', 'cellid']) & set(full.labels.loc[full.labels.partition == 'test', 'cellid'])
    scored = score_result(view, 'test', values=(.95, .1), digest=view_hash)
    result, evaluation = op.apply_operating_point(view, scored, chosen, input_manifest_sha256=view_hash,
        selection_dataset=full, selection_manifest_sha256=full_hash)
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('threshold') == chosen['selection']['threshold'] == .8
    # [AGENT EDIT END]
    assert evaluation['aggregate']['true_positive_pairs'] == evaluation['aggregate']['correctly_unmatched_units'] == 1
    assert set(result.units.cellid) == set(scored.units.cellid)  # Actual view IDs are retained.
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is read with .get (Task 12) | date: 2026-09-22]
    assert result.provenance.get('verified_view_correspondence_sha256')
    # [AGENT EDIT END]
    altered = copy.deepcopy(view)
    altered.pairs.loc[altered.pairs.index[-1], 'siteid_a'] = 'invented_prefix:test0'
    with pytest.raises(ValueError, match='unverified observation namespace'):
        op.apply_operating_point(altered, scored, chosen, input_manifest_sha256=view_hash,
            selection_dataset=full, selection_manifest_sha256=full_hash)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: generated notes describe only the remaining native-score and checkpoint checks (Task 2 fix round 2) | date: 2026-09-27]
def test_generated_notes_claim_no_fitted_model(tmp_path):
    op._notes(tmp_path, 'select', tmp_path)
    text = ((tmp_path / 'METHODS.md').read_text() + (tmp_path / 'report.md').read_text()).lower()
    assert 'fitted model' not in text
    assert 'local probability calibration' not in text
    assert 'checkpoint hashes' in text
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a candidate-pair backend's scores are uncalibrated like every native score (Plan B Task 9) | date: 2026-09-28]
def test_random_baseline_scores_are_uncalibrated():
    from nems_lbhb.unit_match.contract import registry
    from tests.unit_match.example_backend import RandomBaselineBackend
    ds = dataset()
    registry.register(RandomBaselineBackend())
    try:
        result = score_result(ds, backend='random_baseline')
        assert op._validate_scores(ds, result, 'calibration', HASH) == 'uncalibrated_score'
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a sparse candidate table no longer defines coverage (Plan B Task 9 fix round 1) | date: 2026-09-29]
        sparse = copy.deepcopy(ds)
        dropped = ('calibration0-u0', 'calibration1-u1')
        keep = ~((sparse.pairs.cellid_a == dropped[0]) & (sparse.pairs.cellid_b == dropped[1]))
        assert (~keep).sum() == 1
        sparse.pairs = sparse.pairs[keep].reset_index(drop=True)
        sparse_result = score_result(sparse, values=(.95, .2, .1), backend='random_baseline')
        assert set(sparse_result.units.cellid) == set(result.units.cellid)
        with pytest.raises(ValueError, match='Saved score pair coverage is incomplete'):
            op._validate_scores(sparse, sparse_result, 'calibration', HASH)
        # [AGENT EDIT END]
    finally:
        registry.clear_for_tests()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a calibrated score kind is refused (Follow-up Task 5) | date: 2026-09-29]
def test_calibrated_score_kind_is_refused():
    ds = dataset()
    # a calibrated block needs a calibration_reference to pass result.validate(), so the kind check is what refuses
    result = dataclasses.replace(score_result(ds), score=ScoreBlock(ScoreKind.CALIBRATED_PROBABILITY, 'match_score',
                                                                    calibration_reference='synthetic_calibration'))
    with pytest.raises(ValueError) as err:
        op._validate_scores(ds, result, 'calibration', HASH)
    assert str(err.value) == "Score kind 'calibrated_probability' does not match the expected 'uncalibrated_score'"
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: same-session score rows are ignored for selection and coverage (native-same-day Task 5) | date: 2026-10-11]
def _with_same_session_rows(ds, partition):
    scores = score_result(ds, partition)
    extra = pd.DataFrame([dict(cellid_a=f'{partition}{s}-u0', cellid_b=f'{partition}{s}-u1', fingerprint_r=.8,
                               match_score=.99, official_probability_ab=.99, official_probability_ba=.99,
                               selected=False, decision='same_session') for s in (0, 1)])
    return dataclasses.replace(scores, pairs=pd.concat([scores.pairs, extra], ignore_index=True))


def test_same_session_rows_leave_selection_unchanged():
    ds = dataset()
    scores = _with_same_session_rows(ds, 'calibration')
    assert len(scores.pairs) == 6
    chosen = select(ds, scores)
    curve = chosen['selection']['calibration_curve']
    assert [c['threshold'] for c in curve] == [0.1, 0.2, 0.8, 0.95]
    assert [c['true_positive_pairs'] for c in curve] == [2, 2, 2, 1]
    assert [c['false_positive_pairs'] for c in curve] == [0, 0, 0, 0]
    assert chosen['selection']['threshold'] == 0.8
    result, metrics = op.apply_operating_point(ds, _with_same_session_rows(ds, 'test'), chosen, input_manifest_sha256=HASH)
    assert metrics['aggregate']['true_positive_pairs'] == 2
    assert metrics['aggregate']['false_positive_pairs'] == 0
    assert result.tracks == (('test0-u0', 'test1-u0'), ('test0-u1', 'test1-u1'))
# [AGENT EDIT END]
