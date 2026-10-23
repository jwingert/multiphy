import json

import pytest

from tests.test_benchmark import dataset
from multiphy.backends import MatchResult, save_result, load_result
from multiphy.benchmark import save_dataset, load_dataset
from multiphy.benchmark_collection import create_collection
from multiphy.comparison_report import inspect_run, write_comparison
from multiphy.evaluation import evaluate_result


def make_run(root, name, data_path, *, units_override=None):
    ds = load_dataset(data_path)
    units = ds.units if units_override is None else units_override
    pairs = ds.pairs[['cellid_a', 'cellid_b']].copy()
    pairs = pairs[pairs.cellid_a.isin(units.cellid) & pairs.cellid_b.isin(units.cellid)]
    from multiphy.contract.provenance import Provenance
    from multiphy.contract.results import Decision, ScoreBlock, ScoreKind, TrackKind
    pairs['match_score'] = .1
    pairs['selected'] = False
    pairs['decision'] = Decision.BELOW_THRESHOLD.value
    result = MatchResult('functional_likelihood', '1', name,
        units, pairs, (), tuple(units.cellid), (),
        ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score'), TrackKind.COMMON, 'mutual_margin_best_first',
        Provenance.from_json(dict(ds.metadata, benchmark_input=str(data_path), benchmark_partition='test'))).validate()
    run = root / name
    run.mkdir()
    save_result(result, run / 'result.json')
    (run / 'run_manifest.json').write_text(json.dumps(dict(status='completed', backend='functional_likelihood',
        input_path=str(data_path), dataset_id=ds.metadata['dataset_id'], partition='test')), encoding='utf-8')
    (run / 'evaluation.json').write_text(json.dumps(dict(combined=evaluate_result(result, ds.labels))), encoding='utf-8')
    return run


def test_same_population_grouped_but_missing_counterparts_separated(tmp_path):
    original = save_dataset(dataset(), tmp_path / 'original', cache_root=tmp_path)
    missing = create_collection([original], tmp_path / 'missing', dataset_id='missing',
        omissions={str(original): [load_dataset(original).units.cellid.iloc[-1]]}, root=tmp_path)
    same = create_collection([original], tmp_path / 'same', dataset_id='collection', root=tmp_path)
    a = make_run(tmp_path, 'a', original)
    b = make_run(tmp_path, 'b', same)
    c = make_run(tmp_path, 'c', missing)
    assert set(load_dataset(same).units.cellid) != set(load_dataset(original).units.cellid)
    assert inspect_run(a)['population_id'] == inspect_run(b)['population_id']
    assert inspect_run(a)['population_id'] != inspect_run(c)['population_id']
    output = write_comparison([a, b, c], 'report', root=tmp_path)
    text = (output / 'comparison.md').read_text(encoding='utf-8')
    assert text.count('## Population ') == 2
    assert 'not independently verified across-day' in text
    assert 'Not defined' in text and 'Known unmatched observations' in text
    assert 'removes 1 whole observations' in text
    assert 'not a score or an additional scientific condition' in text
    assert (output / 'METHODS.md').exists()
    with pytest.raises(FileExistsError):
        write_comparison([a], output, root=tmp_path)
    with pytest.raises(ValueError, match='stay under'):
        write_comparison([a], tmp_path.parent / 'escape', root=tmp_path)


def test_stale_metrics_and_incomplete_runs_rejected(tmp_path):
    data = save_dataset(dataset(), tmp_path / 'data', cache_root=tmp_path)
    run = make_run(tmp_path, 'run', data)
    evaluation = json.loads((run / 'evaluation.json').read_text(encoding='utf-8'))
    evaluation['combined']['false_negative_pairs'] = 99
    (run / 'evaluation.json').write_text(json.dumps(evaluation), encoding='utf-8')
    with pytest.raises(ValueError, match='disagrees with saved tracks'):
        inspect_run(run)
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest['status'] = 'failed'
    (run / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    with pytest.raises(ValueError, match='not completed'):
        inspect_run(run)


def test_partial_partition_cannot_be_labeled_as_full_population(tmp_path):
    data = save_dataset(dataset(), tmp_path / 'data', cache_root=tmp_path)
    ds = load_dataset(data)
    run = make_run(tmp_path, 'run', data, units_override=ds.units.iloc[:-1].copy())
    with pytest.raises(ValueError, match='exact declared'):
        inspect_run(run)


def test_local_aggregate_schema_without_backend_or_partition_fields(tmp_path):
    data = save_dataset(dataset(), tmp_path / 'data', cache_root=tmp_path)
    run = make_run(tmp_path, 'local', data)
    from multiphy.backends import result_from_pairs
    saved = load_result(run / 'result.json')
    ds = load_dataset(data)
    result = result_from_pairs('unitmatch', '1', ds.units, saved.pairs[['cellid_a', 'cellid_b', 'match_score']],
        run_id=saved.run_id, provenance=dict(ds.metadata, benchmark_input=str(data), benchmark_partition='test',
            evidence_display_name='Stimulus fingerprint and shared-stimulus firing rate'))
    save_result(result, run / 'result.json', overwrite=True)
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest.pop('backend')
    manifest.pop('partition')
    manifest['run_id'] = result.run_id
    (run / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    metrics = evaluate_result(result, load_dataset(data).labels)
    metrics.pop('run_id')
    metrics.pop('backend_id')
    (run / 'evaluation.json').write_text(json.dumps(dict(aggregate=metrics)), encoding='utf-8')
    inspected = inspect_run(run)
    assert inspected['partition'] == 'test'
    assert inspected['result'].backend_id == 'unitmatch'
    assert 'Stimulus fingerprint' not in inspected['name']


@pytest.mark.parametrize('assignment', ['native_intermediate', 'mutual_margin_best_first',
                                       'common_constrained_operating_point'])
def test_report_names_actual_tracking_policy_and_calibration_target(tmp_path, assignment):
    data = save_dataset(dataset(), tmp_path / 'data', cache_root=tmp_path)
    run = make_run(tmp_path, 'tracking', data)
    result = load_result(run / 'result.json')
    result.backend_id = 'unitmatch'
    result.provenance = result.provenance.merged(extra=dict(assignment=assignment, threshold=.83))
    if assignment == 'common_constrained_operating_point':
        result.provenance = result.provenance.merged(extra=dict(
            assignment='common_constrained', assignment_selection='calibration_operating_point'))
        result.provenance = result.provenance.merged(extra={'target_empirical_false_match_fraction': .01})
    save_result(result, run / 'result.json', overwrite=True)
    manifest = json.loads((run / 'run_manifest.json').read_text(encoding='utf-8'))
    manifest['backend'] = 'unitmatch'
    (run / 'run_manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    (run / 'evaluation.json').write_text(json.dumps(dict(combined=evaluate_result(result, load_dataset(data).labels))), encoding='utf-8')
    output = write_comparison([run], 'report', root=tmp_path)
    text = (output / 'comparison.md').read_text(encoding='utf-8')
    if assignment == 'native_intermediate':
        assert 'Native intermediate tracking first requires' in text
    else:
        assert 'Native intermediate tracking first requires' not in text
        assert 'The common tracker requires candidate competition' in text
    if assignment == 'common_constrained_operating_point':
        assert 'calibration target: 1% incorrect assigned pairs' in text
        assert 'does not guarantee the same error fraction on test data' in text
    assert 'Recorded threshold: 0.83' in text
