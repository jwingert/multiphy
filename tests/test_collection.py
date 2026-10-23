import json
from pathlib import Path
import numpy as np
import pytest
from tests.test_benchmark import source
from multiphy.benchmark import build_dataset, save_dataset, load_dataset
from multiphy.benchmark_collection import create_collection, missing_counterpart_ids
from multiphy.benchmark_pairs import build_benchmark_pairs


def track(units, pairs, **kwargs):
    from multiphy.backends import result_from_pairs
    return result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', **kwargs)


def archive(tmp_path, name, well, partition):
    ds = build_dataset([source(well=well, session=name, partition=partition)],
        dataset_id=name, extraction_options={'before': 2, 'after': 3})
    return save_dataset(ds, tmp_path / name, cache_root=tmp_path)


def test_collection_reuses_archives_and_preserves_disjoint_sessions_and_labels(tmp_path):
    a = archive(tmp_path, 'a', 'animal:1', 'train')
    b = archive(tmp_path, 'b', 'animal:2', 'test')
    out = create_collection([a, b], tmp_path / 'all', dataset_id='all', root=tmp_path)
    ds = load_dataset(out)
    assert len(ds.units) == 8 and len(ds.geometry) == 4
    assert set(ds.labels.partition) == {'train', 'test'}
    assert set(p.name for p in out.iterdir()) == {'manifest.json', 'METHODS.md'}
    original = load_dataset(a)
    for cid in original.units.cellid:
        mapped = next(c for c in ds.units.cellid if c.endswith(':' + cid))
        np.testing.assert_array_equal(ds.features[mapped]['waveform'], original.features[cid]['waveform'])
    assert all(r.cellid.split('-')[0] == r.session_id for r in ds.units.itertuples())
    assert ds.waveform_bundle().provenance['sample_rate'] == 1000.
    with pytest.raises(FileExistsError):
        create_collection([a,b], out, dataset_id='all', root=tmp_path)


def test_missing_counterpart_view_is_reproducible_and_has_no_new_waveforms(tmp_path):
    a = archive(tmp_path, 'a', 'animal:1', 'test')
    ds = load_dataset(a)
    ids = missing_counterpart_ids(ds, fraction=.5, seed=3)
    assert ids == missing_counterpart_ids(ds, fraction=.5, seed=3)
    assert len(ids) == 1
    out = create_collection([a], tmp_path / 'missing', dataset_id='missing', root=tmp_path,
        omissions={str(a):ids}, omission_rule='hash seed3 half of later observations')
    view = load_dataset(out)
    assert len(view.units) == 3 and len(view.pairs) == 2
    assert sorted(view.labels.groupby('source_unit_id').size()) == [1, 2]
    assert ids[0] not in view.features
    assert view.exclusions[-1]['reason'] == 'prespecified missing counterpart'


def test_collection_result_opens_shared_display_and_legacy_ids_remain_stable(tmp_path):
    from multiphy.viewer.modes import same_day
    a = archive(tmp_path, 'a', 'animal:1', 'test')
    b = archive(tmp_path, 'b', 'animal:2', 'test')
    out = create_collection([a, b], tmp_path / 'collection', dataset_id='collection', root=tmp_path)
    ds = load_dataset(out)
    result = track(ds.units, ds.pairs.assign(p_global=0.),
        provenance=dict(ds.metadata, benchmark_partition='test'))
    data = same_day.run_data(ds, result)
    assert len(set(data.sites().values())) == 4
    assert set(data.unit_map) == set(data.units().cellid)
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    manifest.pop('observation_id_policy')
    (out / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    legacy = load_dataset(out)
    assert set(legacy.units.cellid) == set(load_dataset(a).units.cellid) | set(load_dataset(b).units.cellid)


def test_collection_rejects_leakage_duplicate_sources_and_changed_inputs(tmp_path):
    a = archive(tmp_path, 'a', 'animal:1', 'train')
    b = archive(tmp_path, 'b', 'animal:1', 'test')
    with pytest.raises(ValueError, match='leakage'):
        create_collection([a,b],tmp_path/'bad',dataset_id='bad',root=tmp_path)
    with pytest.raises(ValueError, match='twice'):
        create_collection([a,a],tmp_path/'dup',dataset_id='dup',root=tmp_path)
    out = create_collection([a],tmp_path/'good',dataset_id='good',root=tmp_path)
    (a/'manifest.json').write_text((a/'manifest.json').read_text(encoding='utf-8')+' ', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_dataset(out)


def test_collection_rejects_unknown_omissions_and_cycles(tmp_path):
    a = archive(tmp_path, 'a', 'animal:1', 'test')
    with pytest.raises(ValueError, match='absent'):
        create_collection([a],tmp_path/'bad',dataset_id='bad',root=tmp_path,
            omissions={str(a):['absent']})
    with pytest.raises(ValueError, match='Cyclic'):
        load_dataset(a, _ancestors=(a.resolve(),))


def single_repeat_archive(tmp_path, name, well, partition, min_repeats=4, **metadata):
    ds = build_dataset([source(well=well, session=name, partition=partition)],
        dataset_id=name, extraction_options={'before': 2, 'after': 3})
    # Explicit synthetic response means: one repeated condition observed once
    # per temporal segment, with no available second independent trial half.
    for feature in ds.features.values():
        fingerprint = np.array([0., 2., 1., 3., 0., 1.])
        feature['fingerprints'] = {'sound:left': fingerprint}
        feature['trial_counts'] = {'sound:left': 1}
        feature['fingerprint_halves'] = {'sound:left': np.column_stack((fingerprint, np.full(6, np.nan)))}
        feature['half_trial_counts'] = {'sound:left': np.array([1, 0])}
    ds.metadata.update(pair_options={'min_repeats': min_repeats}, **metadata)
    ds.pairs = build_benchmark_pairs(ds.units, ds.features, ds.geometry, min_repeats=min_repeats)
    assert ds.pairs.fingerprint_r.isna().all()
    return save_dataset(ds, tmp_path / name, cache_root=tmp_path)


def test_explicit_repeat_override_rebuilds_measured_pairs_without_mutating_archives(tmp_path):
    paths = [single_repeat_archive(tmp_path, name, f'animal:{i}', partition)
             for i, (name, partition) in enumerate([('train', 'train'), ('cal', 'calibration'), ('test', 'test')])]
    before = {str(path / child): (path / child).read_bytes()
              for path in paths for child in ('manifest.json', 'features.json', 'features.npz', 'pairs.csv', 'reference_labels.csv')}
    original = [load_dataset(path) for path in paths]
    out = create_collection(paths, tmp_path / 'low_repeat_view', dataset_id='low_repeat',
                            pair_options={'min_repeats': 1}, root=tmp_path)
    ds = load_dataset(out)
    assert ds.pairs.fingerprint_status.eq('measured').all()
    assert np.allclose(ds.pairs.fingerprint_r, 1.)
    assert ds.pairs.reliability_a.isna().all() and ds.pairs.reliability_b.isna().all()
    assert set(ds.labels.partition) == {'train', 'calibration', 'test'}
    assert ds.metadata['pair_options'] == {'min_repeats': 1, 'min_reliability': None, 'max_distance_um': None}
    provenance = ds.metadata['pair_options_override_provenance']
    assert provenance['source_archives_unchanged'] is True
    assert provenance['reference_labels_used_for_pair_evidence'] is False
    assert all(s['pair_options'] == {'min_repeats': 4} for s in provenance['source_options'])
    manifest = json.loads((out / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['pair_options_override'] == ds.metadata['pair_options']
    assert set(p.name for p in out.iterdir()) == {'manifest.json', 'METHODS.md'}
    units = ds.units.set_index('cellid')
    assert all(units.loc[p.cellid_a, 'session_id'] == p.siteid_a and
               units.loc[p.cellid_b, 'session_id'] == p.siteid_b for p in ds.pairs.itertuples())
    for old in original:
        for row in old.labels.itertuples():
            retained = ds.labels[(ds.labels.well == row.well) &
                (ds.labels.source_unit_id == row.source_unit_id)]
            old_start = old.features[row.cellid]['start_sample']
            cid = next(c for c in retained.cellid if ds.features[c]['start_sample'] == old_start)
            assert ds.labels.set_index('cellid').loc[cid, 'partition'] == row.partition
            np.testing.assert_array_equal(ds.features[cid]['waveform'], old.features[row.cellid]['waveform'])
            np.testing.assert_array_equal(ds.features[cid]['fingerprints']['sound:left'],
                                          old.features[row.cellid]['fingerprints']['sound:left'])
    assert all(Path(path).read_bytes() == data for path, data in before.items())


def test_override_works_after_omission_and_default_keeps_unmeasured_source_pairs(tmp_path):
    source_path = single_repeat_archive(tmp_path, 'a', 'animal:1', 'test')
    ds = load_dataset(source_path)
    omitted = missing_counterpart_ids(ds, fraction=.5, seed=3)
    default = create_collection([source_path], tmp_path / 'default', dataset_id='default', root=tmp_path)
    assert load_dataset(default).pairs.fingerprint_r.isna().all()
    assert 'pair_options_override' not in json.loads((default / 'manifest.json').read_text(encoding='utf-8'))
    overridden = create_collection([source_path], tmp_path / 'missing', dataset_id='missing', root=tmp_path,
        omissions={str(source_path): omitted}, pair_options={'min_repeats': 1})
    view = load_dataset(overridden)
    assert len(view.units) == 3 and len(view.pairs) == 2
    assert np.isfinite(view.pairs.fingerprint_r).all()
    assert sorted(view.labels.groupby('source_unit_id').size()) == [1, 2]


def test_explicit_override_allows_different_source_pair_settings_only(tmp_path):
    a = single_repeat_archive(tmp_path, 'a', 'animal:1', 'train', min_repeats=4)
    b = single_repeat_archive(tmp_path, 'b', 'animal:2', 'test', min_repeats=7)
    with pytest.raises(ValueError, match='pair_options'):
        create_collection([a, b], tmp_path / 'bad', dataset_id='bad', root=tmp_path)
    good = create_collection([a, b], tmp_path / 'good', dataset_id='good',
                             pair_options={'min_repeats': 1}, root=tmp_path)
    assert np.isfinite(load_dataset(good).pairs.fingerprint_r).all()
    incompatible_extraction = single_repeat_archive(tmp_path, 'd', 'animal:4', 'calibration',
                                                    extraction_options={'before': 2, 'after': 3, 'seed': 12})
    with pytest.raises(ValueError, match='extraction_options'):
        create_collection([a, incompatible_extraction], tmp_path / 'extract_bad', dataset_id='bad',
                          pair_options={'min_repeats': 1}, root=tmp_path)


def test_collection_ignores_stored_runclass(tmp_path):
    old = single_repeat_archive(tmp_path, 'c', 'animal:3', 'calibration', runclass='OTHER')
    a = single_repeat_archive(tmp_path, 'a', 'animal:1', 'train')
    out = create_collection([old, a], tmp_path / 'mixed', dataset_id='mixed',
                            pair_options={'min_repeats': 1}, root=tmp_path)
    assert 'runclass' not in load_dataset(out).metadata


@pytest.mark.parametrize('options', [[], {'min_repeats': 0}, {'min_repeats': -1},
    {'min_repeats': 1.5}, {'min_repeats': True}, {'unknown': 1}, {'min_reliability': np.nan},
    {'min_reliability': 1.1}, {'max_distance_um': -1}, {'max_distance_um': np.inf}])
def test_bad_override_is_rejected_before_inputs_or_outputs_are_touched(tmp_path, options):
    with pytest.raises(ValueError):
        create_collection([tmp_path / 'missing_input'], tmp_path / 'invalid', dataset_id='invalid',
                          pair_options=options, root=tmp_path)
    assert not (tmp_path / 'invalid').exists()


def test_override_keeps_cache_containment_and_cli_wiring(tmp_path, monkeypatch, capsys):
    from multiphy import benchmark_collection as collection
    with pytest.raises(ValueError, match='must stay under'):
        create_collection([], tmp_path.parent / 'outside_cache', dataset_id='outside',
                          pair_options={'min_repeats': 1}, root=tmp_path)
    captured = []
    monkeypatch.setattr(collection, 'create_collection', lambda *args, **kwargs: captured.append(kwargs) or 'view')
    collection.main(['--input', 'source', '--output', 'view', '--dataset-id', 'view', '--min-stimulus-repeats', '1'])
    assert captured[0]['pair_options']['min_repeats'] == 1
    assert capsys.readouterr().out.strip() == 'view'
