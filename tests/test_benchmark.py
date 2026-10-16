from dataclasses import replace
import numpy as np
import pytest

from multiphy.benchmark import SourceRecording, build_dataset, save_dataset, load_dataset


def source(well='animal:1', session='source1', partition='test'):
    voltage = np.zeros((1000, 4))
    spikes = {'one': np.arange(10, 980, 10), 'two': np.arange(15, 980, 10)}
    voltage[spikes['one'], 0] = -5
    voltage[spikes['two'], 2] = -7
    trials = [dict(stimulus='sound:left', start=x, stop=x+50) for x in range(0, 950, 100)]
    return SourceRecording(session, well, partition, 1000.,
        np.column_stack((np.zeros(4), np.arange(4)*20)), np.array([[0, 500], [500, 1000]]),
        spikes, ('one', 'two'), lambda a, b: voltage[a:b], 'synthetic sample-clock voltage',
        {'raw_source': 'synthetic array'}, stimuli=trials)


def dataset(**kwargs):
    return build_dataset([source()], dataset_id='test', extraction_options={'before': 2, 'after': 3}, **kwargs)


def test_dataset_preserves_references_separately_and_independent_waveforms():
    ds = dataset()
    assert len(ds.units) == 4 and len(ds.pairs) == 4
    assert 'same_unit' not in ds.pairs
    assert all('source_unit' not in f for f in ds.features.values())
    assert set(ds.labels.source_unit_id) == {'one', 'two'}
    assert ds.labels.groupby('source_unit_id').size().tolist() == [2, 2]
    bundle = ds.waveform_bundle()
    assert bundle.waveforms.shape == (4, 5, 4, 2)
    assert np.all(bundle.waveform_spike_counts > 2)
    assert bundle.peth.shape == (4, 1)
    assert set(bundle.provenance['source_groups'].values()) == {'animal:1'}
    assert all(np.min(t) >= 0 and np.max(t) < .5 for t in bundle.spike_times_seconds)


def test_roundtrip_cache_and_hash_validation(tmp_path):
    ds = dataset()
    path = save_dataset(ds, tmp_path / 'benchmark', cache_root=tmp_path)
    loaded = load_dataset(path)
    assert loaded.metadata == ds.metadata
    assert loaded.labels.equals(ds.labels)
    np.testing.assert_array_equal(loaded.waveform_bundle().waveforms, ds.waveform_bundle().waveforms)
    assert (path / 'METHODS.md').exists()
    with pytest.raises(FileExistsError):
        save_dataset(ds, path, cache_root=tmp_path)
    (path / 'pairs.csv').write_text('tampered')
    with pytest.raises(ValueError, match='hash mismatch'):
        load_dataset(path)


def test_outputs_cannot_escape_selected_cache_root(tmp_path):
    with pytest.raises(ValueError):
        save_dataset(dataset(), tmp_path.parent / 'elsewhere', cache_root=tmp_path)


def test_omissions_preserve_conditional_unmatched_reference():
    ds = dataset(omitted=[('source1', 'one', 1)])
    assert len(ds.units) == 3 and len(ds.pairs) == 2
    assert ds.labels.groupby('source_unit_id').size().to_dict() == {'one': 1, 'two': 2}
    assert ds.exclusions[0]['reason'] == 'prespecified omission'


def test_wells_cannot_cross_partitions():
    a = source(partition='train')
    b = source(session='source2', partition='test')
    with pytest.raises(ValueError, match='leakage'):
        build_dataset([a, b], dataset_id='bad', extraction_options={'before': 2, 'after': 3})


def test_source_recordings_are_never_cross_paired():
    ds = build_dataset([source(), source(well='animal:2', session='source2')], dataset_id='two',
                       extraction_options={'before': 2, 'after': 3})
    assert len(ds.pairs) == 8
    groups = ds.units.set_index('cellid').recording_group
    assert all(groups[a] == groups[b] for a, b in zip(ds.pairs.cellid_a, ds.pairs.cellid_b))
    subset = ds.subset(ds.units.cellid.iloc[:4])
    assert len(subset.units) == 4 and len(subset.pairs) == 4


def test_bundle_probe_type_tracks_selected_sources_not_unselected_holdouts():
    a, b = source(), source(well='animal:2', session='source2')
    a.provenance['probe_type'] = 'Neuropixels 1.0'
    b.provenance['probe_type'] = 'Neuropixels 2.0'
    ds = build_dataset([a, b], dataset_id='two', extraction_options={'before': 2, 'after': 3})
    assert 'probe_type' not in ds.waveform_bundle().provenance
    assert ds.subset(ds.units.cellid.iloc[:4]).waveform_bundle().provenance['probe_type'] == 'Neuropixels 1.0'


def test_reference_in_features_is_rejected():
    ds = dataset()
    ds.features[ds.units.cellid.iloc[0]]['source_unit'] = 'one'
    with pytest.raises(ValueError, match='Source identity'):
        ds.validate()


def test_recording_groups_cannot_pool_original_recordings():
    ds = build_dataset([source(), source(well='animal:2', session='source2')], dataset_id='two',
                       extraction_options={'before': 2, 'after': 3})
    ds.units['recording_group'] = 'pooled'
    with pytest.raises(ValueError, match='original recording identities'):
        ds.validate()


def test_ambiguous_omissions_and_source_identity_types_are_rejected():
    with pytest.raises(ValueError, match='unambiguous'):
        build_dataset([source(), source(well='animal:2')], dataset_id='two',
                      omitted=[('source1', 'one', 1)])
    s = source()
    s.single_unit_ids = (1, '1')
    s.spike_samples = {1: s.spike_samples['one'], '1': s.spike_samples['two']}
    with pytest.raises(ValueError, match='nonempty strings'):
        build_dataset([s], dataset_id='two')
