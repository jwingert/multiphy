import gzip
import json
import numpy as np
import pytest

from multiphy.extract_benchmark import SnippetReader, plan_snippets, extract_manifest
from multiphy.benchmark import load_dataset


@pytest.mark.parametrize('compressed', [False, True])
def test_forward_reader_retains_overlaps_gaps_and_last_sample(tmp_path, compressed):
    voltage = np.arange(1000, dtype='<i2').reshape(250, 4)
    raw = tmp_path / ('voltage.dat.gz' if compressed else 'voltage.dat')
    raw.write_bytes(gzip.compress(voltage.tobytes()) if compressed else voltage.tobytes())
    centers = np.array([2, 3, 4, 55, 56, 247])
    reader = SnippetReader(raw, centers, n_channels=4, n_samples=250, directory=tmp_path / 'snips', before=2, after=3)
    try:
        for center in centers:
            np.testing.assert_array_equal(reader(center-2, center+3), voltage[center-2:center+3])
        with pytest.raises(ValueError, match='outside'):
            reader(23, 28)
    finally:
        reader.close()


def test_reader_preprocessing_is_explicit_and_keeps_raw_cache(tmp_path):
    voltage = np.arange(400, dtype='<i2').reshape(100, 4)
    raw = tmp_path / 'voltage.dat'
    raw.write_bytes(voltage.tobytes())
    reader = SnippetReader(raw, [20, 30], n_channels=4, n_samples=100,
        directory=tmp_path / 'snips', before=2, after=3, voltage_scale=.5,
        common_median=True, baseline_samples=2)
    try:
        np.testing.assert_array_equal(reader.data[0], voltage[18:23])
        np.testing.assert_allclose(reader(18, 23), 0)
    finally:
        reader.close()


def test_compressed_metadata_cannot_silently_reinterpret_voltage_shape(tmp_path):
    raw = tmp_path / 'voltage.dat.gz'
    raw.write_bytes(gzip.compress(np.arange(1000, dtype='<i2').tobytes()))
    with pytest.raises(ValueError, match='Compressed raw byte count'):
        SnippetReader(raw, [20, 30], n_channels=2, n_samples=250,
                      directory=tmp_path / 'snips', before=2, after=3)


def test_interrupted_construction_closes_memmap_before_cache_cleanup(tmp_path, monkeypatch):
    closed = []
    original = SnippetReader.close
    def close(reader):
        closed.append(reader.data)
        original(reader)
    def interrupted(*args):
        raise KeyboardInterrupt
    monkeypatch.setattr(SnippetReader, 'close', close)
    monkeypatch.setattr(SnippetReader, '_prefetch', interrupted)
    with pytest.raises(KeyboardInterrupt):
        SnippetReader(tmp_path / 'not-opened', [20, 30], n_channels=2, n_samples=250,
                      directory=tmp_path / 'snips', before=2, after=3)
    assert len(closed) == 1 and (closed[0] is None or closed[0]._mmap.closed)


@pytest.mark.parametrize('separate_scratch', [False, True])
def test_manifest_end_to_end_and_clock_refusal(tmp_path, monkeypatch, separate_scratch):
    voltage = np.zeros((1000, 4), dtype='<i2')
    spikes = {'one': np.arange(10, 980, 10), 'two': np.arange(15, 980, 10)}
    voltage[spikes['one'], 0] = -5
    voltage[spikes['two'], 2] = -7
    (tmp_path / 'raw.dat.gz').write_bytes(gzip.compress(voltage.tobytes()))
    np.savez(tmp_path / 'spikes.npz', **spikes)
    np.save(tmp_path / 'geometry.npy', np.column_stack((np.zeros(4), np.arange(4)*20)))
    source = dict(source_session='original1', well='animal:1', partition='test', sample_rate=1000,
        n_samples=1000, raw_voltage='raw.dat.gz', spikes_file='spikes.npz', geometry_file='geometry.npy',
        single_unit_ids=['one', 'two'], windows=[[0, 500], [500, 1000]],
        preprocessing='synthetic source voltage; no digital filter',
        clock_verification={'sample_index_origin': 'zero_based_voltage_row', 'method': 'synthetic known sample positions'})
    config = dict(schema_version=1, dataset_id='small', recordings=[source],
                  extraction_options={'before': 2, 'after': 3, 'max_spikes': 10})
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(config))
    scratch = tmp_path / 'scratch' if separate_scratch else None
    observed = []
    original = SnippetReader._prefetch
    def inspect_location(reader, *args):
        observed.append(reader.cache_path)
        return original(reader, *args)
    monkeypatch.setattr(SnippetReader, '_prefetch', inspect_location)
    saved = extract_manifest(path, root=tmp_path / 'cache', scratch_root=scratch)
    ds = load_dataset(saved)
    assert len(ds.units) == 4
    for row in ds.features.values():
        assert row['waveform'][2].min() in (-5, -7)
    runtime = scratch or tmp_path / 'cache' / '_runtime'
    assert observed and all(runtime in p.parents for p in observed)
    assert not list(runtime.iterdir())
    assert tmp_path / 'cache' in saved.parents
    if separate_scratch:
        assert not (tmp_path / 'cache' / '_runtime').exists()
    source['clock_verification'] = {}
    config['dataset_id'] = 'unverified'
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='verified zero-based'):
        extract_manifest(path, root=tmp_path / 'cache', scratch_root=scratch)
    assert not list(runtime.iterdir())


def test_scratch_location_rejects_relative_repository_and_symlink_escape(tmp_path):
    from pathlib import Path
    from multiphy.run.config import extraction_scratch_root
    with pytest.raises(ValueError, match='absolute'):
        extraction_scratch_root('local-scratch')
    repository = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='code repository'):
        extraction_scratch_root(repository / 'generated-snippets')
    alias = tmp_path / 'repo-alias'
    alias.symlink_to(repository, target_is_directory=True)
    with pytest.raises(ValueError, match='code repository'):
        extraction_scratch_root(alias / 'generated-snippets')


def test_reader_failure_cleans_only_owned_scratch_child(tmp_path):
    from multiphy.extract_benchmark import open_recording_sources
    scratch = tmp_path / 'scratch'
    scratch.mkdir()
    sentinel = scratch / 'other-job.dat'
    sentinel.write_text('retain another job')
    np.savez(tmp_path / 'spikes.npz', one=np.arange(10, 490, 10))
    np.save(tmp_path / 'geometry.npy', np.column_stack((np.zeros(2), np.arange(2)*20)))
    (tmp_path / 'raw.dat.gz').write_bytes(gzip.compress(np.zeros((500, 2), dtype='<i2').tobytes())[:-8])
    source = dict(source_session='original1', well='animal:1', partition='test', sample_rate=1000,
        n_samples=500, raw_voltage='raw.dat.gz', spikes_file='spikes.npz', geometry_file='geometry.npy',
        single_unit_ids=['one'], windows=[[0, 500]], preprocessing='synthetic',
        clock_verification={'sample_index_origin': 'zero_based_voltage_row', 'method': 'known synthetic positions'})
    with pytest.raises(EOFError):
        with open_recording_sources({'recordings': [source]}, tmp_path / 'manifest.json',
                                    options={'before': 2, 'after': 3}, root=tmp_path / 'cache', scratch_root=scratch):
            pytest.fail('Truncated gzip must not be accepted')
    assert list(scratch.iterdir()) == [sentinel]
    assert sentinel.read_text() == 'retain another job'
