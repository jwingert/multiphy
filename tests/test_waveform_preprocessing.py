"""Scientific invariants and whole-recording references, using synthetic voltage."""
import gzip
import json

import numpy as np
import pytest
from scipy.signal import butter, sosfiltfilt

from multiphy.waveform_preprocessing import SharedWaveformPreprocessor
from multiphy.extract_benchmark import FilteredSnippetReader


def whole_trace_reference(voltage, sample_rate, *, scale=1., reference_channels=None):
    """Independent direct SciPy reference; never calls production preprocessing."""
    sos = butter(3, 150., btype='highpass', fs=sample_rate, output='sos')
    filtered = sosfiltfilt(sos, np.asarray(voltage, dtype=np.float64) * scale, axis=0)
    reference = filtered if reference_channels is None else filtered[:, reference_channels]
    return filtered - np.median(reference, axis=1, keepdims=True)


def synthetic_voltage(sample_rate=3000, n_seconds=3, n_channels=5):
    rng = np.random.default_rng(421)
    count = int(sample_rate * n_seconds)
    time = np.arange(count) / sample_rate
    common = 700 * np.sin(2 * np.pi * 19 * time) + 70 * np.sin(2 * np.pi * 530 * time)
    voltage = common[:, None] + rng.normal(0, 2, (count, n_channels))
    voltage += np.linspace(-200, 200, count)[:, None] * np.arange(1, n_channels + 1)
    for center in (100, sample_rate - 2, sample_rate + 2, 2 * sample_rate + 5, count - 100):
        width = 1.4
        waveform = -220 * np.exp(-((np.arange(count) - center) / width) ** 2)
        voltage[:, -1] += waveform
        voltage[:, -2] += .4 * waveform
    return np.rint(voltage).astype('<i2')


def test_transform_matches_independent_whole_trace_reference_and_preserves_input():
    voltage = synthetic_voltage()
    original = voltage.copy()
    processor = SharedWaveformPreprocessor(3000, 5)
    actual = processor.transform(voltage, voltage_scale=.195)
    expected = whole_trace_reference(voltage, 3000, scale=.195)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_array_equal(voltage, original)
    assert np.isfinite(actual).all()


def test_common_noise_is_removed_without_whitening_or_amplitude_normalization():
    fs = 3000
    time = np.arange(6000) / fs
    common = 400 * np.sin(2 * np.pi * 20 * time) + 40 * np.sin(2 * np.pi * 650 * time)
    pulse = -100 * np.exp(-((np.arange(len(time)) - 3000) / 2.) ** 2)
    voltage = np.tile(common[:, None], (1, 5))
    voltage[:, 3] += pulse
    voltage[:, 4] += 2 * pulse
    processor = SharedWaveformPreprocessor(fs, 5, reference_channels=[0, 1, 2])
    result = processor.transform(voltage)
    np.testing.assert_allclose(result[:, :3], 0, atol=1e-10)
    np.testing.assert_allclose(result[:, 4], 2 * result[:, 3], rtol=1e-8, atol=1e-10)
    assert abs(result[3000, 3]) > 40
    np.testing.assert_allclose(processor.transform(voltage, voltage_scale=.195),
                               .195 * result, rtol=1e-8, atol=1e-10)


def test_reference_channel_selection_does_not_subtract_active_channels():
    voltage = np.zeros((3000, 4))
    voltage[:, 2:] = (-100 * np.exp(-((np.arange(3000) - 1500) / 2.) ** 2))[:, None]
    selected = SharedWaveformPreprocessor(3000, 4, reference_channels=[0, 1]).transform(voltage)
    expected = whole_trace_reference(voltage, 3000, reference_channels=[0, 1])
    np.testing.assert_allclose(selected, expected, rtol=1e-7, atol=1e-9)
    np.testing.assert_allclose(selected[:, :2], 0, atol=1e-10)
    all_channels = SharedWaveformPreprocessor(3000, 4).transform(voltage)
    assert abs(selected[1500, 2]) > 1.9 * abs(all_channels[1500, 2])


@pytest.mark.parametrize('overrides', [
    {'sample_rate': 0}, {'sample_rate': np.nan}, {'n_channels': 0},
    {'highpass_hz': 0}, {'highpass_hz': 1500}, {'highpass_hz': np.nan},
    {'filter_order': 0}, {'padding_seconds': -1},
    {'reference_channels': []}, {'reference_channels': [0, 4]},
])
def test_invalid_filter_or_reference_configuration_is_rejected(overrides):
    arguments = dict(sample_rate=3000, n_channels=4)
    arguments.update(overrides)
    with pytest.raises((ValueError, TypeError)):
        SharedWaveformPreprocessor(**arguments)


def test_invalid_voltage_shape_and_nonfinite_values_are_rejected():
    processor = SharedWaveformPreprocessor(3000, 4)
    for voltage in (np.zeros(100), np.zeros((100, 3)), np.full((100, 4), np.nan)):
        with pytest.raises(ValueError):
            processor.transform(voltage)
    for scale in (0, -1, np.nan):
        with pytest.raises(ValueError):
            processor.transform(np.zeros((100, 4)), voltage_scale=scale)


def test_metadata_is_json_serializable_and_identifies_filter_and_reference():
    processor = SharedWaveformPreprocessor(3000, 4, reference_channels=[0, 2])
    metadata = json.loads(json.dumps(processor.metadata(), allow_nan=False))
    assert metadata['profile'] == 'shared_unwhitened_v1'
    assert metadata['highpass_hz'] == 150.
    assert metadata['filter_order'] == 3
    assert metadata['padding_seconds'] == .05
    assert metadata['reference_channels'] == [0, 2]


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('fs', [3000, 30000])
def test_streamed_filter_matches_whole_trace_across_chunk_and_raw_boundaries(tmp_path, compressed, fs):
    voltage = synthetic_voltage(fs)
    raw = tmp_path / ('voltage.dat.gz' if compressed else 'voltage.dat')
    raw.write_bytes(gzip.compress(voltage.tobytes()) if compressed else voltage.tobytes())
    before, after = 9, 13
    samples = np.array([before, before + 1, before + 2, fs - 2, fs - 1,
                        fs, fs + 1, fs + 2, 2 * fs - 1, len(voltage) - after])
    config = SharedWaveformPreprocessor(fs, 5, reference_channels=[0, 1, 2]).metadata()
    reader = FilteredSnippetReader(raw, samples, n_channels=5, n_samples=len(voltage),
        directory=tmp_path / 'snippets', sample_rate=fs, preprocessing=config,
        before=before, after=after, voltage_scale=.195)
    try:
        expected = whole_trace_reference(voltage, fs, scale=.195, reference_channels=[0, 1, 2])
        assert reader.data.dtype == np.float32
        assert reader.data.shape == (len(samples), before + after, 5)
        assert reader.processing_metadata['profile'] == 'shared_unwhitened_v1'
        for sample in samples:
            actual = reader(int(sample - before), int(sample + after))
            np.testing.assert_allclose(actual, expected[sample - before:sample + after],
                                       rtol=2e-4, atol=2e-3)
        with pytest.raises(ValueError):
            reader(22, 44)
    finally:
        reader.close()


def test_streamed_reader_rejects_unknown_preprocessing_profile(tmp_path):
    raw = tmp_path / 'voltage.dat'
    raw.write_bytes(np.zeros((1000, 4), dtype='<i2').tobytes())
    with pytest.raises(ValueError):
        FilteredSnippetReader(raw, [100, 200], n_channels=4, n_samples=1000,
            directory=tmp_path / 'snippets', sample_rate=3000,
            preprocessing={'profile': 'unknown_whitening_or_normalization'})


def test_shared_manifest_preserves_processing_provenance_and_filtered_waveforms(tmp_path):
    from multiphy.extract_benchmark import extract_manifest
    from multiphy.benchmark import load_dataset

    fs = 3000
    voltage = synthetic_voltage(fs)
    spikes = {'one': np.arange(100, len(voltage) - 100, 60)}
    voltage[spikes['one'], 4] -= 200
    (tmp_path / 'raw.dat.gz').write_bytes(gzip.compress(voltage.tobytes()))
    np.savez(tmp_path / 'spikes.npz', **spikes)
    np.save(tmp_path / 'geometry.npy', np.column_stack((np.zeros(5), np.arange(5) * 20)))
    processing = SharedWaveformPreprocessor(fs, 5, reference_channels=[0, 1, 2]).metadata()
    source = dict(source_session='original', well='animal:well1', partition='test',
        sample_rate=fs, n_samples=len(voltage), raw_voltage='raw.dat.gz',
        spikes_file='spikes.npz', geometry_file='geometry.npy', single_unit_ids=['one'],
        windows=[[0, 4500], [4500, 9000]], voltage_scale=.195,
        preprocessing='Synthetic acquired voltage; shared high-pass and median reference',
        waveform_preprocessing=processing,
        clock_verification={'sample_index_origin': 'zero_based_voltage_row',
                            'method': 'synthetic known voltage-row positions'})
    config = dict(schema_version=1, dataset_id='shared', recordings=[source],
                  extraction_options={'before': 9, 'after': 13, 'max_spikes': 20})
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps(config), encoding='utf-8')
    saved = extract_manifest(manifest, root=tmp_path / 'cache')
    dataset = load_dataset(saved)
    provenance = dataset.metadata['sources'][0]['provenance']
    stored = provenance['waveform_preprocessing']
    assert all(stored[key] == value for key, value in processing.items())
    assert stored['output_dtype'] == 'float32'
    assert stored['chunk_seconds'] == 1.
    assert provenance['voltage_scale'] == .195
    assert provenance['common_median'] is True
    assert provenance['baseline_samples'] == 0
    assert dataset.metadata['extraction_manifest_sha256']
    expected = whole_trace_reference(voltage, fs, scale=.195, reference_channels=[0, 1, 2])
    assert len(dataset.features) == 2
    for feature in dataset.features.values():
        centers = feature['waveform_spike_samples']
        mean = np.mean([expected[center - 9:center + 13] for center in centers], axis=0)
        np.testing.assert_allclose(feature['waveform'], mean, rtol=2e-4, atol=2e-3)
    assert 'shared_unwhitened_v1' in (saved / 'METHODS.md').read_text(encoding='utf-8')
    assert not list((tmp_path / 'cache' / '_runtime').iterdir())
