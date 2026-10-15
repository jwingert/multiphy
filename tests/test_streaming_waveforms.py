# [AGENT EDIT START | agent: codex | user: wingertj | reason: independently verify exact streaming waveform statistics, selection identity and integrity failures | date: 2026-09-18]
import gzip
import json
from pathlib import Path

import numpy as np
import pytest

from nems_lbhb.unit_match import extract_benchmark as extraction
from nems_lbhb.unit_match.features import average_spike_waveforms, temporal_split_unit_features
from nems_lbhb.unit_match.streaming_waveforms import (
    StreamingWaveformSummaries, WaveformSummaryPlan, plan_waveform_summaries,
)
from nems_lbhb.unit_match.waveform_preprocessing import SharedWaveformPreprocessor


def assert_summary_equal(actual, expected):
    assert actual.keys() == expected.keys()
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key], err_msg=key)


@pytest.mark.parametrize('compressed', [False, True])
@pytest.mark.parametrize('fs', [3000, 30000])
def test_filtered_streaming_is_exactly_the_existing_quantized_welford(tmp_path, compressed, fs):
    rng = np.random.default_rng(71)
    n = 3 * fs
    before, after = 9, 13
    voltage = rng.integers(-32000, 32000, size=(n, 5), dtype=np.int16)
    raw = tmp_path / ('raw.dat.gz' if compressed else 'raw.dat')
    raw.write_bytes(gzip.compress(voltage.tobytes()) if compressed else voltage.tobytes())
    base = np.unique(np.r_[np.arange(before, n - after, fs // 60),
        before, before + 1, fs - 1, fs, fs + 1, 2 * fs - 1, n - after])
    spikes = {'one': base, 'identical_selection': base.copy(), 'other': base[::2]}
    windows = [[0, n // 2], [n // 2, n]]
    options = dict(before=before, after=after, max_spikes=21, seed=9, min_spikes=2)
    plan = plan_waveform_summaries(spikes, windows, **options)
    np.testing.assert_array_equal(plan.samples, extraction.plan_snippets(spikes, windows, **options))
    assert len(plan.selections) == 12  # Identical units share, distinct full/half selections do not.
    boundary_selection = np.array([before, before + 1, fs - 1, fs, fs + 1, 2 * fs - 1, n - after])
    plan = WaveformSummaryPlan([*plan.selections, boundary_selection], before=before, after=after)
    kwargs = dict(n_channels=5, n_samples=n, sample_rate=fs, before=before, after=after,
        preprocessing=SharedWaveformPreprocessor(fs, 5, reference_channels=[0, 1, 2]).metadata(),
        voltage_scale=.195)
    normal = extraction.FilteredSnippetReader(raw, plan.samples, directory=tmp_path / 'normal', **kwargs)
    streamed = extraction.FilteredSnippetReader(raw, plan.samples, directory=tmp_path / 'streamed',
                                                 summary_plan=plan, **kwargs)
    try:
        assert streamed.data is None and not streamed.lookup
        assert not list((tmp_path / 'streamed').iterdir())
        assert normal.data.dtype == np.float32
        assert_summary_equal(
            average_spike_waveforms(boundary_selection, streamed, (0, n), 5, before=before, after=after),
            average_spike_waveforms(boundary_selection, normal, (0, n), 5, before=before, after=after))
        for unit, values in spikes.items():
            arguments = dict(sample_rate=fs, n_channels=5, source_unit=unit, source_session='day',
                su_type='SU', preprocessing='synthetic shared', **options)
            expected = temporal_split_unit_features(values, normal, windows, **arguments)
            actual = temporal_split_unit_features(values, streamed, windows, **arguments)
            for wanted, observed in zip(expected, actual):
                for key in ('waveform', 'waveform_halves', 'waveform_standard_error',
                            'waveform_spike_samples', 'half_waveform_spike_samples',
                            'n_waveform_spikes', 'n_eligible_waveform_spikes', 'isi', 'acg', 'spike_samples'):
                    np.testing.assert_array_equal(observed[key], wanted[key], err_msg=key)
            # SEMs of the two internal halves are not persisted in features,
            # but the generic summary hook must reproduce them exactly too.
            for index, (start, stop) in enumerate(windows):
                mid = (start + stop) // 2
                for offset, bounds in enumerate(((start, stop), (start, mid), (mid, stop))):
                    call = dict(before=before, after=after, max_spikes=21, seed=9 + index * 3 + offset)
                    assert_summary_equal(average_spike_waveforms(values, streamed, bounds, 5, **call),
                                         average_spike_waveforms(values, normal, bounds, 5, **call))
        with pytest.raises(ValueError, match='not individual snippets'):
            streamed(int(plan.samples[0] - before), int(plan.samples[0] + after))
        with pytest.raises(ValueError, match='selection differs'):
            streamed.waveform_summary([before, before + 2], before=before, after=after, n_channels=5)
        assert 'streaming' in streamed.processing_metadata['storage']
        left, right = normal.processing_metadata.copy(), streamed.processing_metadata.copy()
        left.pop('storage'); right.pop('storage')
        assert left == right
    finally:
        normal.close()
        streamed.close()


def test_exact_selection_key_distinguishes_same_endpoints_and_count_and_blocks_unverified_reads():
    a, b = [10, 20, 40], [10, 30, 40]
    plan = WaveformSummaryPlan([a, b, a], before=2, after=3)
    store = StreamingWaveformSummaries(plan, 2)
    assert len(plan.selections) == 2
    kwargs = dict(before=2, after=3, n_channels=2)
    with pytest.raises(RuntimeError, match='integrity'):
        store.waveform_summary(a, **kwargs)
    with pytest.raises(ValueError, match='increasing union'):
        store.add(20, np.ones((5, 2), dtype=np.float32))
    with pytest.raises(ValueError, match='float32'):
        store.add(10, np.ones((5, 2), dtype=np.float64))
    for center in plan.samples:
        store.add(center, np.full((5, 2), center ** 2 + .1234567, dtype=np.float32))
    with pytest.raises(RuntimeError, match='integrity'):
        store.waveform_summary(a, **kwargs)
    store.verify_complete()
    first = store.waveform_summary(a, **kwargs)
    second = store.waveform_summary(b, **kwargs)
    assert not np.array_equal(first['waveform'], second['waveform'])
    assert all(x is None for x in store.means + store.m2)
    first['waveform'][:] = 0
    assert store.waveform_summary(a, **kwargs)['waveform'].any()
    with pytest.raises(ValueError, match='dimensions'):
        store.waveform_summary(a, before=1, after=4, n_channels=2)
    with pytest.raises(ValueError, match='selection differs'):
        store.waveform_summary([10, 21, 40], **kwargs)
    store.close()
    with pytest.raises(RuntimeError, match='integrity'):
        store.waveform_summary(a, **kwargs)


@pytest.mark.parametrize('failure', ['truncated', 'crc', 'byte_count', 'nonfinite'])
def test_failed_voltage_stream_never_publishes_summaries_or_writes_snippets(tmp_path, monkeypatch, failure):
    raw = tmp_path / 'raw.dat.gz'
    voltage = np.ones((9000, 4), dtype='<f4')
    if failure == 'nonfinite':
        voltage[100, 1] = np.nan
    content = bytearray(gzip.compress(voltage.tobytes()))
    if failure == 'truncated':
        content = content[:-8]
    elif failure == 'crc':
        content[-8] ^= 1
    raw.write_bytes(content)
    plan = WaveformSummaryPlan([[100, 200]], before=9, after=13)
    stores = []
    original = StreamingWaveformSummaries.close
    def close(store):
        stores.append(store)
        original(store)
    monkeypatch.setattr(StreamingWaveformSummaries, 'close', close)
    with pytest.raises((ValueError, EOFError, OSError)):
        extraction.FilteredSnippetReader(raw, plan.samples, n_channels=4,
            n_samples=9001 if failure == 'byte_count' else 9000, sample_rate=3000,
            preprocessing=SharedWaveformPreprocessor(3000, 4).metadata(), dtype='<f4',
            directory=tmp_path / 'snippets', before=9, after=13, summary_plan=plan)
    assert stores and all(s.closed and not s.verified and not s.completed for s in stores)
    assert not list((tmp_path / 'snippets').iterdir())


def test_insufficient_internal_half_has_identical_exclusion_and_other_valid_summaries(tmp_path):
    raw = tmp_path / 'raw.dat'
    raw.write_bytes(np.zeros((3000, 4), dtype='<i2').tobytes())
    spikes = {'one': np.arange(50, 1200, 30)}
    options = dict(before=3, after=4, max_spikes=12)
    plan = plan_waveform_summaries(spikes, [[0, 3000]], **options)
    kwargs = dict(n_channels=4, n_samples=3000, sample_rate=3000,
        preprocessing=SharedWaveformPreprocessor(3000, 4).metadata(), before=3, after=4)
    normal = extraction.FilteredSnippetReader(raw, plan.samples, directory=tmp_path / 'normal', **kwargs)
    streamed = extraction.FilteredSnippetReader(raw, plan.samples, directory=tmp_path / 'stream', summary_plan=plan, **kwargs)
    try:
        arguments = dict(sample_rate=3000, n_channels=4, source_unit='one', source_session='day',
                         su_type='SU', preprocessing='synthetic', **options)
        a = temporal_split_unit_features(spikes['one'], normal, [[0, 3000]], **arguments)[0]
        b = temporal_split_unit_features(spikes['one'], streamed, [[0, 3000]], **arguments)[0]
        assert not a['eligible'] and not b['eligible']
        assert a['exclusion_reason'] == b['exclusion_reason']
        np.testing.assert_array_equal(a['waveform'], b['waveform'])
        assert 'waveform_halves' not in b
    finally:
        normal.close(); streamed.close()


def test_cli_opt_in_is_explicit_and_default_is_unchanged(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(extraction, 'extract_manifest', lambda *a, **k: calls.append((a, k)) or Path('/tmp/example'))
    extraction.main(['--manifest', '/tmp/input.json'])
    extraction.main(['--manifest', '/tmp/input.json', '--waveform-storage', 'streaming_summary'])
    assert [kwargs['waveform_storage'] for _, kwargs in calls] == ['snippets', 'streaming_summary']
    with pytest.raises(SystemExit):
        extraction.main(['--manifest', '/tmp/input.json', '--waveform-storage', 'unknown'])


def test_sparse_filter_blocks_and_exact_plan_dimension_refusal(tmp_path):
    fs, n = 3000, 15000
    voltage = np.random.default_rng(18).integers(-500, 500, size=(n, 4), dtype=np.int16)
    raw = tmp_path / 'sparse.dat.gz'
    raw.write_bytes(gzip.compress(voltage.tobytes()))
    selected = np.array([10, 11, 20, n - 21, n - 20])  # Three empty one-second blocks.
    plan = WaveformSummaryPlan([selected], before=9, after=13)
    kwargs = dict(n_channels=4, n_samples=n, sample_rate=fs, before=9, after=13,
                  preprocessing=SharedWaveformPreprocessor(fs, 4).metadata())
    normal = extraction.FilteredSnippetReader(raw, selected, directory=tmp_path / 'normal', **kwargs)
    streamed = extraction.FilteredSnippetReader(raw, selected, directory=tmp_path / 'stream', summary_plan=plan, **kwargs)
    try:
        assert_summary_equal(average_spike_waveforms(selected, streamed, (0, n), 4, before=9, after=13),
                             average_spike_waveforms(selected, normal, (0, n), 4, before=9, after=13))
        with pytest.raises(ValueError, match='differs'):
            extraction.FilteredSnippetReader(raw, selected[:-1], directory=tmp_path / 'bad', summary_plan=plan, **kwargs)
        assert not (tmp_path / 'bad').exists()
    finally:
        normal.close(); streamed.close()


def test_storage_validation_precedes_any_source_or_cache_access(tmp_path):
    manifest = {'recordings': [{'waveform_preprocessing': None}]}
    for storage in ('unrecognized', 'streaming_summary'):
        with pytest.raises(ValueError):
            with extraction.open_recording_sources(manifest, tmp_path / 'absent.json', options={},
                                                    root=tmp_path / 'cache', waveform_storage=storage):
                pytest.fail('Invalid mode or unsupported legacy preprocessing must not open sources')
    assert not (tmp_path / 'cache').exists()
# [AGENT EDIT END]
