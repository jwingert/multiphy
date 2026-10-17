import copy
import json
from pathlib import Path

import numpy as np
import pytest

from multiphy.benchmark import load_dataset, save_dataset
from multiphy.extract_benchmark import extract_manifest
from multiphy.features import temporal_split_unit_features
from multiphy.multiday_reuse import reuse_day_summary


def source(tmp_path, session='DAY001a', windows=None):
    voltage = np.random.default_rng(47).normal(0, .2, (1000, 4)).astype('<f4')
    spikes = {'one': np.arange(10, 980, 10), 'two': np.arange(15, 980, 10)}
    voltage[spikes['one'], 0] -= np.where(spikes['one'] < 400, 5., 9.)
    voltage[spikes['two'], 2] -= 7.
    voltage.tofile(tmp_path / f'{session}.dat')
    np.savez(tmp_path / f'{session}-spikes.npz', **spikes)
    geometry = np.column_stack((np.zeros(4), np.arange(4) * 20))
    np.save(tmp_path / f'{session}-geometry.npy', geometry)
    spec = dict(source_session=session, well='animal:1', partition='test', sample_rate=1000., n_samples=1000,
        raw_voltage=f'{session}.dat', spikes_file=f'{session}-spikes.npz', geometry_file=f'{session}-geometry.npy',
        single_unit_ids=['one', 'two'], windows=windows or [[0, 400], [400, 1000]], dtype='<f4',
        preprocessing='synthetic physical voltage',
        clock_verification={'sample_index_origin': 'zero_based_voltage_row', 'method': 'known synthetic clock'},
        stimuli=[dict(stimulus='sound:left:60', start=i, stop=i + 50) for i in range(0, 1000, 100)],
        baseline_windows=[[i+50, i+100] for i in range(0, 1000, 100)])
    options = dict(before=2, after=3, max_spikes=10, min_spikes=20, seed=0)
    path = tmp_path / f'{session}.json'
    path.write_text(json.dumps(dict(schema_version=1, dataset_id=session,
        recordings=[spec], extraction_options=options)))
    return path, spec, options, voltage, spikes


def reused(archive, manifest, spec, options):
    return reuse_day_summary(archive, manifest, spec, session_date='2026-01-01',
                             analysis_window=[0, 1000], extraction_options=options)


def test_pooled_raw_moments_independent_halves_and_complete_spike_features(tmp_path, monkeypatch):
    manifest, spec, options, voltage, spikes = source(tmp_path)
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    original = load_dataset(archive)
    real_open = Path.open
    raw = tmp_path / spec['raw_voltage']
    def guarded_open(path, *args, **kwargs):
        assert path.resolve() != raw.resolve(), 'Reuse must not open raw voltage'
        return real_open(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'open', guarded_open)
    summary = reused(archive, manifest, spec, options)
    for unit, all_spikes in spikes.items():
        f = summary['features']['DAY001a-' + unit]
        selections = f['half_waveform_spike_samples']
        assert not np.intersect1d(*selections).size
        snippets = np.stack([voltage[s-2:s+3] for s in np.concatenate(selections)]).astype(float)
        np.testing.assert_allclose(f['waveform'], snippets.mean(axis=0), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(f['waveform_standard_error'], snippets.std(axis=0, ddof=1) / np.sqrt(len(snippets)), rtol=1e-6, atol=1e-7)
        rows = [original.features[c] for c in original.labels[original.labels.source_unit_id == unit].cellid]
        rows.sort(key=lambda r: r['start_sample'])
        np.testing.assert_array_equal(f['waveform_halves'], np.stack([r['waveform'] for r in rows], axis=-1))
        direct = temporal_split_unit_features(all_spikes, lambda a, b: voltage[a:b], [[0, 1000]],
            sample_rate=1000., n_channels=4, source_unit=unit, source_session='DAY001a', su_type='SU',
            preprocessing=spec['preprocessing'], stimuli=spec['stimuli'], baseline_windows=spec['baseline_windows'], **options)[0]
        for key in ('spike_samples', 'isi', 'acg', 'rate_session', 'spont'):
            np.testing.assert_array_equal(f[key], direct[key])
        np.testing.assert_array_equal(f['fingerprints']['sound:left:60'], direct['fingerprints']['sound:left:60'])
        assert f['trial_counts'] == direct['trial_counts']
        assert f['half_trial_counts']['sound:left:60'] == (5, 5)
        np.testing.assert_array_equal(f['fingerprint_halves']['sound:left:60'], direct['fingerprint_halves']['sound:left:60'])
        assert f['waveform_half_windows'] == [[0, 400], [400, 1000]]
        assert f['n_waveform_spikes'] == 20 and f['summary_reuse']['fresh_full_mean_limit'] == 10
        assert not {'source_unit_id', 'source_unit', 'source_session', 'partition', 'reference_kind'} & set(f)
    assert summary['sources'][0]['provenance']['summary_reuse']['voltage_read'] is False




@pytest.mark.parametrize('windows', [[[0, 400], [450, 1000]], [[0, 400], [400, 900]]])
def test_gaps_or_partial_days_are_rejected(tmp_path, windows):
    manifest, spec, options, *_ = source(tmp_path, windows=windows)
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    with pytest.raises(ValueError, match='two contiguous parts'):
        reused(archive, manifest, spec, options)


def test_changed_source_manifest_or_inputs_are_rejected(tmp_path):
    manifest, spec, options, *_ = source(tmp_path)
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    old = manifest.read_bytes()
    manifest.write_bytes(old + b'\n')
    with pytest.raises(ValueError, match='manifest hash'):
        reused(archive, manifest, spec, options)
    manifest.write_bytes(old)
    np.save(tmp_path / spec['geometry_file'], np.ones((4, 2)))
    with pytest.raises(ValueError, match='geometry source changed'):
        reused(archive, manifest, spec, options)


@pytest.mark.parametrize('defect', ['excluded', 'reused_selection', 'missing_unit', 'changed_clock'])
def test_incomplete_or_inconsistent_archive_is_rejected(tmp_path, defect):
    manifest, spec, options, *_ = source(tmp_path)
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    ds = load_dataset(archive)
    if defect == 'excluded':
        ds.exclusions.append({'reason': 'prespecified omission'})
    elif defect == 'reused_selection':
        rows = list(ds.features.values())[:2]
        rows[1]['waveform_spike_samples'] = rows[0]['waveform_spike_samples'].copy()
    elif defect == 'missing_unit':
        ds = ds.subset(ds.labels[ds.labels.source_unit_id == 'one'].cellid)
    else:
        ds.metadata['sources'][0]['provenance']['clock_verification']['method'] = 'different clock'
    edited = save_dataset(ds, f'edited-{defect}', cache_root=tmp_path / 'cache')
    with pytest.raises(ValueError, match='complete source|selection|every declared|sample clock'):
        reused(edited, manifest, spec, options)


def test_crossing_stimulus_trial_is_rejected_instead_of_silently_dropped(tmp_path):
    manifest, spec, options, *_ = source(tmp_path)
    spec['stimuli'].append(dict(stimulus='boundary_trial', start=375, stop=425))
    content = json.loads(manifest.read_text())
    content['recordings'][0] = spec
    manifest.write_text(json.dumps(content))
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    with pytest.raises(ValueError, match='stimulus trial crosses'):
        reused(archive, manifest, spec, options)
@pytest.mark.parametrize('n_trials', [9, 10])
def test_reused_trial_halves_and_reliability_match_fresh_with_unequal_outer_counts(tmp_path, n_trials):
    from multiphy.benchmark_pairs import _reliability
    manifest, spec, options, voltage, spikes = source(tmp_path, windows=[[0, 300], [300, 1000]])
    options['bin_seconds'] = .01
    # Deliberately vary response shape over chronological trials, so an outer-
    # window split would yield different half means and measured reliability.
    spikes['one'] = spikes['one'][~((spikes['one'] // 100 >= 3) & (spikes['one'] % 100 == 30))]
    np.savez(tmp_path / spec['spikes_file'], **spikes)
    spec['stimuli'] = list(reversed(spec['stimuli'][:n_trials]))
    contents = json.loads(manifest.read_text())
    contents.update(recordings=[spec], extraction_options=options)
    manifest.write_text(json.dumps(contents))
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    summary = reused(archive, manifest, spec, options)
    f = summary['features']['DAY001a-one']
    direct = temporal_split_unit_features(spikes['one'], lambda a, b: voltage[a:b], [[0, 1000]],
        sample_rate=1000., n_channels=4, source_unit='one', source_session='DAY001a', su_type='SU',
        preprocessing=spec['preprocessing'], stimuli=spec['stimuli'], baseline_windows=spec['baseline_windows'], **options)[0]
    condition = 'sound:left:60'
    assert f['half_trial_counts'] == direct['half_trial_counts'] == {condition: (n_trials // 2, n_trials - n_trials // 2)}
    np.testing.assert_array_equal(f['fingerprints'][condition], direct['fingerprints'][condition])
    np.testing.assert_array_equal(f['fingerprint_halves'][condition], direct['fingerprint_halves'][condition])
    assert np.isfinite(_reliability(f, [condition]))
    assert _reliability(f, [condition]) == _reliability(direct, [condition])
    assert f['waveform_half_windows'] == [[0, 300], [300, 1000]]
    assert 'independent of waveform outer windows' in f['summary_reuse']['fingerprint_half_definition']
def test_pooled_moments_weight_unequal_retained_spike_counts():
    from multiphy.multiday_reuse import _pool_waveforms
    rng = np.random.default_rng(12)
    snippets = [rng.normal(-3, .5, (3, 5, 2)), rng.normal(8, 2, (7, 5, 2))]
    rows = [dict(n_waveform_spikes=len(x), waveform=x.mean(axis=0).astype(np.float32),
        waveform_standard_error=(x.std(axis=0, ddof=1) / np.sqrt(len(x))).astype(np.float32)) for x in snippets]
    mean, error = _pool_waveforms(rows)
    whole = np.concatenate(snippets)
    np.testing.assert_allclose(mean, whole.mean(axis=0), rtol=1e-6)
    np.testing.assert_allclose(error, whole.std(axis=0, ddof=1) / np.sqrt(len(whole)), rtol=1e-6)
