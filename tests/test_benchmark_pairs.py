# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify label-free temporal benchmark pair construction | date: 2026-09-17]
import copy

import numpy as np
import pandas as pd
import pytest

from multiphy.benchmark_pairs import build_benchmark_pairs


def fixture_data():
    units = pd.DataFrame({'cellid': ['opaque1', 'opaque2', 'opaque3', 'opaque4'],
                          'session_id': ['early', 'early', 'late', 'late'],
                          'recording_group': ['animal:well:original'] * 4,
                          'su_type': ['SU'] * 4})
    geometry = np.column_stack((np.zeros(6), np.arange(6) * 20.))
    wave = np.array([0., 0., -1., -4., 2., 1., 0., 0.])[:, None] * np.array([.1, .2, 1., .5, .2, .1])[None, :]
    fingerprint = np.array([1., 3., 2., 7., 2., 1.])
    row = dict(eligible=True, waveform=wave, waveform_halves=np.stack((wave, wave), axis=-1),
               sample_rate=30000., spont=2., rate_session=8.,
               isi=np.linspace(0, 1, 8), acg=np.linspace(1, 0, 8),
               fingerprints={'sound:left': fingerprint, 'sound:right': fingerprint[::-1]},
               fingerprint_halves={'sound:left': np.stack((fingerprint, fingerprint), axis=-1),
                                   'sound:right': np.stack((fingerprint[::-1], fingerprint[::-1]), axis=-1)},
               trial_counts={'sound:left': 6, 'sound:right': 4},
               half_trial_counts={'sound:left': (3, 3), 'sound:right': (2, 2)},
               fingerprint_bin_seconds=.05, source_unit='not_to_be_read')
    features = {cid: copy.deepcopy(row) for cid in units.cellid}
    return units, features, {'early': geometry, 'late': geometry}


def test_builds_all_ordinary_pairs_without_reading_source_identity():
    units, features, geometry = fixture_data()
    pairs = build_benchmark_pairs(units, features, geometry)
    assert len(pairs) == 4
    assert set(pairs.cellid_a) == {'opaque1', 'opaque2'}
    assert set(pairs.cellid_b) == {'opaque3', 'opaque4'}
    assert not {'same_unit', 'source_unit', 'p_global', 'posterior'} & set(pairs.columns)
    for i, row in enumerate(features.values()):
        row['source_unit'] = f'arbitrary_changed_truth_{i}'
        row['source_session'] = f'another_unused_identity_{i}'
    pd.testing.assert_frame_equal(pairs, build_benchmark_pairs(units, features, geometry))
    assert pairs.fingerprint_r.to_numpy() == pytest.approx(np.ones(4))
    assert pairs.waveform_r.to_numpy() == pytest.approx(np.ones(4))
    assert pairs.d_log_spont.to_numpy() == pytest.approx(np.zeros(4))


def test_stimulus_condition_join_respects_channel_and_order():
    units, features, geometry = fixture_data()
    late = features['opaque3']
    late['fingerprints'] = dict(reversed(list(late['fingerprints'].items())))
    late['trial_counts']['sound:right'] = 1
    late['fingerprints']['different_sound:left'] = np.arange(6.)
    late['trial_counts']['different_sound:left'] = 8
    row = build_benchmark_pairs(units, features, geometry).iloc[0]
    assert row.shared_conditions == ('sound:left',)
    assert row.n_shared_bins == 6
    assert row.trial_counts_a == (6,)
    assert row.trial_counts_b == (6,)
    assert row.fingerprint_r == pytest.approx(1.)


def test_reliability_is_measured_not_invented_and_optional_gate_preserves_pair():
    units, features, geometry = fixture_data()
    del features['opaque3']['fingerprint_halves']
    original = build_benchmark_pairs(units, features, geometry)
    row = original.iloc[0]
    assert row.fingerprint_r == pytest.approx(1.)
    assert np.isnan(row.reliability_b)
    gated = build_benchmark_pairs(units, features, geometry, min_reliability=.3)
    assert len(gated) == 4
    assert np.isnan(gated.iloc[0].fingerprint_r)
    assert gated.iloc[0].fingerprint_status == 'missing_or_low_split_half_reliability'
    assert gated.iloc[0].isi_cosine == pytest.approx(1.)


def test_no_matching_stimuli_remains_a_physiology_candidate():
    units, features, geometry = fixture_data()
    features['opaque3']['fingerprints'] = {'unshared': np.arange(6.)}
    features['opaque3']['trial_counts'] = {'unshared': 5}
    row = build_benchmark_pairs(units, features, geometry).iloc[0]
    assert row.n_shared_conditions == 0
    assert np.isnan(row.fingerprint_r)
    assert np.isfinite(row.acg_cosine)


def test_separate_source_recording_groups_never_generate_pairs():
    units, features, geometry = fixture_data()
    units.loc[units.session_id == 'late', 'recording_group'] = 'other_source'
    pairs = build_benchmark_pairs(units, features, geometry)
    assert pairs.empty
    assert 'fingerprint_r' in pairs.columns


def test_non_single_units_filtered_and_missing_fresh_features_rejected():
    units, features, geometry = fixture_data()
    units.loc[units.cellid == 'opaque4', 'su_type'] = 'MU'
    del features['opaque4']
    assert len(build_benchmark_pairs(units, features, geometry)) == 2
    features['opaque3']['eligible'] = False
    with pytest.raises(ValueError, match='eligible independently'):
        build_benchmark_pairs(units, features, geometry)


@pytest.mark.parametrize('bad_field', ['shape', 'bin_duration', 'geometry', 'sample_rate'])
def test_incompatible_measurement_coordinates_fail_instead_of_truncating(bad_field):
    units, features, geometry = fixture_data()
    if bad_field == 'shape':
        features['opaque3']['fingerprints']['sound:left'] = np.arange(8.)
    elif bad_field == 'bin_duration':
        features['opaque3']['fingerprint_bin_seconds'] = .1
    elif bad_field == 'geometry':
        geometry['late'] = geometry['late'][::-1]
    else:
        features['opaque3']['sample_rate'] = 25000.
    with pytest.raises(ValueError):
        build_benchmark_pairs(units, features, geometry)


def test_geometry_location_changes_only_explicit_spatial_gate():
    units, features, geometry = fixture_data()
    features['opaque3']['waveform'] = np.roll(features['opaque3']['waveform'], 3, axis=1)
    exhaustive = build_benchmark_pairs(units, features, geometry)
    assert len(exhaustive) == 4
    assert exhaustive.iloc[0].distance_um > 10
    gated = build_benchmark_pairs(units, features, geometry, max_distance_um=10)
    assert len(gated) == 2
# [AGENT EDIT END]
