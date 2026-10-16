# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify external method input contracts and native output mappings without live data | date: 2026-09-17]
"""Optional packages are mocked only at their verified API boundary here."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: types.ModuleType for the fake DANT motion module (drift-view Task 6) | date: 2026-10-08]
import types
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tempfile.tempdir points run_dant's TemporaryDirectory under tmp_path (native-same-day Task 4) | date: 2026-10-11]
import tempfile
# [AGENT EDIT END]

import numpy as np
import pandas as pd
import pytest

from multiphy import external_backends as ext
from multiphy.backends import BackendUnavailable


def inputs():
    units = pd.DataFrame({'cellid': ['a', 'b', 'c', 'noise'],
                          'session_id': ['s1', 's2', 's2', 's1'],
                          'su_type': ['SU', 'SU', 'SU', 'MU']})
    rng = np.random.default_rng(5)
    bundle = ext.RawWaveformBundle(tuple(units.cellid), rng.normal(size=(4, 82, 4, 2)),
        tuple(units.session_id), {'s1': np.zeros((4, 2)), 's2': np.zeros((4, 2))},
        tuple(np.array([.01, .02, .04]) for _ in range(4)),
        {'raw_source': 'synthetic-test', 'split_method': 'independent temporal groups'},
        waveform_spike_counts=np.ones((4, 2)) * 20)
    return units, bundle


def test_absent_raw_halves_never_fall_back_to_templates():
    units, bundle = inputs()
    with pytest.raises(BackendUnavailable, match='RawWaveformBundle'):
        ext.run_unitmatch(units)
    with pytest.raises(ValueError, match='shape'):
        ext.run_unitmatch(units, waveform_bundle=replace(bundle, waveforms=bundle.waveforms[..., 0]))
    with pytest.raises(ValueError, match='provenance'):
        ext.run_unitmatch(units, waveform_bundle=replace(bundle, provenance={}))


def test_bundle_filters_single_units_and_checks_session_mapping():
    units, bundle = inputs()
    eligible, excluded, order, waves, _, sid, switch = ext._prepare_bundle(units, bundle)
    assert list(eligible.cellid) == ['a', 'b', 'c']
    assert excluded == ('noise',)
    assert waves.shape == (3, 82, 4, 2)
    assert list(sid) == [0, 1, 1]
    assert list(switch) == [0, 1, 3]
    with pytest.raises(ValueError, match='session disagrees'):
        ext._prepare_bundle(units, replace(bundle, session_ids=('s2', 's2', 's2', 's1')))


# [AGENT EDIT START | agent: codex | user: wingertj | reason: protect native adjacent-session semantics for real recording chronology | date: 2026-09-18]
def test_real_dates_order_native_sessions_without_relabeling_units_or_waveforms():
    units, bundle = inputs()
    units['session_date'] = ['2026-01-02', '2026-01-01', '2026-01-01', '2026-01-02']
    eligible, _, order, waves, _, sid, switch = ext._prepare_bundle(units, bundle)
    assert list(eligible.cellid) == ['b', 'c', 'a']
    assert list(eligible.session_id) == ['s2', 's2', 's1']
    assert order.tolist() == [1, 2, 0]
    np.testing.assert_array_equal(waves, bundle.waveforms[order])
    assert sid.tolist() == [0, 0, 1] and switch.tolist() == [0, 2, 3]
    units.loc[2, 'session_date'] = '2026-01-03'
    with pytest.raises(ValueError, match='one explicit recording date'):
        ext._prepare_bundle(units, bundle)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _matrix_pairs returns every unit pair (native-same-day Task 3) | date: 2026-10-11]
def test_directional_probabilities_not_averaged_into_a_match():
    units, bundle = inputs()
    eligible = ext._prepare_bundle(units, bundle)[0]
    matrix = np.array([[1, .99, .2], [.15, 1, .1], [.3, .1, 1]])
    pairs = ext._matrix_pairs(eligible, matrix)
    assert pairs.match_score.tolist() == [.15, .2, .1]
    assert pairs.official_probability_ab.tolist() == [.99, .2, .1]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-day native group keeps one unit per day and proposes the rest (native-same-day Task 3) | date: 2026-10-11]
def _um_same_day(monkeypatch, prob, tracking, n_spikes=(100., 50., 80.)):
    """``_um_result`` on units a (day 0), b and c (day 1), the native tracker grouping all three."""
    from multiphy.backends import unitmatch as um_adapter
    monkeypatch.setattr(um_adapter, '_module', {'UnitMatchPy.assign_unique_id': SimpleNamespace(
        assign_unique_id=lambda p, prm, i: [np.array([0, 0, 0])] * 4)}.__getitem__)
    monkeypatch.setattr(um_adapter, '_source_info', lambda module: {'synthetic_api_mock': True})
    units, bundle = inputs()
    eligible = ext._prepare_bundle(units, bundle)[0]
    eligible = eligible.assign(n_spikes=list(n_spikes))
    return ext._um_result('unitmatch', units, eligible, np.array(prob), np.array([0, 1, 1]),
                          {}, {}, tracking, {}, {})


def test_unitmatch_same_day_group_keeps_best_unit(monkeypatch):
    result = _um_same_day(monkeypatch, [[1, .97, .9], [.98, 1, .6], [.95, .6, 1]], 'native_intermediate')
    assert result.tracks == (('a', 'b'),)
    assert result.unmatched == ('c',)
    assert len(result.pairs) == 3
    assert result.pairs.decision.tolist() == ['native_track', 'proposed_link', 'proposed_link']
    assert result.pairs.selected.tolist() == [True, False, False]
    assert result.pairs.match_score.tolist() == [.97, .9, .6]


def test_unitmatch_tie_keeps_more_spikes(monkeypatch):
    result = _um_same_day(monkeypatch, [[1, .97, .97], [.97, 1, .6], [.97, .6, 1]], 'native_intermediate')
    assert result.tracks == (('a', 'c'),)
    assert result.unmatched == ('b',)


def test_unitmatch_common_tracking_marks_same_day_pairs(monkeypatch):
    result = _um_same_day(monkeypatch, [[1, .97, .9], [.98, 1, .6], [.95, .6, 1]], 'common_constrained')
    assert len(result.pairs) == 3
    row = result.pairs[(result.pairs.cellid_a == 'b') & (result.pairs.cellid_b == 'c')]
    assert row.decision.tolist() == ['same_session']
    assert result.tracks == (('a', 'b'),)
# [AGENT EDIT END]


def test_unitmatch_calls_official_scoring_and_native_tracker(monkeypatch):
    units, bundle = inputs()
    calls = []
    def extract(waves, geometry, info, param):
        calls.append('extract')
        assert waves.shape[0] == 3
        return {}
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fake UnitMatchPy.metric_functions; scoring computes one session drift and applies no pair (drift-view Task 5 fix round 2); uses the module-level types import (drift-view post-run fix) | date: 2026-10-08]
    mf = types.ModuleType('UnitMatchPy.metric_functions')
    mf.drift_n_sessions = lambda *a, **kw: ([[0.0, 0.0, 0.0]],)
    mf.apply_drift_correction_basic = lambda *a, **kw: None
    mf.apply_drift_correction_per_shank = lambda *a, **kw: None
    mf.shank_ID_per_session = lambda *a, **kw: None
    def metrics(extracted, switch, within, param, niter):
        calls.append('metrics')
        assert niter == 2
        mf.drift_n_sessions()
        param['n_expected_matches'] = 3
        return None, np.eye(3, dtype=bool), {}, None
    # [AGENT EDIT END]
    probability = np.array([[1, .97, .1], [.98, 1, .1], [.1, .1, 1]])
    def native(prob, param, info):
        calls.append('native')
        assert param['match_threshold'] == .5
        return [np.array([0, 0, 2])] * 4
    modules = {
        'UnitMatchPy.overlord': SimpleNamespace(extract_parameters=extract, extract_metric_scores=metrics),
        'UnitMatchPy.bayes_functions': SimpleNamespace(get_parameter_kernels=lambda *a, **kw: None,
            apply_naive_bayes=lambda *a, **kw: np.stack([1-probability.ravel(), probability.ravel()], axis=1)),
        'UnitMatchPy.assign_unique_id': SimpleNamespace(assign_unique_id=native),
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the adapter records drift on metric_functions (drift-view Task 5 fix round 2) | date: 2026-10-08]
        'UnitMatchPy.metric_functions': mf,
        # [AGENT EDIT END]
    }
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: patch the defining adapter module; format-2 provenance and score block (Task 9) | date: 2026-09-22]
    from multiphy.backends import unitmatch as um_adapter
    monkeypatch.setattr(um_adapter, '_module', modules.__getitem__)
    monkeypatch.setattr(um_adapter, '_source_info', lambda module: {'synthetic_api_mock': True})
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the stub param carries the shank keys the drift capture reads (drift-view Task 5 fix round 2) | date: 2026-10-08]
    monkeypatch.setattr(um_adapter, '_um_setup', lambda w, g, sid, sw, p:
        ({'n_units': len(w), 'no_shanks': 1, 'shank_dist': 0.0}, {'session_id': sid},
         (sid[:, None] != sid).astype(int)))
    # [AGENT EDIT END]
    result = ext.run_unitmatch(units, waveform_bundle=bundle, threshold=.9)
    assert calls == ['extract', 'metrics', 'native']
    assert result.tracks == (('a', 'b'),)
    assert result.unmatched == ('c',)
    assert result.excluded_units == ('noise',)
    assert result.provenance.get('assignment') == 'native_intermediate'
    assert result.provenance.get('threshold') == .5
    assert result.score.kind.value == 'uncalibrated_score'
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a fake pyDANT.MotionEstimation whose initial motion run_dant records (drift-view Task 6) | date: 2026-10-08]
def _fake_dant_motion_module():
    """A fake ``pyDANT.MotionEstimation`` for two sessions; its initial motion is all zeros."""
    module = types.ModuleType('pyDANT.MotionEstimation')
    motion = SimpleNamespace(LinearScale=np.array(0.001), Linear=np.zeros(2), Constant=np.zeros(2))
    module.initializeMotion = lambda settings, waveforms: (waveforms, motion)
    module.computeMotion = lambda settings: motion
    module.computeWaveformFeatures = lambda settings, waveforms, motion: None
    return module


def _fake_dant_lookup(run, motion_module):
    """``_module`` for run_dant: the runner for ``'pyDANT.Runner'``, the fake motion module otherwise."""
    return lambda name: SimpleNamespace(runDANT=run) if name == 'pyDANT.Runner' else motion_module
# [AGENT EDIT END]


def test_dant_exports_milliseconds_and_aligned_peth_then_preserves_native_tracks(monkeypatch):
    units, bundle = inputs()
    bundle = replace(bundle, peth=np.arange(12).reshape(4, 3),
                     peth_bin_keys=('soundA@0ms', 'soundA@50ms', 'soundB@0ms'))
    settings = {'clustering': {'features': ['Waveform', 'PETH']},
                'motionEstimation': {'features': [['AutoCorr']]}}
    captured = []
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fake runner initialises motion on the fake motion module (drift-view Task 6) | date: 2026-10-08]
    motion_module = _fake_dant_motion_module()
    # [AGENT EDIT END]
    def run(config):
        root = Path(config['path_to_data'])
        captured.append(root)
        np.testing.assert_equal(np.load(root / 'spike_times/Unit0.npy'), [10, 20, 40])
        np.testing.assert_equal(np.load(root / 'peth.npy'), bundle.peth[:3])
        assert np.load(root / 'waveform_all.npy').shape == (3, 4, 82)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fake runner initialises motion on the fake motion module (drift-view Task 6) | date: 2026-10-08]
        motion_module.initializeMotion(config, None)
        # [AGENT EDIT END]
        output = Path(config['output_folder'])
        output.mkdir()
        np.savez(output / 'Output.npz', {'IdxCluster': np.array([1, 1, -1]),
                                        'SimilarityMatrix': np.ones((3, 3)) * 4})
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: patch the defining adapter module (Task 9) | date: 2026-09-22]
    from multiphy.backends import dant as dant_adapter
    monkeypatch.setattr(dant_adapter, '_source_info', lambda _: {'synthetic_api_mock': True})
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _module also returns the fake motion module (drift-view Task 6) | date: 2026-10-08]
    monkeypatch.setattr(dant_adapter, '_module', _fake_dant_lookup(run, motion_module))
    # [AGENT EDIT END]
    result = ext.run_dant(units, waveform_bundle=bundle, settings=settings)
    assert result.tracks == (('a', 'b'),)
    assert result.unmatched == ('c',)
    assert result.pairs.match_score.isna().all()
    assert result.pairs.dant_similarity.eq(4).all()
    assert not captured[0].exists()
    assert 'output_folder' not in settings


def test_dant_missing_stimulus_data_does_not_fabricate_peth():
    units, bundle = inputs()
    with pytest.raises(ValueError, match='PETH requires'):
        ext.run_dant(units, waveform_bundle=bundle,
            settings={'clustering': {'features': ['PETH']}, 'motionEstimation': {'features': []}})


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-session DANT cluster keeps the best-similarity unit per day and proposes the rest (native-same-day Task 4) | date: 2026-10-11]
def test_dant_same_session_group_keeps_best_similarity(monkeypatch, tmp_path):
    monkeypatch.setattr(tempfile, 'tempdir', str(tmp_path))
    units, bundle = inputs()
    motion_module = _fake_dant_motion_module()
    def run(config):
        motion_module.initializeMotion(config, None)
        output = Path(config['output_folder'])
        output.mkdir()
        np.savez(output / 'Output.npz', {'IdxCluster': np.array([1, 1, 1]),
                                        'SimilarityMatrix': np.array([[0, 5, 3], [5, 0, 1], [3, 1, 0]])})
    from multiphy.backends import dant as dant_adapter
    monkeypatch.setattr(dant_adapter, '_module', _fake_dant_lookup(run, motion_module))
    monkeypatch.setattr(dant_adapter, '_source_info', lambda _: {'synthetic_api_mock': True})
    result = ext.run_dant(units, waveform_bundle=bundle,
        settings={'clustering': {'features': ['Waveform']}, 'motionEstimation': {'features': []}})
    assert result.tracks == (('a', 'b'),)
    assert result.unmatched == ('c',)
    assert len(result.pairs) == 3
    assert result.pairs.decision.tolist() == ['native_cluster', 'proposed_link', 'proposed_link']
    assert result.pairs.dant_same_cluster.tolist() == [True, True, True]
    row = result.pairs[(result.pairs.cellid_a == 'b') & (result.pairs.cellid_b == 'c')]
    assert row.dant_similarity.tolist() == [1.0]
# [AGENT EDIT END]


def test_deepunitmatch_requires_explicit_probe_compatible_checkpoint(tmp_path):
    units, bundle = inputs()
    with pytest.raises(BackendUnavailable, match='explicit trained checkpoint'):
        ext.run_deepunitmatch(units, waveform_bundle=bundle)
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.write_bytes(b'test: never loaded')
    with pytest.raises(ValueError, match='probe type'):
        ext.run_deepunitmatch(units, waveform_bundle=bundle,
                              checkpoint=checkpoint, checkpoint_probe='NP2')


def test_training_requires_heldout_groups_before_creating_files(tmp_path):
    units, bundle = inputs()
    with pytest.raises(ValueError, match='disjoint'):
        ext.prepare_deepunitmatch_training(units, bundle, source_checkout=tmp_path,
            output_dir=tmp_path / 'out', training_groups=['s1'], evaluation_groups=['s1'])
    assert not (tmp_path / 'out').exists()
# [AGENT EDIT END]
