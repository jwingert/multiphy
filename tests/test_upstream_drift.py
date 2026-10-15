# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tests that the upstream backends save the drift they applied, read through the drift frame (drift-view Task 5) | date: 2026-10-08]
"""Upstream backends save the drift their own code applied; the drift frame reads it back."""
import hashlib
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match.backends._upstream import Call, recording
from nems_lbhb.unit_match.backends.unitmatch import _unitmatch_registration
from nems_lbhb.unit_match.contract.inputs import RawWaveformBundle
from nems_lbhb.unit_match.run.runner import cache_runtime
from nems_lbhb.unit_match.viewer.drift_frame import drift_frame, unit_depth

from .synthetic_drift import make_drift_days

SESSIONS = ['day0', 'day1', 'day2']


def _bundle(units, features, geometry, waveforms):
    """A RawWaveformBundle of ``units`` per the plan's synthetic drift bundle recipe."""
    ids = tuple(units.cellid)
    waves = np.pad(np.stack(waveforms), ((0, 0), (21, 20), (0, 0), (0, 0)))
    spikes = tuple(features[c]['spike_samples'] / features[c]['sample_rate'] for c in ids)
    counts = np.array([[max(len(s) // 2, 1), max(len(s) - len(s) // 2, 1)] for s in spikes], float)
    return RawWaveformBundle(ids, waves, tuple(units.session_id), dict(geometry), spikes,
                             {'raw_source': 'synthetic', 'split_method': 'halves'},
                             waveform_spike_counts=counts)


def _drift_bundle():
    days = make_drift_days(n_days=3, n_neurons=30, offsets_um=(0.0, 40.0, 10.0), seed=0)
    ids = tuple(days.units.cellid)
    bundle = _bundle(days.units, days.features, days.geometry,
                     [days.features[c]['waveform_halves'] for c in ids])
    return days, bundle


def _two_shank_bundle():
    a = make_drift_days(n_days=3, n_neurons=45, offsets_um=(0.0, 40.0, 10.0), seed=0)
    b = make_drift_days(n_days=3, n_neurons=45, offsets_um=(0.0, 20.0, -10.0), seed=1)
    features, waveforms = {}, {}
    for prefix, days, first in (('A', a, True), ('B', b, False)):
        for cellid in days.units.cellid:
            halves = days.features[cellid]['waveform_halves']
            zeros = np.zeros_like(halves)
            features[prefix + cellid] = days.features[cellid]
            waveforms[prefix + cellid] = (np.concatenate([halves, zeros], axis=1) if first
                                          else np.concatenate([zeros, halves], axis=1))
    units = pd.concat([a.units.assign(cellid='A' + a.units.cellid),
                       b.units.assign(cellid='B' + b.units.cellid)], ignore_index=True)
    units = units.sort_values(['session_id', 'cellid'], ignore_index=True)
    geometry = {s: np.vstack([a.geometry[s], a.geometry[s] + [250.0, 0.0]]) for s in a.sessions}
    bundle = _bundle(units, features, geometry, [waveforms[c] for c in units.cellid])
    return units, bundle


def _prov(registration, group='P'):
    return {'group_runs': {group: {'run_id': 'r', 'provenance': {'group': group,
                                                                 'registration': registration}}}}


def _offsets(frame):
    return [c.offsets_um for c in frame.columns]


def _drift_call(rows):
    return Call('drift_n_sessions', (), {}, (np.array(rows, float), None, None), None)


def _basic_call(k, n_pairs):
    return Call('apply_drift_correction_basic', (np.zeros((n_pairs, 2), int), k), {}, None, 'drift_n_sessions')


def test_recording_restores_and_nests():
    module = types.ModuleType('fake_mf')

    def inner(x):
        return 2 * x

    def outer(x):
        return module.inner(x)

    module.inner, module.outer = inner, outer
    with recording(module, 'outer', 'inner') as calls:
        module.outer(3)
    assert calls == [Call('inner', (3,), {}, 6, 'outer'), Call('outer', (3,), {}, 6, None)]
    assert module.outer is outer and module.inner is inner
    with pytest.raises(RuntimeError):
        with recording(module, 'outer', 'inner'):
            raise RuntimeError('boom')
    assert module.outer is outer and module.inner is inner


def test_unitmatch_registration_through_drift_frame():
    calls = [_basic_call(0, 12), _basic_call(1, 9), _drift_call([[0, 1.0, -40.0], [0, 2.0, -10.0]])]
    reg = _unitmatch_registration(calls, SESSIONS, ['u0', 'v0', 'w0'], np.array([0, 1, 2, 3]),
                                  {'no_shanks': 1, 'shank_dist': 100.0}, 'P', [])
    assert reg['offsets_um'] == {'day0': 0.0, 'day1': -40.0, 'day2': -10.0}
    assert reg['shank_offsets_um'] == {}
    assert reg['diagnostics']['P']['drifts'] == [['day0', 'day1', -40.0, 1.0, 12],
                                                 ['day1', 'day2', -10.0, 2.0, 9]]
    frame = drift_frame(_prov(reg), 'P', SESSIONS, saves_offsets=True)
    assert unit_depth(frame, frame.columns[1], 'v0', 100.0, 30.0) == 60.0
    assert unit_depth(frame, frame.columns[2], 'w0', 100.0, 30.0) == 90.0
    assert frame.method == 'native'


def test_unitmatch_break_starts_group():
    calls = [_basic_call(1, 4), _drift_call([[0, 0.0, 0.0], [0, 1.5, 6.0]])]
    reg = _unitmatch_registration(calls, SESSIONS, ['u0', 'v0', 'w0'], np.array([0, 1, 2, 3]),
                                  {'no_shanks': 1, 'shank_dist': 100.0}, 'P', [])
    assert reg['diagnostics']['P']['groups'] == [['day0'], ['day1', 'day2']]
    frame = drift_frame(_prov(reg), 'P', SESSIONS, saves_offsets=True)
    assert frame.anchor == 'day1'
    assert _offsets(frame) == [(), (0.0,), (6.0,)]
    assert frame.columns[0].reason == 'no matches to day1'


def test_unitmatch_per_shank_registration():
    calls = [Call('shank_ID_per_session', (None, None, 1, None), {}, np.array([0.0, 1.0, 1.0]),
                  'apply_drift_correction_per_shank'),
             Call('apply_drift_correction_per_shank', (np.zeros((20, 2), int), 0), {},
                  (np.array([[0, 1.0, -40.0], [0, 2.0, -20.0]]), None, None), 'drift_n_sessions'),
             _basic_call(1, 9),
             _drift_call([[0, 1.5, -30.0], [0, 0.5, -10.0]])]
    reg = _unitmatch_registration(calls, SESSIONS, ['u0', 'u1', 'v0', 'v1', 'v2', 'w0'],
                                  np.array([0, 2, 5, 6]), {'no_shanks': 2, 'shank_dist': 225.0}, 'P', [])
    assert reg['offsets_um'] == {'day0': 0.0, 'day2': -10.0}
    assert reg['shank_offsets_um'] == {'day1': [-40.0, -20.0]}
    assert reg['unit_shanks'] == {'v0': 0, 'v1': 1, 'v2': 1}
    assert reg['shank_width_um'] == 225.0
    diag = reg['diagnostics']['P']
    assert diag['drifts'] == [['day0', 'day1', -30.0, 1.5, 20], ['day1', 'day2', -10.0, 0.5, 9]]
    assert diag['shank_drifts'] == [['day0', 'day1', 0, -40.0, 1.0], ['day0', 'day1', 1, -20.0, 2.0]]
    frame = drift_frame(_prov(reg), 'P', SESSIONS, saves_offsets=True)
    assert _offsets(frame) == [(0.0,), (-40.0, -20.0), (-10.0,)]


def _real_frame(result):
    return drift_frame({'group_runs': {'P': {'provenance': result.provenance.to_json()}}}, 'P', SESSIONS,
                       saves_offsets=True)


def test_real_unitmatch_offsets_match_truth(tmp_path):
    with cache_runtime(root=tmp_path):
        pytest.importorskip('UnitMatchPy.overlord')
        import UnitMatchPy.metric_functions as mf
        from nems_lbhb.unit_match.backends.unitmatch import run_unitmatch
        days, bundle = _drift_bundle()
        result = run_unitmatch(days.units, waveform_bundle=bundle, tracking='common_constrained',
                               run_id='real', provenance={'group': 'P'})
        expected_sha = hashlib.sha256(Path(mf.__file__).read_bytes()).hexdigest()
    reg = result.provenance.get('registration')
    assert reg['method'] == 'native'
    assert reg['shank_offsets_um'] == {}
    assert reg['diagnostics']['P']['sources'][0]['sha256'] == expected_sha
    frame = _real_frame(result)
    assert frame.method == 'native'
    truth = {'day1': -40.0, 'day2': -10.0}
    for column in frame.columns[1:]:
        shift = unit_depth(frame, column, 'x', 100.0, 30.0) - 100.0
        assert abs(shift - truth[column.session]) <= 2.0, (column.session, shift)


def test_real_unitmatch_per_shank(tmp_path):
    with cache_runtime(root=tmp_path):
        pytest.importorskip('UnitMatchPy.overlord')
        from nems_lbhb.unit_match.backends.unitmatch import run_unitmatch
        units, bundle = _two_shank_bundle()
        result = run_unitmatch(units, waveform_bundle=bundle, tracking='common_constrained',
                               run_id='real', provenance={'group': 'P'})
    reg = result.provenance.get('registration')
    for day, truth in (('day1', [-40.0, -20.0]), ('day2', [-10.0, 10.0])):
        assert np.allclose(reg['shank_offsets_um'][day], truth, atol=3.0), (day, reg['shank_offsets_um'][day])
    assert reg['offsets_um'] == {'day0': 0.0}
    later = units[units.session_id.isin(['day1', 'day2'])].cellid
    assert reg['unit_shanks'] == {c: (0 if c.startswith('A') else 1) for c in later}
    drifts = reg['diagnostics']['P']['drifts']
    for row, day in zip(drifts, ['day1', 'day2']):
        assert row[2] == pytest.approx(float(np.mean(reg['shank_offsets_um'][day])))
    frame = _real_frame(result)
    assert frame.anchor == 'day0'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tests that dant runs save the motion DANT kept, read through the drift frame (drift-view Task 6) | date: 2026-10-08]
import copy
import functools
from types import SimpleNamespace

from nems_lbhb.unit_match.backends._upstream import prepare_bundle
from nems_lbhb.unit_match.backends.dant import _dant_registration, default_dant_settings, run_native_dant
from nems_lbhb.unit_match.contract.errors import ContractViolation
from nems_lbhb.unit_match.viewer.drift_frame import electrode_depths

DAYS = ['D1', 'D2', 'D3']


def _motion(constant, linear=(0.0, 0.0, 0.0)):
    return SimpleNamespace(LinearScale=np.array(0.001), Linear=np.array(linear, float),
                           Constant=np.array(constant, float))


def _record(constant, linear=(0.0, 0.0, 0.0)):
    return {'linear_scale': 0.001, 'linear': [float(v) for v in linear],
            'constant': [float(v) for v in constant]}


def test_dant_kept_round_after_revert(tmp_path):
    config = {'path_to_data': str(tmp_path), 'output_folder': str(tmp_path)}
    similarity = np.zeros((3, 3))
    similarity[0, 1] = similarity[1, 0] = similarity[1, 2] = similarity[2, 1] = 0.9
    for name, data in {'session_index': [1, 2, 3], 'SimilarityPairs': [[0, 1], [1, 2]],
                       'SimilarityMatrix': similarity, 'SimilarityThreshold': 0.8}.items():
        np.save(tmp_path / (name + '.npy'), data)
    fitted = iter([_motion([-2.0, 2.0, 0.0]), _motion([-3.0, 1.0, 2.0])])
    module = types.ModuleType('fake.MotionEstimation')
    module.initializeMotion = lambda settings, waveforms: (waveforms, _motion([0.0, 0.0, 0.0]))
    module.computeMotion = lambda settings: next(fitted)
    module.computeWaveformFeatures = lambda settings, waveforms, motion: None
    round_1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    round_2 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])

    def run(settings):
        waveforms, motion = module.initializeMotion(settings, np.zeros(1))
        kept = []
        for clusters in (round_1, round_2):
            np.save(tmp_path / 'ClusterMatrix.npy', clusters)
            motion = module.computeMotion(settings)
            kept.append(motion)
            module.computeWaveformFeatures(settings, waveforms, motion)
        module.computeWaveformFeatures(settings, waveforms, kept[0])

    log = {}
    run_native_dant(SimpleNamespace(runDANT=run), module, config, 0, log)
    assert [r['pairs'] for r in log['rounds']] == [[(1, 2)], [(1, 2), (2, 3)]]
    assert log['final']['constant'] == [-2.0, 2.0, 0.0]
    reg = _dant_registration(log, DAYS, ['a', 'b', 'c'], np.array([0, 1, 2]), None, 'P', [])
    assert reg['diagnostics']['P']['groups'] == [['D1', 'D2'], ['D3']]
    frame = drift_frame(_prov(reg), 'P', DAYS, saves_offsets=True)
    assert frame.anchor == 'D1'
    assert unit_depth(frame, frame.columns[1], 'b', 100.0, 0.0) == 96.0
    assert frame.columns[2].reason == 'no matches to D1'


def test_dant_kept_initial_motion():
    log = {'initial': _record([0.0, 0.0, 0.0]),
           'rounds': [{'pairs': [(1, 2)], 'motion': _record([-2.0, 2.0, 0.0])}],
           'final': _record([0.0, 0.0, 0.0])}
    for final in (log['final'], None):
        reg = _dant_registration(dict(log, final=final), DAYS, ['a', 'b', 'c'], np.array([0, 1, 2]),
                                 None, 'P', [])
        assert reg['diagnostics']['P']['note'] == 'dant kept initial motion'
        assert reg['offsets_um'] == {}
    frame = drift_frame(_prov(reg), 'P', DAYS, saves_offsets=True)
    assert frame.no_offsets == 'dant kept initial motion'


def test_dant_equal_rounds_refuse():
    motion = _record([-2.0, 2.0, 0.0])
    log = {'initial': _record([0.0, 0.0, 0.0]),
           'rounds': [{'pairs': [(1, 2)], 'motion': motion}, {'pairs': [(1, 2), (2, 3)], 'motion': motion}],
           'final': motion}
    with pytest.raises(ContractViolation):
        _dant_registration(log, DAYS, ['a', 'b', 'c'], np.array([0, 1, 2]), None, 'P', [])


def test_dant_linear_unit_offsets():
    final = {'linear_scale': 0.001, 'linear': [0.0, 500.0, 0.0], 'constant': [1.0, 2.0, 3.0]}
    log = {'initial': _record([0.0, 0.0, 0.0]),
           'rounds': [{'pairs': [(1, 2), (2, 3)], 'motion': final}], 'final': final}
    reg = _dant_registration(log, DAYS, ['a', 'b', 'c'], np.array([0, 1, 2]),
                             np.array([[0, 100], [0, 200], [0, 300]], float), 'P', [])
    assert reg['offsets_um'] == {'D1': -1.0, 'D2': -2.0, 'D3': -3.0}
    assert reg['depth_scale'] == {'D1': 1.0, 'D2': 0.5, 'D3': 1.0}
    assert reg['unit_offsets_um'] == {'a': -1.0, 'b': -102.0, 'c': -3.0}
    frame = drift_frame(_prov(reg), 'P', DAYS, saves_offsets=True)
    assert unit_depth(frame, frame.columns[1], 'b', 200.0, 0.0) == 99.0
    np.testing.assert_array_equal(electrode_depths(frame, frame.columns[1], np.array([[0.0, 200.0]])), [99.0])


def _spy_dant(monkeypatch, linear_correction):
    """Real DANT on the drift bundle, with a spy keeping each applied motion and its call's locations."""
    import pyDANT.MotionEstimation as me
    from nems_lbhb.unit_match.backends.dant import run_dant
    original = me.computeWaveformFeatures
    seen = []

    @functools.wraps(original)
    def spy(user_settings, waveform_all, motion):
        locations = np.load(Path(user_settings['output_folder']) / 'locations.npy', allow_pickle=False)
        seen.append((copy.deepcopy(motion), locations))
        return original(user_settings, waveform_all, motion)

    monkeypatch.setattr(me, 'computeWaveformFeatures', spy)
    days, bundle = _drift_bundle()
    settings = default_dant_settings()
    settings['waveformCorrection']['linear_correction'] = linear_correction
    result = run_dant(days.units, waveform_bundle=bundle, settings=settings, run_id='real', seed=0,
                      provenance={'group': 'P'})
    return days, bundle, result, seen


def test_real_dant_offsets_are_kept_motion(tmp_path, monkeypatch):
    with cache_runtime(root=tmp_path):
        pytest.importorskip('pyDANT.MotionEstimation')
        days, bundle, result, seen = _spy_dant(monkeypatch, False)
    reg = result.provenance.get('registration')
    motion = seen[-1][0]
    np.testing.assert_array_equal([reg['offsets_um'][s] for s in SESSIONS], -np.asarray(motion.Constant))
    assert 'depth_scale' not in reg
    assert reg['diagnostics']['P']['groups'] == [SESSIONS]
    frame = _real_frame(result)
    for column in frame.columns[1:]:
        assert unit_depth(frame, column, 'x', 100.0, 30.0) - 100.0 < 0, column.session


def test_real_dant_linear_offsets(tmp_path, monkeypatch):
    with cache_runtime(root=tmp_path):
        pytest.importorskip('pyDANT.MotionEstimation')
        days, bundle, result, seen = _spy_dant(monkeypatch, True)
        eligible, _, _, _, _, sid, _ = prepare_bundle(days.units, bundle)
    reg = result.provenance.get('registration')
    motion, locations = seen[-1]
    scale = float(np.asarray(motion.LinearScale))
    linear = [float(v) for v in np.asarray(motion.Linear, float)]
    constant = [float(v) for v in np.asarray(motion.Constant, float)]
    assert reg['offsets_um'] == {s: -constant[k] for k, s in enumerate(SESSIONS)}
    assert reg['depth_scale'] == {s: 1 - scale * linear[k] for k, s in enumerate(SESSIONS)}
    assert reg['unit_offsets_um'] == {
        c: float(-(scale * linear[sid[i]] * locations[i, 1] + constant[sid[i]]))
        for i, c in enumerate(eligible.cellid)}
    assert reg['depth_scale']['day1'] != 1.0 and reg['depth_scale']['day2'] != 1.0
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tests that deepunitmatch runs save each day pair's drift, read through the drift frame (drift-view Task 7) | date: 2026-10-08]
import contextlib
import dataclasses

from nems_lbhb.unit_match.backends.deepunitmatch import _drift_correct_pair, _pair_registration


def _fake_pair_mf(correct):
    """A fake UnitMatchPy.metric_functions whose ``drift_correct_session_pair`` runs ``correct(module, ...)``."""
    module = types.ModuleType('fake.metric_functions')
    module.apply_drift_correction_basic = lambda *args: None
    module.apply_drift_correction_per_shank = lambda *args: None
    module.shank_ID_per_session = lambda *args: None

    def drift_correct_session_pair(labels, sub_switch, cent, trajectory, did, param):
        correct(module, sub_switch, cent, trajectory, param)
        return trajectory

    module.drift_correct_session_pair = drift_correct_session_pair
    return module


def test_pair_drift_captured_with_fake_mf():
    trajectory, cent, sub_switch = np.zeros((3, 5, 2, 2)), np.zeros((3, 5, 2)), np.array([0, 2, 5])
    module = _fake_pair_mf(lambda m, s, c, t, p: m.apply_drift_correction_basic(np.zeros((5, 2)), 0, s, c, t))
    module.apply_drift_correction_basic = lambda pairs, sid, s, c, t: (np.array([0.0, 1.5, -7.0]), t, c)
    out, drift = _drift_correct_pair(module, np.eye(5), sub_switch, cent, trajectory, {})
    assert out is trajectory
    assert drift == {'depth_um': -7.0, 'x_um': 1.5, 'n_matches': 5}
    out, drift = _drift_correct_pair(_fake_pair_mf(lambda m, s, c, t, p: None), np.eye(5), sub_switch, cent,
                                     trajectory, {})
    assert out is trajectory and drift is None
    reg = _pair_registration([{'session_a': 'D1', 'session_b': 'D2', 'depth_um': -7.0, 'x_um': 1.5,
                               'n_matches': 5}], {}, 100.0, 'P', [])
    frame = drift_frame(_prov(reg), 'P', ['D1', 'D2'], saves_offsets=True)
    assert _offsets(frame) == [(0.0,), (-7.0,)]
    assert frame.method == 'native'


def test_pair_drift_per_shank_with_fake_mf():
    trajectory, cent, sub_switch = np.zeros((3, 5, 2, 2)), np.zeros((3, 5, 2)), np.array([0, 2, 5])
    param = {'no_shanks': 2, 'shank_dist': 225.0}
    module = _fake_pair_mf(lambda m, s, c, t, p: m.apply_drift_correction_per_shank(np.zeros((30, 2)), 0, s, c, t, p))

    def per_shank(pairs, sid, s, c, t, p):
        module.shank_ID_per_session(c, s, 1, p)
        return np.array([[0, 1.0, -40.0], [0, 2.0, -20.0]]), t, c

    module.apply_drift_correction_per_shank = per_shank
    module.shank_ID_per_session = lambda c, s, sid, p: np.array([1.0, 0.0, 1.0])
    out, drift = _drift_correct_pair(module, np.eye(5), sub_switch, cent, trajectory, param)
    assert out is trajectory
    assert drift == {'shank_depth_um': [-40.0, -20.0], 'shank_x_um': [1.0, 2.0], 'n_matches': 30,
                     'shanks': [1, 0, 1]}


class _FakeTensor:
    """The fake ``clip_sim`` result: ``.detach().cpu().numpy()`` gives the array back."""
    def __init__(self, array):
        self.array = array

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.array


def _deep_drift_bundle():
    """The synthetic drift bundle grown to 384 channels: rows above the probe's top, waveforms zero there."""
    days, bundle = _drift_bundle()
    geometry = days.geometry['day0']
    xs, ys = np.unique(geometry[:, 0]), np.unique(geometry[:, 1])
    pitch, n_rows = ys[1] - ys[0], (384 - len(geometry)) // len(xs)
    extra = np.array([(x, ys[-1] + pitch * (k + 1)) for k in range(n_rows) for x in xs])
    grown = {s: np.vstack([g, extra]) for s, g in bundle.channel_positions.items()}
    waves = np.pad(bundle.waveforms, ((0, 0), (0, 0), (0, len(extra)), (0, 0)))
    return days, dataclasses.replace(bundle, waveforms=waves, channel_positions=grown,
                                     provenance=dict(bundle.provenance, probe_type='synthetic'))


def _fake_deep_modules(neuron_of_position):
    """Fake DeepUnitMatch modules: a match is ``1.0`` exactly for two units of one true neuron."""
    test = types.ModuleType('DeepUnitMatch.testing.test')
    test.__file__ = __file__

    def get_matches(frame, sub_sim, sub_sid, path, dist_thresh=None):
        cells, rows = np.asarray(frame['cells']), []
        for s1 in np.unique(sub_sid):
            for s2 in np.unique(sub_sid):
                for i in cells[sub_sid == s1]:
                    for j in cells[sub_sid == s2]:
                        rows.append({'RecSes1': s1, 'RecSes2': s2,
                                     'match': 1.0 if neuron_of_position[i] == neuron_of_position[j] else 0.0})
        return pd.DataFrame(rows)

    test.get_matches = get_matches
    prep = types.ModuleType('DeepUnitMatch.utils.param_fun')
    prep.get_snippets = lambda waves, geometry, sid, save_path, unit_ids, param: (
        np.zeros((len(waves), 4, 2)), None, np.asarray(unit_ids))
    helpers = types.ModuleType('DeepUnitMatch.utils.helpers')
    helpers.create_dataframe = lambda good, sub_sim, session_list: {'cells': np.concatenate(good)}
    losses = types.ModuleType('DeepUnitMatch.utils.losses')
    losses.clip_sim = lambda first, second: _FakeTensor(np.eye(len(first)))
    return {'testing.test': test, 'utils.param_fun': prep, 'utils.helpers': helpers, 'utils.losses': losses}


def test_real_deepunitmatch_pair_drifts(tmp_path, monkeypatch):
    with cache_runtime(root=tmp_path):
        pytest.importorskip('UnitMatchPy.metric_functions')
        from nems_lbhb.unit_match.backends import deepunitmatch
        days, bundle = _deep_drift_bundle()
        eligible = prepare_bundle(days.units, bundle)[0]
        neuron = dict(zip(days.truth.cellid, days.truth.neuron))
        torch = types.SimpleNamespace(float64='float64', no_grad=contextlib.nullcontext,
                                      as_tensor=lambda x, dtype=None, device=None: np.asarray(x, float))
        real_module = deepunitmatch._module
        fakes = _fake_deep_modules([neuron[c] for c in eligible.cellid])
        monkeypatch.setattr(deepunitmatch, '_module', lambda name: torch if name == 'torch' else real_module(name))
        monkeypatch.setattr(deepunitmatch, '_deep_module', fakes.__getitem__)
        monkeypatch.setattr(deepunitmatch, 'load_deepunitmatch_checkpoint',
                            lambda checkpoint, device='cpu': lambda x: x)
        monkeypatch.setattr(deepunitmatch, '_deep_preprocessing',
                            lambda waves, geometry, probe, parameters=None: dict(parameters or {}))
        monkeypatch.setattr(deepunitmatch, '_deep_training_metadata', lambda checkpoint, bundle, preprocessing: {})
        checkpoint = tmp_path / 'checkpoint'
        checkpoint.write_bytes(b'0')
        result = deepunitmatch.run_deepunitmatch(days.units, waveform_bundle=bundle, checkpoint=checkpoint,
                                                 checkpoint_probe='synthetic', tracking='common_constrained',
                                                 run_id='real', provenance={'group': 'P'})
    reg = result.provenance.get('registration')
    assert reg['method'] == 'native'
    on_day = {s: set(days.truth.neuron[days.truth.session_id == s]) for s in SESSIONS}
    truth = {('day0', 'day1'): -40.0, ('day0', 'day2'): -10.0, ('day1', 'day2'): 30.0}
    assert [(p['session_a'], p['session_b']) for p in reg['pair_drifts']] == list(truth)
    for pair in reg['pair_drifts']:
        key = (pair['session_a'], pair['session_b'])
        assert abs(pair['depth_um'] - truth[key]) <= 1.0, (key, pair['depth_um'])
        assert pair['n_matches'] == len(on_day[key[0]] & on_day[key[1]]), key
    frame = _real_frame(result)
    assert frame.anchor == 'day0'
    for column, shift in zip(frame.columns[1:], (-40.0, -10.0)):
        assert abs(unit_depth(frame, column, 'x', 100.0, 30.0) - 100.0 - shift) <= 1.0, column.session
# [AGENT EDIT END]
