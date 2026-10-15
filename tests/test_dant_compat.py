# [AGENT EDIT START | agent: codex | user: wingertj | reason: verify empty native drift guard and declared DANT defaults | date: 2026-09-18]
from types import SimpleNamespace

import numpy as np
import pytest

from nems_lbhb.unit_match.dant_compat import run_native_dant
from nems_lbhb.unit_match.dant_settings import default_dant_settings


class Motion:
    def __init__(self, constant):
        self.Constant = np.asarray(constant)
        self.Linear = np.zeros(2)

    def save(self, folder):
        np.save(str(folder) + '/retained_motion.npy', self.Constant)


@pytest.mark.parametrize('accepted,later_empty', [(False, False), (True, False), (True, True)])
def test_empty_guard_retains_current_motion_and_normal_fit_is_untouched(tmp_path, accepted, later_empty):
    config = {'path_to_data': str(tmp_path), 'output_folder': str(tmp_path)}
    for name, data in {'session_index': [1, 2], 'SimilarityPairs': [[0, 1]],
                       'SimilarityMatrix': [[0, .9], [.9, 0]],
                       'ClusterMatrix': [[0, int(accepted)], [int(accepted), 0]],
                       'SimilarityThreshold': .8}.items():
        np.save(tmp_path / (name + '.npy'), data)
    calls = []
    original = Motion([-2., 2.])
    fitted = Motion([-3., 3.])
    def initialize(settings, waveforms):
        return waveforms, original
    def compute(settings):
        calls.append(settings)
        return fitted
    module = SimpleNamespace(initializeMotion=initialize, computeMotion=compute)
    def native(settings):
        module.initializeMotion(settings, np.zeros(1))
        result = module.computeMotion(settings)
        np.testing.assert_array_equal(result.Constant, fitted.Constant if accepted else original.Constant)
        if later_empty:
            # The fallback must retain the last real estimate, not restart at zero.
            np.save(tmp_path / 'ClusterMatrix.npy', np.zeros((2, 2)))
            np.testing.assert_array_equal(module.computeMotion(settings).Constant, fitted.Constant)
        np.random.random(3)
    before = np.random.get_state()
    events = run_native_dant(SimpleNamespace(runDANT=native), module, config, 3)
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    assert module.computeMotion is compute and module.initializeMotion is initialize
    assert len(calls) == int(accepted)
    assert len(events) == int(not accepted or later_empty)
    if events:
        assert events[0]['eligible_pairs'] == 0
        assert events[0]['constant_shift_um'] == ([-3., 3.] if accepted else [-2., 2.])


def test_guard_restores_functions_when_native_pipeline_raises():
    compute, initialize = lambda settings: None, lambda settings, waves: (waves, Motion([0, 0]))
    module = SimpleNamespace(computeMotion=compute, initializeMotion=initialize)
    def failing(settings):
        raise RuntimeError('native failure')
    with pytest.raises(RuntimeError, match='native failure'):
        run_native_dant(SimpleNamespace(runDANT=failing), module, {}, 1)
    assert module.computeMotion is compute and module.initializeMotion is initialize


def test_defaults_declare_stimulus_in_both_stages_and_return_fresh_configs():
    baseline = default_dant_settings()
    stimulus = default_dant_settings('waveform_acg_stimulus')
    published = default_dant_settings('upstream_example')
    assert 'PETH' not in baseline['clustering']['features']
    assert all('PETH' not in f for f in baseline['motionEstimation']['features'])
    assert 'PETH' in stimulus['clustering']['features']
    assert all('PETH' in f for f in stimulus['motionEstimation']['features'])
    assert 'PETH' not in published['clustering']['features']
    assert all('PETH' in f for f in published['motionEstimation']['features'])
    baseline['clustering']['max_distance'] = 0
    assert default_dant_settings()['clustering']['max_distance'] == 100
# [AGENT EDIT END]
