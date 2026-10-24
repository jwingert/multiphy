"""Optimizer diagnostics expose failures without changing scientific scoring policy."""
import json

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import OptimizeResult

from multiphy.backends.local import evidence as ev
from multiphy.backends.local.functional_likelihood import _mixture
from multiphy.backends.local.tracker import result_from_pairs
from multiphy.contract.provenance import Provenance


@pytest.mark.parametrize('success', [True, False])
@pytest.mark.parametrize('objective', [1.25, np.nan])
def test_logistic_fit_diagnostics_preserve_returned_parameters(monkeypatch, success, objective):
    parameters = np.array([1.5, -0.25])
    calls = []

    def minimize(fun, initial, *, method):
        calls.append((initial.copy(), method, fun(parameters)))
        return OptimizeResult(x=parameters, success=np.bool_(success), status=np.int64(0 if success else 2),
                              message='converged' if success else 'ABNORMAL', nit=np.int64(7), fun=objective)

    monkeypatch.setattr('scipy.optimize.minimize', minimize)
    x, y = np.array([[-1.0], [1.0]]), np.array([-1.0, 1.0])
    diagnostics = {}
    assert ev._logistic_fit(x, y, diagnostics=diagnostics) is parameters
    assert ev._logistic_fit(x, y) is parameters  # existing callers still receive only the parameter array
    np.testing.assert_array_equal(calls[0][0], np.zeros(2))
    np.testing.assert_array_equal(calls[0][0], calls[1][0])
    assert calls[0][1:] == calls[1][1:]
    assert diagnostics == dict(attempted=True, method='L-BFGS-B', success=success,
                               status=0 if success else 2, message='converged' if success else 'ABNORMAL',
                               iterations=7, objective=objective if np.isfinite(objective) else None)
    assert json.loads(json.dumps(diagnostics, allow_nan=False)) == diagnostics


def test_logistic_fit_diagnostics_allow_missing_optional_fields(monkeypatch):
    parameters = np.array([0.0, 0.0])
    monkeypatch.setattr('scipy.optimize.minimize', lambda *args, **kwargs: OptimizeResult(x=parameters))
    diagnostics = {}
    assert ev._logistic_fit([[0.0]], [1.0], diagnostics=diagnostics) is parameters
    assert diagnostics == dict(attempted=True, method='L-BFGS-B')


@pytest.mark.parametrize('same,diff,reason', [
    ({}, {}, 'both reference classes are required'),
    ({'fingerprint_r': [0.7] * 4}, {'fingerprint_r': [0.2] * 4}, 'fewer than 20 finite reference rows'),
    ({'fingerprint_r': [0.7] * 16, 'd_log_spont': [0.1]}, {'fingerprint_r': [0.2] * 16},
     'references are not row-aligned'),
])
def test_ensemble_diagnostics_distinguish_unattempted_fits(same, diff, reason):
    ensemble = ev.Ensemble(same, diff, {'stim': [('fingerprint_r', 1)]})
    assert ensemble.fit_diagnostics == dict(attempted=False, reason=reason)
    assert not ensemble.fitted
    assert ensemble.weights == {'stim': 1.0} and ensemble.bias == 0.0


def test_failed_degenerate_fit_keeps_missing_evidence_policy(monkeypatch):
    # This injected outcome demonstrates a diagnostic path, not the cause of a past CI failure.
    monkeypatch.setattr('scipy.optimize.minimize', lambda *args, **kwargs: OptimizeResult(
        x=np.zeros(2), success=False, status=2, message='ABNORMAL', nit=0, fun=np.nan))
    same = {'fingerprint_r': np.linspace(0.7, 0.95, 16)}
    diff = {'fingerprint_r': np.linspace(-0.3, 0.6, 16)}
    ensemble = ev.Ensemble(same, diff, {'stim': [('fingerprint_r', 1)]})
    assert ensemble.fitted and ensemble.weights == {'stim': 0.0} and ensemble.bias == 0.0
    assert ensemble.fit_diagnostics['success'] is False
    table = pd.DataFrame(ensemble.score({'fingerprint_r': np.linspace(-0.2, 0.9, 12)}))
    table['cellid_a'], table['cellid_b'] = 'a', [f'b{i}' for i in range(12)]
    streams = _mixture(table, ensemble.reference_scores(same), ensemble.reference_scores(diff), ['global'])
    assert table.p_global.isna().all()
    assert streams['global']['unavailable'].startswith('LinAlgError:')
    assert 'no substitute score' in streams['global']['reason']
    units = pd.DataFrame(dict(cellid=['a', *table.cellid_b], session_id=['A', *(['B'] * 12)],
                              recording_group='g', su_type='SU'))
    diagnostics = {'ensemble_fits': {'g': {'scoring': ensemble.fit_diagnostics}}}
    result = result_from_pairs('diagnostic', '1', units, table, score_column='p_global', provenance=diagnostics)
    previous = result_from_pairs('diagnostic', '1', units, table, score_column='p_global')
    assert not result.tracks and (result.pairs.decision == 'missing_evidence').all()
    for key in ('input_pair_digest', 'run_config_digest'):
        assert result.provenance.get(key) == previous.provenance.get(key)
    serialized = json.loads(json.dumps(result.provenance.to_json(), allow_nan=False))
    assert Provenance.from_json(serialized).to_json() == serialized
