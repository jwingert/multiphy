import json

import numpy as np
import pandas as pd
import pytest

from multiphy.evidence import CalibratedPairModel


def population(seed, n=160):
    rng = np.random.default_rng(seed)
    label = (rng.random(n) < .25).astype(int)
    return pd.DataFrame({'fingerprint_r': label * .65 + rng.normal(0, .12, n),
                         'd_log_spont': (1 - label) * .7 + rng.normal(0, .15, n),
                         'same_unit': label})


def fitted_model():
    a, b = population(0), population(1)
    return CalibratedPairModel(['fingerprint_r', 'd_log_spont']).fit(
        a, b, training_groups=['fit_recording'] * len(a),
        calibration_groups=['cal_recording'] * len(b),
        reference='Synthetic independent recordings; ordinary pairs with 25% expected matches')


def test_calibration_predicts_untouched_population_and_roundtrips():
    model = fitted_model()
    heldout = population(2)
    p = model.predict_proba(heldout)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert np.mean((p - heldout.same_unit.to_numpy()) ** 2) < .08
    exported = json.loads(json.dumps(model.to_dict()))
    restored = CalibratedPairModel.from_dict(exported)
    np.testing.assert_allclose(p, restored.predict_proba(heldout))
    assert exported['training_groups'] == ['fit_recording']
    assert exported['calibration_groups'] == ['cal_recording']


def test_same_source_group_cannot_fit_and_calibrate_even_different_rows():
    a, b = population(0), population(1)
    with pytest.raises(ValueError, match='overlap'):
        CalibratedPairModel(['fingerprint_r']).fit(
            a, b, training_groups=['same_recording'] * len(a),
            calibration_groups=['same_recording'] * len(b), reference='not independent')


def test_missing_evidence_is_unavailable_not_confident_match():
    model = fitted_model()
    result = model.predict_proba(pd.DataFrame({'fingerprint_r': [np.nan, .7],
                                               'd_log_spont': [np.nan, np.nan]}))
    assert np.isnan(result[0])
    assert np.isfinite(result[1])


def test_calibration_requires_two_classes_and_source_provenance():
    a, b = population(0), population(1)
    b['same_unit'] = 1
    with pytest.raises(ValueError, match='both 0 and 1'):
        CalibratedPairModel(['fingerprint_r']).fit(
            a, b, training_groups=['fit'] * len(a), calibration_groups=['cal'] * len(b),
            reference='bad class sample')


def test_serialized_invalid_scales_and_overlapping_groups_rejected():
    spec = fitted_model().to_dict()
    spec['scale'][0] = 0
    with pytest.raises(ValueError, match='scale'):
        CalibratedPairModel.from_dict(spec)
    spec = fitted_model().to_dict()
    spec['calibration_groups'] = spec['training_groups']
    with pytest.raises(ValueError, match='provenance'):
        CalibratedPairModel.from_dict(spec)


@pytest.mark.parametrize('name', ['same_unit', 'source_unit_id', 'partition', 'cellid_a'])
def test_reference_labels_and_identifiers_cannot_be_features(name):
    with pytest.raises(ValueError, match='cannot be model features'):
        CalibratedPairModel([name])


@pytest.mark.parametrize('key,value', [('training_groups', []), ('reference', None),
                                     ('n_training', 1), ('calibration_prevalence', float('nan'))])
def test_import_cannot_bypass_fit_provenance_invariants(key, value):
    spec = fitted_model().to_dict()
    spec[key] = value
    with pytest.raises(ValueError):
        CalibratedPairModel.from_dict(spec)
def test_calibrated_pair_model_kept(tmp_path):
    from multiphy.backends.local.calibrated import CalibratedPairModel as Kept
    assert Kept(('fingerprint_r',)).properties == ('fingerprint_r',)
