# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the ensemble fits the streams a run measured, and only those (event-optional Task 6) | date: 2026-10-10]
"""Streams follow the run's evidence: which streams a run has, and an ensemble over any stream dict."""
import numpy as np
import pytest

from multiphy.backends.local import evidence as ev
from multiphy.backends.local.evidence import run_streams
from multiphy.backends.local.examples import fit_ensemble, labeled_examples

from .synthetic_drift import make_drift_days, observations_by_session

KEYS = ('sound_0', 'sound_1', 'sound_2', 'sound_3')


@pytest.fixture(autouse=True)
def _in_tmp_path(tmp_path, monkeypatch):
    # every test runs in its own tmp_path, so a stray relative write never lands in the repo
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope='module')
def drift_examples():
    days = make_drift_days(seed=0)
    by_session = observations_by_session(days)
    observations = {c: o for session in by_session.values() for c, o in session.items()}
    session_of = dict(zip(days.units.cellid, days.units.session_id))
    return labeled_examples(observations, session_of, KEYS)


def test_run_streams_by_evidence():
    expected = {(True, True): ['stim', 'phys', 'wave'], (True, False): ['stim', 'phys'],
                (False, True): ['phys', 'wave'], (False, False): ['phys']}
    for (events, waveform), names in expected.items():
        streams = run_streams(events, waveform=waveform)
        assert list(streams) == names, (events, waveform)
        phys = [('d_log_spont', -1), ('isi_cosine', +1), ('acg_cosine', +1)]
        assert streams['phys'] == (phys + [('d_log_rate_session', -1)] if events else phys)
        if events:
            assert streams['stim'] == [('fingerprint_r', +1)]
        if waveform:
            assert streams['wave'] == [('waveform_r_local', +1), ('footprint_r_local', +1), ('d_spike_width_ms', -1)]


def test_generic_ensemble_matches_fixed_streams(drift_examples):
    # the numbers the two fixed streams gave at e596dd1d
    ensemble, same_ref, _ = fit_ensemble(drift_examples, ev.STREAMS)
    assert list(ensemble.weights) == ['stim', 'phys']
    np.testing.assert_allclose([ensemble.weights['stim'], ensemble.weights['phys'], ensemble.bias],
                               [1.809718, 0.16562, -0.815351], rtol=0, atol=1e-6)
    np.testing.assert_allclose(ensemble.reference_scores(same_ref)['s_global'][:3],
                               [8.450413, 8.50653, 8.332968], rtol=0, atol=1e-6)


def test_ensemble_scores_only_its_streams(drift_examples):
    ensemble, same_ref, _ = fit_ensemble(drift_examples, run_streams(False, waveform=True))
    scores = ensemble.reference_scores(same_ref)
    assert sorted(scores) == ['s_global', 's_phys', 's_wave']
    row = drift_examples.iloc[0].to_dict()
    assert sorted(ensemble.score(row)) == ['s_global', 's_phys', 's_wave']
    # the wave properties had references, so the wave stream scored rather than sat neutral at zero
    assert np.any(np.asarray(scores['s_wave']) != 0)
# [AGENT EDIT END]
