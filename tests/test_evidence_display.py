# [AGENT EDIT START | agent: codex | user: wingertj | reason: test unknown diagnostic outcomes separately from actual failures and show only actual method inputs | date: 2026-09-18]
import os
from types import SimpleNamespace

import numpy as np
import pytest


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a required PySide6 import and the viewer's display module (viewer-split Task 6) | date: 2026-10-06]
@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtWidgets as widgets
    app = widgets.QApplication.instance() or widgets.QApplication([])
    from nems_lbhb.unit_match.viewer import display
    return app, display


def _actions():
    """An ``Actions`` whose every call does nothing."""
    from nems_lbhb.unit_match.viewer.state import Actions
    return Actions(select_pair=lambda a, b: None, set_verdict=lambda _: None, curation=lambda: None,
                   track_state=lambda: None, take_over=lambda: None, update_controls=lambda: None,
                   cluster_lookup=lambda: {}, review_threshold=lambda: .9, merge_refusal=lambda members: None)
# [AGENT EDIT END]


@pytest.mark.parametrize('saved,expected', [
    ({}, ['not evaluated'] * 4),
    ({'gates_passed': np.nan}, ['not evaluated'] * 4),
    ({'gates_passed': 'prox,isi'}, ['pass', 'fail', 'pass', 'fail']),
    ({'gates_passed': ''}, ['fail'] * 4),
    ({'gate_prox': True, 'gate_rate': False, 'gate_isi': np.nan}, ['pass', 'fail', 'not evaluated', 'not evaluated']),
    ({'gate_prox': False, 'gates_passed': 'prox,rate,isi,sw'}, ['fail', 'not evaluated', 'not evaluated', 'not evaluated']),
])
def test_gate_display_preserves_saved_failures_and_does_not_invent_missing_ones(qt, saved, expected):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer's EvidencePanel over an Actions of no-ops; colours from viewer.display (viewer-split Task 6) | date: 2026-10-06]
    from nems_lbhb.unit_match.viewer.panels.tables import EvidencePanel
    from nems_lbhb.unit_match.viewer.state import PairState
    _, display = qt
    state = PairState()
    state.row = dict(cellid_a='a-1', cellid_b='b-1', depth_delta_um=5., d_log_spont=.02,
                     isi_cosine=.99, d_spike_width_ms=.01, match_score=.9, **saved)
    panel = EvidencePanel(state, _actions())
    panel.refresh()
    assert [panel.table.item(i, 2).text() for i in range(4)] == expected
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each gate cell carries the theme token of its outcome (launcher-gui Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer import theme
    for i, outcome in enumerate(expected):
        assert panel.table.item(i, 2).data(theme.TONE_ROLE) == (
            'muted' if outcome == 'not evaluated' else 'accept' if outcome == 'pass' else 'reject')
    # [AGENT EDIT END]
    panel.close()
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the input labels come from the Qt-free display module; the test needs no Qt (viewer-split Task 2) | date: 2026-10-06]
def test_method_inputs_distinguish_dant_motion_from_clustering_and_local_model():
    from nems_lbhb.unit_match.viewer import display
    result = SimpleNamespace(backend_id='dant', provenance={'parameters': {
        'clustering': {'features': ['Waveform', 'AutoCorr']},
        'motionEstimation': {'features': [['AutoCorr', 'PETH']]}}})
    assert display.result_input_label(result) == 'DANT clustering: waveform, spike timing; fingerprint for motion only'
    result.provenance['parameters']['clustering']['features'].append('PETH')
    assert display.result_input_label(result) == 'DANT clustering: waveform, spike timing, stimulus fingerprint'
    result.backend_id = 'unitmatch'
    assert display.result_input_label(result) == 'Spike waveform and spatial shape'
    result.backend_id = 'deepunitmatch'
    assert display.result_input_label(result) == 'Learned waveform features'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: former constrained fixture is a unitmatch result built by result_from_pairs; saved calibrated-model properties do not override its declared inputs; functional_likelihood labelled from its stream provenance (Task 3 fix round 1) | date: 2026-09-27]
    import pandas as pd
    from nems_lbhb.unit_match.backends import result_from_pairs
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'match_score': [.95]})
    saved = result_from_pairs('unitmatch', '1', units, pairs, run_id='saved-model',
        provenance={'calibrated_model': {'properties': ['isi_cosine', 'acg_cosine']}})
    assert display.result_input_label(saved) == 'Spike waveform and spatial shape'
    likelihood = result_from_pairs('functional_likelihood', '1', units, pairs, run_id='likelihood',
        provenance={'feature_stream': 'physiology', 'evidence_display_name': 'Functional likelihood'})
    assert display.result_input_label(likelihood) == 'Functional likelihood: Physiology only'
    # [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the evidence table names the peak-channel waveform and footprint correlations and the all-channel footprint correlation (event-optional Task 1) | date: 2026-10-09]
def test_evidence_rows_name_waveform_and_footprint():
    from nems_lbhb.unit_match.viewer.display import EVIDENCE_FIELDS
    rows = [('waveform_r_local', 'waveform correlation, peak channels', None),
            ('footprint_r_local', 'footprint correlation, peak channels', None),
            ('footprint_r', 'footprint correlation, all channels', None)]
    keys = [key for key, _, _ in EVIDENCE_FIELDS]
    assert 'waveform_r_local' in keys
    start = keys.index('waveform_r_local')
    assert EVIDENCE_FIELDS[start:start + 3] == rows
    assert start > keys.index('d_spike_width_ms')
    assert not [key for key in keys if key.startswith('template')]
# [AGENT EDIT END]
