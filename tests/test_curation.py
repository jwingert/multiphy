# [AGENT EDIT START | agent: codex | user: wingertj | reason: regression coverage for curation persistence and Qt refresh | date: 2026-09-17]
"""Curation tests use only temporary ledgers and synthetic unit identities."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match import clusters as cl
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every test reads settings from a fresh folder under the pytest base temp (day-table-keys Task 6) | date: 2026-10-14]
from tests.unit_match.settings_fixtures import tmp_settings  # noqa: F401
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each unit's siteid, built from its synthetic id (Plan B Task 4) | date: 2026-09-28]
def _sites(units):
    return {u: u.split('-')[0] for u in units}
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the example is a Curation controller over one 4-unit track; the ledger tests are removed (run-curation Task 5) | date: 2026-10-05]
@pytest.fixture
def example(tmp_path):
    from nems_lbhb.unit_match import curation
    units = ['a-01', 'b-01', 'c-01', 'd-01']
    pairs = pd.DataFrame({'cellid_a': units[:3], 'cellid_b': units[1:],
                          'p_global': [.99, .98, .97]})
    controller = curation.Curation.open(tmp_path / 'curation', SimpleNamespace(tracks=[units], unmatched=[]))
    yield tmp_path, units, pairs, controller
    controller.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the cluster tests build the viewer's ClusterPanel over DictData with an Actions of fakes over the same Curation fixture (viewer-split Task 6) | date: 2026-10-06]
def _viewer_state(units, pairs, arrays=None):
    """A viewer ``PairState`` holding the units and pairs as a ``DictData`` run."""
    from nems_lbhb.unit_match.viewer.state import PairState
    from tests.unit_match.viewer_fakes import DictData
    state = PairState()
    state.set_data(DictData(units, arrays or {}, {}, _sites(units), pairs))
    return state


def _actions(controller, pairs):
    """An ``Actions`` of fakes over the ``Curation`` controller."""
    from nems_lbhb.unit_match.viewer.state import Actions
    lookup = cl.posterior_lookup(pairs)
    return Actions(select_pair=lambda a, b: None, set_verdict=lambda v: None, curation=lambda: controller,
                   track_state=lambda: controller.state, take_over=lambda: None, update_controls=lambda: None,
                   cluster_lookup=lambda: lookup, review_threshold=lambda: .9,
                   merge_refusal=lambda members: None)
# [AGENT EDIT END]


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - a required PySide6 import, so the no-matcher check never skips (Plan B Task 11) | date: 2026-09-28]
    from PySide6 import QtWidgets as widgets
    # [AGENT EDIT END]
    app = widgets.QApplication.instance() or widgets.QApplication([])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer's Qt binding module stands in for gui (viewer-split Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer import qt as gui
    # [AGENT EDIT END]
    yield app, gui


def test_refresh_split_drop_merge_and_selection_update(qt, example):
    app, gui = qt
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer's ClusterPanel, waveform and evidence panels over DictData and an Actions of fakes (viewer-split Task 6) | date: 2026-10-06]
    from nems_lbhb.unit_match.viewer import registry
    from nems_lbhb.unit_match.viewer.modes import same_day
    from nems_lbhb.unit_match.viewer.panels import plots  # noqa: F401  (registers the plot panels)
    from nems_lbhb.unit_match.viewer.panels.tables import ClusterPanel, EvidencePanel
    root, units, pairs, controller = example
    state = _viewer_state(units, pairs, {unit: {'wf': np.array([0., -1., .5])} for unit in units})
    actions = _actions(controller, pairs)
    panel = ClusterPanel(state, actions, same_day.WORDS)
    waveform = registry.PANELS['waveform'](state)
    evidence = EvidencePanel(state, actions)
    # [AGENT EDIT END]
    panel.refresh_list()
    assert panel.members.rowCount() == 4
    assert set(state.selection) == set(units)
    assert len(waveform.axes[0].lines) == 4
    panel.members.selectRow(0)
    panel.split_selected()
    assert panel.current_cluster()[1] == [units[0]]
    assert panel.members.rowCount() == 1
    assert state.selection == [units[0]]
    assert state.row is None
    assert evidence.posterior.text() == 'No pair selected'
    assert len(waveform.axes[0].lines) == 1
    # Rebuilding with the same current table row must still refresh contents.
    panel.refresh_list(preferred_id=cl.cluster_id(units[1:]))
    assert panel.members.rowCount() == 3
    panel.members.selectRow(0)
    panel.drop_member()
    assert panel.current_cluster()[1] == [units[1]]
    assert panel.members.rowCount() == 1
    assert state.selection == [units[1]]
    panel.table.selectAll()
    panel.merge_selected()
    assert panel.members.rowCount() == 4
    assert set(state.selection) == set(units)
    assert len(waveform.axes[0].lines) == 4
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every unit is held by exactly one track (run-curation Task 5) | date: 2026-10-05]
    held = [m for track in controller.state.tracks.values() for m in track['members']]
    assert sorted(held) == sorted(units)
    # [AGENT EDIT END]
    panel.hide_done.setChecked(True)
    panel.accept()
    assert panel.table.rowCount() == panel.members.rowCount() == 0
    assert state.selection == [] and state.row is None
    assert not waveform.axes[0].lines
    app.processEvents()
    panel.close()
    waveform.close()
    evidence.close()


def test_two_and_one_selected_member_clear_stale_pair(qt, example):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer's ClusterPanel over DictData and an Actions of fakes (viewer-split Task 6) | date: 2026-10-06]
    app, gui = qt
    from nems_lbhb.unit_match.viewer.modes import same_day
    from nems_lbhb.unit_match.viewer.panels.tables import ClusterPanel
    root, units, pairs, controller = example
    state = _viewer_state(units, pairs)
    panel = ClusterPanel(state, _actions(controller, pairs), same_day.WORDS)
    # [AGENT EDIT END]
    panel.refresh_list()
    selection = panel.members.selectionModel()
    flags = gui.QtCore.QItemSelectionModel.Select | gui.QtCore.QItemSelectionModel.Rows
    selection.select(panel.members.model().index(0, 0), flags)
    selection.select(panel.members.model().index(1, 0), flags)
    assert set(state.selection) == set(units[:2])
    assert state.row is not None
    selection.select(panel.members.model().index(1, 0),
                     gui.QtCore.QItemSelectionModel.Deselect | gui.QtCore.QItemSelectionModel.Rows)
    assert state.selection == units[:1]
    assert state.row is None
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pending Qt events run while the panel lives (viewer-split Task 6) | date: 2026-10-06]
    app.processEvents()
    # [AGENT EDIT END]
    panel.close()

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: test_curator_switches_backend_without_reusing_curation removed: runs no longer have ledger roots (run-curation Task 5) | date: 2026-10-05]
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table lists each day in place of the candidates table (native-same-day Task 10 fix round 1) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Build-cache import is removed; the shown result keeps native scores and refuses a colliding run id (Plan C Task 7 fix round 1) | date: 2026-10-02]
def test_result_keeps_native_scores_and_base_window_imports_nothing(qt, tmp_path, monkeypatch):
# [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the App window over the fake mode, its layout file under tmp_path (viewer-split Task 7) | date: 2026-10-06]
    from nems_lbhb.unit_match import backends
    from nems_lbhb.unit_match.viewer.app import App
    from tests.unit_match.viewer_fakes import FAKE_MODE, DictData
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: curation calls pass each unit's siteid (Plan B Task 4) | date: 2026-09-28]
    units = pd.DataFrame({'cellid': ['a-01', 'b-01'], 'su_type': ['SU', 'SU'],
                          'siteid': ['a', 'b']})
    # [AGENT EDIT END]
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'],
                          'p_global': [.99], 'posterior': [.99], 'fingerprint_r': [.7],
                          'siteid_a': ['a'], 'siteid_b': ['b'],
                          'reliability_a': [.2], 'reliability_b': [.8]})
    meta = {'animal': 'test', 'well': 1, 'built': 'synthetic', 'siteids': ['a', 'b'],
            'day_pairs': [{'siteid_a': 'a', 'siteid_b': 'b', 'days_apart': 1}], 'runclass': 'BNT'}
    arrays = {u: {'wf': np.array([0., -1., .5])} for u in units.cellid}
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the App window over the fake mode (viewer-split Task 7) | date: 2026-10-06]
    window = App(FAKE_MODE, layout_file=tmp_path / 'layout.json')
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no Build cache: the result is shown directly (Plan C Task 7 fix round 1) | date: 2026-10-02]
    # [AGENT EDIT END]
    reversed_pairs = pd.DataFrame({'cellid_a': ['b-01'], 'cellid_b': ['a-01'],
                                   'native_score': [.96]})
    imported = backends.result_from_pairs(
        'unitmatch', 'official-test', units, reversed_pairs, score_column='native_score',
        run_id='unitmatch1', provenance={'block_id': 'test_w1', 'runclass': 'BNT',
                                       'cache_built': 'synthetic', 'assignment': 'native_intermediate'})
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: Provenance is frozen; write via merged() (Task 7 ruling) | date: 2026-09-22]
    imported.provenance = imported.provenance.merged(extra={'threshold': .5})
    # [AGENT EDIT END]
    path = tmp_path / 'result.json'
    backends.save_result(imported, path)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the base window refuses an import; the result is shown through set_match_result (Plan C Task 7 fix round 1) | date: 2026-10-02]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a window without a selection refuses an import; the result is shown through set_data; no cluster_threshold (viewer-split Task 7, D17) | date: 2026-10-06]
    with pytest.raises(NotImplementedError, match='viewer --run'):
        window.import_match_result(path)
    window.set_data(DictData(units, arrays, meta, _sites(units.cellid), result=backends.load_result(path),
                             evidence=pairs))
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    assert window.state.pairs.match_score.iloc[0] == .96
    assert window.review_threshold == .5
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the cluster_threshold assert is dropped with the attribute (viewer-split Task 7, D17) | date: 2026-10-06]
    # [AGENT EDIT END]
    assert window.state.pairs.fingerprint_r.iloc[0] == .7
    assert window.state.pairs.reliability_a.iloc[0] == .8
    assert window.state.pairs.reliability_b.iloc[0] == .2
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tracking rule moved to the header's hover text | date: 2026-09-23]
    assert 'Common constrained tracking' in window.backend_header.toolTip()
    # [AGENT EDIT END]
    assert len(window.tracked._rows) == 1
    assert window.days.column_ids('a') == ['a-01']
    assert window.days.column_ids('b') == ['b-01']
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the cache_built check left with the Build-cache import (Plan C Task 7 fix round 1) | date: 2026-10-02]
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no result-digest guard, so the collision block is removed (run-curation Task 5, D21) | date: 2026-10-05]
    # [AGENT EDIT END]
    window.close()
# [AGENT EDIT END]


def test_native_backend_without_confidence_does_not_use_cached_probability():
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'],
                          'match_score': [np.nan], 'p_global': [.99]})
    assert cl.score_column(pairs) == 'match_score'
    assert cl.posterior_lookup(pairs) == {}

def test_zero_height_fixed_aspect_canvas_defers_and_resumes_render(qt):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the canvas is the viewer's panels.plots.base.SafeCanvas (viewer-split Task 4) | date: 2026-10-06]
    from nems_lbhb.unit_match.viewer.panels.plots import base
    figure = base.Figure(figsize=(1, 1))
    canvas = base.SafeCanvas(figure)
    # [AGENT EDIT END]
    axis = figure.subplots()
    axis.plot([0, 1], [0, 2])
    axis.set_aspect('equal', adjustable='box')
    draws = []
    canvas.mpl_connect('draw_event', lambda event: draws.append(event))
    # A dock may allocate no height before its final layout. Fixed-aspect
    # rendering used to raise inside an asynchronous Qt draw callback here.
    canvas.resize(100, 0)
    figure.set_size_inches(1, 0, forward=False)
    canvas.draw()
    assert draws == []
    # Deferral must not leave the canvas permanently blank after resizing.
    canvas.resize(300, 200)
    figure.set_size_inches(3, 2, forward=False)
    canvas.draw()
    assert len(draws) == 1
    assert np.asarray(canvas.buffer_rgba()).shape == (200, 300, 4)
    canvas.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the GUI runs no matcher; a Build cache opens unassigned (Task 3) | date: 2026-09-27]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _block_cache removed: it fed the Build-cache tests (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _quiet_curator removed: nothing called it and gui.Curator is deleted (viewer-split final review) | date: 2026-10-07]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the unassigned_block_result tests are removed with the Build cache (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the unassigned view shows measured features and no cached matching scores (Task 3 fix round 1) | date: 2026-09-27]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the stream labels come from the Qt-free display module and are asserted literally (viewer-split Task 2) | date: 2026-10-06]
def test_result_input_label_names_the_stream(tmp_path):
    from nems_lbhb.unit_match.viewer import display
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the D30 stream labels; a result without saved weight_stim names its combined stream physiology + waveform (event-optional Task 12) | date: 2026-10-10]
    stream_labels = {'combined': 'Functional likelihood: Physiology + waveform',
                     'stimulus': 'Functional likelihood: Event response only',
                     'physiology': 'Functional likelihood: Physiology only'}
    # [AGENT EDIT END]
# [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: new tests use tmp_path; the label is read from a saved and reloaded result (Task 3 review r3) | date: 2026-09-27]
    from nems_lbhb.unit_match.backends import load_result, result_from_pairs, save_result
    # [AGENT EDIT END]
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'p_global': [.95]})
    labels = set()
    for stream in ('combined', 'stimulus', 'physiology'):
        result = result_from_pairs('functional_likelihood', '1', units, pairs, score_column='p_global', run_id=stream)
        result.provenance = result.provenance.merged(
            extra=dict(feature_stream=stream, evidence_display_name='Functional likelihood'))
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: label the reloaded result (Task 3 review r3) | date: 2026-09-27]
        save_result(result, tmp_path / (stream + '.json'))
        result = load_result(tmp_path / (stream + '.json'))
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Qt-free display module and literal labels (viewer-split Task 2) | date: 2026-10-06]
        label = display.result_input_label(result)
        assert label == stream_labels[stream]
        assert 'inputs: ' + label in display.result_summary(result).splitlines()
        # [AGENT EDIT END]
        labels.add(label)
    assert len(labels) == 3
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the precompute test is removed with the Build cache (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


def test_gui_refuses_backend_argument(qt, tmp_path, monkeypatch, capsys):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the one viewer's main refuses --backend (viewer-split Task 8) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer import app
    with pytest.raises(SystemExit) as exc:
        app.main(['--run', 'x', '--backend', 'x'])
    # [AGENT EDIT END]
    assert exc.value.code == 2
    assert 'unrecognized arguments: --backend x' in capsys.readouterr().err
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pins tmp_settings: with no fixture requested, the settings file lies in a fresh settings folder under the pytest base temp, never ~/.config/multiphy or an after folder (day-table-keys Task 6) | date: 2026-10-14]
def test_settings_live_in_a_fresh_tmp_folder():
    import re
    from pathlib import Path
    from nems_lbhb.unit_match.viewer import theme
    p = Path(theme.settings().fileName())
    assert p.name == 'multiphy.ini'
    assert p.parent.name == 'multiphy'
    assert re.fullmatch(r'settings\d+', p.parent.parent.name) is not None
# [AGENT EDIT END]
