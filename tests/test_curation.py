"""Curation tests use only temporary ledgers and synthetic unit identities."""
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from multiphy import clusters as cl
from tests.settings_fixtures import tmp_settings  # noqa: F401


def _sites(units):
    return {u: u.split('-')[0] for u in units}


@pytest.fixture
def example(tmp_path):
    from multiphy import curation
    units = ['a-01', 'b-01', 'c-01', 'd-01']
    pairs = pd.DataFrame({'cellid_a': units[:3], 'cellid_b': units[1:],
                          'p_global': [.99, .98, .97]})
    controller = curation.Curation.open(tmp_path / 'curation', SimpleNamespace(tracks=[units], unmatched=[]))
    yield tmp_path, units, pairs, controller
    controller.close()


def _viewer_state(units, pairs, arrays=None):
    """A viewer ``PairState`` holding the units and pairs as a ``DictData`` run."""
    from multiphy.viewer.state import PairState
    from tests.viewer_fakes import DictData
    state = PairState()
    state.set_data(DictData(units, arrays or {}, {}, _sites(units), pairs))
    return state


def _actions(controller, pairs):
    """An ``Actions`` of fakes over the ``Curation`` controller."""
    from multiphy.viewer.state import Actions
    lookup = cl.posterior_lookup(pairs)
    return Actions(select_pair=lambda a, b: None, set_verdict=lambda v: None, curation=lambda: controller,
                   track_state=lambda: controller.state, take_over=lambda: None, update_controls=lambda: None,
                   cluster_lookup=lambda: lookup, review_threshold=lambda: .9,
                   merge_refusal=lambda members: None)


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from PySide6 import QtWidgets as widgets
    app = widgets.QApplication.instance() or widgets.QApplication([])
    from multiphy.viewer import qt as gui
    yield app, gui


def test_refresh_split_drop_merge_and_selection_update(qt, example):
    app, gui = qt
    from multiphy.viewer import registry
    from multiphy.viewer.modes import same_day
    from multiphy.viewer.panels import plots  # noqa: F401  (registers the plot panels)
    from multiphy.viewer.panels.tables import ClusterPanel, EvidencePanel
    root, units, pairs, controller = example
    state = _viewer_state(units, pairs, {unit: {'wf': np.array([0., -1., .5])} for unit in units})
    actions = _actions(controller, pairs)
    panel = ClusterPanel(state, actions, same_day.WORDS)
    waveform = registry.PANELS['waveform'](state)
    evidence = EvidencePanel(state, actions)
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
    held = [m for track in controller.state.tracks.values() for m in track['members']]
    assert sorted(held) == sorted(units)
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
    app, gui = qt
    from multiphy.viewer.modes import same_day
    from multiphy.viewer.panels.tables import ClusterPanel
    root, units, pairs, controller = example
    state = _viewer_state(units, pairs)
    panel = ClusterPanel(state, _actions(controller, pairs), same_day.WORDS)
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
    app.processEvents()
    panel.close()


def test_result_keeps_native_scores_and_base_window_imports_nothing(qt, tmp_path, monkeypatch):
    from multiphy import backends
    from multiphy.viewer.app import App
    from tests.viewer_fakes import FAKE_MODE, DictData
    units = pd.DataFrame({'cellid': ['a-01', 'b-01'], 'su_type': ['SU', 'SU'],
                          'siteid': ['a', 'b']})
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'],
                          'p_global': [.99], 'posterior': [.99], 'fingerprint_r': [.7],
                          'siteid_a': ['a'], 'siteid_b': ['b'],
                          'reliability_a': [.2], 'reliability_b': [.8]})
    meta = {'animal': 'test', 'well': 1, 'built': 'synthetic', 'siteids': ['a', 'b'],
            'day_pairs': [{'siteid_a': 'a', 'siteid_b': 'b', 'days_apart': 1}], 'runclass': 'BNT'}
    arrays = {u: {'wf': np.array([0., -1., .5])} for u in units.cellid}
    window = App(FAKE_MODE, layout_file=tmp_path / 'layout.json')
    reversed_pairs = pd.DataFrame({'cellid_a': ['b-01'], 'cellid_b': ['a-01'],
                                   'native_score': [.96]})
    imported = backends.result_from_pairs(
        'unitmatch', 'official-test', units, reversed_pairs, score_column='native_score',
        run_id='unitmatch1', provenance={'block_id': 'test_w1', 'runclass': 'BNT',
                                       'cache_built': 'synthetic', 'assignment': 'native_intermediate'})
    imported.provenance = imported.provenance.merged(extra={'threshold': .5})
    path = tmp_path / 'result.json'
    backends.save_result(imported, path)
    with pytest.raises(NotImplementedError, match='viewer --run'):
        window.import_match_result(path)
    window.set_data(DictData(units, arrays, meta, _sites(units.cellid), result=backends.load_result(path),
                             evidence=pairs))
    assert window.state.pairs.match_score.iloc[0] == .96
    assert window.review_threshold == .5
    assert window.state.pairs.fingerprint_r.iloc[0] == .7
    assert window.state.pairs.reliability_a.iloc[0] == .8
    assert window.state.pairs.reliability_b.iloc[0] == .2
    assert 'Common constrained tracking' in window.backend_header.toolTip()
    assert len(window.tracked._rows) == 1
    assert window.days.column_ids('a') == ['a-01']
    assert window.days.column_ids('b') == ['b-01']
    window.close()


def test_native_backend_without_confidence_does_not_use_cached_probability():
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'],
                          'match_score': [np.nan], 'p_global': [.99]})
    assert cl.score_column(pairs) == 'match_score'
    assert cl.posterior_lookup(pairs) == {}

def test_zero_height_fixed_aspect_canvas_defers_and_resumes_render(qt):
    from multiphy.viewer.panels.plots import base
    figure = base.Figure(figsize=(1, 1))
    canvas = base.SafeCanvas(figure)
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








def test_result_input_label_names_the_stream(tmp_path):
    from multiphy.viewer import display
    stream_labels = {'combined': 'Functional likelihood: Physiology + waveform',
                     'stimulus': 'Functional likelihood: Event response only',
                     'physiology': 'Functional likelihood: Physiology only'}
    from multiphy.backends import load_result, result_from_pairs, save_result
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'p_global': [.95]})
    labels = set()
    for stream in ('combined', 'stimulus', 'physiology'):
        result = result_from_pairs('functional_likelihood', '1', units, pairs, score_column='p_global', run_id=stream)
        result.provenance = result.provenance.merged(
            extra=dict(feature_stream=stream, evidence_display_name='Functional likelihood'))
        save_result(result, tmp_path / (stream + '.json'))
        result = load_result(tmp_path / (stream + '.json'))
        label = display.result_input_label(result)
        assert label == stream_labels[stream]
        assert 'inputs: ' + label in display.result_summary(result).splitlines()
        labels.add(label)
    assert len(labels) == 3




def test_gui_refuses_backend_argument(qt, tmp_path, monkeypatch, capsys):
    from multiphy.viewer import app
    with pytest.raises(SystemExit) as exc:
        app.main(['--run', 'x', '--backend', 'x'])
    assert exc.value.code == 2
    assert 'unrecognized arguments: --backend x' in capsys.readouterr().err


def test_settings_live_in_a_fresh_tmp_folder():
    import re
    from pathlib import Path
    from multiphy.viewer import theme
    p = Path(theme.settings().fileName())
    assert p.name == 'multiphy.ini'
    assert p.parent.name == 'multiphy'
    assert re.fullmatch(r'settings\d+', p.parent.parent.name) is not None
