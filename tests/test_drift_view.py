# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tests of the Qt-free drift frame reader (drift-view Task 1) | date: 2026-10-08]
"""The Qt-free drift frame: a run's saved registration read into each day's drawn depths."""
import dataclasses
import inspect

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match.viewer.drift_frame import (Column, DriftFrame, drawn_group, drift_frame,
                                                     electrode_depths, group_label, group_sessions,
                                                     group_units, unit_depth)
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered unit positions and distances under test (browse-same-day Task 1) | date: 2026-10-12]
import math

from nems_lbhb.unit_match.viewer.drift_frame import registered_positions, unit_distance
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the drift panel tests build a PairState over DictData and open a project run (drift-view Task 4) | date: 2026-10-08]
import os

from nems_lbhb.unit_match import backends
from nems_lbhb.unit_match.backends.local import functional_likelihood as fl
from nems_lbhb.unit_match.contract import registry as contract_registry
from nems_lbhb.unit_match.project.folder import Project
from nems_lbhb.unit_match.project.selection import Selection
from nems_lbhb.unit_match.run.project_run import run_selection
from nems_lbhb.unit_match.viewer import display, registry
from nems_lbhb.unit_match.viewer.app import open_run
from nems_lbhb.unit_match.viewer.panels import plots  # noqa: F401  (registers the plot panels)
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt through viewer/qt.py, the binding pin (drift-view Task 4, fix round 1) | date: 2026-10-08]
from nems_lbhb.unit_match.viewer.qt import QtWidgets
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.state import PairState

from .synthetic_day import extracted_project
from .viewer_fakes import DictData
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tests of the Qt-free drift frame reader (drift-view Task 1) | date: 2026-10-08]


def _prov(registration, group='P'):
    return {'group_runs': {group: {'run_id': 'r', 'provenance': {'group': group,
                                                                 'registration': registration}}}}


def _offsets(frame):
    return [c.offsets_um for c in frame.columns]


def test_drift_frame_reads_groups():
    prov = _prov({'method': 'matches', 'offsets_um': {'D1': 5.0, 'D2': 17.5, 'D3': 0.0},
                  'diagnostics': {'P': {'groups': [['D1', 'D2'], ['D3']]}}})
    frame = drift_frame(prov, 'P', ['D1', 'D2', 'D3'], saves_offsets=True)
    assert frame.anchor == 'D1'
    assert frame.anchor_offset_um == 5.0
    assert _offsets(frame) == [(5.0,), (17.5,), ()]
    assert [c.reason for c in frame.columns] == ['', '', 'no matches to D1']
    assert frame.method == 'matches'
    assert frame.no_offsets == ''
    assert unit_depth(frame, frame.columns[1], 'b:1', 120.0, 27.0) == 132.5
    assert unit_depth(frame, frame.columns[2], 'c:1', 300.0, 43.0) == 300.0


def test_anchor_moves_when_day1_alone():
    prov = _prov({'method': 'matches', 'offsets_um': {'D1': 0.0, 'D2': 3.0, 'D3': -4.0},
                  'diagnostics': {'P': {'groups': [['D1'], ['D2', 'D3']]}}})
    frame = drift_frame(prov, 'P', ['D1', 'D2', 'D3'], saves_offsets=True)
    assert frame.anchor == 'D2'
    assert _offsets(frame) == [(), (3.0,), (-4.0,)]
    assert unit_depth(frame, frame.columns[2], 'c', 10.0, 0.0) == 3.0
    assert frame.columns[0].reason == 'no matches to D2'


def test_anchor_tie_takes_earliest_first_day():
    sessions = ['D1', 'D2', 'D3', 'D4', 'D5']
    prov = _prov({'method': 'matches', 'offsets_um': {s: 0.0 for s in sessions},
                  'diagnostics': {'P': {'groups': [['D1'], ['D4', 'D5'], ['D2', 'D3']]}}})
    assert drift_frame(prov, 'P', sessions, saves_offsets=True).anchor == 'D2'


def test_no_offsets_reasons():
    sessions = ['D1', 'D2']
    cases = [
        ({}, True, 'run predates offsets'),
        ({}, False, 'backend saves no offsets'),
        (_prov({'method': 'none', 'offsets_um': {}}), True, 'registration off'),
        (_prov({'method': 'matches', 'offsets_um': {}, 'diagnostics': {}}), True,
         'group not registered'),
        (_prov({'method': 'native', 'offsets_um': {},
                'diagnostics': {'P': {'note': 'dant kept initial motion'}}}), True,
         'dant kept initial motion'),
    ]
    for prov, saves, reason in cases:
        frame = drift_frame(prov, 'P', sessions, saves_offsets=saves)
        assert frame.no_offsets == reason
        assert frame.anchor is None
        assert _offsets(frame) == [(), ()]
        assert [c.reason for c in frame.columns] == ['', '']


def test_pair_drifts_shift_direct_to_anchor():
    prov = _prov({'method': 'native', 'pair_drifts': [
        {'session_a': 'D1', 'session_b': 'D2', 'depth_um': 10.0, 'n_matches': 7},
        {'session_a': 'D1', 'session_b': 'D3', 'depth_um': None, 'n_matches': 0},
        {'session_a': 'D2', 'session_b': 'D3', 'depth_um': -4.0, 'n_matches': 6}]})
    frame = drift_frame(prov, 'P', ['D1', 'D2', 'D3'], saves_offsets=True)
    assert frame.anchor == 'D1'
    assert _offsets(frame) == [(0.0,), (10.0,), ()]
    assert frame.columns[2].reason == 'no matches to D1'


def test_per_shank_depths():
    prov = _prov({'method': 'native', 'offsets_um': {'D1': 0.0},
                  'shank_offsets_um': {'D2': [-40.0, -20.0]}, 'unit_shanks': {'b:1': 1, 'b:2': 0},
                  'shank_width_um': 225.0, 'diagnostics': {'P': {'groups': [['D1', 'D2']]}}})
    frame = drift_frame(prov, 'P', ['D1', 'D2'], saves_offsets=True)
    d2 = frame.columns[1]
    assert d2.offsets_um == (-40.0, -20.0)
    assert unit_depth(frame, d2, 'b:1', 100.0, 11.0) == 80.0
    assert unit_depth(frame, d2, 'b:2', 100.0, 261.0) == 60.0
    assert unit_depth(frame, d2, 'b:3', 100.0, 261.0) == 80.0
    assert unit_depth(frame, d2, 'b:4', 100.0, float('nan')) is None
    geometry = np.array([[11.0, 0.0], [261.0, 0.0], [500.0, 0.0]])
    np.testing.assert_array_equal(electrode_depths(frame, d2, geometry), [-40.0, -20.0, -40.0])
    pair = _prov({'method': 'native', 'pair_drifts': [
        {'session_a': 'D1', 'session_b': 'D2', 'shank_depth_um': [-40.0, -20.0],
         'shank_x_um': [0.0, 0.0], 'n_matches': 50}],
        'unit_shanks': {'b:1': 1, 'b:2': 0}, 'shank_width_um': 225.0})
    frame = drift_frame(pair, 'P', ['D1', 'D2'], saves_offsets=True)
    assert unit_depth(frame, frame.columns[1], 'b:1', 100.0, 11.0) == 80.0


def test_linear_depths_reanchor():
    prov = _prov({'method': 'native', 'offsets_um': {'D1': 5.0, 'D2': 20.0, 'D3': 0.0},
                  'depth_scale': {'D1': 1.0, 'D2': 0.5, 'D3': 1.0}, 'unit_offsets_um': {'b:1': 30.0},
                  'diagnostics': {'P': {'groups': [['D1', 'D2']]}}})
    frame = drift_frame(prov, 'P', ['D1', 'D2', 'D3'], saves_offsets=True)
    d2, d3 = frame.columns[1], frame.columns[2]
    np.testing.assert_allclose(electrode_depths(frame, d2, np.array([[0.0, 100.0], [0.0, 300.0]])),
                               [65.0, 165.0])
    assert unit_depth(frame, d2, 'b:1', 100.0, 0.0) == 125.0
    assert unit_depth(frame, d2, 'b:9', 100.0, 0.0) == 65.0
    assert unit_depth(frame, d3, 'c:1', 300.0, 0.0) == 300.0
    prov = _prov({'method': 'native', 'offsets_um': {'D1': 0.0, 'D2': 10.0, 'D3': 4.0},
                  'depth_scale': {'D1': 1.0, 'D2': 0.5, 'D3': 0.8},
                  'diagnostics': {'P': {'groups': [['D1'], ['D2', 'D3']]}}})
    frame = drift_frame(prov, 'P', ['D1', 'D2', 'D3'], saves_offsets=True)
    assert frame.anchor == 'D2'
    y = np.array([[0.0, 100.0]])
    np.testing.assert_allclose(electrode_depths(frame, frame.columns[2], y), [148.0])
    np.testing.assert_allclose(electrode_depths(frame, frame.columns[1], y), [100.0])


@pytest.mark.parametrize('registration, sessions', [
    ({'method': 'matches', 'offsets_um': {'D1': 'x'}}, ['D1', 'D2', 'D3']),
    ({'method': 'native', 'offsets_um': {'D1': 0.0},
      'shank_offsets_um': {'D2': [1.0, 2.0], 'D3': [1.0]}, 'shank_width_um': 225.0},
     ['D1', 'D2', 'D3']),
    ({'method': 'native', 'pair_drifts': [{'session_a': 'D1', 'session_b': 'D2', 'depth_um': 3.0}]},
     ['D1', 'D2', 'D3']),
    ({'method': 'native', 'pair_drifts': [
        {'session_a': 'D2', 'session_b': 'D1', 'depth_um': 3.0, 'n_matches': 4}]}, ['D1', 'D2']),
])
def test_unreadable_registration_raises(registration, sessions):
    with pytest.raises(ValueError, match='registration of P'):
        drift_frame(_prov(registration), 'P', sessions, saves_offsets=True)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - a saved null groups and a zero-match pair's shank list without a width are refused (drift-view Task 1) | date: 2026-10-08]
@pytest.mark.parametrize('registration', [
    {'method': 'matches', 'offsets_um': {'D1': 0.0, 'D2': 1.0}, 'diagnostics': {'P': {'groups': None}}},
    {'method': 'native', 'pair_drifts': [
        {'session_a': 'D1', 'session_b': 'D2', 'shank_depth_um': [1.0, 2.0], 'n_matches': 0}]},
])
def test_null_groups_and_zero_match_shank_list_raise(registration):
    with pytest.raises(ValueError, match='registration of P'):
        drift_frame(_prov(registration), 'P', ['D1', 'D2'], saves_offsets=True)
# [AGENT EDIT END]


def test_group_sessions_orders_and_matches_exactly():
    units = pd.DataFrame({'cellid': ['a', 'b', 'c'], 'recording_group': ['P', 'P', 'Q'],
                          'session_id': ['D1', 'D2', 'D3'],
                          'session_date': ['2026-01-02', '2026-01-01', '2026-01-03']})
    assert list(group_units(units, 'P').cellid) == ['a', 'b']
    assert group_sessions(units, 'P') == ['D2', 'D1']
    assert group_sessions(units, 'P ') == []
    assert drawn_group(units, ['c']) == 'Q'
    assert drawn_group(units, []) == 'P'
    assert drawn_group(units, ['zz']) == 'P'


def test_group_label_joins_json_list():
    assert group_label('["P", "D1"]') == 'P · D1'
    assert group_label('P') == 'P'
    assert group_label('[broken') == '[broken'


def test_produced_signatures():
    params = inspect.signature(drift_frame).parameters
    assert list(params) == ['provenance', 'group', 'sessions', 'saves_offsets']
    assert params['saves_offsets'].kind is inspect.Parameter.KEYWORD_ONLY
    assert inspect.signature(drift_frame).return_annotation is DriftFrame
    assert [f.name for f in dataclasses.fields(DriftFrame)] == [
        'group', 'anchor', 'method', 'columns', 'no_offsets', 'anchor_scale', 'anchor_offset_um',
        'unit_offsets_um', 'unit_shanks', 'shank_width_um']
    assert [f.name for f in dataclasses.fields(Column)] == ['session', 'scale', 'offsets_um', 'reason']
    assert list(inspect.signature(unit_depth).parameters) == ['frame', 'column', 'cellid', 'depth',
                                                              'x_um']
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: backends that save a registration declare day_offsets in their spec (drift-view Task 2); pair_logistic saves one too (drift-view post-run fix) | date: 2026-10-08]
def test_backends_declare_day_offsets():
    from nems_lbhb.unit_match.contract import registry
    from nems_lbhb.unit_match.contract.backend import Provides
    ids = ('functional_likelihood', 'unitmatch', 'dant', 'deepunitmatch', 'pair_logistic')
    assert [registry.get(b).spec.provides.day_offsets for b in ids] == [True, True, True, True, True]
    assert Provides(frozenset(), frozenset()).day_offsets is False
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the drift panel draws each day of the drawn group in the anchor's frame, with hover text and the selection joined (drift-view Task 4) | date: 2026-10-08]
READS_GROUPS = {'method': 'matches', 'offsets_um': {'D1': 5.0, 'D2': 17.5, 'D3': 0.0},
                'diagnostics': {'P': {'groups': [['D1', 'D2'], ['D3']]}}}


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _table():
    return pd.DataFrame({'cellid': ['a:1', 'b:1', 'c:1', 'a:2'], 'siteid': ['D1', 'D2', 'D3', 'D1'],
                         'session_id': ['D1', 'D2', 'D3', 'D1'], 'recording_group': 'P',
                         'session_date': ['2026-01-01', '2026-01-02', '2026-01-03', '2026-01-01'],
                         'su_type': 'SU', 'depth': [100.0, 120.0, 300.0, 200.0],
                         'x_um': [11.0, 27.0, 43.0, 59.0]})


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the panel fixture takes a units table, for the no-depth case (drift-view Task 4, fix round 2) | date: 2026-10-08]
def _drift_panel(qt, registration=READS_GROUPS, meta=None, provenance=True, table=None):
    """The drift panel over the fixture's ``DictData``, refreshed with nothing selected."""
    table = _table() if table is None else table
    # [AGENT EDIT END]
    pairs = pd.DataFrame({'cellid_a': ['a:1'], 'cellid_b': ['b:1'], 'match_score': [0.99]})
    kwargs = {'provenance': _prov(registration)} if provenance else {}
    result = backends.result_from_pairs('dict_data', '1', table, pairs, run_id='dict_data', **kwargs)
    state = PairState()
    state.set_data(DictData(table, {}, meta or {}, dict(zip(table.cellid, table.siteid)), result=result))
    panel = registry.PANELS['drift'](state)
    panel.refresh()
    qt.processEvents()          # the idle redraw runs while the panel lives
    return panel


def _points(panel):
    return panel.points.sort_values('cellid').reset_index(drop=True)


def test_drift_panel_draws_registered_days(qt):
    panel = _drift_panel(qt)
    points = _points(panel)
    assert list(points.cellid) == ['a:1', 'a:2', 'b:1', 'c:1']
    assert list(points.x) == [0.0, 48.0, 104.0, 208.0]
    assert list(points.y) == [100.0, 200.0, 132.5, 300.0]
    assert list(points['shift']) == [0.0, 0.0, 12.5, 0.0]
    assert list(points.filled) == [True, False, True, False]
    assert list(points.muted) == [False, False, False, True]
    ax = panel.axes[0]
    assert [t.get_text() for t in ax.get_xticklabels()] == ['D1', 'D2', 'D3\nunlinked']
    assert ax.get_ylabel() == 'depth in D1 frame (µm)'
    assert ax.get_title() == 'P'
    panel.close()


def test_drift_hover_text(qt):
    panel = _drift_panel(qt)
    assert panel.hover_text(104.0, 132.5) == 'b:1 · D2 · +12.5 µm · matches'
    assert panel.hover_text(112.0, -1000.0) == 'D2 · +12.5 µm · matches'
    assert panel.hover_text(200.0, -1000.0) == 'D3 · unlinked · no matches to D1'
    assert panel.hover_text(-500.0, 0.0) == ''
    panel.close()


def test_drift_selection_joined(qt):
    panel = _drift_panel(qt)
    panel.state.set_selection(['a:1', 'b:1'])
    qt.processEvents()
    (line,) = [ln for ln in panel.axes[0].lines if ln.get_gid() == 'selection']
    assert list(line.get_xdata()) == [0.0, 104.0]
    assert list(line.get_ydata()) == [100.0, 132.5]
    panel.close()


def test_drift_panel_no_offsets_tag(qt):
    panel = _drift_panel(qt, provenance=False)
    ax = panel.axes[0]
    assert ax.get_title() == 'P · no offsets'
    points = _points(panel)
    assert list(points.y) == [100.0, 200.0, 120.0, 300.0]
    assert not points.muted.any()
    assert ax.get_ylabel() == 'depth (µm)'
    assert panel.hover_text(112.0, -1000.0) == 'D2 · no offsets · backend saves no offsets'
    panel.close()


def test_drift_panel_reads_backend_flag(qt, monkeypatch):
    monkeypatch.setattr(display, 'backend_spec',
                        lambda backend_id: contract_registry.get('functional_likelihood').spec)
    panel = _drift_panel(qt, provenance=False)
    assert panel.hover_text(112.0, -1000.0) == 'D2 · no offsets · run predates offsets'
    panel.close()


def test_drift_hover_range_on_varying_column(qt):
    registration = {'method': 'native', 'offsets_um': {'D1': 0.0, 'D2': 10.0},
                    'depth_scale': {'D1': 1.0, 'D2': 0.5}, 'diagnostics': {'P': {'groups': [['D1', 'D2']]}}}
    panel = _drift_panel(qt, registration, meta={'geometry': {'D2': np.array([[27.0, 0.0], [27.0, 100.0]])}})
    assert panel.hover_text(112.0, -1000.0) == 'D2 · -40.0 to +10.0 µm · native'
    b1 = panel.points.set_index('cellid').loc['b:1']
    assert (b1.x, b1.y) == (104.0, 70.0)
    assert panel.hover_text(104.0, 70.0) == 'b:1 · D2 · -50.0 µm · native'
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: electrodes alone set the depth axis when no unit has a depth (drift-view Task 4, fix round 2) | date: 2026-10-08]
def test_drift_panel_autoscales_electrodes_without_depths(qt):
    geometry = np.array([[11.0, 100.0], [11.0, 150.0], [11.0, 200.0]])
    panel = _drift_panel(qt, meta={'geometry': {'D1': geometry}}, table=_table().drop(columns='depth'))
    assert len(panel.points) == 0
    lo, hi = panel.axes[0].get_ylim()
    assert lo <= 100.0 and hi >= 200.0
    panel.close()
# [AGENT EDIT END]


def test_drift_panel_shows_unreadable_registration(qt):
    panel = _drift_panel(qt, {'method': 'matches', 'offsets_um': {'D1': 'x'},
                              'diagnostics': {'P': {'groups': [['D1']]}}})
    assert any(t.get_text().startswith('ValueError: registration of P') for t in panel.axes[0].texts)
    panel.close()


def test_open_run_draws_drift_unselected(qt, tmp_path, monkeypatch):
    real = fl.run_functional_likelihood
    registration = {'method': 'matches', 'offsets_um': {'D1': 0.0, 'D2': 12.5},
                    'diagnostics': {'P': {'groups': [['D1', 'D2']]}}}

    def registered(*args, **kwargs):
        result = real(*args, **kwargs)
        return dataclasses.replace(result, provenance=result.provenance.merged(
            extra={'registration': registration}))

    monkeypatch.setattr(fl, 'run_functional_likelihood', registered)
    folder = extracted_project(tmp_path, days=('D1', 'D2'))
    run = run_selection(Selection(Project.open(folder.parent, folder.name), ('D1', 'D2')), 'functional_likelihood')
    window = open_run(run)
    window.state.set_selection([])          # open_run selects the first unit; clear it
    qt.processEvents()
    assert not window.state.selection
    panel = window._docks['drift'].widget()
    assert panel.frame.anchor == 'D1'
    units = window.state.data.units()
    finite = units[np.isfinite(units.depth.astype(float))]
    assert sorted(panel.points.cellid) == sorted(finite.cellid)
    depth = dict(zip(finite.cellid, finite.depth.astype(float)))
    session = dict(zip(finite.cellid, finite.session_id))
    for row in panel.points.itertuples():
        assert row.y == depth[row.cellid] + {'D1': 0.0, 'D2': 12.5}[session[row.cellid]]
    window._save_layout = lambda: None
    window.close()
    window.deleteLater()
    qt.processEvents()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered unit positions and the 2D registered distance between two units (browse-same-day Task 1) | date: 2026-10-12]
def test_registered_positions_in_drift_frame():
    p = registered_positions(_table(), _prov(READS_GROUPS))
    assert p['depth'].to_dict() == {'a:1': 100.0, 'b:1': 132.5, 'c:1': 300.0, 'a:2': 200.0}
    assert p['x_um'].to_dict() == {'a:1': 11.0, 'b:1': 27.0, 'c:1': 43.0, 'a:2': 59.0}
    assert set(p['group']) == {'P'}
    assert round(unit_distance(p, 'a:1', 'b:1'), 4) == 36.225
    assert round(unit_distance(p, 'a:1', 'c:1'), 4) == 202.5438      # c:1 is unlinked: measured depth


def test_registered_positions_raw_without_registration():
    p = registered_positions(_table(), {})
    assert p.loc['b:1', 'depth'] == 120.0
    assert round(unit_distance(p, 'a:1', 'b:1'), 4) == 25.6125


def test_unit_distance_missing_is_nan():
    p = registered_positions(_table().drop(columns='x_um'), _prov(READS_GROUPS))
    assert p.loc['b:1', 'depth'] == 132.5
    assert math.isnan(unit_distance(p, 'a:1', 'b:1'))
    table = _table()
    table.loc[table.cellid == 'b:1', 'recording_group'] = 'Q'
    p = registered_positions(table, _prov(READS_GROUPS))
    assert p.loc['b:1', 'depth'] == 120.0
    assert math.isnan(unit_distance(p, 'a:1', 'b:1'))
    assert math.isnan(unit_distance(p, 'a:1', 'zz'))
    table = _table()
    table['depth'] = ['100', 'deep', 300.0, None]
    depths = registered_positions(table, _prov(READS_GROUPS))['depth'].to_dict()
    assert sorted(depths) == ['a:1', 'a:2', 'b:1', 'c:1']
    assert depths['a:1'] == 100.0 and depths['c:1'] == 300.0
    assert math.isnan(depths['b:1']) and math.isnan(depths['a:2'])


@pytest.mark.parametrize('provenance', [
    _prov({'method': 'matches', 'offsets_um': {'D1': 'x'}}),
    {'group_runs': {'P': 'x'}},
    _prov({'method': 'matches', 'offsets_um': {'D1': 5.0, 'D3': 0.0},
           'shank_offsets_um': {'D2': [10.0, 20.0]}, 'shank_width_um': 100.0, 'unit_shanks': {'b:1': 5},
           'diagnostics': {'P': {'groups': [['D1', 'D2'], ['D3']]}}}),
])
def test_unreadable_registration_leaves_group_without_depth(provenance):
    p = registered_positions(_table(), provenance)
    assert all(math.isnan(d) for d in p['depth'])
    assert list(p['x_um']) == [11.0, 27.0, 43.0, 59.0]


def test_registered_position_signatures():
    assert (str(inspect.signature(registered_positions))
            == '(units: pandas.core.frame.DataFrame, provenance) -> pandas.core.frame.DataFrame')
    assert (str(inspect.signature(unit_distance))
            == '(positions: pandas.core.frame.DataFrame, a: str, b: str) -> float')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a nullable Float64/Int64 coordinate with pd.NA is a missing value, not an abort (browse-same-day Task 1 fix round 1) | date: 2026-10-12]
def test_registered_positions_nullable_missing_coordinates():
    table = _table()
    table['depth'] = pd.array([100.0, 120.0, 300.0, None], dtype='Float64')
    table['x_um'] = pd.array([11, 27, None, 59], dtype='Int64')
    p = registered_positions(table, _prov(READS_GROUPS))
    assert p.loc['a:1', 'depth'] == 100.0 and p.loc['b:1', 'depth'] == 132.5
    assert math.isnan(p.loc['a:2', 'depth'])
    assert math.isnan(p.loc['c:1', 'x_um']) and p.loc['b:1', 'x_um'] == 27.0
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a left click on a unit dot emits that unit; pan, zoom, the right button, a double-click and empty space emit nothing (browse-same-day Task 4) | date: 2026-10-12]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the click lands only after the double-click interval; tag wraps the whole function (browse-same-day Task 6) | date: 2026-10-12]
def test_drift_click_emits_unit(qt):
    from typing import Optional  # noqa: F401  (the signature string names it)

    from matplotlib.backend_bases import MouseButton, MouseEvent

    from nems_lbhb.unit_match.viewer.panels.plots.drift import DriftPanel

    panel = _drift_panel(qt)
    clicked = []
    panel.unit_clicked.connect(clicked.append)

    def press(x, y, **kwargs):
        px, py = panel.axes[0].transData.transform((x, y))
        kwargs.setdefault('button', MouseButton.LEFT)
        ev = MouseEvent('button_press_event', panel.canvas, px, py, **kwargs)
        panel.canvas.callbacks.process('button_press_event', ev)
        _wait()

    px, py = panel.axes[0].transData.transform((104.0, 132.5))
    panel.canvas.callbacks.process('button_press_event',
                                   MouseEvent('button_press_event', panel.canvas, px, py, button=MouseButton.LEFT))
    assert clicked == []
    _wait()
    assert clicked == ['b:1']
    press(104.0, 220.0)
    assert clicked == ['b:1']
    press(104.0, 132.5, button=MouseButton.RIGHT)
    assert clicked == ['b:1']
    press(104.0, 132.5, dblclick=True)
    assert clicked == ['b:1']
    panel.tools.pan()
    press(104.0, 132.5)
    assert clicked == ['b:1']
    panel.tools.pan()
    press(0.0, 100.0)
    assert clicked == ['b:1', 'a:1']
    assert panel.unit_at(104.0, 132.5) == 'b:1'
    assert panel.unit_at(104.0, 220.0) is None
    assert panel.unit_at(-500.0, 0.0) is None
    assert panel.hover_text(104.0, 132.5) == 'b:1 · D2 · +12.5 µm · matches'
    assert str(inspect.signature(DriftPanel.unit_at)) == '(self, x: float, y: float) -> Optional[str]'
    panel.close()
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a dot click waits out the double-click interval, a double-click only resets, a stale click emits nothing, and a same-run same-group redraw keeps the zoom (browse-same-day Task 6) | date: 2026-10-12]
def _wait():
    """Wait out the double-click interval, and a margin."""
    from PySide6 import QtTest
    QtTest.QTest.qWait(QtWidgets.QApplication.doubleClickInterval() + 100)


def _two_group_table():
    """``_table()`` with c:1 in recording group ``'Q'``."""
    table = _table()
    table.loc[table.cellid == 'c:1', 'recording_group'] = 'Q'
    return table


def _press(panel, x, y, **kwargs):
    """A synthetic matplotlib press at data point ``(x, y)``, left button unless given."""
    from matplotlib.backend_bases import MouseButton, MouseEvent
    px, py = panel.axes[0].transData.transform((x, y))
    kwargs.setdefault('button', MouseButton.LEFT)
    panel.canvas.callbacks.process('button_press_event',
                                   MouseEvent('button_press_event', panel.canvas, px, py, **kwargs))


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the test checks the click timer is precise; tag wraps the whole function (browse-same-day Task 6 fix round 1) | date: 2026-10-12]
def test_drift_click_waits_out_double_click(qt):
    from PySide6 import QtTest

    from nems_lbhb.unit_match.viewer.qt import QtCore

    panel = _drift_panel(qt)
    assert panel._click_timer.timerType() == QtCore.Qt.PreciseTimer   # a coarse timer can fire ~5% early
    panel.resize(600, 400)
    panel.show()
    qt.processEvents()
    panel.refresh()
    qt.processEvents()
    resets = []
    panel.reset_view = lambda: resets.append(True)
    clicked = []
    panel.unit_clicked.connect(clicked.append)
    px, py = panel.axes[0].transData.transform((104.0, 132.5))
    dpr = panel.canvas.device_pixel_ratio
    point = QtCore.QPoint(round(px / dpr), round((panel.canvas.figure.bbox.height - py) / dpr))
    QtTest.QTest.mouseClick(panel.canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
    QtTest.QTest.mouseDClick(panel.canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
    _wait()
    assert clicked == []
    assert resets == [True]
    QtTest.QTest.mouseClick(panel.canvas, QtCore.Qt.LeftButton, QtCore.Qt.NoModifier, point)
    assert clicked == []
    _wait()
    assert clicked == ['b:1']
    panel.close()

    panel = _drift_panel(qt, table=_two_group_table())
    clicked = []
    panel.unit_clicked.connect(clicked.append)
    panel.state.set_selection(['c:1'])
    qt.processEvents()
    assert [(r.x, r.y) for r in panel.points.itertuples() if r.cellid == 'c:1'] == [(0.0, 300.0)]
    _press(panel, 0.0, 300.0)
    panel.state.set_selection(['a:1'])
    _wait()
    assert clicked == []
    assert panel.axes[0].get_title() == 'P'
    panel.close()
# [AGENT EDIT END]


def test_drift_zoom_kept_across_selection(qt):
    panel = _drift_panel(qt, table=_two_group_table())
    ax = panel.axes[0]
    state = panel.state

    def zoom():
        ax.set_xlim(-10.0, 10.0)
        ax.set_ylim(90.0, 150.0)

    def full_p():
        return ax.get_xlim() == (-20.0, 156.0) and ax.get_ylim() == (95.0, 205.0)

    def zoomed():
        return ax.get_xlim() == (-10.0, 10.0) and ax.get_ylim() == (90.0, 150.0)

    assert full_p()
    zoom()
    state.set_selection(['a:1'])
    state.set_selection(['a:1', 'b:1'])
    assert zoomed()
    panel.reset_view()
    assert full_p()
    zoom()
    state.set_selection(['c:1'])
    assert ax.get_xlim() == (-20.0, 20.0) and ax.get_ylim() == (283.5, 316.5)
    state.set_selection(['a:1'])
    assert full_p()
    zoom()
    panel.restyle()
    assert full_p()
    zoom()
    _press(panel, 0.0, 100.0, dblclick=True)
    assert full_p()
    zoom()
    other = _drift_panel(qt, table=_two_group_table())
    state.set_data(other.state.data)
    state.changed.emit()
    assert full_p()
    zoom()
    state.changed.emit()
    assert zoomed()
    other.close()
    panel.close()
# [AGENT EDIT END]
