# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: curation reads each unit's siteid, not a cellid prefix (Plan B Task 4) | date: 2026-09-28]
"""Curation counts days from each unit's siteid, so dated and dashed day names stay separate days."""
import inspect
import os
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match import clusters as cl
from nems_lbhb.unit_match import store as st


@pytest.fixture(scope='module')
def gui():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the GUI checks are required; a missing PySide6 fails instead of skipping (Plan B Task 4 fix round 1) | date: 2026-09-28]
    import PySide6.QtWidgets as widgets
    # [AGENT EDIT END]
    widgets.QApplication.instance() or widgets.QApplication([])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fixture imports the viewer App module; the gui module is deleted (viewer-split Task 9) | date: 2026-10-07]
    from nems_lbhb.unit_match.viewer import app as module
    # [AGENT EDIT END]
    return module


def _verdicts(rows):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: verdict frames carry no is_decoy (run-curation Task 5, D20) | date: 2026-10-05]
    return pd.DataFrame([dict(cellid_a=a, cellid_b=b, siteid_a=sa, siteid_b=sb, verdict=v)
                         for a, b, sa, sb, v in rows])
    # [AGENT EDIT END]


def test_describe_counts_days_by_siteid():
    members = ['2026-01-01:0', '2026-01-02:0']
    info = cl.describe(members, {}, {m: m.split(':')[0] for m in members})
    assert info['n_days'] == 2
    assert info['same_day_collision'] is False
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the sites map decides, not the cellid prefix (Plan B Task 4 fix round 1) | date: 2026-09-28]
    apart = cl.describe(['x:0', 'x:1'], {}, {'x:0': 'D1', 'x:1': 'D2'})
    assert apart['n_days'] == 2 and apart['same_day_collision'] is False
    together = cl.describe(['a:0', 'b:0'], {}, {'a:0': 'S', 'b:0': 'S'})
    assert together['n_days'] == 1 and together['same_day_collision'] is True
    # [AGENT EDIT END]


def test_describe_refuses_missing_site():
    with pytest.raises(ValueError, match='no siteid for'):
        cl.describe(['2026-01-01:0', '2026-01-02:0'], {}, {'2026-01-01:0': '2026-01-01'})
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a map of the right length with a wrong key still refuses (Plan B Task 4 fix round 1) | date: 2026-09-28]
    with pytest.raises(ValueError, match=r"no siteid for \['2026-01-02:0'\]"):
        cl.describe(['2026-01-01:0', '2026-01-02:0'], {},
                    {'2026-01-01:0': '2026-01-01', 'other:0': '2026-01-02'})
    # [AGENT EDIT END]


def test_chains_count_sites_from_rows():
    _, summary = st.chains(_verdicts([('D-1:0', 'D-2:0', 'D-1', 'D-2', 'accept')]))
    assert summary.n_sites.tolist() == [2]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the row siteids decide, not the cellid prefix (Plan B Task 4 fix round 1) | date: 2026-09-28]
    _, same = st.chains(_verdicts([('A-1:0', 'B-1:0', 'S', 'S', 'accept')]))
    assert same.n_sites.tolist() == [1]
    _, apart = st.chains(_verdicts([('X-1:0', 'X-1:1', 'D1', 'D2', 'accept')]))
    assert apart.n_sites.tolist() == [2]


def test_chains_refuse_missing_row_sites():
    rows = _verdicts([('D-1:0', 'D-2:0', 'D-1', 'D-2', 'accept'),
                      ('D-1:0', 'D-3:0', 'D-1', None, 'reject')])
    with pytest.raises(ValueError, match='missing for: D-1:0/D-3:0'):
        st.chains(rows)
    with pytest.raises(ValueError, match='missing for: D-1:0/D-3:0'):
        st.inconsistencies(rows)
    bare = rows.drop(columns=['siteid_a', 'siteid_b']).iloc[:0]
    with pytest.raises(ValueError, match='pair rows need siteid_a and siteid_b'):
        st.chains(bare)
    with pytest.raises(ValueError, match='pair rows need siteid_a and siteid_b'):
        st.inconsistencies(bare)
    # [AGENT EDIT END]


def test_inconsistencies_use_row_sites():
    bad = st.inconsistencies(_verdicts([
        ('D-1:0', 'D-2:0', 'D-1', 'D-2', 'accept'),
        ('D-2:0', 'D-3:0', 'D-2', 'D-3', 'accept'),
        ('D-1:0', 'D-3:0', 'D-1', 'D-3', 'reject')]))
    assert len(bad) == 1
    assert bad.problem.iloc[0] == 'closing edge rejected'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: two ends on one row site are not a triangle, whatever their prefixes (Plan B Task 4 fix round 1) | date: 2026-09-28]
    same_day = st.inconsistencies(_verdicts([
        ('A:0', 'B:0', 'S1', 'S2', 'accept'),
        ('B:0', 'C:0', 'S2', 'S1', 'accept'),
        ('A:0', 'C:0', 'S1', 'S1', 'reject')]))
    assert len(same_day) == 0
    # [AGENT EDIT END]


def test_unit_label_strips_siteid():
    assert cl.unit_label('D1:0', 'D1') == '0'
    assert cl.unit_label('PRN067a-001-1', 'PRN067a') == '001-1'
    assert cl.unit_label('2026-01-01:3', '2026-01-01') == '3'
    assert cl.unit_label('x', 'y') == 'x'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: only the supplied siteid is stripped (Plan B Task 4 fix round 1) | date: 2026-09-28]
    assert cl.unit_label('D1-0', 'D2') == 'D1-0'
    assert cl.unit_label('D1x:0', 'D1') == 'D1x:0'
    assert cl.unit_label('D1_0', 'D1') == 'D1_0'
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the label and tuning checks build a PairState over DictData and the new plot panels (viewer-split Task 4) | date: 2026-10-06]
@pytest.fixture(scope='module')
def viewer():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    import PySide6.QtWidgets as widgets
    widgets.QApplication.instance() or widgets.QApplication([])
    from nems_lbhb.unit_match.viewer import registry, state
    from nems_lbhb.unit_match.viewer.panels import plots  # noqa: F401  (registers the plot panels)
    return SimpleNamespace(PANELS=registry.PANELS, PairState=state.PairState, QApplication=widgets.QApplication)


def _panel(viewer, panel_id, units, meta=None, sites=None, arrays=None):
    """A ``panel_id`` panel over a PairState holding ``DictData`` of the dicts, with ``units`` selected."""
    from .viewer_fakes import DictData
    sites = sites if sites is not None else {u: u.split(':')[0] for u in units}
    state = viewer.PairState()
    state.set_data(DictData(list(units), arrays or {}, meta or {}, sites))
    panel = viewer.PANELS[panel_id](state)
    state.set_selection(list(units))
    # the selection's idle redraw runs now, while the panel lives; a pending one would fire on a deleted canvas
    viewer.QApplication.processEvents()
    return panel


def test_label_for_reads_state_sites(viewer):
    two = _panel(viewer, 'waveform', ['2026-01-01:0', '2026-01-01:1'])
    assert two.label_for('2026-01-01:0') == '2026-01-01 0'
    one = _panel(viewer, 'waveform', ['2026-01-01:0'])
    assert one.label_for('2026-01-01:0') == '2026-01-01'
    shared = _panel(viewer, 'waveform', ['p:0', 'q:0'], sites={'p:0': 'dayA', 'q:0': 'dayA'})
    assert shared.label_for('p:0') == 'dayA p:0'
    split_days = _panel(viewer, 'waveform', ['p:0', 'p:1'], sites={'p:0': 'dayA', 'p:1': 'dayB'})
    assert split_days.label_for('p:0') == 'dayA'


def test_tuning_panel_without_freq_axis(viewer):
    panel = _panel(viewer, 'tuning', ['2026-01-01:0'], meta={})
    assert [t.get_text() for t in panel.axes[0].texts] == ['no tone tuning']
    tuned = _panel(viewer, 'tuning', ['u:0'], meta={'freq_axis': {'dayA': [500., 1000., 2000.]}},
                   sites={'u:0': 'dayA'}, arrays={'u:0': {'ftc': np.array([1., 2., 1.])}})
    assert [t.get_text() for t in tuned.axes[0].texts] == []
    assert len(tuned.axes[0].lines) >= 1
# [AGENT EDIT END]


def test_unit_sites_refuses_missing_siteid():
    units = pd.DataFrame({'cellid': ['a:0', 'b:0'], 'siteid': ['a', np.nan]})
    with pytest.raises(ValueError, match='missing:'):
        cl.unit_sites(units)
    assert cl.unit_sites(units.iloc[:1]) == {'a:0': 'a'}
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an absent siteid column refuses, even for an empty table (Plan B Task 4 fix round 1) | date: 2026-09-28]
    with pytest.raises(ValueError, match='missing: a:0, b:0'):
        cl.unit_sites(units.drop(columns=['siteid']))
    with pytest.raises(ValueError, match='unit table needs a siteid for every unit · missing:'):
        cl.unit_sites(pd.DataFrame({'cellid': []}))
    with pytest.raises(ValueError, match='missing: b:0'):
        cl.unit_sites(pd.DataFrame({'cellid': ['a:0', 'b:0'], 'siteid': ['a', '  ']}))
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the quiet Qt curator is removed; the site checks call data.display_pairs (viewer-split Task 2) | date: 2026-10-06]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the pair table is prepared by the Qt-free data.display_pairs (viewer-split Task 2) | date: 2026-10-06]
def test_display_pairs_needs_every_site():
    from nems_lbhb.unit_match import backends
    from nems_lbhb.unit_match.viewer import data
    units = pd.DataFrame({'cellid': ['a-01', 'b-01'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'], 'p_global': [.99]})
    result = backends.result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', run_id='r1')
    with pytest.raises(ValueError, match='no siteid for 1 units'):
        data.display_pairs(result, {'a-01': 'a'})
    with pytest.raises(ValueError, match='no siteid for 1 units: b-01'):
        data.display_pairs(result, {'a-01': 'a', 'z-01': 'b'})


def test_display_pairs_fills_missing_pair_sites():
    from nems_lbhb.unit_match import backends
    from nems_lbhb.unit_match.viewer import data
    units = pd.DataFrame({'cellid': ['a-01', 'b-01', 'c-01'], 'su_type': ['SU', 'SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['a-01', 'a-01'], 'cellid_b': ['b-01', 'c-01'], 'p_global': [.99, .98]})
    evidence = pd.DataFrame({'cellid_a': ['a-01', 'a-01'], 'cellid_b': ['b-01', 'c-01'],
                             'siteid_a': ['dayA', None], 'siteid_b': ['dayB', '']})
    result = backends.result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', run_id='r1')
    sites = {'a-01': 'dayA', 'b-01': 'dayB', 'c-01': 'dayC'}
    shown = data.display_pairs(result, sites, pair_evidence=evidence)
    got = shown.set_index(['cellid_a', 'cellid_b'])[['siteid_a', 'siteid_b']]
    assert got.loc[('a-01', 'b-01')].tolist() == ['dayA', 'dayB']
    assert got.loc[('a-01', 'c-01')].tolist() == ['dayA', 'dayC']
    day_pairs = data.complete_day_pairs([], shown)
    assert {frozenset((d['siteid_a'], d['siteid_b'])) for d in day_pairs} == {
        frozenset(('dayA', 'dayB')), frozenset(('dayA', 'dayC'))}
    assert all(d['days_apart'] == 'unknown' for d in day_pairs)
# [AGENT EDIT END]


def test_curation_signatures(gui):
    assert 'sites' in inspect.signature(cl.describe).parameters
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the window is the viewer's App; its setter takes one RunData, whose units carry their siteids (viewer-split Task 7) | date: 2026-10-06]
    from nems_lbhb.unit_match.viewer.app import App
    assert not hasattr(cl, 'site_of')
    assert not hasattr(cl, 'export')
    assert not hasattr(App, 'open_pair_anywhere')
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the exact specified interfaces (Plan B Task 4 fix round 1) | date: 2026-09-28]
    describe = inspect.signature(cl.describe).parameters
    assert list(describe) == ['members', 'lookup', 'sites', 'threshold']
    assert describe['sites'].default is inspect.Parameter.empty
    assert describe['threshold'].default == cl.DEFAULT_THRESHOLD
    unit_sites = inspect.signature(cl.unit_sites)
    assert list(unit_sites.parameters) == ['units']
    assert unit_sites.parameters['units'].annotation is pd.DataFrame and unit_sites.return_annotation is dict
    unit_label = inspect.signature(cl.unit_label)
    assert [(n, p.annotation) for n, p in unit_label.parameters.items()] == [('cellid', str), ('siteid', str)]
    assert unit_label.return_annotation is str
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: App.set_data takes the run data and a keyword-only curation folder (viewer-split Task 7) | date: 2026-10-06]
    setter = inspect.signature(App.set_data).parameters
    assert list(setter) == ['self', 'data', 'curation_dir']
    assert setter['curation_dir'].kind is inspect.Parameter.KEYWORD_ONLY
    assert setter['curation_dir'].default is None
    # [AGENT EDIT END]
    # [AGENT EDIT END]
# [AGENT EDIT END]
