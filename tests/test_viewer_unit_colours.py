# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Day ramp colours each selected unit by its site's rank by recording date; Light keeps its own ramp (graphite-palettes Task 5) | date: 2026-10-11]
"""Unit colours by recording day: ``RunData.site_order`` and the Day ramp in ``PlotPanel.selected``.

Every settings file a test makes lives under ``tmp_path`` (``theme_env``)."""
import os

import numpy as np
import pandas as pd
import pytest
from PySide6 import QtWidgets

from multiphy.viewer import display, unit_palettes
from multiphy.viewer.panels.plots.waveform import WaveformPanel
from multiphy.viewer.state import PairState

from .theme_fixtures import theme_env  # noqa: F401  (a fixture, used by name)
from .viewer_fakes import DictData

SITES = {'s2-01': 'S2', 's1-01': 'S1', 's3-01': 'S3', 's1-02': 'S1', 's0-01': 'S0'}
DATED = pd.DataFrame({'cellid': ['s2-01', 's1-01', 's3-01', 's1-02'], 'siteid': ['S2', 'S1', 'S3', 'S1'],
                      'session_date': ['2024-01-02', '2024-01-02', '2024-01-01', '2024-01-02']})
SELECTION = ['s2-01', 's1-01', 's3-01', 's1-02']


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _colours(data, selection):
    state = PairState()
    state.set_data(data)
    state.set_selection(selection)
    panel = WaveformPanel(state)
    out = [(c, colour) for c, _, colour in panel.selected()]
    panel.close()
    return out


def test_site_order_by_date_then_site():
    assert DictData(DATED, {}, {}, SITES).site_order() == ['S3', 'S1', 'S2']
    undated_first = pd.concat([pd.DataFrame({'cellid': ['s0-01'], 'siteid': ['S0'], 'session_date': [np.nan]}),
                               DATED], ignore_index=True)
    assert DictData(undated_first, {}, {}, SITES).site_order() == ['S3', 'S1', 'S2', 'S0']
    assert DictData(['s2-01', 's1-01', 's3-01'], {}, {}, SITES).site_order() == ['S1', 'S2', 'S3']


def test_day_ramp_colours_by_site_rank(qt, theme_env, monkeypatch):
    display.set_theme('graphite')
    monkeypatch.setattr(unit_palettes, '_palette', 'ramp')
    dated = DictData(DATED, {}, {}, SITES)
    by_rank = [('s2-01', '#F7EA92'), ('s1-01', '#FF8FA4'), ('s3-01', '#4E7CC8'), ('s1-02', '#FF8FA4')]
    assert _colours(dated, SELECTION) == by_rank
    one_site = _colours(DictData(['s1-01', 's1-02'], {}, {}, SITES), ['s1-01', 's1-02'])
    assert one_site == [('s1-01', '#4E7CC8'), ('s1-02', '#4E7CC8')]
    display.set_theme('dark')
    assert _colours(dated, SELECTION) == by_rank
    monkeypatch.setattr(unit_palettes, '_palette', 'tol')
    assert [colour for _, colour in _colours(dated, SELECTION)] == ['#77AADD', '#EE8866', '#EEDD88', '#FFAABB']


def test_light_ignores_day_ramp(qt, theme_env, monkeypatch):
    display.set_theme('light')
    monkeypatch.setattr(unit_palettes, '_palette', 'ramp')
    colours = [colour for _, colour in _colours(DictData(DATED, {}, {}, SITES), SELECTION)]
    assert colours == ['#1F6F4A', '#B5451F', '#3F6FA0', '#7A5C9E']
# [AGENT EDIT END]
