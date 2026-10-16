# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the simple plot panels read RunData and draw what the old viewers drew; the registry refuses a duplicate id (viewer-split Task 4) | date: 2026-10-06]
"""The registered plot panels over a ``PairState`` holding a mode's ``RunData``."""
import os

import numpy as np
import pytest
from PySide6 import QtWidgets

from multiphy.project.selection import selection_of_run
from multiphy.run.run_dir import load_result
from multiphy.viewer import registry
from multiphy.viewer.modes import MODES
from multiphy.viewer.panels import plots  # noqa: F401  (registers the plot panels)
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.state import PairState

from .viewer_fakes import DictData
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the old viewers and their comparison harness are gone; the runs are a plain project run of each mode (viewer-split Task 9) | date: 2026-10-07]
from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection

from .synthetic_day import extracted_project
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the deleted-canvas test builds a bare SafeCanvas and deletes its C++ object (day-table-keys Task 1) | date: 2026-10-14]
import sys

import shiboken6
from matplotlib.figure import Figure

from multiphy.viewer.panels.plots.base import SafeCanvas
# [AGENT EDIT END]


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an across-days run of D1, D2, D3 and a same-day run of D1, from extracted_project and run_selection (viewer-split Task 9) | date: 2026-10-07]
@pytest.fixture(scope='module')
def runs(tmp_path_factory):
    path = extracted_project(tmp_path_factory.mktemp('panels'), days=('D1', 'D2', 'D3'))
    project = Project.open(path.parent, path.name)
    return {'across_days': run_selection(Selection(project, ('D1', 'D2', 'D3')), 'functional_likelihood'),
            'same_day': run_selection(Selection(project, ('D1',)), 'functional_likelihood')}
# [AGENT EDIT END]


def _state(run):
    """A ``PairState`` holding the run's ``RunData``, loaded by its mode."""
    selection = selection_of_run(run)
    mode_id = 'same_day' if selection.same_day else 'across_days'
    state = PairState()
    state.set_data(MODES[mode_id].load(selection, selection.dataset(), load_result(run / 'result.json'),
                                       selection.day_extras()))
    return state


def test_register_panel_refuses_duplicate():
    before = dict(registry.PANELS)

    class Again(PlotPanel):
        panel_id = 'waveform'

    with pytest.raises(ValueError, match='already registered'):
        registry.register_panel(Again)
    assert registry.PANELS == before


def test_tuning_without_tuning_part(qt):
    state = PairState()
    state.set_data(DictData(['u:0'], {'u:0': {'ftc': np.array([1., 2., 1.])}}, {}, {'u:0': 'dayA'}))
    panel = registry.PANELS['tuning'](state)
    state.set_selection(['u:0'])
    qt.processEvents()          # the idle redraw runs while the panel lives
    assert [t.get_text() for t in panel.axes[0].texts] == ['no tone tuning']
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the probe panel has no registered box, before and after a selection (drift-view Task 3) | date: 2026-10-08]
def test_probe_has_no_registered_box(qt, runs):
    state = _state(runs['same_day'])
    panel = registry.PANELS['probe'](state)
    assert [b.text() for b in panel.findChildren(QtWidgets.QCheckBox)] == ['whole probe']
    state.set_selection([state.data.units().cellid.iloc[0]])
    qt.processEvents()          # the idle redraw runs while the panel lives
    assert [b.text() for b in panel.findChildren(QtWidgets.QCheckBox)] == ['whole probe']
    panel.close()
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the merged fingerprint and mixture panels draw what the old viewers drew, with dotted rules and one short mixture tag (viewer-split Task 5) | date: 2026-10-06]
def test_fingerprint_rules_between_conditions(qt):
    trace = {'fingerprints': {'A': np.ones(10), 'B': np.ones(10)}, 'trial_counts': {'A': 10, 'B': 10},
             'fingerprint_bin_seconds': .02}
    state = PairState()
    state.set_data(DictData(['u:0', 'v:0'], {'u:0': trace, 'v:0': trace}, {}, {'u:0': 'dayA', 'v:0': 'dayB'}))
    panel = registry.PANELS['fingerprint'](state)
    state.set_selection(['u:0', 'v:0'])
    qt.processEvents()          # the idle redraw runs while the panel lives
    rules = [ln for ln in panel.axes[0].get_lines() if ln.get_linestyle() == ':']
    assert len(rules) == 1
    assert list(rules[0].get_xdata()) == [0.2, 0.2]
    panel.close()


def test_mixture_tag_without_fit(qt, runs):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: select one unit of the day and one unit of each day, without a day pair, in place of the deleted harness selections (viewer-split Task 9) | date: 2026-10-07]
    for mode_id in ('same_day', 'across_days'):
        state = _state(runs[mode_id])
        panel = registry.PANELS['mixture'](state)
        units = state.data.units()
        chosen = [units.cellid.iloc[0]] if mode_id == 'same_day' else \
            units.groupby('siteid', sort=True).cellid.first().tolist()
        state.set_selection(chosen)
        qt.processEvents()          # the idle redraw runs while the panel lives
    # [AGENT EDIT END]
        assert all(not ax.axison for ax in panel.fig.axes), mode_id
        assert [t.get_text() for t in panel.axes[0].texts] == ['mixture: not fitted'], mode_id
        panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram are two panels, one plot each (browse-same-day Task 5) | date: 2026-10-12]
def test_isi_and_acg_panels_draw_one_plot_each(qt):
    trace = {'isi': np.full(40, 0.025), 'acg': np.arange(1.0, 6.0)}
    state = PairState()
    state.set_data(DictData(['u:0', 'v:0'], {'u:0': trace, 'v:0': trace}, {}, {'u:0': 'dayA', 'v:0': 'dayB'}))
    isi = registry.PANELS['isi'](state)
    acg = registry.PANELS['acg'](state)
    state.set_selection(['u:0', 'v:0'])
    qt.processEvents()          # the idle redraw runs while the panels live
    assert 'spiking' not in registry.PANELS
    assert isi.panel_title == 'Interval histogram'
    assert acg.panel_title == 'Autocorrelogram'
    assert len(isi.axes) == 1
    assert len(isi.axes[0].get_lines()) == 2
    assert isi.axes[0].get_title() == 'ISI'
    assert isi.axes[0].get_xscale() == 'log'
    assert len(acg.axes) == 1
    lines = acg.axes[0].get_lines()
    assert len(lines) == 3
    assert list(lines[0].get_xdata()) == [-4.5, -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 4.5]
    assert list(lines[0].get_ydata()) == [5.0, 4.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert list(lines[-1].get_xdata()) == [0, 0]
    assert acg.axes[0].get_title() == 'autocorr'
    isi.close()
    acg.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a redraw queued before the canvas was deleted is skipped, neither raised out of processEvents nor handed to sys.excepthook (day-table-keys Task 1) | date: 2026-10-14]
def test_deleted_canvas_skips_queued_redraw(qt, monkeypatch):
    canvas = SafeCanvas(Figure())
    canvas.draw_idle()                      # queues QTimer.singleShot(0, canvas._draw_idle)
    shiboken6.delete(canvas)
    hooked = []
    monkeypatch.setattr(sys, 'excepthook', lambda kind, exc, tb: hooked.append(exc))
    assert QtWidgets.QApplication.processEvents() is None
    assert hooked == []
# [AGENT EDIT END]
