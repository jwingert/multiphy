# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the App window, from gui.py:1898-2710 (gui.Curator) with set_data in place of set_match_result (gui.py:2214-2362), its plots from the panel registry, its tables built with the mode's words and one Actions, and the per-mode layout file of multiday_viewer.py:404-429 (viewer-split Task 7) | date: 2026-10-06]
"""``App``: one window that shows any mode's ``RunData``.

Every panel is a dock: drag, resize, float, or stack into tabs. The default arrangement is a 2x2
plot grid in the middle, browsing on the left and the verdict with its supporting tables on the
right. The layout is saved to the mode's layout file on close, restored next time, and can be
reset from the View menu.
"""
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the command line and the import's selection hash (viewer-split Task 8) | date: 2026-10-06]
import argparse
import dataclasses
import hashlib
# [AGENT EDIT END]
import contextlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from nems_lbhb.unit_match import clusters as cl
from nems_lbhb.unit_match import curation
from nems_lbhb.unit_match import store as st
from nems_lbhb.unit_match.contract import names
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: import_match_result declares its MatchResult return (viewer-split Task 8 fix round 1) | date: 2026-10-07]
from nems_lbhb.unit_match.contract.results import MatchResult
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project run's selection, result, run list, layout path and mode (viewer-split Task 8) | date: 2026-10-06]
from nems_lbhb.unit_match import backends, paths
from nems_lbhb.unit_match.project.selection import SELECTION_FILE, Selection, selection_of_run
from nems_lbhb.unit_match.run.project_run import selection_runs
from nems_lbhb.unit_match.viewer.modes import MODES
# [AGENT EDIT END]
from nems_lbhb.unit_match.run import catalog
from nems_lbhb.unit_match.run.catalog import RunEntry
from nems_lbhb.unit_match.viewer import display
from nems_lbhb.unit_match.viewer.data import Mode, RunData
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: cells carry theme tokens; the window restyles its plots and tables on a theme switch (launcher-gui Task 8) | date: 2026-10-07]
from nems_lbhb.unit_match.viewer import theme
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: main sets the saved unit palette (graphite-palettes Task 4) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer import unit_palettes
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.panels.plots.base import PlotPanel
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the Settings and Help menus (launcher-gui Task 9) | date: 2026-10-07]
from nems_lbhb.unit_match.viewer import menus
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: main sets the logo icon (logo) | date: 2026-10-13]
from nems_lbhb.unit_match.viewer import logo
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one saved key per action (day-table-keys Task 4) | date: 2026-10-14]
from nems_lbhb.unit_match.viewer import keys
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.panels import plots  # noqa: F401  (registers the plot panels)
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table replaces the candidates table, so no candidate cells are built here (native-same-day Task 10) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer.panels.tables import (BrowsePanel, ChainPanel, ClusterPanel, DayTablePanel,
                                                       EvidencePanel, RivalPanel)
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.qt import QtCore, QtGui, QtWidgets
from nems_lbhb.unit_match.viewer.registry import PANELS
from nems_lbhb.unit_match.viewer.state import Actions, PairState

# the panel ids the default layout places; any other registered plot is tabbed with the fingerprint
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the default layout places the drift panel in its own dock (drift-view Task 4) | date: 2026-10-08]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram take the spiking dock's place (browse-same-day Task 5) | date: 2026-10-12]
LAYOUT_PLOTS = ('waveform', 'footprint', 'isi', 'acg', 'fingerprint', 'tuning', 'probe', 'wfmap', 'mixture', 'drift')
# [AGENT EDIT END]
# [AGENT EDIT END]
# a layout file that cannot be read, or holds the wrong shape, leaves the default layout
LAYOUT_ERRORS = (OSError, ValueError, KeyError, TypeError, AttributeError)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: raise a Qt slot exception at the end of a block instead of letting PySide6 send it to sys.excepthook (viewer-split Task 7, fix round 1) | date: 2026-10-06]
@contextlib.contextmanager
def _raising_slot_errors():
    """Collect the exceptions Qt slots raise inside the block (PySide6 hands them to ``sys.excepthook``, not to
    the emitter) and raise the first one when the block ends."""
    caught = []
    hook = sys.excepthook
    sys.excepthook = lambda kind, exc, tb: caught.append(exc)
    try:
        yield
    finally:
        sys.excepthook = hook
    if caught:
        raise caught[0]
# [AGENT EDIT END]


class App(QtWidgets.QMainWindow):
    """The viewer window of one ``Mode``: ``set_data`` shows a run, curated through its curation folder or
    inspect-only."""

    def __init__(self, mode: Mode, *, layout_file: Path, selection: Optional[Selection] = None):
        super().__init__()
        self.setWindowTitle('Cross-day unit match curation')
        self.resize(1500, 860)
        self.mode = mode
        self.selection = selection
        self._layout_file = Path(layout_file)
        self.curation = None
        self.match_result = None
        self.state = PairState()
        self.cluster_lookup = {}
        self.review_threshold = cl.DEFAULT_THRESHOLD
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the selection's (dataset, day extras), read on first use; the run list shows only with a selection (viewer-split Task 8) | date: 2026-10-06]
        self._inputs = None
        # saved-run list state
        # [AGENT EDIT END]
        self._run_entries = []
        self._imported_runs = []
        self._rejected_runs = set()
        self._run_items = []
        self._shown_key = None
        self.setDockNestingEnabled(True)
        self.setDockOptions(QtWidgets.QMainWindow.AnimatedDocks |
                            QtWidgets.QMainWindow.AllowNestedDocks |
                            QtWidgets.QMainWindow.AllowTabbedDocks |
                            QtWidgets.QMainWindow.GroupedDragging)
        self.setTabPosition(QtCore.Qt.AllDockWidgetAreas, QtWidgets.QTabWidget.North)
        self._docks = {}
        self._build()
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project window lists its selection's runs (viewer-split Task 8) | date: 2026-10-06]
        if self.selection is not None:
            self._use_run_list()
        # [AGENT EDIT END]
        self.state.changed.connect(self._selection_changed)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a live theme switch restyles the window (launcher-gui Task 8) | date: 2026-10-07]
        theme.notifier().changed.connect(self.restyle)
        # [AGENT EDIT END]
        self._restore_layout()
        # a new window has no run, so it starts inspect-only with its controls disabled
        self._update_controls()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: restyle redraws the plots and repaints the tables in the active theme without refilling a table or emitting a change (launcher-gui Task 8) | date: 2026-10-07]
    def restyle(self) -> None:
        """Redraw every plot panel and repaint every table in the active theme.

        Nothing is refilled and ``state.changed`` is not emitted, so the selection and unsaved notes stay.
        """
        for d in self._docks.values():
            panel = d.widget()
            if isinstance(panel, PlotPanel):
                panel.restyle()
        for table in self.findChildren(QtWidgets.QTableWidget):
            table.viewport().update()
    # [AGENT EDIT END]

    # ------------------------------------------------------------ construction
    def _actions(self) -> Actions:
        """The window calls the table panels may make."""
        return Actions(
            select_pair=self.select_pair, set_verdict=self.set_verdict, curation=lambda: self.curation,
            track_state=self.track_state, take_over=self.take_over, update_controls=self._update_controls,
            cluster_lookup=lambda: self.cluster_lookup, review_threshold=lambda: self.review_threshold,
            merge_refusal=lambda members: self.mode.merge_refusal(self.state.data, members))

    def _dock(self, panel, area, title=None):
        d = QtWidgets.QDockWidget(title or panel.panel_title, self)
        d.setObjectName(f'um_{panel.panel_id}_dock')
        d.setWidget(panel)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the dock body paints the style sheet's rounded card, its content inset from the card's edge (graphite-palettes Task 3) | date: 2026-10-11]
        panel.setAttribute(QtCore.Qt.WA_StyledBackground, True)
        if panel.layout() is not None:
            panel.layout().setContentsMargins(8, 6, 8, 8)
        # [AGENT EDIT END]
        d.setAllowedAreas(QtCore.Qt.AllDockWidgetAreas)
        d.setFeatures(QtWidgets.QDockWidget.DockWidgetClosable |
                      QtWidgets.QDockWidget.DockWidgetFloatable |
                      QtWidgets.QDockWidget.DockWidgetMovable)
        self.addDockWidget(area, d)
        self._docks[panel.panel_id] = d
        self._view_menu.addAction(d.toggleViewAction())
        return d

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table built and wired into the Browse column; the pair-list and candidate connections removed (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Browse is the front tab of the default layout; a drift-dot click adds the unit to the selection or takes it out, through the day table; tag wraps the whole function (browse-same-day Task 4 fix round 1) | date: 2026-10-12]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram docks take the spiking dock's place, tabbed together, the interval histogram in front; tag wraps the whole function (browse-same-day Task 5) | date: 2026-10-12]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one shortcut per keys.ACTIONS entry, keyed from the saved keys and rebound on a save; the fixed A R U, undo, redo and Up / Down shortcuts removed; tag wraps the whole function (day-table-keys Task 4) | date: 2026-10-14]
    def _build(self):
        # no central widget: every panel is a dock, so all of them can be moved
        spacer = QtWidgets.QWidget()
        spacer.setMaximumSize(0, 0)
        self.setCentralWidget(spacer)

        menu = self.menuBar()
        self._view_menu = menu.addMenu('&View')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Settings and Help follow View (launcher-gui Task 9) | date: 2026-10-07]
        menus.install_menus(self)
        # [AGENT EDIT END]
        self.header = QtWidgets.QLabel('')
        self.header.setStyleSheet('font-size:14px; font-weight:600; padding:4px 10px;')
        tb = QtWidgets.QToolBar('header')
        tb.setObjectName('um_header_toolbar')
        tb.setMovable(False)
        tb.addWidget(self.header)
        # one minimal backend label; tracker, score and run id on hover
        stretch = QtWidgets.QWidget()
        stretch.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        tb.addWidget(stretch)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run's evidence tags sit before the backend label, amber, hover = reasons (event-optional Task 12) | date: 2026-10-10]
        self.evidence_bar = QtWidgets.QWidget()
        self._evidence_line = QtWidgets.QHBoxLayout(self.evidence_bar)
        self._evidence_line.setContentsMargins(0, 0, 0, 0)
        self._evidence_line.setSpacing(8)
        self._evidence_labels = []
        self.evidence_bar.setVisible(False)
        tb.addWidget(self.evidence_bar)
        # [AGENT EDIT END]
        self.backend_header = QtWidgets.QLabel('')
        self.backend_header.setStyleSheet('font-size:14px; padding:4px 10px;')
        tb.addWidget(self.backend_header)
        self.addToolBar(QtCore.Qt.TopToolBarArea, tb)

        words, actions = self.mode.words, self._actions()
        self.browse = BrowsePanel(self.state, words)
        self.tracked = ClusterPanel(self.state, actions, words)
        self.days = DayTablePanel(self.state, actions, self.mode.curates)
        self.browse.set_days(self.days)
        self.rivals = RivalPanel(self.state, actions)
        self.evidence = EvidencePanel(self.state, actions)
        self.chains = ChainPanel(self.state)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an Export button under the tracked panel's buttons when the mode exports, from multiday_viewer.py:295-302 (viewer-split Task 8) | date: 2026-10-06]
        self.export_btn = None
        if self.mode.exports:
            self.export_btn = QtWidgets.QPushButton('Export')
            self.export_btn.setToolTip('accepted tracks → csv')
            self.export_btn.setEnabled(False)
            self.export_btn.clicked.connect(self.export_accepted)
            layout = self.tracked.layout()
            grid = next(item.layout() for item in (layout.itemAt(i) for i in range(layout.count()))
                        if item.layout() is not None and item.layout().indexOf(self.tracked.buttons['Merge']) >= 0)
            grid.addWidget(self.export_btn, grid.rowCount(), 0)
        # [AGENT EDIT END]
        ids = list(self.mode.plots) + [i for i in PANELS if i not in self.mode.plots]
        self.plots = [PANELS[i](self.state) for i in ids]

        left = self._dock(self.browse, QtCore.Qt.LeftDockWidgetArea)
        left.setMinimumWidth(300)

        # the 2x2 plot grid
        p = {x.panel_id: x for x in self.plots}
        d_wave = self._dock(p['waveform'], QtCore.Qt.RightDockWidgetArea)
        d_foot = self._dock(p['footprint'], QtCore.Qt.RightDockWidgetArea)
        d_isi = self._dock(p['isi'], QtCore.Qt.RightDockWidgetArea)
        d_acg = self._dock(p['acg'], QtCore.Qt.RightDockWidgetArea)
        d_fp = self._dock(p['fingerprint'], QtCore.Qt.RightDockWidgetArea)
        d_tun = self._dock(p['tuning'], QtCore.Qt.RightDockWidgetArea)
        d_prb = self._dock(p['probe'], QtCore.Qt.RightDockWidgetArea)
        d_wfm = self._dock(p['wfmap'], QtCore.Qt.RightDockWidgetArea)
        d_mix = self._dock(p['mixture'], QtCore.Qt.RightDockWidgetArea)
        self.splitDockWidget(left, d_wave, QtCore.Qt.Horizontal)
        self.splitDockWidget(d_wave, d_foot, QtCore.Qt.Horizontal)   # top row
        self.splitDockWidget(d_wave, d_isi, QtCore.Qt.Vertical)      # bottom row
        self.splitDockWidget(d_isi, d_fp, QtCore.Qt.Horizontal)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the drift panel gets its own dock beside the footprint and probe pane (drift-view Task 4) | date: 2026-10-08]
        # split before the footprint becomes a tab group: Qt adds a dock split from a tabbed one as another tab
        d_drf = self._dock(p['drift'], QtCore.Qt.RightDockWidgetArea)
        self.splitDockWidget(d_foot, d_drf, QtCore.Qt.Horizontal)
        # [AGENT EDIT END]
        self.tabifyDockWidget(d_foot, d_prb)    # the probe sits behind the footprint
        self.tabifyDockWidget(d_foot, d_wfm)
        self.tabifyDockWidget(d_fp, d_tun)      # tuning and the mixture share a pane
        self.tabifyDockWidget(d_fp, d_mix)
        self.tabifyDockWidget(d_isi, d_acg)     # the autocorrelogram sits behind the interval histogram
        for panel in self.plots:                # any other registered plot shares that pane too
            if panel.panel_id not in LAYOUT_PLOTS:
                self.tabifyDockWidget(d_fp, self._dock(panel, QtCore.Qt.RightDockWidgetArea))
        d_isi.raise_()
        d_fp.raise_()
        d_foot.raise_()

        d_ev = self._dock(self.evidence, QtCore.Qt.RightDockWidgetArea)
        self.splitDockWidget(d_foot, d_ev, QtCore.Qt.Horizontal)
        d_riv = self._dock(self.rivals, QtCore.Qt.RightDockWidgetArea)
        self.splitDockWidget(d_ev, d_riv, QtCore.Qt.Vertical)
        d_ch = self._dock(self.chains, QtCore.Qt.RightDockWidgetArea)
        self.tabifyDockWidget(d_riv, d_ch)
        d_riv.raise_()
        for d in (d_ev, d_riv):
            d.setMinimumWidth(330)

        # tabbed onto the browser only after every split is in place, or the
        # splits operate on a tab group and push the other docks off-screen
        d_trk = self._dock(self.tracked, QtCore.Qt.LeftDockWidgetArea)
        self.tabifyDockWidget(left, d_trk)
        left.raise_()                      # Browse is the default way in

        QtCore.QTimer.singleShot(0, lambda: self.resizeDocks(
            [left, d_ev], [360, 380], QtCore.Qt.Horizontal))

        self._view_menu.addSeparator()
        self._view_menu.addAction('Reset layout', self._reset_layout)
        self._view_menu.addAction('Save layout now', self._save_layout)

        self.browse.import_btn.clicked.connect(self.import_result_dialog)
        self.days.picked.connect(self.tracked.show_track_of)
        self.tracked.track_shown.connect(self.days.follow_track)
        self.days.curated.connect(self._day_curated)
        p['drift'].unit_clicked.connect(self.days.toggle)

        # one shortcut per action; _bind_keys gives each its saved key, and an unbound one (no key) never fires
        days = self.days
        slots = {'up': lambda: self._step(-1), 'down': lambda: self._step(1),
                 'left': lambda: days.move_day(-1), 'right': lambda: days.move_day(1),
                 'track': days.string_track, 'clear': days.clear_string, 'skip': days.drop_day,
                 'accept': lambda: self.set_verdict('accept'), 'reject': lambda: self.set_verdict('reject'),
                 'unsure': lambda: self.set_verdict('unsure'), 'merge': days.merge, 'accept_track': days.accept,
                 'reject_track': days.reject, 'unmatch': days.unmatch_unit,
                 'undo': lambda: self._undo_redo('undo'), 'redo': lambda: self._undo_redo('redo')}
        self.shortcuts = {}
        for aid, _label, _default, _hover in keys.ACTIONS:
            shortcut = QtGui.QShortcut(QtGui.QKeySequence(), self)
            shortcut.activated.connect(slots[aid])
            self.shortcuts[aid] = shortcut
        # the writing shortcuts are kept so _update_controls can disable them
        self.verdict_shortcuts = [self.shortcuts[a] for a in ('accept', 'reject', 'unsure')]
        self.undo_shortcuts = [self.shortcuts[a] for a in ('undo', 'redo')]
        self.day_shortcuts = [self.shortcuts[a] for a in ('merge', 'accept_track', 'reject_track', 'unmatch')]
        self._default_state = None
        self._bind_keys()
        # a bound method of the window, so the connection ends with it
        keys.notifier().changed.connect(self._bind_keys)
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each shortcut takes its action's saved key; a save or reset rebinds every window (day-table-keys Task 4) | date: 2026-10-14]
    def _bind_keys(self) -> None:
        """Give each action's shortcut its key from ``keys.bindings()``; an unbound action (``''``) gets no key, so
        its shortcut never fires."""
        for aid, key in keys.bindings().items():
            self.shortcuts[aid].setKey(QtGui.QKeySequence(key))
    # [AGENT EDIT END]

    # ------------------------------------------------------------------ layout
    def _save_layout(self):
        """Write the layout beside the file as ``<name>.tmp``, then replace the file, so no reader sees a partial one."""
        self._layout_file.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._layout_file.with_name(self._layout_file.name + '.tmp')
        tmp.write_text(json.dumps(dict(geometry=bytes(self.saveGeometry().toBase64()).decode(),
                                       state=bytes(self.saveState().toBase64()).decode(),
                                       docks=sorted(self._docks)), indent=2))
        os.replace(tmp, self._layout_file)

    def _restore_layout(self):
        """Restore the saved layout; a missing, unreadable or wrong-shape file, or one saved with other docks,
        leaves the default."""
        self._default_state = self.saveGeometry(), self.saveState()
        try:
            saved = json.loads(self._layout_file.read_text())
            if sorted(saved.get('docks', [])) != sorted(self._docks):
                return
            geometry = QtCore.QByteArray.fromBase64(saved['geometry'].encode())
            state = QtCore.QByteArray.fromBase64(saved['state'].encode())
        except LAYOUT_ERRORS:
            return
        self.restoreGeometry(geometry)
        self.restoreState(state)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Reset layout fits Browse to the day columns again; tag wraps the whole function (day-table widths) | date: 2026-10-13]
    def _reset_layout(self):
        if self._default_state:
            self.restoreGeometry(self._default_state[0])
            self.restoreState(self._default_state[1])
            QtCore.QTimer.singleShot(0, self, self._fit_days)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Browse opens wide enough for up to five day columns at their narrowest; more days scroll (day-table widths) | date: 2026-10-13]
    def _fit_days(self) -> None:
        """Size the Browse dock so the day table shows its first ``min(days, FIT_DAYS)`` day columns at their
        narrowest; more days scroll. The window never resizes: Browse stops where the other docks, or the
        Tracked units dock tabbed with it, reach their minimum widths. A floating or hidden Browse, or a run
        with no days, keeps its size.

        ``resizeDocks`` moves a dock only part of the way, so it repeats, with a layout pass each time, until
        the width lands or stops changing (at most 20 times)."""
        browse = self._docks['browse']
        width = self.days.fit_width()
        if not width or browse.isFloating() or not self.days.isVisible():
            return
        target = browse.width() + width - self.days.scroll.viewport().width()
        for _ in range(20):
            before = browse.width()
            self.resizeDocks([browse], [target], QtCore.Qt.Horizontal)
            self.layout().activate()
            if browse.width() in (target, before):
                return
    # [AGENT EDIT END]

    def closeEvent(self, event):
        # a failed layout save still releases the run's lock
        try:
            self._save_layout()
        finally:
            if self.curation is not None:
                self.curation.close()
                self.curation = None
        super().closeEvent(event)

    # --------------------------------------------------------------- selection
    def _selection_changed(self):
        row = self.state.row
        if self.state.data is None:
            self.header.setText('Selected units: ' + ', '.join(self.state.selection)
                                if self.state.selection else 'No units selected')
        else:
            self.header.setText(self.mode.header(self.state.data, self.state.selection, row))
        if row is None:
            return
        prev = self._verdict_of(row.cellid_a, row.cellid_b)
        self.evidence.notes.setPlainText(prev['notes'] if prev else '')
        self.evidence.status.setText(
            f"already curated: {prev['verdict']} by {prev['curator']}" if prev else 'not yet curated')

    def import_result_dialog(self):
        if self.state.data is None:
            self.statusBar().showMessage('Open a run before importing a result.', 10000)
            return
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, 'Import matching result for the selected cache', '', 'Matching result (*.json)')
        if not path:
            return
        try:
            self.import_match_result(path)
        except Exception as exc:
            self.statusBar().showMessage(f'Import rejected: {exc}', 20000)
            self.browse.info.setText(f'Import rejected: {exc}')

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project window imports a completed run of its own selection, from multiday_viewer.py:392-408; the selection's inputs are read once; open_result builds the run data and run list before set_data (viewer-split Task 8) | date: 2026-10-06]
    def import_match_result(self, path) -> MatchResult:
        """Open the completed run whose ``result.json`` is ``path``; it must be a run of this window's selection.

        Returns the imported ``MatchResult``. A window without a selection imports nothing.
        """
        if self.selection is None:
            raise NotImplementedError('only a project viewer imports a result: viewer --run DIR')
        run_dir = Path(path).expanduser().resolve().parent
        selection_of_run(run_dir)
        dataset, _ = self._selection_inputs()
        if hashlib.sha256((run_dir / SELECTION_FILE).read_bytes()).hexdigest() \
                != dataset.metadata['input_manifest_sha256']:
            raise ValueError('result belongs to another selection')
        entry = catalog.read_run(run_dir)
        if entry is None:
            raise ValueError(f'{run_dir.name} is not a completed run')
        imported = backends.load_result(entry.result_path)
        remembered = self._run_entry(entry.result_path, imported)
        self.open_result(imported, run_dir)
        self._remember_run(remembered)
        return imported

    def _selection_inputs(self) -> tuple:
        """``(selection.dataset(), selection.day_extras())``, read on first use and kept."""
        if self._inputs is None:
            self._inputs = (self.selection.dataset(), self.selection.day_extras())
        return self._inputs

    def open_result(self, result, run_dir: Path) -> None:
        """Show ``result`` of the selection's run in ``run_dir``, curated in the project's
        ``curation/runs/<run folder name>`` when the mode curates.

        The run data and the run list are built before ``set_data`` switches controllers, so a failure there
        leaves the shown run, its controller and its lock in place.
        """
        dataset, extras = self._selection_inputs()
        data = self.mode.load(self.selection, dataset, result, extras)
        runs = _canonical_runs(selection_runs(self.selection))
        curation_dir = (self.selection.project.path / 'curation' / 'runs' / Path(run_dir).name
                        if self.mode.curates else None)
        self.set_data(data, curation_dir=curation_dir)
        self.refresh_runs(runs)

    def export_accepted(self) -> None:
        """Write the run's accepted tracks, replayed from the history on disk, to a new CSV; the path,
        or the error, shows in the status bar. ``self.curation.state`` is left untouched."""
        try:
            state = curation.replay(self.curation.result.tracks, self.curation.result.unmatched,
                                    curation.read_history(self.curation.curation_dir))
            out = curation.write_export(state, self.state.sites, self.curation.curation_dir)
        except Exception as exc:
            self.statusBar().showMessage(str(exc), 12000)
            return
        self.statusBar().showMessage(str(out), 12000)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the header's evidence tags: one amber label per tag, hover = its reasons (event-optional Task 12) | date: 2026-10-10]
    def _show_evidence_tags(self, tags: list) -> None:
        """Replace the header's evidence tags with one amber ``QLabel`` per ``(text, tooltip)``."""
        for label in self._evidence_labels:
            self._evidence_line.removeWidget(label)
            label.hide()
            label.deleteLater()
        self._evidence_labels = []
        for text, tooltip in tags:
            label = QtWidgets.QLabel(text)
            label.setToolTip(tooltip)
            label.setProperty('tone', 'unsure')
            label.setStyleSheet('font-size:11px;')
            theme.repolish(label)
            self._evidence_line.addWidget(label)
            self._evidence_labels.append(label)
        self.evidence_bar.setVisible(bool(tags))

    def evidence_tags_shown(self) -> list:
        """``[(text, tooltip)]`` of the header's evidence tags, in order."""
        return [(label.text(), label.toolTip()) for label in self._evidence_labels]
    # [AGENT EDIT END]

    # ---------------------------------------------------------------- run data
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no day-pair list fill; the tracked list fills under a signal blocker and the day table opens unpicked (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    def set_data(self, data: RunData, *, curation_dir: Optional[Path] = None) -> None:
        """Show ``data``; curate it through ``curation_dir``'s history, or inspect it only (``curation_dir`` is
        ``None`` or the mode does not curate).

        Every fallible preparation runs before the controller switch, so a failure there leaves the shown run,
        its controller and its lock in place. A failed render closes the new controller and re-raises.
        """
        result = data.result
        result.validate()
        review_threshold = float(result.provenance.get('threshold', cl.DEFAULT_THRESHOLD))
        cluster_lookup = cl.posterior_lookup(data.pairs())
        header_text = display.backend_header_text(result)
        header_summary = display.result_summary(result)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run's evidence tags are read before the controller switch (event-optional Task 12) | date: 2026-10-10]
        evidence_tags = display.run_evidence_tags(result.provenance)
        # [AGENT EDIT END]
        # the next run's controller opens first; only then is the old one closed and replaced,
        # so a failed open leaves the old run, controller and lock in place
        old = self.curation
        try:
            new = (curation.Curation.open(Path(curation_dir), result)
                   if curation_dir is not None and self.mode.curates else None)
            if (new is not None and old is not None and new.read_only and not old.read_only
                    and (new.lock_owner or {}).get('token') == old.token):
                new.close()  # the same run, locked by this window's own controller: keep that one
                new = old
            if old is not None and old is not new:
                # a failed close of the old controller closes the new one and keeps the old
                try:
                    old.close()
                except BaseException:
                    if new is not None:
                        new.close()
                    raise
        except Exception as exc:
            self.statusBar().showMessage(f'run switch failed · {exc}', 20000)
            raise
        self.curation = new
        self.tracked.confirming_take_over = False
        try:
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a panel that fails while the run renders reaches the cleanup below (viewer-split Task 7, fix round 1) | date: 2026-10-06]
            with _raising_slot_errors():
                self.review_threshold = review_threshold
                self.match_result = result
                self._shown_key = (result.backend_id, result.run_id)
                self.backend_header.setText(header_text)
                self.backend_header.setToolTip(header_summary)
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the header shows the run's evidence tags (event-optional Task 12) | date: 2026-10-10]
                self._show_evidence_tags(evidence_tags)
                # [AGENT EDIT END]
                self.state.set_data(data)
                self.cluster_lookup = cluster_lookup
                self.setWindowTitle(data.title)
                self.statusBar().showMessage(
                    f'{result.backend_id} · {result.run_id[:12]} · {len(result.unmatched)} unmatched · '
                    f'{len(result.excluded_units)} non-SU excluded', 9000)
                self.browse.info.setText(data.info)
                self.browse.info.setToolTip(data.info_tip)
                self.browse.import_btn.setEnabled(True)
                self.browse.import_btn.setToolTip(data.words.import_tip)
                # chains from the controller's verdict rows; the controls follow the new controller
                self.chains.show_for(self._verdict_frame())
                blocker = QtCore.QSignalBlocker(self.tracked)
                self.tracked.refresh_list()
                del blocker
                self.days.show_run()
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Browse fits the new run's day columns once the window has laid out (day-table widths) | date: 2026-10-13]
                QtCore.QTimer.singleShot(0, self, self._fit_days)
                # [AGENT EDIT END]
                self._update_controls()
                # keep the run list on the run being shown
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a project window always lists its runs (viewer-split Task 8) | date: 2026-10-06]
                if self.selection is not None:
                    self._fill_run_box()
                # [AGENT EDIT END]
                self.state.changed.emit()
            # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed lock release still detaches the controller and updates the controls; the render error is raised, chained from the close error (viewer-split Task 7, fix round 2) | date: 2026-10-06]
        except BaseException as render_exc:
            controller, self.curation = self.curation, None
            try:
                if controller is not None:
                    controller.close()
            except BaseException as close_exc:
                self._update_controls()
                raise render_exc from close_exc
            self._update_controls()
            raise
        # [AGENT EDIT END]
    # [AGENT EDIT END]

    # ------------------------------------------------- tracks, verdicts, controls
    def track_state(self) -> curation.CurationState:
        """The controller's state, or the result's own tracks replayed with no history when inspect-only."""
        if self.curation is not None:
            return self.curation.state
        if self.match_result is None:
            return curation.replay([], [], [])
        return curation.replay(self.match_result.tracks, self.match_result.unmatched, [])

    def _verdict_of(self, cellid_a, cellid_b) -> Optional[dict]:
        return self.track_state().verdicts.get(st.canonical(cellid_a, cellid_b))

    def _verdict_frame(self) -> pd.DataFrame:
        """One row per verdict, with each unit's siteid, for the chain checks."""
        sites = self.state.sites or {}
        rows = [dict(cellid_a=a, cellid_b=b, siteid_a=sites.get(a), siteid_b=sites.get(b), verdict=v['verdict'])
                for (a, b), v in self.track_state().verdicts.items()]
        return pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'siteid_a', 'siteid_b', 'verdict'])

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's actions write only when writable (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's merge, track and unmatch keys join the writable controls; the movement keys stay enabled; tag wraps the whole function (day-table-keys Task 4) | date: 2026-10-14]
    def _update_controls(self):
        """Enable each mutation control from inspect-only, read-only, and the current row's units and status. The
        verdict, undo and day-table keys work only when writable; the movement keys always do."""
        writable = self.curation is not None and not self.curation.read_only
        panel = self.tracked
        panel.inspect_tag.setVisible(self.curation is None)
        cid, members = panel.current_cluster() if hasattr(panel, '_rows') else (None, [])
        status = panel.current_status() if cid else None
        held = status in ('proposed', 'unmatched', 'accepted')
        enabled = {
            'Accept': writable and status in ('proposed', 'unmatched'),
            'Reject': writable and held and len(members) >= 2,
            'Split selected': writable and held and len(members) >= 2,
            'Unmatch unit': writable and held and len(members) >= 2,
            'Merge': writable,
            'Reopen': writable and status in ('accepted', 'rejected'),
        }
        for label, on in enabled.items():
            panel.buttons[label].setEnabled(bool(on))
        for control in (list(self.evidence.verdict_buttons) + list(self.verdict_shortcuts)
                        + list(self.undo_shortcuts) + list(self.day_shortcuts)):
            control.setEnabled(writable)
        for button in self.days.buttons.values():
            button.setEnabled(writable)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Export is enabled whenever a controller is open, read-only included, from multiday_viewer.py:329-332 (viewer-split Task 8) | date: 2026-10-06]
        if self.export_btn is not None:
            self.export_btn.setEnabled(self.curation is not None)
        # [AGENT EDIT END]
        self._update_lock_tag()
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    def _update_lock_tag(self):
        """Show why the viewer cannot write, and Take over (then Confirm / Cancel) for another host's lock."""
        panel, c = self.tracked, self.curation
        text, tip, can_take_over = '', '', False
        if c is not None and c.write_failed:
            text, tip = 'write failed', c.write_failed
        elif c is not None and c.taken_over:
            text = 'taken over'
        elif c is not None and c.read_only:
            owner = c.lock_owner or {}
            text = 'read-only · locked'
            tip = f"{owner.get('host')} pid {owner.get('pid')} · {c.curation_dir / curation.LOCK}"
            can_take_over = c.remote_lock
        if not can_take_over:
            panel.confirming_take_over = False
        panel.lock_tag.setText(text)
        panel.lock_tag.setToolTip(tip)
        panel.lock_tag.setVisible(bool(text))
        panel.take_over_btn.setVisible(can_take_over and not panel.confirming_take_over)
        panel.confirm_take_over_btn.setVisible(can_take_over and panel.confirming_take_over)
        panel.cancel_take_over_btn.setVisible(can_take_over and panel.confirming_take_over)

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table redraws from the replayed history (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    def take_over(self):
        """Take the remote lock, then show the replayed history and set the controls."""
        self.tracked.confirming_take_over = False
        if self.curation is None:
            return
        try:
            self.curation.take_over()
        except (OSError, ValueError) as exc:  # PermissionError is an OSError
            self.statusBar().showMessage(f'take over failed · {exc}', 12000)
            self._update_controls()
            return
        self.tracked.refresh_list()
        self.chains.show_for(self._verdict_frame())
        self.days.refresh()
        self._update_controls()
    # [AGENT EDIT END]

    # ------------------------------------------------------------------ run list
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run box lists selection_runs plus the remembered imports; refresh_runs takes the list and reads nothing; the Refresh button lists again and shows a failure in the status bar (viewer-split Task 8) | date: 2026-10-06]
    def _use_run_list(self):
        """Show the run row, with a Refresh button that lists the selection's runs again."""
        self.browse.use_runs()
        self.browse.run_box.activated.connect(self._run_chosen)
        self.refresh_btn = QtWidgets.QPushButton('Refresh')
        self.refresh_btn.setToolTip('list the runs again')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2 - the Refresh button lists the runs in its own handler; refresh_runs requires the prepared list and reads nothing (viewer-split Task 8) | date: 2026-10-07]
        self.refresh_btn.clicked.connect(self.list_runs_again)
        self.browse.run_row.layout().addWidget(self.refresh_btn)

    def list_runs_again(self):
        """The Refresh button: list the selection's runs again; a failure shows in the status bar and
        leaves the run box as it was."""
        try:
            runs = _canonical_runs(selection_runs(self.selection))
        except Exception as exc:
            self.statusBar().showMessage(f'run listing failed · {exc}', 12000)
            return
        self.refresh_runs(runs)

    def refresh_runs(self, runs: list):
        """Show ``runs`` (catalog entries with resolved paths) and the remembered imports in the run box,
        reading nothing."""
        self._run_entries = list(runs)
        self._fill_run_box()
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: listed and remembered entries hold resolved paths, so filling the run box reads nothing (viewer-split Task 8 fix round 1) | date: 2026-10-07]
    def _fill_run_box(self):
        listed = {e.result_path for e in self._run_entries}
        entries = self._run_entries + [e for e in self._imported_runs if e.result_path not in listed]
    # [AGENT EDIT END]
        entries.sort(key=lambda e: e.finished, reverse=True)
        box = self.browse.run_box
        blocker = QtCore.QSignalBlocker(box)
        box.clear()
        self._run_items = []
        shown = next((e for e in entries if (e.backend_id, e.run_id) == self._shown_key), None)
        if shown is None and self.match_result is not None:
            unsaved = self.match_result.backend_id == 'unassigned'
            box.addItem('none' if unsaved else f'{names.backend_short(self.match_result.backend_id)} · not saved')
            box.setItemData(0, display.result_summary(self.match_result), QtCore.Qt.ToolTipRole)
            self._run_items.append(None)
        for entry, label in zip(entries, catalog.run_labels(entries)):
            box.addItem(label)
            box.setItemData(box.count() - 1, catalog.run_tooltip(entry), QtCore.Qt.ToolTipRole)
            if entry.result_path in self._rejected_runs:
                box.model().item(box.count() - 1).setEnabled(False)
            self._run_items.append(entry)
        box.setCurrentIndex(self._run_items.index(shown) if shown is not None else 0)
        del blocker

    def _run_chosen(self, index):
        entry = self._run_items[index] if 0 <= index < len(self._run_items) else None
        if entry is None or (entry.backend_id, entry.run_id) == self._shown_key:
            self._fill_run_box()
            return
        try:
            self.import_match_result(entry.result_path)
        except Exception as exc:
            self._rejected_runs.add(entry.result_path)
            self.statusBar().showMessage(f'rejected · {exc}', 20000)
        self._fill_run_box()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the imported entry is built (the file stat) before the run switch and remembered after it, reading nothing (viewer-split Task 8) | date: 2026-10-06]
    @staticmethod
    def _run_entry(path, result) -> RunEntry:
        """The run-list entry of an imported result, finished at its file's modification time."""
        path = Path(path).expanduser().resolve()
        finished = datetime.fromtimestamp(path.stat().st_mtime).astimezone()
        return RunEntry(path, result.backend_id, result.run_id, finished, {})

    def _remember_run(self, entry: RunEntry):
        """List an imported result even when it lives outside the scanned run folders."""
        if all(e.result_path != entry.result_path for e in self._imported_runs):
            self._imported_runs.append(entry)
        self._fill_run_box()
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: select_pair finds the run's pair row in either order, else selects the two units; a day-table action refreshes Tracked units on its track (native-same-day Task 10) | date: 2026-10-11]
    # ------------------------------------------------------------------ pairs
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a given selection stays selected with the pair or without a row (day-table-keys Task 3) | date: 2026-10-14]
    def select_pair(self, cellid_a: str, cellid_b: str, selection: Optional[list] = None) -> None:
        """Open the run's pair row joining ``cellid_a`` and ``cellid_b``, in either order, with its day pair
        (``None`` within one site); ``selection`` (when given) stays selected, else the two units. Without a row
        that selection is selected and ``state.day_pair`` stays."""
        P = self.state.pairs
        hit = P[((P.cellid_a == cellid_a) & (P.cellid_b == cellid_b)) |
                ((P.cellid_a == cellid_b) & (P.cellid_b == cellid_a))] if len(P) else P
        if not len(hit):
            self.state.set_selection(selection if selection is not None else [cellid_a, cellid_b])
            self.evidence.status.setText('no pair evidence')
            return
        row = hit.iloc[0]
        self.state.day_pair = next((dp for dp in self.state.data.day_pairs()
                                    if {dp['siteid_a'], dp['siteid_b']} == {row.siteid_a, row.siteid_b}), None)
        self.state.set_pair(row, selection)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day-table action keeps the pick; a hidden track leaves no Tracked units row current (D36) (native-same-day Task 12) | date: 2026-10-12]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's string comes back selected after Tracked units re-lists; tag wraps the whole function (day-table-keys Task 3) | date: 2026-10-14]
    def _day_curated(self, cellid) -> None:
        """A day-table action was written: list the tracks again on the one now holding ``cellid``; when a filter
        hides it, no row is current and the pick and selection stay. Then the day table's string, when there is
        one, is selected again with its active pair."""
        self.tracked.refresh_list(preferred_id=self.days.track_of(cellid), fallback=False)
        self.days.reselect()
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # ---------------------------------------------------------------- verdicts
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table redraws in place of the candidate verdict cell (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    def set_verdict(self, verdict):
        r = self.state.row
        # a verdict is one Curation.do row; inspect-only writes nothing, a refusal shows its message
        if r is None or self.curation is None:
            return
        try:
            self.curation.do('verdict', cellid_a=r.cellid_a, cellid_b=r.cellid_b, verdict=verdict,
                             notes=self.evidence.notes.toPlainText())
        except (ValueError, PermissionError) as exc:
            self.evidence.status.setText(str(exc))
            return
        finally:
            self._update_controls()
        self.statusBar().showMessage(
            f'saved "{verdict}" for {r.cellid_a} / {r.cellid_b}', 6000)
        self.days.refresh()
        self.chains.show_for(self._verdict_frame())
        self._step(1)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table redraws from the undone or redone history (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's string comes back selected after Tracked units re-lists; tag wraps the whole function (day-table-keys Task 3) | date: 2026-10-14]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: with a string, Tracked units re-lists on the picked unit's track or no row, so an undone track never re-picks and drops the string; tag wraps the whole function (day-table-keys Task 3 fix round 1) | date: 2026-10-14]
    def _undo_redo(self, operation):
        """Undo or redo one action; a refusal ('nothing to undo') shows in the status bar. With a day-table
        string, Tracked units lists the track now holding the picked unit (no row current when a filter hides
        it), and the string is selected again with its active pair; else the previous row stays current."""
        if self.curation is None or self.curation.read_only:
            return
        try:
            getattr(self.curation, operation)()
        except (ValueError, PermissionError) as exc:
            self.statusBar().showMessage(str(exc), 4000)
            return
        finally:
            self._update_controls()
        days = self.days
        if days.picked_unit is not None and (days.strung or days.extras):
            # the undone or redone track may be the shown one: falling back to another row would re-pick
            self.tracked.refresh_list(preferred_id=days.track_of(days.picked_unit), fallback=False)
        else:
            self.tracked.refresh_list()
        days.reselect()
        self.chains.show_for(self._verdict_frame())
        days.refresh()
    # [AGENT EDIT END]
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Up / Down step the day table's active column (native-same-day Task 10) | date: 2026-10-11]
    def _step(self, d):
        self.days.step(d)
    # [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open any completed project run in its selection's mode, and the python -m nems_lbhb.unit_match.viewer command, from multiday_viewer.py:431-455 and benchmark_viewer.py:395-424 (viewer-split Task 8) | date: 2026-10-06]
def _canonical_runs(runs) -> list:
    """``runs`` (catalog entries) with resolved result paths, read before the run box is filled."""
    return [dataclasses.replace(e, result_path=e.result_path.resolve()) for e in runs]


def open_run(run_dir) -> App:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the refusal before the window is built (event-optional Task 12) | date: 2026-10-10]
    """The viewer window on the completed project run in ``run_dir``, in the mode its selection records.

    The result is read first; a run with mixture references but no valid ``run_evidence`` is refused
    (``display.require_run_evidence``) before any window is built. A failure after the window is built
    closes it, releasing any lock it took, and re-raises.
    """
    # [AGENT EDIT END]
    selection = selection_of_run(run_dir)
    entry = catalog.read_run(run_dir)
    if entry is None:
        raise ValueError(f'{run_dir} is not a completed run')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a run saved before event-optional matchers is refused before any window or panel is built (D56) (event-optional Task 12) | date: 2026-10-10]
    result = backends.load_result(entry.result_path)
    display.require_run_evidence(result.provenance)
    # [AGENT EDIT END]
    mode = MODES[selection.mode]
    window = App(mode, layout_file=paths.resolve_cache_path(mode.layout_name, root=selection.project.root),
                 selection=selection)
    try:
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the result read and checked above (event-optional Task 12) | date: 2026-10-10]
        window.open_result(result, entry.result_path.parent)
        # [AGENT EDIT END]
    except BaseException as exc:
        try:
            window.close()
        except BaseException as close_exc:
            raise exc from close_exc
        raise
    return window


def main(argv: Optional[list] = None) -> int:
    """Open ``--run DIR`` in its viewer and run the Qt event loop."""
    parser = argparse.ArgumentParser(prog='python -m nems_lbhb.unit_match.viewer', allow_abbrev=False,
                                     description='Open a completed project run in the unit-match viewer.')
    parser.add_argument('--run', required=True, metavar='DIR', help='a completed project run folder')
    args = parser.parse_args(argv)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer opens in the saved theme, dark first (launcher-gui Task 8) | date: 2026-10-07]
    theme.apply_theme(app, theme.saved_theme())
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer opens in the saved unit palette (graphite-palettes Task 4) | date: 2026-10-11]
    unit_palettes.set_palette(theme.saved_unit_palette())
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every window of the viewer carries the logo icon (logo) | date: 2026-10-13]
    app.setWindowIcon(logo.icon())
    # [AGENT EDIT END]
    window = open_run(Path(args.run))
    window.show()
    return app.exec()
# [AGENT EDIT END]
