# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the rival candidates table, from gui.py:1275-1349; a click opens a pair through actions.select_pair; fix round 2: the constructor declares state: PairState and actions: Actions (viewer-split Task 6, fix round 2) | date: 2026-10-06]
"""Every neuron inside the proximity gate, with the reason each was rejected."""
import numpy as np

from multiphy import clusters as cl
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: cells carry theme tokens instead of colours (launcher-gui Task 8) | date: 2026-10-07]
from multiphy.viewer import theme
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: P_SOUND renamed P_EVENT (launcher-gui Task 10) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the wave column reads P_WAVE (event-optional Task 12) | date: 2026-10-10]
from multiphy.viewer.display import P_PHYS, P_EVENT, P_WAVE, headline_probability, pct, probability
# [AGENT EDIT END]
# [AGENT EDIT END]
from multiphy.viewer.panels.tables.base import NumericItem, _table
from multiphy.viewer.state import Actions, PairState
from multiphy.viewer.qt import QtCore, QtWidgets
# [AGENT EDIT END]


class RivalPanel(QtWidgets.QWidget):
    """Every neuron inside the proximity gate, with the reason each was rejected.

    The judgment is whether the best candidate clearly beats the runner-up, so
    the rejected rivals belong on screen; a pair shown alone invites agreement.
    """
    panel_id, panel_title = 'rivals', 'Rivals'
    GATES = ('prox', 'rate', 'isi', 'sw')

    def __init__(self, state: PairState, actions: Actions, parent=None):
        super().__init__(parent)
        self.state, self.actions = state, actions
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        hint = QtWidgets.QLabel('Candidates for the same day-1 neuron · click to open · '
                                'the question is whether the best clearly beats the rest')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a hint label the app style sheet colours (launcher-gui Task 8) | date: 2026-10-07]
        hint.setWordWrap(True); hint.setProperty('role', 'hint')
        # [AGENT EDIT END]
        lay.addWidget(hint)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a wave column after phys (event-optional Task 12) | date: 2026-10-10]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the stream column is evt (launcher-gui Task 10) | date: 2026-10-07]
        self.table = _table(['day-2 neuron', 'P', 'evt', 'phys', 'wave', 'depth', 'why not'],
                            [96, 44, 44, 44, 44, 52, 96])
        # [AGENT EDIT END]
        # [AGENT EDIT END]
        self.table.horizontalHeaderItem(1).setToolTip('combined probability')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the evt column tooltip (launcher-gui Task 10) | date: 2026-10-07]
        self.table.horizontalHeaderItem(2).setToolTip('event-response fingerprint only')
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the physiology tooltip names the session rate event runs add to the stream (event-optional Task 12, final review fix round 1) | date: 2026-10-10]
        self.table.horizontalHeaderItem(3).setToolTip(
            'spontaneous rate, interval histogram and autocorrelogram; session rate too in event runs')
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the wave column tooltip; depth moves one column right (event-optional Task 12) | date: 2026-10-10]
        self.table.horizontalHeaderItem(4).setToolTip('waveform and footprint correlation and spike width')
        self.table.horizontalHeaderItem(5).setToolTip('depth difference, um')
        # [AGENT EDIT END]
        self.table.sortByColumn(1, QtCore.Qt.DescendingOrder)
        lay.addWidget(self.table, 1)
        self.table.cellClicked.connect(self._clicked)
        state.changed.connect(self.refresh)

    def _clicked(self, row, _col):
        it = self.table.item(row, 0)
        if it is None:
            return
        pair = it.data(QtCore.Qt.UserRole)
        if pair:
            self.actions.select_pair(*pair)

    def refresh(self):
        r = self.state.row
        self.table.setSortingEnabled(False)
        self.table.setRowCount(0)
        if r is None or self.state.pairs is None:
            self.table.setSortingEnabled(True)
            return
        P = self.state.pairs
        rank = cl.score_column(P)
        sub = P[(P.siteid_a == r.siteid_a) & (P.siteid_b == r.siteid_b) &
                (P.cellid_a == r.cellid_a)].sort_values(rank, ascending=False)
        self.table.setRowCount(len(sub))
        for i, (_, q) in enumerate(sub.iterrows()):
            failed = [g for g in self.GATES
                      if f'gate_{g}' in q.index and not bool(q.get(f'gate_{g}'))]
            pq, _ = headline_probability(q)
            why = ('fails ' + '+'.join(failed)) if failed else 'in'
            cells = [QtWidgets.QTableWidgetItem(cl.unit_label(q.cellid_b, self.state.sites[q.cellid_b])),
                     NumericItem(pq, f'{pq:.3f}' if np.isfinite(pq) else '--'),
                     # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: P_SOUND renamed P_EVENT (launcher-gui Task 10) | date: 2026-10-07]
                     NumericItem(probability(q, P_EVENT), pct(probability(q, P_EVENT))),
                     # [AGENT EDIT END]
                     NumericItem(probability(q, P_PHYS), pct(probability(q, P_PHYS))),
                     # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the wave cell after phys (event-optional Task 12) | date: 2026-10-10]
                     NumericItem(probability(q, P_WAVE), pct(probability(q, P_WAVE))),
                     # [AGENT EDIT END]
                     NumericItem(q.get('depth_delta_um', np.nan)),
                     QtWidgets.QTableWidgetItem(why)]
            cells[0].setData(QtCore.Qt.UserRole, (q.cellid_a, q.cellid_b))
            for j, it in enumerate(cells):
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shown pair and the failed rivals carry theme tokens (launcher-gui Task 8) | date: 2026-10-07]
                if q.cellid_b == r.cellid_b:
                    theme.fill(it, 'accent_tint')
                elif failed:
                    theme.tone(it, 'muted')
                # [AGENT EDIT END]
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
# [AGENT EDIT END]
