"""Every neuron inside the proximity gate, with the reason each was rejected."""
import numpy as np

from multiphy import clusters as cl
from multiphy.viewer import theme
from multiphy.viewer.display import P_PHYS, P_EVENT, P_WAVE, headline_probability, pct, probability
from multiphy.viewer.panels.tables.base import NumericItem, _table
from multiphy.viewer.state import Actions, PairState
from multiphy.viewer.qt import QtCore, QtWidgets


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
        hint.setWordWrap(True); hint.setProperty('role', 'hint')
        lay.addWidget(hint)
        self.table = _table(['day-2 neuron', 'P', 'evt', 'phys', 'wave', 'depth', 'why not'],
                            [96, 44, 44, 44, 44, 52, 96])
        self.table.horizontalHeaderItem(1).setToolTip('combined probability')
        self.table.horizontalHeaderItem(2).setToolTip('event-response fingerprint only')
        self.table.horizontalHeaderItem(3).setToolTip(
            'spontaneous rate, interval histogram and autocorrelogram; session rate too in event runs')
        self.table.horizontalHeaderItem(4).setToolTip('waveform and footprint correlation and spike width')
        self.table.horizontalHeaderItem(5).setToolTip('depth difference, um')
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
                     NumericItem(probability(q, P_EVENT), pct(probability(q, P_EVENT))),
                     NumericItem(probability(q, P_PHYS), pct(probability(q, P_PHYS))),
                     NumericItem(probability(q, P_WAVE), pct(probability(q, P_WAVE))),
                     NumericItem(q.get('depth_delta_um', np.nan)),
                     QtWidgets.QTableWidgetItem(why)]
            cells[0].setData(QtCore.Qt.UserRole, (q.cellid_a, q.cellid_b))
            for j, it in enumerate(cells):
                if q.cellid_b == r.cellid_b:
                    theme.fill(it, 'accent_tint')
                elif failed:
                    theme.tone(it, 'muted')
                self.table.setItem(i, j, it)
        self.table.setSortingEnabled(True)
