# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the chains and consistency text, from gui.py:1472-1508; fix round 2: the constructor declares state: PairState (viewer-split Task 6, fix round 2) | date: 2026-10-06]
"""Chains across days, and triangles that disagree."""
from nems_lbhb.unit_match import store as st
from nems_lbhb.unit_match.viewer.state import PairState
from nems_lbhb.unit_match.viewer.qt import QtWidgets


class ChainPanel(QtWidgets.QWidget):
    """Chains across days, and triangles that disagree."""
    panel_id, panel_title = 'chains', 'Chains and consistency'

    def __init__(self, state: PairState, parent=None):
        super().__init__(parent)
        self.state = state
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.text = QtWidgets.QPlainTextEdit()
        self.text.setReadOnly(True)
        lay.addWidget(self.text)

    def show_for(self, verdicts):
        if verdicts is None or not len(verdicts):
            self.text.setPlainText('no verdicts yet')
            return
        _, summ = st.chains(verdicts)
        bad = st.inconsistencies(verdicts)
        lines = []
        if len(summ):
            n = int(summ.n_sites.max())
            lines.append(f'{len(summ)} chain{"" if len(summ) == 1 else "s"}, '
                         f'longest spans {n} day{"" if n == 1 else "s"}')
            for _, x in summ.head(10).iterrows():
                lines.append(f'  {x.chain_id}: {x.n_units} units over {x.n_sites} days')
        else:
            lines.append('no accepted matches yet')
        if len(bad):
            lines.append(f'\n{len(bad)} inconsistent triangle(s):')
            for _, b in bad.head(6).iterrows():
                lines.append(f'  {b.cellid_a} / {b.cellid_b} via {b.via}: {b.problem}')
        self.text.setPlainText('\n'.join(lines))
# [AGENT EDIT END]
