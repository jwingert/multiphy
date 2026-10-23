"""Chains across days, and triangles that disagree."""
from multiphy import store as st
from multiphy.viewer.state import PairState
from multiphy.viewer.qt import QtWidgets


class ChainPanel(QtWidgets.QWidget):
    """Chains across days, and triangles that disagree."""
    panel_id, panel_title = 'chains', 'Pair chains and consistency (advanced)'

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
            self.text.setPlainText('No pair verdicts yet. Track approvals are separate.')
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
            lines.append('no same-neuron pair verdicts yet')
        if len(bad):
            lines.append(f'\n{len(bad)} inconsistent triangle(s):')
            for _, b in bad.head(6).iterrows():
                lines.append(f'  {b.cellid_a} / {b.cellid_b} via {b.via}: {b.problem}')
        self.text.setPlainText('\n'.join(lines))
