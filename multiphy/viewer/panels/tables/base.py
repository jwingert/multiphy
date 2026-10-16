"""What every table panel shares: a cell that sorts on its number, and a read-only sortable table."""
import numpy as np

from multiphy.viewer.qt import QtWidgets
from multiphy.viewer import theme


class NumericItem(QtWidgets.QTableWidgetItem):
    """Table cell that sorts on a number while displaying formatted text.

    A plain QTableWidgetItem sorts lexically, which puts 0.9 above 0.85.
    """

    def __init__(self, value, text=None):
        super().__init__(text if text is not None else
                         ('--' if value is None or not np.isfinite(value) else f'{value:.2f}'))
        self._v = -np.inf if value is None or not np.isfinite(value) else float(value)

    def __lt__(self, other):
        if isinstance(other, NumericItem):
            return self._v < other._v
        return super().__lt__(other)


def _table(headers, widths=None):
    t = QtWidgets.QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    t.verticalHeader().setVisible(False)
    t.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.setSortingEnabled(True)
    t.horizontalHeader().setSectionsClickable(True)
    t.setItemDelegate(theme.ToneDelegate(t))
    if widths:
        for i, w in enumerate(widths):
            t.setColumnWidth(i, w)
    return t
