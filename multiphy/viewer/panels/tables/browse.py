"""The navigation column: the saved-run list, Import and the day table."""
from multiphy.viewer.data import Words
from multiphy.viewer.state import PairState
from multiphy.viewer.qt import QtWidgets


class BrowsePanel(QtWidgets.QWidget):
    """Run, Import, the day table. The navigation column."""
    panel_id, panel_title = 'browse', 'Browse'

    def __init__(self, state: PairState, words: Words, parent=None):
        super().__init__(parent)
        self.state = state
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.run_row = QtWidgets.QWidget()
        runs = QtWidgets.QHBoxLayout(self.run_row)
        runs.setContentsMargins(0, 0, 0, 0)
        runs.addWidget(QtWidgets.QLabel('<b>Run</b>'))
        self.run_box = QtWidgets.QComboBox()
        self.run_box.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Preferred)
        self.run_box.setToolTip('Saved runs of this archive · each has its own curation history')
        runs.addWidget(self.run_box, 1)
        self.run_row.hide()
        lay.addWidget(self.run_row)
        buttons = QtWidgets.QHBoxLayout()
        self.import_btn = QtWidgets.QPushButton('Import…')
        self.import_btn.setToolTip(words.import_tip)
        buttons.addWidget(self.import_btn)
        lay.addLayout(buttons)
        self.info = QtWidgets.QLabel('')
        self.info.setWordWrap(True)
        self.info.setProperty('role', 'hint')
        lay.addWidget(self.info)

    def use_runs(self):
        """Show the saved-run list."""
        self.run_row.show()

    def set_days(self, panel: QtWidgets.QWidget) -> None:
        """Insert ``panel`` (the day table) with stretch 1 just above the info line."""
        lay = self.layout()
        lay.insertWidget(lay.indexOf(self.info), panel, 1)
