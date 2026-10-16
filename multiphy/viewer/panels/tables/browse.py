# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the navigation column, from gui.py:1183-1274 without the Well and Sounds cache boxes; its words come from the mode; fix round 2: the constructor declares state: PairState (viewer-split Task 6, fix round 2) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table replaces the day pairs and candidates (native-same-day Task 10) | date: 2026-10-11]
"""The navigation column: the saved-run list, Import and the day table."""
# [AGENT EDIT END]
from multiphy.viewer.data import Words
from multiphy.viewer.state import PairState
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no sorted candidates table, so no QtCore (native-same-day Task 10) | date: 2026-10-11]
from multiphy.viewer.qt import QtWidgets
# [AGENT EDIT END]


class BrowsePanel(QtWidgets.QWidget):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table replaces the day pairs and candidates (native-same-day Task 10) | date: 2026-10-11]
    """Run, Import, the day table. The navigation column."""
    # [AGENT EDIT END]
    panel_id, panel_title = 'browse', 'Browse'

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the pair section, the Candidates row and the candidates table removed (native-same-day Task 10 fix round 1) | date: 2026-10-11]
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
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a hint label the app style sheet colours (launcher-gui Task 8) | date: 2026-10-07]
        self.info.setProperty('role', 'hint')
        # [AGENT EDIT END]
        lay.addWidget(self.info)
    # [AGENT EDIT END]

    def use_runs(self):
        """Show the saved-run list."""
        self.run_row.show()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table sits just above the info line (native-same-day Task 10) | date: 2026-10-11]
    def set_days(self, panel: QtWidgets.QWidget) -> None:
        """Insert ``panel`` (the day table) with stretch 1 just above the info line."""
        lay = self.layout()
        lay.insertWidget(lay.indexOf(self.info), panel, 1)
    # [AGENT EDIT END]
# [AGENT EDIT END]
