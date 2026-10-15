# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the shared selection state the panels listen to, from gui.py:343-370; it holds one RunData (viewer-split Task 4) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no candidates table: the state holds the pair on screen (native-same-day Task 10) | date: 2026-10-11]
"""``PairState``: the run on screen (one ``RunData``), the day pair, the pair on screen and the selected units.

Panels listen to ``changed``; nothing polls.
"""
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the imports Actions needs (viewer-split Task 6) | date: 2026-10-06]
from dataclasses import dataclass
from typing import Callable, Optional
# [AGENT EDIT END]

import pandas as pd

from nems_lbhb.unit_match.viewer.qt import QtCore


class PairState(QtCore.QObject):
    """The pair currently under review. Panels listen, nothing polls."""
    changed = QtCore.Signal()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: candidates removed; the row is the pair on screen (native-same-day Task 10 fix round 1) | date: 2026-10-11]
    def __init__(self):
        super().__init__()
        self.data = None           # the RunData on screen
        self.day_pair = None
        self.row = None            # the pair on screen, a pandas Series
        self.selection = []        # cellids to plot; defaults to the pair
    # [AGENT EDIT END]

    @property
    def sites(self) -> dict:
        """``{cellid: siteid}`` of the run, or ``{}`` before one is set."""
        return self.data.sites() if self.data is not None else {}

    @property
    def pairs(self) -> pd.DataFrame:
        return self.data.pairs() if self.data is not None else pd.DataFrame()

    @property
    def units(self):
        return self.data.units() if self.data is not None else None

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: candidates removed (native-same-day Task 10) | date: 2026-10-11]
    def set_data(self, data):
        """Show ``data``; the day pair, row and selection are cleared and nothing is emitted."""
        self.data = data
        self.day_pair = None
        self.row = None
        self.selection = []
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a pair opens with a given selection kept selected, else the two units (day-table-keys Task 3) | date: 2026-10-14]
    def set_pair(self, row, selection: Optional[list] = None) -> None:
        """Show the pair ``row``; ``selection`` (when given, each unit once) stays selected, else its two
        units are selected."""
        self.row = row
        self.selection = list(dict.fromkeys(selection)) if selection is not None else [row.cellid_a, row.cellid_b]
        self.changed.emit()
    # [AGENT EDIT END]

    def set_selection(self, cellids):
        """Select units without retaining unrelated pair evidence."""
        self.selection = list(dict.fromkeys(cellids))
        self.row = None
        self.changed.emit()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the window calls a table panel may make, as one frozen set of callables (viewer-split Task 6) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: select_pair takes the selection that stays selected; tag wraps the whole class (day-table-keys Task 3) | date: 2026-10-14]
@dataclass(frozen=True)
class Actions:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: select_pair opens a pair, not a candidate (native-same-day Task 10) | date: 2026-10-11]
    """Every window call a table panel makes; a panel reaches nothing else of the window.

    ``select_pair(cellid_a, cellid_b, selection=None)`` opens a pair, ``selection`` (when given) staying selected;
    ``set_verdict(verdict)`` records one; ``curation()``
    returns the controller or ``None``; ``track_state()`` the current ``CurationState``; ``take_over()``
    takes the run's lock; ``update_controls()`` sets the mutation controls; ``cluster_lookup()`` returns the
    ``{(cellid_a, cellid_b): score}`` dict; ``review_threshold()`` a float; ``merge_refusal(members)`` why
    ``members`` cannot become one track, or ``None``.
    """
    # [AGENT EDIT END]
    select_pair: Callable[..., None]
    set_verdict: Callable[[str], None]
    curation: Callable[[], object]
    track_state: Callable[[], object]
    take_over: Callable[[], None]
    update_controls: Callable[[], None]
    cluster_lookup: Callable[[], dict]
    review_threshold: Callable[[], float]
    merge_refusal: Callable[[list], Optional[str]]
# [AGENT EDIT END]
# [AGENT EDIT END]
