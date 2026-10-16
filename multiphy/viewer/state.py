"""``PairState``: the run on screen (one ``RunData``), the day pair, the pair on screen and the selected units.

Panels listen to ``changed``; nothing polls.
"""
from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd

from multiphy.viewer.qt import QtCore


class PairState(QtCore.QObject):
    """The pair currently under review. Panels listen, nothing polls."""
    changed = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.data = None           # the RunData on screen
        self.day_pair = None
        self.row = None            # the pair on screen, a pandas Series
        self.selection = []        # cellids to plot; defaults to the pair

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

    def set_data(self, data):
        """Show ``data``; the day pair, row and selection are cleared and nothing is emitted."""
        self.data = data
        self.day_pair = None
        self.row = None
        self.selection = []

    def set_pair(self, row, selection: Optional[list] = None) -> None:
        """Show the pair ``row``; ``selection`` (when given, each unit once) stays selected, else its two
        units are selected."""
        self.row = row
        self.selection = list(dict.fromkeys(selection)) if selection is not None else [row.cellid_a, row.cellid_b]
        self.changed.emit()

    def set_selection(self, cellids):
        """Select units without retaining unrelated pair evidence."""
        self.selection = list(dict.fromkeys(cellids))
        self.row = None
        self.changed.emit()


@dataclass(frozen=True)
class Actions:
    """Every window call a table panel makes; a panel reaches nothing else of the window.

    ``select_pair(cellid_a, cellid_b, selection=None)`` opens a pair, ``selection`` (when given) staying selected;
    ``set_verdict(verdict)`` records one; ``curation()``
    returns the controller or ``None``; ``track_state()`` the current ``CurationState``; ``take_over()``
    takes the run's lock; ``update_controls()`` sets the mutation controls; ``cluster_lookup()`` returns the
    ``{(cellid_a, cellid_b): score}`` dict; ``review_threshold()`` a float; ``merge_refusal(members)`` why
    ``members`` cannot become one track, or ``None``.
    """
    select_pair: Callable[..., None]
    set_verdict: Callable[[str], None]
    curation: Callable[[], object]
    track_state: Callable[[], object]
    take_over: Callable[[], None]
    update_controls: Callable[[], None]
    cluster_lookup: Callable[[], dict]
    review_threshold: Callable[[], float]
    merge_refusal: Callable[[list], Optional[str]]
