"""The day table: one column per day, every unit of that day, beside its links.

Rules:

- ``show_run()`` builds one column per site of ``site_order()``: the day name on two lines (date over site)
  above a table of ``unit``, ``µm`` (distance), ``sc`` (score) and ``s`` (status). It clears the pick, the
  cursor, the selection, the string and its extras, the held orders, the header sorts and the shown held
  orders, and changes no ``PairState``; each day's held order then starts in status order (``unmatched``,
  ``track``, ``accepted``, then cellid).
- Widths: ``unit``, ``µm``, ``sc`` and ``s`` start at ``WIDTHS``; ``spread`` shares a wider table's spare
  width among all four in proportion to them, on every resize of its viewport. A day column is never
  narrower than those widths plus frame and scrollbar, and grows with the panel; days past the panel's width
  scroll sideways. ``fit_width()`` is the scroll width that shows the first ``min(days, FIT_DAYS)`` day
  columns at their narrowest.
- Status: one letter, its word on hover: ``a`` (accepted) for a unit of an accepted track, ``m`` (matched)
  for a track of two or more, else ``u`` (unmatched).
- Held order: with no pick, for the picked unit's own day, and for a day whose header was clicked since the
  last pick, a column shows its held order (kept to its current units, then any other unit by cellid).
  Accept, Reject, Merge and Unmatch unit never change it.
- A unit's rank against another is their pair row's finite score, else its finite similarity, else missing;
  its distance is the 2D registered distance (``unit_distance``), else missing. A day's reference
  (``reference``) is the unit its ``µm`` and ``sc`` cells and header sorts measure against: none with no pick;
  the picked unit for its own day; else, walking ``site_order()`` from the day's neighbour toward the own day
  and stopping before it, the first strung day's unit, else the picked unit. The own day keeps its held
  order; every other day, unless it shows its held order, sorts by the pick key to its reference: the picked
  unit's current track first, then distance (ascending), then rank (descending), then cellid; a missing
  distance or rank sorts last. The same-day column sorts to the picked unit.
- Cells: ``µm`` is the distance to the column's reference in whole µm, ``sc`` the rank against it; both read
  ``--`` with no reference, for the reference itself, or with no value. Header hovers name the reference
  (``distance to <r> · µm``, ``score to <r>``); with none, ``HEADER_TIPS``.
- A header click sorts that column into its held order: ``unit`` by cellid, ``µm`` by distance then rank to
  the reference, ``sc`` by rank then distance to it (both by cellid with no reference), ``s`` by status then
  cellid; the same header again reverses it. The sort arrow shows on a column showing a held order a header
  set; it points down for ``sc`` only.
- A unit with proposed links (the partner may be on the same day or another day) shows ``⇄``; its tooltip
  names each partner and rank.
- Fill: the selected units in ``accent_tint``; the units the backend grouped with the picked one (its track
  in the result, and its proposed-link partners) in ``proposal_tint``. Unit text: bold for the picked
  unit's current track, toned ``accept`` when that track is accepted; else toned ``reject`` for a unit that
  shares a rejected track with the picked unit.
- Same-day column: pinned left of the days, outside their scroll. Empty until a pick; each pick fills it
  with the picked unit's other own-day units by the pick key to the picked unit, its reference, headed
  ``same day · <day name>`` (date over site). Only its header re-orders it (as a day column's header does)
  until the next pick. Its ``unit`` tooltip lists only the proposed-link lines; the distance is in ``µm``.
  A click there, with or without Ctrl, adds or removes an extra beside the string. Its width is fixed at a
  day column's narrowest.
- String: a click picks a unit and empties the string. The string holds at most one unit per other day
  (``strung``, never the own day) and ``extras``: same-day clicks and, after ``string_track``, a track's
  further units on one day. ``selected`` is the picked unit, the strung units in day order, then the extras,
  each once; stringing a unit takes it out of ``extras``. ``toggle`` (a Ctrl-click or a drift-dot click) on
  another day's unit sets or drops that day's strung unit and makes that day active; on an own-day unit it
  adds an extra (the same-day column active) or takes it out; on the picked unit the own day is active.
- Active day (``active_site``): its header is toned ``accent`` while a unit is picked and the same-day column
  is not active; the same-day header is toned ``accent`` while it is. ``cursor`` is the active unit: the
  active day's strung unit, the active extra, else the picked unit.
- ``move_day(d)`` makes the day ``d`` steps away active (clamped), stringing its top unit when it has none.
  ``step(d)`` moves the active extra ``d`` rows in the same-day column; in the own day picks the unit ``d``
  rows away in its held order (no pair opens); in another day moves its strung unit ``d`` rows (its top unit
  when none). ``drop_day()`` drops the active day's strung unit; ``clear_string()`` empties the string;
  ``string_track()`` strings the picked unit's current track (``no link`` with fewer than 2 units). After
  each: an active other day with a strung unit opens the pair of its reference and that unit, an active extra
  the pair of the picked unit and it, else the selection is set; ``selected`` stays selected either way.
  ``reselect()`` emits that again after a curation action, when there is a string.
- Accept and Reject act on the picked unit's track; Merge joins the whole tracks of the selected units;
  Unmatch unit takes the active unit (``cursor``) out of its track. A refused action shows its message only.
"""
from typing import Optional

import numpy as np
import pandas as pd

from multiphy import clusters as cl
from multiphy.viewer import theme
from multiphy.viewer.drift_frame import registered_positions, unit_distance
from multiphy.viewer.panels.tables.base import _table
from multiphy.viewer.qt import QtCore, QtWidgets
from multiphy.viewer.state import Actions, PairState

STATUS_ORDER = {'unmatched': 0, 'track': 1, 'accepted': 2}
# a status shows as one letter; its hover names it
STATUS_LETTERS = {'unmatched': 'u', 'track': 'm', 'accepted': 'a'}
STATUS_WORDS = {'unmatched': 'unmatched', 'track': 'matched', 'accepted': 'accepted'}
HEADERS = ('unit', 'µm', 'sc', 's')
HEADER_TIPS = ('', 'distance to the picked unit · µm', 'score to the picked unit',
               'status · u unmatched · m matched · a accepted')
# the narrowest widths that show `318-1 ⇄`, `9999`, `0.97` and a status letter, and each header beside its sort
# arrow, under the theme's compact header padding (theme.py `QHeaderView[role="compact"]`); sum 172
WIDTHS = (64, 40, 40, 28)
# Browse opens wide enough for this many day columns; more days scroll
FIT_DAYS = 5


def _numbers(values) -> np.ndarray:
    """``values`` as floats; a non-numeric or non-finite cell is NaN."""
    out = pd.to_numeric(pd.Series(values), errors='coerce').to_numpy(dtype=float, copy=True, na_value=np.nan)
    out[~np.isfinite(out)] = np.nan
    return out


def _text(rank: float) -> str:
    return '--' if not np.isfinite(rank) else f'{rank:.2f}'


def spread(width: int) -> list:
    """``WIDTHS`` widened to share ``width``'s spare pixels: every column but the last gains
    ``extra * w // sum(WIDTHS)``, the last takes the remainder, so the list sums to ``max(width, sum(WIDTHS))``."""
    total = sum(WIDTHS)
    extra = max(0, width - total)
    widths = [w + extra * w // total for w in WIDTHS[:-1]]
    return widths + [max(width, total) - sum(widths)]


class _Spread(QtCore.QObject):
    """An event filter on a day table's viewport: each resize sets every column to ``spread`` of the viewport's
    width, so a vertical scrollbar appearing (a narrower viewport) re-spreads too."""

    def __init__(self, table):
        super().__init__(table)
        self._table = table

    def eventFilter(self, watched, event) -> bool:
        if event.type() == QtCore.QEvent.Resize:
            for i, w in enumerate(spread(self._table.viewport().width())):
                self._table.setColumnWidth(i, w)
        return False


def _narrowest(table) -> int:
    """The narrowest width of a day table that shows its four columns: ``WIDTHS`` plus frame and scrollbar."""
    return sum(WIDTHS) + 2 * table.frameWidth() + table.verticalScrollBar().sizeHint().width()


def _day_table():
    """A day table of the four ``HEADERS`` at ``WIDTHS`` (a wider viewport's spare width shared among them by
    ``_Spread``), compact headers with ``HEADER_TIPS``, no selection and no Qt sorting, never narrower than
    ``_narrowest``."""
    table = _table(list(HEADERS), list(WIDTHS))
    table.setSortingEnabled(False)
    table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
    header = table.horizontalHeader()
    header.setProperty('role', 'compact')
    # the minimum first: a section set narrower than the style's minimum is held at it
    header.setMinimumSectionSize(min(WIDTHS))
    for i, w in enumerate(WIDTHS):
        table.setColumnWidth(i, w)
    table.viewport().installEventFilter(_Spread(table))
    for i, tip in enumerate(HEADER_TIPS):
        table.horizontalHeaderItem(i).setToolTip(tip)
    table.setMinimumWidth(_narrowest(table))
    return table


def _set_tips(table, ref: Optional[str]) -> None:
    """``table``'s header hovers: ``µm`` and ``sc`` name ``ref``; with no reference, ``HEADER_TIPS``."""
    tips = HEADER_TIPS if ref is None else ('', f'distance to {ref} · µm', f'score to {ref}', HEADER_TIPS[3])
    for i, tip in enumerate(tips):
        table.horizontalHeaderItem(i).setToolTip(tip)


def _two_lines(day_name: str) -> str:
    """``day_name`` with its first `` · `` a line break: the date over the site."""
    return day_name.replace(' · ', '\n', 1)


class DayTablePanel(QtWidgets.QWidget):
    """One column per day of every unit: the picked unit's own day in its held order, every other day sorted to its reference unit, and the picked unit's other own-day units in the pinned same-day column."""
    panel_id, panel_title = 'days', 'Day table'
    picked = QtCore.Signal(str)
    curated = QtCore.Signal(str)

    def __init__(self, state: PairState, actions: Actions, curates: bool, parent=None):
        super().__init__(parent)
        self.state, self.actions, self.curates = state, actions, curates
        self.picked_unit = None
        self.cursor = None
        self.active_site = None
        self.selected = []
        # site → its strung cellid, never the picked unit's own day; extras: cellids beside the string
        self.strung = {}
        self.extras = []
        self.held = {}
        self.sorted_by = {}
        self.held_shown = set()
        self._positions = pd.DataFrame()
        self.tables = {}
        self.headers = {}
        self._ranks = {}
        self._links = {}
        self._grouped = {}
        self.same_ids = []
        self.same_sorted_by = None
        self.same_day_active = False
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.scroll = QtWidgets.QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAsNeeded)
        self._columns = QtWidgets.QWidget()
        self._columns_layout = QtWidgets.QHBoxLayout(self._columns)
        self._columns_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll.setWidget(self._columns)
        body = QtWidgets.QHBoxLayout()
        same_box = QtWidgets.QVBoxLayout()
        self.same_header = QtWidgets.QLabel('same day')
        self.same_table = _day_table()
        self.same_table.setFixedWidth(_narrowest(self.same_table))
        self.same_table.cellClicked.connect(self._same_clicked)
        self.same_table.horizontalHeader().sectionClicked.connect(self._same_header_clicked)
        same_box.addWidget(self.same_header)
        same_box.addWidget(self.same_table, 1)
        body.addLayout(same_box)
        body.addWidget(self.scroll, 1)
        lay.addLayout(body, 1)

        row = QtWidgets.QHBoxLayout()
        self.buttons = {}
        verdicts = {'Accept': 'accept', 'Reject': 'reject'}
        for label, tip, slot in (('Accept', "accept the picked unit's track", self.accept),
                                 ('Reject', "reject the picked unit's track", self.reject),
                                 ('Merge', 'join the tracks of the selected units', self.merge),
                                 ('Unmatch unit', 'take the active unit out of its track', self.unmatch_unit)):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            if label in verdicts:
                b.setProperty('verdict', verdicts[label])
            b.clicked.connect(slot)
            b.setVisible(curates)
            row.addWidget(b)
            self.buttons[label] = b
        row.addStretch(1)
        lay.addLayout(row)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.status.setProperty('role', 'hint')
        lay.addWidget(self.status)

    # ------------------------------------------------------------- columns
    def show_run(self) -> None:
        """One column per site of the run, headed by its day name on two lines (date over site); clears the
        pick, the cursor, the selection, the string and its extras, the held orders and the header sorts, then
        holds each day's units in status order."""
        while self._columns_layout.count():
            item = self._columns_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.tables, self.headers = {}, {}
        self.picked_unit = self.cursor = self.active_site = None
        self.selected = []
        self.strung, self.extras = {}, []
        self.held, self.sorted_by, self.held_shown = {}, {}, set()
        self.same_ids, self.same_sorted_by, self.same_day_active = [], None, False
        self.same_table.setRowCount(0)
        self.same_header.setText('same day')
        _set_tips(self.same_table, None)
        data = self.state.data
        if data is not None:
            for site in data.site_order():
                column = QtWidgets.QWidget()
                box = QtWidgets.QVBoxLayout(column)
                box.setContentsMargins(0, 0, 0, 0)
                header = QtWidgets.QLabel(_two_lines(data.day_name(site)))
                table = _day_table()
                table.cellClicked.connect(lambda r, c, t=table: self._clicked(t, r))
                table.horizontalHeader().sectionClicked.connect(lambda s, site=site: self._header_clicked(site, s))
                box.addWidget(header)
                box.addWidget(table, 1)
                self._columns_layout.addWidget(column)
                self.tables[site], self.headers[site] = table, header
        self._read_run()
        if data is not None:
            for site, units in self._site_units().items():
                self.held[site] = sorted(units, key=self._status_key)
        self.refresh()

    def fit_width(self) -> int:
        """The day scroll's viewport width that shows the first ``min(days, FIT_DAYS)`` day columns at their
        narrowest; 0 with no day columns."""
        lay = self._columns_layout
        columns = [lay.itemAt(i).widget() for i in range(min(lay.count(), FIT_DAYS))]
        if not columns:
            return 0
        margins = lay.contentsMargins()
        return (sum(c.minimumSizeHint().width() for c in columns) + max(lay.spacing(), 0) * (len(columns) - 1)
                + margins.left() + margins.right())

    def _read_run(self) -> None:
        """The run's ranks by unit pair, proposed links, the backend's grouping and the units' registered
        positions; empty without a run."""
        self._ranks, self._links, self._grouped = {}, {}, {}
        self._positions = pd.DataFrame()
        data = self.state.data
        if data is None:
            return
        self._positions = registered_positions(data.units(),
                                               data.result.provenance if data.result is not None else {})
        pairs = data.pairs()
        self._links = cl.proposed_links(pairs)
        if len(pairs):
            score = _numbers(pairs[cl.score_column(pairs)]) if cl.score_column(pairs) in pairs else \
                np.full(len(pairs), np.nan)
            column = getattr(getattr(data.result, 'score', None), 'similarity_column', None)
            similarity = _numbers(pairs[column]) if column and column in pairs else np.full(len(pairs), np.nan)
            rank = np.where(np.isfinite(score), score, similarity)
            for a, b, r in zip(pairs.cellid_a.astype(str), pairs.cellid_b.astype(str), rank):
                key = frozenset((a, b))
                if key not in self._ranks or not np.isfinite(self._ranks[key]):
                    self._ranks[key] = r
        for track in getattr(data.result, 'tracks', ()) or ():
            members = [str(m) for m in track]
            for m in members:
                self._grouped.setdefault(m, set()).update(members)

    def rank(self, a: str, b: str) -> float:
        """The rank of ``a`` against ``b``: their pair row's score, else its similarity, else NaN."""
        return self._ranks.get(frozenset((a, b)), np.nan)

    def distance(self, a: str, b: str) -> float:
        """The 2D registered distance of ``a`` and ``b`` (``unit_distance``); NaN when it has none."""
        return unit_distance(self._positions, a, b)

    def _site_units(self) -> dict:
        """Each site's units: the members of every current track, each once."""
        sites = self.state.sites
        by_site = {}
        for t in self.actions.track_state().tracks.values():
            for m in t['members']:
                by_site.setdefault(sites.get(m), {})[m] = None
        return {site: list(units) for site, units in by_site.items()}

    def _status_key(self, c: str) -> tuple:
        return (STATUS_ORDER[self._status(c)], c)

    def _held_order(self, site: str, units: list) -> list:
        """``held[site]`` kept to ``units``, then any unit not in it by cellid."""
        present = set(units)
        kept = [c for c in self.held.get(site, []) if c in present]
        seen = set(kept)
        return kept + sorted(c for c in units if c not in seen)

    def _shows_held(self, site: str) -> bool:
        """True when ``site``'s column shows its held order: no pick, the picked unit's own day, or a header
        sort since the last pick."""
        p = self.picked_unit
        return p is None or self.state.sites.get(p) == site or site in self.held_shown

    def reference(self, site: str) -> Optional[str]:
        """The unit ``site``'s column measures ``µm`` and ``sc`` and its header sorts against, and sorts to
        when it does not show its held order (the own day always does): ``None`` with no pick; the picked unit
        for its own day; else, walking ``site_order()`` from ``site``'s neighbour toward the own day and stopping
        before it, the first strung day's unit, else the picked unit."""
        p = self.picked_unit
        if p is None:
            return None
        own = self.state.sites.get(p)
        order = self.state.data.site_order() if self.state.data is not None else []
        if site == own or site not in order or own not in order:
            return p
        here, there = order.index(site), order.index(own)
        toward = 1 if there > here else -1
        for i in range(here + toward, there, toward):
            if order[i] in self.strung:
                return self.strung[order[i]]
        return p

    def _pick_key(self, c: str, ref: str, mates: set) -> tuple:
        """The pick key to ``ref``: ``mates`` first, then registered distance (ascending), then rank
        (descending), then cellid; a missing distance or rank sorts last."""
        distance = self.distance(c, ref)
        rank = self.rank(c, ref)
        return (c not in mates, not np.isfinite(distance), distance if np.isfinite(distance) else 0.,
                not np.isfinite(rank), -rank if np.isfinite(rank) else 0., c)

    def _score_key(self, c: str, ref: str) -> tuple:
        """The ``sc`` header's key to ``ref``: rank (descending), then registered distance (ascending), then
        cellid; a missing rank or distance sorts last."""
        rank = self.rank(c, ref)
        distance = self.distance(c, ref)
        return (not np.isfinite(rank), -rank if np.isfinite(rank) else 0.,
                not np.isfinite(distance), distance if np.isfinite(distance) else 0., c)

    def _header_order(self, units: list, section: int, ref: Optional[str]) -> list:
        """``units`` sorted by header ``section`` against ``ref``: ``unit`` by cellid, ``µm`` by the pick key
        without a track, ``sc`` by ``_score_key``, ``s`` by status then cellid; ``µm`` and ``sc`` by cellid
        with no reference."""
        if section == 0 or (section in (1, 2) and ref is None):
            return sorted(units)
        if section == 1:
            return sorted(units, key=lambda c: self._pick_key(c, ref, set()))
        if section == 2:
            return sorted(units, key=lambda c: self._score_key(c, ref))
        return sorted(units, key=self._status_key)

    def _order(self, site: str, units: list, mates: set) -> list:
        """``site``'s held order when it shows one, else ``units`` by the pick key to its reference."""
        if self._shows_held(site):
            return self._held_order(site, units)
        ref = self.reference(site)
        return sorted(units, key=lambda c: self._pick_key(c, ref, mates))

    def _set_indicator(self, site: str) -> None:
        """The sort arrow, set last: Qt flips it on a header click before ``sectionClicked`` runs. It points
        down for ``sc`` only."""
        header = self.tables[site].horizontalHeader()
        if self._shows_held(site) and site in self.sorted_by:
            section, reverse = self.sorted_by[site]
            ascending = section != 2
            if reverse:
                ascending = not ascending
            header.setSortIndicatorShown(True)
            header.setSortIndicator(section, QtCore.Qt.AscendingOrder if ascending else QtCore.Qt.DescendingOrder)
        else:
            header.setSortIndicator(-1, QtCore.Qt.AscendingOrder)

    def column_ids(self, site: str) -> list:
        """The cellids of ``site``'s column, top to bottom."""
        t = self.tables[site]
        return [t.item(i, 0).data(QtCore.Qt.UserRole) for i in range(t.rowCount())]

    def track_of(self, cellid: str) -> Optional[str]:
        """The id of the current track holding ``cellid``, or ``None``."""
        for tid, t in self.actions.track_state().tracks.items():
            if cellid in t['members']:
                return tid
        return None

    def _track(self, cellid) -> Optional[dict]:
        tid = self.track_of(cellid)
        return self.actions.track_state().tracks[tid] if tid is not None else None

    def _status(self, cellid: str) -> str:
        track = self._track(cellid)
        if track is None:
            return 'unmatched'
        if track['status'] == 'accepted':
            return 'accepted'
        return 'track' if len(track['members']) >= 2 else 'unmatched'

    def refresh(self) -> None:
        """Redraw every column and the same-day column from the current pick, selection, track state and
        held orders; each table's header hovers name its reference. The active day's header is toned
        ``accent`` (with a pick, while the same-day column is not active); the same-day header while it is."""
        for site, label in self.headers.items():
            active = self.picked_unit is not None and not self.same_day_active and site == self.active_site
            label.setProperty('tone', 'accent' if active else '')
            theme.repolish(label)
        self.same_header.setProperty('tone', 'accent' if self.same_day_active else '')
        theme.repolish(self.same_header)
        if self.state.data is None or not self.tables:
            return
        track_state = self.actions.track_state()
        by_site = self._site_units()
        p = self.picked_unit
        mine = self._track(p) if p is not None else None
        mates = set(mine['members']) if mine else set()
        accepted = bool(mine) and mine['status'] == 'accepted'
        grouped = (self._grouped.get(p, set()) | set(self._links.get(p, []))) if p is not None else set()
        rejected_with = ({m for ms in track_state.rejected.values() if p in ms for m in ms} - {p}
                         if p is not None else set())
        marks = (p, mates, accepted, grouped, rejected_with)
        for site, table in self.tables.items():
            ref = self.reference(site)
            units = self._order(site, by_site.get(site, []), mates)
            table.setRowCount(len(units))
            for i, c in enumerate(units):
                self._fill_row(table, i, c, site, ref, marks)
            _set_tips(table, ref)
        own = self.state.sites.get(p) if p is not None else None
        self.same_table.setRowCount(len(self.same_ids))
        for i, c in enumerate(self.same_ids):
            self._fill_row(self.same_table, i, c, own, p, marks)
        _set_tips(self.same_table, p)
        for site in self.tables:
            self._set_indicator(site)
        self._set_same_indicator()

    def _fill_row(self, table, i: int, c: str, site, ref: Optional[str], marks: tuple) -> None:
        """Row ``i`` of ``table`` for unit ``c``: the ``unit`` label with ``⇄`` and its proposed-link lines on
        hover, the distance (whole µm) and score to ``ref`` (``--`` with no reference, for ``ref`` itself, or with
        no value), the status letter with its word on hover, and the wash, bold and tone rules."""
        p, mates, accepted, grouped, rejected_with = marks
        lines = [f'proposed link · {o} · {_text(self.rank(c, o))}' for o in self._links.get(c, [])]
        unit = QtWidgets.QTableWidgetItem(cl.unit_label(c, site) + (' ⇄' if c in self._links else ''))
        unit.setToolTip('\n'.join(lines))
        unit.setData(QtCore.Qt.UserRole, c)
        d = self.distance(c, ref) if ref is not None and c != ref else np.nan
        microns = QtWidgets.QTableWidgetItem(f'{d:.0f}' if np.isfinite(d) else '--')
        score = QtWidgets.QTableWidgetItem('--' if ref is None or c == ref else _text(self.rank(c, ref)))
        status = self._status(c)
        letter = QtWidgets.QTableWidgetItem(STATUS_LETTERS[status])
        letter.setToolTip(STATUS_WORDS[status])
        cells = [unit, microns, score, letter]
        wash = 'accent_tint' if c in self.selected else ('proposal_tint' if c in grouped else None)
        for cell in cells:
            theme.fill(cell, wash)
        font = unit.font()
        font.setBold(c in mates)
        unit.setFont(font)
        if c in mates and accepted:
            theme.tone(unit, 'accept')
        elif c in rejected_with:
            theme.tone(unit, 'reject')
        for j, cell in enumerate(cells):
            table.setItem(i, j, cell)

    def _set_same_indicator(self) -> None:
        """The same-day sort arrow, set last: Qt flips it on a header click before ``sectionClicked`` runs. It
        points down for ``sc`` only."""
        header = self.same_table.horizontalHeader()
        header.setSortIndicatorShown(True)
        if self.same_sorted_by is None:
            header.setSortIndicator(-1, QtCore.Qt.AscendingOrder)
            return
        section, reverse = self.same_sorted_by
        ascending = section != 2
        if reverse:
            ascending = not ascending
        header.setSortIndicator(section, QtCore.Qt.AscendingOrder if ascending else QtCore.Qt.DescendingOrder)

    def same_column_ids(self) -> list:
        """The cellids of the same-day column, top to bottom."""
        t = self.same_table
        return [t.item(i, 0).data(QtCore.Qt.UserRole) for i in range(t.rowCount())]

    def _fill_same(self) -> None:
        """Fill ``same_ids`` with the picked unit's other own-day units by the pick key to it; clear its header
        sort; head it ``same day · `` and the day name on two lines."""
        p = self.picked_unit
        own = self.state.sites.get(p)
        mine = self._track(p)
        mates = set(mine['members']) if mine else set()
        units = [c for c in self._site_units().get(own, []) if c != p]
        self.same_ids = sorted(units, key=lambda c: self._pick_key(c, p, mates))
        self.same_sorted_by = None
        data = self.state.data
        self.same_header.setText('same day · ' + _two_lines(data.day_name(own)) if data is not None else 'same day')

    def _same_clicked(self, row, column) -> None:
        item = self.same_table.item(row, 0)
        if item is None:
            return
        self.toggle(item.data(QtCore.Qt.UserRole), same_day=True)

    def _same_header_clicked(self, section) -> None:
        """Re-sort ``same_ids`` once by header ``section`` against the picked unit (``_header_order``); the same
        header again reverses it. No pick: no change."""
        p = self.picked_unit
        before = self.same_sorted_by
        if p is not None and before is not None and before[0] == section:
            self.same_ids = self.same_ids[::-1]
            self.same_sorted_by = (section, not before[1])
        elif p is not None:
            self.same_ids = self._header_order(self.same_ids, section, p)
            self.same_sorted_by = (section, False)
        self.refresh()

    # ------------------------------------------------------------- picking
    def _clicked(self, table, row) -> None:
        item = table.item(row, 0)
        if item is None:
            return
        if QtWidgets.QApplication.keyboardModifiers() & QtCore.Qt.ControlModifier:
            self.toggle(item.data(QtCore.Qt.UserRole))
        else:
            self.pick(item.data(QtCore.Qt.UserRole))

    def _header_clicked(self, site, section) -> None:
        """Sort ``site``'s column by header ``section`` against its reference (``_header_order``) into its held
        order; the same header again reverses it."""
        units = self._site_units().get(site, [])
        order = self._header_order(units, section, self.reference(site))
        before = self.sorted_by.get(site)
        reverse = (not before[1]) if self._shows_held(site) and before is not None and before[0] == section \
            else False
        self.held[site] = order[::-1] if reverse else order
        self.sorted_by[site] = (section, reverse)
        self.held_shown.add(site)
        self.refresh()

    def _set_pick(self, cellid: str) -> None:
        self.picked_unit = self.cursor = cellid
        self.active_site = self.state.sites.get(cellid)
        self.selected = [cellid]
        self.strung, self.extras = {}, []
        self.held_shown = set()
        self.same_day_active = False
        self._fill_same()
        self.refresh()
        for site, table in self.tables.items():
            if site == self.active_site and cellid in self.column_ids(site):
                table.scrollToItem(table.item(self.column_ids(site).index(cellid), 0))
            else:
                table.scrollToTop()
        self.same_table.scrollToTop()

    def pick(self, cellid: str) -> None:
        """Pick ``cellid``: every other day re-sorts to it, its own day keeps its held order, it alone is
        selected and shown, then ``picked`` emits."""
        self._set_pick(cellid)
        self.state.set_selection([cellid])
        self.picked.emit(cellid)

    def toggle(self, cellid: str, *, same_day: bool = False) -> None:
        """A Ctrl-click or a drift-dot click on ``cellid``; with no pick it picks ``cellid`` instead.

        The picked unit: the own day becomes active. An own-day unit: an extra is taken out (the own day
        active) or added (the same-day column active, it the cursor). Another day's unit: that day's strung
        unit is dropped when it is ``cellid``, else ``cellid`` is strung there; that day becomes active. Then
        the selection is emitted (``_apply``), and the row scrolls into view: in the same-day column when
        ``same_day``, else in its day column, which scrolls into view. Never changes the pick, the held orders
        or ``same_ids``."""
        p = self.picked_unit
        if p is None:
            self.pick(cellid)
            return
        own = self.state.sites.get(p)
        site = self.state.sites.get(cellid)
        if cellid == p:
            self.active_site, self.same_day_active, self.cursor = own, False, p
        elif site == own:
            if cellid in self.extras:
                self.extras.remove(cellid)
                self.active_site, self.same_day_active, self.cursor = own, False, p
            else:
                self.extras.append(cellid)
                self.active_site, self.same_day_active, self.cursor = own, True, cellid
        else:
            if self.strung.get(site) == cellid:
                del self.strung[site]
                self.cursor = p
            else:
                self._string(site, cellid)
                self.cursor = cellid
            self.active_site, self.same_day_active = site, False
        self._apply(scroll=False)
        if same_day:
            ids = self.same_column_ids()
            if cellid in ids:
                self.same_table.scrollToItem(self.same_table.item(ids.index(cellid), 0))
            return
        table = self.tables.get(site)
        if table is not None and cellid in self.column_ids(site):
            table.scrollToItem(table.item(self.column_ids(site).index(cellid), 0))
            self.scroll.ensureWidgetVisible(table.parentWidget())

    def follow_track(self, members: list) -> None:
        """Follow a track shown elsewhere: keep a pick inside it, else pick its earliest member quietly."""
        if not members:
            return
        if self.picked_unit in members:
            self.refresh()
            return
        order = {s: i for i, s in enumerate(self.state.data.site_order())}
        sites = self.state.sites
        first = min(members, key=lambda c: (order.get(sites.get(c), len(order)), c))
        self._set_pick(first)

    def step(self, d: int) -> None:
        """Move the active unit ``d`` rows (clamped). With the same-day column active, the active extra moves to
        the unit ``d`` rows away there (none when that is itself or another extra). In the picked unit's own
        day, the unit ``d`` rows away in its held order becomes the pick (no pair opens) and the own day's
        column scrolls into view. In another day, its strung unit moves ``d`` rows, or its top unit is strung
        when it has none. Then ``_apply``."""
        p = self.picked_unit
        if p is None:
            return
        own = self.state.sites.get(p)
        if self.same_day_active and self.cursor in self.extras:
            ids = self.same_ids
            if self.cursor not in ids:
                return
            target = ids[max(0, min(len(ids) - 1, ids.index(self.cursor) + d))]
            if target == self.cursor or target in self.extras:
                return
            self.extras[self.extras.index(self.cursor)] = target
            self.cursor = target
        elif self.same_day_active or self.active_site == own:
            ids = self.column_ids(own) if own in self.tables else []
            if p not in ids:
                return
            target = ids[max(0, min(len(ids) - 1, ids.index(p) + d))]
            if target != p:
                self.pick(target)
                self.scroll.ensureWidgetVisible(self.tables[own].parentWidget())
            return
        else:
            ids = self.column_ids(self.active_site) if self.active_site in self.tables else []
            if not ids:
                return
            at = self.strung.get(self.active_site)
            if at in ids:
                target = ids[max(0, min(len(ids) - 1, ids.index(at) + d))]
                if target == at:
                    return
            else:
                target = ids[0]
            self._string(self.active_site, target)
            self.cursor = target
        self._apply(scroll=True)

    def _string(self, site: str, cellid: str) -> None:
        """String ``cellid`` on ``site``, taking it out of ``extras``."""
        self.strung[site] = cellid
        if cellid in self.extras:
            self.extras.remove(cellid)

    def _emit(self) -> None:
        """An active other day with a strung unit opens the pair of its reference and that unit; the active
        extra opens the pair of the picked unit and it; else the selection is set. ``selected`` stays selected."""
        p, site = self.picked_unit, self.active_site
        selection = list(self.selected)
        if not self.same_day_active and site != self.state.sites.get(p) and site in self.strung:
            self.actions.select_pair(self.reference(site), self.strung[site], selection=selection)
        elif self.same_day_active and self.cursor in self.extras:
            self.actions.select_pair(p, self.cursor, selection=selection)
        else:
            self.state.set_selection(selection)

    def _apply(self, scroll: bool) -> None:
        """Rebuild ``selected`` (the picked unit, the strung units in day order, then ``extras``, each once),
        redraw, emit; with ``scroll``, the active column scrolls to its active unit and every other day column
        but the own day scrolls to the top."""
        p = self.picked_unit
        order = self.state.data.site_order() if self.state.data is not None else []
        self.selected = list(dict.fromkeys([p] + [self.strung[s] for s in order if s in self.strung]
                                           + self.extras))
        self.refresh()
        self._emit()
        if not scroll:
            return
        own = self.state.sites.get(p)
        for site, table in self.tables.items():
            if site == self.active_site and not self.same_day_active:
                ids = self.column_ids(site)
                if self.cursor in ids:
                    table.scrollToItem(table.item(ids.index(self.cursor), 0))
                self.scroll.ensureWidgetVisible(table.parentWidget())
            elif site != own:
                table.scrollToTop()
        ids = self.same_column_ids()
        if self.same_day_active and self.cursor in ids:
            self.same_table.scrollToItem(self.same_table.item(ids.index(self.cursor), 0))

    def move_day(self, d: int) -> None:
        """Make the day ``d`` steps from the active one (the own day while the same-day column is active)
        active, clamped; nothing when that is the same day. The own day: the picked unit is the cursor. Another
        day: its top unit is strung when it has none, and its strung unit is the cursor. Then ``_apply``."""
        p = self.picked_unit
        if p is None or self.state.data is None:
            return
        own = self.state.sites.get(p)
        order = self.state.data.site_order()
        current = own if self.same_day_active else self.active_site
        if current not in order:
            return
        target = order[max(0, min(len(order) - 1, order.index(current) + d))]
        if target == current:
            return
        self.same_day_active = False
        self.active_site = target
        if target == own:
            self.cursor = p
        else:
            ids = self.column_ids(target) if target in self.tables else []
            if target not in self.strung and ids:
                self._string(target, ids[0])
            self.cursor = self.strung.get(target, p)
        self._apply(scroll=True)

    def drop_day(self) -> None:
        """Drop the active day's strung unit; the picked unit is the cursor and the active day stays. Nothing
        with no pick, the own day active, or nothing strung there."""
        p = self.picked_unit
        if p is None or self.active_site == self.state.sites.get(p) or self.active_site not in self.strung:
            return
        del self.strung[self.active_site]
        self.cursor = p
        self._apply(scroll=True)

    def clear_string(self) -> None:
        """Empty the string and its extras; the own day is active and the picked unit the cursor."""
        p = self.picked_unit
        if p is None:
            return
        self.strung, self.extras = {}, []
        self.active_site = self.state.sites.get(p)
        self.same_day_active = False
        self.cursor = p
        self._apply(scroll=True)

    def string_track(self) -> None:
        """String the picked unit's current track: in its stored order, each member but the picked unit is
        strung on its day, or is an extra when it is on the own day or its day is already strung. The own day
        is active. ``no link`` when the track has fewer than 2 units."""
        p = self.picked_unit
        if p is None:
            return
        tid = self.track_of(p)
        members = list(self.actions.track_state().tracks[tid]['members']) if tid is not None else []
        if len(members) < 2:
            self.status.setText('no link')
            return
        own, sites = self.state.sites.get(p), self.state.sites
        self.strung, self.extras = {}, []
        for m in members:
            if m == p:
                continue
            site = sites.get(m)
            if site == own or site in self.strung:
                self.extras.append(m)
            else:
                self.strung[site] = m
        self.active_site = own
        self.same_day_active = False
        self.cursor = p
        self.status.setText('')
        self._apply(scroll=True)

    def reselect(self) -> None:
        """With a pick and a string or extras, emit the string's selection and active pair again (``_emit``);
        else nothing, so the track Tracked units shows stays selected."""
        if self.picked_unit is not None and (self.strung or self.extras):
            self._emit()

    # ------------------------------------------------------------- curation
    def _do(self, operation, **fields) -> bool:
        """Run one action through the controller; True when it was written. A refusal shows its message."""
        controller = self.actions.curation()
        if controller is None:
            self.status.setText('inspect-only')
            return False
        try:
            controller.do(operation, **fields)
        except (ValueError, PermissionError) as exc:
            self.status.setText(str(exc))
            return False
        finally:
            self.actions.update_controls()
        self.refresh()
        self.curated.emit(self.picked_unit)
        return True

    def _verdict(self, operation) -> None:
        if self.picked_unit is None:
            self.status.setText('pick a unit')
            return
        tid = self.track_of(self.picked_unit)
        members = list(self.actions.track_state().tracks[tid]['members']) if tid is not None else []
        self._do(operation, cluster_id=tid, members=members)

    def accept(self) -> None:
        self._verdict('accept')

    def reject(self) -> None:
        self._verdict('reject')

    def merge(self) -> None:
        """Join the whole tracks of the selected units, in selection order."""
        if self.picked_unit is None:
            self.status.setText('pick a unit')
            return
        tracks = self.actions.track_state().tracks
        ids = list(dict.fromkeys(t for t in (self.track_of(c) for c in self.selected) if t is not None))
        union = [m for tid in ids for m in tracks[tid]['members']]
        refusal = self.actions.merge_refusal(union)
        if refusal:
            self.status.setText(refusal)
            return
        if len(ids) < 2:
            self.status.setText('ctrl-click units of 2 or more tracks')
            return
        self._do('merge', parent_ids=ids, members=union)

    def unmatch_unit(self) -> None:
        """Take the cursor's unit out of its track."""
        if self.picked_unit is None:
            self.status.setText('pick a unit')
            return
        self._do('remove_member', cluster_id=self.track_of(self.cursor), members=[self.cursor])
