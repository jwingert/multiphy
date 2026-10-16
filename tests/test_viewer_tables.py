# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the table panels take a mode's Words and reach the window only through Actions (viewer-split Task 6) | date: 2026-10-06]
"""The table panels over a ``PairState``: their words come from the mode, their window calls from ``Actions``."""
import os
from types import SimpleNamespace

import pandas as pd
import pytest
from PySide6 import QtWidgets

from multiphy import curation
from multiphy.viewer.modes import same_day
from multiphy.viewer.panels.tables import BrowsePanel, ClusterPanel
from multiphy.viewer.state import Actions, PairState

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day-table fake, the clusters module and Qt roles for the Task 8 tests (native-same-day Task 8) | date: 2026-10-11]
from multiphy import clusters as cl
from multiphy.viewer.qt import QtCore

from .viewer_fakes import DictData, day_table_data
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the three-day fake for the distance sort (day-table-keys Task 2) | date: 2026-10-14]
from .viewer_fakes import string_days_data
# [AGENT EDIT END]

UNITS = ['a-01', 'b-01', 'c-01', 'd-01']


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def _state():
    state = PairState()
    pairs = pd.DataFrame({'cellid_a': UNITS[:3], 'cellid_b': UNITS[1:], 'p_global': [.99, .98, .97]})
    state.set_data(DictData(UNITS, {}, {}, {u: u.split('-')[0] for u in UNITS}, pairs))
    return state


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: select_pair takes the selection that stays selected (day-table-keys Task 3) | date: 2026-10-14]
def _actions(controller, refusal=None):
    return Actions(select_pair=lambda a, b, selection=None: None, set_verdict=lambda v: None,
                   curation=lambda: controller,
                   track_state=lambda: controller.state, take_over=lambda: None, update_controls=lambda: None,
                   cluster_lookup=lambda: {}, review_threshold=lambda: .9,
                   merge_refusal=lambda members: refusal)
# [AGENT EDIT END]


def test_merge_refusal_shown_before_count_check(qt):
    done = []
    controller = SimpleNamespace(state=curation.initial_state([UNITS], []),
                                 do=lambda *args, **fields: done.append((args, fields)))
    state = _state()
    panel = ClusterPanel(state, _actions(controller, refusal='no'), same_day.WORDS)
    panel.refresh_list()
    panel.table.selectRow(0)
    assert len(panel.selected_ids()) == 1
    panel.merge_selected()
    assert panel.status.text() == 'no'
    assert done == []
    qt.processEvents()
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no pair section or candidates table in Browse; the tracked headers keep the words (native-same-day Task 10) | date: 2026-10-11]
def test_tables_take_words(qt):
    controller = SimpleNamespace(state=curation.initial_state([UNITS], []), do=lambda *a, **f: None)
    state = _state()
    browse = BrowsePanel(state, same_day.WORDS)
    tracked = ClusterPanel(state, _actions(controller), same_day.WORDS)
    assert tracked.table.horizontalHeaderItem(0).text() == 'parts'
    assert tracked.members.horizontalHeaderItem(1).text() == 'temporal part'
    qt.processEvents()
    browse.close()
    tracked.close()
# [AGENT EDIT END]


def test_browse_has_no_cache_boxes(qt):
    browse = BrowsePanel(_state(), same_day.WORDS)
    assert not hasattr(browse, 'block_box')
    assert not hasattr(browse, 'runclass_box')
    browse.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Tracked units flags proposed links, queues them, keeps a rejected track's units out of review, and shows a unit's track without a state change (native-same-day Task 8) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the controller starts from a run's tracks and unmatched units (day-table-keys Task 2) | date: 2026-10-14]
class _Controller:
    """A curation controller in memory, starting from ``data``'s result tracks and unmatched units
    (``day_table_data()`` when ``None``): ``do`` makes a history row and applies it."""

    def __init__(self, data=None):
        data = day_table_data() if data is None else data
        self.state = curation.initial_state([list(t) for t in data.result.tracks], list(data.result.unmatched))
        self.rows = []

    def do(self, operation, **fields):
        row = curation.make_row(operation, **fields)
        self.rows.append(row)
        self.state = curation.apply(self.state, row)
# [AGENT EDIT END]


def _day_table_panel():
    controller = _Controller()
    state = PairState()
    state.set_data(day_table_data())
    panel = ClusterPanel(state, _actions(controller), same_day.WORDS)
    panel.refresh_list()
    return panel, controller, state


def test_proposed_links_lists_partners():
    assert cl.proposed_links(day_table_data().pairs()) == {'A-1': ['A-2'], 'A-2': ['A-1', 'B-1'], 'B-1': ['A-2']}
    assert cl.proposed_links(pd.DataFrame({'cellid_a': ['a'], 'cellid_b': ['b']})) == {}


def test_tracked_flags_proposed_link_and_queues_review(qt):
    panel, _, _ = _day_table_panel()
    rows = panel._rows.set_index('cluster_id')
    track = rows.loc[cl.cluster_id(['A-1', 'B-1'])]
    assert track.flag_text == 'proposed link · A-2'
    assert bool(track.review) is True
    a2 = rows.loc[cl.cluster_id(['A-2'])]
    assert a2.flag_text == 'proposed link · A-1 B-1'
    assert bool(a2.review) is True
    a3 = rows.loc[cl.cluster_id(['A-3'])]
    assert a3.flag_text == 'unmatched'
    assert bool(a3.review) is False
    panel.close()


def test_proposed_flag_stays_after_accept(qt):
    panel, controller, _ = _day_table_panel()
    controller.do('accept', cluster_id=cl.cluster_id(['A-1', 'B-1']))
    panel.refresh_list()
    track = panel._rows.set_index('cluster_id').loc[cl.cluster_id(['A-1', 'B-1'])]
    assert track.flag_text == 'proposed link · A-2'
    assert bool(track.review) is False
    assert track.status == 'accepted'
    panel.close()


def test_rejected_track_units_stay_out_of_review(qt):
    panel, controller, _ = _day_table_panel()
    controller.do('reject', cluster_id=cl.cluster_id(['A-1', 'B-1']))
    panel.refresh_list()
    rows = panel._rows.set_index('cluster_id')
    a1 = rows.loc[cl.cluster_id(['A-1'])]
    assert a1.flag_text == 'proposed link · A-2'
    assert bool(a1.review) is False
    assert a1.status == 'unmatched'
    b1 = rows.loc[cl.cluster_id(['B-1'])]
    assert b1.flag_text == 'proposed link · A-2'
    assert bool(b1.review) is False
    a2 = rows.loc[cl.cluster_id(['A-2'])]
    assert a2.flag_text == 'proposed link · A-1 B-1'
    assert bool(a2.review) is True
    controller.do('reopen', cluster_id=cl.cluster_id(['A-1', 'B-1']))
    panel.refresh_list()
    track = panel._rows.set_index('cluster_id').loc[cl.cluster_id(['A-1', 'B-1'])]
    assert bool(track.review) is True
    assert track.status == 'proposed'
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the test starts on another track so show_track_of makes a real row change (native-same-day Task 8 fix round 1) | date: 2026-10-11]
def test_show_track_of_changes_no_state(qt):
    panel, _, state = _day_table_panel()
    t = panel.table
    a3 = next(i for i in range(t.rowCount()) if t.item(i, 0).data(QtCore.Qt.UserRole) == cl.cluster_id(['A-3']))
    t.setCurrentCell(a3, 0)
    assert panel.current_cluster()[0] == cl.cluster_id(['A-3'])
    assert state.selection == ['A-3']
    selection = list(state.selection)
    shown = []
    panel.track_shown.connect(shown.append)
    panel.show_track_of('B-1')
    assert panel.current_cluster()[0] == cl.cluster_id(['A-1', 'B-1'])
    listed = sorted(panel.members.item(i, 0).data(QtCore.Qt.UserRole) for i in range(panel.members.rowCount()))
    assert listed == ['A-1', 'B-1']
    assert state.selection == selection
    assert shown == []
    panel.close()
# [AGENT EDIT END]


def test_show_track_of_keeps_filters_when_hidden(qt):
    panel, _, _ = _day_table_panel()
    panel.only_review.setChecked(True)
    panel.show_track_of('A-3')
    assert panel.table.currentRow() == -1
    assert panel.members.rowCount() == 0
    assert panel.only_review.isChecked() is True
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table: status-ordered columns, the pick re-sort, the link marks, the wash and curation text, the status header, the step, and the merge, accept and inspect-only buttons (native-same-day Task 9) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day panel takes its run data (browse-same-day Task 2) | date: 2026-10-12]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the controller starts from the panel's run data; tag wraps the whole function (day-table-keys Task 2) | date: 2026-10-14]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the recorded select_pair takes the selection that stays selected; tag wraps the whole function (day-table-keys Task 3) | date: 2026-10-14]
def _day_panel(curates=True, data=None):
    """A ``DayTablePanel`` on ``data`` (``day_table_data()`` when ``None``), its controller (starting from that
    run's tracks) and state, and the recorded ``select_pair`` calls."""
    import dataclasses
    from multiphy.viewer.panels.tables import DayTablePanel
    data = day_table_data() if data is None else data
    controller = _Controller(data)
    state = PairState()
    state.set_data(data)
    opened = []
    actions = dataclasses.replace(_actions(controller),
                                  select_pair=lambda a, b, selection=None: opened.append((a, b)))
    panel = DayTablePanel(state, actions, curates)
    panel.show_run()
    return panel, controller, state, opened
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]


def _cell(panel, site, cellid, col=0):
    t = panel.tables[site]
    row = next(i for i in range(t.rowCount()) if t.item(i, 0).data(QtCore.Qt.UserRole) == cellid)
    return t.item(row, col)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the score is column 2 beside the new µm column; a column's texts top to bottom (day-table-keys Task 2) | date: 2026-10-14]
def _scores(panel, site):
    t = panel.tables[site]
    return [t.item(i, 2).text() for i in range(t.rowCount())]


def _texts(table, col):
    """Column ``col``'s texts of ``table``, top to bottom."""
    return [table.item(i, col).text() for i in range(table.rowCount())]
# [AGENT EDIT END]


def _fill(panel, cellid):
    from multiphy.viewer import theme
    return _cell(panel, cellid[0], cellid).data(theme.FILL_ROLE)


def _tone(panel, cellid):
    from multiphy.viewer import theme
    return _cell(panel, cellid[0], cellid).data(theme.TONE_ROLE)


def _bold(panel, cellid):
    return _cell(panel, cellid[0], cellid).font().bold()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day name shows its date over its site (day-table-keys Task 2) | date: 2026-10-14]
def test_day_columns_start_in_status_order(qt):
    panel, _, _, _ = _day_panel()
    assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    assert panel.column_ids('B') == ['B-2', 'B-1']
    assert panel.headers['A'].text() == '2023-01-01\nA'
    assert panel.headers['B'].text() == '2023-01-02\nB'
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: other days sort the track first, then by distance in the µm column, then by score (day-table-keys Task 2) | date: 2026-10-14]
def test_other_days_sort_track_then_distance(qt):
    panel, _, _, _ = _day_panel(data=string_days_data())
    panel.pick('A-1')
    assert panel.column_ids('B') == ['B-2', 'B-1', 'B-3']
    assert _texts(panel.tables['B'], 1) == ['20', '40', '190']
    assert _texts(panel.tables['B'], 2) == ['--', '0.90', '0.95']
    assert panel.column_ids('C') == ['C-2', 'C-1', 'C-3']
    assert _texts(panel.tables['C'], 1) == ['0', '35', '210']
    b = panel.tables['B']
    assert [b.horizontalHeaderItem(i).toolTip() for i in range(4)] == \
        ['', 'distance to A-1 · µm', 'score to A-1', 'status · u unmatched · m matched · a accepted']
    panel.close()
    panel, _, _, _ = _day_panel(data=day_table_data(depth=[100, 120, 400, 150, 90, 300, 110], x_um=[0] * 7))
    panel.pick('A-1')
    # the track mate B-1 at 200 µm before B-2 at 10 µm
    assert panel.column_ids('B') == ['B-1', 'B-2']
    panel.close()


def test_spread_shares_spare_width(qt):
    from PySide6 import QtTest
    from multiphy.viewer.panels.tables import days
    assert days.spread(244) == [90, 56, 56, 42]
    assert days.spread(172) == [64, 40, 40, 28]
    assert days.spread(186) == [69, 43, 43, 31]
    assert days.spread(100) == [64, 40, 40, 28]
    t = days._day_table()
    t.setRowCount(60)
    t.setFixedWidth(260)
    t.show()
    QtTest.QTest.qWait(50)
    assert t.viewport().width() == 244
    assert [t.columnWidth(i) for i in range(4)] == [90, 56, 56, 42]
    t.close()


def test_day_sort_signatures():
    import inspect
    from multiphy.viewer.panels.tables import DayTablePanel, days
    assert str(inspect.signature(days.spread)) == '(width: int) -> list'
    assert str(inspect.signature(DayTablePanel.reference)) == '(self, site: str) -> Optional[str]'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a pick keeps its own day's held order and re-sorts the other days (browse-same-day Task 2) | date: 2026-10-12]
def test_pick_keeps_own_day_resorts_others(qt):
    panel, _, state, _ = _day_panel()
    panel.pick('A-1')
    assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    assert _scores(panel, 'A') == ['0.90', '0.30', '--', '--', '--']
    assert panel.column_ids('B') == ['B-1', 'B-2']
    assert _scores(panel, 'B') == ['0.95', '0.97']
    assert state.selection == ['A-1']
    panel.pick('B-2')
    assert panel.column_ids('A') == ['A-1', 'A-3', 'A-2', 'A-4', 'A-5']
    assert panel.column_ids('B') == ['B-2', 'B-1']
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an accepted unit shows the status letter a (day-table widths) | date: 2026-10-13]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the status is column 3 beside the new µm column; tag wraps the whole function (day-table-keys Task 2) | date: 2026-10-14]
def test_actions_keep_held_order(qt):
    panel, _, _, _ = _day_panel()
    panel.pick('A-2')
    panel.buttons['Accept'].click()
    assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    assert _cell(panel, 'A', 'A-2', 3).text() == 'a'
    panel.close()
# [AGENT EDIT END]
# [AGENT EDIT END]


def test_unscored_units_sort_by_registered_distance(qt):
    depth, x_um = [100, 120, 400, 220, 150, 110, 300], [0, 0, 0, 0, 300, 0, 0]
    panel, _, _, _ = _day_panel(data=day_table_data(depth=depth, x_um=x_um, offsets={'A': 0.0, 'B': 25.0}))
    panel.pick('A-4')
    assert panel.column_ids('B') == ['B-1', 'B-2']
    panel.close()
    panel, _, _, _ = _day_panel(data=day_table_data(depth=depth, x_um=x_um, offsets=None))
    panel.pick('A-4')
    assert panel.column_ids('B') == ['B-2', 'B-1']
    panel.close()
# [AGENT EDIT END]


def test_unit_cells_mark_proposed_links(qt):
    panel, _, _, _ = _day_panel()
    a2 = _cell(panel, 'A', 'A-2')
    assert a2.text() == '2 ⇄'
    assert a2.toolTip() == 'proposed link · A-1 · 0.90\nproposed link · B-1 · 0.80'
    a3 = _cell(panel, 'A', 'A-3')
    assert a3.text() == '3'
    assert a3.toolTip() == ''
    panel.close()


def test_pick_draws_wash_and_curation_text(qt):
    panel, controller, _, _ = _day_panel()
    panel.pick('A-1')
    assert _fill(panel, 'A-1') == 'accent_tint'
    assert _fill(panel, 'A-2') == 'proposal_tint'
    assert _fill(panel, 'B-1') == 'proposal_tint'
    for c in ('A-3', 'A-4', 'A-5', 'B-2'):
        assert _fill(panel, c) is None
    assert [c for c in ('A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'B-1', 'B-2') if _bold(panel, c)] == ['A-1', 'B-1']
    assert [c for c in ('A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'B-1', 'B-2') if _tone(panel, c)] == []
    controller.do('accept', cluster_id=cl.cluster_id(['A-1', 'B-1']))
    panel.refresh()
    assert _tone(panel, 'A-1') == 'accept'
    assert _tone(panel, 'B-1') == 'accept'
    panel.close()
    panel, controller, _, _ = _day_panel()
    controller.do('reject', cluster_id=cl.cluster_id(['A-1', 'B-1']))
    panel.pick('A-1')
    assert _tone(panel, 'B-1') == 'reject'
    assert _bold(panel, 'B-1') is False
    assert _fill(panel, 'B-1') == 'proposal_tint'
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: header sorts become a column's held order; Up / Down walk the picked unit's day (browse-same-day Task 2) | date: 2026-10-12]
def _indicator(panel, site):
    header = panel.tables[site].horizontalHeader()
    return (header.sortIndicatorSection(), header.sortIndicatorOrder())


def _header_click(panel, site, section):
    panel.tables[site].horizontalHeader().sectionClicked.emit(section)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: sc and s are sections 2 and 3 beside the new µm column (day-table-keys Task 2) | date: 2026-10-14]
def test_header_sorts_become_held_order(qt):
    panel, _, _, _ = _day_panel()
    _header_click(panel, 'A', 0)
    assert panel.column_ids('A') == ['A-1', 'A-2', 'A-3', 'A-4', 'A-5']
    assert _indicator(panel, 'A') == (0, QtCore.Qt.AscendingOrder)
    _header_click(panel, 'A', 0)
    assert panel.column_ids('A') == ['A-5', 'A-4', 'A-3', 'A-2', 'A-1']
    assert _indicator(panel, 'A') == (0, QtCore.Qt.DescendingOrder)
    panel.pick('A-3')
    assert panel.column_ids('A') == ['A-5', 'A-4', 'A-3', 'A-2', 'A-1']
    assert _indicator(panel, 'A') == (0, QtCore.Qt.DescendingOrder)
    _header_click(panel, 'A', 2)
    assert panel.column_ids('A') == ['A-1', 'A-2', 'A-3', 'A-4', 'A-5']
    assert _indicator(panel, 'A') == (2, QtCore.Qt.DescendingOrder)
    _header_click(panel, 'A', 3)
    assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    assert _indicator(panel, 'A') == (3, QtCore.Qt.AscendingOrder)
    panel.close()


def test_header_sort_on_other_day_lasts_until_pick(qt):
    panel, _, _, _ = _day_panel()
    panel.pick('A-1')
    _header_click(panel, 'B', 3)
    assert panel.column_ids('B') == ['B-2', 'B-1']
    assert _indicator(panel, 'B')[0] == 3
    panel.pick('A-2')
    assert panel.column_ids('B') == ['B-1', 'B-2']
    assert _indicator(panel, 'B')[0] == -1
    panel.pick('B-1')
    assert panel.column_ids('B') == ['B-2', 'B-1']
    assert _indicator(panel, 'B')[0] == 3
    panel.close()
# [AGENT EDIT END]


def test_up_down_walk_own_day(qt):
    panel, _, _, opened = _day_panel()
    panel.pick('A-3')
    for d, picked in ((1, 'A-4'), (1, 'A-5'), (-3, 'A-2'), (-1, 'A-2')):
        panel.step(d)
        assert panel.picked_unit == picked
        assert opened == []
        assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    panel.toggle('B-2')
    assert opened == [('A-2', 'B-2')]
    panel.step(-1)
    assert opened[-1] == ('A-2', 'B-1')
    assert panel.picked_unit == 'A-2'
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the pinned same-day column: fill, sort, tooltips, clicks, steps, header sorts and signatures (browse-same-day Task 3) | date: 2026-10-12]
def _same_row(panel, cellid):
    t = panel.same_table
    return next(i for i in range(t.rowCount()) if t.item(i, 0).data(QtCore.Qt.UserRole) == cellid)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the score is column 2 beside the new µm column (day-table-keys Task 2) | date: 2026-10-14]
def _same_scores(panel):
    t = panel.same_table
    return [t.item(i, 2).text() for i in range(t.rowCount())]
# [AGENT EDIT END]


def _same_click(panel, cellid):
    panel.same_table.cellClicked.emit(_same_row(panel, cellid), 0)


def _same_header_click(panel, section):
    panel.same_table.horizontalHeader().sectionClicked.emit(section)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day header shows the day's date over its site (day-table-keys Task 2) | date: 2026-10-14]
def test_same_day_column_empty_until_pick(qt):
    panel, _, _, _ = _day_panel()
    assert panel.same_header.text() == 'same day'
    assert panel.same_column_ids() == []
    panel.pick('A-1')
    assert panel.same_header.text() == 'same day · 2023-01-01\nA'
    panel.pick('B-1')
    assert panel.same_column_ids() == ['B-2']
    assert _same_scores(panel) == ['0.10']
    panel.resize(900, 400)
    panel.show()
    qt.processEvents()
    assert panel.same_table.mapTo(panel, QtCore.QPoint(0, 0)).x() < panel.scroll.mapTo(panel, QtCore.QPoint(0, 0)).x()
    assert panel.scroll.widget().isAncestorOf(panel.same_table) is False
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day column sorts by distance then score, shows the distance in µm, and its unit hover lists only the link lines (day-table-keys Task 2) | date: 2026-10-14]
def test_same_day_column_sorts_distance_then_score(qt):
    panel, _, _, _ = _day_panel(data=day_table_data(depth=[100, 120, 400, 220, 150, 110, 300],
                                                    x_um=[0, 0, 0, 0, 300, 0, 0]))
    panel.pick('A-1')
    t = panel.same_table
    assert panel.same_column_ids() == ['A-2', 'A-4', 'A-3', 'A-5']
    assert _texts(t, 1) == ['20', '120', '300', '304']
    assert _texts(t, 2) == ['0.90', '--', '0.30', '--']
    assert [t.item(i, 0).toolTip() for i in range(4)] == \
        ['proposed link · A-1 · 0.90\nproposed link · B-1 · 0.80', '', '', '']
    panel.close()
    panel, _, _, _ = _day_panel()
    panel.pick('A-1')
    assert panel.same_table.item(_same_row(panel, 'A-4'), 1).text() == '--'
    panel.close()
# [AGENT EDIT END]


def test_same_day_click_toggles_and_steps(qt):
    panel, _, _, opened = _day_panel()
    panel.pick('A-1')
    _same_click(panel, 'A-3')
    assert panel.selected == ['A-1', 'A-3']
    assert opened == [('A-1', 'A-3')]
    assert panel.picked_unit == 'A-1'
    assert panel.same_column_ids() == ['A-2', 'A-3', 'A-4', 'A-5']
    assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    _same_click(panel, 'A-3')
    assert panel.selected == ['A-1']
    _same_click(panel, 'A-2')
    assert opened[-1] == ('A-1', 'A-2')
    panel.step(1)
    assert opened[-1] == ('A-1', 'A-3')
    panel.step(-1)
    assert opened[-1] == ('A-1', 'A-2')
    assert panel.picked_unit == 'A-1'
    assert panel.same_column_ids() == ['A-2', 'A-3', 'A-4', 'A-5']
    assert panel.column_ids('A') == ['A-2', 'A-3', 'A-4', 'A-5', 'A-1']
    panel.close()


def test_same_day_header_sorts_until_pick(qt):
    panel, _, _, _ = _day_panel()
    panel.pick('A-1')
    header = panel.same_table.horizontalHeader()
    _same_header_click(panel, 0)
    assert panel.same_column_ids() == ['A-2', 'A-3', 'A-4', 'A-5']
    assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == (0, QtCore.Qt.AscendingOrder)
    _same_header_click(panel, 0)
    assert panel.same_column_ids() == ['A-5', 'A-4', 'A-3', 'A-2']
    assert (header.sortIndicatorSection(), header.sortIndicatorOrder()) == (0, QtCore.Qt.DescendingOrder)
    panel.pick('A-2')
    assert panel.same_column_ids() == ['A-1', 'A-3', 'A-4', 'A-5']
    assert header.sortIndicatorSection() == -1
    panel.close()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same header again reverses the held same-day list, never a fresh sort (browse-same-day Task 3 fix round 1) | date: 2026-10-12]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the status header is section 3 beside the new µm column; tag wraps the whole function (day-table-keys Task 2) | date: 2026-10-14]
def test_same_day_header_again_reverses_held_list(qt):
    panel, controller, _, _ = _day_panel()
    panel.pick('A-2')
    _same_header_click(panel, 3)
    assert panel.same_column_ids() == ['A-3', 'A-4', 'A-5', 'A-1']
    controller.do('reject', cluster_id=cl.cluster_id(['A-1', 'B-1']))
    panel.refresh()
    assert panel.same_column_ids() == ['A-3', 'A-4', 'A-5', 'A-1']
    _same_header_click(panel, 3)
    assert panel.same_column_ids() == ['A-1', 'A-5', 'A-4', 'A-3']
    panel.close()
# [AGENT EDIT END]
# [AGENT EDIT END]

def test_same_day_signatures():
    import inspect
    from multiphy.viewer.panels.tables import DayTablePanel
    assert str(inspect.signature(DayTablePanel.same_column_ids)) == '(self) -> list'
    assert str(inspect.signature(DayTablePanel.toggle)) == '(self, cellid: str, *, same_day: bool = False) -> None'
# [AGENT EDIT END]

def test_day_merge_joins_whole_tracks(qt):
    import json
    panel, controller, _, _ = _day_panel()
    panel.pick('A-3')
    panel.toggle('B-1')
    panel.buttons['Merge'].click()
    assert json.loads(controller.rows[-1]['parent_ids']) == [cl.cluster_id(['A-3']), cl.cluster_id(['A-1', 'B-1'])]
    merged = controller.state.tracks[cl.cluster_id(['A-1', 'A-3', 'B-1'])]
    assert merged['members'] == ['A-1', 'A-3', 'B-1']
    panel.close()


def test_day_accept_acts_on_picked_track(qt):
    panel, controller, _, _ = _day_panel()
    panel.pick('A-2')
    panel.toggle('B-1')
    panel.buttons['Accept'].click()
    assert controller.state.tracks[cl.cluster_id(['A-2'])]['status'] == 'accepted'
    assert controller.state.tracks[cl.cluster_id(['A-1', 'B-1'])]['status'] == 'proposed'
    panel.close()


def test_inspect_only_day_table_hides_buttons(qt):
    panel, _, _, _ = _day_panel(curates=False)
    assert list(panel.buttons) == ['Accept', 'Reject', 'Merge', 'Unmatch unit']
    assert not any(b.isVisibleTo(panel) for b in panel.buttons.values())
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: day-table number reading leaves the run's columns untouched and reads nullable and string cells as missing (native-same-day Task 9 fix round 1) | date: 2026-10-11]
def test_day_numbers_copy_and_read_missing():
    import numpy as np
    from multiphy.viewer.panels.tables.days import _numbers
    source = pd.Series([0.8, np.inf, -np.inf])
    out = _numbers(source)
    assert source.tolist() == [0.8, np.inf, -np.inf]
    assert out[0] == 0.8 and np.isnan(out[1:]).all()
    nullable = _numbers(pd.Series([0.5, pd.NA], dtype='Float64'))
    assert nullable[0] == 0.5 and np.isnan(nullable[1])
    assert np.isnan(_numbers(pd.Series([1, pd.NA], dtype='Int64'))[1])
    text = _numbers(pd.Series(['0.4', 'x'], dtype='string'))
    assert text[0] == 0.4 and np.isnan(text[1])
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an accepted or rejected track leaves the review queue for every flag; the flags stay (D35) (native-same-day Task 12) | date: 2026-10-12]
def test_decided_track_leaves_review(qt):
    merged = cl.cluster_id(['A-1', 'A-2', 'B-1'])
    panel, controller, _ = _day_table_panel()
    controller.do('merge', parent_ids=[cl.cluster_id(['A-1', 'B-1']), cl.cluster_id(['A-2'])],
                  members=['A-1', 'B-1', 'A-2'])
    panel.refresh_list()
    track = panel._rows.set_index('cluster_id').loc[merged]
    assert track.flag_text == '2 units on one day, proposed link · A-1 A-2 B-1'
    assert bool(track.review) is True
    assert track.status == 'proposed'
    controller.do('accept', cluster_id=merged)
    panel.refresh_list()
    track = panel._rows.set_index('cluster_id').loc[merged]
    assert track.flag_text == '2 units on one day, proposed link · A-1 A-2 B-1'
    assert bool(track.review) is False
    assert track.status == 'accepted'
    panel.close()

    panel, controller, _ = _day_table_panel()
    controller.do('merge', parent_ids=[cl.cluster_id(['A-1', 'B-1']), cl.cluster_id(['A-2'])],
                  members=['A-1', 'B-1', 'A-2'])
    controller.do('reject', cluster_id=merged)
    panel.refresh_list()
    rows = panel._rows.set_index('cluster_id')
    track = rows.loc[merged]
    assert bool(track.review) is False
    assert track.status == 'rejected'
    assert bool(rows.loc[cl.cluster_id(['A-2'])].review) is False
    panel.close()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day column is fixed to its three columns plus frame and scrollbar (browse-same-day Task 6) | date: 2026-10-12]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the narrower widths 64, 40 and 28 (148 = 132 + frame 2 + offscreen scrollbar 14); the last column stretches; tag wraps the whole function (day-table widths) | date: 2026-10-13]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: four columns 64, 40, 40 and 28 (188 = 172 + frame 2 + offscreen scrollbar 14) share the spare width; tag wraps the whole function (day-table-keys Task 2) | date: 2026-10-14]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the class docstring names each day's reference unit; tag wraps the whole function (day-table-keys Task 3) | date: 2026-10-14]
def test_same_day_column_fits_its_columns(qt):
    from PySide6 import QtTest
    from multiphy.viewer.panels.tables import DayTablePanel
    panel, _, _, _ = _day_panel()
    panel.resize(900, 400)
    panel.show()
    qt.processEvents()
    t = panel.same_table
    assert t.width() == 188
    assert not t.horizontalScrollBar().isVisible()
    panel.pick('A-1')
    assert t.width() == 188
    assert [t.columnWidth(i) for i in range(4)] == [69, 43, 43, 31]
    t.setRowCount(60)
    QtTest.QTest.qWait(50)
    assert t.width() == 188
    assert t.verticalScrollBar().isVisible()
    assert not t.horizontalScrollBar().isVisible()
    assert t.viewport().width() == 172
    assert [t.columnWidth(i) for i in range(4)] == [64, 40, 40, 28]
    assert DayTablePanel.__doc__ == ("One column per day of every unit: the picked unit's own day in its held order, "
                                     "every other day sorted to its reference unit, and the picked unit's other "
                                     "own-day units in the pinned same-day column.")
    panel.close()
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: short headers with hover keys, one status letter with its word on hover, and a day column never narrower than its three columns (day-table widths) | date: 2026-10-13]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: four headers with the µm hover; the status letter is column 3 (day-table-keys Task 2) | date: 2026-10-14]
def test_day_headers_and_status_letters(qt):
    panel, _, _, _ = _day_panel()
    tips = ['', 'distance to the picked unit · µm', 'score to the picked unit',
            'status · u unmatched · m matched · a accepted']
    for t in [panel.same_table, *panel.tables.values()]:
        assert [t.horizontalHeaderItem(i).text() for i in range(4)] == ['unit', 'µm', 'sc', 's']
        assert [t.horizontalHeaderItem(i).toolTip() for i in range(4)] == tips
    assert [(_cell(panel, c[0], c, 3).text(), _cell(panel, c[0], c, 3).toolTip()) for c in ('A-1', 'B-1', 'A-3')] \
        == [('m', 'matched'), ('m', 'matched'), ('u', 'unmatched')]
    panel.pick('A-2')
    panel.buttons['Accept'].click()
    assert (_cell(panel, 'A', 'A-2', 3).text(), _cell(panel, 'A', 'A-2', 3).toolTip()) == ('a', 'accepted')
    panel.pick('A-1')
    rows = {panel.same_table.item(i, 0).data(QtCore.Qt.UserRole): panel.same_table.item(i, 3).text()
            for i in range(panel.same_table.rowCount())}
    assert rows == {'A-2': 'a', 'A-3': 'u', 'A-4': 'u', 'A-5': 'u'}
    panel.close()


def test_day_columns_keep_their_narrowest_width_and_scroll(qt):
    from multiphy.viewer.panels.tables import days
    panel, _, _, _ = _day_panel()
    panel.resize(panel.same_table.width() + 120, 400)
    panel.show()
    qt.processEvents()
    assert len(panel.tables) == 2 and panel.scroll.viewport().width() < 2 * 188
    assert panel.scroll.horizontalScrollBar().isVisible()
    for t in panel.tables.values():
        assert t.width() == 188
        assert not t.horizontalScrollBar().isVisible()
    panel.resize(1200, 400)
    qt.processEvents()
    assert not panel.scroll.horizontalScrollBar().isVisible()
    for t in panel.tables.values():
        assert t.width() > 188
        assert sum(t.columnWidth(i) for i in range(4)) == t.viewport().width()
        for i in range(4):
            assert t.columnWidth(i) > days.WIDTHS[i], i
    panel.close()


def test_day_widths_fit_under_the_theme(qt):
    """Under the app style sheet each of the four headers shows unelided beside its sort arrow, and ``318-1 ⇄``,
    ``9999``, ``0.97`` and every status letter fit their columns at ``WIDTHS``."""
    from multiphy.viewer import display, theme
    from multiphy.viewer.panels.tables import days
    panel, _, _, _ = _day_panel()
    panel.setStyleSheet(theme.stylesheet(display.PALETTES['graphite'], 'c', 'v'))
    panel.show()
    qt.processEvents()
    t = panel.tables['A']
    h = t.horizontalHeader()
    assert len(days.WIDTHS) == 4
    for section, width in enumerate(days.WIDTHS):
        h.setSortIndicatorShown(True)
        h.setSortIndicator(section, QtCore.Qt.AscendingOrder)
        opt = QtWidgets.QStyleOptionHeader()
        h.initStyleOption(opt)
        opt.rect = QtCore.QRect(0, 0, width, h.height())
        opt.section = section
        opt.text = t.horizontalHeaderItem(section).text()
        opt.sortIndicator = QtWidgets.QStyleOptionHeader.SortUp
        label = h.style().subElementRect(QtWidgets.QStyle.SE_HeaderLabel, opt, h)
        assert h.fontMetrics().horizontalAdvance(opt.text) <= label.width(), opt.text
    assert t.font().family() == theme.fonts()['qt_mono']
    margin = 2 * (t.style().pixelMetric(QtWidgets.QStyle.PM_FocusFrameHMargin, None, t) + 1)
    fm = t.fontMetrics()
    assert fm.horizontalAdvance('318-1 ⇄') + margin <= days.WIDTHS[0]
    assert fm.horizontalAdvance('9999') + margin <= days.WIDTHS[1]
    assert fm.horizontalAdvance('0.97') + margin <= days.WIDTHS[2]
    assert max(fm.horizontalAdvance(x) for x in days.STATUS_LETTERS.values()) + margin <= days.WIDTHS[3]
    panel.close()
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one strung unit per other day: move, step, drop, clear, string a track, Ctrl-click, same-day extras and the accent header (day-table-keys Task 3) | date: 2026-10-14]
def _tones(panel):
    """Each day header's ``tone`` property, in day order."""
    return [panel.headers[s].property('tone') for s in panel.tables]


def test_wasd_strings_units_across_days(qt):
    from multiphy.viewer import display, theme
    panel, _, state, opened = _day_panel(data=string_days_data())
    panel.pick('A-1')
    assert list(panel.tables) == ['A', 'B', 'C']
    assert _tones(panel) == ['accent', '', '']
    panel.move_day(1)
    assert panel.strung == {'B': 'B-2'}
    assert opened[-1] == ('A-1', 'B-2')
    assert panel.column_ids('C') == ['C-1', 'C-2', 'C-3']
    assert _texts(panel.tables['C'], 1) == ['15', '20', '190']
    assert _texts(panel.tables['C'], 2) == ['0.80', '--', '--']
    assert _tones(panel) == ['', 'accent', '']
    c = panel.tables['C']
    assert c.horizontalHeaderItem(1).toolTip() == 'distance to B-2 · µm'
    assert c.horizontalHeaderItem(2).toolTip() == 'score to B-2'
    panel.move_day(1)
    assert opened[-1] == ('B-2', 'C-1')
    n = len(opened)
    panel.move_day(1)
    assert len(opened) == n
    panel.move_day(-1)
    assert opened[-1] == ('A-1', 'B-2')
    panel.step(1)
    assert opened[-1] == ('A-1', 'B-1')
    assert panel.column_ids('C') == ['C-2', 'C-1', 'C-3']
    assert panel.strung['C'] == 'C-1'
    panel.move_day(-1)
    assert state.selection == ['A-1', 'B-1', 'C-1']
    panel.step(1)
    assert panel.picked_unit == 'A-2'
    assert panel.selected == ['A-2']
    assert 'QLabel[tone="accent"] { color: #5B9DFF; }' in \
        theme.stylesheet(display.PALETTES['graphite'], 'c', 'v')
    panel.close()


def test_drop_and_clear_string(qt):
    panel, _, state, _ = _day_panel(data=string_days_data())
    panel.pick('A-1')
    panel.move_day(1)
    panel.move_day(1)
    panel.move_day(-1)
    panel.drop_day()
    assert panel.strung == {'C': 'C-1'}
    assert panel.active_site == 'B'
    assert state.selection == ['A-1', 'C-1']
    assert panel.column_ids('C') == ['C-2', 'C-1', 'C-3']
    assert _texts(panel.tables['C'], 1) == ['0', '35', '210']
    panel.clear_string()
    assert panel.selected == ['A-1']
    assert panel.active_site == 'A'
    panel.close()


def test_track_key_strings_current_track(qt):
    panel, controller, state, _ = _day_panel(data=string_days_data())
    panel.pick('A-2')
    panel.string_track()
    assert panel.strung == {'B': 'B-3', 'C': 'C-3'}
    assert state.selection == ['A-2', 'B-3', 'C-3']
    panel.pick('A-1')
    panel.string_track()
    assert panel.status.text() == 'no link'
    controller.do('merge', parent_ids=[cl.cluster_id(['A-2', 'B-3', 'C-3']), cl.cluster_id(['A-1']),
                                       cl.cluster_id(['B-1'])],
                  members=['A-2', 'B-3', 'C-3', 'A-1', 'B-1'])
    panel.pick('A-2')
    panel.string_track()
    assert panel.strung == {'B': 'B-1', 'C': 'C-3'}
    assert panel.extras == ['A-1', 'B-3']
    assert panel.selected == ['A-2', 'B-1', 'C-3', 'A-1', 'B-3']
    assert panel.status.text() == ''
    panel.close()


def test_ctrl_click_sets_or_drops_day_unit(qt):
    panel, _, state, opened = _day_panel(data=string_days_data())
    panel.pick('A-1')
    panel.toggle('C-3')
    assert opened[-1] == ('A-1', 'C-3')
    panel.toggle('C-1')
    assert panel.strung == {'C': 'C-1'}
    assert opened[-1] == ('A-1', 'C-1')
    panel.toggle('C-1')
    assert panel.strung == {}
    assert state.selection == ['A-1']
    panel.close()


def test_same_day_extra_beside_string(qt):
    panel, _, state, opened = _day_panel()
    panel.pick('A-1')
    panel.toggle('B-2')
    _same_click(panel, 'A-3')
    assert panel.selected == ['A-1', 'B-2', 'A-3']
    assert opened[-1] == ('A-1', 'A-3')
    panel.step(1)
    assert opened[-1] == ('A-1', 'A-4')
    assert panel.strung == {'B': 'B-2'}
    _same_click(panel, 'A-4')
    assert state.selection == ['A-1', 'B-2']
    panel.close()


def test_string_signatures():
    import inspect
    from multiphy.viewer.app import App
    from multiphy.viewer.panels.tables import DayTablePanel
    assert str(inspect.signature(DayTablePanel.move_day)) == '(self, d: int) -> None'
    assert str(inspect.signature(DayTablePanel.drop_day)) == '(self) -> None'
    assert str(inspect.signature(DayTablePanel.clear_string)) == '(self) -> None'
    assert str(inspect.signature(DayTablePanel.string_track)) == '(self) -> None'
    assert str(inspect.signature(PairState.set_pair)) == '(self, row, selection: Optional[list] = None) -> None'
    assert str(inspect.signature(App.select_pair)) == \
        '(self, cellid_a: str, cellid_b: str, selection: Optional[list] = None) -> None'


def test_reselect_signature():
    import inspect
    from multiphy.viewer.panels.tables import DayTablePanel
    assert str(inspect.signature(DayTablePanel.reselect)) == '(self) -> None'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a step in the own day scrolls its column into view (day-table-keys Task 3 fix round 1) | date: 2026-10-14]
def test_own_day_step_scrolls_its_column_into_view(qt):
    panel, _, _, opened = _day_panel()
    panel.resize(panel.same_table.width() + 330, 400)
    panel.show()
    qt.processEvents()
    bar = panel.scroll.horizontalScrollBar()
    assert bar.isVisible()
    panel.pick('A-3')
    bar.setValue(bar.maximum())
    assert bar.value() > 0
    panel.step(1)
    assert panel.picked_unit == 'A-4'
    assert opened == []
    assert bar.value() == 0
    panel.close()
# [AGENT EDIT END]
