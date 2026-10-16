"""Tracked units: the object a curator actually works on."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from multiphy import clusters as cl
from multiphy.viewer.data import Words
from multiphy.viewer import theme
from multiphy.viewer.panels.tables.base import NumericItem, _table
from multiphy.viewer.state import Actions, PairState
from multiphy.viewer.qt import QtCore, QtWidgets


class ClusterPanel(QtWidgets.QWidget):
    """Tracked units: the object a curator actually works on.

    A well holds thousands of candidate pairs but only tens of clusters, and
    most of those are clean. The queue is driven by the flags that cannot be
    resolved automatically -- a cluster claiming one neuron twice on a day,
    members joined only through a third unit, and a proposed link the backend
    saved (the partner may be on the same day or another day; a single
    unmatched unit included) -- so review time goes where
    the evidence is actually in tension. The flags never clear. An accepted
    or rejected track leaves the review queue for every flag; the flags stay.
    Reopening it, or merging it into a new ``proposed`` track, queues it
    again. A unit a rejected track left ``unmatched`` stays out of the queue
    for its proposed link too.
    """
    panel_id, panel_title = 'clusters', 'Tracked units'
    track_shown = QtCore.Signal(list)

    def __init__(self, state: PairState, actions: Actions, words: Words, parent=None):
        super().__init__(parent)
        self.state, self.actions = state, actions
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)

        top = QtWidgets.QHBoxLayout()
        self.only_review = QtWidgets.QCheckBox('needing review only')
        self.only_review.stateChanged.connect(self.refresh_list)
        top.addWidget(self.only_review)
        self.hide_done = QtWidgets.QCheckBox('hide decided')
        self.hide_done.stateChanged.connect(self.refresh_list)
        top.addWidget(self.hide_done)
        top.addStretch(1)
        self.inspect_tag = QtWidgets.QLabel('inspect-only')
        self.inspect_tag.setToolTip('same-day runs are not curated')
        self.inspect_tag.setProperty('role', 'hint')
        self.inspect_tag.hide()
        top.addWidget(self.inspect_tag)
        self.skipped_tag = QtWidgets.QLabel('')
        self.skipped_tag.setProperty('tone', 'unsure')
        self.skipped_tag.setStyleSheet('font-size:11px;')
        self.skipped_tag.hide()
        top.addWidget(self.skipped_tag)
        self.lock_tag = QtWidgets.QLabel('')
        self.lock_tag.setProperty('tone', 'unsure')
        self.lock_tag.setStyleSheet('font-size:11px;')
        self.lock_tag.hide()
        top.addWidget(self.lock_tag)
        self.take_over_btn = QtWidgets.QPushButton('Take over')
        self.confirm_take_over_btn = QtWidgets.QPushButton('Confirm take over')
        self.confirm_take_over_btn.setToolTip('the other viewer stops saving')
        self.cancel_take_over_btn = QtWidgets.QPushButton('Cancel')
        for b in (self.take_over_btn, self.confirm_take_over_btn, self.cancel_take_over_btn):
            b.hide()
            top.addWidget(b)
        self.confirming_take_over = False
        self.take_over_btn.clicked.connect(lambda: self._ask_take_over(True))
        self.cancel_take_over_btn.clicked.connect(lambda: self._ask_take_over(False))
        self.confirm_take_over_btn.clicked.connect(lambda: self.actions.take_over())
        self.count = QtWidgets.QLabel('')
        self.count.setProperty('role', 'hint')
        top.addWidget(self.count)
        lay.addLayout(top)

        self.table = _table([words.track_column, 'units', 'flags', 'weakest', 'status'],
                            [44, 44, 104, 58, 62])
        self.table.setMinimumWidth(1)
        self.table.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.table.currentCellChanged.connect(lambda r, *a: self.on_cluster(r))
        lay.addWidget(self.table, 2)

        members_row = QtWidgets.QHBoxLayout()
        members_row.addWidget(QtWidgets.QLabel('<b>Members</b>'))
        members_hint = QtWidgets.QLabel('click one to open its strongest link')
        members_hint.setProperty('role', 'hint')
        members_row.addWidget(members_hint)
        members_row.addStretch(1)
        lay.addLayout(members_row)
        self.members = _table(['unit', words.member_column, 'best link'], [96, 82, 72])
        self.members.horizontalHeader().setSectionResizeMode(1, QtWidgets.QHeaderView.ResizeToContents)
        self.members.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.members.setMinimumWidth(1)
        self.members.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Expanding)
        self.members.cellClicked.connect(self._member_clicked)
        self.members.itemSelectionChanged.connect(self._members_selected)
        lay.addWidget(self.members, 1)

        # a single row of five buttons sets a wide minimum on the panel, which
        # forces the whole dock wide and squeezes the plots out; a grid does not
        grid = QtWidgets.QGridLayout()
        grid.setSpacing(4)
        self.buttons = {}
        verdicts = {'Accept': 'accept', 'Reject': 'reject'}
        for n, (label, tip, slot) in enumerate((
                ('Accept', 'accept this tracked unit', self.accept),
                ('Reject', 'reject this tracked unit', self.reject),
                ('Split selected', 'Separate selected members from the remainder; singletons stay unmatched', self.split_selected),
                ('Unmatch unit', 'Detach the selected member; keep it available as unmatched', self.drop_member),
                ('Merge', 'merge the clusters selected above', self.merge_selected),
                ('Reopen', 'back to undecided', self.reopen),
                )):
            b = QtWidgets.QPushButton(label)
            b.setToolTip(tip)
            b.setMinimumWidth(1)
            b.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Fixed)
            if label in verdicts:
                b.setProperty('verdict', verdicts[label])
            b.setStyleSheet('padding:6px 4px;')
            b.clicked.connect(slot)
            grid.addWidget(b, n // 3, n % 3)
            self.buttons[label] = b
        lay.addLayout(grid)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.status.setProperty('role', 'hint')
        lay.addWidget(self.status)

    # ------------------------------------------------------------- rendering
    def refresh_list(self, *_args, preferred_id=None, fallback=True) -> None:
        """Rebuild the list and make the row of ``preferred_id`` (else the previous row) current.

        Without a shown row holding it, the first row is current; with ``fallback=False`` no row is current, the
        Members table is cleared and nothing is emitted, so no ``PairState`` changes.
        """
        previous = self.current_cluster()[0] if hasattr(self, '_rows') else None
        preferred_id = preferred_id or previous
        if self.state.data is None:
            return
        # rows are the controller's held tracks, plus its rejected tracks while decided rows are shown
        track_state = self.actions.track_state()
        held = [SimpleNamespace(cluster_id=tid, members=list(t['members']), status=t['status'])
                for tid, t in track_state.tracks.items()]
        if not self.hide_done.isChecked():
            held += [SimpleNamespace(cluster_id=tid, members=list(members), status='rejected')
                     for tid, members in track_state.rejected.items()]
        self._update_skipped(track_state.skipped)
        lookup, threshold = self.actions.cluster_lookup(), self.actions.review_threshold()
        links = cl.proposed_links(self.state.data.pairs())
        # every unit of a rejected track: the reject was the curator's decision, so its units stay out of the queue
        settled = {m for ms in track_state.rejected.values() for m in ms}
        rows = []
        for r in held:
            info = cl.describe(r.members, lookup, sites=self.state.sites, threshold=threshold)
            marks = []
            if info['same_day_collision']:
                marks.append(f"{info['max_per_day']} units on one day")
            if info['weak_link']:
                marks.append('weak internal link')
            # members are not removed from the partners, so the flag never clears (D16)
            partners = sorted({p for m in r.members for p in links.get(m, [])})
            if partners:
                marks.append('proposed link · ' + ' '.join(partners))
            # a decision (accepted or rejected) takes the track out of the queue for every flag (D35)
            review = r.status in ('proposed', 'unmatched') and (cl.needs_review(info) or (bool(partners) and (
                r.status == 'proposed' or not set(r.members) & settled)))
            rows.append(dict(cluster_id=r.cluster_id, members=r.members,
                             status=r.status, flag_text=', '.join(marks) or ('unmatched' if len(r.members) == 1 else 'clean'),
                             review=review, **info))
        self._rows = pd.DataFrame(rows)
        show = self._rows
        if len(show):
            if self.only_review.isChecked():
                show = show[show.review]
            if self.hide_done.isChecked():
                show = show[show.status.isin(('proposed', 'unmatched'))]
            show = show.sort_values(['review', 'n_days', 'n_members'],
                                    ascending=[False, False, False])
        self._shown = show.reset_index(drop=True)
        t = self.table
        blocker = QtCore.QSignalBlocker(t)
        t.clearSelection()
        t.setSortingEnabled(False)
        t.setRowCount(len(self._shown))
        for i, r in self._shown.iterrows():
            cells = [NumericItem(r.n_days, str(r.n_days)),
                     NumericItem(r.n_members, str(r.n_members)),
                     QtWidgets.QTableWidgetItem(r.flag_text),
                     NumericItem(r.min_internal_posterior),
                     QtWidgets.QTableWidgetItem(r.status)]
            cells[0].setData(QtCore.Qt.UserRole, r.cluster_id)
            if r.review:
                theme.tone(cells[2], 'unsure')
            if r.status == 'accepted':
                theme.tone(cells[4], 'accept')
            elif r.status == 'rejected':
                theme.tone(cells[4], 'reject')
            for j, it in enumerate(cells):
                t.setItem(i, j, it)
        t.setSortingEnabled(True)
        n_rev = int(self._rows.review.sum()) if len(self._rows) else 0
        n_done = int(self._rows.status.isin(('accepted', 'rejected')).sum()) if len(self._rows) else 0
        self.count.setText(f'{len(self._rows)} clusters, {n_rev} need review, {n_done} decided')
        target = next((i for i in range(t.rowCount())
                       if t.item(i, 0).data(QtCore.Qt.UserRole) == preferred_id), None)
        if target is None and not fallback:
            # the preferred row is not shown: no row is current and no state changes (D36)
            t.setCurrentCell(-1, -1)
            del blocker
            self._fill_members([])
            self._controls_changed()
            return
        target = 0 if target is None else target
        if t.rowCount():
            t.setCurrentCell(target, 0)
            t.selectRow(target)
        else:
            t.setCurrentCell(-1, -1)
        del blocker
        # A rebuild can leave the same row current, so Qt emits no change.
        self.on_cluster(t.currentRow())

    def _update_skipped(self, skipped):
        if not skipped:
            self.skipped_tag.hide()
            return
        self.skipped_tag.setText(f'skipped {len(skipped)}')
        self.skipped_tag.setToolTip('\n'.join(f'{op_id or "(no op_id)"} · {reason}' for op_id, reason in skipped))
        self.skipped_tag.show()

    def _ask_take_over(self, confirming):
        """Take over asks in the page; Cancel goes back."""
        self.confirming_take_over = confirming
        self._controls_changed()

    def _controls_changed(self):
        """Ask the window to set its mutation controls."""
        self.actions.update_controls()

    def current_status(self):
        """The status of the current row, or None when no row is current."""
        cid, _ = self.current_cluster()
        if not cid or not len(self._rows):
            return None
        hit = self._rows[self._rows.cluster_id == cid]
        return hit.iloc[0].status if len(hit) else None

    def selected_ids(self):
        ids = []
        for idx in self.table.selectionModel().selectedRows():
            it = self.table.item(idx.row(), 0)
            if it is not None:
                ids.append(it.data(QtCore.Qt.UserRole))
        return [i for i in ids if i]

    def current_cluster(self):
        r = self.table.currentRow()
        it = self.table.item(r, 0) if r >= 0 else None
        if it is None:
            return None, []
        cid = it.data(QtCore.Qt.UserRole)
        hit = self._rows[self._rows.cluster_id == cid]
        return cid, (list(hit.iloc[0].members) if len(hit) else [])

    def on_cluster(self, _row):
        cid, members = self.current_cluster()
        self._fill_members(members)
        # the controls follow the current row's unit count and status
        self._controls_changed()
        self.state.set_selection(members)
        if len(members) == 2:
            self._open_best_pair(members[0], members)
        if cid is not None:
            self.track_shown.emit(list(members))

    def show_track_of(self, cellid: str) -> None:
        """Make current the held track holding ``cellid``; a row the filters hide leaves no row current.

        The filters stay as they are; no ``PairState`` changes and nothing is emitted.
        """
        rows = getattr(self, '_rows', pd.DataFrame())
        hit = rows[(rows.status != 'rejected') & rows.members.map(lambda ms: cellid in ms)] if len(rows) else rows
        cid = hit.iloc[0].cluster_id if len(hit) else None
        members = list(hit.iloc[0].members) if len(hit) else []
        t = self.table
        target = next((i for i in range(t.rowCount())
                       if cid is not None and t.item(i, 0).data(QtCore.Qt.UserRole) == cid), None)
        blocker = QtCore.QSignalBlocker(t)
        t.clearSelection()
        if target is None:
            t.setCurrentCell(-1, -1)
            members = []
        else:
            t.setCurrentCell(target, 0)
            t.selectRow(target)
        del blocker
        self._fill_members(members)
        self._controls_changed()

    def _fill_members(self, members):
        """List ``members`` in the members table, each with its day and best link."""
        blocker = QtCore.QSignalBlocker(self.members)
        self.members.clearSelection()
        self.members.setSortingEnabled(False)
        self.members.setRowCount(len(members))
        look = self.actions.cluster_lookup()
        sites = self.state.sites
        for i, m in enumerate(sorted(members)):
            scores = [look[(m, o)] for o in members if (m, o) in look]
            best = max(scores, default=np.nan)
            cells = [QtWidgets.QTableWidgetItem(cl.unit_label(m, sites[m])),
                     QtWidgets.QTableWidgetItem(self.state.data.day_name(sites[m])),
                     NumericItem(best, '--' if not np.isfinite(best) else f'{best:.2f}')]
            cells[0].setData(QtCore.Qt.UserRole, m)
            for j, it in enumerate(cells):
                self.members.setItem(i, j, it)
        self.members.setSortingEnabled(True)
        self.members.setCurrentCell(-1, -1)
        del blocker

    def selected_members(self):
        return [self.members.item(i.row(), 0).data(QtCore.Qt.UserRole)
                for i in self.members.selectionModel().selectedRows()
                if self.members.item(i.row(), 0) is not None]

    def _members_selected(self):
        chosen = self.selected_members()
        if not chosen:
            _, chosen = self.current_cluster()
        if len(chosen) == 2:
            self._open_best_pair(chosen[0], chosen)
        else:
            self.state.set_selection(chosen)

    def _member_clicked(self, row, _col):
        # itemSelectionChanged already handles every selection size, including
        # Ctrl-click shrinking a multiday selection back to one or two units.
        self._members_selected()

    def _open_best_pair(self, member, members):
        """Open measured evidence synchronously; otherwise show only these units."""
        pairs = self.state.pairs
        others = set(members) - {member}
        if pairs is not None and len(pairs):
            hit = pairs[((pairs.cellid_a == member) & pairs.cellid_b.isin(others)) |
                        ((pairs.cellid_b == member) & pairs.cellid_a.isin(others))]
            if len(hit):
                row = hit.sort_values(cl.score_column(hit), ascending=False).iloc[0]
                self.state.day_pair = next((dp for dp in self.state.data.day_pairs()
                    if {dp['siteid_a'], dp['siteid_b']} ==
                    {row.siteid_a, row.siteid_b}), None)
                self.state.set_pair(row)
                return
        self.state.set_selection(members)

    def _do(self, operation, **fields):
        """Run one action through the controller; True when it was written. A refusal shows its message."""
        controller = self.actions.curation()
        if controller is None:
            self.status.setText('inspect-only')
            return False
        # any failed write (ENOSPC too) still sets the controls; only ValueError/PermissionError are shown
        try:
            controller.do(operation, **fields)
        except (ValueError, PermissionError) as exc:
            self.status.setText(str(exc))
            return False
        finally:
            self._controls_changed()
        return True

    def accept(self):
        self._set('accept')

    def reopen(self):
        """One reopen row for the current track."""
        self._set('reopen')

    def reject(self):
        self._set('reject')

    def _set(self, op):
        cid, members = self.current_cluster()
        if not cid:
            return
        if not self._do(op, cluster_id=cid, members=members):
            return
        self.status.setText(f'{op}ed {cid} ({len(members)} units)')
        self.refresh_list()

    def split_selected(self):
        cid, members = self.current_cluster()
        selected = self.selected_members()
        rest = [m for m in members if m not in selected]
        if not cid or not selected or not rest:
            self.status.setText('Select some, but not all, members to separate from the remainder.')
            return
        if not self._do('split', cluster_id=cid, members=selected):
            return
        self.refresh_list(preferred_id=cl.cluster_id(selected))
        self.status.setText('Split selected members from the remainder; singletons remain unmatched.')

    def drop_member(self):
        cid, members = self.current_cluster()
        r = self.members.currentRow()
        it = self.members.item(r, 0) if r >= 0 else None
        if not cid or it is None:
            return
        m = it.data(QtCore.Qt.UserRole)
        if not self._do('remove_member', cluster_id=cid, members=[m]):
            return
        self.refresh_list(preferred_id=cl.cluster_id([m]))
        self.status.setText(f'{m} is unmatched and remains available for merging.')

    def merge_selected(self):
        ids = self.selected_ids()
        tracks = self.actions.track_state().tracks
        union = [m for tid in ids if tid in tracks for m in tracks[tid]['members']]
        refusal = self.actions.merge_refusal(union)
        if refusal:
            self.status.setText(refusal)
            return
        if len(ids) < 2:
            self.status.setText('select two or more clusters to merge '
                                '(ctrl-click in the list)')
            return
        if not self._do('merge', parent_ids=ids, members=union):
            return
        self.refresh_list(preferred_id=cl.cluster_id(union))
        self.status.setText(f'merged {len(ids)} clusters')
