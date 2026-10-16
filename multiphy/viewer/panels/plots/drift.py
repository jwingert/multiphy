# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the drift panel: each day of the drawn recording group as a probe column with its units, in the anchor day's depth frame read from the run's saved registration (drift-view Task 4) | date: 2026-10-08]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the module docstring names the dot click; tag wraps the whole docstring (browse-same-day Task 4) | date: 2026-10-12]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the click lands after the double-click interval, a double-click only resets, a same-run same-group redraw keeps the zoom; tag wraps the whole docstring (browse-same-day Task 6) | date: 2026-10-12]
"""Every day of one recording group side by side, drawn in the anchor day's depth frame.

Each day is one column: its electrodes and its units, shifted by the registration the run saved
(``viewer.drift_frame``). A day outside the anchor's linked group is drawn raw in the muted tone
and tagged ``unlinked``; a group with no usable offsets is drawn raw with the ``no offsets`` tag.
Hover a column for its shift, or a unit for its own. A left click on a unit's dot (no pan or zoom on)
emits ``unit_clicked`` with its cellid once the double-click interval passes, which adds it to the selection
or takes it out; a double-click only resets the view. A redraw of the same run at the same full-size limits
(a selection change within one recording group) keeps the zoom and pan.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the click imports: QtCore for the signal, Optional, MouseButton (browse-same-day Task 4) | date: 2026-10-12]
from typing import Optional

# [AGENT EDIT END]
import numpy as np
import pandas as pd

from multiphy.viewer import display
from multiphy.viewer.drift_frame import (drawn_group, drift_frame, electrode_depths, group_label,
                                                     group_sessions, group_units, unit_depth)
from multiphy.viewer.panels.plots import probe
from multiphy.viewer.panels.plots.base import PlotPanel
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: QtCore for the unit_clicked signal (browse-same-day Task 4) | date: 2026-10-12]
from matplotlib.backend_bases import MouseButton

from multiphy.viewer.qt import QtCore, QtGui, QtWidgets
# [AGENT EDIT END]
from multiphy.viewer.registry import register_panel

POINT_COLUMNS = ('cellid', 'session', 'x', 'y', 'shift', 'filled', 'muted')
HOVER_PIXELS = 6.0          # a unit's hover text shows within this many display pixels of its dot


def _float_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    """``frame[name]`` as floats; a missing column reads as ``nan``."""
    if name not in frame.columns:
        return np.full(len(frame), np.nan)
    return pd.to_numeric(frame[name], errors='coerce').to_numpy(float)


@register_panel
class DriftPanel(PlotPanel):
    """Each day's probe and units of the drawn group, mapped into the anchor day's depth frame."""
    panel_id, panel_title = 'drift', 'Drift across days'
    always_draws = True
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a left click on a unit dot emits its cellid (browse-same-day Task 4) | date: 2026-10-12]
    unit_clicked = QtCore.Signal(str)
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a single-shot precise click timer, the pending click and the last full-size draw; tag wraps the whole constructor (browse-same-day Task 6) | date: 2026-10-12]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the click timer is precise so it never fires inside a double-click; tag wraps the whole constructor (browse-same-day Task 6 fix round 1) | date: 2026-10-12]
    def __init__(self, state, parent=None):
        super().__init__(state, parent)
        self.points = pd.DataFrame({c: [] for c in POINT_COLUMNS})
        self.frame = None
        self._layout = (0.0, 0.0, 40.0)     # xmin, span, pitch of the last draw
        self._geometries = {}               # session -> its geometry (or None) in the last draw
        self.canvas.mpl_connect('motion_notify_event', self._hover)
        self._click_timer = QtCore.QTimer(self)
        self._click_timer.setSingleShot(True)
        self._click_timer.setTimerType(QtCore.Qt.PreciseTimer)   # a coarse timer can fire ~5% early, inside a double-click
        self._click_timer.timeout.connect(self._emit_pending_click)
        self._pending_click = None          # (RunData, cellid) of a press waiting out the double-click interval
        self._full_draw = None              # (RunData, xlim, ylim) of the last draw, at full size
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a redraw of the same run at the same full-size limits restores the zoom and pan; tag wraps the whole function (browse-same-day Task 6) | date: 2026-10-12]
    def refresh(self):
        """Redraw; when the user zoomed or panned and the new draw has the same run and the same full-size
        limits as the last one, the zoom and pan are put back."""
        ax = self.axes[0]
        view = (ax.get_xlim(), ax.get_ylim())
        last = self._full_draw
        self.points = pd.DataFrame({c: [] for c in POINT_COLUMNS})
        self.frame = None
        self._geometries = {}
        super().refresh()
        new = (self.state.data, ax.get_xlim(), ax.get_ylim()) if self.frame is not None else None
        self._full_draw = new
        if (last is not None and view != last[1:] and new is not None and new[0] is last[0]
                and new[1:] == last[1:]):
            ax.set_xlim(*view[0])
            ax.set_ylim(*view[1])
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: reset redraws at the drawn limits; the inherited relim() ignores the scatter collections, so it hid unselected units and days (drift-view post-run fix) | date: 2026-10-08]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: reset forgets the last full-size draw so it draws at full size; tag wraps the whole function (browse-same-day Task 6) | date: 2026-10-12]
    def reset_view(self):
        self._full_draw = None
        self.refresh()
        self.tools.update()
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a theme change resets zoom and pan (browse-same-day Task 6) | date: 2026-10-12]
    def restyle(self) -> None:
        """Forget the last full-size draw, then recolour and redraw at full size."""
        self._full_draw = None
        super().restyle()
    # [AGENT EDIT END]

    def draw(self, r):
        ax = self.axes[0]
        data = self.state.data
        p = display.colors()
        units = data.units()
        group = drawn_group(units, self.state.selection or [])
        if group is None:
            ax.text(.5, .5, 'no units', ha='center', va='center', transform=ax.transAxes, fontsize=8)
            return
        members = group_units(units, group)
        sessions = group_sessions(units, group)
        spec = display.backend_spec(data.result.backend_id) if data.result is not None else None
        frame = drift_frame(data.result.provenance if data.result is not None else {}, group, sessions,
                            saves_offsets=spec is not None and spec.provides.day_offsets)

        session_column = 'session_id' if 'session_id' in members.columns else 'siteid'
        depth = _float_column(members, 'depth')
        x_um = _float_column(members, 'x_um')
        geometries = {}
        for session in sessions:
            first = members.loc[members[session_column] == session].iloc[0]
            g = data.geometry(first['siteid'])
            geometries[session] = None if g is None else np.asarray(g, dtype=float)

        xs = [g[:, 0] for g in geometries.values() if g is not None and len(g)]
        xs.append(x_um[np.isfinite(x_um)])
        all_x = np.concatenate(xs)
        xmin = float(all_x.min()) if all_x.size else 0.0
        span = float(all_x.max()) - xmin if all_x.size else 0.0
        pitch = span + max(0.5 * span, 40.0)

        tracked = {c for track in (data.result.tracks if data.result is not None else ()) for c in track}
        pairs = data.pairs()
        chosen = pairs[pairs['selected'].fillna(False).astype(bool)]
        tracked |= set(chosen['cellid_a']) | set(chosen['cellid_b'])

        rows, faces, edges = [], [], []
        for k, column in enumerate(frame.columns):
            session = column.session
            muted = column.offsets_um == () and frame.no_offsets == ''
            colour = p['muted'] if muted else p['plot_line']
            g = geometries[session]
            if g is not None and len(g):
                ys = np.unique(g[:, 1])
                spacing = float(np.median(np.diff(ys))) if len(ys) > 1 else 20.0
                squares = probe._squares(g[:, 0] - xmin + k * pitch, electrode_depths(frame, column, g),
                                         0.9 * spacing)
                if muted:
                    squares.set_facecolor('none')
                    squares.set_edgecolor(p['muted'])
                ax.add_collection(squares)
            in_session = (members[session_column] == session).to_numpy()
            for i in np.flatnonzero(in_session):
                cellid = members['cellid'].iloc[i]
                if not np.isfinite(depth[i]):
                    continue
                y = unit_depth(frame, column, cellid, float(depth[i]), float(x_um[i]))
                if y is None:
                    continue
                x = float(x_um[i] - xmin + k * pitch) if np.isfinite(x_um[i]) else k * pitch + span / 2
                filled = cellid in tracked
                rows.append(dict(cellid=cellid, session=session, x=x, y=float(y), shift=float(y - depth[i]),
                                 filled=filled, muted=muted))
                faces.append(colour if filled else 'none')
                edges.append(colour)
        if rows:
            points = pd.DataFrame(rows, columns=list(POINT_COLUMNS))
            ax.scatter(points.x, points.y, s=18, marker='o', facecolors=faces, edgecolors=edges, zorder=2)
        else:
            points = pd.DataFrame({c: [] for c in POINT_COLUMNS})

        # the selected units of this group: a ring each, joined in column order
        order = {column.session: k for k, column in enumerate(frame.columns)}
        by_cell = points.set_index('cellid') if len(points) else None
        rings = []
        for cellid, _, colour in self.selected():
            if by_cell is None or cellid not in by_cell.index:
                continue
            row = by_cell.loc[cellid]
            ax.scatter([row.x], [row.y], s=70, marker='o', facecolors='none', edgecolors=colour,
                       linewidths=1.5, zorder=3)
            rings.append((order[row.session], row.x, row.y))
        if len(rings) >= 2:
            rings.sort(key=lambda ring: ring[0])
            ax.plot([ring[1] for ring in rings], [ring[2] for ring in rings], color=p['plot_line'], lw=1,
                    gid='selection', zorder=2)

        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the depth axis fits the electrodes even with no dot drawn: add_collection extends the data limits without rescaling (drift-view Task 4, fix round 2) | date: 2026-10-08]
        ax.autoscale_view(scalex=False)
        # [AGENT EDIT END]
        ticks = [k * pitch + span / 2 for k in range(len(frame.columns))]
        labels, muted_ticks = [], []
        for column in frame.columns:
            muted = column.offsets_um == () and frame.no_offsets == ''
            labels.append(f'{column.session}\nunlinked' if muted else column.session)
            muted_ticks.append(muted)
        ax.set_xticks(ticks, labels)
        for label, muted in zip(ax.get_xticklabels(), muted_ticks):
            if muted:
                label.set_color(p['muted'])
        gap = pitch - span
        ax.set_xlim(-gap / 2, (len(frame.columns) - 1) * pitch + span + gap / 2)
        ax.set_xlabel('')
        ax.set_ylabel(f'depth in {frame.anchor} frame (µm)' if frame.anchor is not None else 'depth (µm)',
                      fontsize=8)
        parts = [group_label(group)] if data.mode == 'same_day' else []
        if frame.no_offsets:
            parts.append('no offsets')
        ax.set_title(' · '.join(parts), fontsize=7.5)

        self.points = points
        self.frame = frame
        self._layout = (xmin, span, pitch)
        self._geometries = geometries

    # ------------------------------------------------------------------ hover
    def _column_text(self, column) -> str:
        """A column's shift: one value, a range, or why it has none."""
        frame, session = self.frame, column.session
        if frame.no_offsets:
            return f'{session} · no offsets · {frame.no_offsets}'
        if column.offsets_um == ():
            return f'{session} · unlinked · {column.reason}'
        g = self._geometries.get(session)
        if g is not None and len(g):
            shifts = electrode_depths(frame, column, g) - g[:, 1]
        else:
            shifts = self.points.loc[self.points.session == session, 'shift'].to_numpy(float)
        if not len(shifts):
            return f'{session} · {frame.method}'
        lo, hi = float(np.min(shifts)), float(np.max(shifts))
        if f'{lo:+.1f}' == f'{hi:+.1f}':
            return f'{session} · {lo:+.1f} µm · {frame.method}'
        return f'{session} · {lo:+.1f} to {hi:+.1f} µm · {frame.method}'

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the nearest-dot search moves into _nearest_dot, shared by hover_text and unit_at; tag wraps the three functions (browse-same-day Task 4) | date: 2026-10-12]
    def _nearest_dot(self, x: float, y: float) -> Optional[int]:
        """The row of ``self.points`` whose dot is nearest data point ``(x, y)`` in display pixels, when within
        ``HOVER_PIXELS``; ``None`` with no frame, no points or no dot that close."""
        if self.frame is None or not len(self.points):
            return None
        ax = self.axes[0]
        ax.get_xlim(), ax.get_ylim()          # settle a pending autoscale before reading transData
        dots = ax.transData.transform(self.points[['x', 'y']].to_numpy(float))
        here = ax.transData.transform((x, y))
        distance = np.hypot(dots[:, 0] - here[0], dots[:, 1] - here[1])
        i = int(np.argmin(distance))
        return i if distance[i] <= HOVER_PIXELS else None

    def hover_text(self, x: float, y: float) -> str:
        """The text under data point ``(x, y)``: the nearest unit within 6 pixels, else its column; ``''`` off the columns."""
        if self.frame is None:
            return ''
        _, span, pitch = self._layout
        k = round((x - span / 2) / pitch)
        if k not in range(len(self.frame.columns)):
            return ''
        i = self._nearest_dot(x, y)
        if i is not None:
            dot = self.points.iloc[i]
            (column,) = [c for c in self.frame.columns if c.session == dot.session]
            if column.offsets_um != ():
                return f'{dot.cellid} · {dot.session} · {dot["shift"]:+.1f} µm · {self.frame.method}'
            return f'{dot.cellid} · ' + self._column_text(column)
        return self._column_text(self.frame.columns[k])

    def unit_at(self, x: float, y: float) -> Optional[str]:
        """The cellid of the unit whose dot is within ``HOVER_PIXELS`` of data point ``(x, y)``, else ``None``."""
        i = self._nearest_dot(x, y)
        return None if i is None else str(self.points['cellid'].iloc[i])
    # [AGENT EDIT END]

    def _hover(self, ev):
        text = ''
        if ev.inaxes is self.axes[0] and ev.xdata is not None and ev.ydata is not None:
            text = self.hover_text(ev.xdata, ev.ydata)
        if not text:
            QtWidgets.QToolTip.hideText()
        else:
            QtWidgets.QToolTip.showText(QtGui.QCursor.pos(), text, self.canvas)

    # ------------------------------------------------------------------ click
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a left click on a unit dot, with no pan or zoom on, emits unit_clicked; a double-click resets the view (browse-same-day Task 4) | date: 2026-10-12]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the click waits out the double-click interval; a double-click only resets; tag wraps the whole functions (browse-same-day Task 6) | date: 2026-10-12]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the _click docstring names the kept zoom; tag wraps the whole function (browse-same-day Task 6 fix round 1) | date: 2026-10-12]
    def _click(self, ev):
        """A double-click drops any waiting click and only resets the view. A left click on a unit's dot, with
        no pan or zoom on, waits out the double-click interval, then emits ``unit_clicked`` with its cellid
        (``_emit_pending_click``); a later press restarts the wait. Anything else does nothing. A selection
        change within one recording group redraws the same run at the same full-size limits, which keeps the
        zoom and pan (``refresh``)."""
        if ev.dblclick:
            self._click_timer.stop()
            self._pending_click = None
            super()._click(ev)
            return
        if (ev.button != MouseButton.LEFT or self.tools.mode != '' or ev.inaxes is not self.axes[0]
                or ev.xdata is None or ev.ydata is None):
            return
        cellid = self.unit_at(ev.xdata, ev.ydata)
        if cellid is not None:
            self._pending_click = (self.state.data, cellid)
            self._click_timer.start(QtWidgets.QApplication.doubleClickInterval())
    # [AGENT EDIT END]

    def _emit_pending_click(self):
        """Emit the waiting click's cellid when its run is still drawn and still lists the unit."""
        pending, self._pending_click = self._pending_click, None
        if pending is None:
            return
        data, cellid = pending
        if data is self.state.data and cellid in set(self.points['cellid'].astype(str)):
            self.unit_clicked.emit(cellid)
    # [AGENT EDIT END]
    # [AGENT EDIT END]
# [AGENT EDIT END]
