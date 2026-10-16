"""The base of every plot panel: a canvas that tolerates transient dock geometry, a small toolbar,
wheel zoom, and the selected units read from the state's ``RunData``."""
import numpy as np

from multiphy import clusters as cl
from multiphy.viewer import display
import matplotlib

from multiphy.viewer import theme
from multiphy.viewer import unit_palettes
from multiphy.viewer.qt import (Figure, FigureCanvasQTAgg, NavigationToolbar2QT, QtCore,
                                            QtWidgets)


class SafeCanvas(FigureCanvasQTAgg):
    """Canvas that tolerates transient dock geometry and deleted windows. A redraw queued before the canvas was
    deleted is skipped."""

    def _draw_idle(self):
        """Run matplotlib's queued redraw; once the canvas's C++ object is gone, skip it.

        ``draw_idle`` queues ``QTimer.singleShot(0, self._draw_idle)``. When the canvas is deleted before that
        timer fires, the call raises ``RuntimeError: Internal C++ object (SafeCanvas) already deleted.`` inside the
        next ``processEvents``; that one error returns quietly and any other is raised again.
        """
        try:
            super()._draw_idle()
        except RuntimeError as exc:
            if 'already deleted' not in str(exc):
                raise

    def draw(self):
        # Qt can briefly allocate a zero-height canvas while arranging docks.
        # Matplotlib's equal-aspect probe axes cannot render a zero-height
        # figure. A later Qt resize event schedules the normal redraw.
        if (self.width() <= 0 or self.height() <= 0 or
                self.figure.bbox.width <= 0 or self.figure.bbox.height <= 0):
            return
        super().draw()

    def showEvent(self, event):
        try:
            super().showEvent(event)
        except RuntimeError as exc:
            if 'already deleted' not in str(exc):
                raise
            event.accept()


class PlotTools(NavigationToolbar2QT):
    """Reset, pan and box zoom only. Save is there because it costs nothing."""
    toolitems = [t for t in NavigationToolbar2QT.toolitems
                 if t[0] in ('Home', 'Pan', 'Zoom', 'Save')]

    def __init__(self, canvas, parent=None):
        super().__init__(canvas, parent, coordinates=False)
        self.setIconSize(QtCore.QSize(16, 16))
        self.setStyleSheet('QToolBar{border:0;padding:0;spacing:0;}')
        for a in self.actions():
            if a.text() == 'Home':
                a.setToolTip('reset the view (also: double-click a plot)')

    def restyle(self) -> None:
        """Rebuild every tool icon, so a dark palette gets light icons and a light one dark icons."""
        for text, tooltip, image, callback in self.toolitems:
            if callback in self._actions:
                self._actions[callback].setIcon(self._icon(image + '.png'))


class PlotPanel(QtWidgets.QWidget):
    panel_id = 'plot'
    panel_title = 'Plot'
    nrows, ncols = 1, 1
    always_draws = False      # True: draw whenever a run is loaded, even with no row or selection

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.setMinimumSize(1, 1)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(1)
        self.fig = Figure(figsize=(3, 2.2), layout='constrained')
        self.canvas = SafeCanvas(self.fig)
        self.header_row = QtWidgets.QHBoxLayout()
        self.header_row.setContentsMargins(0, 0, 0, 0)
        self.build_controls(self.header_row)      # panels add their own widgets
        self.header_row.addStretch(1)
        self.tools = PlotTools(self.canvas, self)
        self.header_row.addWidget(self.tools)
        lay.addLayout(self.header_row)
        lay.addWidget(self.canvas)
        self.axes = self.fig.subplots(self.nrows, self.ncols, squeeze=False).ravel()
        for a in self.axes:
            a.tick_params(labelsize=7)
        theme.style_figure(self.fig)
        self.canvas.mpl_connect('scroll_event', self._wheel)
        self.canvas.mpl_connect('button_press_event', self._click)
        state.changed.connect(self.refresh)

    def build_controls(self, row):
        """Hook for a panel that needs its own controls beside the toolbar."""

    # ------------------------------------------------------------ navigation
    def _wheel(self, ev):
        """Wheel zooms about the cursor, on the axis the cursor is over."""
        ax = ev.inaxes
        if ax is None or ev.xdata is None:
            return
        f = 1 / 1.25 if ev.button == 'up' else 1.25
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        # a log axis has to be scaled in log space or the view walks sideways
        if ax.get_xscale() == 'log' and ev.xdata > 0:
            lo, hi, c = np.log10(max(x0, 1e-12)), np.log10(max(x1, 1e-12)), np.log10(ev.xdata)
            ax.set_xlim(10 ** (c + (lo - c) * f), 10 ** (c + (hi - c) * f))
        else:
            ax.set_xlim(ev.xdata + (x0 - ev.xdata) * f, ev.xdata + (x1 - ev.xdata) * f)
        if ev.ydata is not None and ax.get_yscale() != 'log':
            ax.set_ylim(ev.ydata + (y0 - ev.ydata) * f, ev.ydata + (y1 - ev.ydata) * f)
        self.canvas.draw_idle()

    def _click(self, ev):
        if ev.dblclick:
            self.reset_view()

    def reset_view(self):
        for a in self.axes:
            a.relim()
            a.autoscale(True)
        self.tools.update()
        self.canvas.draw_idle()

    def selected(self):
        """[(cellid, UnitData, colour)] for every selected unit the run has, in selection order.

        With the Day ramp on Graphite and Dark, a unit's colour is its site's rank by recording date
        (``RunData.site_order``) along the ramp, so units of one site share a colour; otherwise colours follow
        selection order (``display.unit_colors``).
        """
        data = self.state.data
        if data is None:
            return []
        sel = [c for c in (self.state.selection or []) if c in data.unit_map]
        if unit_palettes.by_day() and display.uses_unit_palette():
            order = data.site_order()
            ramp = unit_palettes.day_ramp(len(order))
            sites = data.sites()
            return [(c, data.unit(c), ramp[order.index(str(sites[c]))]) for c in sel]
        cols = display.unit_colors(len(sel))
        return [(c, data.unit(c), cols[i]) for i, c in enumerate(sel)]

    def label_for(self, cellid):
        """The unit's day name; with the unit label once two selected units share its day."""
        sel = [c for c, _, _ in self.selected()]
        sites = self.state.sites
        site = sites[cellid]
        day_name = self.state.data.day_name(site)
        if sum(1 for c in sel if sites[c] == site) > 1:
            return f'{day_name} {cl.unit_label(cellid, site)}'
        return day_name

    def refresh(self):
        with matplotlib.rc_context(theme.plot_rc(display.colors())):
            r = self.state.row
            for a in self.axes:
                a.clear()
                a.tick_params(labelsize=7)
            theme.style_figure(self.fig)
            if r is not None or self.state.selection or (self.always_draws and self.state.data is not None):
                try:
                    self.draw(r)
                except Exception as exc:          # never let one panel kill the app
                    self.axes[0].text(.5, .5, f'{type(exc).__name__}: {exc}',
                                      ha='center', va='center', fontsize=7, wrap=True,
                                      transform=self.axes[0].transAxes)
            self.canvas.draw_idle()

    def restyle(self) -> None:
        """Recolour the toolbar icons and redraw in the active theme; zoom and pan reset."""
        self.tools.restyle()
        self.refresh()

    def draw(self, r):
        raise NotImplementedError
