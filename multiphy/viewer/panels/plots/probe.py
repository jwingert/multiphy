"""The probe as it was recorded, with each selected unit filled in on its electrodes."""
import numpy as np

from multiphy.features import probe_geometry
from multiphy.viewer import display
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.qt import QtWidgets
from multiphy.viewer.registry import register_panel


@register_panel
class ProbePanel(PlotPanel):
    """The probe as it was recorded, with each unit filled in on its electrodes.

    Every electrode is one square of one fixed size, centred on the coordinate
    the recording record gives it. Those coordinates come from the site's own
    Open Ephys settings.xml, which is the only place the channel-to-electrode
    mapping is stored, and it is not a formality: Neuropixels records 384 of
    960 electrodes and the selection changes between days, so the same channel
    number sits up to 3840 um apart on different days of one well. Drawing an
    assumed map would put a unit on the wrong piece of silicon.

    An electrode a unit appears on is coloured. The square is split into one
    vertical strip per selected unit, and a unit always owns the same strip, so
    reading down the probe follows one colour in one column and an electrode
    shared by three units shows three strips side by side. Amplitude is
    deliberately absent here: it made a shared electrode unreadable, and it is
    already on the footprint and waveform views. This view answers which units
    touch which electrodes, and nothing else.
    """
    panel_id, panel_title = 'probe', 'Probe map'
    REL = 0.10                 # an electrode is part of the footprint above this

    def build_controls(self, row):
        self.whole_probe = QtWidgets.QCheckBox('whole probe')
        self.whole_probe.setToolTip('off: just around the units on screen')
        self.whole_probe.stateChanged.connect(lambda _: self.refresh())
        row.addWidget(self.whole_probe)

    def geometry_for(self, cid, n_chan):
        """The electrodes this unit's day recorded from."""
        g = self.state.data.geometry(self.state.sites[cid])
        if g is not None and len(g) == n_chan:
            return np.asarray(g, float)
        return probe_geometry(n_chan)

    def draw(self, r):
        ax = self.axes[0]
        data = self.state.data
        sel = [(c, u, col) for c, u, col in self.selected() if u.ptp is not None]
        if not sel:
            ax.text(.5, .5, 'no channel profile', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        n_chan = len(sel[0][1].ptp)
        if any(len(u.ptp) != n_chan for _, u, _ in sel):
            ax.text(.5, .5, 'selected units are on probes with different\n'
                            'channel counts', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        geo = {}
        for cid, _, _ in sel:
            site = self.state.sites[cid]
            if site not in geo:
                geo[site] = self.geometry_for(cid, n_chan).copy()

        # One square per electrode position, one fixed size, whatever day it
        # came from: an electrode is a physical thing, and two days recording
        # the same one must not draw it twice at two sizes.
        allg = np.vstack(list(geo.values()))
        sites = np.unique(allg, axis=0)
        ys = np.unique(sites[:, 1])
        pitch = float(np.median(np.diff(ys))) if len(ys) > 1 else 20.0
        side = pitch * 0.9
        ax.add_collection(_squares(sites[:, 0], sites[:, 1], side))

        # one vertical strip per selected unit, always in the same slot
        n = max(len(sel), 1)
        strip = side / n
        focus = []
        for k, (cid, u, col) in enumerate(sel):
            g = geo[self.state.sites[cid]]
            amp = np.asarray(u.ptp, float)
            pk = float(np.nanmax(amp)) if np.isfinite(amp).any() else 0.0
            if pk <= 0:
                continue
            on = np.where(amp >= self.REL * pk)[0]
            x0 = -side / 2 + k * strip
            ax.add_collection(_strips(g[on, 0] + x0, g[on, 1] - side / 2,
                                      strip, side, col))
            j = int(np.nanargmax(amp))
            ax.plot([], [], color=col, lw=4,
                    label=f'{self.label_for(cid)} · ch {j + 1} · {len(on)} sites')
            focus.append(g[j, 1])

        # 'box' not 'datalim': the limits below are the ones wanted, and
        # datalim would pad them out to fill a tall panel with empty probe
        ax.set_aspect('equal', adjustable='box')
        pad = max(4 * pitch, 60.0)
        if focus and not self.whole_probe.isChecked():
            ax.set_ylim(min(focus) - pad, max(focus) + pad)
        else:
            ax.set_ylim(sites[:, 1].min() - pad, sites[:, 1].max() + pad)
        ax.set_xlim(sites[:, 0].min() - side, sites[:, 0].max() + side)
        ax.set_xlabel('x (um)', fontsize=8)
        ax.set_ylabel('depth along the probe (um)', fontsize=8)
        srcs = sorted({data.geometry_source(self.state.sites[cid]) for cid, _, _ in sel})
        spans = {f"{g[:, 1].min():.0f}-{g[:, 1].max():.0f}" for g in geo.values()}
        ax.set_title(f'{n_chan} electrodes · {" · ".join(srcs)}'
                     + (f' · spans {" and ".join(sorted(spans))} um'
                        if len(spans) > 1 else ''), fontsize=7.5)
        ax.legend(fontsize=6, frameon=False, loc='upper right')


def _strips(x, y, w, h, color):
    """One filled strip per electrode, all the same size."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    return PatchCollection(
        [Rectangle((xi, yi), w, h) for xi, yi in zip(x, y)],
        facecolor=color, edgecolor='none', alpha=0.9, zorder=1)


def _squares(x, y, side):
    """One collection for every site, so a 384-site probe stays cheap to draw."""
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle
    p = display.colors()
    pc = PatchCollection(
        [Rectangle((xi - side / 2, yi - side / 2), side, side)
         for xi, yi in zip(x, y)],
        facecolor=p['raised'], edgecolor=p['border'], linewidths=0.4, zorder=0)
    return pc
