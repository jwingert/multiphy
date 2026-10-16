"""Every selected unit's waveform drawn on the electrodes it was recorded from."""
import numpy as np

from multiphy.features import probe_geometry
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.qt import QtWidgets
from multiphy.viewer.registry import register_panel


@register_panel
class WaveformMapPanel(PlotPanel):
    """Every unit's waveform drawn where it was recorded, as phy draws it.

    One trace per electrode, positioned at that electrode's own coordinate, so
    the shape of a unit's spatial footprint is visible rather than summarised
    into an amplitude. Two units that share a peak electrode are told apart
    here, by how their waveforms spread and how they differ in shape, which is
    what a curator is actually judging.

    The traces are drawn wider than the electrode spacing, as phy does: the
    layout places them, it does not scale them, or a 16 um column spacing would
    leave nothing legible.
    """
    panel_id, panel_title = 'wfmap', 'Waveforms on the probe'
    GAINS = ('0.5', '1', '2', '4', '8')

    def build_controls(self, row):
        row.addWidget(QtWidgets.QLabel('channels'))
        self.nch = QtWidgets.QSpinBox()
        self.nch.setRange(3, 41)
        self.nch.setValue(15)
        self.nch.setSingleStep(2)
        self.nch.setToolTip('how many electrodes either side of the peak to draw')
        self.nch.valueChanged.connect(lambda _: self.refresh())
        row.addWidget(self.nch)
        row.addWidget(QtWidgets.QLabel('gain'))
        self.gain = QtWidgets.QComboBox()
        self.gain.addItems(self.GAINS)
        self.gain.setCurrentText('2')
        self.gain.setToolTip('vertical scale of the traces, in electrode rows')
        self.gain.currentIndexChanged.connect(lambda _: self.refresh())
        row.addWidget(self.gain)
        self.norm = QtWidgets.QCheckBox('normalise')
        self.norm.setChecked(True)
        self.norm.setToolTip('scale each unit by its own peak, so shapes compare; '
                             'off compares absolute amplitude')
        self.norm.stateChanged.connect(lambda _: self.refresh())
        row.addWidget(self.norm)

    def peak_of(self, cellid):
        """The unit's peak channel, or None when it has none."""
        peak = self.state.data.unit(cellid).peak_chan
        if peak is None or not np.isfinite(peak):
            return None
        return int(peak)

    def geometry_for(self, cid, n_chan):
        g = self.state.data.geometry(self.state.sites[cid])
        if g is not None and len(g) == n_chan:
            return np.asarray(g, float)
        return probe_geometry(n_chan)

    def draw(self, r):
        ax = self.axes[0]
        sel = [(c, u, col) for c, u, col in self.selected()
               if u.patch is not None and u.ptp is not None]
        if not sel:
            ax.text(.5, .5, 'no template patch', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        half_draw = int(self.nch.value())
        gain = float(self.gain.currentText())
        norm = self.norm.isChecked()

        # one common scale unless the curator asks for absolute amplitude
        peaks = [float(np.nanmax(np.asarray(u.ptp, float))) for _, u, _ in sel]
        common = max([p for p in peaks if np.isfinite(p)] or [1.0])

        drawn, xs, ys = 0, [], []
        for (cid, u, col), pk_amp in zip(sel, peaks):
            n_chan = len(u.ptp)
            g = self.geometry_for(cid, n_chan)
            off = 0.0
            peak = self.peak_of(cid)
            patch = np.asarray(u.patch, float)
            if peak is None or patch.ndim != 2:
                continue
            half_patch = patch.shape[0] // 2
            nt = patch.shape[1]
            ys_ = np.unique(g[:, 1])
            pitch = float(np.median(np.diff(ys_))) if len(ys_) > 1 else 20.0
            xs_ = np.unique(g[:, 0])
            xstep = float(np.median(np.diff(xs_))) if len(xs_) > 1 else pitch
            width = 1.5 * xstep
            scale = gain * pitch / (pk_amp if norm else common)
            t = (np.arange(nt) / max(nt - 1, 1) - 0.5) * width
            lab = self.label_for(cid)
            for k in range(patch.shape[0]):
                ch = peak - half_patch + k
                if ch < 0 or ch >= n_chan or abs(ch - peak) > half_draw:
                    continue
                w = patch[k]
                if not np.isfinite(w).any() or np.allclose(w, 0):
                    continue
                ax.plot(g[ch, 0] + t, g[ch, 1] + off + w * scale, color=col,
                        lw=0.9, solid_capstyle='round',
                        label=lab if drawn == 0 else None)
                xs.append(g[ch, 0]); ys.append(g[ch, 1] + off)
                drawn += 1
            if drawn:
                ax.plot([], [], color=col, lw=2, label=lab)
        if not drawn:
            ax.text(.5, .5, 'no waveform in range', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        xs, ys = np.array(xs), np.array(ys)
        ax.set_xlim(xs.min() - width * 0.7, xs.max() + width * 0.7)
        ax.set_ylim(ys.min() - 2 * pitch, ys.max() + 2 * pitch)
        ax.set_xlabel('x (um), traces widened to be legible', fontsize=7.5)
        ax.set_ylabel('depth along the probe (um)', fontsize=8)
        ax.set_title(f'{len(sel)} units · {2 * half_draw + 1} electrodes about each peak'
                     + (' · each scaled to its own peak' if norm
                        else ' · one absolute scale'), fontsize=7.5)
        h, l = ax.get_legend_handles_labels()
        seen = dict(zip(l, h))
        ax.legend(seen.values(), seen.keys(), fontsize=6, frameon=False,
                  loc='upper right')
