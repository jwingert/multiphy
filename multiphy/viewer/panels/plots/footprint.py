"""Each selected unit's peak-to-peak amplitude along the probe, in measured coordinates."""
import numpy as np

from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.registry import register_panel


@register_panel
class FootprintPanel(PlotPanel):
    panel_id, panel_title = 'footprint', 'Footprint along the probe'

    def draw(self, r):
        ax = self.axes[0]
        data = self.state.data
        # the same coordinates the probe view and the extraction use, so this
        # axis and the cached depth cannot drift apart
        sel = [(cid, u, c, data.geometry(self.state.sites[cid]))
               for cid, u, c in self.selected() if u.ptp is not None]
        if any(g is None or len(g) != len(u.ptp) for _, u, _, g in sel):
            ax.text(.5, .5, 'no measured geometry', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        centres = []
        for cid, u, c, measured in sel:
            y = np.asarray(measured, float)[:, 1]
            ax.plot(u.ptp, y, color=c, lw=1.2, label=self.label_for(cid))
            centres.append(float(np.asarray(y)[int(np.argmax(u.ptp))]))
        if centres:
            ax.set_ylim(min(centres) - 250, max(centres) + 250)
        ax.set_xlabel('amplitude', fontsize=8)
        ax.set_ylabel('Depth along the probe (µm)', fontsize=8)
        ax.legend(fontsize=6, frameon=False, ncol=2 if len(centres) > 4 else 1)
        ax.set_title('Measured coordinates', fontsize=8)
