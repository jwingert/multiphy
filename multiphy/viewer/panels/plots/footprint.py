# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the footprint panel, from gui.py:500-530 with the days viewer's measured-coordinates labels (multiday_viewer.py:206-211) in both modes, reading UnitData (viewer-split Task 4) | date: 2026-10-06]
"""Each selected unit's peak-to-peak amplitude along the probe, in measured coordinates."""
import numpy as np

from nems_lbhb.unit_match.features import probe_geometry
from nems_lbhb.unit_match.viewer.panels.plots.base import PlotPanel
from nems_lbhb.unit_match.viewer.registry import register_panel


@register_panel
class FootprintPanel(PlotPanel):
    panel_id, panel_title = 'footprint', 'Footprint along the probe'

    def draw(self, r):
        ax = self.axes[0]
        data = self.state.data
        centres = []
        for cid, u, c in self.selected():
            if u.ptp is None:
                continue
            n = len(u.ptp)
            # the same coordinates the probe view and the extraction use, so
            # this axis and the cached depth cannot drift apart
            measured = data.geometry(self.state.sites[cid])
            geometry = np.asarray(measured, float) if measured is not None and len(measured) == n else probe_geometry(n)
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the footprint draws measured depths, with no day offset (drift-view Task 3) | date: 2026-10-08]
            y = geometry[:, 1]
            # [AGENT EDIT END]
            ax.plot(u.ptp, y, color=c, lw=1.2, label=self.label_for(cid))
            centres.append(float(np.asarray(y)[int(np.argmax(u.ptp))]))
        if centres:
            ax.set_ylim(min(centres) - 250, max(centres) + 250)
        ax.set_xlabel('amplitude', fontsize=8)
        ax.set_ylabel('Depth along the probe (µm)', fontsize=8)
        ax.legend(fontsize=6, frameon=False, ncol=2 if len(centres) > 4 else 1)
        ax.set_title('Measured coordinates', fontsize=8)
# [AGENT EDIT END]
