# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the peak-channel waveform panel, from gui.py:485-498, reading UnitData (viewer-split Task 4) | date: 2026-10-06]
"""The peak-channel waveform of each selected unit."""
import numpy as np

from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.registry import register_panel


@register_panel
class WaveformPanel(PlotPanel):
    panel_id, panel_title = 'waveform', 'Peak-channel waveform'

    def draw(self, r):
        ax = self.axes[0]
        for cid, u, c in self.selected():
            if u.waveform is not None:
                w = np.asarray(u.waveform, float)
                ax.plot(w / (np.abs(w).max() + 1e-9), color=c, lw=1.4,
                        label=self.label_for(cid))
        ax.set_xlabel('sample', fontsize=8)
        ax.set_ylabel('normalised amplitude', fontsize=8)
        ax.legend(fontsize=6, frameon=False, ncol=2 if len(self.selected()) > 4 else 1)
# [AGENT EDIT END]
