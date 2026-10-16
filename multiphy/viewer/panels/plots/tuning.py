"""Each selected unit's tone tuning curve on its site's frequency axis."""
import numpy as np

from multiphy.viewer.display import colors
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.registry import register_panel


@register_panel
class TuningPanel(PlotPanel):
    panel_id, panel_title = 'tuning', 'Tone tuning curve'

    def draw(self, r):
        ax = self.axes[0]
        data = self.state.data
        if 'tuning' not in data.available():
            ax.text(.5, .5, 'no tone tuning', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        drew = False
        for cid, u, c in self.selected():
            site = self.state.sites[cid]
            f = np.array(data.freq_axis(site) or [])
            if u.tuning is not None and len(f) == len(u.tuning) and len(f):
                ax.semilogx(np.maximum(f, 1.0), u.tuning, color=c, lw=1.3,
                            label=self.label_for(cid))
                drew = True
        if not drew:
            ax.text(.5, .5, 'no tuning curve', ha='center', va='center',
                    transform=ax.transAxes, fontsize=8)
            return
        ax.set_xlabel('frequency (Hz)', fontsize=8)
        ax.set_ylabel('evoked rate', fontsize=8)
        ax.axhline(0, color=colors()['muted'], lw=0.6)
        ax.legend(fontsize=6, frameon=False, ncol=2 if len(self.selected()) > 4 else 1)
