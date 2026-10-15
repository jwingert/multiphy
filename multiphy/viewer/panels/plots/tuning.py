# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the tone tuning panel, from gui.py:964-995, reading UnitData; a run without the tuning part says so (viewer-split Task 4) | date: 2026-10-06]
"""Each selected unit's tone tuning curve on its site's frequency axis."""
import numpy as np

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: colours come from the active theme's tokens (launcher-gui Task 8) | date: 2026-10-07]
from nems_lbhb.unit_match.viewer.display import colors
# [AGENT EDIT END]
from nems_lbhb.unit_match.viewer.panels.plots.base import PlotPanel
from nems_lbhb.unit_match.viewer.registry import register_panel


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
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the zero rule takes the theme's muted token (launcher-gui Task 8) | date: 2026-10-07]
        ax.axhline(0, color=colors()['muted'], lw=0.6)
        # [AGENT EDIT END]
        ax.legend(fontsize=6, frameon=False, ncol=2 if len(self.selected()) > 4 else 1)
# [AGENT EDIT END]
