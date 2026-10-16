# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and autocorrelogram panel, from gui.py:797-825, reading UnitData (viewer-split Task 4) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram are two panels, one plot each (browse-same-day Task 5) | date: 2026-10-12]
"""Each selected unit's interval histogram (``IntervalHistogramPanel``) and autocorrelogram
(``AutocorrelogramPanel``), one plot per panel."""
import numpy as np

from multiphy.extract.unit_features import ISI_BINS
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: colours come from the active theme's tokens (launcher-gui Task 8) | date: 2026-10-07]
from multiphy.viewer.display import colors
# [AGENT EDIT END]
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.registry import register_panel


@register_panel
class IntervalHistogramPanel(PlotPanel):
    panel_id, panel_title = 'isi', 'Interval histogram'
    ncols = 1

    def draw(self, r):
        centres = np.sqrt(ISI_BINS[:-1] * ISI_BINS[1:])
        (a0,) = self.axes
        for cid, u, c in self.selected():
            if u.isi is not None:
                a0.semilogx(centres, u.isi, color=c, lw=1.3)
        a0.set_xlabel('interval (s)', fontsize=8); a0.set_ylabel('fraction', fontsize=8)
        a0.set_title(f"ISI  {r.get('isi_cosine', np.nan):.2f}" if r is not None else 'ISI', fontsize=8)


@register_panel
class AutocorrelogramPanel(PlotPanel):
    panel_id, panel_title = 'acg', 'Autocorrelogram'
    ncols = 1

    def draw(self, r):
        (a1,) = self.axes
        for cid, u, c in self.selected():
            if u.acg is not None:
                # An autocorrelogram is symmetric by construction, so only the
                # positive lags are stored; mirroring here is exact, and the
                # centred form is the convention (phy draws it this way) because
                # it makes the refractory notch read at a glance.
                y = np.asarray(u.acg, float)
                lag = np.arange(len(y)) + 0.5
                a1.plot(np.concatenate([-lag[::-1], lag]),
                        np.concatenate([y[::-1], y]), color=c, lw=1.1)
        a1.set_xlabel('lag (ms)', fontsize=8)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the zero-lag rule takes the theme's muted token (launcher-gui Task 8) | date: 2026-10-07]
        a1.axvline(0, color=colors()['muted'], lw=0.6)
        # [AGENT EDIT END]
        a1.set_title(f"autocorr  {r.get('acg_cosine', np.nan):.2f}" if r is not None else 'autocorr', fontsize=8)
# [AGENT EDIT END]
# [AGENT EDIT END]
