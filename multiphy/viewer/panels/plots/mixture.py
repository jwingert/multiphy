"""How each probability is formed, for the day pair on screen: the fitted mixture of every stream."""
import numpy as np

from multiphy.backends.local import evidence as ev
import matplotlib

from multiphy.viewer import theme
from multiphy.viewer.display import STREAM_WORDS, colors, shows_scores
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.registry import register_panel


@register_panel
class MixturePanel(PlotPanel):
    """How each probability is formed, for the day pair on screen.

    One column per stream the day pair's run saved a ``weight_<stream>`` for, in the order event response,
    physiology, waveform, then the combined column; a run without events has no event-response column. The
    observed spread of scores is modelled as a blend
    of two fixed references. Which scores make up that spread depends on where
    the day pair came from: a project run's group runs fit their mixture on
    every gated candidate of the session pair (``mixture_day_pairs``), while a
    build stream's day pair holds its best-candidate scores. The two references
    are what a score looks like when it IS the same neuron, from a unit against
    its own other half within a session, and what it looks like when it is not,
    from simultaneously recorded neighbours inside the gate. Only the mixing
    fraction is fitted.

    Seeing the streams side by side is the point: they are largely
    independent, so a pair can be convincing on one and not the other, and the
    global column shows what the fitted weighting makes of the combination.

    A day pair without a fitted mixture, or a selection without a candidate row,
    shows the tag ``mixture: not fitted``.
    """
    panel_id, panel_title = 'mixture', 'Matching model provenance'
    ORDER = [('stim', 'event-response fingerprint', 's_stim', 'p_stim'),
             ('phys', 'physiology', 's_phys', 'p_phys'),
             ('wave', 'waveform', 's_wave', 'p_wave'),
             ('global', 'combined', 's_global', 'p_global')]

    def columns(self) -> list:
        """The ``ORDER`` entries the day pair on screen saved a ``weight_<stream>`` for, then the combined one."""
        dp = self.state.day_pair or {}
        return [entry for entry in self.ORDER if entry[0] == 'global' or f'weight_{entry[0]}' in dp]

    def draw(self, r):
        if r is not None and (self.state.day_pair or {}).get('streams'):
            self._draw_fitted(r)
            return
        for ax in self.axes:
            ax.set_axis_off()
        self.axes[0].text(.5, .5, 'mixture: not fitted', ha='center', va='center', fontsize=8,
                          transform=self.axes[0].transAxes)

    def _draw_fitted(self, r):
        """The fitted-mixture axes of the day pair on screen, one per ``columns()`` entry."""
        if not shows_scores(r):
            self.axes[0].text(.5, .5, 'no matching scores', ha='center', va='center',
                              fontsize=8, transform=self.axes[0].transAxes)
            return
        dp = self.state.day_pair or {}
        streams = dp.get('streams') or {}
        columns = self.columns()
        wt = 'weights  ' + '   '.join(f"{STREAM_WORDS[key]} {dp[f'weight_{key}']:.2f}"
                                      for key, *_ in columns if key != 'global')
        for ax, (key, title, scol, pcol) in zip(self.axes, columns):
            st_ = streams.get(key) or {}
            same = np.asarray(st_.get('same') or [], float)
            diff = np.asarray(st_.get('diff') or [], float)
            obs = np.asarray(st_.get('observed') or [], float)
            pi = st_.get('pi')
            if len(same) < 5 or len(diff) < 5 or pi is None or not np.isfinite(pi):
                ax.text(.5, .5, 'not fitted', ha='center', va='center',
                        fontsize=8, transform=ax.transAxes)
                ax.set_title(title, fontsize=8)
                continue
            lo = float(min(same.min(), diff.min(), obs.min() if len(obs) else same.min()))
            hi = float(max(same.max(), diff.max(), obs.max() if len(obs) else same.max()))
            pad = 0.1 * (hi - lo + 1e-9)
            grid = np.linspace(lo - pad, hi + pad, 300)
            # draw the Mixture the backend fits, its kernel bandwidth included
            model = ev.Mixture(same, diff)
            model.pi = float(pi)
            a = pi * model._same(grid)
            b = (1 - pi) * model._diff(grid)
            pal = colors()
            if len(obs):
                ax.hist(obs, bins=np.linspace(lo - pad, hi + pad, 22), density=True,
                        color=pal['plot_fill'], edgecolor=pal['panel'])
            ax.fill_between(grid, 0, a, color=pal['accept'], alpha=.35)
            ax.fill_between(grid, 0, b, color=pal['reject'], alpha=.30)
            ax.plot(grid, a + b, color=pal['plot_line'], lw=1.2)
            val = r.get(scol, np.nan)
            if np.isfinite(val):
                ax.axvline(val, color=pal['plot_line'], lw=1.5, ls='--')
            ax.set_xlabel('log-likelihood score', fontsize=7)
            ax.tick_params(labelsize=6)
            p = r.get(pcol, np.nan)
            ax.set_title(f'{title}\n{100*pi:.0f}% same neuron'
                         + (f',  this pair {100*p:.0f}%' if np.isfinite(p) else ''),
                         fontsize=7.5)
            ax2 = ax.twinx()
            post = model.posterior(grid)      # the model's own posterior
            post = np.where((grid >= lo) & (grid <= hi), post, np.nan)
            # Past the last reference value the ratio of two kernel tails is
            # pure extrapolation, and it bends the curve back down: at Prince
            # well 16 the physiology probability climbs to 96% at the largest
            # reference score and then falls to 48% beyond it, where no
            # reference of either class exists. Draw the probability only over
            # the span the references cover. No candidate pair falls outside
            # it -- 0 of 83 there, in all three streams -- so nothing that was
            # actually scored is hidden.
            ax2.plot(grid, 100 * post, color=pal['plot_second'], lw=1.3)
            ax2.axhline(80, color=pal['plot_second'], lw=0.7, ls=':')
            ax2.set_ylim(0, 105)
            ax2.tick_params(labelsize=6, colors=pal['plot_second'], labelfontfamily=theme.fonts()['mpl_mono'])
            if key == 'global':
                ax2.set_ylabel('P(same), %', fontsize=7, color=pal['plot_second'])
        self.axes[0].set_ylabel('density', fontsize=7)
        self.fig.suptitle(f'{dp.get("siteid_a")} vs {dp.get("siteid_b")},'
                          f' {dp.get("days_apart")}d apart    {wt}', fontsize=7.5)

    def refresh(self):
        """Rebuild the figure (the twin axes have to be dropped explicitly or they accumulate), then draw
        when a row is shown or the selection is non-empty."""
        with matplotlib.rc_context(theme.plot_rc(colors())):
            self.fig.clear()
            self.axes = self.fig.subplots(self.nrows, len(self.columns()), squeeze=False).ravel()
            theme.style_figure(self.fig)
            for a in self.axes:
                a.tick_params(labelsize=6)
            r = self.state.row
            if r is not None or self.state.selection:
                try:
                    self.draw(r)
                except Exception as exc:          # never let one panel kill the app
                    self.axes[0].text(.5, .5, f'{type(exc).__name__}: {exc}', ha='center',
                                      va='center', fontsize=7, wrap=True,
                                      transform=self.axes[0].transAxes)
            self.canvas.draw_idle()
