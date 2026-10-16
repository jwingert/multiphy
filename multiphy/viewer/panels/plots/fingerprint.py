"""Each selected unit's peristimulus histogram over the exact stimulus conditions every selected unit shares."""
import numpy as np

from multiphy.viewer import display
from multiphy.viewer.panels.plots.base import PlotPanel
from multiphy.viewer.qt import QtWidgets
from multiphy.viewer.registry import register_panel


@register_panel
class FingerprintPanel(PlotPanel):
    """Peristimulus histogram over the conditions every selected unit shares.

    One trace per unit: the trial-averaged firing rate through each shared
    condition in turn, laid end to end, with a dotted rule between conditions.

    The condition box narrows that to one condition at full width, which is the only
    way to see whether two units agree on the detail of a response or only on
    which conditions they like.
    """
    panel_id, panel_title = 'fingerprint', 'Fingerprint, shared conditions'
    ALL = '— all conditions —'

    def build_controls(self, row):
        row.addWidget(QtWidgets.QLabel('condition'))
        self.stim_box = QtWidgets.QComboBox()
        self.stim_box.setMinimumWidth(190)
        self.stim_box.setMaxVisibleItems(30)
        self.stim_box.setToolTip('all conditions end to end, or one at full width')
        self.stim_box.addItem(self.ALL)
        self.stim_box.currentIndexChanged.connect(self._on_stim)
        row.addWidget(self.stim_box)
        self._want_key = None            # the condition the curator chose, by identity

    def _on_stim(self, _i):
        if self._filling:
            return
        self._want_key = self.stim_box.currentData()
        self.refresh()

    _filling = False

    def shared(self):
        """[(condition, condition, {})] for every exact condition all selected units share.

        Narrowed to the row's declared conditions when a row is shown; a condition is kept only with
        equal non-zero lengths, equal positive bin widths and enough repeats in every unit.
        """
        selected = self.selected()
        if not selected:
            return []
        common = set.intersection(*(set(u.fingerprints) for _, u, _ in selected))
        row = self.state.row
        declared = display.declared_conditions(row.get('shared_conditions')) if row is not None else None
        if declared is not None:
            common &= set(declared)
        repeat_min = self.state.data.min_repeats()
        out = []
        for condition in sorted(common):
            lengths = {len(u.fingerprints[condition]) for _, u, _ in selected}
            steps = [u.fingerprint_bin_seconds for _, u, _ in selected]
            if (len(lengths) != 1 or next(iter(lengths)) == 0
                    or any(s is None or not np.isfinite(s) or s <= 0 for s in steps)
                    or not np.allclose(steps, steps[0], rtol=1e-9, atol=0)
                    or any(u.trial_counts.get(condition, 0) < repeat_min for _, u, _ in selected)):
                continue
            out.append((condition, condition, {}))
        return out

    @staticmethod
    def _short(name):
        n = str(name)
        if n.startswith('STIM_'):
            n = n[5:]
        return n if len(n) <= 34 else n[:31] + '…'

    def _fill_box(self, shared):
        keys = [k for k, _, _ in shared]
        have = [self.stim_box.itemData(i) for i in range(self.stim_box.count())]
        if have[1:] == keys:
            return
        self._filling = True
        self.stim_box.clear()
        self.stim_box.addItem(self.ALL, None)
        for k, name, _ in shared:
            self.stim_box.addItem(self._short(name), k)
        i = keys.index(self._want_key) + 1 if self._want_key in keys else 0
        self.stim_box.setCurrentIndex(i)
        self._filling = False
        if i == 0:
            self._want_key = None

    def draw(self, row):
        ax = self.axes[0]
        selected, shared = self.selected(), self.shared()
        self._fill_box(shared)
        chosen = [s[0] for s in shared if self._want_key is None or s[0] == self._want_key]
        if not chosen:
            ax.text(.5, .5, 'No shared stimulus conditions with aligned bins and sufficient repeats',
                    ha='center', va='center', wrap=True, transform=ax.transAxes, fontsize=8)
            return
        for cid, unit, color in selected:
            dt = unit.fingerprint_bin_seconds
            trace = np.concatenate([unit.fingerprints[s] for s in chosen])
            ax.plot(np.arange(len(trace)) * dt, trace, color=color, lw=1.3, label=self.label_for(cid))
        first = selected[0][1]
        boundary = 0
        for condition in chosen[:-1]:
            boundary += len(first.fingerprints[condition]) * first.fingerprint_bin_seconds
            ax.axvline(boundary, color=display.colors()['muted'], lw=.7, ls=':')
        ax.set_xlabel('Time within stimulus (s)' if len(chosen) == 1 else 'Stimulus time concatenated (s)', fontsize=8)
        ax.set_ylabel('Spikes/s', fontsize=8)
        ax.set_title(f'{len(chosen)} exact conditions · {len(selected)} {self.state.data.words.observations}',
                     fontsize=8)
        ax.legend(fontsize=6, frameon=False)
