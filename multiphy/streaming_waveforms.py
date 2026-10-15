# [AGENT EDIT START | agent: codex | user: wingertj | reason: exact-selection streaming waveform summaries without temporary snippet files | date: 2026-09-18]
"""Bounded chronological Welford summaries for explicitly planned selections."""
import numpy as np

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: waveform_sample_selection lives in extract/unit_features.py (multiphy project folder Task 11) | date: 2026-09-25]
from .extract.unit_features import waveform_sample_selection
# [AGENT EDIT END]


class WaveformSummaryPlan:
    """Retain exact selections; equal selections may safely share one accumulator."""
    def __init__(self, selections, *, before, after):
        if any(not np.isfinite(x) or int(x) != x or x < 1 for x in (before, after)):
            raise ValueError('Summary crop dimensions must be positive integers.')
        self.before, self.after = int(before), int(after)
        self.selections, self.lookup = [], {}
        for selected in selections:
            raw = np.asarray(selected)
            if (raw.ndim != 1 or len(raw) < 2 or not np.isfinite(raw).all()
                    or not np.equal(raw, np.floor(raw)).all() or (raw < before).any()
                    or (np.diff(raw) <= 0).any()):
                raise ValueError('Each summary selection requires increasing distinct valid sample indices.')
            values = np.array(raw, dtype='<i8', copy=True)
            key = values.tobytes()  # Exact bytes, not a potentially colliding digest.
            if key not in self.lookup:
                self.lookup[key] = len(self.selections)
                values.flags.writeable = False
                self.selections.append(values)
        if not self.selections:
            raise ValueError('No valid waveform summary selections were planned.')
        samples = np.concatenate(self.selections)
        groups = np.repeat(np.arange(len(self.selections), dtype=np.int32),
                           [len(x) for x in self.selections])
        order = np.argsort(samples, kind='stable')
        self.event_samples, self.event_groups = samples[order], groups[order]
        self.samples = np.unique(samples)
        self.offsets = np.searchsorted(self.event_samples, self.samples)
        for array in (self.event_samples, self.event_groups, self.samples, self.offsets):
            array.flags.writeable = False


def plan_waveform_summaries(spikes, windows, *, before=41, after=41, max_spikes=1000,
                            seed=0, min_spikes=20, **unused):
    """Use the same full/half bounds, seeds and eligibility as temporal features."""
    selections = []
    for values in spikes.values():
        values = np.asarray(values)
        for split_index, (start, stop) in enumerate(windows):
            part = values[(values >= start) & (values < stop)]
            if len(part) < min_spikes:
                continue
            mid = (start + stop) // 2
            for offset, bounds in enumerate(((start, stop), (start, mid), (mid, stop))):
                try:
                    selected, _ = waveform_sample_selection(part, bounds, before, after,
                        max_spikes, seed + split_index * 3 + offset)
                except ValueError as exc:
                    if 'two complete waveform snippets' not in str(exc):
                        raise
                    continue
                selections.append(selected)
    return WaveformSummaryPlan(selections, before=before, after=after)


class StreamingWaveformSummaries:
    """Consume each union sample once; publish only after stream integrity checks."""
    def __init__(self, plan, n_channels):
        if not isinstance(plan, WaveformSummaryPlan) or int(n_channels) != n_channels or n_channels < 1:
            raise ValueError('Provide a waveform summary plan and positive channel count.')
        self.plan, self.n_channels = plan, int(n_channels)
        self.shape = (plan.before + plan.after, self.n_channels)
        size = len(plan.selections)
        self.means, self.m2, self.completed = [None] * size, [None] * size, [None] * size
        self.counts = np.zeros(size, dtype=np.int64)
        self.next_sample = 0
        self.verified = self.closed = False

    def add(self, center, snippet):
        if self.closed or self.verified or self.next_sample >= len(self.plan.samples):
            raise ValueError('Summary stream cannot accept more samples.')
        if center != self.plan.samples[self.next_sample]:
            raise ValueError('Summary samples must follow the exact increasing union plan.')
        snippet = np.asarray(snippet)
        if snippet.dtype != np.dtype('float32') or snippet.shape != self.shape or not np.isfinite(snippet).all():
            raise ValueError('Summary input must be a finite float32-quantized waveform snippet.')
        voltage = np.asarray(snippet, dtype=float)
        start = self.plan.offsets[self.next_sample]
        stop = (self.plan.offsets[self.next_sample + 1] if self.next_sample + 1 < len(self.plan.samples)
                else len(self.plan.event_groups))
        for group in self.plan.event_groups[start:stop]:
            if self.means[group] is None:
                self.means[group] = np.zeros(self.shape, dtype=float)
                self.m2[group] = np.zeros(self.shape, dtype=float)
            self.counts[group] += 1
            count = self.counts[group]
            mean, m2 = self.means[group], self.m2[group]
            delta = voltage - mean
            mean += delta / count
            m2 += delta * (voltage - mean)
            if count == len(self.plan.selections[group]):
                self.completed[group] = (mean.astype(np.float32),
                    np.sqrt(m2 / (count - 1) / count).astype(np.float32))
                self.means[group] = self.m2[group] = None
        self.next_sample += 1

    def verify_complete(self):
        if self.closed or self.next_sample != len(self.plan.samples) or any(x is None for x in self.completed):
            raise ValueError('Cannot publish incomplete waveform summaries.')
        self.verified = True

    def waveform_summary(self, selected, *, before, after, n_channels):
        if self.closed or not self.verified:
            raise RuntimeError('Waveform summaries are unavailable until stream integrity is verified.')
        if (before, after, n_channels) != (self.plan.before, self.plan.after, self.n_channels):
            raise ValueError('Waveform summary dimensions differ from the verified plan.')
        selected = np.asarray(selected)
        if (selected.ndim != 1 or not np.isfinite(selected).all()
                or not np.equal(selected, np.floor(selected)).all()):
            raise ValueError('Waveform summary selection differs from the verified plan.')
        index = self.plan.lookup.get(selected.astype('<i8').tobytes())
        if index is None:
            raise ValueError('Waveform summary selection differs from the verified plan.')
        mean, sem = self.completed[index]
        return dict(waveform=mean.copy(), waveform_standard_error=sem.copy())

    def close(self):
        self.closed, self.verified = True, False
        self.means.clear()
        self.m2.clear()
        self.completed.clear()
# [AGENT EDIT END]
