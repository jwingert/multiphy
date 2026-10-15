# [AGENT EDIT START | agent: codex | user: wingertj | reason: shared filtered referenced unwhitened waveform standard | date: 2026-09-17]
"""Common voltage preprocessing, applied before cropping spike snippets."""
import numpy as np
from scipy.signal import butter, sosfiltfilt


class SharedWaveformPreprocessor:
    """Preserve physical amplitudes and channel geometry; never whiten."""
    def __init__(self, sample_rate, n_channels, highpass_hz=150., filter_order=3,
                 padding_seconds=.05, reference_channels=None):
        self.sample_rate = float(sample_rate)
        self.n_channels = int(n_channels)
        self.highpass_hz = float(highpass_hz)
        self.filter_order = int(filter_order)
        self.padding_seconds = float(padding_seconds)
        if (not np.isfinite(self.sample_rate) or self.sample_rate <= 0
                or n_channels != self.n_channels or self.n_channels < 2
                or not 0 < self.highpass_hz < self.sample_rate / 2
                or filter_order != self.filter_order or not 1 <= self.filter_order <= 8
                or not np.isfinite(self.padding_seconds)
                or self.padding_seconds < 5 / self.highpass_hz):
            raise ValueError('Invalid shared waveform filter or insufficient padding (minimum five cutoff periods).')
        channels = np.arange(self.n_channels) if reference_channels is None else np.asarray(reference_channels)
        if (channels.ndim != 1 or len(channels) < 2 or not np.isfinite(channels).all()
                or not np.equal(channels, np.floor(channels)).all()
                or channels.min() < 0 or channels.max() >= self.n_channels
                or len(np.unique(channels)) != len(channels)):
            raise ValueError('Reference channels must contain at least two distinct valid channel indices.')
        self.reference_channels = channels.astype(int)
        self.sos = butter(self.filter_order, self.highpass_hz, fs=self.sample_rate,
                          btype='highpass', output='sos')

    def transform(self, voltage, voltage_scale=1.):
        voltage = np.asarray(voltage, dtype=float)
        if (voltage.ndim != 2 or voltage.shape[1] != self.n_channels
                or not np.isfinite(voltage).all() or not np.isfinite(voltage_scale)
                or voltage_scale <= 0):
            raise ValueError('Invalid voltage dimensions, values, or acquisition scaling.')
        filtered = sosfiltfilt(self.sos, voltage * voltage_scale, axis=0)
        filtered -= np.median(filtered[:, self.reference_channels], axis=1, keepdims=True)
        return filtered

    def metadata(self):
        return dict(profile='shared_unwhitened_v1', highpass_hz=self.highpass_hz,
                    filter_order=self.filter_order, padding_seconds=self.padding_seconds,
                    reference_channels=self.reference_channels.tolist(),
                    filter='Butterworth SOS forward/backward with odd endpoint extension',
                    reference='samplewise median after filtering', whitening=False,
                    baseline_subtraction=False, amplitude_normalization=False,
                    alignment='native exported spike timestamps; no per-spike realignment')
# [AGENT EDIT END]
