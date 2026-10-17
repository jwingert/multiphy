"""
Per-unit spiking statistics for cross-day matching.

Spike width, trial windows and split halves, and the spiking statistics
(inter-spike-interval histogram and autocorrelogram, imported from
``extract.unit_features``).

Split halves are taken by ALTERNATING TRIALS, never by alternate spikes:
alternate spikes double every interval, so the half-data interval histogram and
autocorrelogram describe a different process from the full-data ones and cannot
be compared with them.

"""
import re
import warnings

import numpy as np
import pandas as pd

from .extract.unit_features import (ACG_BINS, ISI_BINS, _checked_sample_spikes, _isi_acg,
                                    average_spike_waveforms, temporal_sample_windows,
                                    temporal_split_unit_features, waveform_sample_selection)

warnings.filterwarnings('ignore')

SPIKE_FS = 30000.0
FP_FS = 20            # 50 ms bins for the fingerprint


def spike_width_ms(wf, fs=SPIKE_FS):
    """Trough-to-peak width. Extracellular spikes are trough-dominant."""
    wf = np.asarray(wf, float)
    if wf.size < 5:
        return np.nan
    tr = int(np.argmin(wf))
    if tr >= wf.size - 2:
        return np.nan
    pk = tr + int(np.argmax(wf[tr:]))
    return 1000.0 * (pk - tr) / fs if pk > tr else np.nan






def _windows(ep, name):
    sub = ep[ep.name == name]
    return sub[['start', 'end']].to_numpy() if len(sub) else np.zeros((0, 2))


def _in_windows(st, win):
    if len(win) == 0 or len(st) == 0:
        return np.array([]), 0.0
    idx = np.searchsorted(win[:, 0], st, side='right') - 1
    ok = (idx >= 0) & (idx < len(win))
    keep = np.zeros(len(st), bool)
    keep[ok] = st[ok] < win[idx[ok], 1]
    return st[keep], float((win[:, 1] - win[:, 0]).sum())


def _split_by_trial(st, trial_win):
    if len(trial_win) == 0 or len(st) == 0:
        return st[:0], st[:0]
    idx = np.searchsorted(trial_win[:, 0], st, side='right') - 1
    ok = (idx >= 0) & (idx < len(trial_win))
    inside = np.zeros(len(st), bool)
    inside[ok] = st[ok] < trial_win[idx[ok], 1]
    return st[inside & (idx % 2 == 0)], st[inside & (idx % 2 == 1)]
