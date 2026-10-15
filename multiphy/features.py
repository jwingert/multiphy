"""
Per-unit features for cross-day matching.

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: template, tuning, fingerprint and rate extraction moved to unit_match_lbhb.baphy_features (Plan C final review fix round 1) | date: 2026-10-02]
The lab-free helpers the feature code uses: probe geometry and channel maps,
spike width, trial windows and split halves, stimulus keys, and the spiking
statistics (inter-spike-interval histogram and autocorrelogram, imported from
``extract.unit_features``). The template, tuning, fingerprint and rate
extraction that reads celldb, BAPHY and spike files lives in
``nems_lbhb.unit_match_lbhb.baphy_features``.
# [AGENT EDIT END]

Split halves are taken by ALTERNATING TRIALS, never by alternate spikes:
alternate spikes double every interval, so the half-data interval histogram and
autocorrelogram describe a different process from the full-data ones and cannot
be compared with them.

# [AGENT EDIT START | agent: claude-opus-5 | user: wingertj | reason: cross-day match curation tool | date: 2026-09-14]
"""
import glob
import os
import re
import warnings

import numpy as np
import pandas as pd
import scipy.io as sio

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: celldb is imported inside the four functions that query it, so importing this module loads no lab module (multiphy project folder Task 1) | date: 2026-09-25]
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: ISI_BINS, ACG_BINS, _isi_acg and the temporal extraction functions moved verbatim to extract/unit_features.py and are imported back, so old imports keep working (multiphy project folder Task 11) | date: 2026-09-25]
from .extract.unit_features import (ACG_BINS, ISI_BINS, _checked_sample_spikes, _isi_acg,
                                    average_spike_waveforms, temporal_sample_windows,
                                    temporal_split_unit_features, waveform_sample_selection)
# [AGENT EDIT END]

warnings.filterwarnings('ignore')

SPIKE_FS = 30000.0
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: ISI_BINS and ACG_BINS moved verbatim to extract/unit_features.py and are imported back above (multiphy project folder Task 11) | date: 2026-09-25]
# [AGENT EDIT END]
FP_FS = 20            # 50 ms bins for the fingerprint
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: FTC_FS, MIN_SPIKES and MIN_TEST_REPEATS moved to nems_lbhb.unit_match_lbhb.baphy_features (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]

# Which recording the fingerprint, the rates and the spiking statistics come
# from. They all come from the SAME recording: the physiology references are
# calibrated on within-session split halves, so mixing a rate from one block
# with an interval histogram from another would compare two different states.
DEFAULT_RUNCLASS = 'BNT'

# Runclasses that present a repeated set of complex sounds, so a peristimulus
# histogram over the sounds two days share is a fingerprint. Counted on the
# recordings themselves, stimuli with at least MIN_TEST_REPEATS presentations:
# BNT 12-16 per site, OLP 282, NFB 14-92. BNB is deliberately absent -- it
# presents ONE noise burst whose epoch is named after its level (STIM_316 one
# day, STIM_3162 the next), so two days share no stimulus to compare on.
# Any other runclass can still be named; it will report having no shared sounds
# rather than fail.
FINGERPRINT_RUNCLASSES = ('BNT', 'OLP', 'NFB')


# ------------------------------------------------------------------ geometry
def npx1_geometry(nchan=384):
    """Neuropixels 1.0 as an idealisation: two sites per 20 um row, four
    staggered columns. Only a fallback -- prefer probe_geometry(), which reads
    the channel map the sort actually used.
    """
    k = np.arange(nchan)
    return np.stack([np.array([16.0, 48.0, 0.0, 32.0])[k % 4], 20.0 * (k // 2)], axis=1)


GEOM = npx1_geometry()
UM_PER_CHAN = 10.0

# The channel maps KiloSort was run with. These are the real site coordinates,
# and they are not what the idealisation above produces: the Neuropixels map
# puts its columns at x = 11, 27, 43, 59 and starts at y = 20, and the order of
# the two sites within a row alternates. The y difference is a constant, so it
# cancels in every depth difference and in the registration, but the column a
# channel sits in is simply wrong without the map.
CHANMAP_DIR = '/auto/data/code/KiloSort'

# What to use when only the channel count is known. Several maps share a count
# (64D against 64M, 128D against 128P) and nothing in the sort record we can
# read says which, so this is a default the viewer lets you override.
CHANMAP_DEFAULT = {384: 'chanMap_neuropixPhase3A.mat',
                   128: 'chanMap_128D.mat',
                   64: 'chanMap.mat',
                   4: 'chanMap_Thomas_Tetrode.mat'}
_CHANMAP_CACHE = {}


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _SITE_GEOM, settings_files, geometry_from_settings and site_geometry moved to nems_lbhb.unit_match_lbhb.baphy_features (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


def chanmap_files(n_chan=None, directory=CHANMAP_DIR):
    """Available channel maps, optionally only those with `n_chan` sites."""
    out = []
    for f in sorted(glob.glob(os.path.join(directory, 'chanMap*.mat'))):
        g = load_chanmap(f)
        if g is not None and (n_chan is None or len(g) == n_chan):
            out.append(os.path.basename(f))
    return out


def load_chanmap(path):
    """(n, 2) site coordinates from a KiloSort channel map, or None."""
    if path in _CHANMAP_CACHE:
        return _CHANMAP_CACHE[path]
    g = None
    try:
        m = sio.loadmat(path, squeeze_me=True)
        if 'xcoords' in m and 'ycoords' in m:
            g = np.stack([np.asarray(m['xcoords'], float),
                          np.asarray(m['ycoords'], float)], axis=1)
    except Exception:
        g = None                      # a v7.3 or malformed file is not fatal
    _CHANMAP_CACHE[path] = g
    return g


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no siteid: the site's settings.xml geometry (site_geometry) moved to nems_lbhb.unit_match_lbhb.baphy_features, which this module never imports (Plan C Task 7) | date: 2026-10-02]
def probe_geometry(n_chan, name=None, directory=CHANMAP_DIR):
    """Assumed electrode coordinates for a channel count, best source first.

    A KiloSort channel map matching the channel count comes first, and the
    Neuropixels idealisation is the last resort so a probe with no record at
    all still plots. Both are assumptions about the recording; a recording's
    measured geometry travels with its day instead.
    """
# [AGENT EDIT END]
    if name:
        g = load_chanmap(name if os.path.isabs(name)
                         else os.path.join(directory, name))
        if g is not None and len(g) == n_chan:
            return g
    want = CHANMAP_DEFAULT.get(int(n_chan))
    if want:
        g = load_chanmap(os.path.join(directory, want))
        if g is not None and len(g) == n_chan:
            return g
    for f in chanmap_files(n_chan, directory):
        g = load_chanmap(os.path.join(directory, f))
        if g is not None:
            return g
    return npx1_geometry(n_chan)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: geometry_source, spk_path and load_sortinfo moved to nems_lbhb.unit_match_lbhb.baphy_features (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


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


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: template_features, chan_unit_from_cellid, site_templates, _parmfiles and _load moved to nems_lbhb.unit_match_lbhb.baphy_features (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: tuning and ftc_tuning moved to nems_lbhb.unit_match_lbhb.baphy_features (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]


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


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: _isi_acg moved verbatim to extract/unit_features.py and is imported back above (multiphy project folder Task 11) | date: 2026-09-25]
# [AGENT EDIT END]


def stimulus_key(epoch):
    """Canonical identity of a stimulus, so the same sound matches across the
    two BNT naming styles.

    Half the sites name epochs as a plain file (STIM_00seq4.wav) and half as a
    binaural pair (STIM_00seq4:1+NULL:2). Joining on the raw string treated
    those as unrelated, which split the archive almost in half. Measured within
    a site that presents both, a neuron's response to STIM_00seq4.wav and to
    STIM_00seq4:1+NULL:2 correlates at 0.52 across 222 cells, against 0.013 for
    two genuinely different sounds: same stimulus, different spelling.

    The configuration is kept in the key, so one sound with the other channel
    silent is NOT matched to that sound competing against a second one, and
    channel offsets stay distinct. Only the spelling is normalised.
    """
    body = str(epoch)
    if body.startswith('STIM_'):
        body = body[len('STIM_'):]
    parts = [p for p in body.split('+') if p]
    items = []
    for p in parts:
        bits = p.split(':')
        name = bits[0].replace('.wav', '')
        if name.upper() == 'NULL':
            continue
        offset = bits[2] if len(bits) > 2 else ''
        items.append((name, offset))
    if not items:
        return None
    if len(items) == 1:
        # a lone sound is the same stimulus however it is spelled, provided no
        # channel offset was applied
        name, offset = items[0]
        return ('mono', name, offset)
    return ('multi',) + tuple(sorted(items))


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fingerprint_and_rate moved to nems_lbhb.unit_match_lbhb.baphy_features (Plan C Task 7) | date: 2026-10-02]
# [AGENT EDIT END]
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: temporal_sample_windows, _checked_sample_spikes, waveform_sample_selection, average_spike_waveforms and temporal_split_unit_features moved verbatim to extract/unit_features.py and are imported back above (multiphy project folder Task 11) | date: 2026-09-25]
# [AGENT EDIT END]
