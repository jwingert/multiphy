# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: synthetic recording days with a known rigid probe drift per day (functional-likelihood Task 4) | date: 2026-09-27]
"""Synthetic recording days of one neuron population under a known rigid drift per day.

``make_drift_days`` places ``n_neurons`` neurons on a 96-channel, 4-column probe and records
them on ``n_days`` days; on day ``k`` every neuron sits ``offsets_um[k]`` deeper and is present
with probability ``presence``. Each neuron keeps its waveform, spontaneous rate and condition
PETHs across days, so the same neuron is the same unit on every day it appears. Cellids are
opaque; the neuron behind each is in ``truth`` and ``ground_truth``. ``truth_offsets`` is the
per-day depth correction that registers every day to the mean probe position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from multiphy.backends.local.references import trial_splits, unit_observation
from multiphy.contract.inputs import TrialTiming

SAMPLE_RATE = 30000.0
COLUMNS_X_UM = (11.0, 27.0, 43.0, 59.0)
ROWS_Y_UM = tuple(float(y) for y in range(0, 461, 20))
WAVEFORM_SAMPLES = 41
TROUGH_SAMPLE = 20
TRIAL_SECONDS, BASELINE_SECONDS, STIMULUS_SECONDS = 1.5, 0.25, 1.25
PETH_BINS, PETH_BIN_SECONDS = 25, 0.05
REFRACTORY_SAMPLES = int(0.002 * SAMPLE_RATE)
SPREAD_UM = 30.0
HALF_NOISE_UV = 2.0


@dataclass(frozen=True)
class DriftDays:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: timing is empty for days made without events (event-optional Task 2) | date: 2026-10-09]
    """``units`` (one row per recorded unit), ``features`` (cellid -> feature dict), ``geometry``
    and ``timing`` (session -> channel positions / ``TrialTiming``; ``timing`` is empty for days
    made without events), ``truth`` (cellid, neuron, session_id, true depth and x),
    ``truth_offsets`` (session -> depth correction in µm) and ``sessions`` in day order."""
    # [AGENT EDIT END]
    units: pd.DataFrame
    features: dict
    geometry: dict
    timing: dict
    truth: pd.DataFrame
    truth_offsets: dict
    sessions: tuple


def probe_geometry() -> np.ndarray:
    """(96, 2) channel x, y in µm: 24 rows of the 4 columns."""
    return np.array([(x, y) for y in ROWS_Y_UM for x in COLUMNS_X_UM], dtype=float)


def _shape(lag: int) -> np.ndarray:
    t = np.arange(WAVEFORM_SAMPLES)
    shape = (-np.exp(-0.5 * ((t - TROUGH_SAMPLE) / 1.5) ** 2)
             + 0.4 * np.exp(-0.5 * ((t - TROUGH_SAMPLE - lag) / 3.0) ** 2))
    return shape / -shape[TROUGH_SAMPLE]


def _spikes(rng, rate_per_ms: np.ndarray) -> np.ndarray:
    """Poisson spikes per 1 ms bin, jittered within the bin, then a 2 ms refractory period."""
    counts = rng.poisson(rate_per_ms * 1e-3)
    ms = np.repeat(np.arange(len(counts)), counts)
    per_ms = int(SAMPLE_RATE // 1000)
    samples = np.sort(ms * per_ms + rng.integers(0, per_ms, len(ms)))
    kept, last = [], -REFRACTORY_SAMPLES
    for sample in samples:
        if sample - last >= REFRACTORY_SAMPLES:
            kept.append(sample)
            last = sample
    return np.asarray(kept, dtype=np.int64)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: events=False gives drift days without trial timing (event-optional Task 2) | date: 2026-10-09]
def make_drift_days(*, n_days: int = 3, n_neurons: int = 30, offsets_um: tuple = (0.0, 40.0, -60.0),
                    n_conditions: int = 4, n_repeats: int = 10, presence: float = 0.8, seed: int = 0,
                    events: bool = True) -> DriftDays:
    """``n_days`` synthetic days of ``n_neurons`` neurons; day ``k`` is shifted by ``offsets_um[k]`` µm.

    ``events=False`` returns ``timing={}``; every random draw is kept, so the units, spikes and
    waveforms are the same as with events.
    """
# [AGENT EDIT END]
    if len(offsets_um) != n_days:
        raise ValueError(f'{len(offsets_um)} offsets for {n_days} days; give one offset per day')
    if not 0 < presence <= 1:
        raise ValueError(f'presence is {presence}; it must be in (0, 1]')
    rng = np.random.default_rng(seed)
    geometry = probe_geometry()
    depth = rng.uniform(80, 380, n_neurons)
    x = rng.uniform(11, 59, n_neurons)
    amplitude = rng.uniform(50, 150, n_neurons)
    lag = rng.integers(6, 15, n_neurons)
    spont = rng.uniform(2, 10, n_neurons)
    peth = rng.gamma(1.0, 20.0, (n_neurons, n_conditions, PETH_BINS))

    n_trials = n_conditions * n_repeats
    trial_samples = int(TRIAL_SECONDS * SAMPLE_RATE)
    baseline_samples = int(BASELINE_SECONDS * SAMPLE_RATE)
    stop = n_trials * trial_samples
    ms_per_trial, baseline_ms = int(TRIAL_SECONDS * 1000), int(BASELINE_SECONDS * 1000)
    bin_ms = int(PETH_BIN_SECONDS * 1000)
    mean_offset = float(np.mean(offsets_um))

    units, features, geometries, timings, truth, truth_offsets, sessions = [], {}, {}, {}, [], {}, []
    for k in range(n_days):
        session = f'day{k}'
        sessions.append(session)
        geometries[session] = geometry.copy()
        truth_offsets[session] = -(float(offsets_um[k]) - mean_offset)
        order = rng.permutation(np.repeat(np.arange(n_conditions), n_repeats))
        starts = np.arange(n_trials) * trial_samples
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day without events keeps its trial order draw but records no timing (event-optional Task 2) | date: 2026-10-09]
        if events:
            timings[session] = TrialTiming(
                session, SAMPLE_RATE, (0, stop), np.column_stack((starts, starts + trial_samples)),
                tuple(dict(stimulus=f'sound_{c}', start=int(s + baseline_samples), stop=int(s + trial_samples))
                      for s, c in zip(starts, order)),
                np.column_stack((starts, starts + baseline_samples)),
                {'sample_index_origin': 'zero_based_voltage_row', 'method': 'synthetic'}).validate()
        # [AGENT EDIT END]
        present = np.flatnonzero(rng.random(n_neurons) < presence)
        opaque = rng.permutation(len(present))
        for j, neuron in zip(opaque, present):
            cellid = f'{session}-u{j:03d}'
            rate = np.full(n_trials * ms_per_trial, spont[neuron])
            for trial, condition in enumerate(order):
                begin = trial * ms_per_trial + baseline_ms
                rate[begin:begin + PETH_BINS * bin_ms] += np.repeat(peth[neuron, condition], bin_ms)
            position = np.array([x[neuron], depth[neuron] + offsets_um[k]])
            distance = np.linalg.norm(geometry - position, axis=1)
            waveform = (amplitude[neuron] * _shape(int(lag[neuron]))[:, None]
                        * np.exp(-distance ** 2 / (2 * SPREAD_UM ** 2))[None, :])
            halves = waveform[..., None] + rng.normal(0, HALF_NOISE_UV, waveform.shape + (2,))
            features[cellid] = dict(start_sample=0, stop_sample=stop, sample_rate=SAMPLE_RATE,
                                    spike_samples=_spikes(rng, rate), fingerprint_bin_seconds=PETH_BIN_SECONDS,
                                    waveform=halves.mean(axis=2), waveform_halves=halves)
            units.append(dict(cellid=cellid, session_id=session, su_type='SU', recording_group='synthetic',
                              session_date=f'2026-01-{k + 1:02d}'))
            truth.append(dict(cellid=cellid, neuron=f'n{neuron:03d}', session_id=session,
                              depth_um=float(position[1]), x_um=float(position[0])))
    units_table = pd.DataFrame(units).sort_values(['session_id', 'cellid'], ignore_index=True)
    truth_table = pd.DataFrame(truth).set_index('cellid').loc[units_table.cellid].reset_index()
    return DriftDays(units_table, features, geometries, timings, truth_table, truth_offsets, tuple(sessions))


def observations_by_session(days: DriftDays, *, n_splits: int = 20, seed: int = 0) -> dict:
    """session -> cellid -> ``unit_observation`` on that day's ``trial_splits``."""
    observations: dict = {}
    for session in days.sessions:
        timing = days.timing[session]
        splits = trial_splits(timing, *timing.analysis_window, n_splits=n_splits, seed=seed)
        cellids = days.units.cellid[days.units.session_id == session]
        observations[session] = {cellid: unit_observation(days.features[cellid], days.geometry[session], timing, splits)
                                 for cellid in cellids}
    return observations


def ground_truth(days: DriftDays) -> pd.DataFrame:
    """The benchmark label table: ``cellid``, ``source_unit_id`` (the neuron), ``source_session`` and ``well``."""
    return pd.DataFrame({'cellid': days.truth.cellid, 'source_unit_id': days.truth.neuron,
                         'source_session': 'synthetic', 'well': 'synthetic'})
# [AGENT EDIT END]
