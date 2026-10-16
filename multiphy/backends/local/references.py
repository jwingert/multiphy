"""Same-unit references: alternating halves and resampled splits of a day's trials or 5 s chunks.

``alternating_references`` is the original odd/even reference (moved verbatim from
``functional_likelihood.py``). ``trial_splits`` draws ``n_splits`` two-way splits of a
session's trials, baseline windows and per-condition stimulus repeats: split 0 alternates,
every later split is a seeded, stratified random half. ``unit_halves`` measures one unit on
both halves of one split, and ``unit_observation`` gathers a unit's waveform summary, its
alternating references and its halves on every split. ``group_observations`` observes every
single unit of each recording group on its own day's splits.

Each reader takes a ``ReferenceWindows`` or a ``TrialTiming``, which share their attribute
names. ``evidence_by_group`` decides once per recording group what its days measured
(``run_evidence``), and ``reference_windows`` builds a unit's windows from it: the day's trials
and stimuli in an event run, else ``chunk_windows`` of the unit's window (``CHUNK_SECONDS``) as
trials and no stimuli; the day's baseline windows when every day has them, else the chunks.
Nothing here reads a lab format.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import evidence as ev
from ...benchmark_pairs import _waveform_summary
from ...contract.inputs import RunEvidence, TrialTiming, run_evidence
from ...contract.results import single_units
from ...features import _checked_sample_spikes, _in_windows, _isi_acg, _split_by_trial
from .neighbourhood import unit_neighbourhood

CHUNK_SECONDS = 5.0


def chunk_windows(start: int, stop: int, sample_rate: float, seconds: float = CHUNK_SECONDS) -> np.ndarray:
    """``(n, 2)`` windows of ``int(round(seconds * sample_rate))`` samples from ``start``, each ending by ``stop``.

    The leftover shorter than one chunk is dropped; a window shorter than one chunk gives shape
    ``(0, 2)``. A non-finite or non-positive ``sample_rate`` or ``seconds``, a chunk under one
    sample, or ``start >= stop`` raises ``ValueError``.
    """
    for name, value in (('sample_rate', sample_rate), ('seconds', seconds)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{name} is {value!r}; chunk windows need a finite positive {name}')
    if start >= stop:
        raise ValueError(f'chunk windows need start < stop, not [{start}, {stop})')
    width = int(round(seconds * sample_rate))
    if width < 1:
        raise ValueError(f'{seconds:g} s at {sample_rate:g} Hz is {width} samples; a chunk needs at least one sample')
    starts = start + width * np.arange((stop - start) // width)
    return np.column_stack((starts, starts + width))


@dataclass(frozen=True)
class ReferenceWindows:
    """The windows one unit's references are measured from, under ``TrialTiming``'s attribute names.

    ``trial_windows``: the day's trials (event runs) or ``chunk_windows`` of the unit's window.
    ``stimuli``: the day's stimulus repeats (event runs) or none. ``baseline_windows``: the day's
    baseline windows (baseline spont) or the chunks (whole-recording spont). Samples are on the
    acquisition clock at ``sample_rate``, inside ``analysis_window``.
    """
    sample_rate: float
    analysis_window: tuple
    trial_windows: np.ndarray
    stimuli: list
    baseline_windows: np.ndarray

    def validate(self) -> 'ReferenceWindows':
        """Raise ``ValueError`` unless the rate is positive, each window array is ``(n, 2)`` integer
        windows, ordered, disjoint and inside ``analysis_window``, and every stimulus lies inside a trial."""
        if not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError(f'sample_rate is {self.sample_rate!r}; reference windows need a finite positive rate')
        start, stop = self.analysis_window
        if not all(np.isfinite(x) and int(x) == x for x in (start, stop)) or not 0 <= start < stop:
            raise ValueError(f'analysis window {self.analysis_window!r} is not an integer [start, stop) interval')
        for name in ('trial_windows', 'baseline_windows'):
            windows = np.asarray(getattr(self, name))
            if (windows.ndim != 2 or windows.shape[1] != 2 or not np.isfinite(windows).all()
                    or not np.equal(windows, np.floor(windows)).all()):
                raise ValueError(f'{name} must be (n, 2) integer sample windows')
            if (windows[:, 1] <= windows[:, 0]).any():
                raise ValueError(f'{name} must each start before they stop')
            if (windows[1:, 0] < windows[:-1, 1]).any():
                raise ValueError(f'{name} must be ordered and must not overlap')
            if (windows[:, 0] < start).any() or (windows[:, 1] > stop).any():
                raise ValueError(f'{name} must lie inside the analysis window [{start}, {stop})')
        trials = np.asarray(self.trial_windows)
        for stimulus in self.stimuli:
            a, b = stimulus['start'], stimulus['stop']
            if not a < b or not ((trials[:, 0] <= a) & (trials[:, 1] >= b)).any():
                raise ValueError(f"stimulus {stimulus['stimulus']!r} at [{a}, {b}) is not inside a trial window")
        return self


def reference_windows(feature: dict, timing: Optional[TrialTiming], evidence: RunEvidence) -> ReferenceWindows:
    """The validated windows ``feature``'s unit is measured from in a run that measured ``evidence``.

    With ``evidence.events``: the timing's trials and stimuli; without: ``chunk_windows`` of the
    unit's ``[start_sample, stop_sample)`` as trials and no stimuli. Baselines: the timing's
    baseline windows when ``evidence.spont == 'baseline'``, else the chunk windows. A run that
    reads timing takes the timing's rate and analysis window; one that reads none, the unit's.
    """
    start, stop = feature['start_sample'], feature['stop_sample']
    rate = float(feature['sample_rate'])
    baseline = evidence.spont == 'baseline'
    if evidence.reads_timing:
        timing.validate()
    chunks = None if evidence.events and baseline else chunk_windows(start, stop, rate)
    return ReferenceWindows(
        sample_rate=float(timing.sample_rate) if evidence.reads_timing else rate,
        analysis_window=tuple(timing.analysis_window) if evidence.reads_timing else (start, stop),
        trial_windows=np.asarray(timing.trial_windows) if evidence.events else chunks,
        stimuli=list(timing.stimuli) if evidence.events else [],
        baseline_windows=np.asarray(timing.baseline_windows).reshape(-1, 2) if baseline else chunks).validate()


def alternating_references(feature, timing, *, min_repeats=5, min_spikes=200):
    """Recompute original odd/even trial references; never reuse chronological halves.

    On a ``ReferenceWindows`` without stimuli (a no-event unit's, whose trials are its 5 s
    chunks), the full ``rate_session``, ``isi`` and ``acg`` use only the spikes inside the chunks;
    ``n_spikes`` and ``reference_eligible`` count every spike in the window.
    """
    timing.validate()
    start, stop = feature['start_sample'], feature['stop_sample']
    fs = float(feature['sample_rate'])
    if fs != timing.sample_rate or not timing.analysis_window[0] <= start < stop <= timing.analysis_window[1]:
        raise ValueError('Feature window/sample rate disagrees with verified timing')
    spikes = _checked_sample_spikes(feature['spike_samples'])
    if ((spikes < start) | (spikes >= stop)).any():
        raise ValueError('Retained full spike train escapes its observation window')
    trials = np.asarray(timing.trial_windows)
    if ((trials[:, 0] < start) & (trials[:, 1] > start)).any() or ((trials[:, 0] < stop) & (trials[:, 1] > stop)).any():
        raise ValueError('Legacy reference observation cuts a complete trial')
    trials = trials[(trials[:, 0] >= start) & (trials[:, 1] <= stop)] / fs
    baseline = np.asarray(timing.baseline_windows).reshape(-1, 2)
    baseline = baseline[(baseline[:, 0] >= start) & (baseline[:, 1] <= stop)] / fs
    times = spikes / fs
    parts = _split_by_trial(times, trials)
    hist = [_isi_acg(part) for part in parts]
    rates = []
    for windows in (baseline, baseline[::2], baseline[1::2]):
        selected, duration = _in_windows(times, windows)
        rates.append(len(selected) / duration if duration else np.nan)
    chunked = isinstance(timing, ReferenceWindows) and not timing.stimuli
    full = _in_windows(times, trials)[0] if chunked else times
    result = dict(spont=rates[0], spont_a=rates[1], spont_b=rates[2],
        isi=_isi_acg(full)[0], acg=_isi_acg(full)[1], isi_a=hist[0][0], isi_b=hist[1][0],
        acg_a=hist[0][1], acg_b=hist[1][1], n_spikes=len(times),
        rate_session=len(full) / (full[-1] - full[0]) if len(full) > 1 else np.nan,
        fingerprints={}, halves={}, per_stim={}, trial_counts={}, reference_eligible=len(spikes) >= min_spikes)
    width = int(round(feature['fingerprint_bin_seconds'] * fs))
    if width < 1:
        raise ValueError('Fingerprint bin duration must span acquisition samples')
    table = pd.DataFrame(list(timing.stimuli), columns=['stimulus', 'start', 'stop'])
    table = table[(table.start >= start) & (table.stop <= stop)]
    for condition, rows in table.groupby('stimulus', sort=True):
        rows = rows.sort_values('start')
        if len(rows) < min_repeats:
            continue
        durations = (rows.stop - rows.start).to_numpy()
        if not (durations == durations[0]).all():
            raise ValueError('Exact acoustic conditions must have identical durations')
        edges = np.arange(int(durations[0] // width) + 1) * width
        if len(edges) < 2:
            continue
        counts = [np.histogram(spikes[(spikes >= t.start) & (spikes < t.start + edges[-1])] - t.start, edges)[0]
                  for t in rows.itertuples()]
        values = np.asarray(counts, float) / (width / fs)
        result['fingerprints'][condition] = values.mean(axis=0)
        result['halves'][condition] = (values[::2].mean(axis=0), values[1::2].mean(axis=0))
        result['trial_counts'][condition] = len(rows)
        count = sum(np.searchsorted(spikes, b) - np.searchsorted(spikes, a) for a, b in zip(rows.start, rows.stop))
        result['per_stim'][condition] = count / (durations.sum() / fs)
    keys = sorted(result['halves'])
    result['reliability'] = (ev.spearman_brown(ev.pearson(
        np.concatenate([result['halves'][k][0] for k in keys]),
        np.concatenate([result['halves'][k][1] for k in keys]))) if keys else np.nan)
    return result


@dataclass(frozen=True)
class TrialSplit:
    """One two-way split of a session: 0/1 per trial, per baseline window and per repeat of each condition.

    Arrays follow the windows fully inside the split's ``[start_sample, stop_sample)``, in start
    order; ``stimulus_half`` maps each condition to its repeats' halves.
    """
    index: int
    trial_half: np.ndarray
    baseline_half: np.ndarray
    stimulus_half: dict


def _inside(windows, start, stop) -> np.ndarray:
    windows = np.asarray(windows).reshape(-1, 2)
    return windows[(windows[:, 0] >= start) & (windows[:, 1] <= stop)]


def _repeats(timing, start, stop) -> dict:
    """Condition -> its repeats inside ``[start, stop)`` as (repeat, 2) sample windows in start order."""
    by_condition: dict = {}
    for stimulus in timing.stimuli:
        if stimulus['start'] >= start and stimulus['stop'] <= stop:
            by_condition.setdefault(stimulus['stimulus'], []).append((stimulus['start'], stimulus['stop']))
    return {c: np.asarray(sorted(by_condition[c]), dtype=float).reshape(-1, 2) for c in sorted(by_condition)}


def _random_half(rng, n: int) -> np.ndarray:
    perm = rng.permutation(n)
    half = np.ones(n, int)
    half[perm[:n // 2]] = 0
    return half


def trial_splits(timing: Union[ReferenceWindows, TrialTiming], start_sample: int, stop_sample: int, *,
                 n_splits: int = 20, seed: int = 0) -> tuple:
    """``n_splits`` ``TrialSplit``s of the windows fully inside ``[start_sample, stop_sample)``.

    Split 0 alternates (``np.arange(n) % 2``, the halves of ``alternating_references``).
    Split ``k >= 1`` draws from ``np.random.default_rng([seed, k])``: trials, then baseline
    windows, then each condition in sorted order, each put ``n // 2`` into half 0.
    """
    if not isinstance(n_splits, int) or n_splits < 1:
        raise ValueError(f'n_splits is {n_splits!r}; trial splits need at least one split (split 0 alternates)')
    timing.validate()
    n_trials = len(_inside(timing.trial_windows, start_sample, stop_sample))
    n_baseline = len(_inside(timing.baseline_windows, start_sample, stop_sample))
    repeats = {c: len(r) for c, r in _repeats(timing, start_sample, stop_sample).items()}
    splits = [TrialSplit(0, np.arange(n_trials) % 2, np.arange(n_baseline) % 2,
                         {c: np.arange(n) % 2 for c, n in repeats.items()})]
    for k in range(1, n_splits):
        rng = np.random.default_rng([seed, k])
        trial_half = _random_half(rng, n_trials)
        baseline_half = _random_half(rng, n_baseline)
        splits.append(TrialSplit(k, trial_half, baseline_half, {c: _random_half(rng, n) for c, n in repeats.items()}))
    return tuple(splits)


def _checked_spikes(feature, timing) -> np.ndarray:
    """The unit's spike samples, with the checks ``alternating_references`` applies."""
    timing.validate()
    start, stop = feature['start_sample'], feature['stop_sample']
    if (float(feature['sample_rate']) != timing.sample_rate
            or not timing.analysis_window[0] <= start < stop <= timing.analysis_window[1]):
        raise ValueError('Feature window/sample rate disagrees with verified timing')
    spikes = _checked_sample_spikes(feature['spike_samples'])
    if ((spikes < start) | (spikes >= stop)).any():
        raise ValueError('Retained full spike train escapes its observation window')
    trials = np.asarray(timing.trial_windows)
    if ((trials[:, 0] < start) & (trials[:, 1] > start)).any() or ((trials[:, 0] < stop) & (trials[:, 1] > stop)).any():
        raise ValueError('Reference observation cuts a complete trial')
    return spikes


def _count(spikes, windows) -> int:
    return int(sum(np.searchsorted(spikes, b) - np.searchsorted(spikes, a) for a, b in windows))


def unit_halves(feature: dict, timing: Union[ReferenceWindows, TrialTiming], split: TrialSplit, *,
                min_repeats: int = 5) -> tuple:
    """``(half_0, half_1)``: the unit measured on each half of ``split``.

    Each half holds ``fingerprints`` (condition -> mean rate PETH over the half's repeats,
    binned as ``alternating_references`` bins, for conditions with at least ``min_repeats``
    repeats), ``per_stim`` (condition -> rate over those repeats), ``spont`` (rate in the
    half's baseline windows), ``isi`` and ``acg`` (of spikes inside the half's trials),
    ``rate`` (spikes in the half's trials over their duration) and ``n_spikes`` (spikes in
    the half's trials). A silent unit has rate 0 over any window of positive duration; a rate
    is NaN only when the half has no window. ``excluded`` maps each condition left out of
    ``fingerprints`` and ``per_stim`` to its named reason (too few repeats, a condition
    shorter than one bin, or no repeat in this half).
    """
    spikes = _checked_spikes(feature, timing)
    start, stop = feature['start_sample'], feature['stop_sample']
    fs = float(feature['sample_rate'])
    trials = _inside(timing.trial_windows, start, stop)
    baseline = _inside(timing.baseline_windows, start, stop)
    repeats = _repeats(timing, start, stop)
    if (len(split.trial_half) != len(trials) or len(split.baseline_half) != len(baseline)
            or set(split.stimulus_half) != set(repeats)
            or any(len(split.stimulus_half[c]) != len(r) for c, r in repeats.items())):
        raise ValueError(f'Split {split.index} was drawn for other windows than [{start}, {stop}) of this timing; '
                         'draw trial_splits over the feature window')
    width = int(round(feature['fingerprint_bin_seconds'] * fs))
    if width < 1:
        raise ValueError('Fingerprint bin duration must span acquisition samples')
    times = spikes / fs
    halves = []
    for h in (0, 1):
        own_trials = trials[np.asarray(split.trial_half) == h]
        own_baseline = baseline[np.asarray(split.baseline_half) == h]
        # exposure comes from the windows themselves: _in_windows reports 0 s for an empty spike train
        in_trials, _ = _in_windows(times, own_trials / fs)
        in_baseline, _ = _in_windows(times, own_baseline / fs)
        trial_seconds = float((own_trials[:, 1] - own_trials[:, 0]).sum()) / fs
        baseline_seconds = float((own_baseline[:, 1] - own_baseline[:, 0]).sum()) / fs
        isi, acg = _isi_acg(in_trials)
        half = dict(fingerprints={}, per_stim={}, excluded={},
                    spont=len(in_baseline) / baseline_seconds if baseline_seconds else np.nan,
                    isi=isi, acg=acg, rate=len(in_trials) / trial_seconds if trial_seconds else np.nan,
                    n_spikes=len(in_trials))
        for condition, rows in repeats.items():
            if len(rows) < min_repeats:
                half['excluded'][condition] = (f'{len(rows)} repeats in [{start}, {stop}); '
                                               f'fingerprints need min_repeats={min_repeats}')
                continue
            durations = rows[:, 1] - rows[:, 0]
            if not (durations == durations[0]).all():
                raise ValueError('Exact acoustic conditions must have identical durations')
            edges = np.arange(int(durations[0] // width) + 1) * width
            if len(edges) < 2:
                half['excluded'][condition] = (f'condition lasts {int(durations[0])} samples, '
                                               f'shorter than one {width}-sample fingerprint bin')
                continue
            own = rows[np.asarray(split.stimulus_half[condition]) == h]
            if not len(own):
                half['excluded'][condition] = f'no repeat of {len(rows)} falls in half {h} of split {split.index}'
                continue
            counts = [np.histogram(spikes[(spikes >= a) & (spikes < a + edges[-1])] - a, edges)[0] for a in own[:, 0]]
            half['fingerprints'][condition] = (np.asarray(counts, float) / (width / fs)).mean(axis=0)
            half['per_stim'][condition] = _count(spikes, own) / ((own[:, 1] - own[:, 0]).sum() / fs)
        halves.append(half)
    return tuple(halves)


def unit_observation(feature: dict, geometry: np.ndarray, timing: Union[ReferenceWindows, TrialTiming],
                     splits: Sequence[TrialSplit], *, min_repeats: int = 5, min_spikes: int = 200) -> dict:
    """A unit's waveform summary, alternating references and halves on every split.

    Adds ``splits`` (one ``unit_halves`` pair per split), ``geometry``, ``waveform_halves``
    (the feature's, or ``None``), ``sw_halves`` (the spike width of each waveform half, or ``None``)
    and ``neighbourhood`` (``unit_neighbourhood`` of the mean waveform half, or ``None``).
    """
    observation = dict(_waveform_summary(feature, geometry),
                       **alternating_references(feature, timing, min_repeats=min_repeats, min_spikes=min_spikes))
    waveform_halves: Optional[np.ndarray] = feature.get('waveform_halves')
    sw_halves = None
    if waveform_halves is not None:
        waveform_halves = np.asarray(waveform_halves, dtype=float)
        whole = {key: value for key, value in feature.items() if key != 'waveform_halves'}
        sw_halves = tuple(_waveform_summary(dict(whole, waveform=waveform_halves[..., h]), geometry)['sw_ms']
                          for h in (0, 1))
    neighbourhood = (unit_neighbourhood(waveform_halves.mean(axis=2), geometry)
                     if waveform_halves is not None else None)
    observation.update(splits=[unit_halves(feature, timing, s, min_repeats=min_repeats) for s in splits],
                       geometry=np.asarray(geometry, dtype=float), waveform_halves=waveform_halves,
                       sw_halves=sw_halves, neighbourhood=neighbourhood)
    return observation


@dataclass(frozen=True)
class GroupObservations:
    """One recording group ready to match.

    ``sessions`` in day order; ``observations`` maps cellid -> ``unit_observation`` for the
    units that pass the spike minimum, and the reliability minimum in an event group;
    ``session_of`` maps those cellids to their session; ``keys`` are the sorted conditions every
    kept observation measures, ``()`` in a no-event group.
    """
    group: str
    sessions: tuple
    observations: dict
    session_of: dict
    keys: tuple


def _session_order(members: pd.DataFrame, features: dict) -> list:
    """Session ids of ``members`` by ``(session_date, session_id)``, else by first ``start_sample``."""
    sessions = list(members.groupby('session_id', sort=True))
    if 'session_date' in members:
        return [s for s, _ in sorted(sessions, key=lambda item: (str(item[1].session_date.iloc[0]), item[0]))]
    return [s for s, _ in sorted(sessions, key=lambda item: (
        min(features[c]['start_sample'] for c in item[1].cellid), item[0]))]


def _single_unit_members(units: pd.DataFrame) -> pd.DataFrame:
    """The explicit single units of ``units``; each needs a recording group, and a session stays in one group."""
    eligible, _ = single_units(units)
    if 'recording_group' not in eligible or eligible.recording_group.isna().any():
        raise ValueError('Original recording groups must be explicit')
    if eligible.groupby('session_id').recording_group.nunique().gt(1).any():
        raise ValueError('A matcher session cannot span original recording groups')
    return eligible


def evidence_by_group(units: pd.DataFrame, features: dict, timing_by_session: dict, *, min_repeats: int) -> dict:
    """``{group: run_evidence(...)}``: what each recording group's days measured.

    A group's sessions are those of its explicit single units in ``_session_order``; each is
    given the timing ``timing_by_session`` holds for it, and a session it lacks has no events.
    """
    evidence = {}
    for group, members in _single_unit_members(units).groupby('recording_group', sort=True):
        sessions = _session_order(members, features)
        evidence[str(group)] = run_evidence(
            sessions, {s: timing_by_session[s] for s in sessions if s in timing_by_session}, min_repeats=min_repeats)
    return evidence


def require_waveform_halves(units: pd.DataFrame, features: dict, backend: str) -> None:
    """Raise ``ValueError`` naming ``backend`` unless every explicit single unit's feature has waveform halves."""
    eligible, _ = single_units(units)
    missing = [c for c in eligible.cellid if (features.get(c) or {}).get('waveform_halves') is None]
    if missing:
        raise ValueError(f'{backend} needs waveform halves from raw voltage; {len(missing)} units have none, '
                         f'first {missing[0]} (templates tier). Re-extract from raw voltage.')


def _channel_layout(geometry: np.ndarray) -> tuple:
    """Sorted unique channel columns (x) and sorted row spacings (y, with multiplicities) of a probe."""
    geometry = np.asarray(geometry, dtype=float)
    return np.unique(geometry[:, 0]), np.sort(np.diff(np.unique(geometry[:, 1])))


def _measured_conditions(observation: dict) -> set:
    """Conditions with a fingerprint in the alternating references and in both halves of every split."""
    measured = set(observation['fingerprints'])
    for halves in observation['splits']:
        for half in halves:
            measured &= set(half['fingerprints'])
    return measured


def _check_aligned(group: str, observations: dict, features: dict, keys) -> None:
    """Raise unless every kept unit has one fingerprint bin width and one length per shared condition."""
    bins = sorted({float(features[c]['fingerprint_bin_seconds']) for c in observations})
    if len(bins) != 1:
        raise ValueError(f'group {group}: fingerprint bin widths {bins} s differ; re-extract every day with one '
                         'bin width, no duration substitution')
    for key in keys:
        lengths = sorted({len(fingerprint[key]) for o in observations.values()
                          for fingerprint in [o['fingerprints'], *(h['fingerprints'] for halves in o['splits']
                                                                   for h in halves)]})
        if len(lengths) != 1:
            raise ValueError(f'group {group}: condition {key!r} has fingerprint lengths {lengths}; every day needs '
                             'the same condition duration, no alias or duration substitution')


def group_observations(units, features, geometry, timing_by_session, *, min_repeats: int = 5,
                       min_spikes: int = 200, min_reliability: float = 0.30, n_splits: int = 20,
                       seed: int = 0, evidence: Optional[dict] = None) -> tuple:
    """``(groups, skipped)``: a ``GroupObservations`` per matchable recording group, and the rest.

    ``evidence`` maps each group to its ``RunEvidence`` (``None``: ``evidence_by_group``). Every
    explicit single unit is observed with ``unit_observation`` on its ``reference_windows`` and one
    ``trial_splits`` of them per session window. An event group keeps the units with
    ``reference_eligible`` and ``reliability >= min_reliability``; a group with no condition every
    kept unit measures goes to ``skipped`` as ``{'group', 'reason'}``, and kept units with different
    fingerprint bin widths, or different lengths of one shared condition, raise ``ValueError``. A
    no-event group keeps the units with ``reference_eligible``, with keys ``()``; its observations'
    full ``rate_session``, ``isi`` and ``acg`` use only the spikes inside their chunks, and a unit
    whose window holds fewer than 2 chunks goes to ``skipped`` as ``{'group', 'session', 'cellid',
    'reason'}``. In every group, fewer than 2 sessions holding a kept unit goes to ``skipped`` as
    ``{'group', 'reason'}``, and a session with no kept unit is named as ``{'group', 'session', 'reason'}``.
    """
    if not isinstance(min_repeats, int) or min_repeats < 2 or not isinstance(min_spikes, int) or min_spikes < 2:
        raise ValueError('Reference repeat and spike minima must be integers at least two')
    eligible = _single_unit_members(units)
    if evidence is None:
        evidence = evidence_by_group(units, features, timing_by_session, min_repeats=min_repeats)
    groups, skipped = [], []
    for group, members in eligible.groupby('recording_group', sort=True):
        group = str(group)
        sessions = _session_order(members, features)
        layouts = [_channel_layout(geometry[s]) for s in sessions]
        if any(not (np.array_equal(x, layouts[0][0]) and np.array_equal(y, layouts[0][1])) for x, y in layouts[1:]):
            raise ValueError('sessions of a group need the same channel columns and spacing')
        if len(sessions) < 2:
            skipped.append(dict(group=group, reason=f'{len(sessions)} session with single units; matching needs at least 2'))
            continue
        events = evidence[group].events
        observations, session_of = {}, {}
        for session in sessions:
            timing = timing_by_session.get(session)
            windows_by_span: dict = {}
            for cellid in members.cellid[members.session_id == session]:
                feature = features[cellid]
                window = (feature['start_sample'], feature['stop_sample'])
                if window not in windows_by_span:
                    windows = reference_windows(feature, timing, evidence[group])
                    short = not events and len(windows.trial_windows) < 2
                    windows_by_span[window] = (windows, None if short else trial_splits(
                        windows, *window, n_splits=n_splits, seed=seed))
                windows, splits = windows_by_span[window]
                if splits is None:
                    skipped.append(dict(group=group, session=session, cellid=cellid, reason=(
                        f'{cellid}: window [{window[0]}, {window[1]}) holds fewer than 2 {CHUNK_SECONDS:g} s chunks; '
                        'this unit is not matched')))
                    continue
                observation = unit_observation(feature, geometry[session], windows, splits,
                                               min_repeats=min_repeats, min_spikes=min_spikes)
                if observation['reference_eligible'] and (not events or observation['reliability'] >= min_reliability):
                    observations[cellid], session_of[cellid] = observation, session
        kept = [s for s in sessions if s in set(session_of.values())]
        for session in sessions:
            if session not in kept:
                n_units = int((members.session_id == session).sum())
                skipped.append(dict(group=group, session=session, reason=(
                    f'0 of {n_units} single units have at least {min_spikes} spikes, {min_repeats} repeats and '
                    f'reliability >= {min_reliability:g}; this day is not matched' if events else
                    f'0 of {n_units} single units have at least {min_spikes} spikes; this day is not matched')))
        if len(kept) < 2:
            skipped.append(dict(group=group, reason=(
                f'{len(kept)} of {len(sessions)} sessions hold a unit with at least {min_spikes} spikes and '
                f'reliability >= {min_reliability:g}; matching needs at least 2' if events else
                f'{len(kept)} of {len(sessions)} sessions hold a unit with at least {min_spikes} spikes; '
                'matching needs at least 2')))
            continue
        if not events:
            groups.append(GroupObservations(group, tuple(kept), observations, session_of, ()))
            continue
        keys = tuple(sorted(set.intersection(*(_measured_conditions(o) for o in observations.values()))))
        if not keys:
            skipped.append(dict(group=group, reason=(
                f'no condition with at least {min_repeats} repeats is measured by all {len(observations)} kept units')))
            continue
        _check_aligned(group, observations, features, keys)
        groups.append(GroupObservations(group, tuple(kept), observations, session_of, keys))
    return tuple(groups), skipped
