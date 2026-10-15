# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: single input object with declared availability | date: 2026-09-22]
"""What a backend may consume.

MatchingInput carries every input any backend could want; each field is optional and
``available()`` reports which are set. The runner compares that set with a backend's
declared ``requires`` before running. RawWaveformBundle is the transport the archive
containers already produce; ``from_bundle`` unpacks it and ``bundle()`` repacks it for
the published-method adapters.
"""
from __future__ import annotations
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run_evidence counts conditions per day and pairs the days (event-optional Task 3) | date: 2026-10-09]
import itertools
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence
# [AGENT EDIT END]

import numpy as np
import pandas as pd

from .provenance import Provenance
from .results import single_units


class Input(str, Enum):
    UNITS = 'units'
    CANDIDATE_PAIRS = 'candidate_pairs'
    FEATURES = 'features'
    WAVEFORM_HALVES = 'waveform_halves'
    GEOMETRY = 'geometry'
    SPIKE_TIMES = 'spike_times'
    FINGERPRINT = 'fingerprint'
    TIMING = 'timing'
    # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no fixed-model input (Task 2) | date: 2026-09-27]
    CHECKPOINT = 'checkpoint'


INPUT_HELP = {
    Input.UNITS: 'unit table with cellid, session_id and su_type',
    Input.CANDIDATE_PAIRS: 'candidate pair table with per-pair evidence columns',
    Input.FEATURES: 'per-unit feature dicts (fingerprints, ISI, ACG, rates, templates)',
    Input.WAVEFORM_HALVES: 'fresh raw waveform means, shape (unit, sample, channel, 2)',
    Input.GEOMETRY: 'channel positions in micrometers per session',
    Input.SPIKE_TIMES: 'spike times in seconds per unit',
    Input.FINGERPRINT: 'aligned stimulus-response fingerprint (unit, stimulus-time-bin)',
    Input.TIMING: 'verified trial and baseline windows per session on the acquisition clock',
    Input.CHECKPOINT: 'a trained checkpoint path with its declared probe type',
}

LABEL_COLUMNS = frozenset({'source_unit_id', 'same_unit', 'partition', 'source_unit'})


@dataclass(frozen=True)
class CheckpointRef:
    path: str
    probe_type: str
    sha256: str | None = None


# --- RawWaveformBundle: moved verbatim from external_backends.py ---
@dataclass(frozen=True)
class RawWaveformBundle:
    """Independently extracted raw means, never duplicated sorting templates.

    Waveforms: (unit, sample, channel, 2 temporal halves). channel_positions is
    a mapping from session ID to (channel, 2) coordinates in micrometers.
    spike_times_seconds has one vector per unit and is required by DANT.
    provenance must identify raw source and split/extraction method.
    """
    cellids: tuple
    waveforms: np.ndarray
    session_ids: tuple
    channel_positions: dict
    spike_times_seconds: tuple = ()
    provenance: dict = field(default_factory=dict)
    peth: np.ndarray = None
    peth_bin_keys: tuple = ()
    waveform_spike_counts: np.ndarray = None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: named by what it holds: LegacyTiming -> TrialTiming (Task 3) | date: 2026-09-27]
# --- TrialTiming: verified acquisition-clock trial, stimulus and baseline windows ---
@dataclass(frozen=True)
class TrialTiming:
# [AGENT EDIT END]
    source_session: str
    sample_rate: float
    analysis_window: tuple
    trial_windows: np.ndarray
    stimuli: tuple
    baseline_windows: np.ndarray
    provenance: dict

    def validate(self):
        if not self.source_session or not np.isfinite(self.sample_rate) or self.sample_rate <= 0:
            raise ValueError('Timing requires a source session and positive acquisition sample rate')
        if self.provenance.get('sample_index_origin') != 'zero_based_voltage_row' or not self.provenance.get('method'):
            raise ValueError('Verified acquisition-clock timing provenance is required')
        start, stop = self.analysis_window
        if not all(np.isfinite(x) and int(x) == x for x in (start, stop)) or not 0 <= start < stop:
            raise ValueError('Invalid verified analysis window')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an empty baseline list is accepted; such a day reads spont from the whole recording (event-optional Task 3) | date: 2026-10-09]
        for name in ('trial_windows', 'baseline_windows'):
            windows = np.asarray(getattr(self, name))
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an empty baseline list or tuple is read as (0, 2) windows, as the (0, 2) array is (event-optional Task 3, final review fix round 1) | date: 2026-10-10]
            if name == 'baseline_windows' and not windows.size:
                windows = windows.reshape(0, 2)
            # [AGENT EDIT END]
            if (windows.ndim != 2 or windows.shape[1] != 2 or (name == 'trial_windows' and not len(windows))
                    or not np.isfinite(windows).all() or not np.equal(windows, np.floor(windows)).all()
                    or (windows[:, 0] < start).any() or (windows[:, 1] > stop).any()
                    or (windows[:, 1] <= windows[:, 0]).any()
                    or (windows[1:, 0] < windows[:-1, 1]).any()):
                raise ValueError(f'{name} must contain complete ordered disjoint acquisition intervals')
        # [AGENT EDIT END]
        for trial in self.stimuli:
            a, b = trial['start'], trial['stop']
            if (not isinstance(trial['stimulus'], str) or not trial['stimulus']
                    or not all(np.isfinite(x) and int(x) == x for x in (a, b)) or not start <= a < b <= stop):
                raise ValueError('Complete exact-condition stimulus timing is required')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the containment loop reads baselines as (n, 2), n may be 0 (event-optional Task 3) | date: 2026-10-09]
        for a, b in [*np.asarray(self.baseline_windows).reshape(-1, 2).tolist(),
                     *[(t['start'], t['stop']) for t in self.stimuli]]:
            if not ((self.trial_windows[:, 0] <= a) & (self.trial_windows[:, 1] >= b)).any():
                raise ValueError('Stimulus or baseline is outside its nominal TRIALSTART/TRIALSTOP interval')
        # [AGENT EDIT END]
        return self

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: timing built from a day's source/ entry (functional-likelihood Task 4) | date: 2026-09-27]
    @classmethod
    def from_source_entry(cls, entry: dict) -> 'TrialTiming':
        """The validated timing of a day's ``source/`` recording entry (``read_source(...)['entry']``).

        Trials are checked first: a day whose source has no trial events has no timing.
        """
        trials = entry.get('trials')
        if trials is None or not len(trials):
            raise ValueError(f"day {entry.get('day')}: timing needs trial events and this day's source has none; "
                             'add an events reader')
        provenance = dict(entry['clock_verification'], day=entry.get('day'), project=entry.get('project'))
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day without baseline windows gets an empty (0, 2) baseline array (event-optional Task 3) | date: 2026-10-09]
        baselines = (np.asarray(entry['baseline_windows']) if len(entry['baseline_windows'])
                     else np.empty((0, 2), np.int64))
        return cls(source_session=entry['source_session'], sample_rate=float(entry['sample_rate']),
                   analysis_window=tuple(entry['analysis_window']), trial_windows=np.asarray(trials),
                   stimuli=tuple(dict(stimulus) for stimulus in entry['stimuli']),
                   baseline_windows=baselines, provenance=provenance).validate()
        # [AGENT EDIT END]
    # [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one function decides which evidence a run's days measured; each fallback is a saved reason (event-optional Task 3) | date: 2026-10-09]
NO_EVENTS_TAG = 'no events'
WHOLE_RECORDING_TAG = 'spont · whole recording'
_TAG_ORDER = (NO_EVENTS_TAG, WHOLE_RECORDING_TAG)


@dataclass(frozen=True)
class RunEvidence:
    """The evidence a run's days measured, decided once by ``run_evidence``.

    ``events``: every day has timing and at least one condition has ``min_repeats`` repeats on every day.
    ``spont``: ``'baseline'`` when every day has baseline windows, else ``'whole_recording'`` on every day.
    ``shared``: ``{'<day_a>+<day_b>': n}``, the conditions with ``min_repeats`` repeats on both days of a pair.
    ``reasons``: one warning line per fallback, each starting with its tag.
    """
    events: bool
    spont: str
    shared: dict
    reasons: tuple

    @property
    def reads_timing(self) -> bool:
        """True when the run reads any timing: its events, or its baseline windows for spont."""
        return self.events or self.spont == 'baseline'

    def to_dict(self) -> dict:
        return dict(events=self.events, spont=self.spont, shared=dict(self.shared), reasons=list(self.reasons))


def run_evidence(sessions: Sequence[str], timing_by_session: dict, *, min_repeats: int = 5) -> RunEvidence:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: min_repeats must be an int >= 2 (event-optional Task 14) | date: 2026-10-10]
    """The evidence the days ``sessions`` measured, from ``timing_by_session`` (keyed only by days with timing).

    A day's measured conditions are the stimuli with at least ``min_repeats`` entries in its timing; a day
    without timing measures none. Events need every day timed and a condition measured on every day.
    ``min_repeats`` must be an int >= 2: a reference splits a condition's repeats in two.
    """
    # [AGENT EDIT END]
    sessions = tuple(sessions)
    if not sessions:
        raise ValueError('no sessions')
    if not all(isinstance(session, str) for session in sessions):
        raise ValueError(f'sessions must be strings: {sessions!r}')
    if len(set(sessions)) != len(sessions):
        raise ValueError(f'duplicate sessions: {sorted(s for s, n in Counter(sessions).items() if n > 1)}')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one repeat is refused, as group_observations refuses it (event-optional Task 14) | date: 2026-10-10]
    if isinstance(min_repeats, bool) or not isinstance(min_repeats, int) or min_repeats < 2:
        raise ValueError(f'min_repeats must be an int >= 2, not {min_repeats!r}')
    # [AGENT EDIT END]
    timed = {session: timing_by_session[session] for session in sessions if session in timing_by_session}
    measured = {}
    for session in sessions:
        counts = Counter(trial['stimulus'] for trial in timed[session].stimuli) if session in timed else Counter()
        measured[session] = {condition for condition, n in counts.items() if n >= min_repeats}
    shared = {f'{a}+{b}': len(measured[a] & measured[b]) for a, b in itertools.combinations(sessions, 2)}
    events = len(timed) == len(sessions) and bool(set.intersection(*measured.values()))
    no_baseline = [session for session in sessions if session not in timed
                   or not len(np.asarray(timed[session].baseline_windows).reshape(-1, 2))]
    spont = 'whole_recording' if no_baseline else 'baseline'
    reasons = []
    if not events:
        pairs = ' · '.join(f'{k} {n}' for k, n in shared.items())
        reasons.append(f'{NO_EVENTS_TAG} · shared conditions per day pair: {pairs or "none"}')
    if spont == 'whole_recording':
        reasons.append(f'{WHOLE_RECORDING_TAG} · days without baseline windows: {", ".join(no_baseline)}')
    return RunEvidence(events=events, spont=spont, shared=shared, reasons=tuple(reasons))


def evidence_tags(reasons: Sequence[str]) -> list:
    """``[(tag, distinct reasons joined by newlines)]`` in tag order; a tag with no reason is left out."""
    grouped = {tag: [] for tag in _TAG_ORDER}
    for reason in reasons:
        tag = next((t for t in _TAG_ORDER if isinstance(reason, str) and reason.startswith(t)), None)
        if tag is None:
            raise ValueError(f'unknown evidence reason: {reason!r}')
        if reason not in grouped[tag]:
            grouped[tag].append(reason)
    return [(tag, '\n'.join(found)) for tag, found in grouped.items() if found]
# [AGENT EDIT END]


@dataclass(frozen=True)
class MatchingInput:
    units: pd.DataFrame
    group: str
    candidate_pairs: pd.DataFrame | None = None
    features: dict | None = None
    waveform_halves: np.ndarray | None = None
    waveform_cellids: tuple = ()
    waveform_session_ids: tuple = ()
    waveform_spike_counts: np.ndarray | None = None
    geometry: dict | None = None
    spike_times: dict | None = None
    fingerprint: np.ndarray | None = None
    fingerprint_bin_keys: tuple = ()
    timing: dict | None = None
    # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no fitted_model field (Task 2) | date: 2026-09-27]
    checkpoint: CheckpointRef | None = None
    waveform_provenance: dict = field(default_factory=dict)
    provenance: Provenance = field(default_factory=Provenance)

    def available(self) -> frozenset:
        have = {Input.UNITS}
        checks = {
            Input.CANDIDATE_PAIRS: self.candidate_pairs is not None,
            Input.FEATURES: bool(self.features),
            Input.WAVEFORM_HALVES: self.waveform_halves is not None and np.size(self.waveform_halves) > 0,
            Input.GEOMETRY: bool(self.geometry),
            Input.SPIKE_TIMES: bool(self.spike_times),
            Input.FINGERPRINT: self.fingerprint is not None and np.size(self.fingerprint) > 0,
            Input.TIMING: bool(self.timing),
            Input.CHECKPOINT: self.checkpoint is not None,
        }
        have.update(k for k, v in checks.items() if v)
        return frozenset(have)

    def validate(self) -> 'MatchingInput':
        if not isinstance(self.group, str) or not self.group:
            raise ValueError('group must be a nonempty string')
        leaked = LABEL_COLUMNS & set(self.units.columns)
        if leaked:
            raise ValueError(f'Reference label columns do not belong in matcher inputs: {sorted(leaked)}')
        eligible, _ = single_units(self.units)
        ids = set(eligible.cellid)
        if self.candidate_pairs is not None:
            if not {'cellid_a', 'cellid_b'} <= set(self.candidate_pairs.columns) and len(self.candidate_pairs):
                raise ValueError('candidate_pairs requires cellid_a and cellid_b')
            if len(self.candidate_pairs):
                ends = set(self.candidate_pairs.cellid_a.astype(str)) | set(self.candidate_pairs.cellid_b.astype(str))
                if not ends <= set(self.units.cellid.astype(str)):
                    raise ValueError('Pair evidence references unknown unit IDs')
        if self.waveform_halves is not None:
            w = np.asarray(self.waveform_halves)
            if w.ndim != 4 or w.shape[3] != 2 or w.shape[0] != len(self.waveform_cellids):
                raise ValueError('waveform halves require shape (unit, sample, channel, 2) aligned to waveform_cellids')
            if not np.isfinite(w).all():
                raise ValueError('waveform halves must be finite; a missing half cannot be duplicated')
            if len(self.waveform_session_ids) != len(self.waveform_cellids):
                raise ValueError('one waveform session id per waveform cellid')
            missing = ids - set(self.waveform_cellids)
            if missing:
                raise ValueError(f'eligible units without waveforms: {sorted(missing)}')
            if not self.geometry:
                raise ValueError('waveforms require channel geometry per session')
            for sess in set(self.waveform_session_ids):
                g = np.asarray(self.geometry.get(sess, np.empty((0, 0))), dtype=float)
                if g.shape != (w.shape[2], 2) or not np.isfinite(g).all():
                    raise ValueError(f'geometry for session {sess!r} must be finite (channel, 2) matching the waveforms')
            if self.spike_times is not None:
                missing = set(self.waveform_cellids) - set(self.spike_times)
                if missing:
                    raise ValueError(f'spike times missing for units: {sorted(missing)}')
            if self.waveform_spike_counts is not None:
                c = np.asarray(self.waveform_spike_counts)
                if c.shape != (len(self.waveform_cellids), 2):
                    raise ValueError('waveform_spike_counts must be (unit, 2)')
        if self.fingerprint is not None:
            f = np.asarray(self.fingerprint, dtype=float)
            keys = tuple(self.fingerprint_bin_keys)
            if f.ndim != 2 or f.shape[0] != len(self.waveform_cellids) or f.shape[1] != len(keys) or len(keys) < 2:
                raise ValueError('fingerprint must be (unit, bin) aligned to waveform_cellids with at least two bins')
            if len(set(keys)) != len(keys):
                raise ValueError('fingerprint bin keys must be unique')
            if not np.isfinite(f).all():
                raise ValueError('fingerprint values must be finite')
        if self.checkpoint is not None and not isinstance(self.checkpoint, CheckpointRef):
            raise ValueError('checkpoint must be a CheckpointRef')
        return self

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no fitted_model parameter (Task 2) | date: 2026-09-27]
    @classmethod
    def from_bundle(cls, units, bundle, *, group, pairs=None, features=None, timing=None,
                    checkpoint=None, provenance=None):
        # [AGENT EDIT END]
        if not isinstance(bundle, RawWaveformBundle):
            raise ValueError('from_bundle requires a RawWaveformBundle')
        ids = tuple(map(str, bundle.cellids))
        spikes = ({i: np.asarray(t, dtype=float) for i, t in zip(ids, bundle.spike_times_seconds)}
                  if len(bundle.spike_times_seconds) else None)
        return cls(units=units, group=str(group), candidate_pairs=pairs, features=features,
                   waveform_halves=np.asarray(bundle.waveforms, dtype=float), waveform_cellids=ids,
                   waveform_session_ids=tuple(map(str, bundle.session_ids)),
                   waveform_spike_counts=bundle.waveform_spike_counts,
                   geometry={str(k): np.asarray(v, dtype=float) for k, v in bundle.channel_positions.items()},
                   spike_times=spikes, fingerprint=bundle.peth, fingerprint_bin_keys=tuple(bundle.peth_bin_keys),
                   timing=timing, checkpoint=checkpoint,  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no fitted_model (Task 2) | date: 2026-09-27]
                   waveform_provenance=dict(bundle.provenance), provenance=provenance or Provenance())

    def bundle(self) -> RawWaveformBundle:
        if self.waveform_halves is None:
            raise ValueError('this input carries no waveforms')
        return RawWaveformBundle(self.waveform_cellids, self.waveform_halves, self.waveform_session_ids,
                                 self.geometry, tuple(self.spike_times[i] for i in self.waveform_cellids) if self.spike_times else (),
                                 dict(self.waveform_provenance), peth=self.fingerprint,
                                 peth_bin_keys=tuple(self.fingerprint_bin_keys),
                                 waveform_spike_counts=self.waveform_spike_counts)
# [AGENT EDIT END]
