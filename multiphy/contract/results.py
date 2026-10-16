"""What every backend returns.

A MatchResult partitions the eligible single units into tracks and unmatched units,
carries the pair evidence it used, and says explicitly what kind of score it emitted
(none, uncalibrated, calibrated probability) and what kind of tracking produced the
tracks (native rule or the common constrained tracker).
``validate()`` enforces every structural rule; ``validate(spec)`` additionally checks the
emissions against what the backend's spec declares it can provide.
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd

from .provenance import Provenance, json_safe, payload_digest


class ScoreKind(str, Enum):
    NONE = 'none'
    UNCALIBRATED = 'uncalibrated_score'
    CALIBRATED_PROBABILITY = 'calibrated_probability'


class TrackKind(str, Enum):
    NATIVE = 'native'
    COMMON = 'common'


class Decision(str, Enum):
    SELECTED = 'selected'
    SUPPORTING_TRACK = 'supporting_track'
    BELOW_THRESHOLD = 'below_threshold'
    MISSING_EVIDENCE = 'missing_evidence'
    AMBIGUOUS_COMPETITION = 'ambiguous_competition'
    SESSION_COLLISION = 'session_collision'
    MEASURED_CONTRADICTION = 'measured_contradiction'
    SAME_SESSION = 'same_session'
    NATIVE_TRACK = 'native_track'
    UNMATCHED_BY_NATIVE_TRACKING = 'unmatched_by_native_tracking'
    NATIVE_CLUSTER = 'native_cluster'
    UNMATCHED_BY_NATIVE_CLUSTERING = 'unmatched_by_native_clustering'
    SCORED = 'scored'
    SCORED_NOT_ASSIGNED = 'scored_not_assigned'
    NOT_RUN = 'not_run'
    # A pair inside one native group that the one-unit-per-session rule separated.
    PROPOSED_LINK = 'proposed_link'


DECISIONS = frozenset(d.value for d in Decision)
REQUIRED_PAIR_COLUMNS = frozenset({'cellid_a', 'cellid_b', 'match_score', 'decision', 'selected'})


@dataclass(frozen=True)
class ScoreBlock:
    kind: ScoreKind
    column: str | None
    similarity_column: str | None = None
    calibration_reference: str | None = None

    def to_json(self):
        return {'kind': self.kind.value, 'column': self.column,
                'similarity_column': self.similarity_column, 'calibration_reference': self.calibration_reference}

    @classmethod
    def from_json(cls, d):
        return cls(ScoreKind(d['kind']), d.get('column'), d.get('similarity_column'), d.get('calibration_reference'))


def single_units(units):
    """Filter explicit SU labels; missing or unknown quality never implies SU.

    Session IDs prefer an explicit ``session_id``, then ``siteid``, then the cell-ID prefix.
    """
    units = units.copy()
    if 'cellid' not in units:
        raise ValueError('units requires cellid')
    if units.cellid.isna().any() or (units.cellid.astype(str).str.strip() == '').any():
        raise ValueError('cellid must not be empty')
    units['cellid'] = units.cellid.astype(str)
    if units.cellid.duplicated().any():
        raise ValueError('cellid must uniquely identify each unit')
    quality = units.get('su_type', pd.Series('', index=units.index))
    keep = quality.fillna('').astype(str).str.strip().str.upper().eq('SU')
    excluded = tuple(sorted(units.loc[~keep, 'cellid']))
    units = units.loc[keep].copy()
    session = units.get('session_id', pd.Series(index=units.index, dtype=object))
    session = session.where(session.notna() & session.astype(str).str.strip().ne(''))
    if 'siteid' in units:
        site = units.siteid.where(units.siteid.notna() & units.siteid.astype(str).str.strip().ne(''))
        session = session.fillna(site)
    units['session_id'] = session.fillna(units.cellid.str.split('-').str[0]).astype(str)
    return units.reset_index(drop=True), excluded


@dataclass
class MatchResult:
    backend_id: str
    version: str
    run_id: str
    units: pd.DataFrame
    pairs: pd.DataFrame
    tracks: tuple
    unmatched: tuple
    excluded_units: tuple
    score: ScoreBlock
    track_kind: TrackKind
    tracking_rule: str
    provenance: Provenance

    def validate(self, spec=None):
        if not all(isinstance(x, str) and x for x in (self.backend_id, self.version, self.run_id, self.tracking_rule)):
            raise ValueError('backend_id, version, run_id and tracking_rule must be nonempty strings')
        if not isinstance(self.provenance, Provenance):
            raise ValueError('provenance must be a Provenance')
        if not isinstance(self.score, ScoreBlock) or not isinstance(self.track_kind, TrackKind):
            raise ValueError('score must be a ScoreBlock and track_kind a TrackKind')
        units, _ = single_units(self.units)
        if len(units) != len(self.units):
            raise ValueError('Result units must all have explicit SU quality')
        if 'session_id' not in self.units or any(not isinstance(v, str) or not v.strip()
                                                  for col in ('cellid', 'session_id') for v in self.units[col]):
            raise ValueError('Result units require explicit string cellid and session_id values')
        eligible = set(units.cellid)
        members = [u for t in self.tracks for u in t] + list(self.unmatched)
        if any(len(t) < 2 for t in self.tracks):
            raise ValueError('Singleton tracks belong in unmatched')
        if len(members) != len(set(members)) or set(members) != eligible:
            raise ValueError('Tracks and unmatched must partition eligible single units')
        if set(self.excluded_units) & eligible:
            raise ValueError('Excluded units cannot also be eligible')
        if not REQUIRED_PAIR_COLUMNS <= set(self.pairs.columns):
            raise ValueError(f'Result pairs missing columns: {sorted(REQUIRED_PAIR_COLUMNS - set(self.pairs.columns))}')
        if not set(self.pairs.cellid_a).union(self.pairs.cellid_b) <= eligible:
            raise ValueError('Pair endpoints must be eligible single units')
        seen = set()
        for r in self.pairs.itertuples():
            key = tuple(sorted((r.cellid_a, r.cellid_b)))
            if key in seen or r.cellid_a == r.cellid_b:
                raise ValueError('Pair evidence must be unique and cannot compare a unit to itself')
            seen.add(key)
            if r.decision not in DECISIONS:
                raise ValueError(f'Unknown decision {r.decision!r}')
            if not isinstance(r.selected, (bool, np.bool_)):
                raise ValueError('Pair selected flags must be Boolean')
            if pd.notna(r.match_score) and (not np.isfinite(r.match_score) or not 0 <= r.match_score <= 1):
                raise ValueError('match_score must be missing or finite in [0, 1]')
        scores = pd.to_numeric(self.pairs.match_score, errors='coerce') if len(self.pairs) else pd.Series(dtype=float)
        if self.score.kind is ScoreKind.NONE:
            if self.score.column is not None or scores.notna().any():
                raise ValueError('score kind none requires column None and all-missing match_score')
        else:
            if self.score.column is None or (len(self.pairs) and self.score.column not in self.pairs):
                raise ValueError('score column must name a pairs column')
        if self.score.kind is ScoreKind.CALIBRATED_PROBABILITY and not self.score.calibration_reference:
            raise ValueError('Calibrated probabilities require a calibration_reference')
        if self.score.similarity_column and len(self.pairs) and self.score.similarity_column not in self.pairs:
            raise ValueError('similarity column must name a pairs column')
        sessions = units.set_index('cellid').session_id.to_dict()
        if any(len({sessions[u] for u in t}) != len(t) for t in self.tracks):
            raise ValueError('Result contains a same-session track collision')
        if spec is not None:
            if self.backend_id != spec.id:
                raise ValueError(f'Result backend_id {self.backend_id!r} does not match spec {spec.id!r}')
            if self.score.kind not in spec.provides.score_kinds:
                raise ValueError(f'score kind {self.score.kind.value} not declared in spec.provides')
            if self.track_kind not in spec.provides.track_kinds:
                raise ValueError(f'track_kind {self.track_kind.value} not declared in spec.provides')
            if self.score.similarity_column is not None and not spec.provides.similarity:
                raise ValueError('similarity column emitted but spec.provides.similarity is False')
        return self

    def payload(self):
        out = {'backend_id': self.backend_id, 'version': self.version, 'run_id': self.run_id,
               'tracks': [list(t) for t in self.tracks], 'unmatched': list(self.unmatched),
               'excluded_units': list(self.excluded_units), 'score': self.score.to_json(),
               'track_kind': self.track_kind.value, 'tracking_rule': self.tracking_rule,
               'provenance': self.provenance.to_json()}
        for key in ('units', 'pairs'):
            table = getattr(self, key)
            if not table.columns.is_unique or not all(isinstance(c, str) for c in table.columns):
                raise ValueError('Result table columns must be unique strings')
            out[key] = {'columns': list(table.columns), 'records': table.to_dict(orient='records')}
        return json_safe(out)

    def digest(self):
        self.validate()
        return payload_digest(self.payload())
