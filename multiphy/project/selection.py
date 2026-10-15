# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one selection record of extracted project days feeds every project run (Plan B Task 1) | date: 2026-09-28]
"""A selection of extracted project days: the record a project run keeps, and the dataset it matches.

``Selection(project, days)`` names one day (``same_day``: the day's two temporal halves are
matched against each other) or two or more days (``across_days``: each day's full-day units are
matched across days). Its ``record()`` is written once as a run's ``selection.json``; its bytes
and their sha256 identify the run's input, and ``selection_of_run`` refuses a run whose project
changed since. ``dataset()`` reads each day's hash-verified features: the halves archive for one
day, the full-day rows for several. A day without extracted features, and a templates-tier day
(no waveform halves, so no geometry for any backend), are refused. No module here imports a lab
module.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from ..benchmark import BenchmarkDataset, fingerprint_matrix, load_dataset
from ..benchmark_pairs import build_benchmark_pairs
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: selection_inputs names Input; days read their entry alone (launcher-gui Task 1) | date: 2026-10-07]
from ..contract.inputs import Input, RawWaveformBundle, TrialTiming
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a selection decides its evidence through run_evidence (event-optional Task 4) | date: 2026-10-09]
from ..contract.inputs import RunEvidence, run_evidence
# [AGENT EDIT END]
from ..extract.full_day import load_full_day
from ..sources.normalize import read_source_entry
# [AGENT EDIT END]
from .folder import MANIFEST_FILE, RUNS_DIR, Project

SCHEMA_VERSION = 1
KIND = 'multiphy_selection'
SELECTION_FILE = 'selection.json'
FEATURES_DIR = 'features'
SOURCE_DIR = 'source'
MIN_REPEATS = 4
HALVES_SESSIONS = ('recording0000_part000', 'recording0000_part001')  # the halves archive's session ids
SPLIT_METHOD = ('one full-day observation per unit and day; waveform halves are the day\'s two temporal halves, '
                'averaged over disjoint spike selections')


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class AcrossDaysDataset:
    """Full-day units of two or more days of one project, with the pairs that join different days."""
    units: pd.DataFrame
    pairs: pd.DataFrame
    features: dict
    geometry: dict
    metadata: dict

    def subset(self, cellids: list) -> AcrossDaysDataset:
        """The units ``cellids``, the pairs among them and the geometry of their days."""
        wanted = set(cellids)
        if not wanted or not wanted <= set(self.units.cellid):
            raise ValueError('subset needs known unit ids')
        units = self.units[self.units.cellid.isin(wanted)].reset_index(drop=True)
        pairs = self.pairs[self.pairs.cellid_a.isin(wanted) & self.pairs.cellid_b.isin(wanted)].reset_index(drop=True)
        sessions = set(units.session_id)
        return AcrossDaysDataset(units, pairs, {c: self.features[c] for c in units.cellid},
                                 {s: g for s, g in self.geometry.items() if s in sessions}, dict(self.metadata))

    def waveform_bundle(self) -> RawWaveformBundle:
        """The matcher inputs: waveform halves, spike times, geometry and shared-condition fingerprints."""
        ids = tuple(self.units.cellid)
        rows = [self.features[c] for c in ids]
        peth, keys = fingerprint_matrix(rows)
        provenance = dict(self.metadata, split_method=SPLIT_METHOD,
                          source_groups=dict(zip(self.units.session_id, self.units.well)))
        if not self.metadata.get('probe_type'):
            provenance.pop('probe_type', None)
        rates = {float(r['sample_rate']) for r in rows}
        if len(rates) == 1:
            provenance['sample_rate'] = next(iter(rates))
        return RawWaveformBundle(
            ids, np.stack([r['waveform_halves'] for r in rows]), tuple(self.units.session_id), self.geometry,
            tuple((np.asarray(r['spike_samples']) - r['start_sample']) / r['sample_rate'] for r in rows),
            provenance, peth=peth, peth_bin_keys=keys,
            waveform_spike_counts=np.array([[len(h) for h in r['half_waveform_spike_samples']] for r in rows]))


@dataclass(frozen=True)
class Selection:
    """One day, or two or more distinct days, of ``project``; ``days`` is sorted by ``(date, day)``."""
    project: Project
    days: tuple

    def __post_init__(self):
        days = tuple(self.days)
        if not days or len(set(days)) != len(days):
            raise ValueError('choose one day, or two or more distinct days')
        manifest = self.project.read()
        records = manifest['days']
        for day in days:
            if day not in records:
                raise ValueError(f'no day {day} in project {self.project.name}')
        days = tuple(sorted(days, key=lambda day: (records[day]['date'], day)))
        for day in days:
            if records[day]['features']['full'] is None:
                raise ValueError(f'day {day} has no features; run extract first')
        for day in days:
            if records[day]['tier'] == 'templates':
                raise ValueError(f'day {day} is templates tier: no waveform halves, so no geometry for any backend; '
                                 'add the day with its raw file to a new project')
        object.__setattr__(self, 'days', days)

    @property
    def same_day(self) -> bool:
        return len(self.days) == 1

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the selection names its mode, which the record keeps and the viewer opens (viewer-split Task 2) | date: 2026-10-06]
    @property
    def mode(self) -> str:
        return 'same_day' if self.same_day else 'across_days'
    # [AGENT EDIT END]

    @property
    def dataset_id(self) -> str:
        return f'{self.project.name}/' + '+'.join(self.days)

    @property
    def _level(self) -> str:
        return 'halves' if self.same_day else 'full'

    def runs_dir(self) -> Path:
        """The day's ``runs/`` for one day, else the project's ``runs/``."""
        if self.same_day:
            return self.project.day_dir(self.days[0]) / RUNS_DIR
        return self.project.path / RUNS_DIR

    def _features_dir(self, records: dict, day: str) -> Path:
        """The day's ``features/halves`` or ``features/full``; a manifest that is not the recorded id is refused."""
        path = self.project.day_dir(day) / FEATURES_DIR / self._level
        if _sha256_file(path / 'manifest.json') != records[day]['features'][self._level]:
            raise ValueError(f'day {day} features/{self._level} no longer match the id recorded in project '
                             f'{self.project.name}')
        return path

    def _halves(self, records: dict) -> BenchmarkDataset:
        return load_dataset(self._features_dir(records, self.days[0]))

    def _full_days(self, records: dict) -> dict:
        return {day: load_full_day(self._features_dir(records, day)) for day in self.days}

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the record, its hash and the dataset come from one manifest read and one load of the features; same-day timing keys are the two halves ids (Plan B Task 1 fix round 1) | date: 2026-09-28]
    def _head(self, manifest: dict) -> dict:
        """The record without ``excluded``: what ``manifest`` alone says about this selection."""
        records, identity = manifest['days'], manifest['identity']
        days = [{'day': day, 'date': records[day]['date'], 'features': self._level,
                 'features_id': records[day]['features'][self._level]} for day in self.days]
        return {'schema_version': SCHEMA_VERSION, 'kind': KIND, 'project': self.project.name,
                'group': identity['group'], 'probe': identity['probe'],
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the record reads the selection's mode (viewer-split Task 2) | date: 2026-10-06]
                'mode': self.mode, 'days': days}
                # [AGENT EDIT END]

    def _load(self, records: dict):
        """The halves archive (one day) or ``{day: full-day features}`` (several), hash-verified against ``records``."""
        return self._halves(records) if self.same_day else self._full_days(records)

    def _record(self, manifest: dict, loaded) -> dict:
        """The record of ``manifest`` with the excluded units of ``loaded`` (the features read under it)."""
        if self.same_day:
            excluded = [{'cellid': str(e['source_unit_id']), 'day': self.days[0],
                         'reason': f"part {e['split_index']}: {e['reason']}"}
                        for e in loaded.exclusions]
        else:
            excluded = [{'cellid': row.cellid, 'day': day, 'reason': row.exclusion_reason}
                        for day, full in loaded.items()
                        for row in full['table'].itertuples() if not row.eligible]
        return dict(self._head(manifest), excluded=excluded)

    @staticmethod
    def _bytes(record: dict) -> bytes:
        return (json.dumps(record, indent=2, sort_keys=True) + '\n').encode()

    def record(self) -> dict:
        """The selection as ``selection.json`` holds it: project identity, days, feature ids and excluded units."""
        manifest = self.project.read()
        return self._record(manifest, self._load(manifest['days']))

    def record_bytes(self) -> bytes:
        return self._bytes(self.record())

    def sha256(self) -> str:
        return hashlib.sha256(self.record_bytes()).hexdigest()

    def dataset(self) -> Union[BenchmarkDataset, AcrossDaysDataset]:
        """One day: its halves archive. Several days: their eligible full-day units and the pairs across days.

        The metadata and ``input_manifest_sha256`` come from one read of project.json and the features
        loaded under it, so a project changed meanwhile cannot give the hash of another selection.
        """
        manifest = self.project.read()
        records, identity = manifest['days'], manifest['identity']
        loaded = self._load(records)
        digest = hashlib.sha256(self._bytes(self._record(manifest, loaded))).hexdigest()
        if self.same_day:
            dataset = loaded
            dataset.metadata.update(input_path=str(self.project.path), input_manifest_sha256=digest,
                                    selection='same_day')
            if identity['probe']:
                for source in dataset.metadata['sources']:
                    source.setdefault('provenance', {})['probe_type'] = identity['probe']
            return dataset
        units, features, geometry = [], {}, {}
        for day, full in loaded.items():
            table = full['table']
            units.append(table[table['eligible']].drop(columns=['eligible', 'exclusion_reason']))
            features.update(full['features'])
            geometry.update(full['geometry'])
        units = pd.concat(units, ignore_index=True)
        pairs = build_benchmark_pairs(units, features, geometry, min_repeats=MIN_REPEATS)
        metadata = {'dataset_id': self.dataset_id, 'project': self.project.name, 'days': list(self.days),
                    'session_dates': {day: records[day]['date'] for day in self.days},
                    'features_ids': {day: records[day]['features']['full'] for day in self.days},
                    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the upstream adapters refuse waveforms without raw_source; each day's source_id is the sha256 of its source/extraction_manifest.json, which names its raw files (e2e fix) | date: 2026-10-11]
                    'raw_source': {day: records[day]['source_id'] for day in self.days},
                    # [AGENT EDIT END]
                    'probe_type': identity['probe'], 'pair_options': {'min_repeats': MIN_REPEATS},
                    'input_path': str(self.project.path), 'input_manifest_sha256': digest,
                    'selection': 'across_days'}
        return AcrossDaysDataset(units, pairs, features, geometry, metadata)

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day's source entry read from its manifest alone; its timing built from that entry (launcher-gui Task 1) | date: 2026-10-07]
    def _day_entry(self, day: str) -> dict:
        return read_source_entry(self.project.day_dir(day) / SOURCE_DIR)

    def _day_timing(self, day: str) -> TrialTiming:
        return TrialTiming.from_source_entry(self._day_entry(day))
# [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day without trial events has no timing and is skipped; the evidence is decided from the timed days (event-optional Task 4) | date: 2026-10-09]
    def _has_trials(self, day: str) -> bool:
        """True when the day's source entry holds trial events."""
        trials = self._day_entry(day).get('trials')
        return trials is not None and len(trials) > 0

    def timing(self) -> dict:
        """Each day's trial timing keyed by day, for the days with trial events only; for one day, keyed by
        both halves' session ids, or ``{}`` when the day has none."""
        if not self.same_day:
            return {day: self._day_timing(day) for day in self.days if self._has_trials(day)}
        if not self._has_trials(self.days[0]):
            return {}
        timing = self._day_timing(self.days[0])
        return {session: timing for session in HALVES_SESSIONS}

    def evidence(self, min_repeats: int = 5) -> RunEvidence:
        """The evidence these days measured (``run_evidence``): one day's two halves, or the days."""
        return run_evidence(HALVES_SESSIONS if self.same_day else self.days, self.timing(), min_repeats=min_repeats)
    # [AGENT EDIT END]
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each day's full-day extras (tone tuning) for the viewers, hash-checked like the features (Plan C Task 5) | date: 2026-10-01]
    def day_extras(self) -> dict:
        """``{day: extras}`` of each day, read from its ``features/full`` once its manifest is the recorded id."""
        records = self.project.read()['days']
        out = {}
        for day in self.days:
            path = self.project.day_dir(day) / FEATURES_DIR / 'full'
            if _sha256_file(path / 'manifest.json') != records[day]['features']['full']:
                raise ValueError(f'day {day} features/full no longer match the id recorded in project '
                                 f'{self.project.name}')
            out[day] = load_full_day(path)['extras']
        return out
    # [AGENT EDIT END]

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: conditions read each day's source entry, so a day without trial events has none (event-optional Task 4) | date: 2026-10-09]
    def conditions(self) -> dict:
        """``{day: frozenset of stimulus conditions}``, read from each day's source entry."""
        return {day: frozenset(s['stimulus'] for s in self._day_entry(day)['stimuli']) for day in self.days}
    # [AGENT EDIT END]


def default_days(project: Project) -> tuple:
    """The project's days with extracted features, sorted by ``(date, day)``."""
    records = project.read()['days']
    return tuple(sorted((day for day, record in records.items() if record['features']['full'] is not None),
                        key=lambda day: (records[day]['date'], day)))


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: predict a selection's inputs from its days' source manifests, without loading spikes (D5) (launcher-gui Task 1) | date: 2026-10-07]
def selection_inputs(project: Project, days: tuple) -> frozenset:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: timing when any day has trial events; fingerprint when every day shares a condition (event-optional Task 4) | date: 2026-10-09]
    """The inputs a selection of ``days`` gives a backend, predicted from each day's source manifest alone.

    Predicts ``build_input(...).available()`` minus ``checkpoint``, which a backend option
    supplies. ``timing`` is predicted when any chosen day's source has trial events (the timing
    holds those days only), and ``fingerprint`` when every day's stimulus conditions share at
    least one. A selection whose shared conditions give fewer than two aligned bins, or whose
    waveform bundle cannot be built, is predicted to have those inputs, and a backend that
    requires one of them fails at Run with ``MissingInputs`` before its run folder exists.
    ``Selection``'s refusals raise unchanged.
    """
    selection = Selection(project, tuple(days))
    inputs = {Input.UNITS, Input.CANDIDATE_PAIRS, Input.FEATURES, Input.WAVEFORM_HALVES, Input.GEOMETRY,
              Input.SPIKE_TIMES}
    if any(selection._has_trials(day) for day in selection.days):
        inputs.add(Input.TIMING)
    if frozenset.intersection(*selection.conditions().values()):
        inputs.add(Input.FINGERPRINT)
    return frozenset(inputs)
    # [AGENT EDIT END]
# [AGENT EDIT END]


def selection_of_run(run_dir: Path) -> Selection:
    """The selection of the project run in ``run_dir``; refused when the project no longer gives the same record."""
    run_dir = Path(run_dir).resolve()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the missing-selection refusal comes before the project lookup, so a folder outside any project gets it too (Plan B Task 1 fix round 2) | date: 2026-09-28]
    path = run_dir / SELECTION_FILE
    if not path.is_file():
        raise FileNotFoundError(f'{run_dir} has no selection.json; it is not a project run')
    project_dir = next((folder for folder in (run_dir, *run_dir.parents) if (folder / MANIFEST_FILE).is_file()), None)
    if project_dir is None:
        raise FileNotFoundError(f'{run_dir} is not inside a project folder; it is not a project run')
    project = Project.open(project_dir.parent, project_dir.name)
    # [AGENT EDIT END]
    saved = path.read_bytes()
    saved_record = json.loads(saved)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a removed day or a cleared features id gets the run-mismatch refusal, checked before Selection's own refusals can fire (Plan B Task 1 review 3) | date: 2026-09-28]
    changed = ValueError(f'{run_dir}/selection.json no longer matches project {project.name}; '
                         'the project changed after this run')
    manifest = project.read()
    for saved_day in saved_record['days']:
        record = manifest['days'].get(saved_day['day'])
        if (record is None or record['features']['full'] is None
                or record['features'][saved_day['features']] != saved_day['features_id']):
            raise changed
    selection = Selection(project, tuple(day['day'] for day in saved_record['days']))
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: compare identity, days and feature ids before loading features, so a changed features id gets the run-mismatch refusal (Plan B Task 1 fix round 1) | date: 2026-09-28]
    if {k: v for k, v in saved_record.items() if k != 'excluded'} != selection._head(manifest):
        raise changed
    if selection._bytes(selection._record(manifest, selection._load(manifest['days']))) != saved:
        raise changed
    # [AGENT EDIT END]
    return selection
# [AGENT EDIT END]
