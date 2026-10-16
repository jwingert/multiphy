# [AGENT EDIT START | agent: codex | user: wingertj | reason: assemble and persist temporal-split identity benchmarks outside the code repository | date: 2026-09-17]
"""Temporal benchmark datasets with reference labels kept out of matcher inputs.

The voltage reader and all input times must already share a verified raw sample
clock. This module does not guess a mapping from trial-relative spike times.
Generated datasets belong beneath paths.CACHE_ROOT, never beside this source.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: SourceRecording.partition may be None (multiphy project folder Task 2) | date: 2026-09-25]
from typing import Optional
# [AGENT EDIT END]

import numpy as np
import pandas as pd

from .evaluation import validate_partitions
from .external_backends import RawWaveformBundle
from .paths import resolve_cache_path


@dataclass
class SourceRecording:
    """One original sort with sample-clock stimuli and measured probe geometry."""
    source_session: str
    well: str
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: halves archives carry no partition; None on every source leaves it out (multiphy project folder Task 2) | date: 2026-09-25]
    partition: Optional[str]
    # [AGENT EDIT END]
    sample_rate: float
    channel_positions: np.ndarray
    windows: np.ndarray
    spike_samples: dict
    single_unit_ids: tuple
    read_samples: object
    preprocessing: str
    provenance: dict
    stimuli: object = None
    baseline_windows: object = None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fingerprint block of BenchmarkDataset.waveform_bundle, moved verbatim so project selections share it (Plan B Task 1) | date: 2026-09-28]
def fingerprint_matrix(rows: list) -> tuple:
    """``(peth, keys)``: each row's fingerprints over the conditions every row shares; ``peth`` is ``None`` without keys."""
    common = set.intersection(*(set(r.get('fingerprints', {})) for r in rows))
    keys, values = [], [[] for _ in rows]
    for stimulus in sorted(common):
        shapes = {np.asarray(r['fingerprints'][stimulus]).shape for r in rows}
        bins = {r['fingerprint_bin_seconds'] for r in rows}
        if len(shapes) != 1 or len(bins) != 1:
            raise ValueError('Stimulus fingerprint bins must be aligned across pseudo-sessions.')
        for k in range(next(iter(shapes))[0]):
            keys.append(json.dumps([stimulus, k, next(iter(bins))]))
        for out, r in zip(values, rows):
            out.extend(r['fingerprints'][stimulus])
    peth = np.asarray(values, dtype=float) if keys else None
    return peth, tuple(keys)
# [AGENT EDIT END]


@dataclass
class BenchmarkDataset:
    units: pd.DataFrame
    pairs: pd.DataFrame
    features: dict
    labels: pd.DataFrame
    geometry: dict
    metadata: dict
    exclusions: list = field(default_factory=list)

    def validate(self):
        required = {'cellid', 'session_id', 'siteid', 'su_type', 'recording_group'}
        if not required <= set(self.units):
            raise ValueError(f'Unit table requires {sorted(required)}')
        if self.units.empty or self.units.cellid.duplicated().any():
            raise ValueError('A benchmark requires unique eligible pseudo-units.')
        if self.units[list(required)].isna().any().any() or not self.units.su_type.eq('SU').all():
            raise ValueError('Benchmark unit metadata must be complete and single-unit only.')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: partition is an optional label column (multiphy project folder Task 2) | date: 2026-09-25]
        if not {'cellid', 'source_unit_id', 'source_session', 'well'} <= set(self.labels):
            raise ValueError('Separate reference labels are required for every pseudo-unit.')
        partitioned = 'partition' in self.labels
        # [AGENT EDIT END]
        ids = set(self.units.cellid)
        if ids != set(self.features) or ids != set(self.labels.cellid) or self.labels.cellid.duplicated().any():
            raise ValueError('Features, units, and reference labels must describe identical unique pseudo-units.')
        if self.labels.source_unit_id.isna().any() or self.labels.source_unit_id.astype(str).str.strip().eq('').any():
            raise ValueError('Source cluster reference IDs must not be missing.')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: partition is an optional label column (multiphy project folder Task 2) | date: 2026-09-25]
        if partitioned:
            validate_partitions(self.labels)
        # [AGENT EDIT END]
        joined = self.units[['cellid', 'session_id', 'recording_group']].merge(
            self.labels[['cellid', 'well', 'source_session', 'source_unit_id']], on='cellid', validate='one_to_one')
        if (joined.groupby('recording_group')[['well', 'source_session']].nunique().gt(1).any().any()
                or joined.groupby(['well', 'source_session']).recording_group.nunique().gt(1).any()
                or joined.groupby('session_id')[['well', 'source_session']].nunique().gt(1).any().any()):
            raise ValueError('Recording groups and pseudo-sessions must preserve original recording identities.')
        if joined.duplicated(['well', 'source_session', 'session_id', 'source_unit_id']).any():
            raise ValueError('Each source unit has at most one observation per pseudo-session.')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: partition is an optional label column (multiphy project folder Task 2) | date: 2026-09-25]
        if partitioned and not set(self.labels.partition) <= {'train', 'calibration', 'test'}:
            raise ValueError('Partition labels must be train, calibration, or test.')
        # [AGENT EDIT END]
        if {'source_unit', 'source_unit_id', 'same_unit', 'partition'} & set(self.units):
            raise ValueError('Reference labels do not belong in matcher unit inputs.')
        if not {'cellid_a', 'cellid_b'} <= set(self.pairs):
            raise ValueError('Pair table requires both endpoint columns even when empty.')
        if not (set(self.pairs.cellid_a) | set(self.pairs.cellid_b)) <= ids:
            raise ValueError('Pair endpoints must be eligible pseudo-units.')
        for r in self.units.itertuples():
            f = self.features[r.cellid]
            if {'source_unit', 'source_unit_id', 'same_unit'} & set(f):
                raise ValueError('Source identity labels must not be serialized as matcher features.')
            g = np.asarray(self.geometry[r.session_id])
            w = np.asarray(f['waveform_halves'])
            if (g.ndim != 2 or g.shape[1] != 2 or not np.isfinite(g).all()
                    or w.ndim != 3 or w.shape[1:] != (len(g), 2) or not np.isfinite(w).all()):
                raise ValueError('Fresh waveform halves must agree with finite measured channel geometry.')
        return self

    def subset(self, cellids):
        wanted = set(cellids)
        if not wanted <= set(self.units.cellid):
            raise ValueError('Subset contains unknown pseudo-unit IDs.')
        units = self.units[self.units.cellid.isin(wanted)].copy()
        pairs = self.pairs[self.pairs.cellid_a.isin(wanted) & self.pairs.cellid_b.isin(wanted)].copy()
        sessions = set(units.session_id)
        return BenchmarkDataset(units, pairs, {k: v for k, v in self.features.items() if k in wanted},
            self.labels[self.labels.cellid.isin(wanted)].copy(),
            {k: v for k, v in self.geometry.items() if k in sessions},
            dict(self.metadata), list(self.exclusions)).validate()

    def waveform_bundle(self):
        """Create official-method inputs; reference source cluster IDs stay separate."""
        self.validate()
        ids = tuple(self.units.cellid)
        rows = [self.features[c] for c in ids]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fingerprint block moved to fingerprint_matrix, shared with project selections (Plan B Task 1) | date: 2026-09-28]
        peth, keys = fingerprint_matrix(rows)
        # [AGENT EDIT END]
        labels = self.labels.set_index('cellid')
        group_map = {}
        for r in self.units.itertuples():
            group = str(labels.loc[r.cellid, 'well'])
            if r.session_id in group_map and group_map[r.session_id] != group:
                raise ValueError('A pseudo-session cannot belong to different holdout wells.')
            group_map[r.session_id] = group
        provenance = dict(self.metadata)
        provenance.update(raw_source=self.metadata['sources'],
                          split_method='disjoint chronological sample windows; independent early/late waveform means',
                          source_groups=group_map)
        active_sources = set(map(tuple, self.labels[['well', 'source_session']].drop_duplicates().to_numpy()))
        source_info = {(s['well'], s['source_session']): s for s in self.metadata['sources']
                       if {'well', 'source_session'} <= set(s)}
        probe_types = {key: source_info.get(key, {}).get('provenance', {}).get('probe_type')
                       for key in active_sources}
        known_types = set(probe_types.values()) - {None, ''}
        if len(known_types) == 1 and all(probe_types.values()):
            provenance['probe_type'] = next(iter(known_types))
        else:
            provenance.pop('probe_type', None)
        provenance['source_probe_types'] = {json.dumps(key): value for key, value in probe_types.items()}
        # [AGENT EDIT START | agent: codex | user: wingertj | reason: retain acquisition rate for learned waveform compatibility | date: 2026-09-18]
        rates = {float(r['sample_rate']) for r in rows}
        provenance['session_sample_rates'] = {str(s): float(r['sample_rate'])
            for s, r in zip(self.units.session_id, rows)}
        if len(rates) == 1:
            provenance['sample_rate'] = next(iter(rates))
        else:
            provenance.pop('sample_rate', None)
        # [AGENT EDIT END]
        return RawWaveformBundle(ids, np.stack([r['waveform_halves'] for r in rows]),
            tuple(self.units.session_id), self.geometry,
            tuple((np.asarray(r['spike_samples']) - r['start_sample']) / r['sample_rate'] for r in rows),
            provenance, peth=peth, peth_bin_keys=tuple(keys),
            waveform_spike_counts=np.array([[len(h) for h in r['half_waveform_spike_samples']] for r in rows]))


def build_dataset(recordings, *, dataset_id, runclass='BNT', omitted=(),
                  extraction_options=None, pair_options=None):
    """Recompute evidence, create opaque pseudo-unit IDs, then form ordinary pairs.

    ``omitted`` contains (source_session, source_unit_id, split_index) tuples for
    prespecified missing observations. Omission is applied after extraction and
    recorded separately; no source identity is given to the matching code.
    """
    from .benchmark_pairs import build_benchmark_pairs
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: temporal_split_unit_features lives in extract/unit_features.py (multiphy project folder Task 11) | date: 2026-09-25]
    from .extract.unit_features import temporal_split_unit_features
    # [AGENT EDIT END]
    recordings = list(recordings)
    if not recordings:
        raise ValueError('Supply at least one original sorted recording.')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: all-None partitions build an archive without a partition; a mix is refused before extraction (multiphy project folder Task 2) | date: 2026-09-25]
    carries_partition = [source.partition is not None for source in recordings]
    if any(carries_partition) and not all(carries_partition):
        raise ValueError('Every source needs a partition, or none does.')
    partitioned = all(carries_partition)
    # [AGENT EDIT END]
    requested = set(tuple(o) for o in omitted)
    omitted_sessions = {o[0] for o in requested}
    for session in omitted_sessions:
        if sum(source.source_session == session for source in recordings) > 1:
            raise ValueError('Omissions require unambiguous source_session identifiers across wells.')
    seen_omissions = set()
    table, labels, features, geometry, exclusions, sources = [], [], {}, {}, [], []
    seen_sources = set()
    for recording_index, source in enumerate(recordings):
        key = (source.well, source.source_session)
        if key in seen_sources:
            raise ValueError('Original recordings cannot be duplicated across dataset inputs.')
        seen_sources.add(key)
        if not source.provenance or not source.provenance.get('raw_source'):
            raise ValueError('Every recording requires raw-source and extraction provenance.')
        g = np.asarray(source.channel_positions, dtype=float)
        if g.ndim != 2 or g.shape[1] != 2 or not np.isfinite(g).all():
            raise ValueError('Supply measured (channel, x/depth) coordinates.')
        source_ids = tuple(source.single_unit_ids)
        if any(not isinstance(c, str) or not c.strip() for c in source_ids):
            raise ValueError('Single-unit source IDs must be nonempty strings.')
        if len(source_ids) != len(set(source_ids)) or not set(source_ids) <= set(source.spike_samples):
            raise ValueError('Single-unit IDs must be unique and have sorted spike samples.')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: leave the partition key out when no source carries one (multiphy project folder Task 2) | date: 2026-09-25]
        entry = dict(source_session=source.source_session, well=source.well,
                     partition=source.partition, preprocessing=source.preprocessing,
                     provenance=source.provenance)
        if not partitioned:
            del entry['partition']
        sources.append(entry)
        # [AGENT EDIT END]
        # IDs vary independently across pseudo-sessions, so suffix equality is
        # not a source-identity hint available to an algorithm.
        serial = 0
        for source_unit in source_ids:
            parts = temporal_split_unit_features(source.spike_samples[source_unit], source.read_samples,
                source.windows, sample_rate=source.sample_rate, n_channels=len(g),
                source_unit=str(source_unit), source_session=source.source_session, su_type='SU',
                preprocessing=source.preprocessing, stimuli=source.stimuli,
                baseline_windows=source.baseline_windows, **(extraction_options or {}))
            for f in parts:
                split = f['split_index']
                observation = (source.source_session, source_unit, split)
                if observation in requested:
                    seen_omissions.add(observation)
                if not f['eligible'] or observation in requested:
                    exclusions.append(dict(source_session=source.source_session, source_unit_id=source_unit,
                        split_index=split, reason='prespecified omission' if observation in requested else f['exclusion_reason']))
                    continue
                session = f'recording{recording_index:04d}_part{split:03d}'
                opaque = hashlib.sha256(f'{dataset_id}:{recording_index}:{serial}'.encode()).hexdigest()[:20]
                cid = f'{session}-{opaque}'
                serial += 1
                geometry[session] = g.copy()
                table.append(dict(cellid=cid, session_id=session, siteid=session, su_type='SU',
                                  recording_group=json.dumps(key)))
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: leave the partition key out when no source carries one (multiphy project folder Task 2) | date: 2026-09-25]
                row = dict(cellid=cid, source_unit_id=str(source_unit),
                           source_session=source.source_session, well=source.well, partition=source.partition)
                if not partitioned:
                    del row['partition']
                labels.append(row)
                # [AGENT EDIT END]
                features[cid] = {k: v for k, v in f.items() if k not in ('source_unit', 'source_session')}
    if seen_omissions != requested:
        raise ValueError('An omission does not identify an extracted source observation.')
    if not table:
        raise ValueError('No temporal observations passed waveform/spike requirements.')
    units = pd.DataFrame(table)
    references = pd.DataFrame(labels)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no partition leakage to check when no source carries one (multiphy project folder Task 2) | date: 2026-09-25]
    if partitioned:
        validate_partitions(references)
    # [AGENT EDIT END]
    pairs = build_benchmark_pairs(units, features, geometry, **(pair_options or {}))
    now = datetime.now(timezone.utc).isoformat()
    metadata = dict(schema_version=1, dataset_id=dataset_id, block_id=dataset_id,
                    runclass=runclass, cache_built=now, sources=sources,
                    extraction_options=extraction_options or {}, pair_options=pair_options or {},
                    reference_kind='within_sort_source_cluster',
                    scope='Conditional source-sort identities; not established across-day ground truth')
    return BenchmarkDataset(units, pairs, features, references, geometry, metadata, exclusions).validate()


def _pack(value, arrays):
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError('Object arrays are not supported in benchmark archives.')
        key = f'array_{len(arrays):07d}'
        arrays[key] = value
        return {'__array__': key}
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return {'__float__': str(value)}
    if isinstance(value, dict):
        if set(value) & {'__array__', '__float__'}:
            raise ValueError('Reserved archive marker in input metadata.')
        return {str(k): _pack(v, arrays) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_pack(v, arrays) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f'Unsupported benchmark metadata type: {type(value).__name__}')


def _unpack(value, arrays):
    if isinstance(value, dict):
        if set(value) == {'__array__'}:
            return arrays[value['__array__']].copy()
        if set(value) == {'__float__'}:
            return float(value['__float__'])
        return {k: _unpack(v, arrays) for k, v in value.items()}
    if isinstance(value, list):
        return [_unpack(v, arrays) for v in value]
    return value


def save_dataset(dataset, path, *, cache_root=None):
    """Save a new cache directory atomically. Existing datasets are immutable."""
    dataset.validate()
    target = resolve_cache_path(path, root=cache_root)
    if target.exists():
        raise FileExistsError(f'Dataset already exists: {target}')
    target.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.benchmark-', dir=target.parent))
    try:
        dataset.units.to_csv(stage / 'units.csv', index=False)
        dataset.pairs.to_csv(stage / 'pairs.csv', index=False)
        dataset.labels.to_csv(stage / 'reference_labels.csv', index=False)
        arrays = {}
        tree = _pack(dict(features=dataset.features, geometry=dataset.geometry,
                          metadata=dataset.metadata, exclusions=dataset.exclusions), arrays)
        np.savez_compressed(stage / 'features.npz', **arrays)
        (stage / 'features.json').write_text(json.dumps(tree, allow_nan=False, indent=2))
        files = ('units.csv', 'pairs.csv', 'reference_labels.csv', 'features.npz', 'features.json')
        manifest = dict(schema_version=1, metadata=dataset.metadata,
                        sha256={name: hashlib.sha256((stage / name).read_bytes()).hexdigest() for name in files})
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        (stage / 'METHODS.md').write_text(_methods(target, dataset))
        # mkdir reserves the destination against concurrent writers; rename
        # replaces only our own empty directory on the same filesystem.
        target.mkdir()
        stage.replace(target)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return target


def load_dataset(path, _ancestors=()):
    """Read a hash-verified, non-pickle dataset without importing data loaders."""
    path = Path(path).resolve()
    # [AGENT EDIT START | agent: codex | user: wingertj | reason: hash-verified dataset collections without duplicated waveform archives | date: 2026-09-18]
    if path in _ancestors:
        raise ValueError('Cyclic benchmark collection references.')
    manifest = json.loads((path / 'manifest.json').read_text())
    if manifest.get('schema_version') == 2 and manifest.get('kind') == 'collection':
        from .benchmark_collection import load_collection
        return load_collection(path, manifest, _ancestors + (path,))
    # [AGENT EDIT END]
    required = {'units.csv', 'pairs.csv', 'reference_labels.csv', 'features.npz', 'features.json'}
    if manifest.get('schema_version') != 1 or set(manifest.get('sha256', {})) != required:
        raise ValueError('Unsupported or incomplete benchmark manifest.')
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((path / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Benchmark content hash mismatch: {name}')
    with np.load(path / 'features.npz', allow_pickle=False) as arrays:
        contents = _unpack(json.loads((path / 'features.json').read_text()), arrays)
    if contents['metadata'] != manifest['metadata']:
        raise ValueError('Benchmark metadata disagree with the manifest.')
    return BenchmarkDataset(pd.read_csv(path / 'units.csv', dtype=str),
        pd.read_csv(path / 'pairs.csv', dtype={'cellid_a': str, 'cellid_b': str}), contents['features'],
        pd.read_csv(path / 'reference_labels.csv', dtype=str), contents['geometry'],
        contents['metadata'], contents['exclusions']).validate()


def _methods(target, dataset):
    return f'''# Temporal single-unit identity benchmark — figure & methods notes

- **Output folder:** `{target}`
- **Artifacts:** `units.csv`, `pairs.csv`, `reference_labels.csv`, `features.npz`, `features.json`, `manifest.json`
- **Generated by:** `multiphy/benchmark.py:build_dataset()/save_dataset()` · **Env:** `{sys.executable}` · **Updated:** `{datetime.now(timezone.utc).date()}`

---

## Goal

Measure recovery of the original single-unit cluster identities from independent temporal observations of one sort. Compare stimulus-response fingerprints and electrophysiological evidence while allowing unmatched units. These reference labels describe within-sort identity, not verified identity across recording days.

## Data lineage

- `features.npz` and `features.json` ← `features.py:temporal_split_unit_features()`; source voltage, sample-clock mapping, preprocessing, and source partitions are recorded in `manifest.json`. Waveforms are recomputed from bounded voltage snippets; full split spike trains determine firing rates, interval histograms, and autocorrelograms. Complete stimulus trials determine condition-aligned mean fingerprints and chronological trial-half means.
- For sources declaring `shared_unwhitened_v1`, `extract_benchmark.py:FilteredSnippetReader()` applies acquisition scaling and `waveform_preprocessing.py:SharedWaveformPreprocessor.transform()` to buffered one-second blocks before cropping. The default is a 150-Hz third-order Butterworth forward/backward high-pass with 50-ms context, followed by a samplewise median over declared reference channels. Physical recording endpoints use odd extension. No whitening, snippet baseline subtraction, or amplitude normalization is applied. Filter context may extend across a temporal split boundary; selected spike snippets cannot. Exact parameters and reference channels are stored in source provenance. Only mean waveforms, standard errors, internal-half means, and selected spike indices persist; temporary processed snippets are removed.
- `units.csv` ← `benchmark.py:build_dataset()`; explicitly selected single units with sufficient spikes and independent waveform halves, represented by opaque pseudo-unit IDs.
- `reference_labels.csv` ← source-sort cluster membership in `benchmark.py:build_dataset()`; labels are kept separate from matching inputs. All temporal parts from a well stay in one partition.
- `pairs.csv` ← `benchmark_pairs.py:build_benchmark_pairs()`; ordinary cross-pseudo-session pairs within each original recording, using measured waveform and response features without source-unit labels. Candidate and feature requirements are recorded in `manifest.json`.

## Panels

No figures are produced by this extraction step. Later method runs consume the same eligible pseudo-unit population. Exclusions and prespecified missing observations are recorded in `features.json`.

## Caveats

Original sorting contamination remains part of the reference. Temporal splitting does not reproduce probe reinsertion. Stimulus-based identity selection can bias conclusions about response stability. Missing or insufficient waveform halves are excluded, never duplicated.
'''
# [AGENT EDIT END]
