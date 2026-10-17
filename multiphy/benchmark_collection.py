"""Immutable, hash-pinned views over existing benchmark waveform archives."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from pathlib import Path
import shutil
import tempfile

import pandas as pd

from .run.config import resolve_cache_path


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _pair_options_override(options):
    """Resolve an explicit rebuild recipe before any output is created."""
    if not isinstance(options, dict):
        raise ValueError('pair_options override must be a dictionary')
    resolved = dict(min_repeats=4, min_reliability=None, max_distance_um=None)
    unknown = set(options) - set(resolved)
    if unknown:
        raise ValueError(f'Unknown pair_options override fields: {sorted(unknown)}')
    resolved.update(options)
    repeats = resolved['min_repeats']
    if isinstance(repeats, bool) or not isinstance(repeats, Integral) or repeats < 1:
        raise ValueError('min_repeats must be a positive integer')
    resolved['min_repeats'] = int(repeats)
    for key in ('min_reliability', 'max_distance_um'):
        value = resolved[key]
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
            raise ValueError(f'{key} must be a finite number or None')
        if (key == 'min_reliability' and not -1 <= value <= 1) or (key == 'max_distance_um' and value < 0):
            raise ValueError(f'Invalid {key} bound')
        resolved[key] = float(value)
    return resolved


def load_collection(path, manifest, ancestors=()):
    from .benchmark import BenchmarkDataset, load_dataset
    override = (_pair_options_override(manifest['pair_options_override'])
                if 'pair_options_override' in manifest else None)
    references = manifest.get('inputs')
    if not references or not isinstance(references, list):
        raise ValueError('Collection requires input benchmark references.')
    datasets, source_keys, seen_ids = [], set(), set()
    geometry, features, exclusions, sources = {}, {}, [], []
    for reference in references:
        source = Path(reference['path'])
        if not source.is_absolute():
            source = Path(path) / source
        if _digest(source / 'manifest.json') != reference['manifest_sha256']:
            raise ValueError('Referenced benchmark manifest hash mismatch.')
        ds = load_dataset(source, _ancestors=ancestors)
        keys = set(map(tuple, ds.labels[['well', 'source_session']].drop_duplicates().to_numpy()))
        if keys & source_keys:
            raise ValueError('A source recording cannot appear twice in a collection.')
        source_keys.update(keys)
        omitted = set(reference.get('omit_cellids', []))
        if not omitted <= set(ds.units.cellid):
            raise ValueError('Omitted observation is absent from its source benchmark.')
        for cid in sorted(omitted):
            label = ds.labels.set_index('cellid').loc[cid]
            exclusions.append(dict(cellid=cid, source_session=label.source_session,
                source_unit_id=label.source_unit_id, well=label.well,
                reason='prespecified missing counterpart', selection=reference.get('omission_rule')))
        if omitted:
            ds = ds.subset(set(ds.units.cellid) - omitted)
        if seen_ids & set(ds.units.cellid):
            raise ValueError('Collection contains colliding observation IDs.')
        seen_ids.update(ds.units.cellid)
        # Independent archives reuse session names. New collections also
        # namespace observation IDs to retain the shared curator's site prefix.
        # Legacy manifests keep their exact original identifiers for audit.
        prefix = reference['manifest_sha256'][:16]
        session_map = {s: f'{prefix}:{s}' for s in ds.units.session_id.unique()}
        ds.units = ds.units.copy()
        policy = manifest.get('observation_id_policy', 'preserve')
        if policy not in ('preserve', 'session_prefix_v1'):
            raise ValueError('Unknown collection observation ID policy.')
        if policy == 'session_prefix_v1':
            id_map = {r.cellid: session_map[r.session_id] + '-' + r.cellid.split('-', 1)[1]
                      for r in ds.units.itertuples()}
            ds.units['cellid'] = ds.units.cellid.map(id_map)
            ds.labels = ds.labels.copy()
            ds.labels['cellid'] = ds.labels.cellid.map(id_map)
            ds.pairs = ds.pairs.copy()
            for column in ('cellid_a', 'cellid_b'):
                ds.pairs[column] = ds.pairs[column].map(id_map)
            ds.features = {id_map[c]: f for c, f in ds.features.items()}
        ds.units['session_id'] = ds.units.session_id.map(session_map)
        ds.units['siteid'] = ds.units.siteid.map(session_map)
        geometry.update({session_map[s]: g for s, g in ds.geometry.items()})
        features.update(ds.features)
        sources.extend(ds.metadata['sources'])
        exclusions.extend(ds.exclusions)
        datasets.append(ds)
    required_common = ('extraction_options',) + (() if override is not None else ('pair_options',))
    for key in required_common:
        if any(ds.metadata.get(key) != datasets[0].metadata.get(key) for ds in datasets[1:]):
            raise ValueError(f'Collection inputs disagree on {key}; use comparable extraction settings.')
    metadata = dict(datasets[0].metadata)
    metadata.pop('runclass', None)
    metadata.update(dataset_id=manifest['dataset_id'], block_id=manifest['dataset_id'],
        sources=sources, cache_built=manifest['created_at'], collection_inputs=references,
        collection_manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest())
    units = pd.concat([d.units for d in datasets], ignore_index=True)
    if override is None:
        pairs = pd.concat([d.pairs for d in datasets], ignore_index=True)
    else:
        from .benchmark_pairs import build_benchmark_pairs
        pairs = build_benchmark_pairs(units, features, geometry, **override)
        metadata.update(pair_options=override, pair_options_override_provenance=dict(
            operation='rebuild_benchmark_pairs_from_retained_features',
            implementation='multiphy.benchmark_pairs.build_benchmark_pairs',
            options=override,
            source_options=[dict(path=reference['path'], manifest_sha256=reference['manifest_sha256'],
                                 pair_options=ds.metadata.get('pair_options', {}))
                            for reference, ds in zip(references, datasets)],
            source_archives_unchanged=True, reference_labels_used_for_pair_evidence=False))
    return BenchmarkDataset(units, pairs, features,
        pd.concat([d.labels for d in datasets], ignore_index=True), geometry,
        metadata, exclusions).validate()


def create_collection(inputs, output, *, dataset_id, omissions=None, omission_rule=None,
                      pair_options=None, root=None):
    """Pin archives and optionally rebuild evidence; never change source features or partitions.

    An explicit pair_options dictionary replaces source pair settings, filling
    unspecified fields with build_benchmark_pairs defaults. None preserves the
    source candidate tables and requires their pair settings to agree.
    """
    override = None if pair_options is None else _pair_options_override(pair_options)
    output = resolve_cache_path(output, root=root)
    if output.exists():
        raise FileExistsError(output)
    if not dataset_id or any(c in dataset_id for c in '/\\') or dataset_id in ('.', '..'):
        raise ValueError('Use a nonempty single-component dataset ID.')
    omissions = omissions or {}
    references = []
    for value in inputs:
        source = Path(value).resolve()
        references.append(dict(path=str(source), manifest_sha256=_digest(source / 'manifest.json'),
            omit_cellids=sorted(omissions.get(str(source), [])), omission_rule=omission_rule))
    if set(omissions) - {r['path'] for r in references}:
        raise ValueError('Omission mapping refers to an unknown input archive.')
    manifest = dict(schema_version=2, kind='collection', dataset_id=dataset_id,
        observation_id_policy='session_prefix_v1',
        created_at=datetime.now(timezone.utc).isoformat(), inputs=references)
    if override is not None:
        manifest['pair_options_override'] = override
    dataset = load_collection(output, manifest, (output,))
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix='.collection-', dir=output.parent))
    try:
        (stage / 'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False))
        (stage / 'METHODS.md').write_text(f'''# Benchmark collection — methods notes

- **Output folder:** `{output}`
- **Artifacts:** `manifest.json`
- **Generated by:** `multiphy/benchmark_collection.py:create_collection()` · **Updated:** {datetime.now(timezone.utc).date()}

## Goal

Compare matching methods on fixed, independently prepared recording wells and explicitly selected missing observations. Original training, calibration and test partitions are preserved.

## Data lineage

- `manifest.json` ← existing benchmark manifests, pinned by SHA256. Each referenced archive verifies all feature, label and geometry hashes on load. Input locations and omitted observation IDs are listed in the manifest; source METHODS describe voltage extraction.
- In-memory inputs ← `load_collection()`: remove declared observations and their incident pairs, disambiguate session IDs, and retain separate reference labels. No waveform arrays are copied to this folder.
- Candidate evidence ← {('`benchmark_pairs.build_benchmark_pairs()` recomputes within-recording candidates from retained waveform, physiology and sound-response features after session/observation namespacing. Explicit resolved settings: `' + json.dumps(override, sort_keys=True) + '`. Input settings and this rebuild operation are recorded in loaded metadata; source archives and features stay unchanged.') if override is not None else '`load_collection()` concatenates original within-recording candidate tables; no pair-option override was requested.'}

## Panels

No figures. The collection contains {len(dataset.units)} retained observations and {sum(len(r['omit_cellids']) for r in references)} deliberately omitted observations. Every method receives the same retained population. Omission rule: {omission_rule or 'none'}.

## Caveats

Conditional source-sort identity is not verified multiday ground truth. Missing-counterpart tests remove complete observations, not raw spikes. Unit IDs selected for omission must be specified before comparing method outcomes. Reference archives must remain available and unchanged. Lowering the required stimulus repeats changes evidence availability, not the recorded number of trials or uncertainty: one trial cannot yield an independent within-observation trial-half reliability estimate. No labels or matching outcomes choose the requested pair settings.
''')
        output.mkdir()
        stage.replace(output)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return output


def missing_counterpart_ids(dataset, fraction=.2, seed=0):
    """Reproducible hash-ranked omissions, independent of scores and outcomes."""
    if not 0 < fraction < 1:
        raise ValueError('Omission fraction must be strictly between zero and one.')
    joined = dataset.units[['cellid', 'session_id', 'recording_group']].merge(dataset.labels, on='cellid')
    omitted = []
    for group, rows in joined.groupby('recording_group', sort=True):
        sessions = sorted(rows.session_id.unique())
        if len(sessions) < 2:
            raise ValueError('Missing-counterpart benchmark requires multiple temporal sessions.')
        for session in sessions[1:]:
            available = rows[rows.session_id == session]
            n = int(len(available) * fraction)
            if n == 0:
                raise ValueError('Requested omission fraction removes no observations.')
            ranked = sorted(available.cellid, key=lambda cid: hashlib.sha256(f'{seed}:{group}:{session}:{cid}'.encode()).hexdigest())
            omitted.extend(ranked[:n])
    return omitted


def main(argv=None):
    from .benchmark import load_dataset
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', action='append', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--dataset-id', required=True)
    parser.add_argument('--omit-fraction', type=float)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--min-stimulus-repeats', type=int,
                        help='Explicitly rebuild collection pairs with this positive per-condition repeat minimum')
    parser.add_argument('--root', metavar='DIR', help='lab root · else MULTIPHY_CACHE_ROOT')
    args = parser.parse_args(argv)
    pair_options = None if args.min_stimulus_repeats is None else _pair_options_override(
        {'min_repeats': args.min_stimulus_repeats})
    omissions, rule = {}, None
    if args.omit_fraction is not None:
        rule = f'hash-ranked fraction={args.omit_fraction:g}, seed={args.seed}, later temporal sessions only; independent of match evidence'
        for path in args.input:
            omissions[str(Path(path).resolve())] = missing_counterpart_ids(load_dataset(path), args.omit_fraction, args.seed)
    print(create_collection(args.input, args.output, dataset_id=args.dataset_id, omissions=omissions,
                            omission_rule=rule, pair_options=pair_options, root=args.root))


if __name__ == '__main__':
    main()
