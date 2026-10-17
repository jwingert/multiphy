"""Create a cache-only report from completed benchmark run directories.

Run: python -m multiphy.comparison_report --run RUN_DIRECTORY
     --run ANOTHER_RUN --output reports/comparison_name
Source datasets are hash-verified and saved evaluations are checked against the
saved tracks. No matcher is fitted, rerun, ranked or tuned by this report.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
from pathlib import Path
import sys
import uuid

import numpy as np

from .backends import load_result
from .benchmark import load_dataset
from .evaluation import evaluate_result
from .run.config import resolve_cache_path
from .reporting_names import FEATURES, METHODS, METRICS, TRACKING, DEFINITIONS, count_range, display
from .contract import names, registry
from .contract.inputs import Input


def _spec(backend_id):
    try:
        return registry.get(backend_id).spec
    except KeyError:
        return None


COUNTS = ('n_units', 'n_tracks', 'n_unmatched', 'n_known_positive_pairs',
          'true_positive_pairs', 'false_positive_pairs', 'false_negative_pairs',
          'true_unmatched_units', 'correctly_unmatched_units',
          'unknown_unmatched_status_units', 'reported_unmatched_with_unknown_status',
          'n_unknown_predicted_pairs')
RATIOS = ('pair_precision', 'pair_recall', 'unmatched_precision', 'unmatched_recall')


def _json(path):
    return json.loads(Path(path).read_text())


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _provenances(result):
    grouped = (result.provenance.get('group_runs') or result.provenance.get('recording_group_runs')
               or result.provenance.get('well_runs'))
    if grouped:
        return [(name, data['provenance']) for name, data in sorted(grouped.items())]
    grouped = result.provenance.get('recording_results')
    return sorted(grouped.items()) if grouped else [('All recordings', result.provenance)]


def method_name(result):
    name = names.backend_short(result.backend_id)
    spec = _spec(result.backend_id)
    option_names = {o.name for o in spec.options} if spec is not None else set()
    if 'preset' in option_names:
        from .backends.dant import PRESETS
        labels = {PRESETS[p['parameters']['_nems_preset']['name']]
                  if '_nems_preset' in p.get('parameters', {}) else
                  'Explicit custom settings: ' + ', '.join(display(f, FEATURES)
                      for f in p['parameters']['clustering']['features'])
                  for _, p in _provenances(result)}
        return name + ' — ' + '; '.join(sorted(labels))
    tracking = result.provenance.get('assignment')
    operating_point = (result.provenance.get('assignment_selection') == 'calibration_operating_point'
                       or tracking == 'common_constrained_operating_point')
    name += (' — ' + display('common_constrained_operating_point' if operating_point else tracking, TRACKING)
             if tracking else '')
    if operating_point:
        target = result.provenance.get('target_empirical_false_match_fraction')
        if target is None:
            raise ValueError('A calibration-selected result must declare its empirical false-match target')
        name += f' (calibration target: {100 * target:g}% incorrect assigned pairs)'
    return name


def inspect_run(run_dir, *, datasets=None):
    """Reject incomplete, stale, mismatched or partial-population comparisons."""
    run_dir = Path(run_dir).resolve()
    manifest = _json(run_dir / 'run_manifest.json')
    if manifest.get('status') != 'completed':
        raise ValueError(f'Run is not completed: {run_dir}')
    result = load_result(run_dir / 'result.json')
    saved = _json(run_dir / 'evaluation.json')
    metrics = saved.get('combined', saved.get('aggregate'))
    if not isinstance(metrics, dict):
        raise ValueError('Unsupported evaluation schema')
    input_path = Path(manifest['input_path']).resolve()
    result_input = result.provenance.get('input_path') or result.provenance.get('benchmark_input')
    if result_input is None or Path(result_input).resolve() != input_path:
        raise ValueError('Result and manifest reference different benchmark inputs')
    if result.run_id != manifest.get('run_id', run_dir.name):
        raise ValueError('Result run identity disagrees with manifest')
    if manifest.get('backend', result.backend_id) != result.backend_id:
        raise ValueError('Result backend disagrees with manifest')
    if metrics.get('run_id', result.run_id) != result.run_id:
        raise ValueError('Saved evaluation run identity disagrees with result')
    if metrics.get('backend_id', result.backend_id) != result.backend_id:
        raise ValueError('Saved evaluation backend disagrees with result')
    digest = hashlib.sha256((input_path / 'manifest.json').read_bytes()).hexdigest()
    for provenance in (manifest, result.provenance):
        if provenance.get('input_manifest_sha256', digest) != digest:
            raise ValueError('Input benchmark manifest has changed')
    datasets = {} if datasets is None else datasets
    if input_path not in datasets:
        datasets[input_path] = load_dataset(input_path)
    dataset = datasets[input_path]
    for key in ('dataset_id', 'block_id', 'runclass', 'cache_built', 'sources', 'extraction_options', 'pair_options'):
        if result.provenance.get(key) != dataset.metadata.get(key):
            raise ValueError(f'Result benchmark provenance disagrees on {key}')
    if manifest['dataset_id'] != dataset.metadata['dataset_id']:
        raise ValueError('Manifest dataset identity disagrees with input')
    partition = result.provenance.get('partition') or result.provenance.get('benchmark_partition')
    if manifest.get('partition', partition) != partition:
        raise ValueError('Manifest and result partitions disagree')
    labels = dataset.labels if partition == 'all' else dataset.labels[dataset.labels.partition == partition]
    expected_ids = set(labels.cellid)
    if set(result.units.cellid) != expected_ids:
        raise ValueError('Run does not contain the exact declared benchmark partition population')
    expected = dataset.units.set_index('cellid').loc[sorted(expected_ids)]
    observed = result.units.set_index('cellid').loc[sorted(expected_ids)]
    for column in ('session_id', 'recording_group', 'su_type'):
        if not observed[column].astype(str).equals(expected[column].astype(str)):
            raise ValueError(f'Result unit {column} disagrees with benchmark')
    verified = evaluate_result(result, labels, expected_partition=None if partition == 'all' else partition)
    for key in COUNTS + RATIOS:
        if key not in metrics or (metrics[key] is None) != (verified[key] is None):
            raise ValueError(f'Saved evaluation is stale or incomplete: {key}')
        if metrics[key] is not None and not np.isclose(metrics[key], verified[key], rtol=0, atol=1e-12):
            raise ValueError(f'Saved evaluation disagrees with saved tracks: {key}')
    # Collection loaders prefix session IDs; canonical membership identifies the
    # same temporal grouping without collapsing distinct sessions or recordings.
    label_rows = labels.set_index('cellid')
    canonical = {cid: tuple(str(label_rows.loc[cid, key]) for key in ('well', 'source_session', 'source_unit_id')) +
        (int(dataset.features[cid]['start_sample']), int(dataset.features[cid]['stop_sample']),
         float(dataset.features[cid]['sample_rate'])) for cid in expected_ids}
    if len(set(canonical.values())) != len(canonical):
        raise ValueError('Source identity and temporal window do not uniquely identify observations')
    sessions = sorted(sorted(canonical[cid] for cid in rows.index) for _, rows in expected.groupby('session_id'))
    source_keys = set(map(tuple, labels[['well', 'source_session']].drop_duplicates().to_numpy()))
    sources = [s for s in dataset.metadata['sources'] if (s['well'], s['source_session']) in source_keys]
    population = {'observations': sorted(canonical.values()), 'sessions': sessions,
                  'sources': sorted(sources, key=lambda s: (s['well'], s['source_session'])),
                  'extraction_options': dataset.metadata.get('extraction_options'),
                  'pair_options': dataset.metadata.get('pair_options'), 'partition': partition}
    return dict(run_dir=run_dir, input_path=input_path, result=result, metrics=verified,
                name=method_name(result), population_id=_hash(population), partition=partition,
                source_names=sorted({str(s['source_session']) for s in sources}),
                input_manifest_sha256=digest, dataset=dataset, labels=labels)


def _cell(value):
    return str(value).replace('|', '\\|').replace('\n', ' ')


def _value(value, ratio=False):
    return 'Not defined' if value is None else (f'{100 * value:.2f}%' if ratio else str(value))


def _population_notes(row):
    dataset, labels = row['dataset'], row['labels']
    notes = []
    for (well, source), group in labels.groupby(['well', 'source_session']):
        features = [dataset.features[c] for c in group.cellid]
        windows = sorted({(f['start_sample'], f['stop_sample'], f['sample_rate']) for f in features})
        intervals = '; '.join(f'[{a / fs:.3f}, {b / fs:.3f}) seconds' for a, b, fs in windows)
        src = next(s for s in dataset.metadata['sources'] if s['well'] == well and s['source_session'] == source)
        split = src.get('provenance', {}).get('split_method', 'Split-construction rule unavailable in saved provenance')
        if 'whole BAPHY trials' in split:
            split = split.replace('whole BAPHY trials', 'whole experimental trials from the BAPHY experiment control system')
            split += '; trial indices are divided into approximately equal contiguous groups, with cuts at trial starts'
        well_description = f'well {well}'
        if ':' in str(well):
            animal, well_number = str(well).rsplit(':', 1)
            if well_number.isdigit():
                well_description += f' (animal {animal}; penetration well identifier {well_number})'
        notes.append(f'Recording {source}, {well_description}: {group.source_unit_id.nunique()} original sorted units represented. '
                     f'Observation windows from recording start: {intervals}; [start, stop) includes the start and excludes the stop. '
                     f'Split rule: {split}.')
    notes.append('Each temporal observation is split again at its sample-window midpoint into two internal waveform halves. UnitMatch compares these independently recomputed early and late means across observations; the two outer observations themselves are the units of matching.')
    options = dataset.metadata.get('extraction_options', {})
    notes.append('Inclusion requires an explicit single-unit label, the selected original sort job, and successfully recomputed features in each retained observation. '
                 f'The configured minimum is {options.get("min_spikes", 20)} spikes per observation; waveform snippets must fit wholly within their own window and both internal halves must be nonempty. '
                 'Source preparation exclusions and reasons remain in the linked dataset methods and manifest. No selection used these matching outcomes.')
    manifest = _json(row['input_path'] / 'manifest.json')
    if manifest.get('kind') == 'collection':
        for source in manifest['inputs']:
            omitted = len(source.get('omit_cellids', []))
            if omitted:
                omission_fraction_note = ''
                fraction_match = re.search(r'fraction=([0-9.]+)', source.get('omission_rule') or '')
                if fraction_match and 'later temporal sessions only' in source['omission_rule']:
                    fraction = float(fraction_match.group(1))
                    original = load_dataset(source['path'])
                    later_counts = []
                    for _, original_group in original.units.groupby('recording_group'):
                        ordered_sessions = sorted(original_group.session_id.unique())
                        later_counts.extend(int(original_group.session_id.eq(session).sum()) for session in ordered_sessions[1:])
                    calculations = '; '.join(f'floor({fraction:g} × {n}) = {int(fraction * n)} omitted' for n in sorted(set(later_counts)))
                    omission_fraction_note = f' The fraction applies to each later segment’s original {count_range(later_counts)} observations, not all observations in both segments: {calculations}.'
                notes.append(f'This population reuses the feature archive [{Path(source["path"]).name}]({source["path"]}) '
                             f'and removes {omitted} whole observations, retaining their opposite-segment counterparts. '
                             f'Omission rule: {source.get("omission_rule") or "Unavailable; explicit observation IDs are saved in the manifest"}.{omission_fraction_note} '
                             'The absent-counterpart population is a subset of that archive, not a new recording or a new waveform extraction.')
    return notes


def _group_label(group):
    try:
        well, source = json.loads(group)
        return f'well {well}, recording {source}'
    except (ValueError, TypeError):
        return str(group)


def _motion_order(row, group):
    labels = row['labels']
    try:
        well, source = json.loads(group)
        labels = labels[(labels.well == well) & (labels.source_session == source)]
    except (ValueError, TypeError):
        pass
    units = row['result'].units[row['result'].units.cellid.isin(labels.cellid)]
    entries = []
    for number, (_, members) in enumerate(units.groupby('session_id', sort=False), 1):
        f = row['dataset'].features[members.cellid.iloc[0]]
        entries.append(f"entry {number}: [{f['start_sample'] / f['sample_rate']:.3f}, "
                       f"{f['stop_sample'] / f['sample_rate']:.3f}) seconds from recording start")
    return '; '.join(entries)


def render_report(runs):
    """Render verified rows; each table has a single identical input population."""
    lines = ['# Temporal neuron matching comparison', '',
        DEFINITIONS['reference'], '',
        'Reference identity comes from the original single-unit spike sort. These are disjoint temporal observations from the same recording, not independently verified across-day neuron identities. Correct and incorrect below mean agreement or disagreement with that original sort.', '',
        'Only the explicitly supplied completed runs are included. Absence of another method is not a failed score. No model or threshold is selected from these results.', '',
        'The partition label “test” reserves source identities for evaluation rather than supervised model fitting or threshold selection. Native UnitMatch and DANT still estimate their own unsupervised similarity distributions, drift, or clustering parameters from the supplied test features; “test” does not mean those methods are frozen pretrained models.', '',
        DEFINITIONS['dant'], '',
        'Pair precision = correct predicted pairs / all evaluated predicted pairs. Pair recall = correct predicted pairs / all known same-neuron pairs. Unmatched precision = correctly unmatched observations / evaluable reported unmatched observations; unmatched recall = correctly unmatched observations / known unmatched observations. A zero denominator is “Not defined”.', '']
    groups = {}
    for row in runs:
        groups.setdefault(row['population_id'], []).append(row)
    for index, (population_id, rows) in enumerate(groups.items(), 1):
        metrics = rows[0]['metrics']
        status = ('With absent counterparts' if metrics['true_unmatched_units'] else
                  'Counterpart availability partly unknown' if metrics['unknown_unmatched_status_units'] else
                  'All retained source neurons have counterparts')
        lines += [f'## Population {index}: {status}', '',
            f"Original recordings: {', '.join(rows[0]['source_names'])}. Partition: {rows[0]['partition']}. "
            f"{metrics['n_units']} single-unit temporal observations; {metrics['n_known_positive_pairs']} known same-neuron pairs; {metrics['true_unmatched_units']} known unmatched observations.", '',
            f'Population verification key: `{population_id}`. This checksum groups runs with identical retained observations and extraction provenance; it is not a score or an additional scientific condition.', '']
        lines += [note + '\n' for note in _population_notes(rows[0])]
        for columns in [('true_positive_pairs', 'false_positive_pairs', 'false_negative_pairs', 'pair_precision', 'pair_recall'),
                        ('n_unmatched', 'correctly_unmatched_units', 'true_unmatched_units', 'unmatched_precision', 'unmatched_recall')]:
            lines += ['| Method | ' + ' | '.join(display(c, METRICS) for c in columns) + ' |',
                      '| --- | ' + ' | '.join('---' for _ in columns) + ' |']
            for row in rows:
                lines.append('| ' + _cell(row['name']) + ' | ' + ' | '.join(_value(row['metrics'][c], c in RATIOS) for c in columns) + ' |')
            lines.append('')
        unknown = {(r['metrics']['unknown_unmatched_status_units'], r['metrics']['n_unknown_predicted_pairs']) for r in rows}
        lines += ['Unknown reference identities are excluded from correctness denominators. '
                  + '; '.join(f'{u} observations with unknown unmatched status, {p} predicted pairs with unknown identity' for u, p in sorted(unknown)) + '.', '',
                  '### Run details', '']
        for row in rows:
            lines += [f"- **{row['name']}** — [saved result]({row['run_dir'] / 'result.json'}), [evaluation]({row['run_dir'] / 'evaluation.json'}), [methods]({row['run_dir'] / 'METHODS.md'})."]
            assignment = row['result'].provenance.get('assignment')
            if (assignment == 'common_constrained_operating_point'
                    or row['result'].provenance.get('assignment_selection') == 'calibration_operating_point'):
                p = row['result'].provenance
                lines.append(f"  - The cutoff {p.get('threshold'):.6g} was selected using calibration identities and frozen before application to these scores. The target is {100 * p.get('target_empirical_false_match_fraction'):g}% incorrect pairs among all assigned calibration pairs after track construction; the observed test precision is reported above. This target does not guarantee the same error fraction on test data. Original native assignments remain separately saved.")
            for group, provenance in _provenances(row['result']):
                spec = _spec(row['result'].backend_id)
                requires = spec.requires if spec is not None else frozenset()
                if (Input.WAVEFORM_HALVES in requires and Input.SPIKE_TIMES not in requires
                        and Input.CHECKPOINT not in requires):
                    threshold = row['result'].provenance.get('threshold', provenance.get('threshold', 'unavailable'))
                    policy = assignment or provenance.get('assignment')
                    detail = (DEFINITIONS['intermediate'] if policy == 'native_intermediate' else
                        'The common tracker requires candidate competition, one observation per session, and no strong measured contradiction.'
                        if policy in ('common_constrained', 'mutual_margin_best_first', 'common_constrained_operating_point') else
                        'Assignments use the native tracking policy named in the method row.')
                    lines.append(f'  - {DEFINITIONS["directions"]} Recorded threshold: {threshold}. {detail}')
                if row['result'].score.similarity_column is not None:
                    lines.append('  - Assignments are the native automatic clustering and automatic curation output; no human edits were applied by this report. Curation removes inconsistent or low-similarity links and may split a proposed group. The native similarity is not a probability; its learned threshold and cluster decisions jointly determine the saved tracks.')
                diag = provenance.get('native_diagnostics', {})
                if 'SimilarityWeights' in diag:
                    weights = ', '.join(f'{display(name, FEATURES)}: {float(weight):.4f}' for name, weight in zip(diag['SimilarityNames'], diag['SimilarityWeights']))
                    lines.append(f'  - Native fitted discriminant coefficients ({_cell(_group_label(group))}): {weights}. {DEFINITIONS["dant_weights"]} Recorded final native similarity threshold: {float(diag["SimilarityThreshold"]):.4f}.')
                for event in provenance.get('compatibility_events', []):
                    lines.append(f"  - Drift compatibility event during iterative motion estimation: no across-session pair passed both native cluster membership and the learned similarity threshold. The adapter retained the previous per-session depth offsets {event['constant_shift_um']} micrometers instead of fitting motion from zero pairs. No motion update was invented; downstream clustering then continued with the retained estimate. {DEFINITIONS['dant_motion']} Vector order is {_motion_order(row, group)}. The event log does not record the iteration number.")
                for caveat in provenance.get('upstream_caveats', []):
                    lines.append('  - Upstream implementation caveat: ' + (DEFINITIONS['dant_caveat'] if 'early-stop reverts' in caveat else _cell(caveat)).rstrip('.') + '.')
            lines.append('')
    lines += ['## Interpretation limits', '',
        'Temporal halves inherit any original sorting mistakes. Deliberately omitted observations test rejection of absent counterparts; they do not simulate all changes caused by a new recording day. Pairs and neurons within a recording are dependent. These counts provide a pilot comparison, not evidence of biological ground-truth accuracy, novelty, or general superiority.', '',
        'UnitMatch uses the tracking policy named in its row; native DANT uses its own clustering and automatic curation. DANT similarity is unbounded. Native Bayesian scores are not automatically calibrated identity probabilities for this laboratory. Methods may use different evidence and learned parameters; all input and output details remain linked above.', '']
    return '\n'.join(lines)


def write_comparison(run_dirs, output_dir=None, *, root=None):
    """Write one immutable report only after all supplied inputs pass validation."""
    paths = [Path(p).resolve() for p in run_dirs]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError('Supply nonempty, unique completed run directories')
    datasets = {}
    runs = [inspect_run(p, datasets=datasets) for p in paths]
    output = resolve_cache_path(output_dir or f'reports/comparison_{uuid.uuid4().hex}', root=root)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'comparison.md').write_text(render_report(runs), encoding='utf-8')
    lineage = '\n'.join(f'- `{r["run_dir"]}/result.json`, `evaluation.json`, `run_manifest.json` ← `runners.py:run_backend_dataset()`; input `{r["input_path"]}`.' for r in runs)
    methods = f'''# Temporal matching comparison — figure & methods notes

- **Output folder:** `{output}`
- **Artifacts:** `comparison.md`, `report_manifest.json`
- **Generated by:** `multiphy/comparison_report.py:write_comparison()` · **Env:** `{sys.executable}` · **Updated:** `{datetime.now(timezone.utc).date()}`

---

## Goal

Compare completed matching methods on identical temporal single-unit observations. Report matching and unmatched decisions relative to original sorted identities, separately for each retained input population.

## Data lineage

{lineage}
- Source benchmark feature archives ← `benchmark.py` and `benchmark_collection.py`; their METHODS document raw voltage extraction, chronological spike partitions, condition-aligned response fingerprints, single-unit selection and omissions.

## Panels

### Matching and unmatched tables
- **Made by:** `render_report()` in `multiphy/comparison_report.py`.
- **Input:** Serialized tracks and unmatched observations; saved pair counts; separately stored original source identity labels.
- **Steps:**
  1. `inspect_run()` verifies completed manifests, portable result hashes, source archive hashes, source provenance, session membership and exact partition populations.
  2. `evaluation.py:evaluate_result()` recomputes pair and unmatched counts from the saved decisions and separated reference labels; disagreements with saved evaluation files are rejected.
  3. `render_report()` groups identical observations and extraction provenance, preserving complete versus missing-counterpart populations. No matching is rerun or tuned. Ratios use the displayed count denominators; zero denominators remain undefined.
- **Shows:** Matching counts and rejection counts for each method on the same observations. Native signed discriminant coefficients are diagnostic parameters, not feature importance. DANT similarity formulas come from the pinned pyDANT IterativeClustering.py:computeAllSimilarityMatrix()/iterativeClustering(); motion-coordinate definitions come from MotionEstimation.py:computeMotion() and IterativeClustering.py:getNearbyPairs(). UnitMatch grouping follows the pinned assign_unique_id.py:assign_unique_id().

## Caveats

Original sorts are conditional references, not independently verified across-day identities. Related pairs are dependent; no confidence intervals or significance tests are implied. No figures are generated.
'''
    (output / 'METHODS.md').write_text(methods, encoding='utf-8')
    manifest = {'created_at': datetime.now(timezone.utc).isoformat(), 'runs': [dict(
        run_dir=str(r['run_dir']), run_id=r['result'].run_id, backend=r['result'].backend_id,
        population_id=r['population_id'], input_path=str(r['input_path']),
        input_manifest_sha256=r['input_manifest_sha256']) for r in runs]}
    (output / 'report_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='append', required=True, help='Completed run directory; repeat per method')
    parser.add_argument('--output', help='New output directory beneath the UnitMatch cache root')
    args = parser.parse_args(argv)
    print(write_comparison(args.run, args.output))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
