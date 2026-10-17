"""Separate select and apply commands for saved matching evidence.

No native method, feature model, or probability transform is fitted here.
Operating-point analysis does not change the existing benchmark workflow.
"""
import argparse
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

import numpy as np
import pandas as pd

from .backends import MatchResult, _json_safe, load_result, result_digest, result_from_pairs, save_result
from .benchmark import load_dataset
from .evaluation import evaluate_result, validate_partitions
from .run.config import new_run_path, resolve_cache_path
from .contract import registry
from .contract.inputs import Input
from .contract.provenance import Provenance
from .contract.results import ScoreBlock, ScoreKind, TrackKind


def _spec(result):
    """The registered spec of a result's backend, or None."""
    try:
        return registry.get(result.backend_id).spec
    except KeyError:
        return None


def _scalar_scores(result):
    """The backend declares a scalar pair score; DANT's native similarity is not one."""
    spec = _spec(result)
    return spec is not None and bool(spec.provides.score_kinds - {ScoreKind.NONE})


def _partition(provenance):
    return provenance.get('partition') or provenance.get('benchmark_partition')


POPULATION_PARAMETERS = ('n_units', 'n_sessions', 'n_expected_matches')
CAVEATS = (
    'The target is empirical assignment FP/(TP+FP), not a bound or guarantee on future error. '
    'Zero accepted pairs has undefined false-match fraction. Candidate pairs and track links are dependent.',
    'A single calibration well and a small number of positive identities give a coarse, unstable estimate; '
    '53 accepted pairs change by about 1.9 percentage points per error. No independent-pair confidence interval is claimed.',
    'Operating-point selection reuses the calibration well; '
    'its observed error is a development-set selection statistic, not independent calibration validation.',
    'Previously inspected pilot tests make this a documented frozen reanalysis, not a newly prospective untouched-test study.',
    'Temporal reference identities inherit the original sort and do not establish cross-day neuron identity.',
    'UnitMatch and DeepUnitMatch score distributions and drift may be estimated unsupervised on each input population. '
    'Only this downstream common tracker threshold is frozen. Native score labels remain uncalibrated.',
    'DANT native similarity, clustering and curation do not define this scalar operating-point family; '
    'keep DANT as a separately labeled native baseline.',
)


def _digest(value):
    return hashlib.sha256(json.dumps(_json_safe(value), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _dataset_identity(dataset, manifest_hash):
    if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
        raise ValueError('A hash-verified full archive manifest identity is required')
    return dict(input_manifest_sha256=manifest_hash,
        **{key: dataset.metadata[key] for key in ('dataset_id', 'block_id', 'runclass', 'cache_built')})


def _validate_full_dataset(dataset):
    dataset.validate()
    validate_partitions(dataset.labels)
    if set(dataset.labels.partition) != {'train', 'calibration', 'test'}:
        raise ValueError('Supply the full archive with train, calibration and test wells')
    joined = dataset.units[['cellid', 'recording_group', 'session_id']].merge(
        dataset.labels[['cellid', 'well', 'source_session']], on='cellid', validate='one_to_one')
    if joined.groupby('recording_group')[['well', 'source_session']].nunique().gt(1).any().any():
        raise ValueError('Each assignment domain must be one original recording')


def _validate_scores(dataset, result, partition, manifest_hash):
    _validate_full_dataset(dataset)
    result.validate()
    if not _scalar_scores(result):
        raise ValueError('Scalar selection supports native UnitMatch and DeepUnitMatch scores; DANT stays a native baseline')
    identity = _dataset_identity(dataset, manifest_hash)
    if any(result.provenance.get(k) != v for k, v in identity.items()):
        raise ValueError('Saved scores do not match the full archive identity')
    if _partition(result.provenance) != partition:
        raise ValueError(f'Saved scores must declare the {partition} partition')
    expected = dataset.units[dataset.units.cellid.isin(dataset.labels.loc[dataset.labels.partition == partition, 'cellid'])]
    columns = ['cellid', 'session_id', 'siteid', 'su_type', 'recording_group']
    if (not set(columns) <= set(result.units) or set(result.units.cellid) != set(expected.cellid)
            or not result.units[columns].sort_values('cellid').reset_index(drop=True).equals(
                expected[columns].sort_values('cellid').reset_index(drop=True))):
        raise ValueError('Scores must cover the exact declared partition population and session identities')
    groups = result.units.set_index('cellid').recording_group.to_dict()
    sessions = result.units.set_index('cellid').session_id.to_dict()
    if any(groups[a] != groups[b] for a, b in zip(result.pairs.cellid_a, result.pairs.cellid_b)):
        raise ValueError('Score pairs must stay within original recordings')
    observed_pairs = {tuple(sorted((a, b))) for a, b in zip(result.pairs.cellid_a, result.pairs.cellid_b)
                      if sessions[a] != sessions[b]}
    expected_pairs = {tuple(sorted((a.cellid, b.cellid)))
        for _, group in expected.groupby('recording_group')
        for a, b in itertools.combinations(group.itertuples(), 2) if a.session_id != b.session_id}
    if observed_pairs != expected_pairs:
        raise ValueError('Saved score pair coverage is incomplete or differs from the required candidate population; retain missing scores as NaN rows')
    expected_kind = 'uncalibrated_score'
    if len(result.pairs) and result.score.kind.value != expected_kind:
        raise ValueError(f'Score kind {result.score.kind.value!r} does not match the expected {expected_kind!r}')
    return expected_kind


def _group_provenance(result, group):
    p = result.provenance
    runs = p.get('group_runs') or p.get('recording_group_runs') or p.get('well_runs')
    if runs:
        return runs[str(group)]['provenance']
    if p.get('recording_results'):
        return p.get('recording_results')[str(group)]
    return p.to_json() if isinstance(p, Provenance) else p


def scoring_signature(dataset, result):
    """Compare science settings, not paths, native assignments, or fitted population size.

    All native parameters are retained except POPULATION_PARAMETERS. File paths,
    run IDs, timestamps, native track decisions and tracker thresholds are not
    scoring identities. Implementation content/revision, checkpoint content,
    acquisition transforms, geometry, sample representation, and candidate recipe
    are retained. Unknown future native parameters are retained conservatively.
    """
    signatures = []
    sources = {(s['well'], s['source_session']): s for s in dataset.metadata['sources']}
    labels = dataset.labels.set_index('cellid')
    for group, units in result.units.groupby('recording_group', sort=True):
        p = _group_provenance(result, group)
        common = dict(backend_id=result.backend_id, backend_version=result.version,
            extraction_options=dataset.metadata.get('extraction_options', {}),
            pair_options=dataset.metadata.get('pair_options', {}))
        required = ('official_source', 'verified_api_revision', 'implementation', 'parameters', 'pair_reduction')
        if any(k not in p for k in required):
            raise ValueError('Native scores are missing upstream/scoring configuration provenance')
        waveform_provenance = p.get('waveform_provenance', {})
        for key in ('dataset_id', 'block_id', 'runclass', 'cache_built', 'sources', 'extraction_options', 'pair_options'):
            if _json_safe(waveform_provenance.get(key)) != _json_safe(dataset.metadata.get(key)):
                raise ValueError('Native waveform scoring provenance differs from the input archive')
        expected_groups = {row.session_id: labels.loc[row.cellid, 'well'] for row in units.itertuples()}
        if waveform_provenance.get('source_groups') != expected_groups:
            raise ValueError('Native waveform source-group mapping differs from the scored recording')
        implementation = p['implementation']
        if not implementation.get('module') or not implementation.get('sha256'):
            raise ValueError('Native implementation module and content hash are required')
        common.update(official_source=p['official_source'], verified_api_revision=p['verified_api_revision'],
            implementation={k: implementation[k] for k in ('module', 'sha256')},
            parameters={k: v for k, v in p['parameters'].items() if k not in POPULATION_PARAMETERS},
            pair_reduction=p['pair_reduction'])
        if p['pair_reduction'] != 'minimum of both directional probabilities':
            raise ValueError('Native scores require the documented directional reduction')
        # The stored score must actually be the declared conservative reduction.
        pairs = result.pairs[result.pairs.cellid_a.isin(units.cellid)]
        if not {'official_probability_ab', 'official_probability_ba'} <= set(pairs):
            raise ValueError('Native scores require both directional outputs')
        reduced = np.minimum(pd.to_numeric(pairs.official_probability_ab, errors='raise').to_numpy(dtype=float),
                             pd.to_numeric(pairs.official_probability_ba, errors='raise').to_numpy(dtype=float))
        if not np.allclose(pairs.match_score, reduced, rtol=0, atol=0, equal_nan=True):
            raise ValueError('Native match_score disagrees with its declared directional reduction')
        spec = _spec(result)
        if spec is not None and Input.CHECKPOINT in spec.requires:
            for key in ('checkpoint_sha256', 'checkpoint_probe', 'network_preprocessing'):
                if not p.get(key):
                    raise ValueError('Deep scores require checkpoint and network configuration provenance')
                common[key] = p[key]
            training = p.get('training', {})
            train_groups = set(training.get('training_groups', ()))
            allowed = set(dataset.labels.loc[dataset.labels.partition == 'train', 'well'])
            if not train_groups or not train_groups <= allowed:
                raise ValueError('Deep training groups must be verified training wells in the full archive')
            common['deep_training_groups'] = sorted(train_groups)
        source_keys = {tuple(labels.loc[c, ['well', 'source_session']]) for c in units.cellid}
        source_configs = []
        for key in sorted(source_keys):
            source = sources[key]
            prov = source['provenance']
            source_configs.append(dict(preprocessing=source['preprocessing'], probe_type=prov.get('probe_type'),
                voltage_scale=prov.get('voltage_scale'), common_median=prov.get('common_median'),
                baseline_samples=prov.get('baseline_samples'), waveform_preprocessing=({k: v for k, v in prov['waveform_preprocessing'].items() if k != 'storage'}
                    if isinstance(prov.get('waveform_preprocessing'), dict) else prov.get('waveform_preprocessing'))))
        # Source filenames and identities differ by definition; measured transforms must not.
        common['source_transforms'] = sorted({_digest(s): s for s in source_configs}.values(), key=_digest)
        common['sample_representations'] = sorted({_digest(dict(
            sample_rate=dataset.features[c]['sample_rate'], waveform_shape=list(np.asarray(dataset.features[c]['waveform_halves']).shape),
            fingerprint_bin_seconds=dataset.features[c].get('fingerprint_bin_seconds'),
            geometry=np.asarray(dataset.geometry[s]).tolist())): dict(
            sample_rate=dataset.features[c]['sample_rate'], waveform_shape=list(np.asarray(dataset.features[c]['waveform_halves']).shape),
            fingerprint_bin_seconds=dataset.features[c].get('fingerprint_bin_seconds'),
            geometry=np.asarray(dataset.geometry[s]).tolist())
            for c, s in zip(units.cellid, units.session_id)}.values(), key=_digest)
        signatures.append(_json_safe(common))
    if not signatures or len({_digest(s) for s in signatures}) != 1:
        raise ValueError('Source recordings use inconsistent model/scoring configurations')
    return signatures[0]


def _assign_partition(result, threshold, contradiction, ambiguity, *, selection_hash=None):
    assigned = []
    kind, reference = result.score.kind, result.score.calibration_reference
    base = result.provenance.to_json()
    for group, units in result.units.groupby('recording_group', sort=True):
        ids = set(units.cellid)
        pairs = result.pairs[result.pairs.cellid_a.isin(ids) & result.pairs.cellid_b.isin(ids)]
        p = dict(base, recording_group=str(group), operating_point_sha256=selection_hash,
                 score_origin_digest=result_digest(result))
        assigned.append(result_from_pairs(result.backend_id, result.version, units, pairs,
            score_kind=kind, threshold=threshold, contradiction_threshold=contradiction,
            ambiguity_margin=ambiguity, provenance=p, calibration_reference=reference))
    p = dict(base, assignment='common_constrained', assignment_selection='calibration_operating_point', threshold=threshold,
        contradiction_threshold=contradiction, ambiguity_margin=ambiguity,
        operating_point_sha256=selection_hash, score_origin_digest=result_digest(result),
        native_assignment_preserved_separately=True)
    return MatchResult(result.backend_id, result.version, 'operating_point_' + uuid.uuid4().hex,
        result.units.copy(), pd.concat([r.pairs for r in assigned], ignore_index=True),
        tuple(t for r in assigned for t in r.tracks), tuple(u for r in assigned for u in r.unmatched), (),
        ScoreBlock(kind, 'match_score', calibration_reference=reference), TrackKind.COMMON, 'common_constrained',
        Provenance.from_json(p)).validate()


def _evaluate_partition(result, labels, partition):
    evaluations = {}
    for group, units in result.units.groupby('recording_group', sort=True):
        ids = set(units.cellid)
        subset = MatchResult(result.backend_id, result.version, result.run_id, units.copy(),
            result.pairs[result.pairs.cellid_a.isin(ids) & result.pairs.cellid_b.isin(ids)].copy(),
            tuple(t for t in result.tracks if set(t) <= ids), tuple(u for u in result.unmatched if u in ids), (),
            result.score, result.track_kind, result.tracking_rule, result.provenance)
        evaluations[str(group)] = evaluate_result(subset, labels, expected_partition=partition)
    keys = ('true_positive_pairs', 'false_positive_pairs', 'false_negative_pairs', 'n_known_positive_pairs',
        'n_unknown_predicted_pairs', 'n_units', 'n_tracks', 'n_unmatched', 'true_unmatched_units', 'correctly_unmatched_units')
    total = {k: sum(e[k] for e in evaluations.values()) for k in keys}
    if total['n_unknown_predicted_pairs']:
        raise ValueError('Unknown cross-recording identities cannot determine an empirical error target')
    n = total['true_positive_pairs'] + total['false_positive_pairs']
    total.update(assigned_pairs=n, empirical_false_match_fraction=total['false_positive_pairs'] / n if n else None,
        pair_recall=total['true_positive_pairs'] / total['n_known_positive_pairs'] if total['n_known_positive_pairs'] else None)
    return dict(aggregate=total, per_original_recording=evaluations)


def select_operating_point(dataset, scores, *, input_manifest_sha256, target_fdr,
                           contradiction_threshold=.1, ambiguity_margin=.02):
    """Search calibration scores only; test identities never enter selection."""
    if not np.isfinite(target_fdr) or not 0 <= target_fdr < 1:
        raise ValueError('target_fdr must lie in [0, 1)')
    if not 0 <= contradiction_threshold <= 1 or not 0 <= ambiguity_margin <= 1:
        raise ValueError('Tracker settings must lie in [0, 1]')
    _validate_scores(dataset, scores, 'calibration', input_manifest_sha256)
    signature = scoring_signature(dataset, scores)
    # Labels outside calibration are used only for full-source partition guards.
    labels = dataset.labels[dataset.labels.partition == 'calibration'].copy()
    sessions = scores.units.set_index('cellid').session_id
    cross = scores.pairs.cellid_a.map(sessions).values != scores.pairs.cellid_b.map(sessions).values
    thresholds = sorted({float(s) for s in scores.pairs.match_score[cross].dropna() if s >= contradiction_threshold})
    curve = []
    for threshold in thresholds:
        result = _assign_partition(scores, threshold, contradiction_threshold, ambiguity_margin)
        metrics = _evaluate_partition(result, labels, 'calibration')['aggregate']
        curve.append(dict(threshold=threshold, **metrics))
    acceptable = [r for r in curve if r['assigned_pairs'] and r['empirical_false_match_fraction'] <= target_fdr]
    best = max(acceptable, key=lambda r: (r['true_positive_pairs'], -r['false_positive_pairs'], r['threshold'])) if acceptable else None
    payload = dict(status='selected' if best else 'no_qualified_operating_point',
        created_at=datetime.now(timezone.utc).isoformat(), dataset_identity=_dataset_identity(dataset, input_manifest_sha256),
        backend_id=scores.backend_id, scoring_signature=signature, scoring_signature_sha256=_digest(signature),
        calibration_score_digest=result_digest(scores), calibration_units_sha256=_digest(sorted(scores.units.cellid)),
        calibration_source_groups=sorted(labels.well.unique()), calibration_original_recordings=len(labels[['well', 'source_session']].drop_duplicates()),
        target_empirical_false_match_fraction=float(target_fdr), threshold=None if best is None else best['threshold'],
        contradiction_threshold=contradiction_threshold, ambiguity_margin=ambiguity_margin,
        calibration_selection=best, calibration_curve=curve,
        selection_rule='Maximize calibration TP subject to FP/(TP+FP) <= target and nonempty assignments; ties fewer FP, then higher cutoff',
        threshold_candidates='Distinct finite cross-session calibration scores >= contradiction threshold only; same-session rows are ignored',
        compatibility_excluded_population_parameters=list(POPULATION_PARAMETERS),
        excluded_parameter_reason='Native population sizes and unsupervised expected-match count vary with the recording; all other native parameters are retained',
        caveats=list(CAVEATS))
    return dict(format='nems_unit_match_operating_point', schema_version=1, sha256=_digest(payload), selection=payload)


def _validate_selection(document):
    if (not isinstance(document, dict) or document.get('format') != 'nems_unit_match_operating_point'
            or document.get('schema_version') != 1 or document.get('sha256') != _digest(document.get('selection'))):
        raise ValueError('Invalid frozen operating-point artifact or content hash')
    selection = document['selection']
    if selection.get('status') != 'selected' or selection.get('threshold') is None:
        raise ValueError('No qualified nonempty operating point was selected; no test threshold may be invented')
    if selection.get('scoring_signature_sha256') != _digest(selection.get('scoring_signature')):
        raise ValueError('Scoring signature hash mismatch')
    return selection


def _same_data(a, b):
    """Compare retained measured content without serializing waveform arrays."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return np.array_equal(np.asarray(a), np.asarray(b), equal_nan=True)
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_same_data(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)) and isinstance(b, (tuple, list)):
        return len(a) == len(b) and all(_same_data(x, y) for x, y in zip(a, b))
    if isinstance(a, (float, np.floating)) and isinstance(b, (float, np.floating)) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


def _records(table, keys):
    return table.sort_values(keys).reset_index(drop=True).to_dict(orient='records')


def _observation_keys(dataset, manifest_hash):
    """Trace verified collection namespaces to opaque leaf observation IDs.

    No reference source-unit labels participate. Every traversed archive is
    hash-validated and each collection prefix is reconstructed from its actual
    referenced manifest hash; arbitrary colon-prefix stripping is forbidden.
    """
    references = dataset.metadata.get('collection_inputs')
    if not references:
        return {r.cellid: ((manifest_hash, r.session_id, r.cellid,
                    dataset.features[r.cellid]['start_sample'], dataset.features[r.cellid]['stop_sample']),
                    {r.session_id}) for r in dataset.units.itertuples()}
    observed = dataset.units.set_index('cellid')
    result = {}
    for reference in references:
        path = Path(reference['path']).expanduser()
        if not path.is_absolute():
            base = getattr(dataset, '_operating_point_archive_path', None)
            if base is None:
                raise ValueError('Relative collection references require the verified containing archive path')
            path = Path(base) / path
        path = path.resolve()
        if _file_hash(path / 'manifest.json') != reference['manifest_sha256']:
            raise ValueError('Collection source manifest changed during observation correspondence validation')
        source = load_dataset(path)
        source._operating_point_archive_path = path
        child = _observation_keys(source, reference['manifest_sha256'])
        omitted = set(reference.get('omit_cellids', ()))
        prefix = reference['manifest_sha256'][:16]
        for row in source.units.itertuples():
            if row.cellid in omitted:
                continue
            session = prefix + ':' + row.session_id
            # load_collection supports precisely these two declared ID policies;
            # its verified archive loader already enforced the actual policy.
            possibilities = {row.cellid, session + '-' + row.cellid.split('-', 1)[1]}
            found = [cid for cid in possibilities if cid in observed.index and observed.loc[cid, 'session_id'] == session]
            if len(found) > 1:
                raise ValueError('Ambiguous verified collection observation correspondence')
            if found:
                cid = found[0]
                if cid in result or observed.loc[cid, 'recording_group'] != row.recording_group:
                    raise ValueError('Collection observation source domain changed')
                key, aliases = child[row.cellid]
                result[cid] = (key, aliases | {session})
    if set(result) != set(dataset.units.cellid):
        raise ValueError('Could not verify every collection observation namespace')
    return result


def _normalized_candidates(dataset, id_map, session_map, keys):
    pairs = dataset.pairs.copy()
    units = dataset.units.set_index('cellid')
    for endpoint in ('a', 'b'):
        column = 'cellid_' + endpoint
        for session_column in ('siteid_' + endpoint, 'session_id_' + endpoint):
            if session_column in pairs:
                if any(value not in keys[cid][1] for cid, value in zip(pairs[column], pairs[session_column])):
                    raise ValueError('Candidate session column uses an unverified observation namespace')
                pairs[session_column] = [session_map[units.loc[c, 'session_id']] for c in pairs[column]]
        pairs[column] = pairs[column].map(id_map)
    return pairs


def _validate_missing_view(dataset, parent, parent_hash, selection, view_hash=None):
    """Allow only test omissions, resolving namespaces through verified sources.

    Test reference identities are compared only after assignment. Normalization
    here is for provenance validation; saved test scores and assignments retain
    their original view IDs, and are never joined using source-cluster labels.
    """
    _validate_full_dataset(parent)
    if selection['dataset_identity'] != _dataset_identity(parent, parent_hash):
        raise ValueError('Explicit selection parent does not match the frozen artifact')
    old = set(parent.units.cellid)
    parent_keys = _observation_keys(parent, parent_hash)
    if set(dataset.units.cellid) <= old:
        id_map = {c: c for c in dataset.units.cellid}
        keys = {c: parent_keys[c] for c in dataset.units.cellid}
    else:
        if view_hash is None:
            raise ValueError('Renamed collection observations require a verified view manifest hash')
        keys = _observation_keys(dataset, view_hash)
        lookup = {value[0]: cid for cid, value in parent_keys.items()}
        if len(lookup) != len(parent_keys) or any(value[0] not in lookup for value in keys.values()):
            raise ValueError('Missing view does not share identical underlying source observations')
        id_map = {cid: lookup[value[0]] for cid, value in keys.items()}
    if len(set(id_map.values())) != len(id_map):
        raise ValueError('Missing view duplicates an underlying observation')
    new = set(id_map.values())
    retained_non_test = set(parent.labels.loc[parent.labels.partition != 'test', 'cellid'])
    if not new < old or not retained_non_test <= new:
        raise ValueError('A missing-counterpart view may omit test observations only')
    session_map = {}
    parent_units = parent.units.set_index('cellid')
    for row in dataset.units.itertuples():
        original_session = parent_units.loc[id_map[row.cellid], 'session_id']
        if session_map.setdefault(row.session_id, original_session) != original_session:
            raise ValueError('Missing view combines different source sessions')
    for key in ('runclass', 'extraction_options', 'pair_options'):
        if not _same_data(parent.metadata.get(key), dataset.metadata.get(key)):
            raise ValueError('Missing view changed source extraction or candidate settings')
    source_key = lambda row: (row['well'], row['source_session'])
    if not _same_data({source_key(s): s for s in parent.metadata['sources']},
                      {source_key(s): s for s in dataset.metadata['sources']}):
        raise ValueError('Missing view changed original source provenance')
    normalized_units = dataset.units.copy()
    normalized_units['cellid'] = normalized_units.cellid.map(id_map)
    for column in ('session_id', 'siteid'):
        normalized_units[column] = normalized_units[column].map(session_map)
    if not _same_data(_records(parent.units[parent.units.cellid.isin(new)], ['cellid']), _records(normalized_units, ['cellid'])):
        raise ValueError('Missing view changed retained unit/session identities')
    labels = dataset.labels.copy()
    labels['cellid'] = labels.cellid.map(id_map)
    cols = ['cellid', 'source_session', 'well', 'partition']
    if not _same_data(_records(parent.labels[parent.labels.cellid.isin(new)][cols], ['cellid']), _records(labels[cols], ['cellid'])):
        raise ValueError('Missing view changed source/partition mapping')
    if not _same_data(_records(parent.labels[parent.labels.partition != 'test'], ['cellid']),
                      _records(labels[labels.partition != 'test'], ['cellid'])):
        raise ValueError('Missing view changed train/calibration references')
    if any(not _same_data(parent.features[id_map[c]], dataset.features[c]) for c in id_map):
        raise ValueError('Missing view changed retained measured features')
    if any(not _same_data(parent.geometry[session_map[s]], g) for s, g in dataset.geometry.items()):
        raise ValueError('Missing view changed measured channel geometry')
    old_pairs = _normalized_candidates(parent, {c: c for c in old},
        {s: s for s in parent.units.session_id}, parent_keys)
    old_pairs = old_pairs[old_pairs.cellid_a.isin(new) & old_pairs.cellid_b.isin(new)]
    new_pairs = _normalized_candidates(dataset, id_map, session_map, keys)
    if not _same_data(_records(old_pairs, ['cellid_a', 'cellid_b']), _records(new_pairs, ['cellid_a', 'cellid_b'])):
        raise ValueError('Missing view changed retained candidate evidence')
    return id_map


def apply_operating_point(dataset, scores, document, *, input_manifest_sha256,
                          selection_dataset=None, selection_manifest_sha256=None):
    """Apply one immutable cutoff to saved test scores; never search or refit."""
    selection = _validate_selection(document)
    _validate_scores(dataset, scores, 'test', input_manifest_sha256)
    different_input = selection['dataset_identity'] != _dataset_identity(dataset, input_manifest_sha256)
    if different_input:
        if selection_dataset is None or selection_manifest_sha256 is None:
            raise ValueError('Frozen selection belongs to another archive; supply its explicit selection parent for a missing view')
        view_id_map = _validate_missing_view(dataset, selection_dataset, selection_manifest_sha256, selection, input_manifest_sha256)
    if selection['scoring_signature'] != scoring_signature(dataset, scores):
        raise ValueError('Frozen selection and test scoring/model/checkpoint configurations differ')
    result = _assign_partition(scores, selection['threshold'], selection['contradiction_threshold'],
                              selection['ambiguity_margin'], selection_hash=document['sha256'])
    result.provenance = result.provenance.merged(extra=dict(
        target_empirical_false_match_fraction=selection['target_empirical_false_match_fraction'],
        calibration_selection=selection['calibration_selection'],
        calibration_source_groups=selection['calibration_source_groups'],
        calibration_original_recordings=selection['calibration_original_recordings']))
    # First access test source identities for performance evaluation, after assignment.
    if different_input:
        ids = set(view_id_map.values())
        normalized_labels = dataset.labels.copy()
        normalized_labels['cellid'] = normalized_labels.cellid.map(view_id_map)
        if not _same_data(_records(selection_dataset.labels[selection_dataset.labels.cellid.isin(ids)], ['cellid']),
                          _records(normalized_labels, ['cellid'])):
            raise ValueError('Missing view changed source identity references')
        result.provenance = result.provenance.merged(extra=dict(
            selection_parent=selection['dataset_identity'],
            verified_view_correspondence_sha256=_digest(view_id_map),
            test_population_policy='Verified test-only omissions from frozen selection parent; threshold unchanged'))
    evaluation = _evaluate_partition(result, dataset.labels[dataset.labels.partition == 'test'], 'test')
    evaluation['combined'] = evaluate_result(result, dataset.labels, expected_partition='test')
    evaluation.update(selection_sha256=document['sha256'], target_empirical_false_match_fraction=selection['target_empirical_false_match_fraction'],
        calibration_selection=selection['calibration_selection'], caveats=list(CAVEATS))
    return result, evaluation


def _load_archive(path):
    path = Path(path).expanduser().resolve()
    dataset = load_dataset(path)
    dataset._operating_point_archive_path = path
    _validate_full_dataset(dataset)
    return dataset, _file_hash(path / 'manifest.json')


def _new_output(dataset, output, root):
    target = resolve_cache_path(output, root=root) if output else new_run_path(dataset.metadata['dataset_id'], root=root)
    target.mkdir(parents=True, exist_ok=False)
    return target


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(_json_safe(value), stream, indent=2, allow_nan=False)
        stream.write('\n')


def _notes(output, stage, input_path, *, selection=None, evaluation=None):
    p = selection['selection'] if selection else {}
    lines = ['# Calibration-selected matching operating point', '',
        f'Operation: **{stage}**. Matching uses the common constrained tracker; these are not native UnitMatch/DeepUnitMatch track assignments.', '']
    if p:
        lines.extend([f"Selection status: **{p['status']}**. Target empirical false-match fraction: {p['target_empirical_false_match_fraction']:.3g}.",
            f"Frozen threshold: {p['threshold']}. Calibration wells: {len(p['calibration_source_groups'])}; original recordings: {p['calibration_original_recordings']}.", ''])
    metrics = evaluation['aggregate'] if evaluation else p.get('calibration_selection')
    if metrics:
        lines.extend(['| Measurement | Count or value |', '|---|---:|'])
        for key, label in (('true_positive_pairs', 'Correct assigned same-source pairs'), ('false_positive_pairs', 'Incorrect assigned pairs'),
                ('false_negative_pairs', 'Missed same-source pairs'), ('n_known_positive_pairs', 'Known same-source pairs'),
                ('assigned_pairs', 'All assigned pairs'), ('empirical_false_match_fraction', 'Observed incorrect fraction among assigned pairs'),
                ('n_unmatched', 'Reported unmatched units'), ('true_unmatched_units', 'Known unmatched units'), ('correctly_unmatched_units', 'Correctly reported unmatched units')):
            lines.append(f"| {label} | {metrics.get(key) if metrics.get(key) is not None else 'Undefined'} |")
        lines.append('')
    lines.extend(CAVEATS)
    (output / 'report.md').write_text('\n'.join(lines) + '\n')
    (output / 'METHODS.md').write_text(f'''# Matching operating-point analysis — methods

- Output folder: `{output}`
- Generated by: `multiphy/operating_points.py`; operation `{stage}`.
- Input full benchmark archive: `{Path(input_path).resolve()}`.

## Goal

Compare assigned-pair yield after selecting a cutoff using calibration reference identities only. The target is empirical FP/(TP+FP), not a guaranteed future error rate. No figures are produced.

## Data lineage and transforms

- `select` reads hash-validated saved calibration scores and the full archive. Full-well partition validation precedes calibration-only threshold search. It reruns `backends.result_from_pairs()` independently per original recording at each distinct finite calibration score from a cross-session pair at or above the fixed contradiction threshold; same-session score rows add no candidate threshold. Ambiguity and contradiction settings are prespecified. `evaluation.evaluate_result()` counts every implied within-track pair, including unmeasured direct links. Empty predictions have undefined false-match fraction. It maximizes TP under the target, breaking ties by fewer FP and then higher threshold. `selection.json` preserves the complete curve and its content hash; no qualifying point is explicit.
- `apply` verifies that same archive and scoring signature (or an explicitly supplied, hash-bound complete selection parent with identical train/calibration content and exact retained test features/candidate evidence/source provenance, allowing test-cell omissions only; collection namespaces are resolved through hash-verified source archives using opaque observation IDs and temporal windows, never reference neuron identities), applies only the frozen cutoff to saved test scores, and then evaluates test identities. `result.json` is a new common-tracker result; the original native result stays unchanged. `evaluation.json` records counts per original recording and summed counts.
- Compatibility retains checkpoint hashes, upstream source revision and implementation hash, all native scoring parameters except `{', '.join(POPULATION_PARAMETERS)}`, acquisition transforms, geometry, crop/sample representation, and candidate settings. Those three omitted values describe population sizes or unsupervised expected-match estimation. Only the operational `waveform_preprocessing.storage` description is ignored; numerical transforms and unknown preprocessing keys remain checked. Filesystem locations, run IDs, native tracker decisions and native tracker thresholds are not scoring identities. Unknown future scoring parameters remain subject to exact comparison. Shared upstream parameters such as `match_threshold` remain checked because they may also affect native score construction.

## Interpretation

''' + '\n\n'.join(CAVEATS) + '\n')


def run_select(input_path, calibration_scores, *, target_fdr, output_dir=None, root=None,
               contradiction_threshold=.1, ambiguity_margin=.02):
    dataset, digest = _load_archive(input_path)
    scores = load_result(calibration_scores)
    document = select_operating_point(dataset, scores, input_manifest_sha256=digest, target_fdr=target_fdr,
        contradiction_threshold=contradiction_threshold, ambiguity_margin=ambiguity_margin)
    output = _new_output(dataset, output_dir, root)
    _write_json(output / 'selection.json', document)
    _write_json(output / 'run_manifest.json', dict(status='completed', operation='select', input_manifest_sha256=digest,
        input_path=str(Path(input_path).resolve()), calibration_scores=str(Path(calibration_scores).resolve()),
        calibration_scores_sha256=_file_hash(calibration_scores), selection_sha256=document['sha256']))
    _notes(output, 'select', input_path, selection=document)
    return output


def run_apply(input_path, test_scores, selection_path, *, output_dir=None, root=None, selection_input=None):
    dataset, digest = _load_archive(input_path)
    scores = load_result(test_scores)
    document = json.loads(Path(selection_path).read_text())
    parent, parent_hash = _load_archive(selection_input) if selection_input else (None, None)
    result, evaluation = apply_operating_point(dataset, scores, document, input_manifest_sha256=digest,
        selection_dataset=parent, selection_manifest_sha256=parent_hash)
    output = _new_output(dataset, output_dir, root)
    result.provenance = result.provenance.merged(extra=dict(benchmark_input=str(Path(input_path).resolve()),
        selection_path=str(Path(selection_path).resolve()), selection_file_sha256=_file_hash(selection_path)))
    save_result(result, output / 'result.json')
    _write_json(output / 'evaluation.json', evaluation)
    _write_json(output / 'run_manifest.json', dict(status='completed', operation='apply', input_manifest_sha256=digest,
        dataset_id=dataset.metadata['dataset_id'], run_id=result.run_id, backend=result.backend_id, partition='test',
        input_path=str(Path(input_path).resolve()), test_scores=str(Path(test_scores).resolve()),
        test_scores_sha256=_file_hash(test_scores), selection_path=str(Path(selection_path).resolve()),
        selection_sha256=document['sha256'], result_digest=result_digest(result),
        selection_input=str(Path(selection_input).resolve()) if selection_input else None, selection_input_sha256=parent_hash))
    _notes(output, 'apply', input_path, selection=document, evaluation=evaluation)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='operation', required=True)
    for operation in ('select', 'apply'):
        command = commands.add_parser(operation)
        command.add_argument('--input', required=True, help='Full train/calibration/test benchmark archive')
        command.add_argument('--output', help='New cache-only output directory')
        command.add_argument('--scores', required=True, help='Saved result.json for the declared partition')
        if operation == 'select':
            command.add_argument('--target-fdr', type=float, required=True, help='Target calibration FP/(TP+FP); empirical, not guaranteed')
            command.add_argument('--contradiction-threshold', type=float, default=.1)
            command.add_argument('--ambiguity-margin', type=float, default=.02)
        else:
            command.add_argument('--selection', required=True, help='Previously frozen selection.json')
            command.add_argument('--selection-input', help='Explicit complete archive used for selection; required for a test-only missing-counterpart view')
    args = parser.parse_args(argv)
    if args.operation == 'select':
        output = run_select(args.input, args.scores, target_fdr=args.target_fdr, output_dir=args.output,
            contradiction_threshold=args.contradiction_threshold, ambiguity_margin=args.ambiguity_margin)
    else:
        output = run_apply(args.input, args.scores, args.selection, output_dir=args.output, selection_input=args.selection_input)
    print(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
