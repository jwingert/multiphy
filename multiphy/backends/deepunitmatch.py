"""Backend ``deepunitmatch``: the official DeepUnitMatch network with UnitMatch Bayes and tracking.

Requires an explicit trained checkpoint whose declared probe type matches the waveform
provenance; the bundled model is NP2 four-shank only and is never silently applied to
another probe. Also stages and runs the official two-stage training recipe.
"""
from __future__ import annotations
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from ..contract.backend import (BackendSpec, Option, Provides, COMMON_TRACKER_OPTIONS,
                                NATIVE_TRACKING_OPTIONS)
from ..contract.errors import BackendUnavailable, ContractViolation, MissingInputs
from ..contract.inputs import Input
from ..contract.results import ScoreKind, TrackKind
from ._upstream import (DEEPUNITMATCH_PAPER_REVISION, UNITMATCH_REVISION, UNITMATCH_SOURCE, availability,
                        deep_module as _deep_module, module as _module, prepare_bundle as _prepare_bundle,
                        source_info as _source_info)
from .unitmatch import _um_geometry, _um_result, _um_setup
from ._upstream import recording as _recording
from .unitmatch import _DRIFT_APPLY, _finite, _read


def _drift_correct_pair(mf, labels, sub_switch, cent, trajectory, param) -> tuple:
    """Run ``mf.drift_correct_session_pair`` and return ``(trajectory, drift)``.

    ``drift`` is ``None`` when UnitMatchPy applied no correction to the pair; else the depth and
    x drift it added to the pair's later day, one value per shank where it corrected per shank
    (with each later-day unit's shank). A capture that cannot be read raises ``ContractViolation``.
    """
    with _recording(mf, 'apply_drift_correction_basic', 'apply_drift_correction_per_shank',
                    'shank_ID_per_session') as calls:
        trajectory = mf.drift_correct_session_pair(labels.astype(bool), sub_switch, cent, trajectory, 0, param)
    applied = [c for c in calls if c.name in _DRIFT_APPLY]
    if len(applied) > 1:
        raise ContractViolation(f'drift_correct_session_pair applied {len(applied)} drift corrections; '
                                'expected at most one')
    if not applied:
        return trajectory, None
    call = applied[0]
    n_matches = _read(call.name, 'args[0] (matched pairs)', lambda: len(call.args[0]))
    if call.name == 'apply_drift_correction_basic':
        drift = _finite(_read(call.name, 'result[0] (drift)', lambda: call.result[0]), (3,), call.name)
        return trajectory, {'depth_um': float(drift[2]), 'x_um': float(drift[1]), 'n_matches': n_matches}
    n_shanks = _read('UnitMatchPy param', "'no_shanks'", lambda: int(param['no_shanks']))
    drift = _finite(_read(call.name, 'result[0] (shank drifts)', lambda: call.result[0]),
                    (n_shanks, 3), call.name)
    ids = [c for c in calls if c.name == 'shank_ID_per_session' and c.parent == 'apply_drift_correction_per_shank']
    if len(ids) != 1:
        raise ContractViolation(f'shank_ID_per_session was called {len(ids)} times by '
                                'apply_drift_correction_per_shank; expected exactly one')
    shank_ids = _read('shank_ID_per_session', 'result (shank ids)', lambda: np.asarray(ids[0].result, float))
    size = _read('drift_correct_session_pair', 'session_switch', lambda: int(sub_switch[2] - sub_switch[1]))
    if (shank_ids.shape != (size,) or not np.isfinite(shank_ids).all()
            or not np.array_equal(shank_ids, np.round(shank_ids))
            or not ((shank_ids >= 0) & (shank_ids < n_shanks)).all()):
        raise ContractViolation(f'shank_ID_per_session returned {shank_ids.shape} values, expected {size} '
                                f'shank numbers in range({n_shanks})')
    return trajectory, {'shank_depth_um': [float(v) for v in drift[:, 2]],
                        'shank_x_um': [float(v) for v in drift[:, 1]], 'n_matches': n_matches,
                        'shanks': [int(s) for s in shank_ids]}


def _pair_registration(pair_drifts: list, unit_shanks: dict, shank_width_um: float, group: str,
                       sources: list) -> dict:
    """The registration of a deepunitmatch run: each day pair's own drift, added to its later day."""
    return {'method': 'native', 'offsets_um': {}, 'pair_drifts': pair_drifts, 'unit_shanks': unit_shanks,
            'shank_width_um': shank_width_um, 'diagnostics': {group: {'sources': sources}}}


def create_deepunitmatch_model(*, autoencoder=False, device='cpu'):
    """Instantiate upstream architecture, float64, without loading a checkpoint.

    Exact February-2026 reproduction requires the paper-era source revision
    DEEPUNITMATCH_PAPER_REVISION; later upstream revisions changed the model.
    No bundled checkpoint is silently applied to another probe type.
    """
    models = _deep_module('utils.mymodel')
    factory = models.SpatioTemporalAutoEncoder_V2 if autoencoder else models.SpatioTemporalCNN_V2
    return factory(n_channel=30, n_time=60, n_output=256).to(device).double()


def _deep_preprocessing(waves, geometry, probe_type, parameters=None):
    """Resolve the official adapter settings from measured geometry, not NP2 defaults."""
    if not probe_type:
        raise ValueError('DeepUnitMatch requires an explicit acquisition probe_type')
    x = np.unique(geometry[0][:, 0])
    n_shanks = 1 + np.sum(np.diff(x) > 50.)
    if len(x) % n_shanks:
        raise ValueError('Probe columns do not divide consistently across measured shanks')
    columns = len(x) // n_shanks
    if str(probe_type) == 'NP1010' and (columns != 4 or n_shanks != 1):
        raise ValueError('NP1010 requires measured four-column, single-shank geometry')
    result = dict(nTime=waves.shape[1], spike_width=waves.shape[1], nChannels=384,
                  ChannelRadius=110, RnChannels=30, RnTime=60, n_xchannelpos=int(columns))
    result.update(parameters or {})
    if (result['n_xchannelpos'] != columns or result['RnChannels'] != 30
            or result['RnTime'] != 60 or result['nChannels'] != 384
            or result['spike_width'] != waves.shape[1] or result['nTime'] != waves.shape[1]):
        raise ValueError('DeepUnitMatch preprocessing conflicts with measured geometry or trained architecture')
    return result


def load_deepunitmatch_checkpoint(checkpoint, *, device='cpu'):
    """Load every encoder parameter strictly; never leave random missing weights."""
    torch = _module('torch')
    saved = torch.load(str(checkpoint), map_location=device, weights_only=True)
    if 'clip_loss' not in saved or 'model' not in saved:
        raise ValueError('DeepUnitMatch matching requires a completed contrastive-stage checkpoint')
    model = create_deepunitmatch_model(device=device)
    model.load_state_dict(saved['model'], strict=True)
    model.eval()
    return model


def _deep_input_signature(bundle):
    """Record shared voltage transforms independently of raw-file names and wells."""
    active = set(bundle.provenance.get('source_groups', {}).values())
    signatures = set()
    for source in bundle.provenance.get('sources', []):
        if source.get('well') not in active:
            continue
        provenance = source.get('provenance', {})
        processing = dict(provenance.get('waveform_preprocessing') or {})
        # This one field describes disk versus in-memory accumulation. Every
        # scientific field, including new future transforms, remains checked.
        processing.pop('storage', None)
        signature = dict(processing, acquisition_voltage_scale=provenance.get('voltage_scale'),
            legacy_common_median=provenance.get('common_median'),
            legacy_baseline_samples=provenance.get('baseline_samples'))
        signatures.add(json.dumps(signature, sort_keys=True))
    if len(signatures) > 1:
        raise ValueError('DeepUnitMatch source waveforms use inconsistent voltage preprocessing')
    return {'shared_voltage_preprocessing': json.loads(next(iter(signatures))) if signatures else None,
            'snippet_before': bundle.provenance.get('extraction_options', {}).get('before'),
            'snippet_after': bundle.provenance.get('extraction_options', {}).get('after')}


def _deep_training_metadata(checkpoint, bundle, preprocessing):
    """Validate our training manifest when accompanying a laboratory checkpoint."""
    for parent in Path(checkpoint).resolve().parents:
        manifest = parent / 'training_manifest.json'
        if not manifest.is_file():
            continue
        recipe = json.loads(manifest.read_text())
        if recipe.get('status') != 'completed':
            raise ValueError('Checkpoint training manifest is not completed')
        if recipe.get('probe_type') != bundle.provenance.get('probe_type'):
            raise ValueError('Training manifest probe_type differs from inference acquisition')
        if recipe.get('sample_rate') != bundle.provenance.get('sample_rate'):
            raise ValueError('Checkpoint and inference waveforms have different sample rates')
        if recipe['preprocessing'] != preprocessing:
            raise ValueError('Checkpoint and inference DeepUnitMatch preprocessing differ')
        if recipe.get('input_signature') != _deep_input_signature(bundle):
            raise ValueError('Checkpoint and inference shared waveform preprocessing or temporal crop differ')
        trained_hash = recipe['source_sha256']['utils/mymodel.py']
        if _source_info(_deep_module('utils.mymodel'))['sha256'] != trained_hash:
            raise ValueError('Installed DeepUnitMatch architecture source differs from training source')
        if recipe.get('checkpoint_sha256') != hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest():
            raise ValueError('Checkpoint hash differs from completed training manifest')
        active = set(bundle.provenance.get('source_groups', {}).values())
        if active & set(recipe['training_groups']):
            raise ValueError('Inference source groups overlap model training groups; use held-out wells')
        return {'training_manifest': str(manifest), 'training_seed': recipe['seed'],
                'training_source_revision': recipe['source_revision'],
                'training_groups': recipe['training_groups'],
                'evaluation_groups': recipe['evaluation_groups']}
    return {'training_manifest': None, 'training_groups_verified': False}


def run_deepunitmatch(units, pairs=None, *, waveform_bundle=None, checkpoint=None,
                     checkpoint_probe=None, params=None, device='cpu',
                     tracking='native_intermediate', native_threshold=None,
                     provenance=None, **options):
    """Official preprocessing/network/Bayes and explicit native/common tracking."""
    eligible, _, _, waves, geometry, sid, switch = _prepare_bundle(units, waveform_bundle)
    if checkpoint is None or not Path(checkpoint).is_file() or not checkpoint_probe:
        raise BackendUnavailable('DeepUnitMatch requires an explicit trained checkpoint and '
                                 'checkpoint_probe; the bundled model is NP2 four-shank only')
    if waveform_bundle.provenance.get('probe_type') != checkpoint_probe:
        raise ValueError('Checkpoint probe type must match waveform provenance probe_type')
    if waves.shape[2] != 384 or waves.shape[1] < 60:
        raise ValueError('Verified DeepUnitMatch preprocessing requires 384 channels and at least 60 samples')
    if any(not np.array_equal(g, geometry[0]) for g in geometry[1:]):
        raise ValueError('Official DeepUnitMatch preprocessing assumes common channel geometry')
    test, prep = _deep_module('testing.test'), _deep_module('utils.param_fun')
    helpers, losses = _deep_module('utils.helpers'), _deep_module('utils.losses')
    torch = _module('torch')
    ov, bf, mf = (_module('UnitMatchPy.' + m) for m in
                  ('overlord', 'bayes_functions', 'metric_functions'))
    param, info, _ = _um_setup(waves, geometry, sid, switch, params)
    preprocessing = _deep_preprocessing(waves, geometry, checkpoint_probe,
        {k: v for k, v in (params or {}).items() if k in
         ('nTime', 'spike_width', 'nChannels', 'ChannelRadius', 'RnChannels', 'RnTime', 'n_xchannelpos')})
    training_metadata = _deep_training_metadata(checkpoint, waveform_bundle, preprocessing)
    with tempfile.TemporaryDirectory(prefix='nems-deepunitmatch-') as root:
        snippets, _, kept = prep.get_snippets(waves, geometry, sid, save_path=root,
                                             unit_ids=np.arange(len(waves)), param=preprocessing)
        if not np.array_equal(kept, np.arange(len(waves))):
            raise ValueError('Official preprocessing rejected units; inspect waveform shape/geometry before matching')
        snippets = np.asarray(snippets)
        model = load_deepunitmatch_checkpoint(checkpoint, device=device)
        # Preserve the explicit bundle ordering, avoiding directory-list order.
        with torch.no_grad():
            first = model(torch.as_tensor(snippets[..., 0], dtype=torch.float64, device=device))
            second = model(torch.as_tensor(snippets[..., 1], dtype=torch.float64, device=device))
            sim = losses.clip_sim(first, second).detach().cpu().numpy()
        extracted = ov.extract_parameters(waves, _um_geometry(geometry), info, param)
        probabilities = np.eye(len(waves))
        sessions = list(dict.fromkeys(eligible.session_id))
        ids = [str(c) for c in eligible.cellid]
        pair_drifts, unit_shanks = [], {}
        for r1 in range(len(geometry)):
            for r2 in range(r1 + 1, len(geometry)):
                ix = np.flatnonzero(np.isin(sid, [r1, r2]))
                sub_sid, sub_sim = sid[ix], sim[np.ix_(ix, ix)]
                good = [np.flatnonzero(sid == r1), np.flatnonzero(sid == r2)]
                frame = helpers.create_dataframe(good, sub_sim, session_list=[r1, r2])
                matches = test.get_matches(frame, sub_sim, sub_sid,
                                           str(Path(root) / 'processed_waveforms'), dist_thresh=50)
                labels = np.eye(len(ix))
                for (s1, s2), group in matches.groupby(['RecSes1', 'RecSes2']):
                    labels[np.ix_(sub_sid == s1, sub_sid == s2)] = group['match'].to_numpy().reshape(
                        np.sum(sub_sid == s1), np.sum(sub_sid == s2))
                sub_switch = np.array([0, len(good[0]), len(ix)])
                cent = extracted['avg_centroid'][:, ix, :].copy()
                trajectory = extracted['avg_waveform_per_tp'][:, ix, :, :].copy()
                trajectory, drift = _drift_correct_pair(mf, labels, sub_switch, cent, trajectory, param)
                if drift is None:
                    pair_drifts.append({'session_a': sessions[r1], 'session_b': sessions[r2],
                                        'depth_um': None, 'x_um': None, 'n_matches': 0})
                else:
                    shanks = drift.pop('shanks', None)
                    if shanks is not None:
                        for j, shank in zip(ix[sub_switch[1]:], shanks):
                            if unit_shanks.setdefault(ids[j], shank) != shank:
                                raise ContractViolation(f'{ids[j]} was put on shank {unit_shanks[ids[j]]} '
                                                        f'and on shank {shank} by two day pairs')
                    pair_drifts.append(dict(session_a=sessions[r1], session_b=sessions[r2], **drift))
                distance = mf.get_Euclidean_dist(mf.flip_dim(trajectory, param, len(ix)), param, len(ix))
                centroid, _ = mf.centroid_metrics(distance, param)
                scores = {'similarity': sub_sim, 'distance': centroid}
                conditions = np.unique(labels)
                if not np.array_equal(conditions, [0, 1]) or len(ix) <= 2:
                    raise ValueError('DeepUnitMatch Bayes reference classes require more usable units')
                kernels = bf.get_parameter_kernels(scores, labels, conditions, param)
                predictors = np.stack(list(scores.values()), axis=2)
                prob = bf.apply_naive_bayes(kernels, np.array([1 - 2/len(ix), 2/len(ix)]),
                                           predictors, param, conditions)[:, 1].reshape(len(ix), -1)
                probabilities[np.ix_(ix, ix)] = prob
    metadata = dict(provenance or {})
    metadata.update(official_source=UNITMATCH_SOURCE, verified_api_revision=UNITMATCH_REVISION,
                    implementation=_source_info(test), checkpoint=str(Path(checkpoint).resolve()),
                    checkpoint_sha256=hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest(),
                    checkpoint_probe=checkpoint_probe, waveform_provenance=waveform_bundle.provenance,
                    network_preprocessing=preprocessing, training=training_metadata,
                    parameters=param, pair_reduction='minimum of both directional probabilities',
                    calibration_status='official Bayes outputs; not validated identity accuracy here')
    metadata.update(used_optional_inputs=[])
    metadata.update(registration=_pair_registration(
        pair_drifts, unit_shanks, float(param['shank_dist']), str((provenance or {}).get('group', '')),
        [_source_info(mf)]))
    return _um_result('deepunitmatch', units, eligible, probabilities, sid, param,
                      info, tracking, metadata, dict(options, native_threshold=native_threshold))


_TRAINING_WORKER = r'''
import json, pathlib, sys, random, hashlib
import numpy as np
import torch
root = pathlib.Path(sys.argv[1]).resolve()
recipe = json.loads((root / 'training_manifest.json').read_text())
source = root / 'official_source' / 'DeepUnitMatch'
sys.path.insert(0, str(source.parent))
sys.path.insert(0, str(source))
from DeepUnitMatch.utils import param_fun, AE_npdataset, npdataset
from DeepUnitMatch.train.train_AE import run_training
from DeepUnitMatch.train.train_finetune import run_finetune
random.seed(recipe['seed'])
np.random.seed(recipe['seed'])
torch.manual_seed(recipe['seed'])
torch.set_num_threads(recipe['cpu_threads'])
torch.use_deterministic_algorithms(True)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(recipe['seed'])
    torch.backends.cudnn.benchmark = False
recipe['execution'] = dict(torch_version=torch.__version__, numpy_version=np.__version__,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    device_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
    deterministic_algorithms=True, data_loader_workers=4)
(root / 'training_manifest.json').write_text(json.dumps(recipe, indent=2))
with np.load(root / 'training_waveforms.npz', allow_pickle=False) as arrays:
    waveforms, session_id = arrays['waveforms'], arrays['session_id']
    geometries = [arrays['geometry']] * len(np.unique(session_id))
    snippets, positions, kept = param_fun.get_snippets(
        waveforms, geometries, session_id, save_path=str(root / 'data'),
        unit_ids=np.arange(len(waveforms)), param=recipe['preprocessing'])
if not np.array_equal(kept, np.arange(len(waveforms))):
    raise ValueError('Official preprocessing rejected training units; inspect inputs before training')
dataset = AE_npdataset.AE_NeuropixelsDataset(str(root / 'data'), batch_size=32)
dataset.np_file_names.sort()
run_training('training', dataset, lr=1e-5, save_freq=recipe['autoencoder_save_freq'],
             total_epoch=recipe['autoencoder_epochs'], cont=False,
             batchsize=32, launch_tensorboard=False)
dataset = npdataset.NeuropixelsDataset(str(root / 'data'), batch_size=32, mode='train')
ordered = sorted(dataset.experiment_unit_map.values(), key=lambda files: str(pathlib.Path(files[0]).parent))
dataset.experiment_unit_map = {i: sorted(files) for i, files in enumerate(ordered)}
dataset.all_files = [(i, file) for i, files in dataset.experiment_unit_map.items() for file in files]
run_finetune('training', dataset, lr_enc=2e-5, lr_proj=1.1e-4,
             save_freq=recipe['contrastive_save_freq'], total_epoch=recipe['contrastive_epochs'],
             cont=False, batchsize=32, launch_tensorboard=False)
checkpoint = source / 'ModelExp' / 'experiments' / 'training' / 'ckpt' / (
    'ckpt_epoch_' + str(recipe['contrastive_epochs'] - 1))
if not checkpoint.is_file():
    raise RuntimeError('Official trainer did not produce the expected final checkpoint')
ae_folder = source / 'ModelExp' / 'AE_experiments' / 'training' / 'ckpt'
ae_final = ae_folder / ('ckpt_epoch_' + str(recipe['autoencoder_epochs'] - 1))
if not ae_final.is_file():
    raise RuntimeError('Official trainer did not produce the expected final autoencoder checkpoint')
for folder, retained in ((ae_folder, ae_final), (checkpoint.parent, checkpoint)):
    for previous in folder.glob('ckpt_epoch_*'):
        if previous != retained:
            previous.unlink()
recipe['autoencoder_checkpoint'] = str(ae_final)
recipe['checkpoint'] = str(checkpoint)
recipe['checkpoint_sha256'] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
recipe['status'] = 'completed'
(root / 'training_manifest.json').write_text(json.dumps(recipe, indent=2))
'''


def prepare_deepunitmatch_training(units, waveform_bundle, *, source_checkout,
                                  output_dir, training_groups, evaluation_groups,
                                  autoencoder_epochs=300, contrastive_epochs=50,
                                  seed=0, cpu_threads=4, preprocessing=None):
    """Stage an auditable official two-stage training run, without starting it.

    The caller chooses output_dir. Copy only upstream Python source there because
    upstream trainers write checkpoints relative to their own module location.
    Training follows the verified *current upstream notebook*, not an assertion
    of exact reproduction of the February preprint. The architecture changed
    after that preprint. Evaluation groups must be held out by the caller before
    this function; upstream random validation splitting is not our benchmark.
    """
    eligible, _, _, waves, geometry, sid, _ = _prepare_bundle(units, waveform_bundle)
    train_groups, eval_groups = set(training_groups), set(evaluation_groups)
    if not train_groups or not eval_groups or train_groups & eval_groups:
        raise ValueError('Supply nonempty disjoint training and held-out evaluation groups')
    source_groups = waveform_bundle.provenance.get('source_groups', {})
    if not set(eligible.session_id) <= set(source_groups):
        raise ValueError('Waveform provenance source_groups must map every pseudo-session to '
                         'its original recording/well holdout group')
    if {source_groups[s] for s in eligible.session_id} != train_groups:
        raise ValueError('Training groups must exactly identify original recording/well groups; '
                         'temporal pieces of one source cannot cross the evaluation boundary')
    if min(autoencoder_epochs, contrastive_epochs, cpu_threads) < 1:
        raise ValueError('Epoch counts and cpu_threads must be positive')
    if any(int(value) != value for value in (autoencoder_epochs, contrastive_epochs, cpu_threads, seed)) or not 0 <= seed < 2 ** 32:
        raise ValueError('Epoch counts, cpu_threads and seed must be integers; seed must fit uint32')
    if waves.shape[2] != 384 or waves.shape[1] < 60:
        raise ValueError('Verified preprocessing requires 384-channel, at least 60-sample raw waveforms')
    if any(not np.array_equal(g, geometry[0]) for g in geometry[1:]):
        raise ValueError('Official preprocessing assumes common channel geometry')
    if min(np.bincount(sid)) < 32:
        raise ValueError('This integration requires at least 32 single units per training session; '
                         'upstream otherwise pads contrastive batches by resampling units')
    prep = _deep_preprocessing(waves, geometry, waveform_bundle.provenance.get('probe_type'), preprocessing)
    input_signature = _deep_input_signature(waveform_bundle)
    sample_rate = waveform_bundle.provenance.get('sample_rate')
    if sample_rate is None or not np.isfinite(sample_rate) or sample_rate <= 0:
        raise ValueError('Training requires a uniform positive waveform provenance sample_rate')
    checkout = Path(source_checkout).resolve()
    revision = subprocess.run(['git', '-C', str(checkout), 'rev-parse', 'HEAD'],
                              capture_output=True, text=True, check=True).stdout.strip()
    if revision != UNITMATCH_REVISION:
        raise ValueError(f'Training wrapper is verified for source revision {UNITMATCH_REVISION}; got {revision}')
    source = checkout / 'UnitMatchPy' / 'DeepUnitMatch'
    if not source.is_dir():
        raise ValueError('source_checkout must be the official UnitMatch repository root')
    dirty = subprocess.run(['git', '-C', str(checkout), 'status', '--porcelain', '--',
                            'UnitMatchPy/DeepUnitMatch'], capture_output=True, text=True,
                           check=True).stdout
    if dirty.strip():
        raise ValueError('Official DeepUnitMatch source has local changes; use a clean pinned checkout')
    target = Path(output_dir).resolve()
    if target.exists():
        raise FileExistsError('Choose a new output directory; training preparation never overwrites a run')
    target.mkdir(parents=True)
    copied = target / 'official_source' / 'DeepUnitMatch'
    for path in source.rglob('*.py'):
        dest = copied / path.relative_to(source)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
    shutil.copy2(checkout / 'UnitMatchPy' / 'LICENSE', copied.parent / 'LICENSE')
    np.savez(target / 'training_waveforms.npz', waveforms=waves, session_id=sid,
             geometry=geometry[0])
    recipe = dict(status='prepared_not_started', source=UNITMATCH_SOURCE,
        source_revision=revision, paper_era_revision=DEEPUNITMATCH_PAPER_REVISION,
        exact_paper_reproduction=False, protocol='verified upstream DUM_training.ipynb',
        architecture='SpatioTemporalAutoEncoder_V2 / SpatioTemporalCNN_V2 (30,60,256)',
        autoencoder_epochs=int(autoencoder_epochs), contrastive_epochs=int(contrastive_epochs),
        autoencoder_save_freq=max(1, int(autoencoder_epochs) - 1),
        contrastive_save_freq=max(1, int(contrastive_epochs) - 1),
        checkpoint_retention='final autoencoder and final contrastive; epoch-zero files removed after success',
        autoencoder_learning_rate=1e-5, encoder_learning_rate=2e-5,
        projection_learning_rate=1.1e-4, batch_size=32, seed=int(seed),
        cpu_threads=int(cpu_threads), preprocessing=prep,
        probe_type=waveform_bundle.provenance['probe_type'], sample_rate=float(sample_rate),
        input_signature=input_signature,
        training_waveforms_sha256=hashlib.sha256((target / 'training_waveforms.npz').read_bytes()).hexdigest(),
        sampler_note='Official final partial contrastive batches are filled by within-session resampling',
        training_cellids=list(eligible.cellid), training_groups=sorted(train_groups),
        training_sessions=list(dict.fromkeys(eligible.session_id)), source_groups=source_groups,
        evaluation_groups=sorted(eval_groups), waveform_provenance=waveform_bundle.provenance,
        validation_note='Upstream random validation is diagnostic only; evaluation groups are separately held out',
        source_sha256={str(p.relative_to(copied)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in copied.rglob('*.py')})
    (target / 'training_manifest.json').write_text(json.dumps(recipe, indent=2, default=str))
    (target / 'METHODS.md').write_text(
        '# DeepUnitMatch training — figure & methods notes\n\n'
        f'- **Output folder:** `{target}`\n'
        '- **Artifacts:** `training_waveforms.npz`, `training_manifest.json`, '
        '`data/processed_waveforms/`, `official_source/DeepUnitMatch/ModelExp/`\n'
        '- **Generated by:** `multiphy/external_backends.py` · '
        f'**Env:** `{sys.executable}` · **Updated:** `2026-09-18`\n\n'
        '## Goal\n\n'
        'Train the official waveform representation using single units and independent temporal '
        'halves of their raw multichannel spike waveforms. Evaluate identity matching later on '
        'held-out original recording/well groups, keeping temporal pieces of one source together. '
        'The manifest records whether preparation, training, or completion has occurred.\n\n'
        '## Data lineage\n\n'
        '- `training_waveforms.npz` ← `prepare_deepunitmatch_training()` in '
        '`multiphy/external_backends.py`; single-unit selection and session ordering '
        'from caller-supplied raw means (raw source and extraction/split method in manifest).\n'
        '- `data/processed_waveforms/` ← official `utils/param_fun.py:get_snippets()`; '
        'crop centrally to 60 samples, apply the upstream linear detrend across the channel axis, '
        'select/sort channels within 110 micrometers by measured probe columns, subtract their global '
        'mean, and pad to 30 channels. The shared unwhitened waveform preparation is recorded in '
        'waveform provenance; these additional network transforms preserve the official implementation.\n'
        '- Autoencoder checkpoints ← official `train/train_AE.py:run_training()`; '
        'mean squared reconstruction loss with Adam, learning rate 1e-5, batch size 32.\n'
        '- Contrastive checkpoints ← official `train/train_finetune.py:run_finetune()`; '
        'paired temporal halves, frozen convolution layers, encoder fully connected layers '
        'at 2e-5 and projection/loss parameters at 1.1e-4. Epoch counts are in the manifest. '
        'Official save_freq writes epoch zero and the final epoch; after successful completion '
        'only the final checkpoint from each stage is retained. Saving frequency does not change '
        'the training updates and uses no held-out identity labels.\n\n'
        '## Panels\n\nNo figures are generated by this wrapper. Training losses are logged by '
        'the upstream trainers; they are optimization diagnostics, not identity error rates.\n\n'
        '## Caveats\n\n'
        'This is the verified current upstream protocol, not a claim of exact reproduction of '
        'the February 2026 preprint. Upstream random validation is not an independent identity '
        'benchmark. The original augmentation shifts alternating channel slots by two; for the '
        'four-column NP1010 ordering this does not preserve physical column identity. It is retained '
        'as part of the official training recipe and requires held-out evaluation on this probe. '
        'The integration requires at least 32 units per session; the upstream sampler still resamples '
        'the final partial batch. The upstream network positional-bias module receives no explicit '
        'positions in this notebook training path. '
        'Source revision, source hashes, settings, training source groups and held-out '
        'evaluation groups are recorded in training_manifest.json.\n')
    return target


def train_deepunitmatch(prepared_dir, *, python_executable=None):
    """Explicitly execute a prepared run, synchronously, logging inside that run."""
    root = Path(prepared_dir).resolve()
    manifest = root / 'training_manifest.json'
    recipe = json.loads(manifest.read_text())
    if recipe['status'] != 'prepared_not_started':
        raise ValueError('Training run has already started; prepare a new run instead of overwriting')
    source = root / 'official_source' / 'DeepUnitMatch'
    if hashlib.sha256((root / 'training_waveforms.npz').read_bytes()).hexdigest() != recipe['training_waveforms_sha256']:
        raise ValueError('Prepared training waveforms changed since staging')
    for rel, expected in recipe['source_sha256'].items():
        if hashlib.sha256((source / rel).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Prepared official training source changed: {rel}')
    recipe['status'] = 'running'
    manifest.write_text(json.dumps(recipe, indent=2))
    try:
        import os
        environment = dict(os.environ, PYTHONHASHSEED=str(recipe['seed']),
                           CUBLAS_WORKSPACE_CONFIG=':4096:8', PYTHONDONTWRITEBYTECODE='1')
        with (root / 'training.log').open('x') as log:
            subprocess.run([python_executable or sys.executable, '-u', '-c', _TRAINING_WORKER, str(root)],
                           cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True, env=environment)
    except Exception:
        recipe['status'] = 'failed; inspect training.log'
        manifest.write_text(json.dumps(recipe, indent=2))
        raise
    return json.loads(manifest.read_text())


SPEC = BackendSpec('deepunitmatch', '1',
    requires=frozenset({Input.UNITS, Input.WAVEFORM_HALVES, Input.GEOMETRY, Input.CHECKPOINT}), optional=frozenset(),
    provides=Provides(frozenset({ScoreKind.UNCALIBRATED}), frozenset({TrackKind.NATIVE, TrackKind.COMMON}),
                      day_offsets=True),
    options=(Option('checkpoint', 'path', None, 'Trained checkpoint file', required=True),
             Option('checkpoint_probe', 'str', None, 'Probe type the checkpoint was trained for', required=True),
             Option('device', 'str', 'cpu', 'torch device'),
             Option('params', 'json', None, 'Overrides for UnitMatchPy default parameters'),
             *NATIVE_TRACKING_OPTIONS, *COMMON_TRACKER_OPTIONS), builtin=True)


class DeepUnitMatchBackend:
    spec = SPEC

    def available(self):
        return availability('DeepUnitMatch')

    def run(self, inp, options, ctx):
        avail = self.available()
        if not avail.ok:
            raise BackendUnavailable(avail.reason)
        ref = inp.checkpoint
        if ref is None:
            raise MissingInputs(SPEC.id, inp.group, {Input.CHECKPOINT})
        if ref.probe_type != inp.waveform_provenance.get('probe_type'):
            raise ContractViolation(f'checkpoint probe type {ref.probe_type!r} does not match waveform probe type '
                                    f'{inp.waveform_provenance.get("probe_type")!r}')
        opts = {**SPEC.defaults(), **options}
        tracker = {k: opts[k] for k in ('threshold', 'contradiction_threshold', 'ambiguity_margin')}
        return run_deepunitmatch(inp.units, inp.candidate_pairs, waveform_bundle=inp.bundle(), checkpoint=ref.path,
                                 checkpoint_probe=ref.probe_type, params=opts['params'], device=opts['device'],
                                 tracking=opts['tracking'], native_threshold=opts['native_threshold'],
                                 provenance=ctx.provenance.to_json(), run_id=ctx.run_id, **tracker)
