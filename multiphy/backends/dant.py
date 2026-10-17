"""Backend ``dant``: official pyDANT native clustering in a disposable directory.

Verified upstream API: jiumao2/pyDANT revision 741240511e6a4ed9315b2c80d71ab5cfe4c588eb.
DANT keeps its native cluster membership and emits an unbounded similarity, never a match
probability. Also holds the named settings presets (adaptations of the upstream
settings.json example, not parameters optimized against benchmark labels) and the narrow
empty drift-reference guard (hold the current motion estimate when no pair passes native
selection; no cluster or match labels are changed).
"""
from __future__ import annotations
from pathlib import Path
import copy
import tempfile
import threading
import uuid

import numpy as np
import pandas as pd

from ..contract.backend import BackendSpec, Option, Provides
from ..contract.errors import BackendUnavailable, MissingInputs
from ..contract.inputs import Input
from ..contract.provenance import Provenance
from ..contract.results import Decision, MatchResult, ScoreBlock, ScoreKind, TrackKind
from ._upstream import (DANT_REVISION, DANT_SOURCE, availability, module as _module,
                        prepare_bundle as _prepare_bundle, source_info as _source_info, version as _version)
import contextlib
import math
from typing import Optional

from ..contract.errors import ContractViolation
from ._upstream import recording as _recording
from .native_groups import label_pairs, resolve_groups

SOURCE = 'https://github.com/jiumao2/pyDANT/blob/741240511e6a4ed9315b2c80d71ab5cfe4c588eb/settings.json'
PRESETS = {
    'waveform_acg': 'Waveform and autocorrelogram, without stimulus fingerprint',
    'waveform_acg_stimulus': 'Waveform, autocorrelogram and stimulus fingerprint in motion estimation and final clustering',
    'upstream_example': 'Published repository example: stimulus fingerprint in motion estimation only',
}


def default_dant_settings(preset='waveform_acg', *, n_jobs=1):
    """Return an independent complete config with documented feature selection."""
    if preset not in PRESETS:
        raise ValueError(f'preset must be one of {tuple(PRESETS)}')
    if not isinstance(n_jobs, int) or n_jobs == 0 or n_jobs < -1:
        raise ValueError('n_jobs must be -1 or a positive integer')
    use_peth = preset != 'waveform_acg'
    final_peth = preset == 'waveform_acg_stimulus'
    return {
        'path_to_data': '', 'output_folder': '',
        'save_intermediate_results': True, 'n_jobs': n_jobs,
        'centering_waveforms': False,
        'spikeLocation': {'location_algorithm': 'monopolar_triangulation', 'n_nearest_channels': 20},
        'waveformCorrection': {'n_nearest_channels': 38, 'linear_correction': False,
                               'n_templates': 2, 'path_to_motion': ''},
        'autoCorr': {'window': 300, 'binwidth': 1, 'gaussian_sigma': 5},
        'ISI': {'window': 100, 'binwidth': 1, 'gaussian_sigma': 1},
        'motionEstimation': {'features': [['AutoCorr'] + (['PETH'] if use_peth else []),
                                          ['Waveform', 'AutoCorr'] + (['PETH'] if use_peth else [])],
                             'max_iter': 15, 'repeat_last_feature_set': True, 'stop_early': True},
        'clustering': {'max_distance': 100, 'features': ['Waveform', 'AutoCorr'] + (['PETH'] if final_peth else []),
                       'n_iter': 10, 'weight_tol': 1e-8},
        'autoCuration': {'auto_split': True},
        '_nems_preset': {'name': preset, 'description': PRESETS[preset], 'settings_source': SOURCE,
                         'changes': 'Explicit stimulus feature inclusion/exclusion; n_jobs is a resource setting. Numerical scientific defaults retained.'},
    }


_DANT_RUN_LOCK = threading.RLock()


def run_native_dant(runner, motion_module, config, seed, motion_log: Optional[dict] = None) -> list:
    """Execute the actual pipeline, restoring patched globals and RNG on exit.

    With ``motion_log``, it receives ``initial`` (the motion ``initializeMotion`` returned),
    ``rounds`` (each fitted round's 1-based session ``pairs`` and ``motion``) and ``final`` (the
    motion of the last ``computeWaveformFeatures`` call, ``None`` with no call), as
    ``_motion_record`` dicts. Without it, behaviour is unchanged. Returns the events list.
    """
    events = []
    if motion_log is not None:
        for name in ('computeMotion', 'initializeMotion'):
            if not callable(getattr(motion_module, name, None)):
                raise BackendUnavailable(f'{motion_module.__name__}.{name} is missing; '
                                         'drift offsets cannot be captured')
        motion_log.update(initial=None, rounds=[], final=None)
    with _DANT_RUN_LOCK:
        state = np.random.get_state()
        compute = getattr(motion_module, 'computeMotion', None)
        initialize = getattr(motion_module, 'initializeMotion', None)
        previous_motion = [None]

        def capture_initial(settings, waveforms):
            result = initialize(settings, waveforms)
            previous_motion[0] = copy.deepcopy(result[1])
            if motion_log is not None:
                motion_log['initial'] = _motion_record(result[1])
            return result

        def guard_empty(settings):
            output = Path(settings['output_folder'])
            sessions = np.load(Path(settings['path_to_data']) / 'session_index.npy', allow_pickle=False)
            pair_indices = np.asarray(np.load(output / 'SimilarityPairs.npy', allow_pickle=False), dtype=np.int64).reshape(-1, 2)
            similarity = np.load(output / 'SimilarityMatrix.npy', allow_pickle=False)
            clusters = np.load(output / 'ClusterMatrix.npy', allow_pickle=False)
            threshold = np.load(output / 'SimilarityThreshold.npy', allow_pickle=False)
            accepted = ((similarity[pair_indices[:, 0], pair_indices[:, 1]] > threshold)
                        & (clusters[pair_indices[:, 0], pair_indices[:, 1]] > 0)
                        & (sessions[pair_indices[:, 0]] != sessions[pair_indices[:, 1]]))
            if accepted.any():
                result = compute(settings)
                previous_motion[0] = copy.deepcopy(result)
                if motion_log is not None:
                    motion_log['rounds'].append({
                        'pairs': sorted({tuple(sorted((int(a), int(b))))
                                         for a, b in sessions[pair_indices[accepted]]}),
                        'motion': _motion_record(result)})
                return result
            if previous_motion[0] is None:
                raise RuntimeError('Cannot retain DANT motion without capturing its native initialization')
            result = copy.deepcopy(previous_motion[0])
            result.save(str(output))
            event = {'event': 'no_eligible_drift_pairs', 'action': 'retained_previous_native_motion',
                     'eligible_pairs': 0, 'constant_shift_um': np.asarray(result.Constant).tolist(),
                     'linear_terms': np.asarray(result.Linear).tolist(),
                     'reason': 'Native drift selection supplied no observations; no motion fit or bootstrap was fabricated'}
            events.append(event)
            print('DANT compatibility guard: no native drift pairs; retaining previous motion estimate.', flush=True)
            return result

        patch = callable(compute) and callable(initialize)
        applying = (_recording(motion_module, 'computeWaveformFeatures') if motion_log is not None
                    else contextlib.nullcontext([]))
        try:
            np.random.seed(seed)
            with applying as applied:
                try:
                    if patch:
                        motion_module.computeMotion = guard_empty
                        motion_module.initializeMotion = capture_initial
                    runner.runDANT(config)
                finally:
                    if patch:
                        motion_module.computeMotion = compute
                        motion_module.initializeMotion = initialize
        finally:
            np.random.set_state(state)
        if motion_log is not None and applied:
            last = applied[-1]
            if len(last.args) <= 2 and 'motion' not in last.kwargs:
                raise ContractViolation('computeWaveformFeatures was called without a motion')
            motion_log['final'] = _motion_record(last.args[2] if len(last.args) > 2 else last.kwargs['motion'])
    return events


def _motion_record(motion) -> dict:
    """A DANT ``Motion`` as plain floats; two motions are equal when their records are ``==``."""
    try:
        record = {'linear_scale': float(np.asarray(motion.LinearScale)),
                  'linear': [float(v) for v in np.asarray(motion.Linear, float)],
                  'constant': [float(v) for v in np.asarray(motion.Constant, float)]}
    except (AttributeError, TypeError, ValueError) as exc:
        raise ContractViolation(f'DANT motion cannot be read: {exc!r}') from exc
    return record


def _dant_components(sessions: list, pairs: list) -> list:
    """Connected components of ``pairs`` over ``sessions``; members and groups in ``sessions`` order."""
    parent = {s: s for s in sessions}

    def root(s):
        while parent[s] != s:
            s = parent[s]
        return s

    for a, b in pairs:
        parent[root(a)] = root(b)
    members = {}
    for s in sessions:
        members.setdefault(root(s), []).append(s)
    return list(members.values())


def _dant_registration(log: dict, sessions: list, cellids: list, sid: np.ndarray,
                       locations: Optional[np.ndarray], group: str, sources: list) -> dict:
    """The registration DANT applied, from the ``run_native_dant`` motion log of one run.

    The applied motion is ``log['final']`` (``log['initial']`` when DANT never called
    ``computeWaveformFeatures``); its groups come from the one round that fitted it. DANT
    subtracts its motion (``ComputeWaveformFeatures.py:58-60``), so each offset is minus it. A
    log that cannot be read this way raises ``ContractViolation``.
    """
    def fail(problem):
        raise ContractViolation(f'DANT registration of {group}: {problem}')

    final = log['final']
    if final is None:
        final = log['initial']
    if final is None:
        fail('DANT applied and initialised no motion')
    n = len(sessions)
    for key in ('linear', 'constant'):
        if len(final[key]) != n or not all(math.isfinite(v) for v in final[key]):
            fail(f'motion {key} has {len(final[key])} values, expected {n} finite values')
    matching = [r for r in log['rounds'] if r['motion'] == final]
    if len({tuple(tuple(p) for p in r['pairs']) for r in matching}) > 1:
        fail('rounds with different pairs fitted the applied motion; the kept round is ambiguous')
    if not matching:
        if final == log['initial']:
            return {'method': 'native', 'offsets_um': {},
                    'diagnostics': {group: {'note': 'dant kept initial motion', 'sources': sources}}}
        fail('DANT applied a motion no round fitted')
    kept = matching[-1]
    pairs = []
    for a, b in kept['pairs']:
        if not (1 <= a <= n and 1 <= b <= n):
            fail(f'session pair ({a}, {b}) is out of range for {n} sessions')
        pairs.append([sessions[a - 1], sessions[b - 1]])
    constant, linear = final['constant'], final['linear']
    registration = {'method': 'native', 'offsets_um': {sessions[k]: -constant[k] for k in range(n)}}
    if any(v != 0 for v in linear):
        scale = final['linear_scale']
        sid = np.asarray(sid)
        loc = None if locations is None else np.asarray(locations, float)
        if (loc is None or loc.ndim != 2 or loc.shape[0] != len(cellids) or loc.shape[1] < 2
                or not np.isfinite(loc).all()):
            fail(f'a linear motion needs finite unit locations of shape ({len(cellids)}, n >= 2)')
        if (sid.shape != (len(cellids),) or not np.issubdtype(sid.dtype, np.integer)
                or not ((sid >= 0) & (sid < n)).all() or not math.isfinite(scale)):
            fail('unit session indices or the linear scale cannot be read')
        registration['depth_scale'] = {sessions[k]: 1 - scale * linear[k] for k in range(n)}
        registration['unit_offsets_um'] = {
            cellids[i]: float(-(scale * linear[sid[i]] * loc[i, 1] + constant[sid[i]]))
            for i in range(len(cellids))}
    registration['diagnostics'] = {group: {'groups': _dant_components(sessions, pairs), 'pairs': pairs,
                                           'motion': final, 'sources': sources}}
    return registration


def run_dant(units, pairs=None, *, waveform_bundle=None, settings=None,
             provenance=None, run_id=None, seed=0, **options):
    """Run official DANT in a disposable directory and preserve native clusters.

    settings is a complete upstream settings mapping; omission selects the
    documented waveform/autocorrelogram preset from default_dant_settings().
    PETH uses an explicitly
    aligned (unit, stimulus-time-bin) array with unique stimulus/time bin keys.
    """
    eligible, excluded, order, waves, geometry, sid, _ = _prepare_bundle(units, waveform_bundle)
    if settings is None:
        settings = default_dant_settings()
    if not isinstance(seed, int) or seed < 0 or seed >= 2 ** 32:
        raise ValueError('DANT random seed must be an integer in [0, 2**32)')
    if len(waveform_bundle.spike_times_seconds) != len(waveform_bundle.cellids):
        raise ValueError('DANT requires spike times in seconds for every input unit')
    counts = np.asarray(waveform_bundle.waveform_spike_counts)
    if (counts.shape != (len(waveform_bundle.cellids), 2)
            or not np.isfinite(counts).all() or np.any(counts <= 0)):
        raise ValueError('DANT requires waveform_spike_counts (unit, 2) to combine half means correctly')
    if any(not np.array_equal(g, geometry[0]) for g in geometry[1:]):
        raise ValueError('DANT input format requires common channel geometry across sessions')
    config = copy.deepcopy(settings)
    features = config['clustering']['features'] + [
        f for block in config['motionEstimation']['features'] for f in block]
    peth = None
    if 'PETH' in features:
        peth = np.asarray(waveform_bundle.peth, dtype=float)
        keys = waveform_bundle.peth_bin_keys
        if (peth.ndim != 2 or peth.shape[0] != len(waveform_bundle.cellids)
                or peth.shape[1] != len(keys) or len(keys) < 2
                or len(set(keys)) != len(keys) or not np.isfinite(peth).all()):
            raise ValueError('PETH requires finite aligned (unit, bin) data and unique stimulus/time bin keys')
    runner = _module('pyDANT.Runner')
    with tempfile.TemporaryDirectory(prefix='nems-dant-') as root:
        data = Path(root) / 'input'
        output = Path(root) / 'output'
        (data / 'spike_times').mkdir(parents=True)
        weights = counts[order] / counts[order].sum(axis=1, keepdims=True)
        full_mean = np.sum(waves * weights[:, None, None, :], axis=3)
        np.save(data / 'waveform_all.npy', full_mean.transpose(0, 2, 1))
        np.save(data / 'channel_locations.npy', geometry[0])
        np.save(data / 'session_index.npy', sid + 1)
        if peth is not None:
            np.save(data / 'peth.npy', peth[order])
        for k, i in enumerate(order):
            times = np.asarray(waveform_bundle.spike_times_seconds[i], dtype=float)
            if times.ndim != 1 or len(times) < 2 or not np.isfinite(times).all():
                raise ValueError('DANT needs at least two finite spike times per unit')
            np.save(data / 'spike_times' / f'Unit{k}.npy', 1000 * times)
        config.update(path_to_data=str(data), output_folder=str(output))
        motion_module = _module('pyDANT.MotionEstimation')
        motion_log = {}
        with _DANT_RUN_LOCK:
            compatibility_events = run_native_dant(runner, motion_module, config, seed, motion_log)
            correction_module = motion_module.computeWaveformFeatures.__module__
        # Rows in waveform_all.npy order, which is eligible order.
        locations = (np.load(output / 'locations.npy', allow_pickle=False)
                     if (output / 'locations.npy').is_file() else None)
        with np.load(output / 'Output.npz', allow_pickle=True) as saved:
            # Only unpickle a file just produced by the explicitly invoked library.
            out = saved['arr_0' if 'arr_0' in saved else 'Output'].item()
        labels = np.asarray(out['IdxCluster']).reshape(-1)
        similarity = np.asarray(out['SimilarityMatrix'], dtype=float)
        diagnostics = {key: np.asarray(out[key]).tolist() for key in
                       ('NumClusters', 'NumUnits', 'SimilarityNames', 'SimilarityWeights', 'SimilarityThreshold',
                        'IdxCluster', 'CurationPairs', 'CurationTypes', 'CurationTypeNames') if key in out}
        diagnostics['SimilarityNames'] = list(config['clustering']['features'])
        for key in ('SimilarityWeights', 'SimilarityThreshold', 'CurationPairs', 'CurationTypes',
                    'CurationTypeNames', 'RunTimeSec', 'motion_constant', 'motion_linear'):
            artifact = output / (key + '.npy')
            if artifact.is_file():
                diagnostics[key] = np.asarray(np.load(artifact, allow_pickle=False)).tolist()
    if len(labels) != len(eligible) or similarity.shape != (len(eligible), len(eligible)):
        raise ValueError('DANT output unit count/order does not match input')
    ids = eligible.cellid.to_numpy()
    sessions = [str(s) for s in sid]
    n_spikes = (eligible['n_spikes'].astype(float).tolist() if 'n_spikes' in eligible
                else [float('nan')] * len(ids))
    groups = resolve_groups(list(ids), sessions, [int(v) if v > 0 else None for v in labels],
                            similarity, n_spikes)
    tracks = list(groups.tracks)
    selected = {u for track in tracks for u in track}
    a, b = np.where(np.triu(np.ones((len(ids), len(ids)), dtype=bool), 1))
    same_cluster = (labels[a] == labels[b]) & (labels[a] > 0)
    result_pairs = pd.DataFrame(dict(cellid_a=ids[a], cellid_b=ids[b],
        match_score=np.full(len(a), np.nan),
        dant_similarity=similarity[a, b], dant_same_cluster=same_cluster))
    result_pairs = label_pairs(result_pairs, dict(zip(ids, sessions)), groups,
                               Decision.NATIVE_CLUSTER.value, Decision.UNMATCHED_BY_NATIVE_CLUSTERING.value)
    metadata = dict(provenance or {})
    metadata.update(official_source=DANT_SOURCE, verified_api_revision=DANT_REVISION,
                    implementation=_source_info(runner), waveform_provenance=waveform_bundle.provenance,
                    parameters=settings, assignment='native_DANT',
                    random_seed=seed, native_diagnostics=diagnostics,
                    compatibility_events=compatibility_events,
                    compatibility_policy='Native pipeline, with previous motion retained only when native drift selection has no eligible pairs',
                    upstream_caveats=['Pinned upstream early-stop reverts corrected waveforms in memory but reloads the last saved motion for final distance gating; upstream behavior retained'],
                    score_meaning='native similarity is unbounded; no match probability supplied',
                    peth_bin_keys=waveform_bundle.peth_bin_keys if peth is not None else (),
                    enforce_session_unique=True,
                    spike_time_conversion='seconds to milliseconds')
    sources = [_source_info(motion_module), _source_info(_module(correction_module))]
    metadata.update(registration=_dant_registration(
        motion_log, list(dict.fromkeys(eligible.session_id)), list(eligible.cellid), sid, locations,
        str((provenance or {}).get('group', '')), sources))
    metadata.update(used_optional_inputs=['fingerprint'] if peth is not None else [],
                    fingerprint_used=peth is not None)
    return MatchResult('dant', _version('pyDANT'), run_id or f'dant_{uuid.uuid4().hex}',
                       eligible, result_pairs, tuple(tracks),
                       tuple(u for u in ids if u not in selected), excluded,
                       ScoreBlock(ScoreKind.NONE, None, similarity_column='dant_similarity'),
                       TrackKind.NATIVE, 'native_DANT', Provenance.from_json(metadata)).validate()


SPEC = BackendSpec('dant', '1',
    requires=frozenset({Input.UNITS, Input.WAVEFORM_HALVES, Input.GEOMETRY, Input.SPIKE_TIMES}),
    optional=frozenset({Input.FINGERPRINT}),
    provides=Provides(frozenset({ScoreKind.NONE}), frozenset({TrackKind.NATIVE}), similarity=True,
                      day_offsets=True),
    options=(Option('preset', 'choice', 'waveform_acg', 'Documented settings preset',
                    choices=('waveform_acg', 'waveform_acg_stimulus', 'upstream_example')),
             Option('settings', 'json', None, 'Complete upstream settings mapping; overrides preset'),
             Option('seed', 'int', 0, 'Random seed')), builtin=True)


class DantBackend:
    spec = SPEC

    def available(self):
        return availability('pyDANT')

    def run(self, inp, options, ctx):
        avail = self.available()
        if not avail.ok:
            raise BackendUnavailable(avail.reason)
        opts = {**SPEC.defaults(), **options}
        settings = opts['settings'] or default_dant_settings(opts['preset'])
        needs_peth = 'PETH' in settings['clustering']['features'] + [f for b in settings['motionEstimation']['features'] for f in b]
        if needs_peth and Input.FINGERPRINT not in inp.available():
            raise MissingInputs(SPEC.id, inp.group, {Input.FINGERPRINT})
        return run_dant(inp.units, inp.candidate_pairs, waveform_bundle=inp.bundle(), settings=settings,
                        provenance=ctx.provenance.to_json(), run_id=ctx.run_id, seed=opts['seed'])
