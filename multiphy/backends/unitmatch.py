"""Backend ``unitmatch``: the official UnitMatchPy scoring with native or common tracking.

Verified upstream API: EnnyvanBeest/UnitMatch revision
95b6fed5db84d12f67487b92d07f0a2664a6122f. This is an API reference, not a claim that an
arbitrary installed version reproduces the original publication. Tracking defaults to the
official intermediate rule; ``common_constrained`` hands the pair probabilities to the local
tracker instead. A native group holding two or more units of one session keeps one per session
(backends/native_groups.py); the others are unmatched and their pairs inside the group are
proposed_link.
"""
from __future__ import annotations
import uuid

import numpy as np
import pandas as pd

from ..contract.backend import (BackendSpec, Option, Provides, COMMON_TRACKER_OPTIONS,
                                NATIVE_TRACKING_OPTIONS)
from ..contract.errors import BackendUnavailable
from ..contract.errors import ContractViolation
from ..contract.inputs import Input
from ..contract.provenance import Provenance
from ..contract.results import Decision, MatchResult, ScoreBlock, ScoreKind, TrackKind, single_units
from ._upstream import (UNITMATCH_REVISION, UNITMATCH_SOURCE, availability, module as _module,
                        prepare_bundle as _prepare_bundle, source_info as _source_info, version as _version)
from ._upstream import recording as _recording
from .local.tracker import result_from_pairs
from .native_groups import label_pairs, resolve_groups

_TRACKER_KEYS = ('run_id', 'threshold', 'contradiction_threshold', 'ambiguity_margin')


def _um_setup(waves, geometry, sid, switch, params):
    default = _module('UnitMatchPy.default_params')
    util = _module('UnitMatchPy.utils')
    param = default.get_default_param(param=dict(params or {}, spike_width=waves.shape[1]))
    param.update(n_units=len(waves), n_sessions=len(geometry), n_channels=waves.shape[2])
    param = util.get_probe_geometry(geometry[0], param)
    groups = [np.flatnonzero(sid == s) for s in range(len(geometry))]
    info = dict(good_units=groups, original_ids=np.arange(len(waves)),
                session_id=sid, session_switch=switch)
    return param, info, (sid[:, None] != sid[None, :]).astype(int)


def _um_geometry(geometry):
    # Match official utils.paths_from_KS: prepend the constant first coordinate.
    # Feature extraction requires three columns; the public bundle keeps x/y.
    return [np.column_stack([np.ones(len(g)), g]) for g in geometry]


_DRIFT_APPLY = ('apply_drift_correction_basic', 'apply_drift_correction_per_shank')


_UNREADABLE = (TypeError, ValueError, IndexError, KeyError, OverflowError)


def _read(name, what, read):
    """``read()``, or ``ContractViolation`` naming ``name`` when its ``what`` cannot be read."""
    try:
        return read()
    except _UNREADABLE as exc:
        raise ContractViolation(f'{name} {what} cannot be read: {exc!r}') from exc


def _finite(values, shape, name):
    array = _read(name, 'drift (not numeric)', lambda: np.asarray(values, float))
    if array.shape != shape or not np.isfinite(array).all():
        raise ContractViolation(f'{name} returned a drift of shape {array.shape}, expected finite {shape}')
    return array


def _unitmatch_registration(calls: list, sessions: list, cellids: list, switch: np.ndarray, param: dict,
                            group: str, sources: list) -> dict:
    """The registration UnitMatchPy applied, from the ``Call`` list of one ``extract_metric_scores``.

    Row ``k`` of ``drift_n_sessions`` aligns day ``k + 1`` to the corrected day ``k``; a pair
    UnitMatchPy did not correct starts a new group. A capture that cannot be read raises
    ``ContractViolation`` naming the call.
    """
    n_rows = len(sessions) - 1
    drift_calls = [c for c in calls if c.name == 'drift_n_sessions']
    if len(drift_calls) != 1:
        raise ContractViolation(f'drift_n_sessions was called {len(drift_calls)} times; expected exactly one')
    rows = _finite(_read('drift_n_sessions', 'result[0] (drift rows)', lambda: drift_calls[0].result[0]),
                   (n_rows, 3), 'drift_n_sessions')
    applied, n_pairs, shank_drifts = {}, {}, {}
    for call in calls:
        if call.name not in _DRIFT_APPLY or call.parent != 'drift_n_sessions':
            continue
        k = _read(call.name, 'args[1] (session index)', lambda: int(call.args[1]))
        if k not in range(n_rows):
            raise ContractViolation(f'{call.name} aligned session index {k + 1}; '
                                    f'{len(sessions)} sessions were recorded')
        if k in applied:
            raise ContractViolation(f'{call.name} aligned session pair {k}/{k + 1} twice')
        applied[k] = call
        n_pairs[k] = _read(call.name, 'args[0] (matched pairs)', lambda: len(call.args[0]))
    n_shanks = _read('UnitMatchPy param', "'no_shanks'", lambda: int(param['no_shanks']))
    shank_width_um = _read('UnitMatchPy param', "'shank_dist'", lambda: float(param['shank_dist']))
    for k, call in applied.items():
        if call.name != 'apply_drift_correction_per_shank':
            continue
        drift = _finite(_read(call.name, 'result[0] (shank drifts)', lambda: call.result[0]),
                        (n_shanks, 3), call.name)
        ids = [c for c in calls if c.name == 'shank_ID_per_session'
               and c.parent == 'apply_drift_correction_per_shank'
               and _read(c.name, 'args[2] (session number)', lambda: int(c.args[2])) == k + 1]
        if len(ids) != 1:
            raise ContractViolation(f'shank_ID_per_session was called {len(ids)} times for session '
                                    f'index {k + 1}; expected exactly one')
        shank_ids = _read('shank_ID_per_session', 'result (shank ids)',
                          lambda: np.asarray(ids[0].result, float))
        size = int(switch[k + 2] - switch[k + 1])
        if (shank_ids.shape != (size,) or not np.isfinite(shank_ids).all()
                or not np.array_equal(shank_ids, np.round(shank_ids))
                or not ((shank_ids >= 0) & (shank_ids < n_shanks)).all()):
            raise ContractViolation(f'shank_ID_per_session for session index {k + 1} returned '
                                    f'{shank_ids.shape} values, expected {size} shank numbers in '
                                    f'range({n_shanks})')
        shank_drifts[k] = (drift, shank_ids.astype(int))
    groups = [[sessions[0]]]
    offsets_um, shank_offsets_um, unit_shanks = {sessions[0]: 0.0}, {}, {}
    for k in range(n_rows):
        day = sessions[k + 1]
        if k in applied:
            groups[-1].append(day)
        else:
            groups.append([day])
        if k in shank_drifts:
            drift, shank_ids = shank_drifts[k]
            shank_offsets_um[day] = [float(v) for v in drift[:, 2]]
            for j, s in enumerate(shank_ids):
                unit_shanks[cellids[int(switch[k + 1]) + j]] = int(s)
        else:
            offsets_um[day] = float(rows[k, 2])
    drifts = [[sessions[k], sessions[k + 1], float(rows[k, 2]), float(rows[k, 1]), n_pairs.get(k, 0)]
              for k in range(n_rows)]
    per_shank = [[sessions[k], sessions[k + 1], i, float(shank_drifts[k][0][i, 2]),
                  float(shank_drifts[k][0][i, 1])]
                 for k in sorted(shank_drifts) for i in range(n_shanks)]
    return {'method': 'native', 'offsets_um': offsets_um, 'shank_offsets_um': shank_offsets_um,
            'unit_shanks': unit_shanks, 'shank_width_um': shank_width_um,
            'diagnostics': {group: {'groups': groups, 'drifts': drifts, 'shank_drifts': per_shank,
                                    'sources': sources}}}


def _matrix_pairs(eligible: pd.DataFrame, matrix: np.ndarray) -> pd.DataFrame:
    """Every unit pair ``a < b`` in ``eligible`` order, scored by the minimum of both directions."""
    matrix = np.asarray(matrix, float)
    if matrix.shape != (len(eligible), len(eligible)) or not np.isfinite(matrix).all():
        raise ValueError('Official backend returned an invalid probability matrix')
    a, b = np.where(np.triu(np.ones(matrix.shape, dtype=bool), 1))
    ids = eligible.cellid.to_numpy()
    return pd.DataFrame({'cellid_a': ids[a], 'cellid_b': ids[b],
                         'match_score': np.minimum(matrix[a, b], matrix[b, a]),
                         'official_probability_ab': matrix[a, b],
                         'official_probability_ba': matrix[b, a]})


def _um_result(backend_id, units, eligible, prob, sid, param, info, tracking, metadata, options):
    """MatchResult from a UnitMatch probability matrix, by native or common tracking.

    Native tracking keeps one unit per session in each native group (``resolve_groups``); the
    group's other pairs are saved with decision ``proposed_link``.
    """
    pairs = _matrix_pairs(eligible, prob)
    options = dict(options)
    native_threshold = options.pop('native_threshold', None)
    if tracking == 'common_constrained':
        options = {k: options[k] for k in _TRACKER_KEYS if k in options}
        return result_from_pairs(backend_id, _version('UnitMatchPy'), units, pairs,
                                 provenance=metadata, **options)
    modes = {'native_liberal': 0, 'native_intermediate': 1, 'native_conservative': 2}
    if tracking not in modes:
        raise ValueError('tracking must be native_liberal/intermediate/conservative or common_constrained')
    assigner = _module('UnitMatchPy.assign_unique_id')
    native_params = dict(param, match_threshold=(param.get('match_threshold', .5)
                         if native_threshold is None else native_threshold))
    if not 0 <= native_params['match_threshold'] <= 1:
        raise ValueError('native_threshold must be in [0, 1]')
    labels = np.asarray(assigner.assign_unique_id(prob, native_params, info)[modes[tracking]])
    ids = eligible.cellid.to_numpy()
    sessions = [str(s) for s in sid]
    n_spikes = (eligible['n_spikes'].astype(float).tolist() if 'n_spikes' in eligible
                else [float('nan')] * len(ids))
    prob = np.asarray(prob, float)
    groups = resolve_groups(list(ids), sessions, [int(v) for v in labels], np.minimum(prob, prob.T), n_spikes)
    pairs = label_pairs(pairs, dict(zip(ids, sessions)), groups, Decision.NATIVE_TRACK.value,
                        Decision.UNMATCHED_BY_NATIVE_TRACKING.value)
    tracks = list(groups.tracks)
    members = {u for track in tracks for u in track}
    metadata.update(assignment=tracking, tracker_implementation=_source_info(assigner),
                    threshold=native_params['match_threshold'], enforce_session_unique=True,
                    same_session_isi_merge_check='not used without original Kilosort directory mapping')
    _, excluded = single_units(units)
    return MatchResult(backend_id, _version('UnitMatchPy'),
        options.get('run_id') or f'{backend_id}_{uuid.uuid4().hex}', eligible, pairs,
        tuple(tracks), tuple(u for u in ids if u not in members), excluded,
        ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score'), TrackKind.NATIVE, tracking,
        Provenance.from_json(metadata)).validate()


def run_unitmatch(units, pairs=None, *, waveform_bundle=None, params=None,
                  tracking='native_intermediate', native_threshold=None,
                  provenance=None, **options):
    """Official waveform metrics/drift/Bayes and explicit native/common tracking."""
    eligible, _, _, waves, geometry, sid, switch = _prepare_bundle(units, waveform_bundle)
    ov, bf = _module('UnitMatchPy.overlord'), _module('UnitMatchPy.bayes_functions')
    param, info, within = _um_setup(waves, geometry, sid, switch, params)
    extracted = ov.extract_parameters(waves, _um_geometry(geometry), info, param)
    mf = _module('UnitMatchPy.metric_functions')
    with _recording(mf, 'drift_n_sessions', 'apply_drift_correction_basic', 'apply_drift_correction_per_shank',
                    'shank_ID_per_session') as calls:
        _, candidate, scores, predictors = ov.extract_metric_scores(
            extracted, switch, within, param, niter=2)
    labels = candidate.astype(int)
    conditions = np.unique(labels)
    if not np.array_equal(conditions, [0, 1]):
        raise ValueError('UnitMatch needs both candidate-match and nonmatch reference classes')
    prior = param['n_expected_matches'] / param['n_units'] ** 2
    kernels = bf.get_parameter_kernels(scores, labels, conditions, param, add_one=1)
    prob = bf.apply_naive_bayes(kernels, np.array([1-prior, prior]),
                               predictors, param, conditions)[:, 1].reshape(len(waves), -1)
    metadata = dict(provenance or {})
    metadata.update(official_source=UNITMATCH_SOURCE, verified_api_revision=UNITMATCH_REVISION,
                    implementation=_source_info(ov), waveform_provenance=waveform_bundle.provenance,
                    parameters=param, pair_reduction='minimum of both directional probabilities',
                    calibration_status='official Bayes outputs; not validated identity accuracy here')
    metadata.update(used_optional_inputs=[])
    metadata.update(registration=_unitmatch_registration(
        calls, list(dict.fromkeys(eligible.session_id)), list(eligible.cellid), switch, param,
        str((provenance or {}).get('group', '')), [_source_info(mf)]))
    return _um_result('unitmatch', units, eligible, prob, sid, param, info,
                      tracking, metadata, dict(options, native_threshold=native_threshold))


SPEC = BackendSpec('unitmatch', '1',
    requires=frozenset({Input.UNITS, Input.WAVEFORM_HALVES, Input.GEOMETRY}), optional=frozenset(),
    provides=Provides(frozenset({ScoreKind.UNCALIBRATED}), frozenset({TrackKind.NATIVE, TrackKind.COMMON}),
                      day_offsets=True),
    options=(Option('params', 'json', None, 'Overrides for UnitMatchPy default parameters'),
             *NATIVE_TRACKING_OPTIONS, *COMMON_TRACKER_OPTIONS), builtin=True)


class UnitMatchBackend:
    spec = SPEC

    def available(self):
        return availability('UnitMatchPy')

    def run(self, inp, options, ctx):
        avail = self.available()
        if not avail.ok:
            raise BackendUnavailable(avail.reason)
        opts = {**SPEC.defaults(), **options}
        tracker = {k: opts[k] for k in ('threshold', 'contradiction_threshold', 'ambiguity_margin')}
        return run_unitmatch(inp.units, inp.candidate_pairs, waveform_bundle=inp.bundle(), params=opts['params'],
                             tracking=opts['tracking'], native_threshold=opts['native_threshold'],
                             provenance=ctx.provenance.to_json(), run_id=ctx.run_id, **tracker)
