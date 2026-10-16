"""Built-in backends, registered on import, plus the old ``backends.py`` public names.

The compatibility names (``run_backend``, ``list_backends``, ``register_backend``,
``save_result``, ``load_result``, ``result_digest``) exist so callers written against the
flat module keep working until the viewer is rewritten (sub-project 3) and the
repository split (sub-project 4). New code imports from ``contract`` and ``run``.
"""
from __future__ import annotations
from collections import namedtuple
from pathlib import Path
import tempfile
import uuid

from ..contract import registry
from ..contract.backend import Availability, BackendSpec, Provides, RunContext  # noqa: F401
from ..contract.backend import COMMON_TRACKER_OPTIONS
from ..contract.errors import BackendUnavailable  # noqa: F401
from ..contract.inputs import Input, MatchingInput
from ..contract.provenance import Provenance, json_safe as _json_safe, payload_digest as _payload_digest  # noqa: F401
from ..contract.results import MatchResult, ScoreKind, TrackKind, single_units  # noqa: F401
from ..run.run_dir import load_result, save_result  # noqa: F401
from .local.tracker import result_from_pairs  # noqa: F401
from .local.functional_likelihood import FunctionalLikelihoodBackend
from .local.pair_logistic import PairLogisticBackend
from .dant import DantBackend
from .deepunitmatch import DeepUnitMatchBackend
from .unitmatch import UnitMatchBackend


def register_builtins():
    """Register every built-in id not already present; the registry calls this after a reset."""
    for backend in (FunctionalLikelihoodBackend(), PairLogisticBackend(),
                    UnitMatchBackend(), DantBackend(), DeepUnitMatchBackend()):
        if backend.spec.id not in registry.registered_ids():
            registry.register(backend)


register_builtins()

_SpecView = namedtuple('SpecView', 'backend_id label version available description')


def result_digest(result):
    return result.digest()


def list_backends():
    """Old shape: objects with backend_id, label, version, available, description."""
    from ..contract.names import backend_help, backend_label
    out = []
    for spec in registry.list_specs():
        ok = registry.get(spec.id).available().ok
        out.append(_SpecView(spec.id, backend_label(spec.id), spec.version, ok, backend_help(spec.id)))
    return tuple(out)


def register_backend(spec, runner):
    """Old API: wrap a bare runner callable in a Backend. Kept for tests written against it."""
    if not callable(runner):
        raise ValueError('Registered backends require a callable runner')

    class _Wrapped:
        def available(self):
            return Availability(True, spec.version, None)

        def run(self, inp, options, ctx):
            extra = {}
            if Input.WAVEFORM_HALVES in inp.available():
                extra['waveform_bundle'] = inp.bundle()
            return runner(inp.units, inp.candidate_pairs, run_id=ctx.run_id,
                          provenance=inp.provenance.to_json(), **extra, **options)
    _Wrapped.spec = spec if isinstance(spec, BackendSpec) else BackendSpec(
        spec.backend_id, spec.version, frozenset({Input.UNITS}), frozenset(),
        Provides(frozenset(ScoreKind), frozenset(TrackKind)), COMMON_TRACKER_OPTIONS)
    registry.register(_Wrapped())


def run_backend(backend_id, units, pairs, *, threshold=.90, contradiction_threshold=.10,
                ambiguity_margin=.02, provenance=None, run_id=None, feature_stream='combined', **kwargs):
    """Deprecated adapter: the pre-contract call shape, routed through the registry.

    Removed when the launcher (S10) calls run.runner directly.
    """
    backend = registry.get(backend_id)
    avail = backend.available()
    if not avail.ok:
        raise BackendUnavailable(avail.reason)
    if kwargs.pop('score_kind', 'uncalibrated_score') != 'uncalibrated_score':
        raise ValueError('Cached local scores cannot be relabeled as calibrated probabilities')
    inp = MatchingInput(units, group=str((provenance or {}).get('block_id', 'compat')), candidate_pairs=pairs,
                        provenance=Provenance.from_json(provenance)).validate()
    unknown = sorted(set(kwargs) - {o.name for o in backend.spec.options})
    if unknown:
        raise ValueError(f'{backend_id} does not accept {unknown}')
    options = {**backend.spec.defaults(), 'threshold': threshold, 'contradiction_threshold': contradiction_threshold,
               'ambiguity_margin': ambiguity_margin, 'feature_stream': feature_stream, **kwargs}
    options = {k: v for k, v in options.items() if k in {o.name for o in backend.spec.options}}
    with tempfile.TemporaryDirectory(prefix='compat-run-') as work:
        ctx = RunContext(run_id or f'{backend_id}_{uuid.uuid4().hex}', Path(work), provenance=inp.provenance)
        result = backend.run(inp, options, ctx)
    expected, excluded = single_units(units)
    if set(result.units.cellid) != set(expected.cellid) or set(result.excluded_units) != set(excluded):
        raise ValueError('Runner must account for every input single unit, including unmatched units')
    if run_id is not None and result.run_id != run_id:
        raise ValueError('Runner must preserve the requested run identity')
    return result.validate(backend.spec)
