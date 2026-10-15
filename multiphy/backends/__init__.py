# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: register built-ins; keep old backends.py names importable | date: 2026-09-22]
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
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: common tracker options for the compatibility register_backend spec (Task 11) | date: 2026-09-22]
from ..contract.backend import COMMON_TRACKER_OPTIONS
# [AGENT EDIT END]
from ..contract.errors import BackendUnavailable  # noqa: F401
from ..contract.inputs import Input, MatchingInput
from ..contract.provenance import Provenance, json_safe as _json_safe, payload_digest as _payload_digest  # noqa: F401
from ..contract.results import MatchResult, ScoreKind, TrackKind, single_units  # noqa: F401
from ..run.run_dir import load_result, save_result  # noqa: F401
from .local.tracker import result_from_pairs  # noqa: F401
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the local matcher is registered by its mechanism id; constrained removed (Task 3) | date: 2026-09-27]
from .local.functional_likelihood import FunctionalLikelihoodBackend
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the calibrated per-project classifier is a registered built-in (functional-likelihood Task 8) | date: 2026-09-27]
from .local.pair_logistic import PairLogisticBackend
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: published-method adapters are registered built-ins (Task 9) | date: 2026-09-22]
from .dant import DantBackend
from .deepunitmatch import DeepUnitMatchBackend
from .unitmatch import UnitMatchBackend
# [AGENT EDIT END]


def register_builtins():
    """Register every built-in id not already present; the registry calls this after a reset."""
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: matching methods only, named by mechanism; pair_logistic added (Task 3, functional-likelihood Task 8) | date: 2026-09-27]
    for backend in (FunctionalLikelihoodBackend(), PairLogisticBackend(),
                    UnitMatchBackend(), DantBackend(), DeepUnitMatchBackend()):
    # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: check ids without listing, so a plugin cannot load first and shadow a built-in (whole-branch review fix 2) | date: 2026-09-23]
        if backend.spec.id not in registry.registered_ids():
        # [AGENT EDIT END]
            registry.register(backend)


register_builtins()

_SpecView = namedtuple('SpecView', 'backend_id label version available description')


def result_digest(result):
    return result.digest()


def list_backends():
    """Old shape: objects with backend_id, label, version, available, description."""
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: a backend with no NAMES entry is listed by its id (whole-branch review fix 3) | date: 2026-09-23]
    from ..contract.names import backend_help, backend_label
    out = []
    for spec in registry.list_specs():
        ok = registry.get(spec.id).available().ok
        out.append(_SpecView(spec.id, backend_label(spec.id), spec.version, ok, backend_help(spec.id)))
    # [AGENT EDIT END]
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
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: synthesized spec declares the common tracker options so validated threshold/contradiction/margin reach the wrapped runner (Task 11) | date: 2026-09-22]
    _Wrapped.spec = spec if isinstance(spec, BackendSpec) else BackendSpec(
        spec.backend_id, spec.version, frozenset({Input.UNITS}), frozenset(),
        Provides(frozenset(ScoreKind), frozenset(TrackKind)), COMMON_TRACKER_OPTIONS)
    # [AGENT EDIT END]
    registry.register(_Wrapped())


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run_backend takes no calibrated_model (Task 2) | date: 2026-09-27]
def run_backend(backend_id, units, pairs, *, threshold=.90, contradiction_threshold=.10,
                ambiguity_margin=.02, provenance=None, run_id=None, feature_stream='combined', **kwargs):
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the launcher, since gui.py is deleted (viewer-split Task 9) | date: 2026-10-07]
    """Deprecated adapter: the pre-contract call shape, routed through the registry.

    Removed when the launcher (S10) calls run.runner directly.
    """
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no id aliases; an unknown id raises the registry's KeyError (Task 1) | date: 2026-09-27]
    backend = registry.get(backend_id)
    # [AGENT EDIT END]
    avail = backend.available()
    if not avail.ok:
        raise BackendUnavailable(avail.reason)
    if kwargs.pop('score_kind', 'uncalibrated_score') != 'uncalibrated_score':
        raise ValueError('Cached local scores cannot be relabeled as calibrated probabilities')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no fitted_model input (Task 2) | date: 2026-09-27]
    inp = MatchingInput(units, group=str((provenance or {}).get('block_id', 'compat')), candidate_pairs=pairs,
                        provenance=Provenance.from_json(provenance)).validate()
    # [AGENT EDIT END]
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an undeclared argument (e.g. the removed calibrated_model) is refused, never discarded (Task 2 fix round 1) | date: 2026-09-27]
    unknown = sorted(set(kwargs) - {o.name for o in backend.spec.options})
    if unknown:
        raise ValueError(f'{backend_id} does not accept {unknown}')
    # [AGENT EDIT END]
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
# [AGENT EDIT END]
