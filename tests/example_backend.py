"""A minimal backend, kept as a live worked example for ``docs/adding_a_backend.md``.

It requires only units and candidate pairs and scores every pair with a seeded uniform
draw, so it needs no waveform archive, checkpoint or upstream package. It is never a
built-in: nothing under ``backends/`` imports this module. A real package registers a
backend like this through ``[project.entry-points."multiphy.backends"]``; this file is
registered in-process, only inside the conformance test.
"""
from __future__ import annotations
import numpy as np

from multiphy.contract.backend import Availability, BackendSpec, Option, Provides
from multiphy.contract.inputs import Input
from multiphy.contract.results import ScoreKind, TrackKind
from multiphy.backends.local.tracker import result_from_pairs

SPEC = BackendSpec(
    'random_baseline', '1',
    requires=frozenset({Input.UNITS, Input.CANDIDATE_PAIRS}),
    optional=frozenset(),
    provides=Provides(frozenset({ScoreKind.UNCALIBRATED}), frozenset({TrackKind.COMMON})),
    options=(Option('seed', 'int', 0, 'Seed for the uniform draw'),))


class RandomBaselineBackend:
    spec = SPEC

    def available(self):
        return Availability(True, SPEC.version, None)

    def run(self, inp, options, ctx):
        pairs = inp.candidate_pairs.copy()
        rng = np.random.default_rng(options.get('seed', 0))
        pairs['match_score'] = rng.uniform(size=len(pairs))
        prov = ctx.provenance.merged(extra={'used_optional_inputs': []})
        return result_from_pairs(SPEC.id, SPEC.version, inp.units, pairs, score_column='match_score',
                                 provenance=prov.to_json(), run_id=ctx.run_id)
