# Adding a backend

A backend is one object: a `spec` (a `BackendSpec` declaring its id, version, inputs,
emissions and options) plus `available()` and `run(inp, options, ctx) -> MatchResult`.
Nothing else in the package needs to know a new backend exists. This is a complete
worked example — `random_baseline`, a twenty-line backend that requires only units
and candidate pairs and scores every pair with a seeded uniform draw. It lives at
`tests/example_backend.py` and is registered only inside
`tests/test_contract_conformance.py`, so this doc's example stays tested
rather than going stale.

## The file

```python
# tests/example_backend.py
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
```

`result_from_pairs` does the rest: it forms tracks from the `match_score` column with
the shared best-first tracker, fills in `excluded_units` (non-`SU` rows), and builds a
valid `MatchResult`. The backend itself only has to produce a score.

## Registering it

A real package registers a backend by declaring an entry point — nothing in this repo
needs to change:

```toml
# pyproject.toml, in a package that depends on multiphy
[project.entry-points."multiphy.backends"]
random_baseline = "my_package.backends:make_random_baseline"
```

where `make_random_baseline()` returns a `RandomBaselineBackend()`.
`registry.list_specs()` loads entry points after the built-ins, and `registry.get()`
loads them on the first miss (`contract/registry.py`), so installing that package is
enough for `random_baseline` to show up everywhere: `list-backends` and the
`--backend` choices of `python -m multiphy.run` and the
subprocess launcher.
 An entry point that fails to load is recorded in
`registry.ENTRY_POINT_ERRORS` (name to error text) and skipped; the built-ins and the
other plugins still load. A new spec's `real_days` defaults to `True`; set it `False`
for a benchmark-only baseline so real-day launches refuse it.

For this demo, with no package to install, the backend is registered in-process:

```python
from multiphy.contract import registry
from tests.example_backend import RandomBaselineBackend
from multiphy.run.__main__ import main

registry.register(RandomBaselineBackend())
try:
    main(['list-backends'])
finally:
    registry.clear_for_tests()
```

```
functional_likelihood Functional likelihood — firing, waveform and optional event-response likelihood ratios available
pair_logistic  Pair logistic — calibrated classifier per project            available
unitmatch      UnitMatch — spike waveform and spatial shape                 unavailable: UnitMatchPy is not importable in this interpreter
dant           DANT — waveform, spike timing, optional stimulus response    unavailable: pyDANT is not importable in this interpreter
deepunitmatch  DeepUnitMatch — learned waveform features                    unavailable: DeepUnitMatch is not importable in this interpreter
random_baseline random_baseline                                              available
```

It shows up with its id in place of a label — `contract.names` has no `'backend'`
entry for it, and `names.backend_label()` returns the id rather than raising (§8 of
`contract.md`); the viewer's method list and the reports do the same. Adding one is a
one-line addition to `contract/names.py`, not required to run.

## Running it

Same in-process registration, then the real CLI entry point, on the small PRN069a
temporal-split benchmark used elsewhere in these tests:

```python
from multiphy.contract import registry
from tests.example_backend import RandomBaselineBackend
from multiphy.run.__main__ import main

registry.register(RandomBaselineBackend())
try:
    main(['run-backend', '--input',
          '/path/to/root/benchmarks/my_benchmark/dataset',
          '--backend', 'random_baseline', '--partition', 'test', '--seed', '0',
          '--root', '/tmp/adding_a_backend_demo_root'])
finally:
    registry.clear_for_tests()
```

```
Matching result: /tmp/adding_a_backend_demo_root/benchmarks/prn069a_temporal_shared_v1/runs/<stamp>_<hex>/result.json
```

The equivalent from a shell, once the backend is registered by an installed package
(the CLI's own process has no in-process registration step to run), is the same
`run-backend` invocation with no extra ceremony: `$PY -m multiphy.run
run-backend --input .../prn069a_temporal_shared_v1/dataset --backend random_baseline
--partition test --seed 0`. Only the in-process demo above is executed as part of
this document, since no such package is installed here.

## Opening a project run in a viewer

Benchmark archives and collections have no GUI view; their CLI scoring is
unchanged. A project run opens with `$PY -m multiphy.viewer --run DIR`
in the mode its `selection.json` records: a same-day run made by
`launch run PROJECT --backend ID --day DAY` opens inspect-only, and an across-days run
is curated in `curation/runs/<run>/history.csv` (see [viewer.md](viewer.md)). No code
under `viewer/` mentions `random_baseline` or any other registered backend id: it
reads `MatchResult` and takes display names from the registry. That command opens an interactive window and is not run as part
of this document; `--help` below confirms the flags it takes without opening one:

```bash
$PY -m multiphy.viewer --help
```

```
usage: python -m multiphy.viewer [-h] --run DIR

Open a completed project run in the unit-match viewer.

optional arguments:
  -h, --help  show this help message and exit
  --run DIR   a completed project run folder
```

## Day offsets

A backend that estimates drift between days sets `Provides(..., day_offsets=True)` in its spec
and saves a `registration` in its provenance, in the shape `contract.md` section 4 gives
(`### Day offsets`). The viewer's drift panel draws every day in the anchor day's frame from that
mapping; it learns whether a backend saves offsets from the spec flag alone, so the viewer still
names no backend id. A run whose group has no `registration` is drawn raw: with
`run predates offsets` when its backend sets the flag, else with `backend saves no offsets`
(the default `False`, as `random_baseline` leaves it).

An adapter around upstream code captures the drift that code applied by swapping the upstream
functions for recorders during the one call, with `_upstream.recording(module, *names)`
(`backends/_upstream.py`); it puts every original back afterwards, also after an error. The
upstream source is never edited. A capture that cannot be read raises `BackendUnavailable` or
`ContractViolation`, so the run writes no result rather than a wrong or missing offset.

## The conformance test

```bash
$PY -m pytest tests/test_contract_conformance.py -k example_backend -v
```

```
tests/test_contract_conformance.py::test_example_backend_conforms PASSED
```

`test_example_backend_conforms` registers `RandomBaselineBackend` through a
test-scoped fixture, runs the same `_assert_conforms` check every built-in backend
passes (synthetic input built from the spec, `result.validate(spec)`, every eligible
unit accounted for, `used_optional_inputs` present and a subset of `spec.optional`),
and calls `registry.clear_for_tests()` on teardown so the example never leaks into
any other test's `registry.list_specs()`.

## What you did not have to touch

- **The runner** (`run/runner.py`) — it calls `registry.get(backend_id)` and reads
  `spec.requires`/`spec.optional`/`spec.defaults()`; it has never heard of
  `random_baseline`.
- **The CLI** (`run/__main__.py`) — `list-backends` and `run-backend --backend ID`
  both come from `registry.list_specs()`/`registry.get()`, and the backend's own
  `--seed` flag was generated from its `Option` by `run/options.add_arguments`.
- **The subprocess launcher** (`run/subprocess.py`'s `build_command`) — it builds its
  argv from the same spec, not from a per-backend table.
- **The reports and the viewer** — they read `MatchResult`, not a backend id, except
  where a display string is looked up by id (§8 of `contract.md`); an unregistered
  name shows as its raw id instead of failing.
- **`contract/names.py`** — optional. Nothing above requires an entry there; it only
  changes whether the CLI and viewer show a curated label or the raw id.
