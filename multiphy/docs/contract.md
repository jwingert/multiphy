# The backend contract

Every unit-matching method — local or an official published implementation — is a
`Backend`: an object with a declared `spec` (a `BackendSpec`), an `available()` probe,
and a `run(inp, options, ctx) -> MatchResult`. The spec is data, not code, so the
runner, the CLI, and the subprocess launcher all generate their behavior from it instead of
special-casing a backend by name. This document explains the objects, their
validation rules, provenance, the runner, the result file format, errors, and how to
add a display name. Every code block below runs as pasted, one at a time, with
`$PY - <<'EOF' ... EOF` where `$PY` is the project interpreter.

## 1. The three objects and the enums

`MatchingInput` is everything a backend might consume; `BackendSpec` is what a
backend declares about itself; `MatchResult` is what every backend returns. Four
enums appear throughout: `Input` names the fields a backend can require or accept
optionally, `ScoreKind`/`TrackKind` classify what a result emits, and `Decision`
names why each candidate pair ended up selected or not.

```python
from multiphy.contract.inputs import Input, MatchingInput
from multiphy.contract.backend import BackendSpec, Option, Provides, RunContext
from multiphy.contract.results import MatchResult, ScoreKind, TrackKind, Decision

print([f for f in MatchingInput.__dataclass_fields__])
print([f for f in BackendSpec.__dataclass_fields__])
print([f for f in MatchResult.__dataclass_fields__])
print([i.value for i in Input])
print([k.value for k in ScoreKind])
print([k.value for k in TrackKind])
```

```
['units', 'group', 'candidate_pairs', 'features', 'waveform_halves', 'waveform_cellids', 'waveform_session_ids', 'waveform_spike_counts', 'geometry', 'spike_times', 'fingerprint', 'fingerprint_bin_keys', 'timing', 'checkpoint', 'waveform_provenance', 'provenance']
['id', 'version', 'requires', 'optional', 'provides', 'options', 'builtin', 'real_days']
['backend_id', 'version', 'run_id', 'units', 'pairs', 'tracks', 'unmatched', 'excluded_units', 'score', 'track_kind', 'tracking_rule', 'provenance']
['units', 'candidate_pairs', 'features', 'waveform_halves', 'geometry', 'spike_times', 'fingerprint', 'timing', 'checkpoint']
['none', 'uncalibrated_score', 'calibrated_probability']
['native', 'common']
```

`BackendSpec.requires`/`optional` are `frozenset[Input]`; every spec must require
`Input.UNITS` (`BackendSpec.__post_init__` enforces this and that no input is both
required and optional). `Provides(score_kinds, track_kinds, similarity=False, day_offsets=False)`
names what a backend's results are allowed to emit. A backend that sets `day_offsets=True` saves a
`registration` in each group's provenance (section 4), and the viewer's drift panel reads that flag
to tell a run made before its backend saved offsets from a backend that saves none.
`MatchResult.validate(spec)` checks a
result against it. `real_days` (default `True`) says whether the backend may run
across real project days; a backend that sets it `False` is a benchmark baseline, and
`run/project_run.py` (`check_real_day_run`, called by `launch run` and
`run-backend --project`) refuses it across days before the run folder exists.
Every registered backend, `functional_likelihood` and `pair_logistic` included,
sets it `True`. The two local matchers take `Input.TIMING` as optional;
`run/project_run.py` passes it from the `source/` entry of each day that has trial
events. On days without events, or days with no condition on every chosen day, either
runs without events, and `launch run` prints a warning for each fallback: `run_evidence`
(`contract/inputs.py`) decides once per recording group what the days measured, 5 s chunks
stand in for trials, and spont comes from the whole recording when a day has no baseline
windows.
`RunContext(run_id, work_dir, provenance=Provenance(), seed=0)` is
what the runner hands a backend's `run()`.

## 2. Validation rules of MatchingInput

`MatchingInput.validate()` rejects an empty `group`, reference-label columns leaked
into `units` (`source_unit_id`, `same_unit`, `partition`, `source_unit` — those belong
to benchmark labels, never to a matcher's input), candidate pairs referencing unknown
unit IDs, and malformed waveform/geometry/fingerprint shapes when those are present
(see `contract/inputs.py` for the full shape checks). `available()` reports which
`Input` values are actually set, and the runner checks `spec.requires <= inp.available()`
before calling a backend.

```python
import pandas as pd
from multiphy.contract.inputs import MatchingInput

units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})

try:
    MatchingInput(units, group='').validate()
except ValueError as exc:
    print('empty group rejected:', exc)

try:
    MatchingInput(units.assign(partition='test'), group='g').validate()
except ValueError as exc:
    print('reference-label leakage rejected:', exc)

pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['Z-9'], 'p_global': [0.5]})
try:
    MatchingInput(units, group='g', candidate_pairs=pairs).validate()
except ValueError as exc:
    print('unknown pair endpoint rejected:', exc)

inp = MatchingInput(units, group='g').validate()
print('available inputs:', sorted(i.value for i in inp.available()))
```

```
empty group rejected: group must be a nonempty string
reference-label leakage rejected: Reference label columns do not belong in matcher inputs: ['partition']
unknown pair endpoint rejected: Pair evidence references unknown unit IDs
available inputs: ['units']
```

## 3. Validation rules of MatchResult, including `validate(spec)`

`MatchResult.validate()` enforces the structural contract regardless of which backend
produced it: tracks and `unmatched` must exactly partition the eligible single units,
a track of one is rejected (singletons belong in `unmatched`), pair endpoints must be
eligible units with no self-pair and no duplicate, `decision` must be a known
`Decision`, and `match_score` must be missing or finite in `[0, 1]`. `validate(spec)`
additionally checks the result against what the backend declared: `backend_id` must
equal `spec.id`, and the emitted `score.kind`/`track_kind` must be in `spec.provides`.

```python
import pandas as pd
from multiphy.backends.local.tracker import result_from_pairs
from multiphy.contract import registry

units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'p_global': [0.95]})
result = result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', run_id='demo')
result.validate()
print('tracks:', result.tracks, 'score kind:', result.score.kind.value, 'track kind:', result.track_kind.value)

spec = registry.get('dant').spec
try:
    result.validate(spec)
except ValueError as exc:
    print('backend_id/spec mismatch rejected:', exc)

below = result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', run_id='demo2')
below.tracks = (('A-1',),)
try:
    below.validate()
except ValueError as exc:
    print('singleton track rejected:', exc)
```

```
tracks: (('A-1', 'B-1'),) score kind: uncalibrated_score track kind: common
backend_id/spec mismatch rejected: Result backend_id 'unitmatch' does not match spec 'dant'
singleton track rejected: Singleton tracks belong in unmatched
```

## 4. Provenance: typed fields, `extra`, `get`

`Provenance` carries a fixed set of typed fields (`dataset_id`, `partition`, `group`,
`calibration_status`, `parameters`, …) plus a free-form `extra` mapping for anything
else a backend wants to record (`used_optional_inputs`, `threshold`,
`feature_stream`, …). `to_json()` flattens `extra` into the same mapping so a saved
result reads like one flat dict; it raises if an `extra` key would shadow a typed
field. `.get(key, default)` reads either a typed field or an `extra` key by name, so
call sites do not need to know which kind a given key is.

```python
from multiphy.contract.provenance import Provenance

p = Provenance(dataset_id='demo', partition='test').merged(extra={'threshold': 0.9})
print('typed field:', p.get('dataset_id'))
print('extra field:', p.get('threshold'))
print('missing with default:', p.get('nope', 'fallback'))
print('to_json:', p.to_json())
print('round trip equal:', Provenance.from_json(p.to_json()) == p)

try:
    p.merged(extra={'dataset_id': 'shadow'}).to_json()
except ValueError as exc:
    print('extra cannot shadow a typed field:', exc)
```

```
typed field: demo
extra field: 0.9
missing with default: fallback
to_json: {'dataset_id': 'demo', 'partition': 'test', 'threshold': 0.9}
round trip equal: True
extra cannot shadow a typed field: extra provenance keys shadow typed fields: ['dataset_id']
```

### Day offsets (`registration`)

A backend whose spec sets `day_offsets=True` saves a `registration` mapping in each group's
provenance: the drift between the group's days that the backend measured or its upstream code
applied. The runner files each group's provenance under `group_runs[group]['provenance']` of the
merged result, and the viewer's drift panel draws it ([viewer.md](viewer.md)). Every value is a
plain `float`, `int`, `str`, `list` or `dict` (the provenance is written as JSON).

| key | value |
|---|---|
| `method` | `str`: how the drift was measured; `native` is the upstream package's own drift correction |
| `offsets_um` | `{day: µm}`: one rigid offset per day |
| `shank_offsets_um` | `{day: [µm per shank]}`: a day corrected per shank (never also in `offsets_um`) |
| `depth_scale` | `{day: factor}`: a linear depth term; a day without one has scale `1.0` |
| `unit_offsets_um` | `{cellid: µm}`: the shift applied to that unit itself; its corrected depth is `depth` plus this |
| `unit_shanks` | `{cellid: shank}`: the shank each unit was corrected on |
| `shank_width_um` | `float`: shank `i` holds `i × shank_width_um < x < (i + 1) × shank_width_um`; any other `x` is shank `0` |
| `diagnostics` | `{group: {...}}`: `groups` (lists of days linked by measured offsets), a `note`, and the producer's own records |
| `pair_drifts` | `[{session_a, session_b, depth_um or shank_depth_um, x_um or shank_x_um, n_matches}]`: the drift of each day pair, in place of `offsets_um` |

A unit's corrected depth is `depth_scale` × `depth` + offset, where the offset is its day's
`offsets_um` entry or its shank's entry of `shank_offsets_um` (a unit in `unit_offsets_um`: `depth`
plus its own shift). In a day-map registration (no `pair_drifts`), corrected depths are compared
only within one `diagnostics[group]['groups']` list; days in different lists share no frame. A
pair registration (`pair_drifts`) saves no `groups`: each entry's drift is added to its
`session_b` to align it directly with `session_a`, only for a pair with matches, and sets no
shared frame across pairs (drift never chains through a third day).

| producer | `method` | saves |
|---|---|---|
| `functional_likelihood` | `matches`, `profile` or `none` | `offsets_um` per day; `diagnostics[group]`: `groups` and the measured pairs |
| `pair_logistic` | `matches`, `profile` or `none` | as `functional_likelihood` (the same `functional_offsets` registration) |
| `unitmatch` | `native` | `offsets_um`, per-shank days in `shank_offsets_um` with `unit_shanks` and `shank_width_um`; `diagnostics[group]`: `groups` (a day pair UnitMatchPy applied no drift to starts a new list), `drifts` (`[day_a, day_b, depth, x, n_pairs]`), `shank_drifts` (`[day_a, day_b, shank, depth, x]`), `sources` |
| `dant` | `native` | `offsets_um` = `-Constant` of the motion DANT kept; with a linear term also `depth_scale` = `1 - LinearScale × Linear` and `unit_offsets_um` (the shift at DANT's own unit location); `diagnostics[group]`: `groups`, `pairs`, `motion`, `sources`. When DANT kept its initial motion: no offsets and the note `dant kept initial motion` |
| `deepunitmatch` | `native` | `pair_drifts`, one per day pair (the drift added to `session_b`, as `depth_um` or per shank `shank_depth_um`; `None` depths and `n_matches` 0 for a pair with no matches), `unit_shanks`, `shank_width_um`; `offsets_um` empty; `diagnostics[group]`: `sources` |

`sources` lists the upstream files the drift came from (`module`, `path`, `sha256`). A capture
that cannot happen (a missing upstream function, an unexpected number of calls, a drift of the
wrong shape or not finite, …) raises `BackendUnavailable` or `ContractViolation` and the run
writes no result: a saved offset is never wrong and never silently missing.

## 5. The runner's nine steps and the run directory contents

`run.runner.run(dataset, backend_id, options, *, group_column, partition=None, ...)`
(`multiphy/run/runner.py`) does, in order: (1) resolve the backend and
check `available()`; (2) validate `options` against the spec; (3) select the
requested partition, if any; (4) confirm every unit has a `group_column` value; (5)
build the base `Provenance` from the dataset's metadata; (6) for each group, build a
`MatchingInput` and check `spec.requires <= inp.available()`, raising `MissingInputs`
before any backend runs (if the dataset's `waveform_bundle()` raised, its message is
kept in the input provenance as `waveform_bundle_error`, and a `MissingInputs` for a
waveform-derived input ends with `; waveform bundle could not be built: <error>`); (7) create the output directory and write the initial
`run_manifest.json` and `METHODS.md`; (8) run the backend once per group inside a
contained cache/tempdir, validating each result against the spec and the contract
invariants (no two observations from one calendar day in a track, the run id is
preserved); (9) merge the per-group results with `merge_results`, `save_result()` the
combined `MatchResult`, and write `evaluation.json` when labels exist or
`availability.json` otherwise; the manifest is updated to `completed` (or `failed`
with the exception recorded) and rewritten one last time in a `finally` block.

The CLI forwards only a `--checkpoint` as an extra input; it builds no `TrialTiming`.
Both local matchers therefore run without events: 5 s chunks stand in for trials, spont
comes from the whole recording, and the reasons are saved as `run_evidence` in each
group's provenance. Only `launch run` prints them as warnings; this command prints none:

```bash
$PY -m multiphy.run run-backend \
  --input /auto/data/nems_db/unit_match/benchmarks/prn069a_temporal_shared_v1/dataset \
  --backend functional_likelihood --partition test \
  --root /tmp/contract_doc_demo_root
```

```
Matching result: /tmp/contract_doc_demo_root/benchmarks/prn069a_temporal_shared_v1/runs/<run>/result.json
```

The group's `run_evidence` in that result's `group_runs` provenance:

```
{"events": false, "spont": "whole_recording",
 "shared": {"recording0000_part000+recording0000_part001": 0},
 "reasons": ["no events · shared conditions per day pair: recording0000_part000+recording0000_part001 0",
             "spont · whole recording · days without baseline windows: recording0000_part000, recording0000_part001"]}
```

For a run with events, pass the timing yourself from Python. `runner.run` requires the
keyword `group_column`; the CLI's default for a benchmark is `recording_group`, and for a
benchmark it runs the `test` partition:

```python
from multiphy.run import runner

# dataset: the benchmark dataset read from the --input folder above.
# timing:  {session_id: TrialTiming}, one entry for every session in dataset.units,
#          e.g. built with nems_lbhb.unit_match_lbhb.prepare_baphy.recover_baphy_timing(spec).
output = runner.run(dataset, 'functional_likelihood', {},
                    group_column='recording_group', partition='test',
                    root='/tmp/contract_doc_demo_root',
                    extra_inputs={'timing': timing})
```

A completed run folder holds `result.json`, `run_manifest.json`, `METHODS.md` and
`evaluation.json` (or `availability.json`).

## 6. Result file format 2

`run.run_dir.save_result()` writes format 2: `{"format": "nems_unit_match_result",
"schema_version": 2, "sha256": <digest of "result">, "result": {...}}`, where
`result` holds `backend_id`, `version`, `run_id`, `units`, `pairs`, `tracks`,
`unmatched`, `excluded_units`, `provenance`, `score`, `track_kind`, and
`tracking_rule`. `load_result()` reads format 2 only and always returns a validated
`MatchResult`; any other format or schema version raises `ValueError` (`unsupported
matching result format or schema version`), and the run catalog does not list such a
run, nor one whose backend id is not registered.

```python
import pandas as pd
from multiphy.backends.local.tracker import result_from_pairs
from multiphy.run import run_dir

units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'p_global': [0.95]})
result = result_from_pairs('unitmatch', '1', units, pairs, score_column='p_global', run_id='demo')
path = run_dir.save_result(result, '/tmp/contract_doc_result.json', overwrite=True)
loaded = run_dir.load_result(path)
print('round trip matches:', loaded.backend_id == result.backend_id, loaded.tracks == result.tracks)

import json
doc = json.loads(open(path).read())
print('format:', doc['format'], 'schema_version:', doc['schema_version'])
print('result keys:', sorted(doc['result']))
```

```
round trip matches: True True
format: nems_unit_match_result schema_version: 2
result keys: ['backend_id', 'excluded_units', 'pairs', 'provenance', 'run_id', 'score', 'track_kind', 'tracking_rule', 'tracks', 'units', 'unmatched', 'version']
```

## 7. Errors and when each is raised

`contract.errors` defines three: `BackendUnavailable` (a backend's `available().ok`
is `False` — an optional package, checkpoint, or environment is missing; raised by
`runner.run()` before starting, and by an adapter's own `run()` if called directly),
`MissingInputs` (a group's `MatchingInput.available()` does not cover
`spec.requires`; raised by `runner.run()` once per group, before any backend runs),
and `ContractViolation` (a result, option set, or run broke a rule the contract
states — an unknown or missing required option in `run.options.validate()`, including a
misspelled keyword passed to `runners.run_backend_dataset`, which is passed through
rather than dropped, a merged
result whose backend version changed between groups, or a backend that changed the
run id or dropped an eligible unit).

```python
import pandas as pd
from multiphy.contract import registry
from multiphy.contract.errors import BackendUnavailable, ContractViolation, MissingInputs
from multiphy.contract.inputs import MatchingInput
from multiphy.run import options as opt

spec = registry.get('functional_likelihood').spec
try:
    opt.validate(spec, {'threshold': 0.9, 'not_an_option': 1})
except ContractViolation as exc:
    print('ContractViolation (unknown option):', exc)

units = pd.DataFrame({'cellid': ['A-1'], 'session_id': ['A'], 'su_type': ['SU']})
inp = MatchingInput(units, group='g').validate()
missing = spec.requires - inp.available()
try:
    if missing:
        raise MissingInputs(spec.id, 'g', missing)
except MissingInputs as exc:
    print('MissingInputs (the runner.run() check before any run):', exc)

backend = registry.get('unitmatch')
avail = backend.available()
print('unitmatch available:', avail.ok, '-', avail.reason)
try:
    backend.run(inp, spec.defaults(), None)
except BackendUnavailable as exc:
    print('BackendUnavailable:', exc)
```

```
ContractViolation (unknown option): backend 'functional_likelihood' does not take options ['not_an_option']
MissingInputs (the runner.run() check before any run): Backend 'functional_likelihood' cannot run on group 'g': missing inputs features, geometry
unitmatch available: False - UnitMatchPy is not importable in this interpreter
BackendUnavailable: UnitMatchPy is not importable in this interpreter
```

## 8. Names: how to add a display string

`contract.names.NAMES` is the one table of human-readable strings, keyed by
namespace (`'backend'`, `'tracking'`, `'evidence'`, `'metric'`, `'feature'`,
`'dant_preset'`, `'definitions'`) then by internal id. `display(namespace, key)`
returns the label and raises `KeyError` for anything not in the table, so an
unexplained internal code can never reach a figure or the viewer. Backend ids are
the one exception: any installed backend may lack an entry, so every place that names
a backend (the viewer's method list and hint, `backends.list_backends()`,
`comparison_report.method_name`, and the CLI through
`runner.backend_label`) calls `backend_label(id)`, `backend_short(id)` or
`backend_help(id)`, which return the `NAMES['backend']` entry when there is one and
otherwise the id itself (an empty string for help). Add a display string for a new
backend id (or any other new code) by adding a `Name(...)` entry under the right
namespace in `contract/names.py`; nothing else needs to change.

```python
from multiphy.contract.names import (backend_help, backend_label, backend_short,
                                                 display, help_text, short)

print(display('backend', 'functional_likelihood'))
print(short('backend', 'functional_likelihood'))
print(help_text('backend', 'functional_likelihood'))
try:
    display('evidence', 'no_such_stream')
except KeyError as exc:
    print('unexplained code refused:', exc)
print(backend_label('functional_likelihood'))
print(repr(backend_label('no_such_backend')), repr(backend_short('no_such_backend')),
      repr(backend_help('no_such_backend')))
```

```
Functional likelihood — firing, waveform and optional event-response likelihood ratios
Functional likelihood
Likelihood ratios from own halves (trials in event runs, else 5 s chunks) and same-day neighbours; a mixture posterior per day pair.
unexplained code refused: 'no_such_stream'
Functional likelihood — firing, waveform and optional event-response likelihood ratios
'no_such_backend' 'no_such_backend' ''
```

A backend with no entry under `'backend'` still runs and is shown by its id:
`list-backends`, the viewer's method list and the reports all go through the helpers
above, so a freshly installed third-party or test backend never crashes them (see
`docs/adding_a_backend.md`).
