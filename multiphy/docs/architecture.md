# Architecture and extensions

This guide maps the package's layers and extension points. For installation and a
quick start, return to the [README](../../README.md). Implementation priorities,
benchmark contracts and remaining work are in [DEVELOPMENT.md](../DEVELOPMENT.md).

## Layers

| module | role | imports |
|---|---|---|
| `features.py` | spiking statistics: spike width, trial split halves, interval histogram and autocorrelogram | numpy/pandas |
| `contract/` | the backend contract: `MatchingInput`, `BackendSpec`, `MatchResult`, `Provenance`, the registry, and every display name | numpy/pandas only |
| `backends/` | single-unit results: the local `functional_likelihood` and `pair_logistic` matchers plus the UnitMatch/DANT/DeepUnitMatch adapters, registered on import | numpy/pandas; upstream packages imported lazily, only when run |
| `run/` | one runner loop, the option schema shared by the CLI and subprocess launcher, the run directory and result file format | numpy/pandas |
| `evaluation.py` | held-out source-cluster evaluation and partition checks | numpy/pandas |
| `store.py` | pair canonical IDs, chains and consistency checks for advanced pair annotations | numpy/pandas only |
| `curation.py` | one writer per run: history, undo/redo, lock, export | numpy/pandas only |
| `viewer/` | the PySide6 viewer: one `App`, the `across_days` and `same_day` modes, registered panels; see the [viewer package map](viewer.md#package-map) | the above, plus Qt (`data.py`, `display.py` and `modes/` import none) |
| `project/selection.py` | a `Selection` of one or more extracted project days: its dataset, trial timing, conditions and `selection.json` | numpy/pandas |
| `run/project_run.py` | runs one registered backend on a selection into its `runs/` folder, and lists a selection's runs | lazy backend imports |
| `launch/` | the launcher: `__main__.py` (the CLI), `commands.py`, `plugins.py`, `forms.py` (forms read from the CLI parser) and `window.py` (the launcher window, `launch gui`) | Qt only in `window.py` |

The [backend contract](contract.md) describes input validation, results, provenance
and the runner. `backends/local/evidence.py` and `store.py` import no UI toolkit
and no recording loader, so the science is testable without a display and the
viewer is replaceable.

## Plugins

A plugin module with `register(api)` adds source readers and launcher commands.
The launcher finds plugins in `<root>/multiphy.json` and the `multiphy.plugins`
entry points. Follow [Adding a source](adding_a_source.md) for readers and
launcher plugins, or [Adding a backend](adding_a_backend.md) for matching methods.

## Advanced pair annotations and track export

In the viewer's optional **Pair review (advanced)** mode, pair verdicts are stored
as edges. `store.chains()` turns accepted pair edges into connected components for
the **Pair chains and consistency (advanced)** panel. If day 1 to day 2 and day 2
to day 3 are accepted but day 1 to day 3 is rejected, `store.inconsistencies()`
reports that triangle instead of resolving it silently. It also reports a closing
edge that has not been accepted.

These chains describe pair annotations. Current track membership and review
decisions are rebuilt separately by `curation.py` from the run's history.
`launch export` exports members of accepted current tracks, including accepted
singletons; pair annotations do not change membership or export eligibility.
See the [curation guide](viewer.md#match-review-and-export) and
[export command and CSV schema](project_folder.md#export).
