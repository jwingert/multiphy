<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the logo heads the README (logo) | date: 2026-10-13] -->
<img src="viewer/logo/logo.svg" alt="multiphy logo: one unit's waveform tracked across three days" width="128">
<!-- [AGENT EDIT END] -->

# unit_match — cross-day single-unit matching and curation

Matching units across days in a semi-chronic preparation, where the probe goes
in afresh each day and the brain moves. The stimulus-response fingerprint is a
central source of evidence; multichannel waveform and firing-pattern evidence
can be compared through the same interface. Performance on chronic implants or
within-session temporal splits does not by itself establish reinsertion accuracy.

## Comparing methods and curating unmatched units

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer runs no matcher; the local method is named by mechanism (Task 3) | date: 2026-09-27] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the cache builder is retired; the viewers open project runs (Plan C Task 7) | date: 2026-10-02] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one viewer runs no matcher (viewer-split final review fix round 1) | date: 2026-10-07] -->
The viewer runs no matcher. Runs arrive through the viewer's run list or
<!-- [AGENT EDIT END] -->
**Import**. Only explicit `SU` labels are eligible; unknown quality and
multi-unit labels are excluded. These labels come from spike-file comments and still require quality review.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: both local matchers are allowed on real days; the limit is per-day trial timing (final review fix round 1) | date: 2026-09-28] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic holds out sessions, not only days (Follow-up Task 4) | date: 2026-09-29] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: functional_likelihood scores event-response likelihood ratios (launcher-gui Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: functional_likelihood scores firing, waveform and optional event-response likelihood ratios from trial or 5 s chunk halves (event-optional Task 13) | date: 2026-10-10] -->
Two local methods learn from every day of their project. `functional_likelihood`
scores firing, waveform and optional event-response likelihood ratios; its same/different
references are each unit's own halves (trials in event runs, else 5 s chunks) and same-day
neighbours within `radius_um`, pooled over the project, and a mixture fitted on every gated
candidate of a day pair gives the posterior. `pair_logistic` fits a logistic model on the
same labeled examples from alternate sessions (days, or a same-day run's two halves) and
calibrates it on the held-out sessions.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic waveform properties on the peak-threshold neighbourhood by default (threshold neighbourhood) | date: 2026-10-04] -->
Its waveform properties compare each pair on the union of both units' channel neighbourhoods (channels within 150 µm of the peak whose largest voltage in the spike window is at least 10% of the peak channel's, plus the 5 nearest), the waveform correlation over the spike window only (`waveform_channels` `peak_threshold`, the default); `full_array` uses every channel and sample.
<!-- [AGENT EDIT END] -->
Both register each day's
depth from confident matches before the proximity gate. Both track with the
common tracker: deterministic best-first assignment with mutual candidate
margins, one unit per recording session, and rejection of merges containing
strongly contradictory measured pairs. It permits unmatched units and missing
sessions; it is not a globally optimal tracker. `functional_likelihood` scores
are uncalibrated posteriors, not validated identity probabilities.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: launch run passes per-day trial timing (Plan B Task 10) | date: 2026-09-28] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: both take trial timing as optional and run without events, with a warning, on days without events or a shared condition (event-optional Task 13) | date: 2026-10-10] -->
Both declare `real_days=True` and take per-day trial timing (`TrialTiming`) as optional;
`launch run` passes it from the `source/` entry of each day with trial events. On days
without events, or days with no condition on every chosen day, either runs without
events: 5 s chunks stand in for trials, spont comes from the whole recording when a day
has no baseline windows, and no event-response stream is fitted. `launch run` prints each
fallback as a `warning · <reason>` line, and the run saves the reasons as `run_evidence`
in its provenance.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each across-days run has one curation history; the old ledgers are gone (run-curation Task 10) | date: 2026-10-05] -->
Split partitions the selected members from the remainder. Drop detaches a
member into an unmatched singleton. Merge combines selected groups, including
unmatched units. Tables and plots refresh together after edits. Each across-days
run has its own history, `curation/runs/<run>/history.csv`, so switching runs
does not touch another run's curation.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no method or evidence-stream switching in the viewer (an empty pair from Task 3 of 2026-09-27, split from one line onto two so the tag check pairs it) (day-table-keys Task 4) | date: 2026-10-14] -->
<!-- [AGENT EDIT END] -->

`backends.MatchResult` provides the common units, pair evidence, tracks,
unmatched units, and provenance contract. `external_backends` calls genuine
optional UnitMatch, DANT, and DeepUnitMatch implementations; missing packages,
raw waveforms, or checkpoints produce explicit errors rather than substitutions.
Published algorithm outputs must state whether native tracking or the common
local tracker formed their groups.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no cache-builder caches to describe (Plan C Task 7) | date: 2026-10-02] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no cache-scoped imports: a viewer imports from runs of its own selection (Plan C Task 7) | date: 2026-10-02] -->
Use **Import matching result…** in a viewer to load the `result.json` of another
completed run of the same selection. Its native match scores remain
authoritative; the viewer adds the measured stimulus and physiology evidence
for inspection. Each imported method/run has
its own curation history. UnitMatch and DeepUnitMatch default to their official
intermediate tracking rule and 0.5 threshold; the local tracker's threshold is
separate. DANT cluster membership is displayed without inventing a probability.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: real days are project runs started by launch run; multiday archives, their runner and exports are retired (Plan B Task 10) | date: 2026-09-28] -->
Real recording days live in a **project** (see
[docs/project_folder.md](docs/project_folder.md)): `launch init` adds each day once,
`launch extract` extracts its features once, and `launch run` runs any registered
backend on a selection of extracted days. `python -m nems_lbhb.unit_match.run
list-backends` names every registered backend and whether this interpreter can run
it; one that cannot runs in `--python EXE` or the verified lab interpreter.

```bash
# every extracted day of project M1, matched across days
python -m nems_lbhb.unit_match.launch run M1 --backend functional_likelihood
# chosen days, or one day's two temporal halves
python -m nems_lbhb.unit_match.launch run M1 --backend unitmatch --days D1 D2
python -m nems_lbhb.unit_match.launch run M1 --backend dant --day D1
```

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher window runs the same commands (launcher-gui Task 11) | date: 2026-10-07] -->
`python -m nems_lbhb.unit_match.launch gui` runs the same commands from one window and
opens runs (see `docs/project_folder.md`).
<!-- [AGENT EDIT END] -->

Each run is a new folder, never replaced: across-days runs in the project's
`runs/`, same-day runs in `days/<day>/runs/`. The folder holds `selection.json`
(the project identity, the days, their feature ids and the excluded units) and
`pairs.csv` before the backend starts, and `result.json` when it ends. A failed
run keeps its folder with manifest status `failed`; running again makes a new
folder. Every refusal comes before the folder exists.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one viewer command opens either kind of run (viewer-split Task 10) | date: 2026-10-07] -->
The viewer opens completed project runs and starts none; one command opens either kind
(see [docs/viewer.md](docs/viewer.md)):

```bash
python -m nems_lbhb.unit_match.viewer --run /path/to/M1/runs/<run>            # across days
python -m nems_lbhb.unit_match.viewer --run /path/to/M1/days/D1/runs/<run>    # one day's halves
```
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one history per across-days run, its lock and launch export; same-day runs are inspect-only; archives have no GUI view (run-curation Task 10) | date: 2026-10-05] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one viewer lists the runs; the across_days mode curates (viewer-split final review fix round 1) | date: 2026-10-07] -->
A run opens only when its `selection.json` still matches the project. The
viewer lists the completed runs of the same selection, and imports a result only
from one of them. In across_days mode the viewer curates its run in
<!-- [AGENT EDIT END] -->
`curation/runs/<run>/history.csv`: one line per curator action, undo and redo
included, the file replaced whole on each write. A merge may put two
observations from one calendar day into one track; the track row flags it.
`history.lock` lets one viewer write: a second viewer of the run opens
read-only, and a lock from another host can be taken over.
`launch export PROJECT --run RUN [--out PATH]` writes the accepted tracks to
`curation/runs/<run>/exports/accepted_tracks_<YYYYMMDD-HHMMSS>.csv`. Same-day
runs are inspect-only. Across-days runs have no temporal benchmark labels,
partitions or accuracy estimates; a same-day run is scored against its day's
source labels.

Benchmark archives and collections have no GUI view; their CLI scoring is
unchanged. Extraction, dependency installation and model training use explicit
commands; see [MATCHING_INTERFACE.md](MATCHING_INTERFACE.md).
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: LBHB wells reach the project path through the lab plugin (Plan C Task 7) | date: 2026-10-02] -->
## LBHB wells

The lab plugin `nems_lbhb.unit_match_lbhb` turns an LBHB well into project days.
Name it in `<root>/multiphy.json`, where `<root>` is `--root DIR` or the default
cache root:

```json
{"plugins": ["nems_lbhb.unit_match_lbhb"]}
```

A project named `<animal>_w<well>_<RUNCLASS>` holds one day per site of that well:
all good recordings of the run class at the site, concatenated in run order, with
the day's FTC tone tuning when it was sorted in the same KiloSort job.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the optional _tag | date: 2026-10-08] -->
An optional `_<tag>` after the run class (`Prince_w16_BNT_e2e`) makes another project of the
same well, for a re-sort or a test: a day is written once, so a changed day needs a new project.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the LBHB example opens its run with the viewer package (viewer-split Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the LBHB example names the launcher window as an alternative (launcher-gui Task 11) | date: 2026-10-07] -->
```bash
python -m nems_lbhb.unit_match.launch lbhb-init Prince_w16_BNT
python -m nems_lbhb.unit_match.launch extract Prince_w16_BNT --tier raw
python -m nems_lbhb.unit_match.launch run Prince_w16_BNT --backend functional_likelihood
python -m nems_lbhb.unit_match.viewer --run DIR
# or all of it from one window:
# python -m nems_lbhb.unit_match.launch gui Prince_w16_BNT
```
<!-- [AGENT EDIT END] -->

`lbhb-init` skips days already in the project, so a rerun completes a failed one.
`python -m nems_lbhb.unit_match.viewer --run DIR` opens any project run in its mode.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: codex | user: wingertj | reason: distinguish generated benchmark and multiday output locations | date: 2026-09-18] -->
See [DEVELOPMENT.md](DEVELOPMENT.md) for implementation priorities, the temporal
split benchmark contract, and the remaining data/training work.
See [BENCHMARKS.md](BENCHMARKS.md) for extraction, comparison, and training
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: project runs and curation live in the project (Plan B Task 10) | date: 2026-09-28] -->
commands. Runner code stays in this package; generated benchmarks and checkpoints
stay under `/auto/data/nems_db/unit_match/benchmarks/`, and each project keeps its
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each run's curation lives under curation/runs/<run>/ (run-curation Task 10 fix round 1) | date: 2026-10-05] -->
own runs under `runs/` and `days/<day>/runs/` and its curation under `curation/runs/<run>/`.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

New benchmark inputs use one shared filtered, median-referenced, unwhitened
waveform representation. Buffered filtering precedes snippet cropping. Only
waveform summaries persist; methods reuse the same archive and perform any
additional method-specific transformations in memory. This tests the methods
on shared inputs, rather than reproducing each author's extraction recipe.

## Layers

<!-- [AGENT EDIT START | agent: codex | user: wingertj | reason: identify actual-day archive and shared curator entry points | date: 2026-09-18] -->
| module | role | imports |
|---|---|---|
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no cache-builder layers; the lab feature functions live in the plugin (Plan C Task 7) | date: 2026-10-02] -->
| `features.py` | probe geometry, spiking statistics and stimulus keys | numpy/scipy |
| `nems_lbhb.unit_match_lbhb` | the LBHB plugin: `lbhb.baphy` reader, `lbhb-init`, well listing (`blocks.py`) and BAPHY features (`baphy_features.py`) | celldb and BAPHY, imported where queried |
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: replace evidence.py/backends.py/external_backends.py rows with the contract/, backends/, run/ layers they moved into (Task 13) | date: 2026-09-22] -->
| `contract/` | the backend contract: `MatchingInput`, `BackendSpec`, `MatchResult`, `Provenance`, the registry, and every display name | numpy/pandas only |
| `backends/` | single-unit results: the local `functional_likelihood` and `pair_logistic` matchers <!-- [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic listed (final review fix round 1) | date: 2026-09-28] --> plus the UnitMatch/DANT/DeepUnitMatch adapters, registered on import | numpy/pandas; upstream packages imported lazily, only when run |
| `run/` | one runner loop, the option schema shared by the CLI and subprocess launcher <!-- [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no Qt dialog reads the option schema since backend_job.py was removed (run-curation final review 2) | date: 2026-10-06] -->, the run directory and result file format | numpy/pandas |
<!-- [AGENT EDIT END] -->
| `evaluation.py` | held-out source-cluster evaluation and partition checks | numpy/pandas |
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: curation.py row; store.py keeps pair ids, chains and consistency (run-curation Task 10) | date: 2026-10-05] -->
| `store.py` | pair canonical ids, chains, consistency | numpy/pandas only |
| `curation.py` | one writer per run: history, undo/redo, lock, export | numpy/pandas only |
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer package row replaces gui.py (viewer-split Task 10) | date: 2026-10-07] -->
| `viewer/` | the PySide6 viewer: one `App`, the `across_days` and `same_day` modes, registered panels; see `docs/viewer.md` | the above, plus Qt (`data.py`, `display.py` and `modes/` import none) |
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the retired archive and archive-runner rows go; project selection and run rows (Plan B Task 10) | date: 2026-09-28] -->
| `project/selection.py` | a `Selection` of one or more extracted project days: its dataset, trial timing, conditions and `selection.json` | numpy/pandas |
| `run/project_run.py` | runs one registered backend on a selection into its `runs/` folder, and lists a selection's runs | lazy backend imports |
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launch/ package row (launcher-gui Task 11) | date: 2026-10-07] -->
| `launch/` | the launcher: `__main__.py` (the CLI), `commands.py`, `plugins.py`, `forms.py` (forms read from the CLI parser) and `window.py` (the launcher window, `launch gui`) | Qt only in `window.py` |
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: multiday_viewer.py is deleted; the viewer/ row covers both modes (viewer-split Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: point to the new contract and adding-a-backend docs (Task 13) | date: 2026-09-22] -->
The contract is documented in `docs/contract.md`; adding a method is `docs/adding_a_backend.md`.
<!-- [AGENT EDIT END] -->

`evidence.py` and `store.py` import no UI toolkit and no recording loader, so
the science is testable without a display and the viewer is replaceable.

## Use

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: curation opens a project run; the cache-builder commands are gone (Plan C Task 7) | date: 2026-10-02] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: curation opens through the viewer package (viewer-split Task 10) | date: 2026-10-07] -->
```bash
# curate a completed project run
python -m nems_lbhb.unit_match.viewer --run DIR
```
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the keys: a verdict steps down; Up / Down step in the day table (native-same-day Task 11) | date: 2026-10-11] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Up / Down walk the picked unit's day (browse-same-day Task 2) | date: 2026-10-12] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Up / Down step the same-day column after a click there (browse-same-day Task 3) | date: 2026-10-12] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one saved key per action: WASD, F, Esc and X string the day table, E R U pair verdicts, Q and the Shift keys curate tracks (day-table-keys Task 4) | date: 2026-10-14] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the keyboard line names Help ▸ Keyboard shortcuts, where the keys are changed (day-table-keys Task 5) | date: 2026-10-14] -->
Keyboard (defaults; `Help` ▸ `Keyboard shortcuts` changes them): `W` / `S` up / down in the active day, `A` / `D` previous / next day, `F` string the picked unit's track, `Esc` clear the string, `X` drop the active day's unit; `E` accept, `R` reject, `U` unsure (a pair verdict, then a step down); `Q` merge, `Shift+E` / `Shift+R` accept / reject the picked unit's track, `Shift+X` unmatch the active unit; `Ctrl+Z` / `Ctrl+Shift+Z` undo / redo.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: curation lives in the project, per backend and run (Plan C Task 7 fix round 1) | date: 2026-10-02] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one history per across-days run and launch export replace the pair and cluster tables (run-curation Task 10) | date: 2026-10-05] -->
An across-days run's curation is `<project>/curation/runs/<run>/history.csv`,
one line per curator action, undo and redo included; the file is replaced whole
on each write. Export the accepted tracks with:

```bash
python -m nems_lbhb.unit_match.launch export PROJECT --run RUN [--out PATH]
```
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

## Choosing the sounds

The fingerprint comes from one recording block, named by runclass, and so do
the rates and the spiking statistics: the physiology references are calibrated
on within-session split halves, so a rate from one block and an interval
histogram from another would compare two different states.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no Sounds-box cache building (Plan C Task 7) | date: 2026-10-02] -->
Each choice is therefore its own project: the run class is the last part of an
LBHB project name.
<!-- [AGENT EDIT END] -->

Checked at Prince and Reishi, counting stimuli with at least five repeats:

| runclass | stimuli per site | joins across days |
|---|---|---|
| BNT | 12–16 | on sound path, test range and binaural setting |
| OLP | 48–282 | on epoch names, which drift; see below |
| NFB | 14–92 | on epoch names; no set-identifying fields in celldb |

BNB is deliberately absent: it presents one noise burst whose epoch is named
after its level (`STIM_316` one day, `STIM_3162` the next), so two days share
no stimulus to compare.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the 290-pair count came from the retired cache's join (Plan C Task 7 fix round 2) | date: 2026-10-02] -->
OLP joins only where the naming convention held. At Prince wells 14 and 15
one day writes `01Chimes-0-1_16Bell-0-1` and the next writes
`01Chimes-0-1-1_16Bell-0-1-1`, an added field with no way to tell from the
names alone whether the extra condition is the same sound.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no cache builder records it (Plan C Task 7) | date: 2026-10-02] -->
Rather than guess, such sounds are not joined.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the project reader keeps literal conditions; stimulus_key normalised only the retired cache (Plan C Task 7 fix round 2) | date: 2026-10-02] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: lbhb.baphy version 2 joins BNT spellings (bignat-names Task 2) | date: 2026-10-11] -->
From `lbhb.baphy` version 2, BNT spellings are joined: a BigNat run writes one
wav:channel per speaker with attenuation dropped, so mono `00seq4.wav`,
binaural `00seq4.wav:1:0+NULL:2:0` and its 10 dB variant all become
`00seq4.wav:1+NULL:2`. A site presenting both spellings responded to them alike
(r 0.52 against 0.013 for two different sounds). Days extracted at version 1
keep their literal names; see below.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

## The criteria

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: candidates come from the backend's proximity gate, not a cache (Plan C Task 7 fix round 2) | date: 2026-10-02] -->
The historical thresholds in `evidence.Criteria` were selected using
within-session split halves. The proximity threshold limits candidate
generation. Rate, interval, and width gates are displayed diagnostics; the
automatic backend does not require every diagnostic gate to pass.
<!-- [AGENT EDIT END] -->

| criterion | threshold | rejects of gated neighbours |
|---|---|---|
| proximity | 40 μm after rigid depth alignment | — |
| spontaneous rate, abs diff of log | 0.293 | 84% |
| inter-spike-interval histogram, cosine | 0.978 | 88% |
| spike width, abs diff | 0.15 ms | displayed diagnostic |

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: three streams of evidence, the event response only with events (event-optional Task 13) | date: 2026-10-10] -->
Score, applied to whatever survives: three streams of evidence (the event response
only in runs with events), each property contributing a likelihood ratio built from its own same-neuron and
different-neuron references, so a value that merely misses a threshold is
penalised rather than discarded and missing evidence is neutral.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the stream is the event response (launcher-gui Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: physiology adds the session rate with events; the waveform stream; tags outside the table rows (event-optional Task 13 fix round 1) | date: 2026-10-10] -->
| stream | properties |
|---|---|
| event response | fingerprint correlation |
| physiology | spontaneous rate, interval histogram, autocorrelogram + session rate (events only) |
| waveform | waveform correlation, footprint correlation, spike width (peak channels) |
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the event response, not sound, comes out heavier (launcher-gui Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams are combined by fitted weights; the Prince well 16 weights predate the waveform stream (event-optional Task 13) | date: 2026-10-10] -->
The streams are combined by weights fitted on the reference pairs where they
exist, not asserted. The event response comes out heavier wherever it has been fitted
(1.23 against 0.70 for physiology at Prince well 16, fitted before the waveform stream).
Rate, interval histogram and autocorrelogram are not independent — rate sets the interval
scale and the autocorrelogram restates the short intervals — so summing their ratios would
be overconfident; fitting the weights and then calibrating the combined score
against the same references absorbs that.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a pair the event response convinces on (launcher-gui Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each stream the run fitted sits beside the combined probability (event-optional Task 13) | date: 2026-10-10] -->
All the probabilities are on screen: the combined one is the headline, the
three streams (the event response only in runs with events) sit beside it, and the
mixture panel shows the reference and observed distributions behind each.
A pair the event response convinces on but the
physiology does not is visible as such rather than averaged away.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the v2 references, mixture and scoring (final review fix round 1) | date: 2026-09-28] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-unit reference is the other half of trials, else of 5 s chunks (event-optional Task 13) | date: 2026-10-10] -->
The project-wide references do not guarantee conservative or accurate cross-day
probabilities. They come from every day of the project: each unit against its
own other half (trials in event runs, else 5 s chunks), and same-day neighbours within
`radius_um`. Each day pair's
mixture is fitted on every gated candidate, not on best candidates, and
candidates are scored half against half, so references and candidates share one
sampling noise. Same-day neighbours are not a guaranteed stand-in for different
units across days.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: no CalibratedPairModel fitting path (Task 2) | date: 2026-09-27] -->

## The views

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fingerprint view shows the event response (launcher-gui Task 10) | date: 2026-10-07] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the mixture view forms each stream and the combined probability; tags outside the table rows (event-optional Task 13 fix round 1) | date: 2026-10-10] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram are two views; tags outside the table rows (browse-same-day Task 5) | date: 2026-10-12] -->
| view | what it answers |
|---|---|
| Waveform | do the peak-channel spikes look alike |
| Footprint | where on the shank the unit sits, all days on one registered axis |
| Probe map | which electrodes each unit appears on |
| Waveforms on the probe | how each unit's spike varies across the electrodes it is on |
| Interval histogram | do the inter-spike interval distributions look alike |
| Autocorrelogram | do the two-sided autocorrelograms look alike |
| Fingerprint | the event response, all shared conditions or one at full width |
| Tuning | tone curve, shown only, never scored |
| Mixture | how each stream's probability and the combined one are formed |
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

The probe map splits each electrode into one vertical strip per selected unit,
and a unit always owns the same strip, so reading down the probe follows one
colour in one column. Amplitude is deliberately absent: with units of different
amplitude on one electrode, an amplitude-scaled mark hides the weaker one, and
the question this view answers is which units touch which electrodes. Amplitude
is on the footprint, and shape is on the waveform view, which draws each unit's
template at the coordinate of every electrode it was recorded on, after phy.

## How two days get linked to the same sounds

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the celldb triage stage served the cache-builder well listing (Plan C Task 7) | date: 2026-10-02] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the project reader keeps literal BAPHY conditions and functional_likelihood intersects them across the group (Plan C Task 7 fix round 2) | date: 2026-10-02] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fingerprint conditions apply to runs with events (event-optional Task 13) | date: 2026-10-10] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: BigNat runs write binaural names from lbhb.baphy version 2 (bignat-names Task 2) | date: 2026-10-11] -->
**The join, on the recordings.** A condition is the BAPHY stimulus string of
a `Stim` event, kept whole by the project reader (channel, NULL and attenuation
fields included), with one exception: from `lbhb.baphy` version 2, a BigNat run
(`ReferenceClass = 'BigNat'`) writes each name as one `wav:channel` per
speaker, `NULL:channel` for a silent speaker, attenuation dropped
(`unit_match_lbhb/bignat.py`), and each recording's provenance
`stimulus_names` maps recorded name → written name.
<!-- [AGENT EDIT END] -->
In a run with events, `functional_likelihood` scores a group of days on the conditions that every
kept unit measures with at least `min_repeats` repeats, in its reference
fingerprint and in both halves of every split, and each fingerprint is those
conditions' histograms concatenated in sorted order. A group with no such
condition is skipped with the reason recorded.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: version-1 days keep two spellings; version 2 writes one (bignat-names Task 2) | date: 2026-10-11] -->
On days extracted at version 1, two spellings of one sound are two
conditions: a BigNat run names a sound `00seq4.wav` (mono) or
`00seq4.wav:1:0+NULL:2:0` (binaural), and days on different spellings share
neither. Within a site presenting both spellings a neuron's responses to the
two correlate at 0.52 across 222 cells, against 0.013 for two genuinely
different sounds. Version 2 writes both as `00seq4.wav:1+NULL:2` and pools
their level variants. The sound field is kept as written, so `00seq4` and
`00seq4.wav` stay distinct. One sound alone and that sound competing against a
second stay distinct, as do channel offsets.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

## Tracked units

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 2: common-tracker guarantees apply only to common tracks (Task 1) | date: 2026-09-27] -->
Common tracks come from the common tracker (`backends/local/tracker.py`: mutual
margins, one unit per session, measured contradictions): `functional_likelihood`
and `pair_logistic` <!-- [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: pair_logistic listed (final review fix round 1) | date: 2026-09-28] --> always, `unitmatch` and `deepunitmatch` only when
`tracking='common_constrained'` is chosen. Native tracks (`dant`, and
`unitmatch`/`deepunitmatch` in their default native modes) keep their backend's
own rules and are not checked for margins or contradictions.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the matcher proposes tracks; curation is described once, per run (run-curation Task 10) | date: 2026-10-05] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a native same-session group keeps one unit per session and proposes the rest (native-same-day Task 11) | date: 2026-10-11] -->
When a native group holds two or more observations from one session, the adapter keeps the one with the highest mean native score to the group's observations from other sessions (missing scores are skipped; a tie goes to more spikes, then the smaller cell ID); the others stay unmatched, and every pair inside the group that is not two members of one track is saved with decision proposed_link.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the matcher proposes tracks; curation is described once, per run (run-curation Task 10) | date: 2026-10-05] -->
A neuron belongs to at most one tracked unit. The matcher proposes the tracks:
a run's `result.json` holds them and its unmatched units. A curator accepts,
rejects, splits, merges and edits them in that run's own history; see
[docs/project_folder.md](docs/project_folder.md), section 2, for
`curation/runs/<run>/history.csv`, `history.lock` and `launch export`.
<!-- [AGENT EDIT END] -->

## Reading the output

`store.chains()` turns accepted edges into connected components, a neuron
followed across days. Matches are stored as edges rather than chains on purpose:
if day 1 to day 2 and day 2 to day 3 are accepted but day 1 to day 3 is
rejected, `store.inconsistencies()` reports that triangle instead of resolving
it silently.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: store.decoy_error_rate is gone (run-curation Task 10) | date: 2026-10-05] -->
<!-- [AGENT EDIT END] -->

## Caveats that matter when using the matches

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: matching on event responses selects for similar event responses (launcher-gui Task 10) | date: 2026-10-07] -->
Matching on event responses selects for similar event responses. The matched set
therefore cannot be used to argue that tuning is stable across days; that
question needs a different identity criterion, or the selection has to be
modelled. Selection can also affect other outcomes correlated with the matching
features. Holding out trials reduces measurement-noise reuse but does not
eliminate this biological selection bias.
<!-- [AGENT EDIT END] -->

Depth alignment comes from unit-density profiles and has a noise floor near
20 μm, measured by running it on same-day recordings where the true offset is
zero. That is why the proximity gate is 40 μm rather than tighter.

Site coordinates come from the KiloSort channel maps in `/auto/data/code/KiloSort`,
chosen by channel count, with an idealised Neuropixels geometry as the fallback
for a probe with no map on disk. Several maps share a channel count (64D against
64M, 128D against 128P) and nothing readable in the sort record says which was
used, so the probe view offers the choice and names the one in use.

Single units and multi-unit clusters are both included. A multi-unit cluster's
fingerprint is a blend, so its match means something weaker.

## Data-entry dependencies

Curation needs `gPenetration.well`, which is the physical-location key.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: blocks moved to the lab plugin (Plan C Task 7) | date: 2026-10-02] -->
`nems_lbhb.unit_match_lbhb.blocks.backfill_worklist()` lists the sites that
block curation and why: `never_entered` rows need a well number,
`well_lumps_sites` rows have a well covering more than one chamber position and
need splitting. Clathrus is the clear case of the latter: 42 sites, all
labelled well 1, with 41 distinct coordinate strings.
<!-- [AGENT EDIT END] -->
