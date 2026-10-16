# Running temporal single-unit comparisons

Code lives in `multiphy/`. Generated input arrays, datasets, method
results, model checkpoints, and reports live beneath the existing cache root:
`/auto/data/nems_db/unit_match/benchmarks/`. The commands below reject output
paths outside that root and never overwrite an existing dataset or run.

## 1. Prepare and extract independent observations

`extract_benchmark.py` consumes a JSON source manifest whose spike times,
stimulus windows, and baseline windows are **zero-based voltage row indices**.
Do not pass trial-relative BAPHY spike samples directly. Record the source of
the conversion in `clock_verification`; a descriptive string alone is not
independent evidence that a conversion is correct.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: prepare_baphy moved to nems_lbhb.unit_match_lbhb (Plan C Task 3) | date: 2026-10-01] -->
For the verified BaphyKiloSort export format, `nems_lbhb/unit_match_lbhb/prepare_baphy.py`
(the LBHB lab package) creates the source inputs without reading continuous voltage.
Supply the actual paths from recording metadata and the completed sort-job record:

```bash
python -m nems_lbhb.unit_match_lbhb.prepare_baphy --parmfile PARMFILE --spikefile SPIKEFILE --recording-folder RECORDING_FOLDER --sort-job SORT_JOB --run-index RUN_INDEX --well ANIMAL:WELL --partition test --dataset-id DATASET_ID
```
<!-- [AGENT EDIT END] -->

It verifies the complete AP sample-number index, AP trial TTLs against the sort
job, measured channel geometry, timestamp-correction backups, and source-sort
provenance. Mixed sort jobs fail unless `--exclude-other-jobs` is explicitly
chosen; excluded units and their source jobs are recorded. The exact inverse
of the historical exporter restores native timestamps with its documented
two-sample offset. This differs from a generic MATLAB-index conversion; the
one-sample template-center convention is retained as a waveform-inspection
caveat.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: BigNat runs write binaural names (bignat-names Task 2) | date: 2026-10-11] -->
Stimulus names are kept whole, except in BigNat runs
(`ReferenceClass = 'BigNat'`), which write one `wav:channel` per speaker with
attenuation dropped; provenance `stimulus_names` maps recorded name → written
name.
<!-- [AGENT EDIT END] -->

The default waveform window is 41 samples before and 41 after the native
timestamp, matching the 82-sample centered input expected by the verified
UnitMatch defaults. New source preparation uses `shared_unwhitened_v1`: a
150-Hz third-order Butterworth forward/backward high-pass, then samplewise
median referencing. `--reference-channels` can exclude known bad contacts from
the reference without changing the saved channel geometry. All recorded neural
channels are used by default; channel quality should be reviewed. There is no
whitening, amplitude normalization, or additional snippet baseline subtraction.

Each manifest contains `schema_version: 1`, a unique `dataset_id`, `runclass`,
`extraction_options`, and a list of `recordings`. A recording supplies:

| Field | Meaning |
|---|---|
| `source_session`, `well`, `partition` | Original recording, animal-qualified well, and `train`, `calibration`, or `test` |
| `raw_voltage`, `dtype`, `n_samples`, `sample_rate` | Interleaved voltage `.dat` or `.dat.gz`, numeric dtype, sample count, and samples/second |
| `spikes_file`, `single_unit_ids` | Non-pickle NPZ containing one sample-index vector for each explicitly accepted single unit |
| `geometry_file` | NPY with measured x/depth coordinates in micrometers, in raw channel order |
| `windows` | Nonoverlapping chronological start/stop sample pairs |
| `stimuli` | Rows with full acoustic-condition `stimulus` label and `start`/`stop` sample indices |
| `baseline_windows` | Nonoverlapping spontaneous-rate start/stop sample pairs |
| `preprocessing` | Acquisition filtering and voltage preprocessing description |
| `clock_verification` | `sample_index_origin: "zero_based_voltage_row"` and a documented verification `method` |

The `waveform_preprocessing` object records the profile, cutoff, filter order,
padding, and reference-channel indices. Filtering uses one-second blocks with
50-ms context on each side before cropping spike snippets; physical recording
endpoints use odd extension. Context may cross a temporal split, while selected
spike snippets cannot. The forward-pass reader temporarily stores only cropped,
processed float32 snippets under `_runtime` and removes them after averaging.
Persistent storage contains float32 means, standard errors, and two internal-half
means, plus selected spike indices and provenance. No full filtered recording or
per-method waveform copies are produced. All methods consume the same archive;
any method-specific transforms occur in memory. This is a shared-input comparison,
not an exact reproduction of each author's raw-data preprocessing recipe.

Older manifests without `waveform_preprocessing` retain their explicitly recorded
`voltage_scale`, `common_median`, and `baseline_samples` behavior. They are never
silently relabeled as filtered data. Such pilot data cannot be upgraded accurately
by filtering their already averaged short waveforms; a new buffered source read is
required. Do not generate parallel preprocessing archives routinely. A future
sensitivity check should use a small sample before justifying another representation.

```bash
python -m multiphy.extract_benchmark --manifest /path/to/source_manifest.json
```

Compressed AP recordings are read forward through gzip into memory; no complete
uncompressed recording is written. The extractor temporarily writes only the
selected snippets, prints their directory and planned size, then removes its
unique temporary directory when it exits normally or handles an exception.
Abrupt termination may leave partial files. The default location is
`/auto/data/nems_db/unit_match/_runtime/extract-*`. An explicit
`--scratch-root /absolute/path/to/scratch` relocates those temporary snippets for
that run while keeping permanent dataset outputs in the matching cache. Paths
inside the code repository, including symlink aliases, are rejected. This option
does not move or alter existing jobs.

An opt-in mode accumulates exactly the same waveform means and standard errors
in memory while reading the compressed source:

```bash
python -m multiphy.extract_benchmark --manifest /path/to/source_manifest.json --waveform-storage streaming_summary
```

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the multiday archive command is retired; real days are project runs (Plan B Task 10) | date: 2026-09-28] -->
Real days do not use this command: `launch extract` extracts them into a project
(see [docs/project_folder.md](docs/project_folder.md)), and `run/project_run.py`
runs a backend on them. The streaming mode writes no temporary spike-snippet file.
<!-- [AGENT EDIT END] --> It preserves selected timestamps, full/half seeds,
float32 snippet quantization, chronological accumulation order, filtering and
complete gzip integrity checks. Memory scales with the number of planned
summaries, not the length of the recording. The default remains `snippets`;
existing jobs are unaffected. Saved provenance distinguishes the operational
storage mode while matching/model compatibility retains the scientific voltage
transforms. Exact numerical-equivalence tests cover both modes; real-recording
deployment and timing are still pending.

The default dataset location is
`/auto/data/nems_db/unit_match/benchmarks/<dataset_id>/dataset/`.
It contains separate matcher inputs and `reference_labels.csv`, content
hashes, exclusions, source/preprocessing provenance, and `METHODS.md`.
Prespecified missing observations can be supplied as
`omitted: [["source_session", "source_unit_id", split_index], ...]`; session
identifiers must be unambiguous across wells. Every method receives the same
remaining eligible population, including observations with no counterpart.

## 2. Run published methods on the test partition

Install the appropriate official implementation in the environment used to
launch the command. The runners never silently install a package or replace
an unavailable method. The API revisions are recorded in
`external_backends.py`; current upstream DeepUnitMatch training is distinct
from exact reproduction of its original preprint.

```bash
python -m multiphy.runners run-backend --input DATASET --backend unitmatch
python -m multiphy.runners run-backend --input DATASET --backend dant --settings SETTINGS.json
python -m multiphy.runners run-backend --input DATASET --backend deepunitmatch --checkpoint CHECKPOINT --checkpoint-probe PROBE_TYPE
```

For a dataset currently being extracted, add `--wait-for-pid EXTRACTION_PID`.
The runner checks its backend dependencies first, waits for that specific
process to exit, then requires a completed dataset manifest before running.
An interrupted extraction without a completed dataset produces an error.

DANT settings must specify the complete upstream configuration. If `PETH` is
enabled, the input consists of condition-aligned stimulus-time bins; unavailable
fingerprints are not fabricated. UnitMatch and DeepUnitMatch use their native
intermediate tracking by default. `--tracking common_constrained` separately
tests their scores through the local assignment rule. Native UnitMatch's 0.5
threshold and the local tracker's threshold are independent options.

Each original recording runs independently, including drift estimation. Runs
write a portable `result.json`, `evaluation.json`, status/provenance manifest,
and `METHODS.md` under a unique `runs/` directory. Native DANT similarity is
retained as similarity, not converted into a purported probability.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: benchmark archives have no GUI view; their CLI scoring is unchanged (run-curation Task 10) | date: 2026-10-05] -->
Benchmark archives and collections have no GUI view; their CLI scoring is
unchanged. The runner reads the persisted waveform/response features directly;
it does not query the database or build a daily cache. A same-day project run
(`launch run PROJECT --backend ID --day DAY`) opens inspect-only with
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day run opens with the viewer package (viewer-split Task 10) | date: 2026-10-07] -->
`python -m multiphy.viewer --run DIR`; reference neuron identities are not shown, and its
<!-- [AGENT EDIT END] -->
observations are within-recording temporal parts rather than recording days.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no fixed local model: section 3 keeps only native-score operating points (Task 2) | date: 2026-09-27] -->
## 3. Select a shared operating point

For a separately labeled comparison at a shared empirical error target,
`operating_points` loads saved native UnitMatch / DeepUnitMatch scores. It
selects one cutoff using calibration labels, then applies that cutoff to saved
test scores without searching test thresholds. The target is **incorrect
assigned pairs / all assigned pairs** after track construction, including
implied links. It is not the false-positive rate among all candidate negatives,
and it is not a guaranteed future error rate.

```bash
python -m multiphy.operating_points select --input FULL_COLLECTION --scores CALIBRATION_SCORES/result.json --target-fdr 0.01 --output FROZEN_SELECTION
python -m multiphy.operating_points apply --input FULL_COLLECTION --scores TEST_RUN/result.json --selection FROZEN_SELECTION/selection.json --output SELECTED_TEST_RUN
```

Native calibration scores come from `runners run-backend --partition calibration`
on that same full collection. DANT remains a native clustering baseline; sweeping
its similarity would define a different algorithm. No qualifying nonempty
calibration assignment produces an explicit unavailable operating point. A small
calibration population and dependent track links limit the risk estimate. Native
results remain separate. The already inspected PRN069 pilot makes this an
exploratory reanalysis.
<!-- [AGENT EDIT END] -->

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer runs no matcher; the local method runs from Python with TrialTiming (Task 3 fix round 2) | date: 2026-09-27] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: prepare_baphy moved to nems_lbhb.unit_match_lbhb (Plan C Task 3) | date: 2026-10-01] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the cache builder is retired (Plan C Task 7) | date: 2026-10-02] -->
The main viewer runs no matcher. To compare the
<!-- [AGENT EDIT END] -->
local `functional_likelihood` method, run it from Python with a `TrialTiming`
per session (`runner.run(..., extra_inputs={'timing': timing})`, timing from
`nems_lbhb.unit_match_lbhb.prepare_baphy.recover_baphy_timing`; see
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no benchmark_viewer --result: project runs come from launch run (run-curation Task 10) | date: 2026-10-05] -->
[docs/contract.md](docs/contract.md), section 5), or on project days with
`launch run` (see [docs/project_folder.md](docs/project_folder.md)), which passes
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run opens with the viewer package (viewer-split Task 10) | date: 2026-10-07] -->
each day's timing and opens with `python -m multiphy.viewer --run DIR`.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->
The `run-backend` CLI builds no timing and refuses this method.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

## 4. Train DeepUnitMatch explicitly

```bash
python -m multiphy.runners prepare-deep --input DATASET --source-checkout OFFICIAL_CHECKOUT
python -m multiphy.runners train-deep --prepared PREPARED_DIRECTORY
```

Preparation selects only the training partition, validates held-out original
wells, and stages the pinned official two-stage training protocol. It does
not start training. Training requires a compatible probe, measured geometry,
and the official dependencies. Upstream random validation is an optimization
diagnostic; final matching accuracy is measured on separately held-out wells.

## Interpretation

### Reuse archives across wells and missing-counterpart tests

`benchmark_collection` writes only a hash-pinned manifest and METHODS, with no
copied waveform arrays. Original train/calibration/test well assignments are
preserved; repeated source recordings, changed source hashes and partition
leakage are rejected. For a complete held-out collection:

```bash
python -m multiphy.benchmark_collection --input TRAIN_DATASET --input CALIBRATION_DATASET --input TEST_DATASET --output CACHE_COLLECTION --dataset-id COLLECTION_ID
```

For a fixed missing-counterpart test, use a test archive alone with
`--omit-fraction 0.2 --seed 20260918`. Observations in later temporal sessions
are selected by a reproducible hash independent of matching evidence. Save the
view to a new cache path and give that same population to every method.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: collections are compared by CLI scoring; they have no GUI view (run-curation Task 10 fix round 1) | date: 2026-10-05] -->
New collections namespace both session and observation identifiers, so multiple
archives combine without session-name collisions. Existing collection manifests
retain their original identifier policy. To compare methods, run each method
against the exact same collection and partition and score the results with
`comparison_report --run RUN_DIRECTORY`; scientifically equivalent populations
from different archive views can be compared in reports but are not
interchangeable. Benchmark archives and collections have no GUI view; the
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one viewer since the split (viewer-split final review) | date: 2026-10-07] -->
viewer opens only project runs (`--run DIR`).
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT END] -->

`benchmark_workflow` combines four explicit archives (`--training`,
`--calibration`, `--test`, `--missing-test`), runs
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no local ablation stages or their preflight (Task 2 fix round 1) | date: 2026-09-27] -->
UnitMatch and both DANT presets, and explicitly starts official
<!-- [AGENT EDIT END] -->
DeepUnitMatch training plus inference. UnitMatch and DeepUnitMatch each run with
native intermediate tracking and the common constrained tracker.
It requires `--output` and `--source-checkout`; defaults are 300 autoencoder
epochs and 50 contrastive epochs. Optional `--training-pid` and
`--calibration-pid` wait on existing extraction processes. Its `workflow.json`
records per-stage status and hashes, holds a lifetime process lock, supports
verified resume, and retains failed attempts. Interrupted stages require outcome
verification before another attempt; an expired observation is not permission
to restart a live job. Failed network attempts restage preparation and training
together. Generated training METHODS record probe-specific assumptions;
only final checkpoints are retained. Running this command authorizes training,
unlike the preparation-only `prepare-deep` command. To reuse an already trained
network, provide `--checkpoint PATH --checkpoint-probe PROBE`; its hash is pinned
and inference verifies the accompanying training provenance and compatibility.

Use `--min-stimulus-repeats N` only as an explicit input policy. It rebuilds pair
evidence from measured saved features in both collections without copying voltage
or fabricating unavailable responses. A lower minimum may retain noisier evidence;
repeat counts and reliability remain recorded.
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the evidence-variant preflight is removed with the local stages (Task 2 fix round 1) | date: 2026-09-27] -->
<!-- [AGENT EDIT END] -->
Confirm shared exact sound conditions across temporal parts while choosing
source recordings; a large number of different sounds does not establish overlap.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: workflow runs are scored by the CLI, not imported into a viewer (run-curation Task 10 fix round 1) | date: 2026-10-05] -->
The workflow uses the same two collection paths for every method, so every
alternative is scored on the same population. Include these runs in
`comparison_report --run RUN_DIRECTORY` (repeat `--run` for each); their
collections have no GUI view.
<!-- [AGENT EDIT END] -->

### Scientific scope

Temporal-split reference identity is conditional on the original sort being
correct. It does not establish multiday reinsertion accuracy or remove
contamination inherited from sorting. Compare match precision, recovery,
unmatched decisions, and false merges/splits on identical populations, then
investigate which benefits survive on real cross-day recordings. Stimulus
fingerprints used for identity selection cannot also provide independent proof
that those same response properties were stable.
