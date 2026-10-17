# Cross-day single-unit matching: priorities and comparison contract

Updated 2026-09-18. The objective is to compare cell-identity methods in one
curation interface, preserve stimulus-response fingerprints as first-class
evidence, and measure whether combining them with electrophysiology improves
matching. A matching method may leave any unit unmatched.

## Priority order

1. **Reliable curation and single-unit selection.** Fix stale tables/plots after
   split, merge, drop, accept, and reject. Split selected members from the
   remainder; represent detached members as unmatched. Preserve independent
   curation for each immutable method/run and exclude non-single-unit labels.
2. **Comparable method outputs.** One `MatchResult` carries single units,
   pair evidence, tracks, unmatched units, excluded units, and method/input
   provenance. Preserve the original cached method as a baseline. Distinguish
   calibrated probabilities, provisional scores, and native cluster membership.
3. **Within-sort temporal benchmark.** Divide recording time into nonoverlapping
   windows and recompute multichannel voltage waveforms and stimulus/spiking
   features using only spikes/trials inside each window. Source cluster IDs are
   reference labels, not matching features. Deliberately omit selected units in
   selected windows to measure abstention and recovery after missing sessions.
4. **Updated local evidence and tracking.** Fit ordinary candidate pairs on
   training sources, calibrate on separate sources, and evaluate on held-out
   sources. Start with explicit candidate competition and session/contradiction
   constraints; subsequently add aligned multichannel waveform evidence.
5. **Official algorithm comparison.** Run published UnitMatch and DANT using
   verified upstream code and fresh waveform inputs. Report native tracking
   separately from published pair evidence followed by the local tracker.
   Include DANT with and without available stimulus-response evidence.
6. **DeepUnitMatch training.** Use the official model and preprocessing for the
   chosen probe; keep the published waveform-only baseline separate from any
   later stimulus-augmented network. Record source revision, probe geometry,
   training sources, hyperparameters, random seed, and checkpoint hash.
7. **Novelty assessment.** Compare precision/recall, unmatched decisions, identity
   merges/splits, and track stability on the same eligible unit population.
   Select operating thresholds without using the final test partition. Compare
   yield at a fixed estimated false-match rate, rather than rewarding larger
   or longer clusters alone.

## Implemented foundations

- `viewer/` (see `docs/viewer.md`), `curation.py`: selection-aware curation, explicit unmatched units,
  consistent plot refresh, run switching, and one history per across-days run
  (`curation/runs/<run>/history.csv`, undo/redo, `history.lock`, `launch export`). The viewer runs no
  matcher: it opens project runs.
- `launch/window.py` (`launch gui`) runs every command as a CLI child from forms read off the
  CLI parser (`launch/forms.py`), greys backends by `selection_inputs` and `check_real_day_run`,
  shows the run's evidence tags from `Selection.evidence` and greys the `stimulus` feature stream
  without events, and cancels with SIGINT, then a kill after 10 s; the launcher and the viewer
  share the dark-first theme of `viewer/theme.py` and the `Settings` and `Help` menus of
  `viewer/menus.py`.
- `backends/`: the `functional_likelihood` and `pair_logistic` local matchers, the common tracker,
  result validation, the backend registry, run provenance.
- `features.py:temporal_sample_windows()`: shared contiguous sample windows.
- `features.py:average_spike_waveforms()`: bounded fresh voltage reads, waveform
  means/standard errors, deterministic waveform-only subsampling, no snippets
  crossing window boundaries.
- `features.py:temporal_split_unit_features()`: recomputed rates, interval and
  autocorrelation histograms, acoustic-condition-specific fingerprints, trial
  counts, and separate waveform halves inside each pseudo-session. Unusable
  splits carry an exclusion reason; missing waveforms are never duplicated.
- `backends/local/calibrated.py:CalibratedPairModel`: regularized feature classification and
  separate source-group calibration, explicit missing-feature indicators,
  JSON-compatible model state. No runner fits or loads it.
- `evaluation.py`: source-group partition validation and held-out evaluation
  against source-cluster reference identities. Across unrelated source
  recordings, cell identity is unknown rather than automatically negative.
- the `backends/` adapters: optional upstream adapters and explicit DeepUnitMatch
  model/training preparation. Input and dependency checks fail explicitly.
- `run/config.py`: shared cache root and generated-output containment.
- `waveform_preprocessing.py`: shared filtered, median-referenced, unwhitened
  voltage representation; 150-Hz third-order forward/backward Butterworth with
  buffered context before cropping. All matchers use one compact waveform archive.
  Preprocessing alternatives require evidence from a small sensitivity check,
  rather than routine duplicate raw-waveform stores.
- `benchmark.py`, `benchmark_pairs.py`: separate reference labels, fresh
  temporal observations, label-free candidate evidence, and hash-verified
  dataset archives.
- `extract_benchmark.py`: forward-only reads of compressed voltage, retaining
  only selected snippets temporarily in the cache; explicit input clocks and
  voltage transforms.
- `runners.py`: official-method comparison/training commands with immutable
  cache outputs.
- `project/selection.py`, `run/project_run.py`, `viewer/modes/across_days.py`: real dated
  project days; `launch run` runs any registered backend on a selection of
  extracted days, and the viewer, in its across_days mode, curates the completed run in the project's
  `curation/runs/<run>/history.csv`, without fabricated across-day identity labels. Explicit
  valid analysis windows remain separate from physical raw-file lengths.
- `multiday_reuse.py`: reuse complete verified two-part raw-derived summaries as
  independent waveform halves, with pooled moments and freshly recomputed full-day
  spike/fingerprint statistics. The actual PRN069 archive passed read-only conversion
  for all 36 units without opening raw voltage; selected sampling and half boundaries
  remain explicit. Stimulus reliability uses exactly the fresh extraction trial split.
- `streaming_waveforms.py`: opt-in exact-selection accumulation while reading
  voltage, without a temporary snippet file. Defaults remain unchanged. Filtering,
  float32 quantization, timestamp order and full gzip validation are preserved;
  67 focused checks passed, including complete actual-day feature equivalence.
  Independent review found no numerical, integrity, memory or changed-default blocker.
  The queued PRN067 attempt02 explicitly selects streaming mode; attempt01 was
  cancelled while verified waiting, before it opened voltage. Its original PRN050
  dependency and permanent cache location remain unchanged. Actual extraction
  and real-recording timing remain pending.
- `operating_points.py`: calibration-only cutoff selection on saved native
  scores and test application. The full calibration curve and source hashes
  are retained.
Independent review identified and corrected incomplete
  score-pair acceptance and nested-collection identifier mismatches. Exact candidate
  coverage and hash-verified source correspondence are now required; 80 combined
  operating-point/backend/streaming checks passed. Real-data results remain pending.

## Validation completed in this implementation pass

Automated tests cover source-group separation, calibration import validation,
temporal waveform boundaries, unmatched evaluation, result serialization,
method-specific curation, actual offscreen Qt edits, and asynchronous process
success/failure/cancellation. The latest complete suite passed 287 tests, including genuine official
autoencoder and contrastive training and checkpoint inference on synthetic
NP1010 inputs. Focused checks follow subsequent report edits. Real UnitMatch and
DANT stimulus-fingerprint runs also completed through the Qt launcher and loaded
into the same curator while its event loop remained responsive.

The shared-preprocessor PRN069a pilot and deliberately omitted-counterpart view
have completed native UnitMatch and both DANT presets. Saved run manifests and
comparison reports under the cache are authoritative for their measured results.
Waveform diagnostics inspect timing, internal-half reliability, sound-response
ranks and actual missed-pair overlays. These are same-recording reference-sort
checks, not proof of across-day accuracy. DANT's no-reference-pair motion guard
and retained upstream motion-state caveat are recorded in each affected run.

Training-waveform extraction for PRN050a (Prince well 10) has finished reading
and verifying the raw stream and is building temporal features. A separate
DeepUnitMatch driver waits for that exact process, then prepares and trains the
300-epoch autoencoder and 50-epoch contrastive model under
`benchmarks/deep_training_shared_v1/`. No laboratory-trained checkpoint exists yet.

The original PRN065a calibration choice proved unsuitable for a stimulus model:
its chronological halves have no common exact sound conditions. The waiting
`heldout_shared_v1` workflow was stopped and recorded as superseded before any
fitting. Its ongoing waveform extraction is retained as an additional source,
not used to claim stimulus calibration. The replacement PRN075a (Prince well 18)
has 53 eligible single units and 12 conditions repeated five times per half;
its source clock and spike reconstruction are verified. Its first extraction process
stopped before producing a dataset; its partial snippets and log are retained.
The dependent `heldout_shared_v2` waiting driver was cancelled before any fitting
because it referred to that stopped process. A root-owned retry now records its
PID and status in `extraction_attempt_02.json` and logs in `extraction_attempt_02.log`.
It uses the existing `_runtime` cache location (3.49 GiB of planned snippets).
The replacement `heldout_shared_v3` driver follows that exact calibration process
and the independent official training driver; it has not started comparison stages.
An optional scratch override is available without changing existing jobs.
PRN069a remains test well 16, excluded from supervised fitting and probability
calibration. Its native pilot results and waveform diagnostics have already been
inspected. Subsequent comparisons are an explicitly documented exploratory
reanalysis, not a newly prospective study on never-seen test recordings.

PRN050a has six shared conditions, four with at least two repeats in both halves.
The planned local comparison therefore explicitly rebuilds candidate evidence at
`min_repeats=2` using zero-copy collection views. The original four-repeat
archives remain unchanged. This choice follows an input-availability audit,
not test outcomes; repeat counts and internal response reliability remain
recorded.

The GUI has a durable cache-root inference environment created by
`dependency_setup.py`, containing pinned Python sources and tested small optional
dependencies without bundled pretrained models or changes to the base Python.
Real UnitMatch and DANT runs through that environment have loaded in the shared
curator without temporary import paths. Network training still uses the verified
full source checkout, separately recorded in its training manifest.

Process state and artifact manifests are authoritative; these are dated snapshots.

## Remaining development gates

- Run `functional_likelihood` and `pair_logistic` on the same fresh temporal
  benchmark inputs as the published methods, with trial timing from verified
  acquisition-clock intervals. `drift_validation` does this for injected shifts on
  extracted project days: `load_project_day` reads a day's halves archive and the
  day's own events (`python -m multiphy.drift_validation --project NAME --day DAY
  --output DIR`); a day without events is validated on waveforms and physiology alone.
  Old cached results open unassigned and provide no comparable benchmark.
- Verify the project path end to end on real days: `launch init`, `extract`,
  `run` (`run/project_run.py`) and `python -m multiphy.viewer --run DIR`, or all of
  it from `launch gui`, with runs that
  keep actual dates and invent no across-day reference labels. PRN067a13 and
  PRN069a08 have verified common probe geometry and exact stimulus overlap;
  PRN067 requires explicitly verified exclusion of four trailing acquisition
  trials without stimulus annotations. Production extraction must retain the
  raw-file sample count while honoring that shorter analysis window. Across-day
  published-method runs on these days have not completed.
- Complete and inspect the training and calibration archives.
- Train the genuine official DeepUnitMatch architecture, then verify saved-model
  inference and its CLI scoring on both test populations. Record the published
  channel-shift augmentation's NP1010 geometry limitation explicitly.
- Finish report readability review and common-interface integration checks,
  including collections that reference multiple archives without copying voltage.
  The expanded workflow runs UnitMatch and both DANT presets against its exact
  complete and missing-counterpart collections, and compares UnitMatch/DeepUnitMatch
  native tracking with the common constrained tracker. It can reuse the separately
  trained, hash-pinned network. Earlier pilot runs remain auditable.
- Produce the final comparable results and an evidence-based novelty assessment.
  More wells and actual repeated-day data remain necessary for biological claims.
- Add a separately labeled calibration-selected operating-point analysis for
  UnitMatch and DeepUnitMatch scores under the common tracker. Selection
  uses calibration labels only and freezes the chosen cutoff before applying it
  to saved test scores.
  Target empirical false-match fraction means incorrect assigned pairs divided
  by all assigned pairs, measured after track construction; it is not a guaranteed
  population error rate. DANT's native clustering remains a separate baseline,
  because a scalar sweep of its unbounded similarity would change the algorithm.
  One calibration well with 53 reference identities limits the risk estimate;
  operating-point selection reuses that well.
  Before completion of the new comparison, fix primary empirical calibration
  target at 1%, with a prespecified 5% sensitivity comparison. These are descriptive
  development-set operating targets, not asserted population risk bounds. Apply
  each selected cutoff unchanged to complete and absent-counterpart populations.

## Data preparation contract

Existing `arrays.npz` files hold averaged sorting templates, not independent
raw-voltage averages for each temporal split. They cannot be copied into two
halves and described as independently measured waveforms.

The caller of the new temporal extractor must establish:

- The raw voltage source, acquisition sample rate, channel order, and physical
  electrode coordinates for every session.
- The exact mapping from saved sorted spike times to voltage sample indices.
  BAPHY trial-relative times are not automatically raw acquisition indices.
- Shared voltage preprocessing: filtering, reference subtraction, physical
  scaling, and any motion correction. Describe it in extractor provenance.
- Stimulus and baseline windows in the same acquisition sample clock. Preserve
  complete acoustic identity, including ear/channel and timing. Only complete
  trials inside each temporal split contribute to its fingerprint.
- Benchmark/training scripts live in `multiphy/`. All generated
  data, checkpoints, and reports live beneath the root's `benchmarks/` (`--root DIR`
  or `MULTIPHY_CACHE_ROOT`); the repository contains code only.

The initial inspected PRN069a BNT recording retains sorted cluster IDs, trial
spike times, and the recorded AP voltage in compressed `continuous.dat.gz`.
The AP sample index was verified contiguous and its TTL starts agree with the
completed sorting job. The historical spike exporter can be inverted exactly
to recover native exported sorter timestamps. AP TTLs must be used: the generic
LFP-derived starts differ by up to 11 AP samples. Native exported timestamps
also have a one-sample convention distinction from MATLAB template-center
coordinates; this is recorded explicitly rather than silently corrected.
Fresh raw-voltage extraction and waveform inspection are complete for PRN069a;
the common one-sample convention uncertainty remains documented.

The first acquisition-filter-only pilot completed extraction and native UnitMatch
execution. It is an integration result, not the final preprocessing standard.
New preparation defaults to `shared_unwhitened_v1`; old manifests remain explicitly
legacy. A single new buffered source read is needed to upgrade that pilot because
its short snippet averages do not retain the surrounding voltage needed for filtering.
The replacement extraction and published-method pilot runs are complete under
the lab root's `benchmarks/prn069a_temporal_shared_v1/`.

Compressed recordings are read forwards through gzip in memory; no complete
uncompressed recording is written. The current extractor temporarily stores only
selected filtered float32 snippets under the shared cache's `_runtime/extract-*`
folders by default. `--scratch-root` can select a separate absolute directory
outside the code repository for future runs; permanent datasets stay in the cache.
The extractor reports the exact temporary path and planned byte count before
reading voltage. It removes its uniquely named directory after summarization or
a handled exception. Abrupt process termination can leave partial files behind;
no broad automatic cleanup deletes another job's snippets. Permanent datasets contain means,
standard errors and feature summaries. Replacing temporary snippets with streaming
summary accumulation is a proposed storage optimization, not current behavior.

## Benchmark separation and interpretation

Use four or more chronological windows where data permit. Each window is a
pseudo-session; the early and late portions inside it provide independent
waveform averages required by UnitMatch/DeepUnitMatch. Keep the full spike train
within a window for firing statistics. Alternating individual spikes changes
the interval distribution and must not be used to estimate that distribution.

All pseudo-sessions from a source recording stay in the same train,
calibration, or test partition. Prefer holding entire wells out to reduce the
chance that a repeated cell appears across partitions. Do not tune on the final
test partition. Preserve deliberately omitted counterparts as unmatched
reference cases. The benchmark measures recovery of source-sort cluster
identities; it does not resolve contamination in that sort or reproduce all
effects of daily reinsertion.

Stimulus-fingerprint matching selects for response similarity. Withholding
trials reduces noise reuse but does not make a response-stability conclusion
independent of identity selection. Report waveform-only, functional-only, and
combined comparisons separately. DANT already uses functional evidence, so
combining waveform and stimulus information alone is not a novelty claim.

## Primary implementations

- UnitMatch and DeepUnitMatch: <https://github.com/EnnyvanBeest/UnitMatch>
- UnitMatch paper: <https://www.nature.com/articles/s41592-024-02440-1>
- DANT implementation: <https://github.com/jiumao2/pyDANT>
- DANT paper: <https://doi.org/10.1016/j.patter.2026.101590>
- DeepUnitMatch preprint: <https://doi.org/10.64898/2026.01.30.702777>

DeepUnitMatch's currently bundled pretrained checkpoint targets four-shank
Neuropixels 2.0. Do not apply it silently to a different probe. The current
upstream architecture differs from the paper-era code; the adapter records
the verified revision and does not describe latest-source training as an exact
reproduction of the original preprint.
