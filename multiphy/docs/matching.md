# Matching methods and evidence

multiphy presents several matching algorithms through one viewer. A backend
proposes pair evidence, tracks and unmatched units; the viewer lets a curator
inspect and edit those hypotheses. The algorithms do not all use the same
features, scores or grouping rules.

See [matching settings and upstream grouping](../MATCHING_INTERFACE.md) for
backend options, [benchmarks](../BENCHMARKS.md) for extraction and evaluation,
and [the viewer guide](viewer.md) for comparison, curation and export.

## What each method contributes

| Backend | Matching evidence | Track formation |
| --- | --- | --- |
| `functional_likelihood` | Local likelihood ratios from physiology, waveform and optional event response | Common constrained tracker |
| `pair_logistic` | Local logistic pair model, calibrated on held-out sessions | Common constrained tracker |
| `unitmatch` | Upstream UnitMatch waveform and spatial features | Native grouping by default; common constrained tracker optional |
| `dant` | Upstream DANT waveform and spike-timing features; stimulus use depends on the preset | Native clustering |
| `deepunitmatch` | Upstream trained waveform network and matching procedure | Native grouping by default; common constrained tracker optional |

The optional adapters call the genuine upstream implementations. Missing
packages, required raw waveforms or compatible checkpoints produce explicit
errors. A stimulus fingerprint is not a requirement or a score for every
algorithm. DANT's `waveform_acg_stimulus` preset uses it for motion estimation and
clustering; `upstream_example` uses it for motion estimation only. The settings
guide describes the available presets and inputs.

When another completed result is imported into the viewer, its native scores
remain authoritative. Additional measured stimulus and physiology evidence is
available for inspection; displaying it does not change the imported method.
DANT group membership is shown without inventing a pair probability.

## Shared observations and waveforms

Only explicit `SU` quality labels are eligible. Missing, unknown and multi-unit
labels are excluded. These source labels still require quality review; the
matching contract does not independently establish that a cluster is a single
neuron.

New raw-voltage extraction uses the shared filtered, median-referenced,
unwhitened waveform representation described in [benchmarks](../BENCHMARKS.md).
Buffered filtering precedes snippet cropping. Persistent waveform data contain
summaries and internal-half means; methods reuse the archive and apply any
additional transformations in memory. This compares methods on shared inputs
and may differ from an author's original extraction recipe. Older archives
retain their recorded preprocessing and are not silently relabeled.

Both local matchers require waveform halves from raw voltage; template-only
features are insufficient. Geometry follows each recording's measured channel
order. Channel-count or incompatible-layout errors must be resolved in the
inputs. Geometry-dependent plots cannot establish a location where measured
geometry is absent.

## The local reference models

`functional_likelihood` and `pair_logistic` build their same/different examples
from the selected project's recording sessions. Same-unit examples compare
each unit's own halves, with waveform displacement jitter. Different-unit
examples are distinct same-session neighbours within `radius_um` (80 µm by
default). References use resampled trial splits when events are available and
5 s recording chunks otherwise. Firing statistics and event fingerprints come
from the same recording; source plugins define which recordings make up a day.

`functional_likelihood` fits one likelihood-ratio model per property and
combines these in streams:

| Stream | Properties |
| --- | --- |
| Event response, when available | Fingerprint correlation |
| Physiology | Spontaneous-rate difference, interval-histogram cosine similarity, autocorrelogram cosine similarity; session-rate difference in event runs |
| Waveform | Waveform correlation, footprint correlation and peak-channel spike-width difference |

Stream weights are fitted from the reference pairs. Rate, interval histogram
and autocorrelogram are related measurements, so their agreement should not be
read as three independent confirmations. The fitted weights do not guarantee
calibrated cross-day identity probabilities. Missing property evidence has a
neutral likelihood-ratio contribution; an event stream that the run did not
measure is omitted entirely.

For each day pair, every candidate inside the proximity gate contributes to
the mixture fit, including candidates that will not become links. Candidates
are scored half against half, using the same measurement scheme as the
references. The mixture estimates the same-unit fraction and returns stream
and combined posteriors. These are provisional, uncalibrated posteriors: the
mixture prior is fitted on the inference candidates, not a held-out calibration
set. A failed or insufficient mixture fit leaves its scores unavailable.

`pair_logistic` instead trains a logistic model on alternating sessions in
recording order and calibrates it on the remaining sessions. A session is a
day in an across-days run, or a temporal half in a same-day run. Its calibration
does not see training sessions, but still uses within-session reference labels
from the same project. In no-event runs it omits fingerprint and shared-event
rate properties, and omits the session-rate property already represented by the
whole-recording spontaneous rate.

The local waveform comparison uses the union of both units' channel
neighbourhoods after alignment. A neighbourhood contains channels less than
150 µm from the peak with at least 10% of the peak channel's largest absolute
voltage in the spike window, plus the five nearest channels. Waveform
correlation uses the spike window; footprint correlation uses peak-to-peak
amplitudes over the snippet. `pair_logistic` calls this default
`waveform_channels='peak_threshold'`; its `full_array` option compares every
channel and waveform sample instead. `functional_likelihood` uses the local
neighbourhood properties.

## Event conditions and missing events

With the CSV events reader, a condition is the complete `condition` label on a
stimulus row. Surrounding spaces are stripped and labels are then compared
exactly across trials and days. Encode every relevant part of the condition,
such as sound and ear, in that label.

The local matchers first decide which evidence the whole recording group can
support. Event matching requires trial timing on every chosen day and at least
one condition with `min_repeats` repetitions on every day (default 5). The
fingerprint then concatenates the sorted shared-condition histograms measured
by every retained unit, in its reference fingerprint and both halves of every
split. If that further intersection is empty, the group is skipped and the
reason recorded. Shared conditions must have matching bin widths and durations;
the matcher refuses incompatible fingerprints.

If a day has no events, or no sufficiently repeated condition is shared across
all chosen days, both local methods run without the event-response stream and
use 5 s chunks in place of trials. If any day lacks baseline windows,
spontaneous-rate estimates use the whole recording on every day of the group.
The launcher prints each fallback as `warning · <reason>` and stores it in
`run_evidence` provenance. Explicitly requesting the stimulus-only likelihood
stream without events is an error.

Reference eligibility remains a separate constraint: the local defaults require
at least 200 spikes and, in event runs, fingerprint reliability of at least
0.30. No-event observations need at least two chunks. Skipped units or groups
do not acquire substitute evidence.

## Depth registration and diagnostic criteria

Both local methods default to `registration='matches'`. Depth-amplitude
profiles provide initial offsets, then confident mutual-best pairs refine one
rigid depth offset per session before the proximity gate. Event runs register
using event response and physiology; no-event runs use physiology and waveform,
aligning the waveform at each tested offset. A day pair with too few confident
matches falls back to its usable profile estimate or has no usable estimate.
Provenance records the source and diagnostics. `profile` and `none` are explicit
alternative registration settings.

The historical thresholds in `backends/local/evidence.Criteria` are useful for
reading local pair diagnostics. Their roles differ:

| Criterion | Default threshold | Role in current local matching |
| --- | --- | --- |
| Proximity | At most 40 µm after registration | Candidate-generation gate |
| Spontaneous rate | Absolute log difference at most 0.293 | Display diagnostic |
| Inter-spike interval histogram | Cosine similarity at least 0.978 | Display diagnostic |
| Spike width | Absolute difference at most 0.15 ms | Display diagnostic |

Rate, interval and width diagnostics are not mandatory pass/fail gates for the
automatic local matchers. Their measured values contribute to the scoring
models instead. These thresholds are not universal requirements for the
optional upstream algorithms. Inspect the registration diagnostics and run
settings rather than assuming a fixed alignment accuracy.

## From pair scores to tracks

The common tracker processes candidate links deterministically from highest
score to lowest. Each endpoint must exceed its competing candidates in the
partner's session by the configured margin; alternatives below the link
threshold still count. A merge must retain one observation per session and
must not contain a measured pair below the contradiction threshold. Missing
evidence is not a measured contradiction. Unmatched observations and missing
sessions are allowed. This is a best-first procedure, not a globally optimal
assignment.

Both local methods always use this tracker. UnitMatch and DeepUnitMatch use it
only with `tracking='common_constrained'`; their default is native intermediate
grouping. DANT retains native clustering. Native grouping does not inherit the
common tracker's margin or contradiction checks. The adapters resolve native
same-session collisions before returning tracks; the exact rule, native
grouping choices and separate native/common cutoffs are documented in
[matching settings](../MATCHING_INTERFACE.md#grouping-rule-and-cutoff).

A result records its backend, pair scores, tracks, unmatched units and
provenance. Comparisons and publications should state which grouping rule formed
the tracks, as well as the scoring method. Curation changes track membership
and review decisions in a separate history for each run. See
[the viewer guide](viewer.md) for those actions and accepted-track export.

## Interpreting the evidence

Inspect the waveform, footprint, interval histogram, autocorrelogram and any
available fingerprint alongside the backend's score. For functional likelihood,
the stream scores and mixture panel expose disagreement between evidence
sources and the distributions behind the combined posterior. The tuning view
is a display and is not itself a scored property.

Within-session references do not guarantee conservative or accurate cross-day
probabilities. Same-day neighbours need not represent different units across
days, and a neuron's measurements can change between recordings. Calibration
on reference examples and temporal-split benchmark performance do not by
themselves establish identity accuracy across recording days. Across-days
project runs have no reference identities or accuracy estimate; see
[benchmarks](../BENCHMARKS.md) for the evaluation contract.

Matching on event responses selects for similar event responses. A matched set
selected this way cannot by itself establish stable tuning across days; that
question needs an independent identity criterion or a model of the selection.
Selection can also affect outcomes correlated with other matching features.
Holding out trials reduces measurement-noise reuse but does not eliminate this
biological selection bias. Apply this caveat to methods and settings that use
the relevant feature, including stimulus-aware DANT settings, rather than
assuming that every backend matches on a fingerprint.
