# Matching methods and run settings

The viewer opens two kinds of **project runs** (see
[`docs/project_folder.md`](docs/project_folder.md)). A **same-day run** matches the
two temporal halves of one extracted day, which are scored against the day's
source labels. An **across-days run** matches two or more extracted days of one
project. It uses every eligible single unit of those days and has no benchmark
split or reference identities. A **track** is a group of observations linked as
one neuron; an unmatched observation remains available for review. Across-day
tracks are identity hypotheses, not established biological ground truth.

Runs start from the launcher, never from a viewer. `launch run` runs any
registered backend; `python -m multiphy.run list-backends` names
every registered backend and whether it is available in this interpreter:

```bash
python -m multiphy.launch run PROJECT --backend ID            # every extracted day, across days
python -m multiphy.launch run PROJECT --backend ID --day D1   # one day's two halves
```

Open a completed run of either kind with one command; it opens in the mode its
`selection.json` records (see [`docs/viewer.md`](docs/viewer.md)):

```bash
python -m multiphy.viewer --run /path/to/PROJECT/runs/<run>
python -m multiphy.viewer --run /path/to/PROJECT/days/<day>/runs/<run>
```

The viewer lists the completed runs of the same selection and starts none. A
run opens only when its `selection.json` still matches its project, and a result
is imported only from a completed run folder of the shown selection.
In across_days mode the viewer curates its run in `curation/runs/<run>/history.csv`: one
line per curator action, undo and redo included, the file replaced whole on each
write. A merge may put two observations from the same calendar day into one
track; the track row flags it. `history.lock` lets one viewer write: a second
viewer of the run opens read-only, and a lock from another host can be taken
over. `launch export PROJECT --run RUN [--out PATH]` writes the accepted tracks to
`curation/runs/<run>/exports/accepted_tracks_<YYYYMMDD-HHMMSS>.csv`. Same-day
runs are inspect-only.

Benchmark archives and collections have no GUI view; their CLI scoring is
unchanged.

## Method inputs

- **UnitMatch** uses measured multichannel spike waveform and spatial features.
- **DANT** uses spike waveforms and spike timing (the autocorrelogram). With
  `--preset waveform_acg_stimulus` it includes aligned stimulus responses in
  both motion estimation and final clustering. The `upstream_example` preset uses
  the fingerprint for motion estimation only. **Motion estimation**
  estimates electrode drift across temporal parts or actual recording sessions,
  according to the run type. DANT returns native
  groups that become neuron tracks: observations assigned to the same neuron.
  Its similarity value is not a calibrated probability for every pair.
- **DeepUnitMatch** uses a saved, trained waveform network, followed by the
  published matching procedure. Running matching does not train a new model.
  Pass the saved checkpoint as `--checkpoint FILE` (it may have a `.pth` extension
  or a name such as `ckpt_epoch_49`) and the exact, case-sensitive **probe model
  identifier** as `--checkpoint-probe ID`, for
  example `NP1010`. Copy the string in `probe_type` from `training_manifest.json`
  in the checkpoint's training-run directory. For an externally obtained model,
  find the equivalent identifier in its training documentation; do not guess it.
  This identifies the electrode geometry the network was trained on. It must
  match the dataset acquisition's `probe_type`; the saved model's sample rate and
  waveform preprocessing must also match. This field does not convert a model
  between probe designs. Channel count alone cannot establish compatibility.

The local `functional_likelihood` method (version 3) runs across days: it fits
its references from every day of the selection, registers each day's depth
before the proximity gate, and reads each day's trial timing (`TrialTiming`,
from the day's `source/` entry) when the day has events. Days without events,
or without a condition on every day, run on physiology and waveform, with 5 s
chunks in place of trials. In a run with events, every day must share one
fingerprint bin width and one duration per shared condition, or the run
refuses. Any registered backend runs across days the same way:

```bash
python -m multiphy.launch run PROJECT --backend unitmatch --days D1 D2
```

Across-days runs have no `--partition` option. Each run is a new folder in the
project's `runs/` (a same-day run: `days/<day>/runs/`); existing runs are never
overwritten. DANT and DeepUnitMatch accept their corresponding settings,
checkpoint, probe and device options described below, after `--backend ID`.

See the [UnitMatch repository](https://github.com/EnnyvanBeest/UnitMatch),
[DANT repository](https://github.com/jiumao2/pyDANT), and
[DeepUnitMatch source](https://github.com/EnnyvanBeest/UnitMatch/tree/95b6fed5db84d12f67487b92d07f0a2664a6122f/UnitMatchPy/DeepUnitMatch) for the
published implementations. These runs share this project's extracted waveform
representation, which may differ from an author's original extraction recipe.


## Python environment and model device

With `--python EXE`, `launch run` runs the backend in that interpreter, even when
the backend is also available in the launcher's own. Without `--python`, an
available backend runs in the launcher's own interpreter, and an unavailable one
runs in the verified durable interpreter below; with neither, it refuses and
names what is missing.
The interpreter needs the chosen method's dependencies: `UnitMatchPy` for UnitMatch,
`pyDANT` for DANT, or `UnitMatchPy` plus PyTorch for DeepUnitMatch. The launcher
installs nothing. `--device` sets DeepUnitMatch's torch device (default `cpu`);
a GPU device such as `cuda:0` requires a CUDA-enabled PyTorch installation and an
available GPU.

## Optional JSON settings

JSON files contain a single object with double-quoted keys, no comments, and no
trailing commas. For UnitMatch/DeepUnitMatch, blank means the published defaults
plus this adapter's measured-geometry settings. An empty override object `{}`
also leaves those defaults unchanged. Nonempty keys must be valid parameters
for the installed official implementation. `--native-threshold`, when given, overrides
a `match_threshold` supplied in parameter JSON.

For DANT, blank means the selected named preset. A custom JSON file is a
**complete DANT settings object**, not a partial override, and replaces that preset.
The [published settings example](https://github.com/jiumao2/pyDANT/blob/741240511e6a4ed9315b2c80d71ab5cfe4c588eb/settings.json)
shows the full format. This project's presets retain its numerical scientific
settings, use one worker, and explicitly choose where the stimulus fingerprint is
included. A complete copy of the selected preset is saved beside each run's log;
the result retains the settings and preset provenance.

### UnitMatch and DeepUnitMatch override schema and example

The top-level value is an object; each key names one published parameter and its
value has that parameter's type. It is a **partial override**: omitted keys keep
the defaults. The same field is used by both methods.

| Example key | JSON type | Meaning |
| --- | --- | --- |
| `max_dist` | Positive number | Maximum spatial separation for candidate matching, in micrometers |
| `channel_radius` | Positive number | Radius around a unit used for waveform features, in micrometers |
| `chunk_size` | Positive integer | Number of rows processed together for pairwise metrics |

A valid example retaining the published values for these parameters is:

```json
{"max_dist": 100, "channel_radius": 150, "chunk_size": 500}
```

`{}` is the minimal valid file. Other supported keys and their types are defined
in the installed `UnitMatchPy/default_params.py:get_default_param()`. This is an
implementation-specific schema, not an unrestricted user metadata object.
Do not change DeepUnitMatch's waveform dimensions or channel layout to fit a
checkpoint: those must agree with measured geometry and training preprocessing.
`--native-threshold` and `--threshold` remain the controls for link assignment.

### DANT complete settings schema and example

Required top-level fields and types are shown below. This complete example is
**Waveform + spike timing**, with the same numerical settings as that preset:

```json
{
  "path_to_data": "",
  "output_folder": "",
  "save_intermediate_results": true,
  "n_jobs": 1,
  "centering_waveforms": false,
  "spikeLocation": {"location_algorithm": "monopolar_triangulation", "n_nearest_channels": 20},
  "waveformCorrection": {"n_nearest_channels": 38, "linear_correction": false, "n_templates": 2, "path_to_motion": ""},
  "autoCorr": {"window": 300, "binwidth": 1, "gaussian_sigma": 5},
  "ISI": {"window": 100, "binwidth": 1, "gaussian_sigma": 1},
  "motionEstimation": {
    "features": [["AutoCorr"], ["Waveform", "AutoCorr"]],
    "max_iter": 15,
    "repeat_last_feature_set": true,
    "stop_early": true
  },
  "clustering": {"max_distance": 100, "features": ["Waveform", "AutoCorr"], "n_iter": 10, "weight_tol": 1e-8},
  "autoCuration": {"auto_split": true}
}
```

Names and string values are case-sensitive. Nested feature values are arrays
of strings: `Waveform` = spike waveform, `AutoCorr` = autocorrelogram (spike
interval structure), `ISI` = interspike interval distribution, and `PETH` =
condition-aligned stimulus-response fingerprint. The `motionEstimation.features`
field is an array of feature arrays, one per successive stage. To include the
fingerprint in both stages and final clustering, use
`[["AutoCorr", "PETH"], ["Waveform", "AutoCorr", "PETH"]]` there and
`["Waveform", "AutoCorr", "PETH"]` in `clustering.features`. Using the named preset
is easier and records that choice directly.

Boolean fields use `true`/`false`; iteration, channel, template and worker counts
are integers (`n_jobs = -1` requests all available workers). Timing windows,
bin widths and smoothing scales are in milliseconds; `max_distance` is in
micrometers. The launcher replaces `path_to_data` and `output_folder` with its
own cache paths; leave them blank in the file. Leave `path_to_motion` blank to
estimate motion in this run. Other native options require the pinned upstream
settings documentation linked above.

## Grouping rule and cutoff

Here a **session** means one temporal part in a benchmark or same-day run, or one
day in an across-days run.
Published conservative, intermediate, and liberal grouping are the official
UnitMatch/DeepUnitMatch track-assignment choices. Both directional scores must
exceed the cutoff before a pair is considered, and pairs are processed from
higher to lower mean score. For each proposed pair linking two existing tracks:

- **Intermediate:** each endpoint must also match the members of the other
  endpoint's track in the same or adjacent sessions. More distant members
  do not need direct support from that endpoint. Adjacent means neighboring
  sessions in the supplied recording order, not necessarily adjacent calendar days.
- **Conservative:** each endpoint must also match every member of the other
  endpoint's track, including distant sessions.
- **Liberal:** the qualifying pair alone joins the tracks, without those extra
  member checks.

These are endpoint checks during sequential grouping, not a claim that every
possible pair within the final track was tested and passed.
When a native group holds two or more observations from one session, the adapter keeps the one with the highest mean native score to the group's observations from other sessions (missing scores are skipped; a tie goes to more spikes, then the smaller cell ID); the others stay unmatched, and every pair inside the group that is not two members of one track is saved with decision proposed_link.
Result validation still refuses a track holding two observations of one session.

Our constrained rule instead uses the published pair scores while enforcing one
observation per session. A proposed pair must beat every other measured
candidate in the partner's session by at least **0.02 score units**, for
both endpoints; alternatives below the link cutoff still count. Otherwise the
pair is rejected as ambiguous. Joining two tracks is also rejected if any
measured pair between their members scores **below 0.10**. Missing evidence is
not treated as such a conflict.

For UnitMatch and DeepUnitMatch, `launch run` takes the grouping rule as
`--tracking` (`native_intermediate` by default, `native_conservative`,
`native_liberal` or `common_constrained`). The competition margin and conflict
threshold are `--ambiguity-margin` and `--contradiction-threshold`. They are not
fields in the optional published parameter JSON. These grouping choices are
different scientific methods.

Scores range from 0 to 1; higher means stronger model support for a link.
Published rules consider pairs strictly above the cutoff. Our constrained rule
considers pairs at or above it. Other grouping constraints still apply, so
passing the cutoff does not guarantee a final link. The published rules' cutoff
is `--native-threshold` (default 0.5, or a `match_threshold` in parameter JSON);
our rule's cutoff is `--threshold` (default 0.9).

## Progress and results

`launch run` runs in the foreground and prints one line when the run completes:
the run folder, backend, days and the number of excluded units. The run folder's
`run_manifest.json` records `running`, then `completed` or `failed` with the
error; a failed run keeps its folder and the command exits with an error. Open a
completed run in its viewer with `--run DIR`; each across-days run keeps its own
history, `curation/runs/<run>/history.csv`. Training, calibration, and test splits must remain separated when
estimating performance; temporal reference identity does not establish accuracy
after a probe is reinserted on a different day.

## Durable inference environment

The default durable interpreter is
`<root>/dependencies/inference-v3/venv/bin/python`, where `<root>` is `--root DIR`
or `MULTIPHY_CACHE_ROOT`. It is absent until built with the setup below.
`launch run` uses it for a backend unavailable in its own interpreter only when
its completed environment manifest and copied-file hashes verify. `--python EXE`
chooses another existing interpreter instead. If verification fails and no
`--python` is given, the run refuses.

The environment contains pinned Python sources for the three methods and the
already tested `hdbscan`/`mat73` dependencies. It inherits the existing scientific
Python installation; it does not modify that installation. No bundled pretrained
models are copied. Choose an actual compatible trained checkpoint separately
for DeepUnitMatch. `environment.json`, `smoke.json`, `setup.log` and `METHODS.md`
in the environment folder record source revisions, file hashes, package versions
and the successful import/network-construction check. That check is not training
or evidence of matching accuracy.

To stage the `inference-v3` environment from local pinned source checkouts and an
existing tested dependency directory, without downloading or installing into the
base interpreter:

```bash
python -m multiphy.dependency_setup \
  --unitmatch-source /path/to/pinned/UnitMatch-checkout \
  --dant-source /path/to/pinned/pyDANT-checkout \
  --overlay /path/to/tested-dependency-directory \
  --output dependencies/inference-v3 \
  --root DIR
```

The source revisions must match the adapter's audited revisions; the overlay
must contain `hdbscan` 0.8.42 and `mat73` 0.65 with intact distribution metadata.
Existing output directories are never overwritten. Only
`dependencies/inference-v3` is the automatic default; for any other output
name, pass its `venv/bin/python` explicitly as `--python`. The staged source
snapshot supports inference, not the separate clean Git checkout required when
preparing future training runs.
