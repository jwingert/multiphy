# The project folder

A **project** is one folder that holds a set of recording days made with one probe. Each day
is added once, from a phy (KiloSort) folder, an optional events CSV and an optional raw voltage
file. multiphy then extracts the day's features, also once, and can check at any time that
the files the day came from are unchanged, runs any registered matcher on its extracted
days, and exports the tracks a curator accepted. Five commands do all of this:

```bash
python -m multiphy.launch [--root DIR] init    PROJECT (--from DIR | --day DAY --sort DIR --date YYYY-MM-DD [--events CSV] [--raw FILE]) [--uv-per-bit X] [--probe TYPE]
python -m multiphy.launch [--root DIR] extract PROJECT [--day DAY ...] [--tier {raw,templates}] [--scratch DIR]
python -m multiphy.launch [--root DIR] status  PROJECT [--verify]
python -m multiphy.launch [--root DIR] run     PROJECT --backend ID [--day DAY | --days DAY DAY ...] [--python EXE] [backend options]
python -m multiphy.launch [--root DIR] export  PROJECT --run RUN [--out PATH]
```
`python -m multiphy.launch [--root DIR] gui [PROJECT]` opens a window that runs the
same commands and opens runs (`### gui` in section 9).

To read spikes, events or voltage in a format other than phy, events CSV or a flat binary
file, see [`adding_a_source.md`](adding_a_source.md).

## 1. The lab root

Every project lives directly under one **lab root**. It is `--root DIR` when given, else the
`MULTIPHY_CACHE_ROOT` environment variable. With neither, every launcher command, `--help`
included, prints `error: no root · pass --root DIR or set MULTIPHY_CACHE_ROOT` and returns 1;
the benchmark CLIs raise a `ValueError` with the same message when they need a root.
`python -m multiphy.viewer --run DIR` takes its root from the run.

- A project is `<root>/<PROJECT>/`. `PROJECT` is a name matching `[A-Za-z0-9][A-Za-z0-9_.-]*`,
  or the path of a folder directly under the root.
- These names are reserved and cannot be projects: `collections`, `reports`, `logs`,
  `_runtime`, `dependencies`, `viewer`, `benchmarks`, `multiday`, `curated`.
- `<root>/_runtime/` holds the temporary voltage snippets of a raw-tier extraction, and each
  extraction removes its own. `extract --scratch DIR` puts them elsewhere. `DIR` must be an
  absolute path outside the code repository.

## 2. The project layout

```
<root>/
  M1/                          the project
    project.json               the project's identity and every day's record (section 4)
    days/
      <day>/                   one folder per day, named after it
        source/                the day as its readers found it; no voltage
          extraction_manifest.json
          spikes.npz           unit id -> int64 spike samples
          geometry.npy         channel x, y in µm
          templates.npz        unit id -> unwhitened template (only when the sort has templates)
          METHODS.md
        features/
          halves/              raw tier only: the day's two halves as a benchmark archive
            units.csv  pairs.csv  reference_labels.csv  features.npz  features.json
            manifest.json  METHODS.md
          full/                one row per unit for the whole day
            units.csv  features.npz  features.json  extras.npz  manifest.json  METHODS.md
          extract.log          one timestamped line per extraction step
        runs/                  same-day project runs of this day's two halves (section 9, `run`)
      ...
    runs/                      across-days project runs
      <run>/                   selection.json  pairs.csv  run_manifest.json  result.json
    curation/
      runs/
        <run>/                 the curation of one across-days run in runs/
          history.csv          one line per curator action, undo and redo included
          history.lock         the one viewer that writes history.csv
          exports/             accepted_tracks_<YYYYMMDD-HHMMSS>.csv, one per export
    logs/
      launcher.log             one line per command run on this project
  _runtime/                    temporary snippets during a raw-tier extraction
```

- A day's `source_id` is the sha256 of `source/extraction_manifest.json`. Its halves id and
  full id are the sha256 of `features/halves/manifest.json` and `features/full/manifest.json`.
  `project.json` records all three.
- While a process writes `project.json`, it holds `project.lock` in the project folder. While
  a day is extracted, `.extract.lock` sits in its day folder. A lock left by a process that
  died stays until you remove it, and the error names it:
  `project.json is locked by <host> <pid>; remove <path> if that process is gone.`
- `status` counts the folders in each day's `runs/` and in the project's `runs/`. Only `run`
  writes `runs/`; `init`, `extract` and `status` never do. The viewer writes `curation/`.
- `curation/runs/<run>/history.csv` is the curation of the across-days run `runs/<run>`: one
  line per curator action, undo and redo included. No line is ever edited; the file is replaced
  whole on each write, so a reader sees the old file or the new one. The viewer rebuilds the
  run's tracks on every open by replaying the history over the run's `result.json`. A line that
  is malformed or no longer applies to that result is skipped and listed, never deleted.
- `history.lock` lets one viewer write a run's history. A second viewer of the run opens
  read-only. A lock left by a process that died on this host is reclaimed when the run opens; a
  lock from another host can be taken over (**Take over**, then **Confirm take over**).
- Same-day runs are inspect-only: the viewer shows them in its same_day mode and writes no
  curation history.
  Closing it still saves the window layout to `viewer/same_day_layout.json` (an across-days run:
  `viewer/across_days_layout.json`) under the lab root
  (the folder that holds every project).
  Benchmark archives and collections have no GUI view; their CLI scoring is unchanged.

## 3. Written once: a changed day goes in a new project

A day is added once. Its `source/` and `features/` are written once and never replaced or
archived, so anything built from a day's features keeps pointing at the same files. A new
sort, a changed source file, another tier or new preprocessing all mean a **new project**.
Both refusals say so:

```
error: Day D1 already exists; a day is added once. Put a changed day in a new project.
error: Day D1 is already extracted; features are written once. Put it in a new project to extract it again.
```

Two more rules follow from this:

- A project holds one day per date. Two days with the same date are refused.
- An extraction that stops part way records nothing. The next `extract` of that day reuses a
  `source/` it left when that was normalized from the same files, and replaces anything else
  it left in `source/` or `features/` (section 9, `extract`).

## 4. `project.json`

| Field | Value |
|---|---|
| `schema_version` | `1` |
| `kind` | `"multiphy_project"` |
| `name` | the project name |
| `created_at` | UTC time, ISO 8601 |
| `identity.group` | the group name, which the launcher sets to the project name; each unit row's `well` and `recording_group` |
| `identity.geometry_sha256` | sha256 of the first extracted day's `geometry.npy`; every later day must have the same geometry |
| `identity.probe` | `null`, or the probe type the first `init --probe TYPE` recorded |
| `identity.subject`, `identity.implant` | `null`; kept for later tools |
| `days` | day name -> day record |

A day record:

| Field | Value |
|---|---|
| `date` | `YYYY-MM-DD` |
| `date_source` | `"user"` (from `--date` or `date.txt`) or `"reader"` |
| `added_at` | UTC time the day was added |
| `readers` | role -> `id@version`, for example `{"sort": "phy@1", "events": "events_csv@1", "raw": "binary@1"}` |
| `reader_options` | the options the readers were given; for a day added by `init`: `day`, `sort_dir`, `events_path`, `raw_path`, `uv_per_bit`, `raw_dtype`, `raw_offset`, `raw_n_channels` |
| `files` | file key -> file identity (below), one per file the readers open |
| `raw_params` | `null` until extraction and for a day without raw; then `{sample_rate, n_channels, dtype, uv_per_bit, gzip}` |
| `tier` | `null` until extraction, then `"raw"` or `"templates"` |
| `source_id` | `null` until extraction, then the `source/` id |
| `features` | `{"halves": id or null, "full": id or null}`; `halves` stays `null` in the templates tier |

A file identity:

| Field | Value |
|---|---|
| `role` | `"sort"`, `"events"` or `"raw"` |
| `given` | the path as given, made absolute; a link is kept as a link |
| `resolved` | the file that path resolved to when the day was added |
| `size_bytes`, `mtime_ns` | of the resolved file |
| `sha256` | of the resolved file, or `null` |
| `sha256_from` | `"init"` (hashed when the day was added), `"extract"`, `"verify"` or `null` |

Every file except the raw file is hashed when the day is added. A `.gz` raw file is hashed
while the raw tier reads it (`"extract"`). A flat raw file is hashed only by `status --verify`
(`"verify"`).

## 5. A day's inputs

### The sort: a phy folder

The `phy` reader reads `params.py`, `spike_times.npy`, `spike_clusters.npy`,
`channel_positions.npy` and `cluster_group.tsv`, or `cluster_KSLabel.tsv` when there is no
`cluster_group.tsv`. Every cluster labelled `good` is a single unit, with the id
`<day>:<cluster>`. `params.py` is parsed, never executed.

- `channel_map.npy` is optional. When present, it gives the raw-file column of each geometry
  row.
- When `templates.npy`, `whitening_mat_inv.npy` and `spike_templates.npy` are all present, each
  unit gets a template: the spike-count-weighted mean of the unwhitened templates its spikes
  carry. The templates tier needs them.
- When there is no raw file, `params.py`'s `dat_path` is used as the raw file if that file
  exists and `--uv-per-bit` is given. Without `--uv-per-bit`, `init` prints the note
  `raw file found at dat_path; pass --uv-per-bit to use it` and the day has no raw file.

### The events CSV

One row per trial, stimulus or baseline window, with a header row:

| Column | Holds |
|---|---|
| `kind` | `trial`, `stimulus` or `baseline` |
| `start`, `stop` | 0-based samples on the sort's clock; `stop` is exclusive |
| `start_s`, `stop_s` | or seconds, rounded to the nearest sample; one file uses one pair, never both |
| `trial` | an integer of at least 1, on trial rows. Stimulus rows need it only when the file has trial rows, and then it names the trial the stimulus lies in |
| `condition` | on stimulus rows: the complete condition label |

```
kind,start,stop,trial,condition
trial,0,57000,1,
baseline,0,6000,,
stimulus,6000,21000,1,noise_L
trial,60000,117000,2,
baseline,60000,66000,,
stimulus,66000,81000,2,noise_R
```

- `condition` labels are stripped of surrounding spaces, then compared exactly, across trials
  and across days. A label must name everything that makes the condition, for example the
  sound and the ear.
- Without trial rows, each stimulus is its own trial, numbered from 1 in start order.
- The analysis window becomes `[0, stop of the last trial)`. Spikes at or after that stop
  are left out. Baseline windows never change the recording's length.
- Baseline windows are never checked against the last trial or the recording's end, so a
  baseline window may run past either. Each baseline rate is computed over the window being
  measured: each half in the raw tier, and the analysis window for the full-day row. A
  baseline window is clipped to that window, and only its part inside counts, with no
  message. A baseline window wholly after the last trial counts toward no rate. Check the
  baseline rows yourself: a mistimed baseline window gives a baseline rate over a different
  stretch than intended.
- Other columns are ignored.

Each of these refuses the day at `extract`. The error names the CSV line (the header is
line 1) for a missing value, a malformed number, an empty window, a bad trial number and a
stimulus row without a `condition`:

- a `kind` other than `trial`, `stimulus` or `baseline`;
- both `start`/`stop` and `start_s`/`stop_s` in one file;
- a missing start or stop, a sample that is not an integer, or seconds that are not a finite
  number;
- a window that is empty or starts before sample 0;
- a missing trial, or one that is not an integer of at least 1;
- a stimulus row without a `condition`;
- trials out of order or overlapping, or one trial number used twice;
- a stimulus outside its trial;
- a condition with two different durations;
- baseline windows out of order or overlapping;
- a last trial that ends after the recording ends, when the recording's length comes from the
  raw file (or from a sort reader's own clock). Without trial rows, the last stimulus is the
  last trial. This is the only check against the recording's end; baseline windows are not
  checked (above).

### The raw file: the `binary` reader

The raw file is a flat interleaved binary file (one row = one sample of every channel),
optionally gzip-compressed (`.gz`). A flat file has no header, so its layout comes from the
sort's `params.py`, whether or not `params.py` names a `dat_path`: `n_channels_dat` (the
channel count, required), `dtype` (default `int16`) and `offset` (default 0). `--uv-per-bit`
has no default: waveforms are always in µV.

`init` refuses a missing raw file, a raw file without `--uv-per-bit`, a `--uv-per-bit` that
is not a positive finite number, and a raw file whose `params.py` has no `n_channels_dat`
(`phy: params.py has no n_channels_dat; a raw file needs its declared channel count.`).
`init` reads no data, so the reader's other refusals come at `extract`. The day then stays in
the project, unextracted: fix the input and add the day to a new project.

| Refused | Error |
|---|---|
| `temp_wh.dat`, KiloSort's whitened copy | `binary: temp_wh.dat is KiloSort's whitened copy, not the acquired stream.` |
| a channel count (`params.py`'s `n_channels_dat`) other than the geometry's row count, such as 385 with a sync channel | `binary: 385 raw channels differ from 384 geometry channels.` |
| a channel map that is not the identity | `binary: the sort's channel map is not the identity; geometry row i must be raw column i.` |
| a byte offset | `binary: a byte offset is not supported.` |
| a file size that is not a whole number of rows | `binary: the raw file size is not a whole number of samples.` |
| a file shorter than the last spike | `binary: the raw file holds fewer samples than the sort.` |
| a `.gz` file with a sort that takes its length from the last spike, as `phy` does | `binary: a gzip raw file needs the sample count from its sort reader.` |

Use the file the acquisition system wrote, never `temp_wh.dat`. A `.gz` raw file needs a
sort reader that knows the recording's length (see `adding_a_source.md`).

`binary` cannot count a flat file's channels; it compares the declared `n_channels_dat` with
the geometry's row count. The declared count must be the file's own: a wrong `n_channels_dat`
that equals the geometry's row count reads rows that mix channels and samples, and is
refused only when the file's size is not a whole number of rows.

## 6. The `init --from` folder

`init --from DIR` adds every folder in `DIR` as a day, named after the folder:

```
/data/incoming/M1/
  D1/
    sort/          the phy folder
    date.txt       2026-01-05
    events.csv     optional
    raw.bin        optional; at most one raw.* file (raw.dat, raw.bin, raw.bin.gz, ...)
  D2/
    sort/
    date.txt       2026-01-06
```

- Every folder in `DIR` is a day folder, hidden ones included. Its name must match
  `[A-Za-z0-9][A-Za-z0-9_.-]*`.
- `date.txt` holds exactly `YYYY-MM-DD`.
- A `raw.*` folder is not a raw file.
- With a `raw.*` file, `--uv-per-bit` is required. It applies to every day.
- Each path is recorded as it lies inside `DIR`, so any entry may be a link (section 8).

Every day folder is checked before anything is written. Each of these refuses the whole
command, and no day is added:

- a missing `sort/` or `date.txt`, or a malformed date;
- more than one `raw.*` file;
- a raw file without `--uv-per-bit`, or with one that is not a positive finite number;
- a raw file whose sort's `params.py` has no `n_channels_dat`;
- a day already in the project;
- two days sharing a date;
- a folder name the project cannot hold;
- a file the readers need that is missing;
- a sort or events file that cannot be opened. `init` checks the sort and events files and
  records the raw file without opening it: an unreadable raw file is refused at `extract`.

`--sort`, `--date`, `--events` and `--raw` belong to the `--day` form, and `--from` refuses
them. `init --day D1 --sort DIR --date YYYY-MM-DD [--events CSV] [--raw FILE]` adds one day.

## 7. Tiers

| | `raw` | `templates` |
|---|---|---|
| when | the default for a day with a raw file | the default for a day without one; `--tier templates` forces it on a day with one |
| reads voltage | yes: snippets around the sampled spikes | no |
| waveform | recomputed from the raw file: scaled by `uv_per_bit`, high-pass filtered, median-referenced, unwhitened | the sort's unwhitened template |
| `features/halves/` | the analysis window cut into two halves | none (`features.halves` is `null`) |
| `features/full/` | each unit's two halves pooled | from the templates and the complete spike trains |
| a unit is ineligible | when a half is, for example with fewer than 20 spikes in it; its full-day row carries that half's reason | with fewer than 20 spikes in the analysis window |
| needs | a raw file | a sort with templates |

The halves are cut at the start of the middle trial, so no trial crosses the cut
(`trial_start`). A day without trials is cut at its sample midpoint (`sample_midpoint`).
Extraction settings are fixed for every project: 41 samples before and 41 after each spike,
at most 100 spikes sampled per half, at least 20 spikes (in each half for `raw`, in the
analysis window for `templates`), seed 0. A `raw` full-day waveform pools the two halves'
samples, so it averages up to 200 spikes. Each unit's `summary_reuse` in `features/full/`
records the two halves' counts (`half_spike_counts`) and their total
(`full_mean_selected_count`).

## 8. Links and `retargeted`

multiphy never creates or changes a link. You may point `--sort`, `--events` and `--raw` at
links, or build an `init --from` folder from links to where the data lives:

```
/data/incoming/M1/D1/sort    -> /data/kilosort/M1_2026-01-05
/data/incoming/M1/D1/raw.bin -> /data/acquired/M1_2026-01-05/continuous.dat
```

Each file's identity records the path as given (`given`) and the file it resolved to then
(`resolved`). Extraction and every check of a file's size, `mtime_ns` and sha256 read
`resolved`. Re-pointing a link afterwards therefore never changes the day. `status` reports
`sources retargeted` when a `given` path now resolves to a different file:

```
D1 · 2026-02-01 · raw · sources retargeted · features ok · runs 0
```

To use the new target, add the day to a new project.

## 9. The commands

`--root DIR` goes before the command. Each command prints its result lines on stdout, and
progress on stderr. Any error prints `error: <message>` on stderr and exits 1, including a
command line that does not parse. Success, and `--help`, exit 0. SIGINT while a
command runs (Ctrl-C, or Cancel in the window) prints `cancelled` on stderr and exits 130; the
log line then ends `cancelled`, written, as for every outcome, only when the project exists.

### `init`

Adds days to a project, and creates the project when it does not exist yet. It prints only
its notes. With `--from`, each note starts with its day, for example
`D2: raw file found at dat_path; pass --uv-per-bit to use it`.

`--probe TYPE` names the project's probe type. The first `init` that gives it records it in
`identity.probe`. A later `init` with a different `--probe` is refused before anything is
written (`project M1 probe is NP1; --probe says NP2`); one without `--probe` records nothing.

```
$ python -m multiphy.launch --root /data/multiphy init M1 --from /data/incoming/M1 --uv-per-bit 0.195 --probe NP1
```

### `extract`

Extracts the named days (`--day D1 D2`, or `--day` repeated), by default every day not yet
extracted, in name order. It prints one line per day: the tier and the first 12 hex digits of
the halves id (`—` in the templates tier) and of the full id. The first refused day stops the
command, and only its error is printed. The days extracted before it stay recorded, and
`status` shows them.

```
$ python -m multiphy.launch --root /data/multiphy extract M1
D1 · raw · halves 32067c0468cc · full 639a84c5b9e5
D2 · raw · halves 2b8d89642d71 · full 38843d3eabe0
```

`extract` refuses these before it writes anything:

- a day whose `.extract.lock` is held;
- a day already extracted;
- a missing or changed file (`<role> source missing: <path>`, `<role> source changed: <path>`);
- an input its reader refuses: the events CSV and `binary` refusals in section 5.

These come after the day is normalized, so they can leave `source/` behind, but no
`features/`:

- a geometry other than the project's;
- the templates tier on a sort without templates;
- a `.gz` raw file whose bytes differ from the sha256 recorded when the extraction started
  (`raw source changed: <path>`).

One comes after `features/` is published: a `.gz` raw file whose sha256 another command
(`status --verify`) recorded during the extraction, and which differs from the bytes the
extraction read. `features/` then stays, unrecorded.

`project.json` records none of what a refused or stopped extraction leaves. The next
`extract` of the day reuses a `source/` normalized from the same recorded files and replaces
any other. It deletes an unrecorded `features/` just before it publishes its own.

### `status`

Prints one line per day, then the number of runs across days:

```
$ python -m multiphy.launch --root /data/multiphy status M1 --verify
D1 · 2026-01-05 · raw · sources ok · features ok · runs 0
D2 · 2026-01-06 · raw · sources ok · features ok · runs 0
across days · runs 0
```

The fields are day · date · tier (`—` before extraction) · sources · features · runs:

- `sources` is the first of these that applies:
  - `missing`: a recorded file is gone;
  - `changed`: a file's size, `mtime_ns` or, with `--verify`, sha256 differs;
  - `retargeted`: section 8;
  - `ok`.
- `features` is `none` before extraction. It is `stale` when a manifest under `features/` no
  longer hashes to its recorded id, or when the sources changed. Otherwise it is `ok`.

Without `--verify`, `status` compares only sizes and `mtime_ns`, and writes nothing to
`project.json`. `--verify` re-hashes every file, the raw file included. It records the sha256
of a raw file that has none yet, with `sha256_from` `"verify"`.

### run

`launch run` runs one registered backend on the project's extracted days and writes a new
run folder. The backend ids come from `python -m multiphy.run list-backends`; each backend's own
options follow `--backend ID`.

- By default every extracted day is matched across days, sorted by date; this needs 2 or more
  extracted days. `--days D1 D2 ...` names two or more days. The run goes in the project's
  `runs/`.
- `--day D1` matches the day's two temporal halves, and the run goes in `days/D1/runs/`.
- A benchmark baseline (a backend that cannot run on real days) is refused across days. A
  backend that reads trial timing runs without events when a chosen day has no trial events or
  no condition is on every chosen day: 5 s chunks stand in for trials, spont comes from the
  whole recording when a day has no baseline windows, and each fallback prints one
  `warning · <reason>` line on stderr before the run and is saved as `run_evidence` in the
  result's provenance.
- A templates-tier day is refused: it has no waveform halves.
- A day without a single unit is refused (`days without a single unit: D2; run without them`),
  and so is one day whose half has none (`day D1 half recording0000_part001 has no single unit`),
  before any warning.
- A backend available in this interpreter runs in process. With `--python EXE`, or when the
  backend is unavailable here, it runs in `EXE` or else the verified lab interpreter; when it
  fails there, the error names the run folder it left.

Every refusal comes before the run folder exists. The folder name is a UTC stamp, the backend
id and a random suffix. Its `selection.json` and `pairs.csv` are written before the backend
starts and its `result.json` when it ends; none is ever replaced. A failed run leaves its
folder with `run_manifest.json` status `failed`, and running again makes a new folder. A run
whose `selection.json` no longer matches the project refuses to open.

It prints one line: the run folder · the backend · the days · the number of units excluded
from the selection (ineligible units).

```
$ python -m multiphy.launch --root /data/multiphy run M1 --backend functional_likelihood
/data/multiphy/M1/runs/20260928T120000Z_functional_likelihood_1a2b3c4d · functional_likelihood · D1,D2 · excluded 3
```

Open a finished run of either kind with `python -m multiphy.viewer --run DIR`; it
opens in the mode the run's `selection.json` records (see [viewer.md](viewer.md)). The viewer
runs no matcher. An
across-days run is curated in `curation/runs/<run>/history.csv` (section 2); a same-day run is
inspect-only.

### export

`launch export PROJECT --run RUN [--out PATH]` writes the tracks a curator accepted in the
across-days run `RUN`, a folder in the project's `runs/`, to a new CSV. The default is
`curation/runs/<run>/exports/accepted_tracks_<YYYYMMDD-HHMMSS>.csv`; `--out PATH` names another
new file. An existing file is never overwritten, and a failed write leaves no file.

- It reads `curation/runs/<run>/history.csv` without taking `history.lock`, so it runs while a
  viewer has the run open.
- The CSV has one row per member of each accepted track, a one-unit track included, with the
  columns `track_id`, `cellid`, `siteid`, `n_units`, `n_days`, `accepted_at` and `curator`.
- A run that is not a completed run in the project's `runs/`, or whose `selection.json` is not
  across days, is refused. Same-day runs have nothing to export.

It prints one line: the CSV · the number of accepted tracks · the number of units.

```
$ python -m multiphy.launch --root /data/multiphy export M1 --run 20260928T120000Z_functional_likelihood_1a2b3c4d
/data/multiphy/M1/curation/runs/20260928T120000Z_functional_likelihood_1a2b3c4d/exports/accepted_tracks_20261005-101500.csv · 12 tracks · 31 units
```

### gui

`launch [--root DIR] gui [PROJECT]` opens the launcher window on the lab root (`--root`, else
`MULTIPHY_CACHE_ROOT`, section 1). `PROJECT` (a name, or a path directly under the root) is shown first,
and a name that is not yet a project fills the `project` field of the init and plugin forms.
`launch gui` itself writes no log line.
The window opens at 1280 × 860 (`OPEN_SIZE`), or at most 90% of the screen's free width and
height, every time; its size is not remembered.

```
$ python -m multiphy.launch --root /data/multiphy gui M1
```

**Left.** The project, its days tagged `· raw`, `· templates` or `· none`, and its runs:
`across days`, then each day's same-day runs. A run is listed once its run manifest and its
result are both written (by a backend this install knows), even when Cancel arrived after
that; an unfinished run is not. `Curate…` opens an across-days run and `Inspect…` a same-day
run, each in its own viewer window, which stays open when the launcher closes. While a run
opens, its button reads `◐ opening · <run>` and stays locked until the viewer shows; clicks
meanwhile are dropped. The spinner turns and the launcher and open viewers stay responsive
while the run is read; they pause while its window is built. A failed open logs `error: …`
and unlocks the button.

**Right.** One tab per command: Initialization (`init`), Extract features (`extract`), Run
matching (`run`), Export tracks (`export`), Status (`status`), then each plugin command under
its own name; hover a tab for its `launch` command. A field is one option of the command,
labelled by its flag in plain words, its help in hover; a required field ends ` *` and Run
waits for it; an empty backend option shows its default in grey; `…` picks a folder or file. A
plugin option a form cannot show reads `CLI only · <flag>` and that command runs from the
terminal.

**The checked days.** `extract` takes the checked days not yet extracted (none checked: every
day not yet extracted when Run is pressed); `run` takes one checked day (its two halves) or several (across days),
all extracted at the raw tier. A backend that cannot run on them is greyed, and hover says why.
For a backend that reads trial timing, tags under the backend name each evidence fallback of the
checked days (`no events`, `spont · whole recording`; the reasons in hover), as `launch run` would
warn; without events the `stimulus` feature stream is greyed. A `min_repeats` that is not an
integer of at least 2 shows one `error` tag instead and disables Run.

**Running, waiting.** Each command runs as `python -m multiphy.launch --root <root> …`
and logs its own line; the log pane shows `$ launch …`, the output and `exit <code>`. One
command runs at a time. A Run pressed meanwhile waits, listed under the log pane with `×` to
remove it and `Clear` to drop them all; a waiting command keeps the root, project and fields it
had when Run was pressed, which days are eligible is judged then, and a failed command does not
stop the next. While commands run or wait only the root is locked; another project can be
browsed and its runs opened. After a run, the new run is selected when its project is still
shown. A project that cannot be read shows `error: …` in the log pane and disables the commands
that need it.

**Cancel.** Cancel acts on the running command only and sends SIGINT once: the command exits
130 and prints `cancelled`, and its log line ends `cancelled` when the project exists; a SIGINT
that arrives while the child is still starting ends it without that line. Still running after
10 s, it is killed: the pane shows `killed`, its log line may be missing, and a killed extract
can leave `days/<day>/.extract.lock`, which the next extract names for removal. Closing the
window while a command runs asks `<command> running · cancel and close?`
(`<command> running · N waiting · cancel and close?` while N wait); `Yes` drops the waiting
commands, cancels the running one and closes when it ends.

**The root.** `Settings` ▸ `Root…` switches it; the left column shows the root read-only,
shortened in the middle, the full path in hover. A switch loads the new root's config and
plugins and builds every form before it switches; a root where any of that fails is refused
with an `error:` line, and the window keeps its root. The source readers plugins add stay
registered for the rest of the process.

**The log pane** is monospace: a child's `[… INFO]` lines are muted; `error:` lines, `exit N`
with N ≠ 0 and `killed` are in the reject colour.

**The menu bar**, shared with the viewer: `Settings` ▸ `Theme` (`Graphite`, the default, `Dark`
or `Light`) and `Settings` ▸ `Unit colours` (see [Theme](viewer.md#theme) in the viewer guide);
`Help` ▸ `Launcher guide` (this file) and `Viewer guide` (`docs/viewer.md`), each in its own
read-only window, then `Keyboard shortcuts`, where the viewer's keys are changed (one key per
action, saved in `multiphy.ini`).
`Help` ▸ `About multiphy` shows the logo and the version (`0.0.0` until launch); the logo is also
the window icon.

### `logs/launcher.log`

Each command on a project that has a `project.json` afterwards, failed ones included, appends
one line. The line holds the UTC time, the command with the arguments given after it (not
`--root`), and the outcome: `ok`, `error: <message>` or `cancelled`:

```
2026-09-26T05:44:17.699623+00:00 · init M1 --from /data/incoming/M1 --uv-per-bit 0.195 · ok
2026-09-26T05:44:27.737662+00:00 · extract M1 · ok
2026-09-26T05:44:34.713980+00:00 · status M1 --verify · ok
2026-09-26T05:46:32.987474+00:00 · extract M1 --day D1 · error: Day D1 is already extracted; features are written once. Put it in a new project to extract it again.
2026-09-26T05:47:02.104311+00:00 · extract M1 · cancelled
```

A command line that does not parse runs nothing and writes no line.
