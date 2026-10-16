<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: user guide to writing and registering a source reader, with one worked example that the tests execute (multiphy project folder Task 16; fix round 1: binary checks only a declared channel count, and events_csv checks only the last trial against the length; fix round 2: the example's code carries the AGENT EDIT tags; final review fix round 1: init always passes the declared channel count) | date: 2026-09-25] -->
# Adding a source

A **source reader** turns one day's original files into a `DaySource`, the one form that
extraction reads. multiphy ships three readers:

- `phy` reads a KiloSort or phy folder;
- `events_csv` reads an events CSV;
- `binary` checks a flat or gzip raw voltage file.

When your spikes, events or voltage come in another format, add a reader. Nothing else in the
package changes: `add_day` records the reader by its id, and `extract_day` and `status` work on
the day as on any other. The launcher's `init` only builds `phy` / `events_csv` / `binary` days,
so a day with your reader is added from Python with `add_day`. Once your reader is installed
through the entry point below, `extract` and `status` run on that day from the command line
too. The project folder itself is described in [`project_folder.md`](project_folder.md).

## Roles

A reader serves one or more of three roles:

- `sort` gives the units and their spikes;
- `events` gives trials, stimuli and baseline windows;
- `raw` gives the voltage file's parameters.

A day needs a sort reader, and events and raw are optional. `read_day` runs a day's readers in
the order sort, raw, events. The sort reader gets `day=None` and starts the day. Each later
reader gets the `DaySource` built so far and returns a new one, usually with
`dataclasses.replace`. A reader that serves several roles runs once, with those roles' files
merged.

## A reader

A reader is any object with:

| Member | |
|---|---|
| `id` | a unique string. The day records it as `readers[role] = "id@version"` |
| `version` | an integer |
| `roles` | a `frozenset`, a subset of `{'sort', 'events', 'raw'}` |
| `files(options)` | returns `{key: Path}` for every file `read` will open |
| `read(files, options, day=None)` | returns a `DaySource` |

### `files`

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the raw-role key rule: key raw or raw_<k> is a raw-role file, never hashed; raw segments (Plan C Task 2) | date: 2026-10-01] -->
`add_day` calls `files(options)` before any data is read. It records each file's identity:
the path as given, the file it resolves to, its size, its `mtime_ns` and its sha256. Every
file except the raw-role ones (below) is hashed. At extraction, every file is verified against
that identity, and `read` gets the resolved paths under the same keys. A file that `files`
does not name is not recorded or checked, so name every file `read` opens.

- File keys must be unique across a day's readers. The built-ins use `events`, `raw` and the
  phy file names (`params`, `spike_times`, ...).
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: raw-role files are not checked by a separate hash before extraction; the voltage pass hashes them (Plan C Task 2 fix round 1) | date: 2026-10-01] -->
- Key `raw`, and any key starting `raw_` (such as `raw_1`, `raw_2`), is always recorded under
  the `raw` role and is never hashed when the day is added, nor checked by a separate hash
  before extraction. Any other key is recorded under the reader's first role.
<!-- [AGENT EDIT END] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: say where raw-role files are hashed: the raw-tier voltage pass and status --verify (Plan C Task 2 fix round 1) | date: 2026-10-01] -->
<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: only one .gz file hashes at no extra read; segment hashing can reread after seeks (Plan C Task 2 fix round 2) | date: 2026-10-01] -->
- The raw tier's voltage pass hashes the stored bytes as it reads them. One `.gz` raw file:
  gzip reads it in order, so the digest costs no extra read; it must match a recorded
  `sha256`, and is recorded with `sha256_from` `"extract"`. Segments: the pass computes one
  digest over all of them in order, and neither compares nor records it. A segment's digest
  takes only the stored bytes read at its hashing cursor; after a seek skips ahead or goes
  back, finishing the segment rereads its stored bytes from that cursor, so a pass that seeks
  can read a segment's bytes more than once. A flat single raw file is not hashed by
  extraction.
<!-- [AGENT EDIT END] -->
- An explicit `status --verify` hashes every raw-role file, and records the digest of one
  that has none, with `sha256_from` `"verify"`.
<!-- [AGENT EDIT END] -->
- A day whose voltage is several recordings on one clock names one `raw_<k>` key per
  recording, and its `DaySource.raw` holds `segments`, a list of
  `{'path': str, 'n_samples': int}` in clock order, in place of `path`. Extraction reads the
  segments as one file, and stops when a segment holds another number of samples than
  declared.
<!-- [AGENT EDIT END] -->

`options` is the day's `reader_options`: one dict, shared by all of the day's readers and
stored in `project.json`. It must be plain JSON, and its names must not collide with the
built-ins' names: `day`, `sort_dir`, `events_path`, `raw_path`, `uv_per_bit`, `raw_dtype`,
`raw_offset`, `raw_n_channels`.

`binary` cannot count a flat file's channels. It compares `raw_n_channels` with the
geometry's row count and refuses a difference. Without `raw_n_channels`, it takes the
geometry's row count, and a file with extra channels can pass as more samples of fewer
channels. `init` always passes the count from `params.py` (`project_folder.md`, section 5).
When you call `add_day` yourself, put the acquired channel count in `raw_n_channels`, as the
example below does.

### `read`

`read` returns a `DaySource`, and never reads voltage: the raw tier reads that later, from the
parameters in `DaySource.raw`. The day's `source/` folder is written once from what `read`
returns, so `read` must give the same answer for the same files.

### `DaySource`

`DaySource` (in `multiphy.sources.base`) is a frozen dataclass. Every time in it
is a 0-based sample on the raw clock, that is, a row of the raw file.

| Field | Holds |
|---|---|
| `sample_rate` | Hz |
| `n_samples` | the recording's length in samples |
| `analysis_window` | `(start, stop)`, stop exclusive; spikes at or after `stop` are left out |
| `spikes` | unit id -> sorted int64 samples, single units only; ids are unique strings, `<day>:<cluster>` for the built-ins |
| `su_type` | unit id -> `'SU'` |
| `geometry` | `(n_channels, 2)` channel x, y in µm; row i is raw-file column i |
| `clock` | `sample_index_origin` `'zero_based_voltage_row'`, `method` (free text), and `n_samples_from`: `'last_spike'`, `'raw_size'`, `'events'` or `'clock'` |
| `provenance` | a JSON-able dict, copied into `source/extraction_manifest.json` |
| `templates` | unit id -> `(n_template_samples, n_channels)` unwhitened template, or `None`; the templates tier needs them |
| `trials`, `baseline_windows` | `(n, 2)` int64 `[start, stop)` windows |
| `stimuli` | a tuple of `{'stimulus', 'start', 'stop', 'trial'}` dicts; `stimulus` is the complete condition label |
| `raw` | the `binary` reader's parameters, or `None` |
| `extras` | optional per-unit arrays that no backend reads |
| `channel_map` | the int64 raw-file column of each geometry row, or `None`; `binary` accepts only the identity |

`clock['n_samples_from']` says where `n_samples` came from, and the later readers depend on
it:

- `binary` sets the length from a flat file's size.
- A gzip file's size says nothing about its length, so a `.gz` raw file needs a sort reader
  that sets `n_samples` itself and says so with `'clock'`. `binary` refuses a `.gz` file after
  a `'last_spike'` length.
- `events_csv` may extend a `'last_spike'` length to the stop of the last trial. It refuses
  a last trial that ends after any other length. Baseline windows are never checked against
  the length; features count only the part of each inside the window being measured.

## Registering a reader

An installed package registers a reader with an entry point. Nothing in this repository
changes:

```toml
# pyproject.toml, in a package that depends on nems_lbhb
[project.entry-points."multiphy.sources"]
example_npz = "my_package.readers:make_npz_sort_reader"
```

`make_npz_sort_reader()` returns the reader.

- Entry points are loaded once, on the first listing (`reader_ids()`) or the first id that is
  not yet registered.
- One that fails to load is recorded in `sources.base.READER_ERRORS` (entry-point name ->
  error) and skipped.
- An entry point whose reader id is already taken, for example by a built-in, is skipped.

Within one process, `register_reader(reader)` registers a reader directly. It refuses an id
that is already taken. The example below uses it, since there is no package to install here.

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the launcher's plugin hook: multiphy.json, register(api), add_reader, add_command and the multiphy.plugins entry points (Plan C Task 1) | date: 2026-10-01] -->
## Plugins

A plugin adds readers and launcher commands without changing this package. It is a module
with a `register(api)` function. The launcher finds plugins in two places, before it parses
the command line:

- `<root>/multiphy.json`, where `<root>` is `--root DIR` given before the command word, else
  `run.config.cache_root()`:

  ```json
  {"plugins": ["my_package.multiphy_plugin"]}
  ```

  `"plugins"` is a list of module names; a missing key means none. Each module is imported
  and its `register(api)` is called.
- the `multiphy.plugins` entry-point group of an installed package, where each entry point
  names a `register` function:

  ```toml
  [project.entry-points."multiphy.plugins"]
  my_plugin = "my_package.multiphy_plugin:register"
  ```

A module named twice registers once. A plugin that fails to import or register stops the
launcher with `error: plugin <name>: <error>`.

`register(api)` calls:

<!-- [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the refused names include export and gui; the launcher window shows plugin commands as tabs (launcher-gui Task 11) | date: 2026-10-07] -->
- `api.add_reader(reader)` registers a reader, as `register_reader` does. Loading the plugins
  again in one process keeps the same reader object; a different object under a taken id is
  refused.
- `api.add_command(name, configure, handler, help='')` adds a launcher command.
  `configure(parser)` adds the command's own options to its `argparse` parser.
  `handler(args, root)` runs it and returns the lines to print. A name that is already a
  command (`init`, `extract`, `status`, `run`, `export`, `gui`, or one an earlier plugin
  added) is refused. The launcher window (`launch gui`) shows each plugin command as a tab
  with one field per option, labelled by its flag. An option a form cannot show (an action
  other than store, store_true or store_false; a `nargs` other than the default, `+` or `*`;
  a positional list; a list with choices) shows `CLI only · <flag>`, and that command runs
  from the terminal.
<!-- [AGENT EDIT END] -->

A plugin command's first argument is the project, as for the built-in commands:

```
$ python -m multiphy.launch --root <root> hello demo --word x
```

Its outcome goes to the project's `logs/launcher.log` like any other command.
<!-- [AGENT EDIT END] -->

## Worked example: a sort in one `.npz` file

`main(root)` does five things:

1. It writes a small day under `root`: 4 channels, 2 units, a `.npz` sort file, an events CSV
   and an int16 raw file.
2. It registers the reader `example.npz`, which serves `sort`.
3. It creates project `demo` under the lab root `root/lab`.
4. It adds the day with `add_day`.
5. It extracts the day and returns what `extract_day` returns.

The test `tests/test_docs_project_folder.py` executes this block as written.
Its first and last lines are this repository's `AGENT EDIT` audit tags; a copy can drop
them.

```python
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: worked example of a sort reader registered in-process, added to project demo and extracted in the raw tier (multiphy project folder Task 16) | date: 2026-09-25]
from pathlib import Path
from typing import Optional

import numpy as np

from multiphy.extract.pipeline import extract_day
from multiphy.project.days import add_day
from multiphy.project.folder import Project
from multiphy.sources.base import DaySource, register_reader


class NpzSortReader:
    """Serves the sort role from one .npz file: spike_times, spike_units, geometry, sample_rate."""
    id = 'example.npz'
    version = 1
    roles = frozenset({'sort'})

    def files(self, options: dict) -> dict:
        return {'sort': Path(options['sort_path'])}

    def read(self, files: dict, options: dict, day: Optional[DaySource] = None) -> DaySource:
        with np.load(files['sort']) as saved:
            times = saved['spike_times'].astype(np.int64)
            units = saved['spike_units']
            geometry = saved['geometry'].astype(float)
            sample_rate = float(saved['sample_rate'])
        spikes = {f"{options['day']}:{unit}": np.sort(times[units == unit]) for unit in np.unique(units)}
        n_samples = int(times.max()) + 1  # until the raw reader sets it from the file size
        return DaySource(
            sample_rate=sample_rate, n_samples=n_samples, analysis_window=(0, n_samples),
            spikes=spikes, su_type={unit: 'SU' for unit in spikes}, geometry=geometry,
            clock={'sample_index_origin': 'zero_based_voltage_row', 'method': 'example .npz spike samples',
                   'n_samples_from': 'last_spike'},
            provenance={'reader': f'{self.id}@{self.version}'})


def main(root):
    """Write a small day under ``root``, add it to project ``demo`` and extract it."""
    root = Path(root)
    inputs = root / 'inputs' / 'D1'
    inputs.mkdir(parents=True)

    # 1. A small day: 4 channels, 2 units, 4 trials, 4 s at 30 kHz.
    rng = np.random.default_rng(0)
    sample_rate, n_samples, uv_per_bit = 30000.0, 120000, 0.195
    geometry = np.array([[0, 0], [32, 0], [0, 20], [32, 20]], dtype=float)  # µm
    spikes = {unit: np.sort(rng.choice(np.arange(200, 116800), size=150, replace=False)) for unit in (0, 1)}
    np.savez(inputs / 'sort.npz',
             spike_times=np.concatenate([spikes[0], spikes[1]]),
             spike_units=np.repeat([0, 1], 150),
             geometry=geometry, sample_rate=sample_rate)

    rows = ['kind,start,stop,trial,condition']
    for i in range(4):
        start = 30000 * i
        rows += [f'trial,{start},{start + 27000},{i + 1},',
                 f'baseline,{start},{start + 3000},,',
                 f'stimulus,{start + 3000},{start + 15000},{i + 1},{"AB"[i % 2]}']
    (inputs / 'events.csv').write_text('\n'.join(rows) + '\n')

    voltage = rng.normal(0, 5, (n_samples, 4))  # µV
    shape = -80 * np.exp(-((np.arange(82) - 41) / 3) ** 2)
    for unit, channel in ((0, 1), (1, 2)):
        for s in spikes[unit]:
            voltage[s - 41:s + 41, channel] += shape
    (voltage / uv_per_bit).round().astype('<i2').tofile(inputs / 'raw.bin')

    # 2. The reader, in-process; an installed package would use the multiphy.sources entry point.
    register_reader(NpzSortReader())

    # 3. A project and its day: add_day records each file's identity and reads no data.
    project = Project.create(root / 'lab', 'demo')
    add_day(project, 'D1',
            readers={'sort': 'example.npz', 'events': 'events_csv', 'raw': 'binary'},
            reader_options={'day': 'D1', 'sort_path': str(inputs / 'sort.npz'),
                            'events_path': str(inputs / 'events.csv'),
                            'raw_path': str(inputs / 'raw.bin'), 'uv_per_bit': uv_per_bit,
                            'raw_n_channels': 4},  # the acquired count; binary checks it against the geometry
            date='2026-01-05', date_source='user')

    # 4. Extract: normalize into source/, cut the halves, read the raw voltage, pool the full day.
    return extract_day(project, 'D1')
# [AGENT EDIT END]
```

`main(root)` returns the day, its tier, its `source_id` and its feature ids. The ids depend
on when the example runs:

```
{'day': 'D1', 'tier': 'raw', 'source_id': '65ea1bf6e8c5…',
 'features': {'halves': 'f09373a7e3f7…', 'full': '56753697cdc2…'}}
```

- The raw reader set the length from the file size: 120000 samples.
- The events reader then set the analysis window to `[0, 117000)`, the stop of the last trial.
- The halves were cut at sample 60000, the start of the third of the four trials
  (`halves rule trial_start, boundary 60000` in `features/extract.log`).
- Both units are eligible in `root/lab/demo/days/D1/features/full/units.csv`.

`project.json` records the sort reader as `example.npz@1`. `status` would report the day
from the command line even without the reader, since it reads only the recorded identities
and features:

```
$ python -m multiphy.launch --root <root>/lab status demo
D1 · 2026-01-05 · raw · sources ok · features ok · runs 0
across days · runs 0
```
<!-- [AGENT EDIT END] -->
