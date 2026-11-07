# Installation and Python environments

The [Windows desktop application](desktop.md) bundles Python, Qt and the core
scientific libraries for recording import, extraction, built-in matching,
curation and export. It needs no separate Python installation. Use the Python/uv
route below for Linux, development, notebooks, custom plugins and optional
inference backends. Both routes use the same [project folders](project_folder.md).

## Python / uv installation

The supported core, GUI and test baseline is **Python 3.12**, with
**CPython 3.12.15** as the pinned reference patch and **uv 0.13.4** as the
environment tool. The core needs numpy, pandas, scipy and matplotlib; the
launcher window and viewer also need PySide6 (the `gui` extra). Install that uv
version using the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/),
then use the committed cross-platform lockfile:

```bash
git clone https://github.com/jwingert/multiphy.git
cd multiphy
uv python install
uv sync --locked --extra gui
```

uv reads `.python-version` and installs the same managed CPython patch on Linux
and Windows, including Windows builds supplied by Astral's
[python-build-standalone](https://docs.astral.sh/uv/guides/install-python/).
From the repository directory, start the launcher with an explicit project root
outside the application directory.

Windows PowerShell:

```powershell
uv run --no-sync python -m multiphy.launch --root "$env:USERPROFILE\multiphy-data" gui
```

Linux:

```bash
uv run --no-sync python -m multiphy.launch --root "$HOME/multiphy-data" gui
```

Run other commands through `uv run --no-sync python ...`, or activate `.venv`
before using the `python ...` examples in the guides:

```bash
# Linux
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

`--no-sync` retains the extras installed above. An unpinned
`pip install -e ".[gui]"` still works on Python 3.12, but does not reproduce this
tested dependency set. For recording import, extraction, matching and export,
continue with the [project and launcher guide](project_folder.md).

On minimal Debian/Ubuntu installations, install Qt's native graphics libraries
as well:

```bash
sudo apt-get update
sudo apt-get install --no-install-recommends -y libegl1 libgl1 libxkbcommon0
```

Headless Linux tests use `QT_QPA_PLATFORM=offscreen`, but still need these shared
libraries. Leave that variable unset when using the interactive launcher/viewer.

### Using an existing Python on Windows

If uv's managed Python cannot run on your Windows machine, choose an installed
CPython 3.12 interpreter that works there. Replace the example path with its
absolute executable path:

```powershell
uv sync --locked --no-managed-python --python "C:\path\to\Python312\python.exe" --extra gui
& .\.venv\Scripts\python.exe -m multiphy.launch --root "$env:USERPROFILE\multiphy-data" gui
```

This keeps the locked dependencies while using that Python installation. A
different Python 3.12 patch is within the supported minor version, but does not
reproduce the pinned CPython 3.12.15 reference exactly.

## Supported baseline and tests

The supported Python range is `>=3.12,<3.13`; neither older nor newer minors are
currently supported. Python 3.9 reached [end of life in October 2025](https://peps.python.org/pep-0596/),
and the previous `>=3.9` claim and 3.9 CI lane were intentionally removed.
Windows coverage remains part of this policy; reducing Python minors is not a
substitute for fixing Windows failures. macOS is not currently covered by CI.

CI installs the locked core, GUI and test extras and runs the full suite,
including GUI and interprocess workflow-lock tests, on Ubuntu and Windows.
The packaging test builds and installs a wheel offline into a temporary
environment, checks its resources and installed CLI, and verifies that imports
come from the wheel rather than the repository or an inherited `PYTHONPATH`.
The lockfile pins all transitive dependencies; the reference stack includes
numpy 2.5.4, pandas 3.1.0, scipy 1.18.1, matplotlib 3.11.2, PySide6 6.12.0 and
pytest 9.1.1. CI prints the actual Python, uv and installed package versions.

To reproduce the test environment and run the suite:

```bash
uv sync --locked --extra gui --extra test
uv run --no-sync python -B -m pytest tests -q -ra --durations=10 -p no:cacheprovider
```

Set `QT_QPA_PLATFORM=offscreen` when running without a display:

```bash
export QT_QPA_PLATFORM=offscreen
```

```powershell
$env:QT_QPA_PLATFORM = "offscreen"
```

The exact patch and dependency lock are a reproducible reference, not a promise
to freeze security updates indefinitely. To refresh the baseline:

1. Choose and review the new Python patch in `.python-version`.
2. Keep uv's version in `pyproject.toml` and CI aligned, and review the build-tool
   constraints.
3. Run `uv lock --upgrade` and verify a fresh environment on **both** CI operating
   systems before publishing the new baseline.

## Optional backends

`functional_likelihood` and `pair_logistic` need nothing beyond the core
installation. UnitMatch, DANT and DeepUnitMatch require separately provisioned
upstream dependencies and, for DeepUnitMatch, a compatible model checkpoint.
They are not included in the Windows desktop bundle or the locked core/GUI
environment. The launcher installs nothing.

```bash
python -m multiphy.run list-backends
```

This lists the registered backends and which ones the current interpreter can
run. `launch run` selects an interpreter in this order:

1. Explicit `--python EXE` (the launcher's **python** field), even if the backend
   is also available locally.
2. The launcher's own interpreter, when the backend is available there.
3. The lab root's shared environment, `<root>/dependencies/inference-v3`, when
   its completed manifest, recorded file hashes and smoke-test record verify.

With none of these, the run refuses; the launcher disables unavailable backends
and shows the reason on hover. An external interpreter used from the desktop
application must contain the same multiphy version as the application.
See [matching methods and run settings](../MATCHING_INTERFACE.md) for backend
options and interpreter selection.

The lockfile and Linux/Windows core/GUI tests do **not** establish compatibility
for optional upstream inference or model training. Provision and smoke-test
those dependencies, checkpoints and their base environment separately.

### Shared inference environment

Build one shared environment per lab root. It holds UnitMatch, DeepUnitMatch
and pyDANT Python sources at pinned revisions, plus hdbscan 0.8.42 and mat73 0.65,
on top of a base Python. In a source installation, each run passes the caller's
own multiphy to it.

The base must be a conda environment or a system Python with multiphy installed
with `gui`, plus torch, scikit-learn, h5py, tqdm, joblib and networkx. It cannot
be another venv, including the locked `.venv`: the shared environment is a venv
with system site packages and cannot see packages installed in another venv.
Run these commands with that prepared base Python, outside the locked `.venv`:

```bash
git clone https://github.com/EnnyvanBeest/UnitMatch
git -C UnitMatch checkout 95b6fed5db84d12f67487b92d07f0a2664a6122f
git clone https://github.com/jiumao2/pyDANT
git -C pyDANT checkout 741240511e6a4ed9315b2c80d71ab5cfe4c588eb
python -m pip install --no-deps --target overlay hdbscan==0.8.42 mat73==0.65
```

On Linux, stage and check the environment:

```bash
python -m multiphy.dependency_setup --root "$HOME/multiphy-data" \
    --unitmatch-source UnitMatch --dant-source pyDANT --overlay overlay
"$HOME/multiphy-data/dependencies/inference-v3/venv/bin/python" -m multiphy.run list-backends
```

On Windows PowerShell:

```powershell
python -m multiphy.dependency_setup --root "$env:USERPROFILE\multiphy-data" --unitmatch-source UnitMatch --dant-source pyDANT --overlay overlay
& "$env:USERPROFILE\multiphy-data\dependencies\inference-v3\venv\Scripts\python.exe" -m multiphy.run list-backends
```

Substitute your existing project root if different. Setup refuses a checkout
that differs from its pinned revision or an existing `inference-v3` folder. A
failed setup leaves the folder with `status: failed`; `environment.json` and
`setup.log` name the cause. Inspect that failure and remove the failed output
folder before retrying. Setup copies Python sources and the checked dependency
overlay, so the original clones and overlay can be deleted afterwards. It does
not copy upstream bundled models or checkpoints. The environment inherits the
base Python's packages, so changing the base changes the shared environment.

`environment.json`, `smoke.json`, `setup.log` and `METHODS.md` record the source
revisions, file hashes, package versions and import/network-construction check.
That check is neither training nor evidence of matching accuracy. The copied
sources support inference; future training preparation requires a separate
clean pinned Git checkout. For alternate output locations, see the
[durable environment reference](../MATCHING_INTERFACE.md#durable-inference-environment).

### Recording, model and device requirements

- UnitMatch and DANT need raw-waveform features: days with a raw file, extracted
  at `--tier raw`. DANT's default settings need at least 38 channels on a shank.
- DeepUnitMatch needs 384 channels, `--checkpoint FILE` and
  `--checkpoint-probe ID`. The model must have been trained for your probe type;
  channel count alone does not establish compatibility. The upstream bundled
  model fits only Neuropixels 2.0 four-shank probes and is not copied by setup.
  Use the exact probe identifier from its training manifest or documentation;
  the sample rate and waveform preprocessing must also match the recording.
- DeepUnitMatch's `--device` defaults to `cpu`. A GPU device such as `cuda:0`
  requires a CUDA-enabled PyTorch installation and an available GPU.

The [matching reference](../MATCHING_INTERFACE.md) covers model settings,
tracking choices and backend-specific options.
