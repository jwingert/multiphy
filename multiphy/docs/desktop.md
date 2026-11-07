# Windows desktop application

multiphy has two installation options. The Windows x64 desktop application bundles
Python, Qt and the core scientific libraries. The Python/uv installation provides the
same application in an environment you can extend for notebooks, plugins and optional
inference backends. Use the [Python setup guide](installation.md)
on Linux and macOS, or when you need that flexibility on Windows. Windows is the only
platform with an executable build workflow; macOS is not covered by the core CI suite.

## Download and install

Download the [v0.0.0 Windows desktop preview](https://github.com/jwingert/multiphy/releases/tag/v0.0.0).
This first Windows release is marked as a prerelease. Under its
**Assets**, choose the installer for regular use or the portable ZIP to run from an
extracted folder. The filenames include the application version, and
`SHA256SUMS.txt` records their hashes. Python/uv remains available alongside both
Windows downloads.

| Option | How to use it |
| --- | --- |
| Installer | Run the Windows x64 setup executable, install for your user account, then open the **multiphy** shortcut. A separate Python installation is unnecessary. |
| Portable ZIP | Extract the ZIP to a writable location and open `multiphy-gui.exe` in the extracted folder. Keep the whole folder, including `multiphy.exe` and `_internal`; the GUI executable is not a standalone file. |
| Python / uv | Follow the [Python setup guide](installation.md) to install the pinned environment, then launch with `uv run --no-sync python -m multiphy.launch --root C:\path\to\multiphy-data gui`. |

The initial prerelease's installer and executables are unsigned. Windows may
show a publisher or application-control warning. Check the release source and compare
the downloaded file's SHA-256 with `SHA256SUMS.txt`; a matching hash confirms the file
matches that release artifact, not that Windows policy permits it.

## First launch and project storage

Opening **multiphy** asks you to choose a project root when one has not been configured.
The root is the folder containing your projects, for example
`C:\Users\you\multiphy-data`. Choose an existing root to see projects you already used
through Python, or a separate folder for new work. The launcher remembers your choice.
Use **Settings → Root…** to switch folders later. An explicit `--root` argument or the
`MULTIPHY_CACHE_ROOT` environment variable takes precedence over the remembered folder.
Cancelling the initial folder dialog closes the application without creating a project.

Keep project data outside the application's installation or portable folder. Each
project stores its recordings' source references, extracted features, matching runs
and curation separately from the application. Installing a new application version
does not require re-importing recordings. Source recording files still need to be
available at the paths recorded in the project.

The desktop application includes:

- manual recording entry and folder import;
- feature extraction, including raw voltage input when its scaling is supplied;
- the `functional_likelihood` and `pair_logistic` matchers;
- the viewer, matching and review actions, curation history, and accepted-track export;
- the project checks and launcher command log.

Follow the [launcher guide](https://github.com/jwingert/multiphy/blob/main/multiphy/docs/project_folder.md)
for recording setup and matching, and the
[viewer guide](https://github.com/jwingert/multiphy/blob/main/multiphy/docs/viewer.md) for
review controls; both are also available in the application's **Help** menu. The project format is shared with
the Python installation; close one instance before editing the same run in another.

## Command-line use

The bundle also includes `multiphy.exe` for PowerShell and scripts. From the extracted
or installed application folder:

```powershell
.\multiphy.exe --help
.\multiphy.exe --version
.\multiphy.exe --root "C:\Users\you\multiphy-data" status M1
.\multiphy.exe --root "C:\Users\you\multiphy-data" extract M1
.\multiphy.exe --root "C:\Users\you\multiphy-data" run M1 --backend functional_likelihood
.\multiphy.exe --root "C:\Users\you\multiphy-data" gui
```

These are the same launcher arguments used after `python -m multiphy.launch` in the
[project guide](project_folder.md). Commands that operate on data require `--root DIR` before the command, or the
`MULTIPHY_CACHE_ROOT` environment variable. The remembered GUI folder is not an implicit
root for batch commands. `--help` and `--version` work without a root.

To use a root for the current PowerShell session:

```powershell
$env:MULTIPHY_CACHE_ROOT = "C:\Users\you\multiphy-data"
.\multiphy.exe status M1
```

## Updates

Updates are explicit downloads; the application does not automatically update itself.
Check **Help → About multiphy** or `multiphy.exe --version` to see the current version.

1. Finish or cancel active jobs and close the launcher and viewers.
2. Read the new release's notes, especially any project-format compatibility changes.
3. Install the newer setup executable, or extract its portable ZIP into a new folder.
   Keep the new portable folder complete rather than copying individual executables
   over the previous version.
4. Open the new version and select your existing project root if prompted.

Your external project folder is retained. Keep ordinary backups of projects and
curation; replacing the application is not a backup. Retaining an older portable
folder can help reproduce work, but opening a project with an older version is only
appropriate when its format remains compatible.

## Optional backends and extensions

UnitMatch, DANT and DeepUnitMatch require separate upstream dependencies and, for
DeepUnitMatch, compatible model checkpoints. They are not bundled. Use the Python/uv
installation and the [optional backend setup](installation.md#optional-backends) to provision these environments.
You can then select their interpreter in the launcher's **python** field or pass
`run --python EXE`. The external environment must contain the same multiphy version
as the desktop application as well as the backend's dependencies. Installing a package
into a system Python does not add it to the bundled application's Python runtime.

Both installation routes can work with the same project root, so you can use the
Python environment for specialized processing and open the resulting completed runs
in the desktop viewer. Optional backends must produce supported multiphy project
results; the desktop bundle does not install or train those models. Keep dependency
setup, model training, custom source readers and other third-party plugin development
in the Python installation.

## Building and checking a Windows release

Build on Windows x64 using the repository's pinned CPython 3.12.15 and uv 0.13.4.
The desktop build uses PyInstaller with separate console and GUI entry points in one
shared application folder. Build dependencies are pinned with hashes separately from
the application's lockfile. The reference installer compiler is Inno Setup 7.1.1;
it is needed only for the installer. The builder also locates an existing Inno Setup 6
installation, but CI uses the pinned 7.1.1 compiler.

From the repository root in PowerShell:

```powershell
uv python install
uv sync --locked --extra gui
uv pip install --python .venv/Scripts/python.exe --no-deps --require-hashes -r packaging/windows/requirements-build.txt
& .\.venv\Scripts\python.exe -B packaging/windows/build.py --installer
```

If Inno Setup is not on its usual path, add `--iscc "C:\path\to\ISCC.exe"`.
Omit `--installer` to build and check only the portable version. Build output goes
under `dist/desktop/`:

| Output | Contents |
| --- | --- |
| `multiphy/` | The complete unpacked application, with both executables and `_internal`. |
| `multiphy-<version>-windows-x64-portable.zip` | A portable copy of that application folder. |
| `multiphy-<version>-windows-x64-setup.exe` | The per-user installer, when requested. |
| `SHA256SUMS.txt` | SHA-256 hashes for the distributable files. |

The installer defaults to `%LOCALAPPDATA%\Programs\multiphy`, adds a Start menu entry
and offers a desktop shortcut. Project data belongs outside that location.

The builder runs the packaged smoke test before producing distribution artifacts.
To repeat it against an existing unpacked build:

```powershell
& .\.venv\Scripts\python.exe -B packaging/windows/smoke.py --dist dist/desktop/multiphy
```

The smoke test uses synthetic recordings in a temporary project to exercise import,
extraction, both built-in matchers, export and GUI startup through the actual
executables. Host Python creates the synthetic recordings and an acceptance record;
the packaged application exports that accepted track and renders the launcher and
viewer. The test also checks packaged resources and root-free help/version commands.
It does not establish compatibility for optional backend environments or replace a
manual desktop check. Before a release, install the setup executable on a clean
Windows account without a Python environment, open an existing project, run the
launcher workflow and check that uninstalling the application retains external data.

Compare an artifact with the published checksums in PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 .\multiphy-<version>-windows-x64-portable.zip
```

Substitute the actual version in the filename, and compare the returned hash to the
same filename in `SHA256SUMS.txt`.

## Publishing

`.github/workflows/windows-desktop.yml` runs in these cases:

- Pull requests that change desktop packaging or its runtime integration build and
  check the artifacts without publishing a release.
- **Run workflow** builds and checks the Windows artifacts and uploads them to the
  workflow run. It does not publish a GitHub Release.
- Pushing a `v<version>` tag builds the same artifacts and attaches them to a GitHub
  Release. The tag must match `multiphy.__version__`; a mismatch fails the build.

The release job prepares a draft, uploads and verifies the artifacts, then publishes
it. The first tag, `v0.0.0`, is a prerelease and is not marked as the latest stable
release. Its notes live in `packaging/windows/release-notes/v0.0.0.md`.

Run the core tests and review a successful manual desktop build before tagging a
release. Check the downloadable installer, ZIP and checksums, and record the included
version and known limitations in the release notes. Signing and automatic updating are
not configured by this workflow.
