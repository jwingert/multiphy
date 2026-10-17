"""Stage pinned source and already installed wheels into a dedicated environment.

No network, dependency resolution, model downloads, matching or training. The
result inherits the existing scientific stack and records its versions. This is
an inference environment, not a full Git checkout for preparing new training.
"""
import argparse
import base64
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .run.config import resolve_cache_path

DEFAULT_DIRECTORY = 'dependencies/inference-v2'
UNITMATCH_REVISION = '95b6fed5db84d12f67487b92d07f0a2664a6122f'
DANT_REVISION = '741240511e6a4ed9315b2c80d71ab5cfe4c588eb'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git(checkout, *args):
    return subprocess.check_output(['git', '-C', str(checkout), *args])


def _source_files(checkout, revision, folders):
    """Accept only committed, unchanged Python files at the audited revision."""
    checkout = Path(checkout).resolve()
    if _git(checkout, 'rev-parse', 'HEAD').decode().strip() != revision:
        raise ValueError(f'Source checkout is not the verified revision: {checkout}')
    files = []
    for folder, destination in folders:
        tracked = _git(checkout, 'ls-tree', '-r', '--name-only', revision, folder).decode().splitlines()
        selected = [p for p in tracked if p.endswith('.py')]
        if not selected:
            raise ValueError(f'No committed Python sources found: {folder}')
        for relative in selected:
            source = checkout / relative
            committed = _git(checkout, 'show', f'{revision}:{relative}')
            if source.is_symlink() or source.read_bytes() != committed:
                raise ValueError(f'Official source differs from committed revision: {source}')
            files.append((source, Path(destination) / Path(relative).relative_to(folder)))
    license_path = checkout / 'LICENSE'
    if license_path.read_bytes() != _git(checkout, 'show', f'{revision}:LICENSE'):
        raise ValueError('Source license differs from committed revision')
    files.append((license_path, Path('licenses') / (checkout.name + '-LICENSE')))
    return files


def _overlay_files(overlay, package, version):
    """Verify installed-wheel RECORD hashes before copying runtime files."""
    overlay = Path(overlay).resolve()
    metadata = f'{package}-{version}.dist-info'
    record = overlay / metadata / 'RECORD'
    files, recorded = [], set()
    for relative, signature, size in csv.reader(record.read_text().splitlines()):
        item = Path(relative)
        if '__pycache__' in item.parts or item.suffix == '.pyc':
            continue
        # hdbscan's wheel also records a build-time pxd outside site-packages;
        # it is not needed at runtime and must never escape the destination.
        if item.is_absolute() or '..' in item.parts:
            continue
        if item.parts[0] not in (package, metadata):
            raise ValueError(f'Unexpected wheel path: {relative}')
        source = overlay / item
        if source.is_symlink() or not source.is_file():
            raise ValueError(f'Missing or indirect installed-wheel file: {source}')
        if signature:
            kind, digest = signature.split('=', 1)
            actual = base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest()).decode().rstrip('=')
            if kind != 'sha256' or actual != digest or source.stat().st_size != int(size):
                raise ValueError(f'Installed-wheel integrity mismatch: {source}')
        elif relative != metadata + '/RECORD':
            raise ValueError(f'Unhashed runtime dependency file: {relative}')
        files.append((source, item))
        recorded.add(item.as_posix())
    for source in (overlay / package).rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts and source.suffix != '.pyc':
            if source.relative_to(overlay).as_posix() not in recorded:
                raise ValueError(f'Unrecorded runtime dependency file: {source}')
    return files


_SMOKE = '''import importlib, importlib.metadata, json, pathlib, sys
import UnitMatchPy, pyDANT.Runner, hdbscan, mat73
from multiphy.backends._upstream import deep_module as _deep_module
from multiphy.backends.deepunitmatch import create_deepunitmatch_model
_deep_module('testing.test')
model = create_deepunitmatch_model(device='cpu')
packages = ('numpy','scipy','pandas','scikit-learn','torch','hdbscan','mat73','PySide6')
def version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return getattr(importlib.import_module(name), '__version__', 'metadata unavailable')
result = dict(python=sys.executable, prefix=sys.prefix,
    origins={name: str(importlib.util.find_spec(name).origin or list(importlib.util.find_spec(name).submodule_search_locations)[0])
             for name in ('UnitMatchPy','DeepUnitMatch','pyDANT','hdbscan','mat73')},
    versions={name: version(name) for name in packages},
    model_type=type(model).__name__, parameter_count=sum(p.numel() for p in model.parameters()),
    model_construction='Random initialization only; no checkpoint loaded, no optimization performed')
pathlib.Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
'''


def verified_interpreter(*, root, directory=DEFAULT_DIRECTORY):
    """Return the durable interpreter only when its completed manifest verifies."""
    try:
        folder = resolve_cache_path(directory, root=root)
        manifest = json.loads((folder / 'environment.json').read_text())
        if manifest.get('status') != 'completed' or manifest.get('schema_version') != 1:
            return None
        interpreter = folder / 'venv' / 'bin' / 'python'
        if manifest['interpreter'] != str(interpreter) or not os.access(interpreter, os.X_OK):
            return None
        files = manifest['sha256']
        required = {'venv/bin/python', 'venv/pyvenv.cfg', 'smoke.json'}
        if not required <= set(files) or not any(p.endswith('/unitmatch-inference.pth') for p in files):
            return None
        for relative, expected in files.items():
            path = (folder / relative).resolve()
            if not path.is_relative_to(folder) or _sha(path) != expected:
                return None
        smoke = json.loads((folder / 'smoke.json').read_text())
        if smoke.get('model_type') != 'SpatioTemporalCNN_V2' or smoke.get('parameter_count', 0) <= 0:
            return None
        if set(smoke.get('origins', {})) != {'UnitMatchPy', 'DeepUnitMatch', 'pyDANT', 'hdbscan', 'mat73'}:
            return None
        if any(not Path(value).is_relative_to(folder) for value in smoke['origins'].values()):
            return None
        return str(interpreter)
    except (OSError, ValueError, KeyError, TypeError):
        return None


def setup_environment(unitmatch_source, dant_source, overlay, *, output_dir=DEFAULT_DIRECTORY, root=None):
    """Build one new local environment; preserve all existing environments/runs."""
    output = resolve_cache_path(output_dir, root=root)
    if output.exists():
        raise FileExistsError(output)
    sources = _source_files(unitmatch_source, UNITMATCH_REVISION,
        [('UnitMatchPy/UnitMatchPy', 'UnitMatchPy'), ('UnitMatchPy/DeepUnitMatch', 'DeepUnitMatch')])
    sources += _source_files(dant_source, DANT_REVISION, [('pyDANT', 'pyDANT')])
    dependencies = _overlay_files(overlay, 'hdbscan', '0.8.42') + _overlay_files(overlay, 'mat73', '0.65')
    output.mkdir(parents=True)
    manifest = dict(schema_version=1, status='building', created_at=datetime.now(timezone.utc).isoformat(),
        base_interpreter=sys.executable, source_revisions={'UnitMatch': UNITMATCH_REVISION, 'DANT': DANT_REVISION},
        source_checkouts={'UnitMatch': str(Path(unitmatch_source).resolve()), 'DANT': str(Path(dant_source).resolve())},
        excluded_artifacts='All upstream non-Python data, bundled models and checkpoints; no Git checkout copied',
        environment_policy='Isolated venv with copied interpreter; inherit existing system scientific packages; no base modifications',
        training_scope='Durable inference sources, not a full Git source_checkout for training preparation')
    manifest_path = output / 'environment.json'
    manifest_path.write_text(json.dumps(manifest, indent=2))
    (output / 'METHODS.md').write_text(f'''# Matching inference dependencies — methods notes

- **Output folder:** `{output}`
- **Artifacts:** `venv/`, `sources/`, `overlay/`, `environment.json`, `smoke.json`, `setup.log`
- **Generated by:** `multiphy/dependency_setup.py:setup_environment()` · **Env:** `{sys.executable}` · **Updated:** `{datetime.now(timezone.utc).date()}`

## Goal

Make the verified published matching implementations available from a durable Python interpreter. This setup performs no scientific analysis, training, model selection or checkpoint loading.

## Data lineage

- `sources/` ← unchanged committed Python files and licenses from UnitMatch `{UNITMATCH_REVISION}` and DANT `{DANT_REVISION}`; bundled pretrained files are excluded.
- `overlay/` ← already tested hdbscan 0.8.42 and mat73 0.65 distributions; runtime file hashes checked against wheel RECORD metadata before copying.
- `venv/` ← current Python with system scientific packages visible and a copied executable; its private `.pth` points only to durable sources, overlay and this project.
- `smoke.json` ← isolated import checks and construction of the official DeepUnitMatch network without loading weights or running optimization.

## Panels

No figures or scientific results. `environment.json` records copied-file hashes, dependency versions and setup status.

## Caveats

Inherited scientific packages remain tied to the existing base environment. Binary hdbscan compatibility is checked by actual import in the copied interpreter. This source snapshot is for inference; training preparation still requires a separate clean pinned Git checkout.
''')
    try:
        for destination, files in (('sources', sources), ('overlay', dependencies)):
            for source, relative in files:
                target = output / destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
        subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', '--copies', '--without-pip', str(output / 'venv')], check=True)
        interpreter = output / 'venv' / 'bin' / 'python'
        env = dict(os.environ)
        env.pop('PYTHONPATH', None)
        runtime = output / 'runtime'
        runtime.mkdir()
        env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', MPLCONFIGDIR=str(runtime),
                   XDG_CACHE_HOME=str(runtime), TMPDIR=str(runtime), OMP_NUM_THREADS='1')
        site = Path(subprocess.check_output([str(interpreter), '-c', 'import sysconfig;print(sysconfig.get_path("purelib"))'], env=env, text=True).strip())
        (site / 'unitmatch-inference.pth').write_text('\n'.join(map(str, [output / 'sources', output / 'overlay', Path(__file__).resolve().parents[1]])) + '\n')
        with (output / 'setup.log').open('w') as log:
            subprocess.run([str(interpreter), '-c', _SMOKE, str(output / 'smoke.json')], env=env,
                           cwd=output, stdout=log, stderr=subprocess.STDOUT, check=True)
        smoke = json.loads((output / 'smoke.json').read_text())
        if any(not Path(value).is_relative_to(output) for value in smoke['origins'].values()):
            raise ValueError('Smoke test imported an optional backend outside its durable snapshot')
        tracked = [p for directory in ('sources', 'overlay') for p in (output / directory).rglob('*') if p.is_file()]
        tracked += [interpreter, output / 'venv' / 'pyvenv.cfg', site / 'unitmatch-inference.pth', output / 'smoke.json']
        manifest.update(status='completed', interpreter=str(interpreter), smoke=smoke,
                        sha256={str(p.relative_to(output)): _sha(p) for p in tracked})
    except BaseException as exc:
        manifest.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False))
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--unitmatch-source', required=True)
    parser.add_argument('--dant-source', required=True)
    parser.add_argument('--overlay', required=True)
    parser.add_argument('--output', default=DEFAULT_DIRECTORY)
    parser.add_argument('--root', metavar='DIR', help='lab root · else MULTIPHY_CACHE_ROOT')
    args = parser.parse_args(argv)
    print(setup_environment(args.unitmatch_source, args.dant_source, args.overlay, output_dir=args.output,
                            root=args.root))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
