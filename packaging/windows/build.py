"""Build and smoke-test the Windows x64 desktop distribution from a locked env."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BUILD = ROOT / 'build' / 'desktop'
ASSETS = BUILD / 'assets'
OUTPUT = ROOT / 'dist' / 'desktop'


def version():
    """Read the version without importing the application or its dependencies."""
    tree = ast.parse((ROOT / 'multiphy' / '__init__.py').read_text(encoding='utf-8'))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__version__'
                for target in node.targets):
            value = ast.literal_eval(node.value)
            if re.fullmatch(r'\d+\.\d+\.\d+', value):
                if all(int(part) <= 65535 for part in value.split('.')):
                    return value
    raise ValueError('Desktop releases require a numeric major.minor.patch __version__.')


def check_environment():
    if sys.platform != 'win32' or platform.machine().lower() not in ('amd64', 'x86_64'):
        raise ValueError('Build the Windows x64 application on Windows x64.')
    expected = (ROOT / '.python-version').read_text(encoding='utf-8').strip()
    if platform.python_version() != expected:
        raise ValueError('Use Python {} from .python-version; found {}.'.format(
            expected, platform.python_version()))
    for line in (HERE / 'requirements-build.txt').read_text(encoding='utf-8').splitlines():
        match = re.match(r'^([A-Za-z0-9_-]+)==([0-9.]+)', line)
        if match:
            name, wanted = match.groups()
            installed = importlib.metadata.version(name)
            if installed != wanted:
                raise ValueError('{} {} required; found {}. Install requirements-build.txt.'.format(
                    name, wanted, installed))


def dependency_distributions():
    """Application GUI dependency closure, independent of local test/dev packages."""
    from packaging.requirements import Requirement
    pending = ['multiphy', 'pyinstaller']
    found = {}
    while pending:
        name = pending.pop()
        canonical = re.sub(r'[-_.]+', '-', name).lower()
        if canonical in found:
            continue
        dist = importlib.metadata.distribution(name)
        found[canonical] = dist
        if canonical == 'pyinstaller':
            continue  # Its bootloader is shipped, its build-time dependencies are not.
        extras = ('', 'gui') if canonical == 'multiphy' else ('',)
        for text in dist.requires or ():
            requirement = Requirement(text)
            if requirement.marker is None or any(
                    requirement.marker.evaluate({'extra': extra}) for extra in extras):
                pending.append(requirement.name)
    return found


def collect_notices(distributions):
    target = ASSETS / 'licenses'
    target.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / 'LICENSE', target / 'multiphy-MIT.txt')
    for source in sorted((HERE / 'licenses').glob('*.txt')):
        shutil.copy2(source, target / source.name)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if not python_license.is_file():
        raise ValueError('Cannot find the Python runtime LICENSE.txt at {}'.format(python_license))
    shutil.copy2(python_license, target / 'Python-LICENSE.txt')
    inventory = []
    for name, dist in sorted(distributions.items()):
        folder = target / '{}-{}'.format(name, dist.version)
        folder.mkdir(parents=True, exist_ok=True)
        metadata = dist.read_text('METADATA')
        if metadata:
            (folder / 'METADATA.txt').write_text(metadata, encoding='utf-8')
        for item in dist.files or ():
            parts = [part.lower() for part in item.parts]
            is_notice = 'licenses' in parts or any(
                item.name.upper().startswith(prefix) for prefix in ('LICENSE', 'COPYING', 'NOTICE'))
            if not is_notice:
                continue
            source = Path(dist.locate_file(item))
            if source.is_file():
                safe_parts = [part for part in item.parts if part not in ('..', '.')]
                destination = folder.joinpath(*safe_parts)
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
        inventory.append({'name': name, 'version': dist.version,
                          'license': dist.metadata.get('License-Expression') or dist.metadata.get('License'),
                          'project_urls': dist.metadata.get_all('Project-URL') or []})
    (target / 'dependency-inventory.json').write_text(
        json.dumps(inventory, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    qt_version = importlib.metadata.version('PySide6')
    (target / 'README.txt').write_text(
        'multiphy is distributed under the MIT license in multiphy-MIT.txt.\n'
        'Third-party components retain their own licenses; see the per-package notices\n'
        'and dependency-inventory.json for exact versions and upstream source links.\n'
        'The inventory includes the GUI dependency closure; not every optional Qt module\n'
        'in those wheels is included in this application.\n\n'
        'Qt / PySide6 / Shiboken {} are used under the LGPLv3 option.\n'
        'LGPL-3.0.txt and GPL-3.0.txt contain the license texts. Qt DLLs remain\n'
        'separate in _internal/PySide6 and can be replaced with compatible builds.\n'
        'No application restriction forbids reverse engineering to debug modifications\n'
        'to these LGPL components. Source and rebuild instructions for multiphy:\n'
        'https://github.com/jwingert/multiphy/blob/main/multiphy/docs/desktop.md\n'
        'Corresponding Qt sources (select the exact version above):\n'
        'https://download.qt.io/official_releases/qt/\n'
        'https://code.qt.io/cgit/qt/qtbase.git/\n'
        'https://code.qt.io/cgit/qt/qtsvg.git/\n'
        'https://code.qt.io/cgit/pyside/pyside-setup.git/\n'.format(qt_version),
        encoding='utf-8')


def make_windows_assets(app_version):
    from PIL import Image
    from PySide6 import QtCore, QtGui, QtSvg
    from PyInstaller.utils.win32.versioninfo import (
        FixedFileInfo, StringFileInfo, StringStruct, StringTable,
        VarFileInfo, VarStruct, VSVersionInfo)

    application = QtGui.QGuiApplication.instance() or QtGui.QGuiApplication(
        ['multiphy-build-icon', '-platform', 'offscreen'])
    renderer = QtSvg.QSvgRenderer(str(ROOT / 'multiphy' / 'viewer' / 'logo' / 'icon_64.svg'))
    if not renderer.isValid():
        raise ValueError('Invalid application SVG icon.')
    image = QtGui.QImage(256, 256, QtGui.QImage.Format.Format_ARGB32)
    image.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(image)
    renderer.render(painter)
    painter.end()
    png = ASSETS / 'multiphy.png'
    if not image.save(str(png)):
        raise ValueError('Could not render the application icon.')
    with Image.open(png) as icon:
        icon.save(ASSETS / 'multiphy.ico', format='ICO',
                  sizes=[(size, size) for size in (16, 24, 32, 48, 64, 128, 256)])
    values = tuple(int(part) for part in app_version.split('.')) + (0,)
    info = VSVersionInfo(
        ffi=FixedFileInfo(filevers=values, prodvers=values, mask=0x3f, flags=0,
                          OS=0x40004, fileType=0x1, subtype=0, date=(0, 0)),
        kids=[StringFileInfo([StringTable('040904B0', [
            StringStruct('CompanyName', 'multiphy contributors'),
            StringStruct('FileDescription', 'multiphy'),
            StringStruct('FileVersion', app_version),
            StringStruct('ProductName', 'multiphy'),
            StringStruct('ProductVersion', app_version),
            StringStruct('LegalCopyright', 'Copyright (c) multiphy contributors'),
        ])]), VarFileInfo([VarStruct('Translation', [1033, 1200])])])
    (ASSETS / 'version.txt').write_text(str(info), encoding='utf-8')
    application.quit()


def run(command, **kwargs):
    print('+ {}'.format(subprocess.list2cmdline([str(part) for part in command])), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, check=True, **kwargs)


def source_info():
    git = shutil.which('git')
    if git is None:
        return {'source_commit': None, 'source_dirty': None}
    head = subprocess.run([git, 'rev-parse', 'HEAD'], cwd=ROOT, text=True, capture_output=True)
    status = subprocess.run([git, 'status', '--porcelain'], cwd=ROOT, text=True, capture_output=True)
    return {'source_commit': head.stdout.strip() if head.returncode == 0 else None,
            'source_dirty': bool(status.stdout.strip()) if status.returncode == 0 else None}


def build_environment():
    """Resolve DLLs from this interpreter and Windows, not unrelated PATH apps."""
    environment = dict(os.environ)
    windows = Path(environment.get('SystemRoot', r'C:\Windows'))
    search = [Path(sys.executable).resolve().parent, Path(sys.base_prefix),
              Path(sys.base_prefix) / 'DLLs', windows / 'System32', windows]
    environment['PATH'] = os.pathsep.join(str(path) for path in search)
    environment['PYTHONUTF8'] = '1'
    for name in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH',
                 'QML_IMPORT_PATH', 'QML2_IMPORT_PATH'):
        environment.pop(name, None)
    return environment


def compiler_path(explicit):
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.is_file():
            raise ValueError('Inno Setup compiler does not exist: {}'.format(candidate))
        return candidate
    command = shutil.which('iscc')
    if command:
        return Path(command)
    for base in ('ProgramFiles', 'ProgramFiles(x86)', 'LOCALAPPDATA'):
        for folder in ('Inno Setup 7', 'Inno Setup 6', 'Programs/Inno Setup 7', 'Programs/Inno Setup 6'):
            candidate = Path(os.environ.get(base, '')) / folder / 'ISCC.exe'
            if candidate.is_file():
                return candidate
    raise ValueError('Install Inno Setup or pass --iscc C:/path/to/ISCC.exe.')


def write_zip(source, destination):
    """Stable ordering/timestamps; the compiler and signing can still change bytes."""
    with zipfile.ZipFile(destination, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in sorted(source.rglob('*')):
            if file.is_file():
                relative = file.relative_to(source.parent).as_posix()
                info = zipfile.ZipInfo(relative, date_time=(2020, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with file.open('rb') as incoming, archive.open(info, 'w', force_zip64=True) as outgoing:
                    shutil.copyfileobj(incoming, outgoing)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--installer', action='store_true', help='also build the per-user Inno Setup installer')
    parser.add_argument('--iscc', type=Path, help='explicit path to the Inno Setup compiler')
    parser.add_argument('--expected-version', help='fail if the source version differs (for release tags)')
    args = parser.parse_args()
    app_version = version()
    if args.expected_version and args.expected_version != app_version:
        parser.error('Expected version {}, source declares {}.'.format(args.expected_version, app_version))
    check_environment()
    compiler = compiler_path(args.iscc) if args.installer else None
    if ASSETS.exists():
        # Only clear this script's generated folder, never a redirected path or
        # an arbitrary --output location that could contain project data.
        expected_assets = ROOT.resolve() / 'build' / 'desktop' / 'assets'
        if ASSETS.resolve() != expected_assets:
            raise ValueError('Refusing to clear redirected build assets: {}'.format(ASSETS))
        shutil.rmtree(ASSETS)
    ASSETS.mkdir(parents=True, exist_ok=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    distributions = dependency_distributions()
    collect_notices(distributions)
    make_windows_assets(app_version)
    info = {'application_version': app_version, 'python_version': platform.python_version(),
            'platform': 'windows-x64', **source_info(),
            'packages': {name: dist.version for name, dist in sorted(distributions.items())}}
    (ASSETS / 'build-info.json').write_text(json.dumps(info, indent=2) + '\n', encoding='utf-8')
    run([sys.executable, '-B', '-m', 'PyInstaller', '--noconfirm', '--clean',
         '--distpath', OUTPUT, '--workpath', BUILD / 'pyinstaller', HERE / 'multiphy.spec'],
        env=build_environment())
    bundle = OUTPUT / 'multiphy'
    shutil.copy2(ROOT / 'LICENSE', bundle / 'LICENSE.txt')
    shutil.copy2(ROOT / 'multiphy' / 'docs' / 'desktop.md', bundle / 'README.md')
    shutil.copy2(ASSETS / 'build-info.json', bundle / 'build-info.json')
    run([sys.executable, '-B', HERE / 'smoke.py', '--dist', bundle])
    stem = 'multiphy-{}-windows-x64'.format(app_version)
    archive = OUTPUT / (stem + '-portable.zip')
    write_zip(bundle, archive)
    artifacts = [archive]
    if compiler:
        run([compiler, '/DAppVersion=' + app_version, '/DSourceDir=' + str(bundle),
             '/DOutputDir=' + str(OUTPUT), '/DIconFile=' + str(ASSETS / 'multiphy.ico'),
             HERE / 'multiphy.iss'])
        installer = OUTPUT / (stem + '-setup.exe')
        if not installer.is_file():
            raise ValueError('Inno Setup did not produce {}'.format(installer))
        artifacts.append(installer)
    sums = []
    for artifact in artifacts:
        digest = hashlib.sha256()
        with artifact.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        sums.append('{}  {}'.format(digest.hexdigest(), artifact.name))
        print('Built {}'.format(artifact), flush=True)
    (OUTPUT / 'SHA256SUMS.txt').write_text('\n'.join(sums) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, importlib.metadata.PackageNotFoundError) as exc:
        raise SystemExit(str(exc))
