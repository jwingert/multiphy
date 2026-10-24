"""The repository installs: the root README's links resolve, and a wheel built from a copy of the
repository carries the package documents, fonts and logo SVGs, the ``multiphy`` console script and no
tests. The wheel is built under ``tmp_path``, never in the repository."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import sysconfig
import venv
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
LINK = re.compile(r'\]\(([^)\s]+)\)|src="([^"]+)"')
EXTERNAL = ('http://', 'https://', 'mailto:', '#')

DATA_FILES = [
    'multiphy/BENCHMARKS.md',
    'multiphy/DEVELOPMENT.md',
    'multiphy/MATCHING_INTERFACE.md',
    'multiphy/docs/adding_a_backend.md',
    'multiphy/docs/adding_a_source.md',
    'multiphy/docs/contract.md',
    'multiphy/docs/project_folder.md',
    'multiphy/docs/viewer.md',
    'multiphy/viewer/fonts/Figtree.ttf',
    'multiphy/viewer/fonts/Figtree-OFL.txt',
    'multiphy/viewer/fonts/JetBrainsMono.ttf',
    'multiphy/viewer/fonts/JetBrainsMono-OFL.txt',
    'multiphy/viewer/logo/logo.svg',
    'multiphy/viewer/logo/icon_16.svg',
    'multiphy/viewer/logo/icon_24.svg',
    'multiphy/viewer/logo/icon_32.svg',
    'multiphy/viewer/logo/icon_48.svg',
    'multiphy/viewer/logo/icon_64.svg',
]


def test_readme_links_resolve():
    text = (REPO / 'README.md').read_text(encoding='utf-8')
    links = [a or b for a, b in LINK.findall(text)]
    links = [link for link in links if not link.startswith(EXTERNAL)]
    assert links == [
        'multiphy/viewer/logo/logo.svg',
        'multiphy/docs/project_folder.md',
        'multiphy/docs/viewer.md',
        'multiphy/MATCHING_INTERFACE.md',
        'multiphy/DEVELOPMENT.md',
        'multiphy/BENCHMARKS.md',
        'multiphy/docs/project_folder.md',
    ]
    assert len(links) == 7
    missing = [link for link in links if not (REPO / link).exists()]
    assert missing == []


def test_wheel_carries_data_and_entry_point(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    ignore = shutil.ignore_patterns('__pycache__')
    shutil.copytree(REPO / 'multiphy', src / 'multiphy', ignore=ignore)
    shutil.copytree(REPO / 'tests', src / 'tests', ignore=ignore)
    shutil.copy2(REPO / 'pyproject.toml', src / 'pyproject.toml')
    shutil.copy2(REPO / 'README.md', src / 'README.md')
    out = tmp_path / 'out'
    # pip's build tracker and wheel staging directories come from tempfile.mkdtemp(), so the
    # subprocess gets a TMPDIR under tmp_path and no inherited build-tracker path.
    tmp = tmp_path / 'tmp'
    tmp.mkdir()
    env = {key: value for key, value in os.environ.items()
           if key not in {'PIP_BUILD_TRACKER', 'PYTHONPATH', 'PYTHONHOME'}}
    env.update(TMPDIR=str(tmp), TMP=str(tmp), TEMP=str(tmp), PYTHONIOENCODING='utf-8')
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-index',
                    '--no-cache-dir', '-w', str(out), str(src)], check=True, capture_output=True, encoding='utf-8',
                   env=env)

    wheels = sorted(out.glob('*.whl'))
    assert len(wheels) == 1
    with zipfile.ZipFile(wheels[0]) as wheel:
        names = wheel.namelist()
        entry_points = [name for name in names if name.endswith('.dist-info/entry_points.txt')]
        assert len(entry_points) == 1
        entry_text = wheel.read(entry_points[0]).decode('utf-8')

    missing = [name for name in DATA_FILES if name not in names]
    assert missing == []
    assert 'multiphy = multiphy.launch.__main__:main' in entry_text.splitlines()
    assert [name for name in names if name.startswith('tests/')] == []
    assert 'multiphy/__init__.py' in names
    assert 'multiphy/launch/__main__.py' in names

    # Install the actual wheel into a fresh environment. Reuse only the test environment's
    # dependencies, not its editable-install .pth hooks: adding a site-packages directory via
    # a .pth file does not execute the .pth files in that directory. The wheel's site-packages
    # precedes those dependency directories, and the checks below run outside the source tree.
    # Include characters outside Windows' legacy cp1252 code page in paths reported by the child.
    installed = tmp_path / 'installed-é-µ-路径'
    venv.EnvBuilder(with_pip=False).create(installed)
    scripts = installed / ('Scripts' if os.name == 'nt' else 'bin')
    python = scripts / ('python.exe' if os.name == 'nt' else 'python')
    subprocess.run([sys.executable, '-m', 'pip', '--python', str(python), 'install', '--no-deps',
                    '--no-index', '--no-cache-dir', str(wheels[0])], check=True, capture_output=True,
                   encoding='utf-8', cwd=tmp_path, env=env)
    # -I ignores PYTHONIOENCODING, so isolated probes explicitly enable UTF-8 mode.
    site = subprocess.run([str(python), '-I', '-X', 'utf8', '-c',
                           'import sysconfig; print(sysconfig.get_path("purelib"))'],
                          check=True, capture_output=True, encoding='utf-8', cwd=tmp_path, env=env)
    dependencies = dict.fromkeys(sysconfig.get_path(key) for key in ('purelib', 'platlib'))
    # ASCII escapes avoid relying on patch-specific .pth decoding and keep Unicode dependency
    # paths valid even when the installed CLI uses a legacy locale.
    (Path(site.stdout.strip()) / 'test-dependencies.pth').write_text(
        f'import sys; sys.path.extend({ascii(list(dependencies))})\n', encoding='ascii')
    smoke = '''
import sys
from pathlib import Path
import multiphy
from multiphy.launch import __main__
from importlib.resources import files
for name, module in list(sys.modules.items()):
    if name == 'multiphy' or name.startswith('multiphy.'):
        assert Path(module.__file__).resolve().is_relative_to(Path(sys.prefix).resolve()), name
assert files('multiphy').joinpath('docs/contract.md').is_file()
assert files('multiphy').joinpath('viewer/fonts/Figtree.ttf').is_file()
assert files('multiphy').joinpath('viewer/logo/logo.svg').is_file()
'''
    subprocess.run([str(python), '-I', '-X', 'utf8', '-c', smoke], check=True, capture_output=True, encoding='utf-8',
                   cwd=tmp_path, env=env)
    command = scripts / ('multiphy.exe' if os.name == 'nt' else 'multiphy')
    help_result = subprocess.run([str(command), '--root', str(tmp_path / 'lab'), '--help'],
                                 check=True, capture_output=True, encoding='utf-8', cwd=tmp_path, env=env)
    assert 'usage:' in help_result.stdout
    assert 'add days · creates a new project' in help_result.stdout
