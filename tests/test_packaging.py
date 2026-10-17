"""The repository installs: the root README's links resolve, and a wheel built from a copy of the
repository carries the package documents, fonts and logo SVGs, the ``multiphy`` console script and no
tests. The wheel is built under ``tmp_path``, never in the repository."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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
    env = {key: value for key, value in os.environ.items() if key != 'PIP_BUILD_TRACKER'}
    env['TMPDIR'] = str(tmp)
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '--no-build-isolation', '--no-index',
                    '--no-cache-dir', '-w', str(out), str(src)], check=True, capture_output=True, text=True,
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
