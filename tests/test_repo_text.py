"""No tracked file names the old package or test paths."""
from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import List, Tuple

REPO = Path(__file__).resolve().parents[1]


def tracked_text() -> List[Tuple[str, str]]:
    """Every tracked or untracked-unignored text file as ``(path, text)``, sorted by path.

    Nested repositories (entries ending ``/``) and the ``.svg`` and ``.ttf`` files are left out.
    """
    listing = subprocess.run(['git', 'ls-files', '--cached', '--others', '--exclude-standard'],
                             cwd=REPO, check=True, capture_output=True, text=True).stdout
    out = []
    for entry in sorted(set(listing.splitlines())):
        if not entry or entry.endswith(('/', '.svg', '.ttf')):
            continue
        out.append((entry, (REPO / entry).read_text(encoding='utf-8')))
    return out


def test_no_tracked_file_names_old_paths():
    old_package = 'nems_lbhb' + '.' + 'unit_match'
    old_folder = 'nems_lbhb' + '/' + 'unit_match'
    old_tests = 'tests' + '.' + 'unit_match'
    old_tests_folder = 'tests' + '/' + 'unit_match'
    old_prefix = 'test_' + 'unit_match' + '_'
    tail = '(?![_A-Za-z0-9])'
    patterns = [re.compile(p) for p in (
        re.escape(old_package) + tail,
        re.escape(old_folder) + tail,
        re.escape(old_tests) + tail,
        re.escape(old_tests_folder) + tail,
        re.escape(old_prefix),
    )]
    files = tracked_text()
    assert len(files) >= 240
    hits = [f'{path}:{number}'
            for path, text in files
            for number, line in enumerate(text.splitlines(), 1)
            if any(p.search(line) for p in patterns)]
    assert hits == []


def test_no_tracked_file_holds_a_tag():
    tag = 'AGENT ' + 'EDIT'
    files = tracked_text()
    assert len(files) >= 240
    hits = [f'{path}:{number}'
            for path, text in files
            for number, line in enumerate(text.splitlines(), 1)
            if tag in line]
    assert hits == []


def _imported_modules(path: Path) -> List[str]:
    """Every module an import in ``path`` can bind, relative imports resolved against its package.

    ``from a import b`` yields both ``a`` and ``a.b``, since ``b`` may be a submodule.
    """
    import ast
    package = list(path.relative_to(REPO).parts[:-1])
    out = []
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'), filename=str(path))):
        if isinstance(node, ast.Import):
            out.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package[:len(package) - (node.level - 1)]
                module = '.'.join(base + ([node.module] if node.module else []))
            else:
                module = node.module or ''
            out.append(module)
            out.extend(f'{module}.{alias.name}' for alias in node.names if alias.name != '*')
    return out


def test_no_module_imports_a_shim():
    import importlib.util
    names = ('paths', 'evidence', 'external' + '_backends', 'dant' + '_settings', 'dant' + '_compat',
             'names')
    shims = ['multiphy' + '.' + name for name in names]
    assert [importlib.util.find_spec(shim) for shim in shims] == [None] * 6

    files = sorted(list((REPO / 'multiphy').rglob('*.py')) + list((REPO / 'tests').rglob('*.py')))
    assert len(files) >= 220
    imports = [f'{path.relative_to(REPO)}: {module}'
               for path in files
               for module in _imported_modules(path)
               if any(module == shim or module.startswith(shim + '.') for shim in shims)]
    assert imports == []

    dotted = re.compile('multiphy' + r'\.(' + '|'.join(names) + r')\b')
    bare = re.compile(r'\b(' + '|'.join(names[2:5]) + r')\b')
    hits = [f'{path}:{number}'
            for path, text in tracked_text()
            for number, line in enumerate(text.splitlines(), 1)
            if dotted.search(line) or bare.search(line)]
    assert hits == []


def test_benchmark_pairs_imports_first_in_a_fresh_interpreter():
    import sys
    done = subprocess.run([sys.executable, '-B', '-c', 'import multiphy.benchmark_pairs'],
                          cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_no_lab_words():
    skipped = {'tests/conftest.py', 'tests/lab_blocker.py', 'tests/test_no_lab_imports.py',
               'tests/test_repo_text.py'}
    removed = ('nems_' + 'unit_match' + '_operating_point', 'nems_' + 'unit_match' + '_result')
    pattern = re.compile('nems' + '_lbhb|nems' + '0|nems' + '_db|unit' + '_match|/' + 'auto/|'
                         + '(?i:' + 'lb' + 'hb|ba' + 'phy|cell' + 'db)')
    files = [(path, text) for path, text in tracked_text() if path not in skipped]
    assert len(files) >= 240
    hits = []
    for path, text in files:
        for token in removed:
            text = text.replace(token, '')
        hits.extend(f'{path}:{number}: {match.group(0)}'
                    for number, line in enumerate(text.splitlines(), 1)
                    for match in pattern.finditer(line))
    assert hits == []


def test_no_runclass():
    word = 'run' + 'class'
    allowed = {('multiphy/benchmark_collection.py', "metadata.pop('" + word + "', None)"): 1,
               ('multiphy/operating_points.py',
                "return {k: v for k, v in identity.items() if k != '" + word + "'}"): 1}
    package = sorted(path.relative_to(REPO).as_posix() for path in (REPO / 'multiphy').rglob('*.py'))
    docs = (['README.md'] + sorted(path.relative_to(REPO).as_posix()
                                   for path in (REPO / 'multiphy' / 'docs').glob('*.md'))
            + ['multiphy/MATCHING_INTERFACE.md', 'multiphy/BENCHMARKS.md', 'multiphy/DEVELOPMENT.md'])
    files = package + docs
    assert len(files) >= 120
    pattern = re.compile(word, re.IGNORECASE)
    seen = {key: 0 for key in allowed}
    hits = []
    for path in files:
        for number, line in enumerate((REPO / path).read_text(encoding='utf-8').splitlines(), 1):
            if not pattern.search(line):
                continue
            key = (path, line.strip())
            if key in allowed and seen[key] < allowed[key]:
                seen[key] += 1
                continue
            hits.append(f'{path}:{number}')
    assert hits == []
