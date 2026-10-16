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
