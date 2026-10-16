# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: public unit_match imports never reach the lab database (multiphy project folder Task 1) | date: 2026-09-25]
"""Importing multiphy loads no lab module."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

from .lab_blocker import run_without_lab

REPO = Path(__file__).resolve().parents[1]


def test_core_modules_import_without_lab():
    code = (
        'import multiphy.benchmark\n'
        'import multiphy.extract_benchmark\n'
        'import multiphy.multiday_reuse\n'
        'import multiphy.streaming_waveforms\n'
        'from multiphy.contract import registry\n'
        "registry.get('functional_likelihood')\n"  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: mechanism id (Task 3) | date: 2026-09-27]
        "print('ok')\n"
    )
    proc = run_without_lab(code, REPO)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip().endswith('ok')


def test_blocker_blocks_lab_modules():
    proc = run_without_lab('import nems_lbhb.db\n', REPO)
    assert proc.returncode == 1
    assert proc.stderr.strip().splitlines()[-1] == 'ImportError: blocked: nems_lbhb'


def test_session_blocks_lab_imports():
    with pytest.raises(ImportError) as exc:
        importlib.import_module('nems_lbhb' + '.unit_match')
    assert str(exc.value) == 'blocked: nems_lbhb'
    assert 'nems_lbhb' not in sys.modules
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the curator and the project-run viewers import and open without the lab database (Plan B Task 11) | date: 2026-09-28]
QT_PRELUDE = (
    'import os\n'
    "os.environ['QT_QPA_PLATFORM'] = 'offscreen'\n"
    'from PySide6 import QtWidgets\n'
    'app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])\n'
)


def test_gui_imports_without_lab():
    code = QT_PRELUDE + (
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the viewer App replaces the deleted gui module (viewer-split Task 9) | date: 2026-10-07]
        'import multiphy.viewer.app\n'
        # [AGENT EDIT END]
        "print('ok')\n"
    )
    proc = run_without_lab(code, REPO)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip().endswith('ok')


def test_project_path_runs_and_opens_without_lab(tmp_path):
    code = QT_PRELUDE + (
        'from pathlib import Path\n'
        'from tests.synthetic_day import extracted_project\n'
        'from multiphy.launch import commands\n'
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the runs open in viewer.app, not the deleted days and benchmark viewers (viewer-split Task 9) | date: 2026-10-07]
        'from multiphy.viewer import app as viewer_app\n'
        # [AGENT EDIT END]
        f'root = extracted_project(Path({str(tmp_path)!r})).parent\n'
        "days = commands.run(root, 'P', backend='functional_likelihood')\n"
        "day = commands.run(root, 'P', backend='functional_likelihood', days=['D1'])\n"
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the windows show their runs and units, and opening attempts no blocked import (Plan B Task 11) | date: 2026-09-28]
        'import sys\n'
        'from multiphy import backends\n'
        'attempts = []\n'
        'class _Recorder:\n'
        '    def find_spec(self, name, path=None, target=None):\n'
        "        if any(name == b or name.startswith(b + '.') for b in BLOCKED):\n"
        '            attempts.append(name)\n'
        '        return None\n'
        'sys.meta_path.insert(0, _Recorder())\n'
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: open both runs with viewer.app.open_run (viewer-split Task 9) | date: 2026-10-07]
        "days_window = viewer_app.open_run(days['run'])\n"
        "day_window = viewer_app.open_run(day['run'])\n"
        # [AGENT EDIT END]
        'days_window.show()\n'
        'day_window.show()\n'
        'app.processEvents()\n'
        'assert attempts == [], attempts\n'
        "assert days_window.match_result.run_id == backends.load_result(days['run'] / 'result.json').run_id\n"
        "assert days_window.state.sites == {f'{d}:{u}': d for d in ('D1', 'D2') for u in range(3)}\n"
        "day_result = backends.load_result(day['run'] / 'result.json')\n"
        'assert day_window.match_result.run_id == day_result.run_id != days_window.match_result.run_id\n'
        'assert set(day_window.state.units.cellid) == set(day_result.units.cellid)\n'
        "assert sorted(day_window.state.sites.values()).count('recording0000_part000') == 3\n"
        "assert sorted(set(day_window.state.sites.values())) == ['recording0000_part000', 'recording0000_part001']\n"
        'assert len(day_window.state.sites) == 6\n'
        # [AGENT EDIT END]
        'days_window.close()\n'
        'day_window.close()\n'
        "print('ok')\n"
    )
    proc = run_without_lab(code, REPO)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip().endswith('ok')
# [AGENT EDIT END]
