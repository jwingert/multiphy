"""Shared extraction code lives in ``extract/``; ``features`` and ``extract_benchmark`` re-export it."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from .lab_blocker import run_without_lab

REPO = Path(__file__).resolve().parents[1]

UNIT_FEATURES_NAMES = ('ISI_BINS', 'ACG_BINS', '_isi_acg', 'temporal_sample_windows',
                       '_checked_sample_spikes', 'waveform_sample_selection',
                       'average_spike_waveforms', 'temporal_split_unit_features')
RAW_PASS_NAMES = ('_voltage_stream', 'plan_snippets', 'SnippetReader', 'FilteredSnippetReader')


def test_moved_names_are_same_objects():
    from multiphy import extract_benchmark, features
    from multiphy.extract import raw_pass, unit_features

    same = {name: getattr(features, name) is getattr(unit_features, name)
            for name in UNIT_FEATURES_NAMES}
    same.update({name: getattr(extract_benchmark, name) is getattr(raw_pass, name)
                 for name in RAW_PASS_NAMES})
    assert len(same) == 12
    assert same == {name: True for name in UNIT_FEATURES_NAMES + RAW_PASS_NAMES}


def test_extract_benchmark_cli_help():
    env = dict(os.environ)
    env['PYTHONPATH'] = str(REPO)
    proc = subprocess.run([sys.executable, '-m', 'multiphy.extract_benchmark', '--help'],
                          cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, proc.stderr


def test_extract_modules_import_without_lab():
    code = (
        'import multiphy.extract.raw_pass\n'
        'import multiphy.extract.unit_features\n'
        'import multiphy.extract.pipeline\n'
        "print('ok')\n"
    )
    proc = run_without_lab(code, REPO)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.rstrip().endswith('ok')
