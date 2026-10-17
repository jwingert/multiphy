"""Each legacy benchmark CLI takes ``--root DIR`` and hands it to the function it calls."""
from __future__ import annotations

import importlib

import pytest

ROOT = '<root>'  # stands for str(tmp_path) in each command line below

CASES = [
    ('benchmark_collection', 'create_collection',
     ['--input', 'a', '--output', 'o', '--dataset-id', 'd', '--root', ROOT]),
    ('benchmark_workflow', 'run_workflow',
     ['--training', 't', '--calibration', 'c', '--test', 'x', '--missing-test', 'm', '--output', 'o',
      '--source-checkout', 's', '--root', ROOT]),
    ('comparison_report', 'write_comparison', ['--run', 'r', '--root', ROOT]),
    ('dependency_setup', 'setup_environment',
     ['--unitmatch-source', 'u', '--dant-source', 'd', '--overlay', 'o', '--root', ROOT]),
    ('extract_benchmark', 'extract_manifest', ['--manifest', 'm', '--root', ROOT]),
    ('operating_points', 'run_select',
     ['--root', ROOT, 'select', '--input', 'i', '--scores', 's', '--target-fdr', '0.1']),
    ('operating_points', 'run_apply',
     ['--root', ROOT, 'apply', '--input', 'i', '--scores', 's', '--selection', 'sel']),
    ('runners', 'run_backend_dataset', ['--root', ROOT, 'run-backend', '--input', 'i', '--backend', 'unitmatch']),
    ('runners', 'prepare_deep_dataset', ['--root', ROOT, 'prepare-deep', '--input', 'i', '--source-checkout', 's']),
    ('runners', 'train_prepared', ['--root', ROOT, 'train-deep', '--prepared', 'p']),
]


@pytest.mark.parametrize('module_name, function, argv', CASES, ids=[f'{m}.{f}' for m, f, _ in CASES])
def test_legacy_clis_pass_root(module_name, function, argv, tmp_path, monkeypatch):
    module = importlib.import_module(f'multiphy.{module_name}')
    calls = []

    def spy(*_args, root=None, **_kwargs):
        calls.append(root)
        return {'checkpoint': 'c'} if function == 'train_prepared' else tmp_path

    monkeypatch.setattr(module, function, spy)
    module.main([str(tmp_path) if arg == ROOT else arg for arg in argv])
    assert calls == [str(tmp_path)]
