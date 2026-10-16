import json
import sys
import pytest

from multiphy.contract import registry
from multiphy.contract.errors import ContractViolation
from multiphy.run import subprocess as sp


def test_command_is_argv_not_shell_and_roundtrips(tmp_path):
    src = tmp_path / 'ds'; src.mkdir(); (src / 'manifest.json').write_text('{"kind": "temporal_benchmark"}')
    program, argv = sp.build_command(src, 'unitmatch', {'tracking': 'common_constrained', 'threshold': .8},
                                     output=tmp_path / 'out', partition='test')
    assert program == sys.executable
    assert argv[:4] == ['-u', '-m', 'multiphy.run', 'run-backend']
    assert '--tracking' in argv and argv[argv.index('--tracking') + 1] == 'common_constrained'
    assert argv[argv.index('--threshold') + 1] == '0.8' and '--partition' in argv
    assert all(isinstance(a, str) and ' ' not in a or a.startswith('{') for a in argv)


def test_unknown_option_and_bad_interpreter_rejected(tmp_path):
    src = tmp_path / 'ds'; src.mkdir(); (src / 'manifest.json').write_text('{}')
    with pytest.raises(ContractViolation, match='bogus'):
        sp.build_command(src, 'dant', {'bogus': 1}, output=tmp_path / 'o')
    with pytest.raises(ValueError, match='interpreter'):
        sp.build_command(src, 'dant', {}, output=tmp_path / 'o', interpreter=tmp_path / 'nope')


def test_missing_manifest_rejected(tmp_path):
    with pytest.raises(ValueError, match='manifest.json'):
        sp.build_command(tmp_path, 'dant', {}, output=tmp_path / 'o')


def test_symlinked_venv_interpreter_keeps_its_prefix(tmp_path):
    real = tmp_path / 'real' / 'python'; real.parent.mkdir(); real.write_text('#!/bin/sh\n'); real.chmod(0o755)
    link = tmp_path / 'venv' / 'python'; link.parent.mkdir(); link.symlink_to(real)
    src = tmp_path / 'ds'; src.mkdir(); (src / 'manifest.json').write_text('{}')
    program, _ = sp.build_command(src, 'dant', {}, output=tmp_path / 'o', interpreter=link)
    assert program == str(link.absolute())
