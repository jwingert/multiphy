# [AGENT EDIT START | agent: codex | user: wingertj | reason: reject altered dependencies and unverifiable durable interpreter defaults | date: 2026-09-18]
import base64
import hashlib
import json
from pathlib import Path

from multiphy.dependency_setup import DEFAULT_DIRECTORY, _overlay_files, verified_interpreter
import pytest


def test_overlay_verifies_wheel_hash_and_rejects_extra_runtime_code(tmp_path):
    package = tmp_path / 'example'
    package.mkdir()
    source = package / '__init__.py'
    source.write_text('x = 1\n')
    metadata = tmp_path / 'example-1.dist-info'
    metadata.mkdir()
    digest = base64.urlsafe_b64encode(hashlib.sha256(source.read_bytes()).digest()).decode().rstrip('=')
    (metadata / 'RECORD').write_text(f'example/__init__.py,sha256={digest},{source.stat().st_size}\nexample-1.dist-info/RECORD,,\n')
    assert len(_overlay_files(tmp_path, 'example', '1')) == 2
    extra = package / 'unexpected.py'
    extra.write_text('pass\n')
    with pytest.raises(ValueError, match='Unrecorded'):
        _overlay_files(tmp_path, 'example', '1')
    extra.unlink()
    source.write_text('x = 2\n')
    with pytest.raises(ValueError, match='integrity mismatch'):
        _overlay_files(tmp_path, 'example', '1')


def test_only_complete_hash_verified_environment_can_become_gui_default(tmp_path):
    folder = tmp_path / DEFAULT_DIRECTORY
    python = folder / 'venv' / 'bin' / 'python'
    python.parent.mkdir(parents=True)
    python.write_text('test fixture only')
    python.chmod(0o700)
    (folder / 'venv' / 'pyvenv.cfg').write_text('fixture')
    pth = folder / 'venv' / 'lib' / 'unitmatch-inference.pth'
    pth.parent.mkdir()
    pth.write_text(str(folder / 'sources'))
    smoke = folder / 'smoke.json'
    smoke.write_text(json.dumps(dict(model_type='SpatioTemporalCNN_V2', parameter_count=1,
        origins={name: str(folder / 'sources' / name)
                 for name in ('UnitMatchPy', 'DeepUnitMatch', 'pyDANT', 'hdbscan', 'mat73')})))
    manifest = dict(schema_version=1, status='completed', interpreter=str(python),
        sha256={str(p.relative_to(folder)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (python, folder / 'venv' / 'pyvenv.cfg', pth, smoke)})
    (folder / 'environment.json').write_text(json.dumps(manifest))
    assert verified_interpreter(root=tmp_path) == str(python)
    pth.write_text('/tmp/unverified-overlay')
    assert verified_interpreter(root=tmp_path) is None
    assert verified_interpreter(root=tmp_path / 'unconfigured') is None
# [AGENT EDIT END]
