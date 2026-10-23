import base64
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

from multiphy import dependency_setup
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
    executable = ('Scripts', 'python.exe') if sys.platform == 'win32' else ('bin', 'python')
    python = folder / 'venv' / Path(*executable)
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
        sha256={p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in (python, folder / 'venv' / 'pyvenv.cfg', pth, smoke)})
    (folder / 'environment.json').write_text(json.dumps(manifest))
    assert verified_interpreter(root=tmp_path) == str(python)
    pth.write_text('/tmp/unverified-overlay')
    assert verified_interpreter(root=tmp_path) is None
    assert verified_interpreter(root=tmp_path / 'unconfigured') is None


def test_default_environment_is_v3():
    assert dependency_setup.DEFAULT_DIRECTORY == 'dependencies/inference-v3'


def test_pth_lines_name_this_repository(tmp_path):
    repository = Path(__file__).resolve().parents[1]
    assert dependency_setup._pth_lines(tmp_path) == [tmp_path / 'sources', tmp_path / 'overlay', repository]
    assert (repository / 'multiphy' / '__init__.py').is_file()


def test_pth_preserves_only_trusted_unicode_paths_under_legacy_locale_read(tmp_path):
    output = tmp_path / '路径-Δ'
    expected = list(map(str, dependency_setup._pth_lines(output)))
    content = dependency_setup._pth_text(output).encode('utf-8')
    assert content.isascii()
    # Model legacy locale-based .pth decoding without changing the host's site or sys.path.
    fake_sys = SimpleNamespace(path=['existing'])

    def import_sys(name, *args):
        assert name == 'sys'
        return fake_sys

    namespace = {'__builtins__': {'__import__': import_sys}}
    exec(content.decode('cp1252'), namespace)
    assert fake_sys.path == ['existing', *expected]
    exec(content.decode('cp1252'), namespace)
    assert fake_sys.path == ['existing', *expected]  # do not duplicate existing entries


@pytest.mark.parametrize(('platform', 'parts'), [
    ('win32', ('Scripts', 'python.exe')),
    ('linux', ('bin', 'python')),
    ('darwin', ('bin', 'python')),
])
def test_venv_interpreter_uses_platform_layout(tmp_path, monkeypatch, platform, parts):
    monkeypatch.setattr(dependency_setup, 'sys', SimpleNamespace(platform=platform))
    assert dependency_setup._venv_interpreter(tmp_path / 'venv') == tmp_path / 'venv' / Path(*parts)


def test_windows_manifest_requires_scripts_executable_and_still_checks_hashes(tmp_path, monkeypatch):
    """A Windows-layout fixture tests verification, not optional upstream imports."""
    monkeypatch.setattr(dependency_setup, 'sys', SimpleNamespace(platform='win32'))
    folder = tmp_path / DEFAULT_DIRECTORY
    files = ('venv/Scripts/python.exe', 'venv/pyvenv.cfg',
             'venv/Lib/site-packages/unitmatch-inference.pth', 'smoke.json')
    for relative in files:
        path = folder / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('fixture', encoding='utf-8')
    interpreter = folder / files[0]
    interpreter.chmod(0o700)
    (folder / 'smoke.json').write_text(json.dumps(dict(
        model_type='SpatioTemporalCNN_V2', parameter_count=1,
        origins={name: str(folder / 'sources' / name)
                 for name in ('UnitMatchPy', 'DeepUnitMatch', 'pyDANT', 'hdbscan', 'mat73')})), encoding='utf-8')
    manifest = dict(schema_version=1, status='completed', interpreter=str(interpreter),
                    sha256={relative: hashlib.sha256((folder / relative).read_bytes()).hexdigest()
                            for relative in files})
    manifest_path = folder / 'environment.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    assert verified_interpreter(root=tmp_path) == str(interpreter)

    # A POSIX interpreter entry cannot stand in for the selected Windows executable.
    digest = manifest['sha256'].pop(files[0])
    manifest['sha256']['venv/bin/python'] = digest
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    assert verified_interpreter(root=tmp_path) is None
    del manifest['sha256']['venv/bin/python']
    manifest['sha256'][files[0]] = digest
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    interpreter.write_text('changed', encoding='utf-8')
    assert verified_interpreter(root=tmp_path) is None


def test_setup_records_native_venv_and_portable_hashes(tmp_path, monkeypatch):
    """Exercise a real native venv, with upstream staging and model smoke stubbed."""
    source = tmp_path / 'stub.py'
    source.write_text('# test fixture only\n', encoding='utf-8')
    monkeypatch.setattr(dependency_setup, '_source_files',
                        lambda *args: [(source, Path('multiphy_dependency_setup_fixture') / '__init__.py')])
    monkeypatch.setattr(dependency_setup, '_overlay_files', lambda *args: [])
    output = tmp_path / 'inference-Δ'
    smoke = dict(model_type='SpatioTemporalCNN_V2', parameter_count=1,
                 origins={name: str(output / 'sources' / name)
                          for name in ('UnitMatchPy', 'DeepUnitMatch', 'pyDANT', 'hdbscan', 'mat73')})
    monkeypatch.setattr(dependency_setup, '_SMOKE',
                        'import pathlib, sys\n'
                        'import multiphy_dependency_setup_fixture as staged\n'
                        'assert pathlib.Path(staged.__file__).is_relative_to(pathlib.Path(sys.argv[1]).parent)\n'
                        f'pathlib.Path(sys.argv[1]).write_text({json.dumps(smoke)!r}, encoding="utf-8")\n'
                        'print("staged Δ")\n')
    assert dependency_setup.setup_environment(tmp_path / 'unitmatch', tmp_path / 'dant',
                                              tmp_path / 'overlay', output_dir=output, root=tmp_path) == output
    executable = ('Scripts', 'python.exe') if sys.platform == 'win32' else ('bin', 'python')
    interpreter = output / 'venv' / Path(*executable)
    manifest = json.loads((output / 'environment.json').read_text(encoding='utf-8'))
    assert manifest['interpreter'] == str(interpreter)
    assert verified_interpreter(root=tmp_path, directory=output) == str(interpreter)
    assert interpreter.is_file() and not interpreter.is_symlink()
    assert {'venv/' + '/'.join(executable), 'venv/pyvenv.cfg', 'smoke.json'} <= manifest['sha256'].keys()
    assert all('\\' not in key and not Path(key).is_absolute() for key in manifest['sha256'])
    (pth_key,) = [key for key in manifest['sha256'] if key.endswith('/unitmatch-inference.pth')]
    assert (output / pth_key).read_text(encoding='utf-8') == dependency_setup._pth_text(output)
    assert 'dependencies — methods notes' in (output / 'METHODS.md').read_text(encoding='utf-8')
    assert (output / 'setup.log').read_text(encoding='utf-8') == 'staged Δ\n'

    # Hash-valid entries are still rejected when they escape the durable snapshot.
    manifest['sha256']['../stub.py'] = hashlib.sha256(source.read_bytes()).hexdigest()
    (output / 'environment.json').write_text(json.dumps(manifest), encoding='utf-8')
    assert verified_interpreter(root=tmp_path, directory=output) is None
