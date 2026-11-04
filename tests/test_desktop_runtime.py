"""Packaged process dispatch and first-launch choices, without touching user data."""
from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from multiphy import desktop, runtime
from multiphy.launch import __main__ as launch
from multiphy.launch.window import child_command
from multiphy.run import subprocess as workers

from .launch_window_fixtures import qt  # noqa: F401


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    folder = tmp_path / 'App with spaces'
    folder.mkdir()
    console = folder / ('multiphy.exe' if sys.platform == 'win32' else 'multiphy')
    gui = folder / ('multiphy-gui.exe' if sys.platform == 'win32' else 'multiphy-gui')
    console.touch()
    gui.touch()
    console.chmod(0o755)
    gui.chmod(0o755)
    internal = folder / '_internal'
    internal.mkdir()
    monkeypatch.setattr(sys, 'frozen', True, raising=False)
    monkeypatch.setattr(sys, '_MEIPASS', str(internal), raising=False)
    monkeypatch.setattr(sys, 'executable', str(gui))
    return SimpleNamespace(folder=folder, console=console, gui=gui, internal=internal)


def test_windowed_app_uses_console_sibling_for_launcher_and_backend_workers(bundle, tmp_path):
    root = tmp_path / 'data with spaces'
    program, args = child_command(root, ['status', 'P'])
    assert program == str(bundle.console)
    assert args == ['--multiphy-module', 'multiphy.launch', '--root', str(root), 'status', 'P']
    program, args = workers.build_selection_command(root / 'P', ('D1', 'D2'), 'functional_likelihood', {},
                                                   output=root / 'result')
    assert program == str(bundle.console)
    assert args[:4] == ['--multiphy-module', 'multiphy.run', 'run-backend', '--project']
    assert '-m' not in args
    assert runtime.execution_directory() == bundle.folder


def test_incomplete_bundle_refuses_worker_without_relaunching_gui(bundle):
    bundle.console.unlink()
    with pytest.raises(ValueError, match='complete application folder'):
        child_command(Path('data'), ['status', 'P'])


def test_explicit_external_python_stays_a_python_module_command(bundle, tmp_path):
    interpreter = tmp_path / 'venv' / 'python.exe'
    interpreter.parent.mkdir()
    interpreter.touch()
    interpreter.chmod(0o755)
    program, args = workers.build_selection_command(tmp_path / 'P', ('D1',), 'functional_likelihood', {},
                                                   output=tmp_path / 'out', interpreter=str(interpreter))
    assert program == str(interpreter)
    assert args[:4] == ['-u', '-m', 'multiphy.run', 'run-backend']


def test_external_environment_removes_only_bundle_paths(bundle, tmp_path, monkeypatch):
    user_modules = tmp_path / 'user-modules'
    monkeypatch.setenv('PYTHONPATH', os.pathsep.join([str(bundle.internal), str(user_modules)]))
    monkeypatch.setenv('PATH', os.pathsep.join([str(bundle.internal / 'Qt'), str(tmp_path / 'tools')]))
    monkeypatch.setenv('PYTHONHOME', str(bundle.internal))
    monkeypatch.setenv('QT_PLUGIN_PATH', str(bundle.internal / 'plugins'))
    monkeypatch.setenv('LD_LIBRARY_PATH', str(bundle.internal))
    monkeypatch.setenv('LD_LIBRARY_PATH_ORIG', '/original/libs')
    env = runtime.child_environment(external=True)
    assert env['PYTHONPATH'] == str(user_modules)
    assert env['PATH'] == str(tmp_path / 'tools')
    assert 'PYTHONHOME' not in env and 'QT_PLUGIN_PATH' not in env
    assert env['LD_LIBRARY_PATH'] == '/original/libs'
    assert env['PYTHONIOENCODING'] == 'utf-8' and env['PYTHONUTF8'] == '1'
    assert os.environ['PYTHONHOME'] == str(bundle.internal)


def test_source_environment_keeps_checkout_and_user_modules(monkeypatch):
    monkeypatch.setenv('PYTHONPATH', 'custom-modules')
    assert runtime.child_environment()['PYTHONPATH'] == os.pathsep.join([str(runtime.execution_directory()), 'custom-modules'])


@pytest.mark.parametrize('code', [1, 2])
def test_external_python_requires_matching_installed_multiphy(bundle, monkeypatch, code):
    monkeypatch.setattr(runtime, 'external_library_path', nullcontext)
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, code, '', '')

    monkeypatch.setattr(runtime.subprocess, 'run', run)
    with pytest.raises(ValueError, match='matching release.*uv setup'):
        runtime.verify_external_interpreter('external-python')
    assert calls[0][0][:2] == ['external-python', '-c']
    assert calls[0][1]['timeout'] == 30
    assert calls[0][1]['stdin'] == subprocess.DEVNULL


def test_external_python_accepts_matching_install(bundle, monkeypatch):
    monkeypatch.setattr(runtime, 'external_library_path', nullcontext)
    monkeypatch.setattr(runtime.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0))
    runtime.verify_external_interpreter('external-python')


def test_cli_worker_dispatch_cannot_evaluate_arbitrary_code(monkeypatch, capsys):
    monkeypatch.setattr(desktop, '_configure_streams', lambda: None)
    assert desktop.cli_main(['--multiphy-module', 'os', '-c', 'anything']) == 1
    assert 'unsupported multiphy worker' in capsys.readouterr().err


def test_cli_worker_dispatch_preserves_arguments(monkeypatch):
    monkeypatch.setattr(desktop, '_configure_streams', lambda: None)
    seen = []
    monkeypatch.setattr(launch, 'main', lambda args: seen.append(args) or 7)
    assert desktop.gui_main(['--multiphy-module', 'multiphy.launch', '--root', 'data', 'status', 'P']) == 7
    assert seen == [['--root', 'data', 'status', 'P']]


def test_packaged_cli_help_and_version_need_no_data_root(monkeypatch, capsys):
    monkeypatch.setattr(desktop, '_configure_streams', lambda: None)
    assert desktop.cli_main(['--help']) == 0
    assert 'usage: multiphy' in capsys.readouterr().out
    assert desktop.cli_main(['--version']) == 0
    assert 'multiphy ' in capsys.readouterr().out
    assert desktop.cli_main(['run', '--backend', 'unknown', '--help']) == 1
    assert 'unknown backend' in capsys.readouterr().err
    assert desktop.cli_main([]) == 0
    assert 'usage: multiphy' in capsys.readouterr().out


def test_source_cli_still_requires_explicit_root(capsys):
    assert launch.main(['--help']) == 1
    assert 'no root' in capsys.readouterr().err


def test_gui_cancel_does_not_launch_or_create_data(qt, monkeypatch, tmp_path):
    from multiphy.viewer.qt import QtWidgets
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', lambda *a: '')
    monkeypatch.setattr(desktop, 'cli_main', lambda args: pytest.fail('Cancellation launched a command'))
    assert desktop.gui_main([]) == 0
    assert not list(tmp_path.iterdir())


def test_gui_selects_and_remembers_root_then_skips_dialog(qt, monkeypatch, tmp_path):
    from multiphy.viewer.qt import QtWidgets
    selected = tmp_path / 'chosen data'
    selected.mkdir()
    dialogs, calls = [], []
    monkeypatch.setattr(QtWidgets.QFileDialog, 'getExistingDirectory', lambda *a: dialogs.append(a) or str(selected))
    monkeypatch.setattr(desktop, 'cli_main', lambda args: calls.append(args) or 0)
    assert desktop.gui_main([]) == 0
    assert desktop.gui_main([]) == 0
    assert len(dialogs) == 1
    assert calls == [['--root', str(selected), 'gui']] * 2
    assert desktop._settings().value('last_root') == str(selected)


def test_gui_explicit_root_overrides_environment_and_saved_root(qt, monkeypatch, tmp_path):
    chosen, old = tmp_path / 'explicit', tmp_path / 'remembered'
    old.mkdir()
    desktop.remember_root(old)
    monkeypatch.setenv('MULTIPHY_CACHE_ROOT', str(tmp_path / 'environment'))
    monkeypatch.setattr(desktop, '_choose_root', lambda *a: pytest.fail('Unexpected folder dialog'))
    calls = []
    monkeypatch.setattr(desktop, 'cli_main', lambda args: calls.append(args) or 0)
    assert desktop.gui_main(['--root', str(chosen), 'gui', 'P']) == 0
    assert calls == [['--root', str(chosen), 'gui', 'P']]
    assert desktop._settings().value('last_root') == str(chosen)


def test_frozen_dependency_setup_refuses_before_writing(bundle, tmp_path):
    from multiphy.dependency_setup import setup_environment
    with pytest.raises(ValueError, match='Python/uv installation'):
        setup_environment('missing', 'missing', 'missing', root=tmp_path / 'data')
    assert not (tmp_path / 'data').exists()


def test_frozen_training_refuses_before_changing_manifest(bundle, tmp_path):
    from multiphy.backends.deepunitmatch import train_deepunitmatch
    with pytest.raises(ValueError, match='Python/uv installation'):
        train_deepunitmatch(tmp_path / 'prepared')
    assert not (tmp_path / 'prepared').exists()


def test_desktop_diagnostic_renders_launcher_and_reports_assets(tmp_path):
    import json
    root = tmp_path / 'data'
    root.mkdir()
    report_path = tmp_path / 'report' / 'check.json'
    # The shipped diagnostic owns its application. Keep its theme, matplotlib
    # settings and QSettings path out of pytest's shared QApplication too.
    process = subprocess.run(
        [sys.executable, '-B', '-c', 'from multiphy.desktop import cli_main; raise SystemExit(cli_main())',
         '--self-check', str(report_path), '--root', str(root)],
        cwd=tmp_path, env=runtime.child_environment(), capture_output=True,
        text=True, encoding='utf-8', timeout=60)
    assert process.returncode == 0, process.stdout + process.stderr
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['status'] == 'passed' and report['logo'] is True
    assert Path(report['launcher_screenshot']).is_file()
    assert not list(root.iterdir())
