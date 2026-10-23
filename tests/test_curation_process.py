"""Curation checks a writer's existence without changing the writer process."""
import ctypes
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from multiphy import curation


@pytest.mark.parametrize('wait_result, expected', [(258, True), (0, False), (0xffffffff, True)])
def test_windows_process_query_is_read_only(monkeypatch, wait_result, expected):
    kernel = SimpleNamespace(OpenProcess=Mock(return_value=42),
                             WaitForSingleObject=Mock(return_value=wait_result), CloseHandle=Mock())
    monkeypatch.setattr(ctypes, 'WinDLL', Mock(return_value=kernel), raising=False)
    assert curation._windows_pid_alive(123) is expected
    kernel.OpenProcess.assert_called_once_with(0x00100000, False, 123)
    kernel.WaitForSingleObject.assert_called_once_with(42, 0)
    kernel.CloseHandle.assert_called_once_with(42)


@pytest.mark.parametrize('error, expected', [(87, False), (5, True), (8, True)])
def test_windows_process_query_preserves_lock_on_uncertain_errors(monkeypatch, error, expected):
    kernel = SimpleNamespace(OpenProcess=Mock(return_value=None),
                             WaitForSingleObject=Mock(), CloseHandle=Mock())
    monkeypatch.setattr(ctypes, 'WinDLL', Mock(return_value=kernel), raising=False)
    monkeypatch.setattr(ctypes, 'get_last_error', lambda: error, raising=False)
    assert curation._windows_pid_alive(123) is expected
    kernel.WaitForSingleObject.assert_not_called()
    kernel.CloseHandle.assert_not_called()


def test_windows_process_ids_cannot_wrap_to_another_process():
    assert curation._windows_pid_alive(0) is False
    assert curation._windows_pid_alive(-1) is False
    assert curation._windows_pid_alive(2 ** 32) is False


@pytest.mark.parametrize('error, expected', [(None, True), (ProcessLookupError(), False),
                                            (PermissionError(), True)])
def test_posix_process_probe_keeps_existing_behavior(monkeypatch, error, expected):
    kill = Mock(side_effect=error)
    monkeypatch.setattr(curation, 'os', SimpleNamespace(name='posix', kill=kill))
    assert curation._pid_alive(123) is expected
    kill.assert_called_once_with(123, 0)


def test_process_query_leaves_live_writer_running():
    child = subprocess.Popen([sys.executable, '-c', 'import sys; sys.stdin.read()'], stdin=subprocess.PIPE)
    try:
        assert curation._pid_alive(child.pid) is True
        assert child.poll() is None
    finally:
        child.communicate(timeout=10)
    assert child.returncode == 0
    assert curation._pid_alive(child.pid) is False
