"""Fixtures shared by the launcher window tests; import them by name into each test module.

``qt`` builds the offscreen ``QApplication``; ``runs`` is one project with an across-day run, a
same-day run, an initialised-only day and an unfinished run folder, under a root path holding ``' · '``;
``wait_until`` spins the event loop until a predicate holds; ``FakeProcess`` stands in for
``CommandProcess``.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import pytest
from PySide6 import QtWidgets
from PySide6 import QtCore
from PySide6.QtTest import QTest

from multiphy.launch import commands
from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection

from .synthetic_day import extracted_project, write_day


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture(scope='module')
def runs(tmp_path_factory) -> dict:
    """``{'root', 'across', 'same'}``: project ``P`` with D1 and D2 extracted, D3 initialised only, a
    D1+D2 run (``across``), a D1 run (``same``) and the empty folder ``P/runs/unfinished``."""
    folder = tmp_path_factory.mktemp('runs') / 'lab · x'
    path = extracted_project(folder, days=('D1', 'D2'))
    root = path.parent
    d3 = write_day(folder / 'in', 'D3', seed=2, date='2026-01-03')
    commands.init_day(root, 'P', day='D3', sort=str(d3.sort_dir), date=d3.date, events=str(d3.events_path),
                      raw=str(d3.raw_path), uv_per_bit=d3.uv_per_bit)
    project = Project.open(root, 'P')
    across = run_selection(Selection(project, ('D1', 'D2')), 'functional_likelihood')
    same = run_selection(Selection(project, ('D1',)), 'functional_likelihood')
    (path / 'runs' / 'unfinished').mkdir()
    return {'root': root, 'across': across, 'same': same}


def wait_until(predicate, timeout_s: float) -> bool:
    """Spin the event loop in 50 ms steps until ``predicate()`` is true or ``timeout_s`` has passed;
    return ``predicate()``."""
    deadline = time.monotonic() + timeout_s
    while not predicate() and time.monotonic() < deadline:
        QTest.qWait(50)
    return predicate()


class FakeProcess(QtCore.QObject):
    """A ``CommandProcess`` that starts nothing: ``start`` records ``(program, arguments, env, cwd)`` in
    ``calls``; ``interrupt`` and ``kill`` count into ``interrupts`` and ``kills``; ``running()`` turns
    false on its own ``ended``, before any slot the window connects."""

    started = QtCore.Signal()
    stdout_line = QtCore.Signal(str)
    stderr_line = QtCore.Signal(str)
    ended = QtCore.Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.calls = []
        self.interrupts = 0
        self.kills = 0
        self._running = False
        self.ended.connect(self._on_ended)  # connected first, so it runs before the window's slot

    def _on_ended(self, _code: int) -> None:
        self._running = False

    def start(self, program: str, arguments: list, env: dict, cwd: str) -> None:
        self.calls.append((program, list(arguments), dict(env), cwd))
        self._running = True

    def interrupt(self) -> None:
        self.interrupts += 1

    def kill(self) -> None:
        self.kills += 1

    def running(self) -> bool:
        return self._running
