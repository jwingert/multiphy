"""Fixtures shared by the launcher window tests; import them by name into each test module.

``qt`` builds the offscreen ``QApplication``; ``runs`` is one project with an across-day run, a
same-day run, an initialised-only day and an unfinished run folder, under a root path holding ``' · '``;
``wait_until`` spins the event loop until a predicate holds; ``open_and_wait`` opens the selected run
and waits until the open ends; ``held_reads`` holds every launcher read at a gate until released
(``HeldReads``); ``FakeProcess`` stands in for ``CommandProcess``.
"""
from __future__ import annotations

import os
from pathlib import Path
import threading
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


def open_and_wait(window, timeout_s: float = 120.0) -> None:
    """``window.open_run()``, then spin the event loop until the open has ended (viewer shown or failed)."""
    window.open_run()
    assert wait_until(lambda: not window.opening(), timeout_s)


class HeldReads:
    """The launcher's reads held at ``gate``: ``calls`` lists ``(run_dir, on_main_thread)`` of each read,
    ``running`` counts the held reads not yet returned, and ``error``, when set, is raised by a read once
    released; ``release()`` opens the gate."""

    def __init__(self):
        self.gate = threading.Event()
        self.calls = []
        self.error = None
        self.running = 0
        self._count = threading.Lock()

    def release(self) -> None:
        self.gate.set()

    def _add(self, step: int) -> None:
        with self._count:
            self.running += step


@pytest.fixture
def held_reads(monkeypatch):
    """Replace ``multiphy.viewer.app.prepare_run`` with a read that waits at the yielded ``HeldReads``'s
    gate, then reads with the original. At teardown, while the patch still holds, it releases, waits for
    every held read to return and its completion to arrive, then closes every visible viewer left."""
    from multiphy.viewer import app as viewer_app
    original = viewer_app.prepare_run
    held = HeldReads()

    def held_prepare(run_dir):
        held.calls.append((Path(run_dir), threading.current_thread() is threading.main_thread()))
        held._add(1)
        try:
            assert held.gate.wait(60), 'the held read was never released'
            if held.error is not None:
                raise held.error
            return original(run_dir)
        finally:
            held._add(-1)

    monkeypatch.setattr(viewer_app, 'prepare_run', held_prepare)
    try:
        yield held
    finally:
        held.release()
        wait_until(lambda: held.running == 0, 120)
        QTest.qWait(100)  # queued completions arrive
        for widget in QtWidgets.QApplication.topLevelWidgets():
            if isinstance(widget, viewer_app.App) and widget.isVisible():
                widget.close()
        QtWidgets.QApplication.processEvents()


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
