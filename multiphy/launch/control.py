"""Private, opt-in cancellation channel for GUI children without a usable console.

Windows' ``os.kill(pid, SIGINT)`` terminates a process rather than raising
``KeyboardInterrupt``. The GUI instead writes one byte to its child's stdin;
the child raises SIGINT locally, which also wakes Python's Windows sleep event.
Ordinary CLI stdin is untouched. This protocol is not an interactive input API.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
import signal
import subprocess
import sys
import threading

STDIN_CONTROL_ENV = 'MULTIPHY_LAUNCH_CONTROL_STDIN'
INTERRUPT = b'\x03'
_stdin_control_claimed = False


def start_control_reader() -> None:
    """Consume the opt-in flag and handle at most one cancellation, until EOF.

    Read a duplicate raw descriptor, not ``sys.stdin.buffer``: a blocked daemon
    must not hold a buffered-stream lock during interpreter shutdown. Consuming
    the flag prevents commands' subprocesses from inheriting this protocol.
    Call inside the CLI's KeyboardInterrupt handler's protected region.
    """
    global _stdin_control_claimed
    if os.environ.pop(STDIN_CONTROL_ENV, None) != '1':
        return
    fd = os.dup(sys.stdin.fileno())
    # The pipe stays private even after EOF or cancellation stops its reader.
    _stdin_control_claimed = True

    def receive() -> None:
        try:
            while True:
                byte = os.read(fd, 1)
                if not byte:
                    return
                if byte == INTERRUPT:
                    # interrupt_main only schedules a handler; on Windows it
                    # does not wake a blocking time.sleep. raise_signal does.
                    if sys.platform == 'win32':
                        signal.raise_signal(signal.SIGINT)
                    else:
                        os.kill(os.getpid(), signal.SIGINT)
                    return  # a repeated cancel must not interrupt cleanup
        finally:
            os.close(fd)

    threading.Thread(target=receive, name='multiphy-cancel', daemon=True).start()


@contextmanager
def _defer_sigint_during_spawn():
    """Delay a Python SIGINT handler until the caller owns the newly created child.

    A signal between the OS spawn and Popen returning would otherwise lose the
    child reference needed for cleanup. Restore and invoke the original handler
    on exit, including after failed construction; a pending cancellation takes
    precedence over a simultaneous construction error. Worker-thread callers do
    not own signal handlers, and ignored/default OS dispositions stay unchanged.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if not callable(previous):
        yield
        return
    pending = []
    signal.signal(signal.SIGINT, lambda signum, frame: pending.append((signum, frame)))
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
        for signum, frame in pending:
            previous(signum, frame)


def run_captured(command: list, *, cwd, env: dict) -> subprocess.CompletedProcess:
    """Capture a backend's output without an uninterruptible Windows pipe join.

    On cancellation, kill and reap this direct child before propagating the
    KeyboardInterrupt to the launcher. This does not promise graceful backend
    cleanup or terminate arbitrary grandchildren created by a backend/plugin.
    Both pipe draining and reaping during exceptional cleanup have deadlines.
    An opted-in GUI's stdin is a private cancellation channel, not backend input;
    ordinary CLI backends keep inheriting stdin.
    """
    process = None
    try:
        with _defer_sigint_during_spawn():
            # An inherited pipe with a pending reader can hang Windows Python
            # startup (CPython #78961). Never pass our private channel to a backend.
            stdin = subprocess.DEVNULL if _stdin_control_claimed else None
            process = subprocess.Popen(command, cwd=cwd, env=dict(env, PYTHONIOENCODING='utf-8'),
                                       stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       encoding='utf-8')
        while True:
            try:
                out, err = process.communicate(timeout=0.1)
                return subprocess.CompletedProcess(command, process.returncode, out, err)
            except subprocess.TimeoutExpired:
                # Windows' unbounded reader-thread joins do not service SIGINT
                # on supported Python versions. Short waits let the main thread
                # run its handler even when a backend is silent or sleeping.
                continue
    except BaseException:
        if process is not None:
            try:
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=2)
                process.communicate(timeout=2)
            except BaseException:
                # Keep the original error/interrupt, including if a grandchild holds
                # the capture pipes open. Do not enter Popen.__exit__, whose stream
                # closes/waits could block here after the cleanup deadline.
                pass
        raise
