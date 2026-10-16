"""Refuse every ``nems_lbhb`` and ``nems0`` module, in a child and in the test session.

``run_without_lab(code, cwd)`` prepends ``PRELUDE`` to ``code``: it installs a meta-path
finder that raises ``ImportError('blocked: <name>')`` for every module in ``BLOCKED`` and
their submodules, so a test can prove that a public import never reaches celldb, the
baphy loaders or legacy NEMS. ``LabBlocker`` is the same finder for the test session
itself: ``tests/conftest.py`` puts one at the front of ``sys.meta_path``, so an unrenamed
lab import fails loudly instead of running the lab checkout's old copy.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BLOCKED = ('nems_lbhb', 'nems0')


class LabBlocker:
    """Meta-path finder that refuses every module in ``BLOCKED`` and their submodules."""

    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + '.') for b in BLOCKED):
            raise ImportError('blocked: ' + name)
        return None


PRELUDE = f'''\
import sys
BLOCKED = {BLOCKED!r}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + '.') for b in BLOCKED):
            raise ImportError('blocked: ' + name)
        return None


sys.meta_path.insert(0, _Blocker())
'''


def run_without_lab(code: str, cwd: Path) -> subprocess.CompletedProcess:
    """Run ``PRELUDE + code`` with ``sys.executable`` in ``cwd``, ``cwd`` on PYTHONPATH."""
    env = dict(os.environ)
    env['PYTHONPATH'] = str(cwd)
    return subprocess.run([sys.executable, '-c', PRELUDE + code], cwd=cwd, env=env,
                          capture_output=True, text=True, timeout=600)
