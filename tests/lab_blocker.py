# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: run code in a fresh interpreter that cannot import lab modules (multiphy project folder Task 1) | date: 2026-09-25]
"""Run Python in a fresh interpreter that refuses to import the lab's modules.

``run_without_lab(code, cwd)`` prepends ``PRELUDE`` to ``code``: it installs a meta-path
finder that raises ``ImportError('blocked: <name>')`` for every module in ``BLOCKED`` and
their submodules, so a test can prove that a public import never reaches celldb, the
baphy loaders or legacy NEMS.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

BLOCKED = ('nems_lbhb.db', 'nems_lbhb.baphy_io', 'nems0')

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
# [AGENT EDIT END]
