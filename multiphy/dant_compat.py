# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: shim; DANT compatibility guard moved to backends/dant.py (Task 9) | date: 2026-09-22]
"""Deprecated import path; the DANT empty drift-reference guard lives in ``backends/dant.py``."""
from .backends.dant import _DANT_RUN_LOCK, run_native_dant  # noqa: F401
# [AGENT EDIT END]
