"""Deprecated import path; the DANT empty drift-reference guard lives in ``backends/dant.py``."""
from .backends.dant import _DANT_RUN_LOCK, run_native_dant  # noqa: F401
