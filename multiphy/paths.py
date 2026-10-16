# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: paths.py is now a shim over run/config.py | date: 2026-09-22]
"""Deprecated import path; cache locations now live in ``run/config.py``."""
from multiphy.run.config import *  # noqa
from multiphy.run.config import CACHE_ROOT  # noqa
# [AGENT EDIT END]
