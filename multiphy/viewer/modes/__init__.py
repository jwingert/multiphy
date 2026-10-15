# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run modes by id (viewer-split Task 3) | date: 2026-10-06]
"""The run modes by id: ``across_days`` (dated days, curated) and ``same_day`` (temporal parts, inspect-only).

Imports no Qt and no lab module.
"""
from nems_lbhb.unit_match.viewer.modes import across_days, same_day

MODES = {'across_days': across_days.ACROSS_DAYS, 'same_day': same_day.SAME_DAY}
# [AGENT EDIT END]
