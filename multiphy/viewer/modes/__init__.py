"""The run modes by id: ``across_days`` (dated days, curated) and ``same_day`` (temporal parts, inspect-only).

Imports no Qt and no lab module.
"""
from multiphy.viewer.modes import across_days, same_day

MODES = {'across_days': across_days.ACROSS_DAYS, 'same_day': same_day.SAME_DAY}
