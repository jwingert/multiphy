# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: rewritten as a shim over contract/names.py NAMES table | date: 2026-09-22]
"""Moved to contract/names.py; these dicts are views on NAMES kept for old imports."""
from multiphy.contract.names import NAMES, display as _display
METRICS = {k: v.label for k, v in NAMES['metric'].items()}
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: registered backend names only; no pilot aliases (Task 1) | date: 2026-09-27]
METHODS = {k: v.short for k, v in NAMES['backend'].items()}
# [AGENT EDIT END]
FEATURES = {k: v.label for k, v in NAMES['feature'].items()}
TRACKING = {k: v.short for k, v in NAMES['tracking'].items()}
DEFINITIONS = {k: v.label for k, v in NAMES['definitions'].items()}
def display(key, namespace):
    return namespace[key]
def count_range(values, *, decimals=0):
    """Display a constant once, otherwise explicitly label a min–max range."""
    values = list(values)
    if not values:
        return 'unavailable'
    lo, hi = min(values), max(values)
    fmt = lambda value: f'{value:.{decimals}f}'
    return fmt(lo) if lo == hi else f'{fmt(lo)}–{fmt(hi)} (minimum–maximum)'
# [AGENT EDIT END]
