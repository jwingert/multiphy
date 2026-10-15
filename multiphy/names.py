# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: rewritten as a shim over contract/names.py NAMES table | date: 2026-09-22]
"""Moved to contract/names.py; these dicts are views on NAMES kept for old imports."""
from nems_lbhb.unit_match.contract.names import NAMES
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: the published waveform methods come from the specs, not an id tuple (Task 12 grep gate) | date: 2026-09-22]
from nems_lbhb.unit_match.contract import registry as _registry
from nems_lbhb.unit_match.contract.inputs import Input as _Input
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: a backend with no NAMES entry is named by its id (whole-branch review fix 3) | date: 2026-09-23]
from nems_lbhb.unit_match.contract.names import backend_label as _backend_label
METHOD_NAMES = {s.id: _backend_label(s.id) for s in _registry.list_specs()
                if _Input.WAVEFORM_HALVES in s.requires}
# [AGENT EDIT END]
# [AGENT EDIT END]
TRACKING_NAMES = {k: NAMES['tracking'][k].label for k in ('native_intermediate', 'native_conservative', 'native_liberal', 'common_constrained')}
TRACKING_HELP = {k: NAMES['tracking'][k].help for k in TRACKING_NAMES}
DANT_PRESET_NAMES = {k: v.label for k, v in NAMES['dant_preset'].items()}
# [AGENT EDIT END]
