"""Moved to contract/names.py; these dicts are views on NAMES kept for old imports."""
from multiphy.contract.names import NAMES
from multiphy.contract import registry as _registry
from multiphy.contract.inputs import Input as _Input
from multiphy.contract.names import backend_label as _backend_label
METHOD_NAMES = {s.id: _backend_label(s.id) for s in _registry.list_specs()
                if _Input.WAVEFORM_HALVES in s.requires}
TRACKING_NAMES = {k: NAMES['tracking'][k].label for k in ('native_intermediate', 'native_conservative', 'native_liberal', 'common_constrained')}
TRACKING_HELP = {k: NAMES['tracking'][k].help for k in TRACKING_NAMES}
DANT_PRESET_NAMES = {k: v.label for k, v in NAMES['dant_preset'].items()}
