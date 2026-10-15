# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Qt-free display texts and colours, from gui.py:52-59, 111-128, 131-296, 1354-1372 (EvidencePanel.FIELDS) and benchmark_viewer.py:175-181 (_declared_conditions) (viewer-split Task 2) | date: 2026-10-06]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the module docstring names the theme's colour tokens (launcher-gui Task 8) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the module docstring names graphite, uses_unit_palette and unit colours from unit_palettes except on Light (graphite-palettes Task 2) | date: 2026-10-11]
"""Qt-free display texts and colours the viewer panels share: score labels, the backend header and its hover
text, gate outcomes, unit colours and the scalar evidence fields. Imports no Qt and no lab module.

The theme's colour tokens live here too: ``PALETTES`` holds one set per name in ``THEMES`` (``graphite``, the
default, then ``dark`` and ``light``), and panels read the active theme's tokens with ``colors()`` when they
draw, so a theme switch recolours them on their next draw. Unit colours come from the Unit colours palette of
``unit_palettes`` on every theme but Light, which keeps its own ramp; ``uses_unit_palette`` says which."""
# [AGENT EDIT END]
# [AGENT EDIT END]
import ast
from typing import Optional

import numpy as np
import pandas as pd

from nems_lbhb.unit_match.contract import names, registry
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the header tags group a run's saved reasons as the launch form does (event-optional Task 12) | date: 2026-10-10]
from nems_lbhb.unit_match.contract.inputs import Input, evidence_tags
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: unit colours come from the active Unit colours palette (graphite-palettes Task 2) | date: 2026-10-11]
from nems_lbhb.unit_match.viewer import unit_palettes
# [AGENT EDIT END]


def backend_spec(backend_id):
    """The registered spec for a result's backend id, or None if unregistered."""
    try:
        return registry.get(backend_id).spec
    except KeyError:
        return None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each palette gains proposal_tint, the wash of the units the backend grouped with the picked one (native-same-day Task 9) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the dark and light colour tokens, the active theme name, set_theme, theme_name and colors replace the fixed colour constants (launcher-gui Task 8) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the graphite tokens come first and are the default theme; dark and light lose the day tokens, dark loses many, both gain plot_grid and plot_spine (graphite-palettes Task 2) | date: 2026-10-11]
# Each theme's colour tokens. Light's many is its own unit ramp; the other themes take the Unit colours palette.
PALETTES = {
    'graphite': {
        'ground': '#0F1114', 'panel': '#171A1F', 'raised': '#1F232A', 'border': '#262B33',
        'text': '#ECEEF1', 'muted': '#9097A3', 'accent': '#5B9DFF', 'accent_ink': '#0F1114',
        'accent_tint': '#1A2740', 'accept': '#3DD68C', 'reject': '#FF7B72', 'unsure': '#E3B341',
        'plot_fill': '#2A2F37', 'plot_line': '#ECEEF1', 'plot_second': '#7AA2F7',
        'plot_grid': '#22262D', 'plot_spine': '#3A414C', 'proposal_tint': '#18202F',
    },
    'dark': {
        'ground': '#0B1119', 'panel': '#18222F', 'raised': '#212D3D', 'border': '#2A3646',
        'text': '#E6EDF3', 'muted': '#8B98A9', 'accent': '#7DD3FC', 'accent_ink': '#0B1119',
        'accent_tint': '#173247', 'accept': '#3FB950', 'reject': '#F0883E', 'unsure': '#D29922',
        'plot_fill': '#3A4757', 'plot_line': '#E6EDF3', 'plot_second': '#6CB6FF',
        'plot_grid': '#243041', 'plot_spine': '#3A4757', 'proposal_tint': '#172A3B',
    },
    'light': {
        'ground': '#EEF1F5', 'panel': '#FFFFFF', 'raised': '#F3F5F8', 'border': '#D5DBE3',
        'text': '#18202B', 'muted': '#5F6B7A', 'accent': '#0284C7', 'accent_ink': '#FFFFFF',
        'accent_tint': '#E0F2FE', 'accept': '#1F6F4A', 'reject': '#B5451F', 'unsure': '#A08A2A',
        'many': ['#1F6F4A', '#B5451F', '#3F6FA0', '#7A5C9E', '#B08B1E',
                 '#2F8F8F', '#A33D6B', '#5D7D2F', '#8A6A4A', '#4A4A8A'],
        'plot_fill': '#CFD6DE', 'plot_line': '#222222', 'plot_second': '#3F6FA0',
        'plot_grid': '#EDF0F4', 'plot_spine': '#B8C1CC', 'proposal_tint': '#EFF8FE',
    },
}
THEMES = ('graphite', 'dark', 'light')
_theme = 'graphite'      # the active theme; the app's entry points apply the saved one
# [AGENT EDIT END]
# [AGENT EDIT END]


def set_theme(name: str) -> None:
    """Make ``name`` the active theme; a name not in ``THEMES`` raises ``ValueError``."""
    global _theme
    if name not in THEMES:
        raise ValueError(f'unknown theme {name!r}')
    _theme = name


def theme_name() -> str:
    """The active theme's name."""
    return _theme


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: only Light has many, so colors() copies it only when present; uses_unit_palette says whether unit colours come from the Unit colours palette (graphite-palettes Task 2) | date: 2026-10-11]
def colors() -> dict:
    """The active theme's colour tokens, as a new dict whose ``many``, when the theme has it, is a new list."""
    p = dict(PALETTES[_theme])
    if 'many' in p:
        p['many'] = list(p['many'])
    return p


def uses_unit_palette() -> bool:
    """Whether unit colours come from the Unit colours palette: True on graphite and dark, False on light, whose
    own ramp is its ``many`` token."""
    return 'many' not in PALETTES[_theme]
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: P_EVENT names the event-response stream; the stored column stays p_stim (launcher-gui Task 10) | date: 2026-10-07]
# The combined probability is what a verdict should rest on, but a cache built
# before the streams existed carries only the event-response one. Name what is actually
# on screen rather than passing one off as the other.
P_COMBINED, P_EVENT, P_PHYS = 'p_global', 'p_stim', 'p_phys'
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the waveform stream's posterior column, and its log-likelihood and posterior among the cached scores (event-optional Task 12) | date: 2026-10-10]
P_WAVE = 'p_wave'
# A matching run's scores: stream log-likelihoods, their mixture posteriors and
# the fingerprint posterior. A result whose score kind is `none`
# (the unassigned view, DANT) never displays them beside its measured features.
CACHED_SCORE_COLUMNS = ('s_stim', 's_phys', 's_wave', 's_global', 'p_stim', 'p_phys', 'p_wave', 'p_global',
                        'posterior')
# [AGENT EDIT END]


def shows_scores(row):
    """False for a row of a result that carries no matching score."""
    return row is None or row.get('score_kind') != 'none'


def probability(row, key=P_COMBINED):
    """One probability from a candidate row, NaN when the cache predates it."""
    try:
        v = row.get(key, np.nan)
        return np.nan if v is None else float(v)
    except (AttributeError, TypeError, ValueError):
        return np.nan          # a stale cache must never take the viewer down


def headline_probability(row):
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fallback is the event-response-only number (launcher-gui Task 10) | date: 2026-10-07]
    """(value, label). Falls back to the event-response-only number, and says so."""
    # [AGENT EDIT END]
    if 'match_score' in row:
        label = row.get('score_display_label') or (
            'P(same neuron)' if row.get('score_kind') == 'calibrated_probability'
            else 'Match score (uncalibrated)')
        return probability(row, 'match_score'), label
    v = probability(row, P_COMBINED)
    if np.isfinite(v):
        return v, 'P(same neuron)'
    v = probability(row, 'posterior')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the fallback label says events only (launcher-gui Task 10) | date: 2026-10-07]
    return v, 'P(same neuron), events only'
    # [AGENT EDIT END]


ASSIGNMENT_LABELS = {
    'native_intermediate': 'Published method’s native tracking (intermediate)',
    'native_liberal': 'Published method’s native tracking (liberal)',
    'native_conservative': 'Published method’s native tracking (conservative)',
    'native_DANT': 'DANT native clustering',
    'native_DANT_with_session_collision_rejection': 'DANT native clustering; same-session conflicts rejected',
    'mutual_margin_best_first': 'Common constrained tracking: candidate competition and contradiction checks',
    'common_constrained': 'Common constrained tracking: candidate competition and contradiction checks',
}
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the stream labels name the event response, physiology and waveform; the combined label, the stream words, the run's evidence tags and the refusal of a run without run_evidence read each group's saved provenance (event-optional Task 12) | date: 2026-10-10]
STREAM_LABELS = {'combined': 'Event response + physiology + waveform',
                 'stimulus': 'Event response only', 'physiology': 'Physiology only',
                 'waveform': 'Waveform only'}
NO_EVENT_COMBINED = 'Physiology + waveform'
# a saved ``weight_<stream>`` key's stream, in column order, and the word that names it on screen
STREAM_WORDS = {'stim': 'events', 'phys': 'physiology', 'wave': 'waveform'}
EVIDENCE_REFUSAL = 'run saved before event-optional matchers · rerun'


def _provenance_levels(provenance) -> list:
    """The result's provenance, then each ``group_runs`` entry's provenance."""
    runs = provenance.get('group_runs') or {}
    return [provenance, *(entry.get('provenance') or {} for entry in runs.values())]


def mixture_references(provenance) -> dict:
    """``{group: saved mixture references}`` of the top level and of every ``group_runs`` entry."""
    out = {}
    for level in _provenance_levels(provenance):
        out.update({str(group): refs for group, refs in (level.get('mixture_references') or {}).items()})
    return out


def group_streams(provenance, group) -> list:
    """The streams (``STREAM_WORDS`` keys, in order) whose ``weight_<stream>`` ``group``'s mixture references
    saved; ``[]`` for a group without saved references."""
    refs = mixture_references(provenance).get(str(group)) or {}
    return [s for s in STREAM_WORDS if f'weight_{s}' in refs]


def combined_label(provenance) -> str:
    """The combined stream's label from every group's saved ``weight_*`` keys: ``Physiology + waveform``
    when no group saved ``weight_stim``."""
    if any('weight_stim' in refs for refs in mixture_references(provenance).values()):
        return STREAM_LABELS['combined']
    return NO_EVENT_COMBINED


def run_evidence_tags(provenance) -> list:
    """``evidence_tags`` of every group's saved ``run_evidence`` reasons: ``[(tag, its reasons)]``."""
    return evidence_tags([reason for level in _provenance_levels(provenance)
                          for evidence in (level.get('run_evidence') or {}).values()
                          for reason in evidence['reasons']])


def _valid_evidence(evidence) -> bool:
    """A saved ``run_evidence`` entry: a dict whose ``reasons`` is a list of ``str`` with known tags."""
    reasons = evidence.get('reasons') if isinstance(evidence, dict) else None
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        return False
    try:
        evidence_tags(reasons)
    except ValueError:
        return False
    return True


def require_run_evidence(provenance) -> None:
    """Refuse a run whose ``mixture_references`` (top level or in a ``group_runs`` entry) come without a valid
    ``run_evidence`` beside them: a dict with an entry per group, each with ``reasons`` a list of ``str`` with
    known tags. Raises ``ValueError(EVIDENCE_REFUSAL)``."""
    for level in _provenance_levels(provenance):
        refs = level.get('mixture_references')
        if refs is None:
            continue
        evidence = level.get('run_evidence')
        if (not isinstance(evidence, dict) or not {str(g) for g in refs} <= {str(g) for g in evidence}
                or not all(_valid_evidence(e) for e in evidence.values())):
            raise ValueError(EVIDENCE_REFUSAL)
# [AGENT EDIT END]
# the scalar evidence rows: (pair column, label, gate id or None)
EVIDENCE_FIELDS = [
    ('depth_delta_um', 'depth difference (um)', 'prox'),
    ('d_log_spont', 'spontaneous rate, |diff log|', 'rate'),
    ('isi_cosine', 'interval histogram (cosine)', 'isi'),
    ('d_spike_width_ms', 'spike width difference (ms)', 'sw'),
    ('acg_cosine', 'autocorrelogram (cosine)', None),
    ('fingerprint_r', 'fingerprint correlation', None),
    ('tuning_r', 'tone tuning correlation', None),
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one footprint row becomes the peak-channel waveform and footprint rows and the all-channel footprint row (event-optional Task 1) | date: 2026-10-09]
    ('waveform_r_local', 'waveform correlation, peak channels', None),
    ('footprint_r_local', 'footprint correlation, peak channels', None),
    ('footprint_r', 'footprint correlation, all channels', None),
    # [AGENT EDIT END]
    ('d_log_rate_session', 'session rate, |diff log|', None),
    ('d_log_rate_shared', 'shared-stimulus rate, |diff log|', None),
    ('reliability_a', 'fingerprint reliability, day 1', None),
    ('reliability_b', 'fingerprint reliability, day 2', None),
    ('posterior', 'cached correlation score (uncalibrated)', None),
    ('dant_similarity', 'DANT native similarity (unbounded)', None),
    ('dant_same_cluster', 'DANT native membership (1=yes)', None),
]


def result_input_label(result):
    """Describe scoring inputs, not every supplementary measurement displayed."""
    provenance = result.provenance
    spec = backend_spec(result.backend_id)
    requires = spec.requires if spec is not None else frozenset()
    if Input.WAVEFORM_HALVES in requires and Input.CHECKPOINT in requires:
        return 'Learned waveform features'
    if (Input.WAVEFORM_HALVES in requires and Input.SPIKE_TIMES not in requires
            and Input.CHECKPOINT not in requires):
        return 'Spike waveform and spatial shape'
    if result.backend_id == 'unassigned':
        return 'Measured features only; matching has not run'
    if Input.WAVEFORM_HALVES in requires and Input.SPIKE_TIMES in requires:
        params = provenance.get('parameters')
        if params is None:
            runs = (provenance.get('group_runs') or provenance.get('recording_group_runs')
                    or provenance.get('well_runs') or {})
            records = [r.get('provenance', {}).get('parameters') for r in runs.values()]
            params = records[0] if records and all(p == records[0] for p in records) else None
        if not isinstance(params, dict):
            return 'DANT native inputs; see saved settings'
        feature_names = {'Waveform': 'waveform', 'AutoCorr': 'spike timing',
                         'ISI': 'spike intervals', 'PETH': 'stimulus fingerprint'}
        features = params.get('clustering', {}).get('features', [])
        motion = params.get('motionEstimation', {}).get('features', [])
        text = 'DANT clustering: ' + ', '.join(feature_names.get(k, k) for k in features)
        if any('PETH' in stage for stage in motion) and 'PETH' not in features:
            text += '; fingerprint for motion only'
        return text if features else 'DANT native inputs; see saved settings'
    properties = provenance.get('calibrated_model', {}).get('properties', [])
    if properties:
        labels = {key: label for key, label, _ in EVIDENCE_FIELDS}
        return 'Fitted model: ' + ', '.join(labels.get(k, k.replace('_', ' ')) for k in properties)
    name = provenance.get('evidence_display_name')
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the combined stream is labelled from the saved weights (event-optional Task 12) | date: 2026-10-10]
    labels = dict(STREAM_LABELS, combined=combined_label(provenance))
    stream = labels.get(provenance.get('feature_stream'))
    if name and stream:
        return f'{name}: {stream}'
    return name or labels.get(provenance.get('feature_stream', 'combined'), 'Inputs specified in saved model')
    # [AGENT EDIT END]


def backend_header_text(result) -> str:
    if result.backend_id == 'unassigned':
        return 'backend: none'
    return f'backend: {names.backend_short(result.backend_id)}'


def result_summary(result) -> str:
    """Hover text for the backend header."""
    if result.backend_id == 'unassigned':
        return 'no matching run · all observations unassigned'
    rule = result.provenance.get('assignment') or result.tracking_rule
    kind = result.score.kind.value
    return '\n'.join((names.backend_label(result.backend_id),
                      'tracking: ' + ASSIGNMENT_LABELS.get(rule, rule.replace('_', ' ')),
                      'score: ' + ('none' if kind == 'none' else kind.replace('_', ' ')),
                      'inputs: ' + result_input_label(result),
                      'run ' + result.run_id))


def diagnostic_gate_outcome(row, gate):
    """None means uncomputed; explicit saved gate decisions remain authoritative."""
    explicit = row.get('gate_' + gate)
    if isinstance(explicit, (bool, np.bool_)):
        return bool(explicit)
    if isinstance(explicit, (int, float, np.integer, np.floating)) and explicit in (0, 1):
        return bool(explicit)
    if any(str(k).startswith('gate_') for k in row.keys()):
        return None
    passed = row.get('gates_passed')
    if isinstance(passed, str):
        return gate in {k.strip() for k in passed.split(',')}
    return None


def merge_display_evidence(result_pairs, cached_pairs):
    """Supplement plots with cached measurements without replacing backend scores."""
    cache = {}
    for row in cached_pairs.to_dict('records'):
        a, b = row['cellid_a'], row['cellid_b']
        cache[(a, b)] = row
        cache[(b, a)] = {
            (key[:-2] + '_b' if key.endswith('_a') else
             key[:-2] + '_a' if key.endswith('_b') else key): value
            for key, value in row.items()}
    records = []
    for row in result_pairs.to_dict('records'):
        values = dict(cache.get((row['cellid_a'], row['cellid_b']), {}))
        values.update(row)  # Native score, selection, and decision always win.
        records.append(values)
    if not records:
        return result_pairs.copy()
    return pd.DataFrame.from_records(records)


def pct(v):
    return f'{100*v:.0f}%' if np.isfinite(v) else '--'


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: unit colours from the active theme's tokens; COL_MUTED and VERDICT_COLOR removed (launcher-gui Task 8) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: unit colours from the Unit colours palette, or Light's own ramp on Light; none for no units (graphite-palettes Task 2) | date: 2026-10-11]
def unit_colors(n: int) -> list:
    """One colour per selected unit in selection order, from the Unit colours palette, or Light's own ramp on
    Light; ``n <= 0`` gives ``[]``."""
    if n <= 0:
        return []
    if uses_unit_palette():
        return unit_palettes.colours(n)
    many = PALETTES[_theme]['many']
    return [many[i % len(many)] for i in range(n)]
# [AGENT EDIT END]
# [AGENT EDIT END]


def declared_conditions(value) -> Optional[list]:
    """A run's declared stimulus conditions as a list of strings, or None when absent or malformed."""
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return None
    return list(value) if isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value) else None
# [AGENT EDIT END]
