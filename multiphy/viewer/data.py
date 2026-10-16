# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the typed run data panels read, and the pair preparation from gui.py:997-1026 (mixture_day_pairs, apply_run_provenance) and gui.py:2222-2257 (set_match_result's pair table and day pairs) (viewer-split Task 2) | date: 2026-10-06]
"""The typed data a viewer panel reads: one frozen ``UnitData`` per unit, a mode's ``Words``, the ``RunData``
a ``Mode`` loads, and the Qt-free preparation of a result's pair table and day pairs.

Panels read ``RunData`` methods, never dicts. Imports no Qt and no lab module.
"""
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

import pandas as pd

from multiphy.viewer import display

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: offsets is no longer a part; RunData.day_offset is deleted (drift-view Task 3) | date: 2026-10-08]
PARTS = frozenset({'isi', 'acg', 'tuning'})
# [AGENT EDIT END]
PAIR_COLUMNS = ('cellid_a', 'cellid_b', 'siteid_a', 'siteid_b', 'match_score', 'decision', 'selected', 'score_kind')


@dataclass(frozen=True, eq=False)
class UnitData:
    """One unit's measured features; a field a mode does not have is ``None`` (or ``{}``), and a panel skips it."""
    cellid: str
    siteid: str
    waveform: object = None
    ptp: object = None
    patch: object = None
    peak_chan: object = None
    fingerprints: dict = field(default_factory=dict)
    trial_counts: dict = field(default_factory=dict)
    fingerprint_bin_seconds: object = None
    isi: object = None
    acg: object = None
    tuning: object = None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no pair words: the day table replaces the day pairs and candidates (native-same-day Task 10) | date: 2026-10-11]
@dataclass(frozen=True)
class Words:
    """The nouns a mode's panels and tables use."""
    track_column: str
    member_column: str
    observations: str
    import_tip: str
# [AGENT EDIT END]


@dataclass(frozen=True, eq=False)
class RunData:
    """One loaded run: its result, units, pairs and day pairs, read by every panel through its methods."""
    mode: str
    result: object
    words: Words
    title: str
    info: str
    info_tip: str
    unit_table: pd.DataFrame
    unit_map: Mapping[str, UnitData]
    pair_table: pd.DataFrame
    day_pair_list: tuple
    geometries: dict
    geometry_sources: dict
    day_names: dict
    freq_axes: dict
    min_fingerprint_repeats: int
    parts: frozenset
    _sites: dict = field(init=False, repr=False)

    def __post_init__(self):
        missing = [c for c in PAIR_COLUMNS if c not in self.pair_table.columns]
        if missing:
            raise ValueError(f'pairs lack {missing}')
        unknown = sorted(set(self.parts) - PARTS)
        if unknown:
            raise ValueError(f'unknown parts {unknown}')
        object.__setattr__(self, '_sites', dict(zip(self.unit_table['cellid'], self.unit_table['siteid'])))

    def unit(self, cellid: str) -> UnitData:
        return self.unit_map[cellid]

    def units(self) -> pd.DataFrame:
        return self.unit_table

    def sites(self) -> dict:
        """``{cellid: siteid}`` of every unit."""
        return self._sites

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the run's sites ranked by recording date, for the Day ramp (graphite-palettes Task 5) | date: 2026-10-11]
    def site_order(self) -> list:
        """The run's site ids as ``str``, each once, in recording order.

        With a ``session_date`` column: by ``(session_date, siteid)``, dates sorted as stored (ISO ``YYYY-MM-DD``
        strings), so sites recorded the same day are ordered by ``siteid`` and sites without a date come last.
        Without the column: the sorted unique ``siteid`` values.
        """
        table = self.unit_table
        if 'session_date' in table.columns:
            ranked = (table[['siteid', 'session_date']].drop_duplicates('siteid')
                      .sort_values(['session_date', 'siteid'], na_position='last', kind='mergesort'))
            return [str(s) for s in ranked['siteid']]
        return sorted({str(s) for s in table['siteid']})
    # [AGENT EDIT END]

    def pairs(self) -> pd.DataFrame:
        return self.pair_table

    def day_pairs(self) -> list:
        return list(self.day_pair_list)

    def geometry(self, site: str):
        """The site's channel positions, or ``None``."""
        return self.geometries.get(site)

    def geometry_source(self, site: str) -> str:
        return self.geometry_sources.get(site, 'unknown')

    def freq_axis(self, site: str) -> Optional[list]:
        return self.freq_axes.get(site)

    def min_repeats(self) -> int:
        return self.min_fingerprint_repeats

    def day_name(self, site: str) -> str:
        return self.day_names.get(site, site)

    def available(self) -> frozenset:
        """The optional parts (``PARTS``) this run has."""
        return self.parts


@dataclass(frozen=True)
class Mode:
    """How one kind of run loads and reads: ``load(selection, dataset, result, extras) -> RunData``,
    ``header(data, cellids, row) -> str`` and ``merge_refusal(data, members) -> Optional[str]``."""
    id: str
    layout_name: str
    curates: bool
    exports: bool
    words: Words
    plots: tuple
    load: Callable
    header: Callable
    merge_refusal: Callable


def display_pairs(result, sites: dict, pair_evidence: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the refusal of a run saved without run_evidence (event-optional Task 12) | date: 2026-10-10]
    """The result's pairs as the panels show them: ``score_kind`` filled, evidence merged, siteids filled from ``sites``.

    The result is validated first, so ``decision`` and ``selected`` come from the contract. A result unit
    missing from ``sites`` is refused, as is a run whose mixture references have no valid ``run_evidence``
    (``display.require_run_evidence``).
    """
    # [AGENT EDIT END]
    result.validate()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a run saved before event-optional matchers is refused before any panel reads it (event-optional Task 12, D56) | date: 2026-10-10]
    display.require_run_evidence(result.provenance)
    # [AGENT EDIT END]
    sites = dict(sites)
    missing = [str(c) for c in result.units.cellid if c not in sites]
    if missing:
        raise ValueError(f"no siteid for {len(missing)} units: {', '.join(missing[:5])}")
    P = result.pairs.copy()
    if 'score_kind' not in P:
        P['score_kind'] = result.score.kind.value
    if pair_evidence is not None:
        if result.score.kind.value == 'none':
            pair_evidence = pair_evidence.drop(columns=list(display.CACHED_SCORE_COLUMNS), errors='ignore')
        P = display.merge_display_evidence(P, pair_evidence)
    assignment = result.provenance.get('assignment', '')
    if assignment.startswith('native_DANT'):
        P['score_display_label'] = 'DANT native membership (no confidence score)'
    for side in ('a', 'b'):
        column = f'siteid_{side}'
        filled = P[f'cellid_{side}'].astype(str).map(sites)
        if column not in P:
            P[column] = filled
        else:
            blank = P[column].isna() | (P[column].astype(str).str.strip() == '')
            P[column] = P[column].astype(object).where(~blank, filled)
    return P


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a site pair within one site is skipped, both copied and appended (native-same-day Task 7) | date: 2026-10-11]
def complete_day_pairs(day_pairs: list, pairs: pd.DataFrame, new_pair: Optional[Callable] = None) -> list:
    """A new list of copies of ``day_pairs``, with each site pair of ``pairs`` it lacks appended once as ``new_pair(a, b)``;
    pairs within one site are skipped."""
    if new_pair is None:
        def new_pair(a, b):
            return {'siteid_a': a, 'siteid_b': b, 'days_apart': 'unknown'}
    out = [dict(dp) for dp in day_pairs if dp['siteid_a'] != dp['siteid_b']]
    known = {frozenset((dp['siteid_a'], dp['siteid_b'])) for dp in out}
    for a, b in pairs[['siteid_a', 'siteid_b']].drop_duplicates().itertuples(index=False, name=None):
        if a != b and frozenset((a, b)) not in known:
            out.append(new_pair(a, b))
            known.add(frozenset((a, b)))
    return out
# [AGENT EDIT END]


def mixture_day_pairs(provenance: dict) -> dict:
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: every saved weight_<stream> key is copied, so a run without events carries its physiology and waveform weights (event-optional Task 12) | date: 2026-10-10]
    """Each group run's fitted mixture per day pair: ``{frozenset((day_a, day_b)): {'streams', 'weight_<stream>', ...}}``,
    with every ``weight_*`` key the group's mixture references saved.

    ``observed`` is every gated candidate the mixture was fit on. A stream whose fit is
    ``unavailable`` is left out. Only ``group_runs`` are read, never top-level references.
    """
    # [AGENT EDIT END]
    out = {}
    for entry in (provenance.get('group_runs') or {}).values():
        p = entry['provenance']
        for sp in p.get('session_pairs', []):
            refs = p.get('mixture_references', {}).get(sp['recording_group'])
            if not refs:
                continue
            streams = {s: {'same': refs['streams'][s]['same'], 'diff': refs['streams'][s]['diff'],
                           'observed': e['observed'], 'pi': e['pi']}
                       for s, e in (sp.get('streams') or {}).items() if 'unavailable' not in e}
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: copy every saved weight_* key (event-optional Task 12) | date: 2026-10-10]
            out[frozenset((sp['session_a'], sp['session_b']))] = {
                'streams': streams, **{k: v for k, v in refs.items() if k.startswith('weight_')}}
            # [AGENT EDIT END]
    return out


def apply_run_provenance(day_pairs: list, provenance: dict) -> list:
    """A new list of copies of ``day_pairs``: each without streams gets its run's mixture; the inputs are untouched.

    The run's registration stays in its provenance, never in a day pair.
    """
    mixtures = mixture_day_pairs(provenance)
    out = []
    for dp in day_pairs:
        dp = dict(dp)
        key = frozenset((dp['siteid_a'], dp['siteid_b']))
        if not dp.get('streams') and key in mixtures:
            dp.update(mixtures[key])
        out.append(dp)
    return out
# [AGENT EDIT END]
