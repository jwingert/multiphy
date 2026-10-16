# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a RunData built from the old hand-made viewer dicts, for panel tests (viewer-split Task 4) | date: 2026-10-06]
"""``DictData``: a ``RunData`` from the hand-made dicts the old viewer tests set on ``PairState``."""
from typing import Optional

import numpy as np
import pandas as pd

from multiphy import backends
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the contract types day_table_data builds its result from (native-same-day Task 8) | date: 2026-10-11]
from multiphy.contract.provenance import Provenance
from multiphy.contract.results import MatchResult, ScoreBlock, ScoreKind, TrackKind
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: FAKE_MODE builds on the viewer's Mode (viewer-split Task 7) | date: 2026-10-06]
from multiphy.viewer.data import Mode, PARTS, RunData, UnitData, complete_day_pairs, display_pairs
# [AGENT EDIT END]
from multiphy.viewer.modes import same_day

# the old per-unit array keys and the UnitData fields they become
ARRAY_FIELDS = {'wf': 'waveform', 'ptp': 'ptp', 'patch': 'patch', 'fingerprints': 'fingerprints',
                'trial_counts': 'trial_counts', 'fingerprint_bin_seconds': 'fingerprint_bin_seconds',
                'isi': 'isi', 'acg': 'acg', 'ftc': 'tuning'}


def _unit_table(units, sites: dict) -> pd.DataFrame:
    """``units`` as a frame with ``cellid``, ``siteid`` and ``su_type``: a frame is copied, a list of cellids is built."""
    if isinstance(units, pd.DataFrame):
        table = units.copy()
    else:
        table = pd.DataFrame({'cellid': [str(u) for u in units]})
    if 'siteid' not in table:
        table['siteid'] = table.cellid.map(sites)
    if 'su_type' not in table:
        table['su_type'] = 'SU'
    return table


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: day pairs carry no list label, as a mode load gives none (native-same-day Task 10 fix round 1) | date: 2026-10-11]
def DictData(units, arrays: dict, meta: dict, sites: dict, pairs: Optional[pd.DataFrame] = None,
             result=None, evidence: Optional[pd.DataFrame] = None) -> RunData:
    """A same-day ``RunData`` from the old dicts: ``units`` (cellids or a frame), ``arrays`` (``{cellid: {wf, ptp,
    patch, fingerprints, trial_counts, fingerprint_bin_seconds, isi, acg, ftc}}``), ``meta`` (``geometry``,
    ``geometry_source``, ``freq_axis``, ``session_dates``, ``min_fingerprint_repeats``, ``day_pairs``) and ``sites``.

    Given ``result``, its pair table is ``display_pairs(result, sites, pair_evidence=evidence)``; without one, a minimal
    valid result is built with ``backends.result_from_pairs`` from ``units`` and ``pairs`` (no pairs when ``None``).
    Every unit of ``units`` gets a ``UnitData``; its parts are the ones the dicts hold.
    """
    sites = dict(sites)
    arrays = arrays or {}
    meta = meta or {}
    table = _unit_table(units, sites)
    if result is None:
        frame = pairs if pairs is not None else pd.DataFrame({'cellid_a': [], 'cellid_b': []})
        result = backends.result_from_pairs('dict_data', '1', table, frame, run_id='dict_data')
    pair_table = display_pairs(result, sites, pair_evidence=evidence)
    peaks = dict(zip(table.cellid, table.peak_chan)) if 'peak_chan' in table else {}
    unit_map = {}
    for cellid in table.cellid:
        held = arrays.get(cellid, {})
        fields = {field: held[key] for key, field in ARRAY_FIELDS.items() if key in held}
        peak = peaks.get(cellid)
        if peak is None and 'ptp' in held:
            peak = int(np.argmax(held['ptp']))
        unit_map[cellid] = UnitData(cellid=cellid, siteid=sites[cellid], peak_chan=peak, **fields)
    freq_axes = dict(meta.get('freq_axis') or {})
    parts = {part for part, key in (('isi', 'isi'), ('acg', 'acg')) if any(key in a for a in arrays.values())}
    if freq_axes:
        parts.add('tuning')
    assert parts <= PARTS
    dates = dict(meta.get('session_dates') or {})
    day_pairs = complete_day_pairs(meta.get('day_pairs') or [], pair_table)
    return RunData(
        mode='same_day', result=result, words=same_day.WORDS, title='', info='', info_tip='',
        unit_table=table, unit_map=unit_map, pair_table=pair_table,
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day pairs, unlabelled (viewer-split Task 7; native-same-day Task 10) | date: 2026-10-11]
        day_pair_list=tuple(day_pairs),
        # [AGENT EDIT END]
        geometries=dict(meta.get('geometry') or {}), geometry_sources=dict(meta.get('geometry_source') or {}),
        day_names={s: f'{d} · {s}' for s, d in dates.items()}, freq_axes=freq_axes,
        min_fingerprint_repeats=meta.get('min_fingerprint_repeats', 4), parts=frozenset(parts))
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a curating fake mode with the same-day words and rules, for App tests (viewer-split Task 7) | date: 2026-10-06]
def _no_load(selection, dataset, result, extras):
    raise NotImplementedError('the fake mode loads no run')


FAKE_MODE = Mode(id='fake', layout_name='viewer/fake_layout.json', curates=True, exports=False,
                 words=same_day.WORDS, plots=same_day.SAME_DAY.plots, load=_no_load,
                 header=same_day.header, merge_refusal=same_day.merge_refusal)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a two-day native run with proposed same-day links, for the Tracked units and day table tests (native-same-day Task 8) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: day_table_data takes depths, x positions and a saved registration for the distance sort (browse-same-day Task 2) | date: 2026-10-12]
def day_table_data(depth: Optional[list] = None, x_um: Optional[list] = None,
                   offsets: Optional[dict] = None) -> RunData:
    """Day A (``A-1``…``A-5``) and day B (``B-1``, ``B-2``): track ``A-1 B-1``, proposed links ``A-1 A-2`` and
    ``A-2 B-1``, same-session and unmatched rows.

    ``depth`` replaces the ``depth`` column (default ``[100, 120, 400, 150, 90, 110, 300]``); ``x_um`` adds an
    ``x_um`` column only when given; ``offsets`` (site → µm) saves a ``matches`` registration of one group in
    the result's provenance, else the provenance is empty.
    """
    cellids = ['A-1', 'A-2', 'A-3', 'A-4', 'A-5', 'B-1', 'B-2']
    units = pd.DataFrame({
        'cellid': cellids,
        'siteid': [c[0] for c in cellids],
        'session_id': [c[0] for c in cellids],
        'su_type': 'SU',
        'depth': list(depth) if depth is not None else [100, 120, 400, 150, 90, 110, 300],
        'session_date': ['2023-01-01' if c[0] == 'A' else '2023-01-02' for c in cellids],
    })
    if x_um is not None:
        units['x_um'] = list(x_um)
    meta = {'session_dates': {'A': '2023-01-01', 'B': '2023-01-02'}}
    rows = [('A-1', 'B-1', .95, 'native_track'),
            ('A-1', 'A-2', .90, 'proposed_link'),
            ('A-2', 'B-1', .80, 'proposed_link'),
            ('A-1', 'A-3', .30, 'same_session'),
            ('A-2', 'A-3', .20, 'same_session'),
            ('B-1', 'B-2', .10, 'same_session'),
            ('A-1', 'B-2', .97, 'unmatched_by_native_tracking'),
            ('A-2', 'B-2', .50, 'unmatched_by_native_tracking'),
            ('A-3', 'B-1', .60, 'unmatched_by_native_tracking'),
            ('A-3', 'B-2', .70, 'unmatched_by_native_tracking')]
    pairs = pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'match_score', 'decision'])
    pairs['selected'] = pd.Series([d == 'native_track' for d in pairs.decision], dtype=object)
    if offsets is not None:
        provenance = Provenance(extra={'group_runs': {'': {'run_id': 'r', 'provenance': {'registration': {
            'method': 'matches', 'offsets_um': offsets, 'diagnostics': {'': {'groups': [sorted(offsets)]}}}}}}})
    else:
        provenance = Provenance()
    result = MatchResult('unitmatch', '1', 'day_table', units, pairs, (('A-1', 'B-1'),),
                         ('A-2', 'A-3', 'A-4', 'A-5', 'B-2'), (),
                         ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score'), TrackKind.NATIVE,
                         'native_intermediate', provenance).validate()
    return DictData(units, {}, meta, {c: c[0] for c in units.cellid}, result=result)
# [AGENT EDIT END]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a three-day native run for the distance sort and the string tests (day-table-keys Task 2) | date: 2026-10-14]
def string_days_data() -> RunData:
    """Days A, B and C (``2023-01-01``, ``2023-01-02``, ``2023-01-03``): units ``A-1``, ``A-2``, ``B-1``, ``B-2``,
    ``B-3``, ``C-1``, ``C-2``, ``C-3`` at ``x_um`` 0 and depths 100, 300, 60, 120, 290, 135, 100, 310; track
    ``A-2 B-3 C-3``; the rest unmatched, with unmatched pairs ``A-1 B-1`` (.90), ``A-1 B-3`` (.95), ``B-1 C-2``
    (.60) and ``B-2 C-1`` (.80); an empty provenance."""
    cellids = ['A-1', 'A-2', 'B-1', 'B-2', 'B-3', 'C-1', 'C-2', 'C-3']
    units = pd.DataFrame({
        'cellid': cellids,
        'siteid': [c[0] for c in cellids],
        'session_id': [c[0] for c in cellids],
        'su_type': 'SU',
        'depth': [100, 300, 60, 120, 290, 135, 100, 310],
        'x_um': [0] * len(cellids),
    })
    meta = {'session_dates': {'A': '2023-01-01', 'B': '2023-01-02', 'C': '2023-01-03'}}
    unmatched = 'unmatched_by_native_tracking'
    rows = [('A-2', 'B-3', .90, 'native_track'),
            ('B-3', 'C-3', .90, 'native_track'),
            ('A-1', 'B-1', .90, unmatched),
            ('A-1', 'B-3', .95, unmatched),
            ('B-1', 'C-2', .60, unmatched),
            ('B-2', 'C-1', .80, unmatched)]
    pairs = pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'match_score', 'decision'])
    pairs['selected'] = pd.Series([d == 'native_track' for d in pairs.decision], dtype=object)
    result = MatchResult('unitmatch', '1', 'string_days', units, pairs, (('A-2', 'B-3', 'C-3'),),
                         ('A-1', 'B-1', 'B-2', 'C-1', 'C-2'), (),
                         ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score'), TrackKind.NATIVE,
                         'native_intermediate', Provenance()).validate()
    return DictData(units, {}, meta, {c: c[0] for c in units.cellid}, result=result)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a native run of any number of days, for the day-table fit and scroll tests (day-table widths) | date: 2026-10-13]
def many_days_data(n_days: int, n_units: int = 4) -> RunData:
    """``n_days`` days ``D00a``, ``D01a``, … of ``n_units`` units each (``D00a-300-1``, ``D00a-301-1``, …): one
    track per unit number across every day, its days linked in day order."""
    sites = [f'D{i:02d}a' for i in range(n_days)]
    cellids = [f'{s}-{300 + u}-1' for s in sites for u in range(n_units)]
    dates = {s: f'2023-01-{i + 1:02d}' for i, s in enumerate(sites)}
    siteids = [c.split('-')[0] for c in cellids]
    units = pd.DataFrame({'cellid': cellids, 'siteid': siteids, 'session_id': siteids, 'su_type': 'SU',
                          'depth': [100. + 10 * int(c.split('-')[1]) for c in cellids],
                          'session_date': [dates[s] for s in siteids]})
    tracks = tuple(tuple(f'{s}-{300 + u}-1' for s in sites) for u in range(n_units))
    rows = [(a, b, .9, 'native_track') for t in tracks for a, b in zip(t, t[1:])]
    pairs = pd.DataFrame(rows, columns=['cellid_a', 'cellid_b', 'match_score', 'decision'])
    pairs['selected'] = pd.Series([True] * len(pairs), dtype=object)
    result = MatchResult('unitmatch', '1', 'many_days', units, pairs, tracks, (), (),
                         ScoreBlock(ScoreKind.UNCALIBRATED, 'match_score'), TrackKind.NATIVE,
                         'native_intermediate', Provenance()).validate()
    return DictData(units, {}, {'session_dates': dates}, dict(zip(cellids, siteids)), result=result)
# [AGENT EDIT END]
