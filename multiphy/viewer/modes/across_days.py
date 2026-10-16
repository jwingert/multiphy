# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the across-days mode: validate_days_result from multiday_viewer.py:36-64, load from display_cache multiday_viewer.py:67-133 returning RunData, the info line from multiday_viewer.py:382-385, the header from multiday_viewer.py:335-340 and the merge refusal from multiday_viewer.py:282-285 (viewer-split Task 3) | date: 2026-10-06]
"""The across-days mode: two or more recording days of one project, curated in the project.

``load`` builds a run's ``RunData`` from its selection's dataset and result. Imports no Qt and no
lab module.
"""
import itertools
from typing import Optional

import numpy as np
import pandas as pd

from multiphy import clusters
from multiphy.backends import MatchResult
from multiphy.benchmark_pairs import _waveform_summary
from multiphy.viewer.data import (Mode, RunData, UnitData, Words, apply_run_provenance,
                                              complete_day_pairs, display_pairs)

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no pair words (native-same-day Task 10) | date: 2026-10-11]
WORDS = Words(track_column='days', member_column='Date / session', observations='dated observations',
              import_tip='Import a result covering every unit of this selection.')
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-day pair row (proposed_link) loads; only the track loop refuses two units of one day (native-same-day Task 7) | date: 2026-10-11]
def validate_days_result(dataset, result: MatchResult) -> MatchResult:
    """Refuse a result that is not of this selection, or whose track holds two units of one day."""
    try:
        result.validate()
    except ValueError as exc:
        if str(exc) == 'Result contains a same-session track collision':
            raise ValueError('a track holds two units from one day') from exc
        raise
    for key in ('input_manifest_sha256', 'dataset_id'):
        if result.provenance.get(key) != dataset.metadata[key]:
            raise ValueError('result belongs to another selection')
    if set(result.units.cellid) != set(dataset.units.cellid):
        raise ValueError('result must cover every unit of the selection')
    expected = dict(zip(dataset.units.cellid, dataset.units.session_id))
    if dict(zip(result.units.cellid, result.units.session_id)) != expected:
        raise ValueError('result session ids disagree with the selection')
    for track in result.tracks:
        days = [expected[c] for c in track]
        if len(set(days)) != len(days):
            raise ValueError('a track holds two units from one day')
    return result
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: day pairs carry no list label (native-same-day Task 10 fix round 1) | date: 2026-10-11]
def load(selection, dataset, result, extras: dict) -> RunData:
    """The run's units, pairs and day pairs; ``extras`` is ``{day: full-day extras}`` (``Selection.day_extras()``),
    whose ``tuning_*`` arrays give each site a frequency axis and each unit its tuning curve."""
    validate_days_result(dataset, result)
    units, arrays = dataset.units.copy(), {}
    for session, group in units.groupby('session_id', sort=True):
        for row in group.itertuples():
            feature = dataset.features[row.cellid]
            summary = _waveform_summary(feature, dataset.geometry[session])
            wave, peak = summary['waveform'], int(np.argmax(summary['ptp']))
            patch = np.zeros((41, wave.shape[0]))
            lo, hi = max(0, peak - 20), min(wave.shape[1], peak + 21)
            patch[lo - peak + 20:hi - peak + 20] = wave[:, lo:hi].T
            arrays[row.cellid] = dict(waveform=wave[:, peak], ptp=summary['ptp'], patch=patch,
                fingerprints=feature.get('fingerprints') or {}, trial_counts=feature.get('trial_counts') or {},
                fingerprint_bin_seconds=feature.get('fingerprint_bin_seconds'),
                **{k: np.asarray(feature[k]) for k in ('isi', 'acg') if k in feature})
            for column, value in dict(peak_chan=peak, n_chan=wave.shape[1], depth=summary['depth'],
                                      x_um=summary['x_um'], amp=summary['amp'], sw_ms=summary['sw_ms']).items():
                units.loc[units.cellid == row.cellid, column] = value
    units['siteid'] = units['session_id']
    days = []
    for well, group in units.groupby('well', sort=True):
        sessions = group[['session_id', 'session_date']].drop_duplicates().sort_values(['session_date', 'session_id'])
        for a, b in itertools.combinations(sessions.itertuples(), 2):
            if a.session_date == b.session_date:
                continue
            days.append(dict(siteid_a=a.session_id, siteid_b=b.session_id, well=well,
                date_a=a.session_date, date_b=b.session_date,
                days_apart=(pd.Timestamp(b.session_date) - pd.Timestamp(a.session_date)).days))
    sites, freq_axes = clusters.unit_sites(units), {}
    for day, day_extras in sorted((extras or {}).items()):
        if 'tuning_freqs' not in day_extras:
            continue
        rows = dict(zip((str(u) for u in day_extras['tuning_unit']), day_extras['tuning_ftc']))
        for cellid in units.loc[units.session_id == day, 'cellid']:
            freq_axes[sites[str(cellid)]] = np.asarray(day_extras['tuning_freqs']).tolist()
            if str(cellid) in rows:
                arrays[cellid]['tuning'] = np.asarray(rows[str(cellid)])
    peaks = dict(zip(units.cellid, units.peak_chan))
    unit_map = {c: UnitData(cellid=c, siteid=sites[str(c)], peak_chan=int(peaks[c]), **a) for c, a in arrays.items()}
    session_rows = units.drop_duplicates('session_id').set_index('session_id')
    dates = session_rows.session_date.to_dict()
    wells = session_rows.well.to_dict()

    def new_pair(a, b):
        return dict(siteid_a=a, siteid_b=b, well=wells[a], date_a=dates[a], date_b=dates[b],
                    days_apart=abs((pd.Timestamp(dates[b]) - pd.Timestamp(dates[a])).days))

    pair_table = display_pairs(result, sites, pair_evidence=dataset.pairs.copy())
    day_pairs = apply_run_provenance(complete_day_pairs(days, pair_table, new_pair), result.provenance)
    dataset_id = dataset.metadata['dataset_id']
    n_wells = units.well.nunique()
    info = (f"{len(result.units)} units · {len(result.unmatched)} unmatched · {len(set(dates.values()))} days"
            + (f' · {n_wells} wells' if n_wells > 1 else ''))
    return RunData(
        mode='across_days', result=result, words=WORDS, title=f'{dataset_id} · days', info=info,
        info_tip=f'{dataset_id} · real recording days\nNo reference identities. Cross-day matches need review.',
        unit_table=units, unit_map=unit_map, pair_table=pair_table, day_pair_list=tuple(day_pairs),
        geometries=dataset.geometry, geometry_sources={s: 'measured geometry' for s in dataset.geometry},
        day_names={s: f'{d} · {s}' for s, d in dates.items()}, freq_axes=freq_axes,
        min_fingerprint_repeats=dataset.metadata['pair_options']['min_repeats'],
        parts=frozenset({'isi', 'acg'} | ({'tuning'} if freq_axes else set())))
# [AGENT EDIT END]


def header(data: RunData, cellids, row) -> str:
    """The wells and the selected observations, else the pair on screen, else the selected units."""
    units = data.units()
    if cellids and 'well' in units:
        selected = units.set_index('cellid').loc[list(cellids)]
        wells = ', '.join(sorted(selected.well.unique()))
        return f'{wells} · Selected observations: ' + ', '.join(cellids)
    if row is not None:
        return f'{row.cellid_a}   vs   {row.cellid_b}'
    return 'Selected units: ' + ', '.join(cellids) if cellids else 'No units selected'


def merge_refusal(data: RunData, members) -> Optional[str]:
    """Why ``members`` cannot become one track: different wells or a missing well or date; else ``None``."""
    rows = data.units().set_index('cellid').reindex(list(members))
    if rows.well.isna().any() or rows.session_date.isna().any() or rows.well.nunique() > 1:
        return 'Merge rejected: selected observations belong to different wells or lack well/date identity.'
    return None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the plots end with the drift panel (drift-view Task 4) | date: 2026-10-08]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram are two plots (browse-same-day Task 5) | date: 2026-10-12]
ACROSS_DAYS = Mode(id='across_days', layout_name='viewer/across_days_layout.json', curates=True, exports=True,
                   words=WORDS, plots=('waveform', 'wfmap', 'isi', 'acg', 'footprint', 'probe', 'fingerprint',
                                       'tuning', 'mixture', 'drift'),
                   load=load, header=header, merge_refusal=merge_refusal)
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
