# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the same-day mode: result_partition and validate_benchmark_result from benchmark_viewer.py:39-98, run_data from display_cache benchmark_viewer.py:100-173 returning RunData, the labels and info line from benchmark_viewer.py:340-349; fix round 1: validate_benchmark_result declares its BenchmarkDataset return (viewer-split Task 3) | date: 2026-10-06]
"""The same-day mode: the two temporal parts of one recording, inspected only.

``run_data`` builds a run's ``RunData`` from its benchmark dataset and result. Imports no Qt and no
lab module.
"""
import itertools
from typing import Optional

import numpy as np
import pandas as pd

from nems_lbhb.unit_match import clusters
from nems_lbhb.unit_match.benchmark import BenchmarkDataset
from nems_lbhb.unit_match.benchmark_pairs import _waveform_summary
from nems_lbhb.unit_match.viewer.data import (Mode, RunData, UnitData, Words, apply_run_provenance,
                                              complete_day_pairs, display_pairs)

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no pair words (native-same-day Task 10) | date: 2026-10-11]
WORDS = Words(track_column='parts', member_column='temporal part', observations='temporal observations',
              import_tip='Compare a saved alternative result for this exact temporal benchmark partition')
# [AGENT EDIT END]


def result_partition(result):
    """Typed ``partition`` (runner) else the pilot ``benchmark_partition``."""
    return result.provenance.get('partition') or result.provenance.get('benchmark_partition')


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a same-part pair row (proposed_link) loads; only a pair crossing recording groups is refused (native-same-day Task 7) | date: 2026-10-11]
def validate_benchmark_result(dataset, result, *, input_manifest_sha256: Optional[str] = None) -> BenchmarkDataset:
    """Require an entire declared partition with exact dataset/session identity."""
    dataset.validate()
    result.validate()
    if input_manifest_sha256 is not None and result.provenance.get('input_manifest_sha256') != input_manifest_sha256:
        raise ValueError('Result input_manifest_sha256 does not match the exact persisted benchmark archive')
    expected = {'block_id': dataset.metadata.get('block_id'),
                'runclass': dataset.metadata.get('runclass'),
                'cache_built': dataset.metadata.get('cache_built')}
    for key, value in expected.items():
        if value is None or result.provenance.get(key) != value:
            raise ValueError(f'Result {key} does not identify this exact benchmark dataset')
    partition = result_partition(result)
    if partition not in ('train', 'calibration', 'test', 'all'):
        raise ValueError('Result must declare its complete benchmark_partition')
    ids = set(dataset.labels.cellid if partition == 'all' else
              dataset.labels.loc[dataset.labels.partition == partition, 'cellid'])
    if not ids or ids != set(result.units.cellid):
        raise ValueError('Result must contain every unit in its declared partition, including unmatched units')
    expected_sessions = dataset.units.set_index('cellid').loc[sorted(ids), 'session_id']
    actual_sessions = result.units.set_index('cellid').loc[sorted(ids), 'session_id']
    if not expected_sessions.equals(actual_sessions):
        raise ValueError('Result session identities disagree with the benchmark')
    expected_groups = dataset.units.set_index('cellid').loc[sorted(ids), 'recording_group']
    if ('recording_group' not in result.units or not expected_groups.equals(
            result.units.set_index('cellid').loc[sorted(ids), 'recording_group'])):
        raise ValueError('Result original recording identities disagree with the benchmark')
    for track in result.tracks:
        if expected_groups.loc[list(track)].nunique() != 1:
            raise ValueError('Benchmark tracks cannot cross original recordings')
    for pair in result.pairs.itertuples():
        if expected_groups[pair.cellid_a] != expected_groups[pair.cellid_b]:
            raise ValueError('Benchmark pair evidence must stay within one original recording')
    return dataset.subset(ids)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: day pairs carry no list label (native-same-day Task 10 fix round 1) | date: 2026-10-11]
def run_data(dataset, result, *, input_manifest_sha256: Optional[str] = None, extras: Optional[dict] = None) -> RunData:
    """The run's units, pairs and temporal part pairs, never its reference identities; ``extras`` is
    ``{day: full-day extras}``, whose ``tuning_*`` arrays give each site a frequency axis and each unit its tuning curve."""
    subset = validate_benchmark_result(dataset, result, input_manifest_sha256=input_manifest_sha256)
    arrays, summaries = {}, []
    for session, units in subset.units.groupby('session_id', sort=True):
        for unit in units.itertuples():
            feature = subset.features[unit.cellid]
            summary = _waveform_summary(feature, subset.geometry[session])
            wave = summary['waveform']
            peak = int(np.argmax(summary['ptp']))
            half = 20
            patch = np.zeros((2 * half + 1, wave.shape[0]))
            start, stop = max(0, peak - half), min(wave.shape[1], peak + half + 1)
            patch[start - peak + half:stop - peak + half] = wave[:, start:stop].T
            arrays[unit.cellid] = {'waveform': wave[:, peak], 'ptp': summary['ptp'], 'patch': patch,
                'fingerprints': feature.get('fingerprints') or {},
                'fingerprint_bin_seconds': feature.get('fingerprint_bin_seconds'),
                'trial_counts': feature.get('trial_counts') or {},
                **{name: np.asarray(feature[name]) for name in ('isi', 'acg') if name in feature}}
            summaries.append({'cellid': unit.cellid, 'peak_chan': peak, 'n_chan': wave.shape[1],
                'depth': summary['depth'], 'x_um': summary['x_um'], 'amp': summary['amp'], 'sw_ms': summary['sw_ms']})
    measured = pd.DataFrame(summaries).set_index('cellid')
    units = subset.units.copy().set_index('cellid')
    for column in measured:
        units[column] = measured[column]
    units = units.reset_index().drop(columns=['source_unit_id'], errors='ignore')
    units['siteid'] = units['session_id']
    day_pairs = []
    for _, group in units.groupby('recording_group', sort=True):
        for a, z in itertools.combinations(sorted(set(group.session_id)), 2):
            day_pairs.append({'siteid_a': a, 'siteid_b': z, 'days_apart': 0,
                              'observation_kind': 'disjoint temporal parts of one recording'})
    sites, freq_axes = clusters.unit_sites(units), {}
    labels = subset.labels.set_index('cellid')
    for day, day_extras in sorted((extras or {}).items()):
        if 'tuning_freqs' not in day_extras:
            continue
        rows = dict(zip((str(u) for u in day_extras['tuning_unit']), day_extras['tuning_ftc']))
        for cellid in labels.index[labels.source_session.astype(str) == str(day)]:
            if cellid not in arrays:
                continue
            freq_axes[sites[str(cellid)]] = np.asarray(day_extras['tuning_freqs']).tolist()
            source_unit = str(labels.at[cellid, 'source_unit_id'])
            if source_unit in rows:
                arrays[cellid]['tuning'] = np.asarray(rows[source_unit])
    peaks = dict(zip(units.cellid, units.peak_chan))
    unit_map = {c: UnitData(cellid=c, siteid=sites[str(c)], peak_chan=int(peaks[c]), **a) for c, a in arrays.items()}
    pair_table = display_pairs(result, sites, pair_evidence=subset.pairs.copy())
    day_pairs = apply_run_provenance(complete_day_pairs(day_pairs, pair_table), result.provenance)
    for dp in day_pairs:
        dp['days_apart'] = 0
    dataset_id = dataset.metadata['dataset_id']
    return RunData(
        mode='same_day', result=result, words=WORDS, title=f'{dataset_id} · benchmark',
        info=f'{len(result.units)} units · {len(result.unmatched)} unmatched · {result_partition(result)} partition',
        info_tip=f'{dataset_id} · temporal parts of one recording, not different days\nReference identities hidden.',
        unit_table=units, unit_map=unit_map, pair_table=pair_table, day_pair_list=tuple(day_pairs),
        geometries=subset.geometry, geometry_sources={s: 'measured geometry' for s in subset.geometry},
        day_names={}, freq_axes=freq_axes,
        min_fingerprint_repeats=dataset.metadata.get('pair_options', {}).get('min_repeats', 4),
        parts=frozenset({'isi', 'acg'} | ({'tuning'} if freq_axes else set())))
# [AGENT EDIT END]


def load(selection, dataset, result, extras: dict) -> RunData:
    """``run_data`` checked against the selection's hash."""
    return run_data(dataset, result, input_manifest_sha256=selection.sha256(), extras=extras)


def header(data: RunData, cellids, row) -> str:
    """The pair on screen, else the selected units."""
    if row is not None:
        return f'{row.cellid_a}   vs   {row.cellid_b}'
    return 'Selected units: ' + ', '.join(cellids) if cellids else 'No units selected'


def merge_refusal(data: RunData, members) -> Optional[str]:
    """A same-day run is inspect-only: nothing merges, so nothing is refused here."""
    return None


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the plots end with the drift panel (drift-view Task 4) | date: 2026-10-08]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the interval histogram and the autocorrelogram are two plots (browse-same-day Task 5) | date: 2026-10-12]
SAME_DAY = Mode(id='same_day', layout_name='viewer/same_day_layout.json', curates=False, exports=False,
                words=WORDS, plots=('waveform', 'footprint', 'probe', 'wfmap', 'isi', 'acg', 'fingerprint',
                                    'tuning', 'mixture', 'drift'),
                load=load, header=header, merge_refusal=merge_refusal)
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
