# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the halves rule and full-day rows pooled from halves match the old reuse path (multiphy project folder Task 9) | date: 2026-09-25]
"""The halves rule, and full-day rows pooled from the halves without a second voltage read."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from nems_lbhb.unit_match.benchmark import load_dataset
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 16 fix round 2: _spike_features for the baseline test moved here from test_docs_project_folder.py | date: 2026-09-25]
from nems_lbhb.unit_match.extract.full_day import (_spike_features, full_day_rows, load_full_day,
                                                   pool_day_row, save_full_day)
# [AGENT EDIT END]
from nems_lbhb.unit_match.extract.split import halves
from nems_lbhb.unit_match.extract_benchmark import extract_manifest
from nems_lbhb.unit_match.multiday_reuse import reuse_day_summary

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: final review fix round 1: the pooled row is checked against an oracle that does not run pool_day_row | date: 2026-09-25]
from .full_day_oracle import assert_pooled_from_halves
# [AGENT EDIT END]
from .synthetic_day import OPTIONS, write_day, write_legacy_inputs

WINDOW = (0, 597000)


def _old_path(sd, tmp_path):
    manifest = write_legacy_inputs(sd, tmp_path / 'legacy', dataset_id='legacy', well='G')
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    spec = json.loads(manifest.read_text())['recordings'][0]
    return manifest, archive, spec, load_dataset(archive)


def _full(sd, spec, dataset):
    return full_day_rows(dataset, sd.spikes, spec, day=sd.day, group='G', session_date=sd.date,
                         analysis_window=WINDOW, windows=np.asarray(spec['windows'], dtype=np.int64),
                         max_spikes=OPTIONS['max_spikes'])


def test_halves_trial_rule(tmp_path):
    sd = write_day(tmp_path, 'D1')
    events = pd.read_csv(sd.events_path)
    trials = events.loc[events.kind == 'trial', ['start', 'stop']].to_numpy(dtype=np.int64)
    h = halves(trials, WINDOW)
    assert h['rule'] == 'trial_start'
    assert h['boundary'] == 300000
    assert h['windows'].dtype == np.int64
    assert h['windows'].tolist() == [[0, 300000], [300000, 597000]]


def test_halves_midpoint_rule():
    h = halves(np.zeros((0, 2), dtype=np.int64), (0, 1000))
    assert h['boundary'] == 500
    assert h['rule'] == 'sample_midpoint'
    assert h['windows'].tolist() == [[0, 500], [500, 1000]]


def test_empty_half_refused():
    with pytest.raises(ValueError, match='empty'):
        halves(np.array([[0, 400]], dtype=np.int64), (0, 1000))


def test_pool_day_row_matches_reuse(tmp_path):
    sd = write_day(tmp_path, 'D1')
    manifest, archive, spec, dataset = _old_path(sd, tmp_path)
    summary = reuse_day_summary(archive, manifest, spec, session_date=sd.date,
                                analysis_window=WINDOW, extraction_options=OPTIONS)
    cellids = dataset.labels.loc[dataset.labels.source_unit_id == 'D1:0', 'cellid']
    rows = sorted([dataset.features[c] for c in cellids], key=lambda row: row['start_sample'])
    pooled = pool_day_row(rows, sd.spikes['D1:0'], spec, analysis_window=WINDOW,
                          windows=np.asarray(spec['windows']), max_spikes=OPTIONS['max_spikes'])
    np.testing.assert_equal(pooled, summary['features']['D1-D1:0'])
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: final review fix round 1: reuse_day_summary now calls pool_day_row, so the row is also checked against an oracle that does not | date: 2026-09-25]
    assert_pooled_from_halves(pooled, rows, sd.spikes['D1:0'], spec, WINDOW)
    # [AGENT EDIT END]


def test_weak_unit_row_ineligible(tmp_path):
    sd = write_day(tmp_path, 'D1', weak_unit=True)
    _, _, spec, dataset = _old_path(sd, tmp_path)
    full = _full(sd, spec, dataset)
    table = full['table'].set_index('cellid')
    assert len(table) == 4
    assert bool(table.loc['D1:4', 'eligible']) is False
    assert table.loc['D1:4', 'exclusion_reason'] == 'Too few spikes in temporal split.'
    assert table.loc[['D1:0', 'D1:1', 'D1:2'], 'eligible'].tolist() == [True, True, True]
    assert table.loc[['D1:0', 'D1:1', 'D1:2'], 'exclusion_reason'].tolist() == ['', '', '']
    assert sorted(full['features']) == ['D1:0', 'D1:1', 'D1:2']
    assert list(full['geometry']) == ['D1']
    np.testing.assert_array_equal(full['geometry']['D1'], sd.geometry)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - full_day_rows refuses halves rows whose windows or spike trains do not match the call, as reuse_day_summary does (multiphy project folder Task 9) | date: 2026-09-25]
def test_full_day_rows_refuses_mismatched_halves(tmp_path):
    sd = write_day(tmp_path, 'D1')
    _, _, spec, dataset = _old_path(sd, tmp_path)
    windows = np.asarray(spec['windows'], dtype=np.int64)

    def call(spikes=sd.spikes, analysis_window=WINDOW, windows=windows):
        return full_day_rows(dataset, spikes, spec, day=sd.day, group='G', session_date=sd.date,
                             analysis_window=analysis_window, windows=windows,
                             max_spikes=OPTIONS['max_spikes'])

    # The rows were extracted on [[0, 300000], [300000, 597000]].
    with pytest.raises(ValueError, match='windows'):
        call(windows=np.array([[0, 290000], [290000, 597000]], dtype=np.int64))
    # The windows do not cover the analysis window the full-day row is computed over.
    with pytest.raises(ValueError, match='windows'):
        call(analysis_window=(0, 598000))
    # A spike train other than the one the halves were extracted from.
    stimuli = [(s['start'], s['stop']) for s in spec['stimuli']]
    train = sd.spikes['D1:0']
    inside = np.zeros(len(train), dtype=bool)
    for start, stop in stimuli:
        inside |= (train >= start) & (train < stop)
    assert 0 < inside.sum() < ((train >= WINDOW[0]) & (train < WINDOW[1])).sum()
    with pytest.raises(ValueError, match='spike train'):
        call(spikes=dict(sd.spikes, **{'D1:0': train[inside]}))
    # The unchanged call still passes.
    assert sorted(call()['features']) == ['D1:0', 'D1:1', 'D1:2']
# [AGENT EDIT END]


def test_full_day_roundtrip(tmp_path):
    sd = write_day(tmp_path, 'D1', weak_unit=True)
    _, _, spec, dataset = _old_path(sd, tmp_path)
    full = _full(sd, spec, dataset)
    target = tmp_path / 'full'
    metadata = {'day': 'D1', 'halves_rule': 'trial_start', 'halves_boundary': 300000}
    full_id = save_full_day(full, target, metadata=metadata)
    assert full_id == hashlib.sha256((target / 'manifest.json').read_bytes()).hexdigest()
    manifest = json.loads((target / 'manifest.json').read_text())
    assert manifest['kind'] == 'multiphy_full_day'
    assert sorted(manifest['sha256']) == ['extras.npz', 'features.json', 'features.npz', 'units.csv']
    loaded = load_full_day(target)
    pd.testing.assert_frame_equal(loaded['table'], full['table'])
    np.testing.assert_equal(loaded['features'], full['features'])
    np.testing.assert_equal(loaded['geometry'], full['geometry'])
    assert loaded['metadata'] == metadata
    assert loaded['extras'] == {}
    with pytest.raises(FileExistsError):
        save_full_day(full, target, metadata=metadata)
    units = target / 'units.csv'
    units.write_text(units.read_text().replace('Too few spikes', 'Too many spikes'))
    with pytest.raises(ValueError, match='hash mismatch: units.csv'):
        load_full_day(target)
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: Task 16 fix round 2: pin project_folder.md's extraction claims here, not in test_docs_project_folder.py; baselines are clipped to the measured window (moved from there), and a full-day waveform pools up to max_spikes sampled spikes per half | date: 2026-09-25]
def test_baseline_counts_only_inside_the_measured_window():
    """project_folder.md, section 5: a baseline window past the measured window adds only its part inside."""
    spikes = np.array([950, 1200], dtype=np.int64)
    feature = _spike_features(spikes, (0, 1000), 1000.0, {'baseline_windows': [[900, 1500]]}, 0.01)
    assert feature['baseline_exposure_samples'] == 100
    assert feature['baseline_spike_count'] == 1
    assert feature['spont'] == 10.0


def test_full_day_waveform_pools_both_halves_samples(tmp_path):
    """project_folder.md, section 7: at most max_spikes are sampled per half, and the full-day
    waveform pools both halves' samples, so it can hold more than max_spikes; the counts are
    kept in each feature's summary_reuse in the saved full/ folder."""
    sd = write_day(tmp_path, 'D1')
    _, _, spec, dataset = _old_path(sd, tmp_path)
    full = _full(sd, spec, dataset)
    save_full_day(full, tmp_path / 'full', metadata={'day': 'D1'})
    loaded = load_full_day(tmp_path / 'full')['features']
    limit = OPTIONS['max_spikes']
    assert sorted(loaded) == ['D1:0', 'D1:1', 'D1:2']
    for feature in loaded.values():
        reuse = feature['summary_reuse']
        assert len(reuse['half_spike_counts']) == 2 and max(reuse['half_spike_counts']) <= limit
        assert reuse['fresh_full_mean_limit'] == limit
        assert (feature['n_waveform_spikes'] == len(feature['waveform_spike_samples'])
                == sum(reuse['half_spike_counts']) == reuse['full_mean_selected_count'])
    assert max(feature['n_waveform_spikes'] for feature in loaded.values()) > limit
# [AGENT EDIT END]
