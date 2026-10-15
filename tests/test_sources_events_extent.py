# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 regression tests: only the last trial stop moves the recording extent (multiphy project folder Task 6) | date: 2026-09-25]
"""Only the stop of the last trial can extend or refuse the recording extent; baselines never do."""
from __future__ import annotations

import dataclasses

import pytest

from nems_lbhb.unit_match.sources.base import clear_readers_for_tests, read_day, register_reader

from .test_sources_events import _Reader, _stub_day


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


def _events(tmp_path, lines, n_samples_from='last_spike'):
    path = tmp_path / 'events.csv'
    path.write_text('\n'.join(lines) + '\n')
    sort_day = _stub_day()
    sort_day = dataclasses.replace(sort_day, clock=dict(sort_day.clock, n_samples_from=n_samples_from))
    register_reader(_Reader('stub.sort', {'sort'}, lambda files, options, day: sort_day))
    return read_day({'sort': 'stub.sort', 'events': 'events_csv'},
                    {'sort': {}, 'events': {'events': path}}, {'events_path': str(path)})


def test_baseline_past_last_trial_keeps_extent(tmp_path):
    day = _events(tmp_path, [
        'kind,start,stop,trial,condition',
        'trial,0,800,1,',
        'stimulus,100,200,1,A',
        'baseline,900,1500,,',
    ])
    assert day.n_samples == 1000
    assert day.clock['n_samples_from'] == 'last_spike'
    assert day.analysis_window == (0, 800)
    assert day.baseline_windows.tolist() == [[900, 1500]]


def test_baseline_only_file_keeps_extent(tmp_path):
    day = _events(tmp_path, [
        'kind,start,stop,trial,condition',
        'baseline,0,500,,',
        'baseline,900,1500,,',
    ])
    assert day.n_samples == 1000
    assert day.clock == _stub_day().clock
    assert day.analysis_window == (0, 1000)
    assert day.trials.shape == (0, 2)
    assert day.baseline_windows.tolist() == [[0, 500], [900, 1500]]


def test_trial_past_known_end_refused(tmp_path):
    with pytest.raises(ValueError, match=r'^events_csv: events end at 1500, after the recording ends at 1000\.$'):
        _events(tmp_path, [
            'kind,start,stop,trial,condition',
            'trial,0,1500,1,',
            'stimulus,100,200,1,A',
        ], n_samples_from='raw_size')


def test_baseline_past_known_end_kept(tmp_path):
    day = _events(tmp_path, [
        'kind,start,stop,trial,condition',
        'trial,0,800,1,',
        'stimulus,100,200,1,A',
        'baseline,900,1500,,',
    ], n_samples_from='raw_size')
    assert day.n_samples == 1000
    assert day.clock['n_samples_from'] == 'raw_size'
# [AGENT EDIT END]
