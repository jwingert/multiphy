# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: reader registry, read_day order and the events_csv rules (multiphy project folder Task 6) | date: 2026-09-25]
"""Readers are found through one registry, run in role order, and events_csv validates trials, stimuli and baselines."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from nems_lbhb.unit_match.sources import base
from nems_lbhb.unit_match.sources.base import (
    READER_ERRORS, DaySource, clear_readers_for_tests, read_day, reader_ids, register_reader,
)


class _Reader:
    def __init__(self, reader_id, roles, read):
        self.id = reader_id
        self.version = 1
        self.roles = frozenset(roles)
        self._read = read

    def files(self, options):
        return {}

    def read(self, files, options, day=None):
        return self._read(files, options, day)


def _stub_day():
    return DaySource(
        sample_rate=1000.0, n_samples=1000, analysis_window=(0, 1000),
        spikes={'D:1': np.array([10, 990], dtype=np.int64)}, su_type={'D:1': 'SU'},
        geometry=np.zeros((4, 2)),
        clock={'sample_index_origin': 'zero_based_voltage_row', 'method': 'stub', 'n_samples_from': 'last_spike'},
        provenance={'reader': 'stub.sort@1'},
    )


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


def _events(tmp_path, lines):
    path = tmp_path / 'events.csv'
    path.write_text('\n'.join(lines) + '\n')
    register_reader(_Reader('stub.sort', {'sort'}, lambda files, options, day: _stub_day()))
    return read_day({'sort': 'stub.sort', 'events': 'events_csv'},
                    {'sort': {}, 'events': {'events': path}}, {'events_path': str(path)})


TWO_TRIALS = [
    'kind,start,stop,trial,condition',
    'trial,0,400,1,',
    'baseline,0,100,,',
    'stimulus,100,200,1,A',
    'trial,400,800,2,',
    'baseline,400,500,,',
    'stimulus,500,600,2,A',
]


def test_events_in_samples(tmp_path):
    day = _events(tmp_path, TWO_TRIALS)
    assert day.trials.tolist() == [[0, 400], [400, 800]]
    assert day.trials.dtype == np.int64
    assert day.analysis_window == (0, 800)
    assert day.stimuli == ({'stimulus': 'A', 'start': 100, 'stop': 200, 'trial': 1},
                           {'stimulus': 'A', 'start': 500, 'stop': 600, 'trial': 2})
    assert day.baseline_windows.tolist() == [[0, 100], [400, 500]]
    assert day.n_samples == 1000 and day.clock['n_samples_from'] == 'last_spike'


def test_events_in_seconds(tmp_path):
    day = _events(tmp_path, [
        'kind,start_s,stop_s,trial,condition',
        'trial,0.0,0.4,1,',
        'baseline,0.0,0.1,,',
        'stimulus,0.1,0.2,1,A',
        'trial,0.4,0.8,2,',
        'baseline,0.4,0.5,,',
        'stimulus,0.5,0.6,2,A',
    ])
    assert day.trials.tolist() == [[0, 400], [400, 800]]


def test_mixed_pairs_refused(tmp_path):
    with pytest.raises(ValueError, match='not both'):
        _events(tmp_path, [
            'kind,start,stop,start_s,stop_s,trial,condition',
            'trial,0,400,0.0,0.4,1,',
        ])


def test_overlapping_trials_refused(tmp_path):
    with pytest.raises(ValueError, match='out of order or overlap'):
        _events(tmp_path, [
            'kind,start,stop,trial,condition',
            'trial,0,500,1,',
            'trial,400,800,2,',
        ])


def test_stimulus_outside_trial_refused(tmp_path):
    with pytest.raises(ValueError, match='outside trial 1'):
        _events(tmp_path, [
            'kind,start,stop,trial,condition',
            'trial,0,400,1,',
            'stimulus,350,450,1,A',
        ])


def test_condition_two_durations_refused(tmp_path):
    with pytest.raises(ValueError, match='condition A has two durations'):
        _events(tmp_path, [
            'kind,start,stop,trial,condition',
            'trial,0,400,1,',
            'stimulus,0,100,1,A',
            'trial,400,800,2,',
            'stimulus,400,520,2,A',
        ])


def test_no_trial_rows(tmp_path):
    day = _events(tmp_path, [
        'kind,start,stop,trial,condition',
        'stimulus,500,600,,B',
        'stimulus,100,200,,A',
        'stimulus,300,400,,A',
    ])
    assert len(day.trials) == len(day.stimuli) == 3
    assert day.trials.tolist() == [[100, 200], [300, 400], [500, 600]]
    assert [s['trial'] for s in day.stimuli] == [1, 2, 3]


def test_events_extend_last_spike_count(tmp_path):
    day = _events(tmp_path, [
        'kind,start,stop,trial,condition',
        'trial,0,1500,1,',
        'stimulus,100,200,1,A',
    ])
    assert day.n_samples == 1500
    assert day.clock['n_samples_from'] == 'events'
    assert day.analysis_window == (0, 1500)


class _EntryPoint:
    def __init__(self, name, load):
        self.name = name
        self.load = load


def test_entry_point_reader_registered(monkeypatch):
    fake = _Reader('x.fake', {'sort'}, lambda files, options, day: _stub_day())
    monkeypatch.setattr(base, '_entry_points', lambda: [_EntryPoint('fake', lambda: (lambda: fake))])
    assert 'x.fake' in reader_ids()
    assert base.get_reader('x.fake@1') is fake


def test_broken_entry_point_recorded(monkeypatch):
    def load():
        raise ImportError('nope')

    monkeypatch.setattr(base, '_entry_points', lambda: [_EntryPoint('bad', load)])
    ids = reader_ids()
    assert READER_ERRORS['bad'] == 'ImportError: nope'
    assert 'events_csv' in ids


def test_read_day_order(tmp_path):
    log = []

    def logging_reader(role):
        def read(files, options, day):
            log.append(role)
            assert (day is None) == (role == 'sort')
            return _stub_day() if day is None else day
        return _Reader(f'log.{role}', {role}, read)

    for role in ('events', 'raw', 'sort'):
        register_reader(logging_reader(role))
    read_day({'events': 'log.events', 'raw': 'log.raw', 'sort': 'log.sort'},
             {'sort': {}, 'raw': {}, 'events': {}}, {})
    assert log == ['sort', 'raw', 'events']
# [AGENT EDIT END]
