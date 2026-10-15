# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the public events_csv reader: trials, stimuli and baseline windows from one CSV file (multiphy project folder Task 6; fix round 1: only the last trial stop moves the recording extent) | date: 2026-09-25]
"""The ``events_csv`` reader: trials, stimuli and baseline windows from one CSV file.

Columns: ``kind`` (``trial``, ``stimulus`` or ``baseline``); either ``start``/``stop`` in
0-based samples on the sort's clock or ``start_s``/``stop_s`` in seconds, never both in one
file (``stop`` is exclusive); ``trial`` (an integer of at least 1) on trial and stimulus rows;
``condition`` on stimulus rows, the complete condition label. Without trial rows each
stimulus is its own trial. The analysis window becomes ``[0, stop of the last trial)``, and
that stop alone may extend a sample count taken from the last spike; baselines never change
the recording extent.
The reader needs the day from the sort reader and returns a new ``DaySource``.
"""
from __future__ import annotations

import csv
import dataclasses
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .base import DaySource

KINDS = ('trial', 'stimulus', 'baseline')
SAMPLE_PAIR = ('start', 'stop')
SECONDS_PAIR = ('start_s', 'stop_s')


def _cell(row: dict, column: str) -> str:
    value = row.get(column)
    return value.strip() if isinstance(value, str) else ''


def _read_rows(path: Path) -> list:
    """``(line, row)`` for every row; ``utf-8-sig`` drops the byte-order mark spreadsheets write."""
    with open(path, newline='', encoding='utf-8-sig') as handle:
        reader = csv.DictReader(handle)
        return [(reader.line_num, row) for row in reader]


def _pair(rows: list) -> tuple:
    """The column pair the file uses for times; both pairs in one file are refused."""
    uses = {pair: any(_cell(row, c) for _, row in rows for c in pair) for pair in (SAMPLE_PAIR, SECONDS_PAIR)}
    if all(uses.values()):
        raise ValueError('events_csv: use start/stop or start_s/stop_s, not both.')
    return SECONDS_PAIR if uses[SECONDS_PAIR] else SAMPLE_PAIR


def _window(row: dict, line: int, pair: tuple, sample_rate: float) -> tuple:
    values = []
    for column in pair:
        text = _cell(row, column)
        if not text:
            raise ValueError(f'events_csv: row {line} has no {column}.')
        if pair == SAMPLE_PAIR:
            try:
                values.append(int(text))
            except ValueError:
                raise ValueError(f'events_csv: row {line} {column} must be an integer, not {text!r}.') from None
        else:
            try:
                seconds = float(text)
            except ValueError:
                seconds = math.nan
            if not math.isfinite(seconds):
                raise ValueError(f'events_csv: row {line} {column} must be a finite number, not {text!r}.')
            values.append(int(round(seconds * sample_rate)))
    start, stop = values
    if start < 0 or stop <= start:
        raise ValueError(f'events_csv: row {line} window [{start}, {stop}) is empty or starts before sample 0.')
    return start, stop


def _trial_id(row: dict, line: int) -> int:
    text = _cell(row, 'trial')
    if not text:
        raise ValueError(f'events_csv: row {line} has no trial.')
    try:
        trial = int(text)
    except ValueError:
        trial = 0
    if trial < 1:
        raise ValueError(f'events_csv: row {line} trial must be an integer of at least 1, not {text!r}.')
    return trial


def _ordered(windows: list) -> bool:
    """True when each window starts at or after the previous one stops."""
    return all(b[0] >= a[1] for a, b in zip(windows, windows[1:]))


def _windows_array(windows: list) -> np.ndarray:
    return np.asarray([[start, stop] for start, stop in windows], dtype=np.int64).reshape(-1, 2)


class EventsCsvReader:
    """Reads the ``events`` role from ``options['events_path']``."""
    id = 'events_csv'
    version = 1
    roles = frozenset({'events'})

    def files(self, options: dict) -> dict:
        return {'events': Path(options['events_path'])}

    def read(self, files: dict, options: dict, day: Optional[DaySource] = None) -> DaySource:
        if day is None:
            raise ValueError('events_csv: needs the day from the sort reader.')
        rows = _read_rows(files['events'])
        for _, row in rows:
            kind = row.get('kind')
            kind = kind.strip() if isinstance(kind, str) else kind
            if kind not in KINDS:
                raise ValueError(f'events_csv: unknown kind {kind}.')
        pair = _pair(rows)

        trial_rows, stimulus_rows, baselines = [], [], []
        for line, row in rows:
            kind = row['kind'].strip()
            window = _window(row, line, pair, day.sample_rate)
            if kind == 'trial':
                trial_rows.append((window, _trial_id(row, line)))
            elif kind == 'stimulus':
                condition = _cell(row, 'condition')
                if not condition:
                    raise ValueError(f'events_csv: row {line} has no condition.')
                stimulus_rows.append((window, condition, line, row))
            else:
                baselines.append(window)

        if trial_rows:
            trials = [window for window, _ in trial_rows]
            if not _ordered(trials):
                raise ValueError('events_csv: trials are out of order or overlap.')
            by_id: dict = {}
            for window, trial in trial_rows:
                if trial in by_id:
                    raise ValueError(f'events_csv: trial {trial} appears twice.')
                by_id[trial] = window
            stimuli = []
            for (start, stop), condition, line, row in stimulus_rows:
                trial = _trial_id(row, line)
                bounds = by_id.get(trial)
                if bounds is None or start < bounds[0] or stop > bounds[1]:
                    raise ValueError(f'events_csv: stimulus at {start} lies outside trial {trial}.')
                stimuli.append({'stimulus': condition, 'start': start, 'stop': stop, 'trial': trial})
            stimuli.sort(key=lambda s: s['start'])
        else:  # each stimulus is its own trial, numbered from 1 in start order
            ordered = sorted(stimulus_rows, key=lambda s: s[0][0])
            trials = [window for window, _, _, _ in ordered]
            if not _ordered(trials):
                raise ValueError('events_csv: trials are out of order or overlap.')
            stimuli = [{'stimulus': condition, 'start': start, 'stop': stop, 'trial': i + 1}
                       for i, ((start, stop), condition, _, _) in enumerate(ordered)]

        durations: dict = {}
        for s in stimuli:
            if durations.setdefault(s['stimulus'], s['stop'] - s['start']) != s['stop'] - s['start']:
                raise ValueError(f"events_csv: condition {s['stimulus']} has two durations.")
        if not _ordered(baselines):
            raise ValueError('events_csv: baseline windows are out of order or overlap.')

        # Only the last trial's stop moves the recording extent; baseline windows never do.
        analysis_window, n_samples, clock = day.analysis_window, day.n_samples, day.clock
        if trials:
            stop = int(trials[-1][1])
            analysis_window = (0, stop)
            if stop > n_samples:
                if day.clock.get('n_samples_from') != 'last_spike':
                    raise ValueError(f'events_csv: events end at {stop}, after the recording ends at {n_samples}.')
                n_samples, clock = stop, dict(day.clock, n_samples_from='events')
        return dataclasses.replace(
            day, n_samples=n_samples, analysis_window=analysis_window, clock=clock,
            trials=_windows_array(trials), stimuli=tuple(stimuli), baseline_windows=_windows_array(baselines))


READER = EventsCsvReader()
# [AGENT EDIT END]
