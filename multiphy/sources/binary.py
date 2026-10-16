"""The ``binary`` reader: a flat interleaved raw file, optionally gzip, serves the ``raw`` role.

It never reads voltage. It checks its options against the sort's geometry and records them
in ``DaySource.raw``. Options: ``raw_path``; ``uv_per_bit``, required, since there is no
default and waveform amplitudes are always in µV; ``raw_dtype`` (default ``'<i2'``);
``raw_offset`` (default 0, the only value supported); ``raw_n_channels`` (default the
geometry's row count, the only value supported). The sort's ``channel_map``, when it has one,
must be the identity: geometry row i is raw column i. A flat file sets the sample count from its
size. A gzip file keeps the sort reader's count, so that count must not come from the last
spike. It refuses ``temp_wh.dat``, KiloSort's whitened copy.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Optional

import numpy as np

from .base import DaySource

WHITENED_COPY = 'temp_wh.dat'
DEFAULT_DTYPE = '<i2'


def _option(options: dict, key: str, default):
    """``options[key]``, or ``default`` when it is absent or ``None``."""
    value = options.get(key)
    return default if value is None else value


def _uv_per_bit(options: dict) -> float:
    value = options.get('uv_per_bit')
    if value is None:
        raise ValueError('binary: uv_per_bit is required.')
    try:
        scale = float(value)
    except (TypeError, ValueError):
        scale = math.nan
    if not (math.isfinite(scale) and scale > 0):
        raise ValueError(f'binary: uv_per_bit must be a positive number, not {value!r}.')
    return scale


class BinaryReader:
    """Reads the ``raw`` role from ``options['raw_path']``; needs the day from the sort reader."""
    id = 'binary'
    version = 1
    roles = frozenset({'raw'})

    def files(self, options: dict) -> dict:
        return {'raw': Path(options['raw_path'])}

    def read(self, files: dict, options: dict, day: Optional[DaySource] = None) -> DaySource:
        if day is None:
            raise ValueError('binary: needs the day from the sort reader.')
        path = Path(files['raw'])
        if path.name == WHITENED_COPY:
            raise ValueError("binary: temp_wh.dat is KiloSort's whitened copy, not the acquired stream.")
        uv_per_bit = _uv_per_bit(options)
        dtype = np.dtype(_option(options, 'raw_dtype', DEFAULT_DTYPE))
        if _option(options, 'raw_offset', 0) != 0:
            raise ValueError('binary: a byte offset is not supported.')
        m = int(day.geometry.shape[0])
        n = int(_option(options, 'raw_n_channels', m))
        if n != m:
            raise ValueError(f'binary: {n} raw channels differ from {m} geometry channels.')
        if day.channel_map is not None and not np.array_equal(day.channel_map, np.arange(m)):
            raise ValueError("binary: the sort's channel map is not the identity; geometry row i must be raw column i.")
        if not path.is_file():
            raise FileNotFoundError(f'binary: {path} is missing.')

        gzip = path.suffix == '.gz'
        n_samples, analysis_window, clock = day.n_samples, day.analysis_window, day.clock
        if gzip:  # the size of a gzip file says nothing about its sample count
            if day.clock.get('n_samples_from') == 'last_spike':
                raise ValueError('binary: a gzip raw file needs the sample count from its sort reader.')
        else:
            row_bytes = n * dtype.itemsize
            size = path.stat().st_size
            if size % row_bytes:
                raise ValueError('binary: the raw file size is not a whole number of samples.')
            rows = size // row_bytes
            if rows < day.n_samples:
                raise ValueError('binary: the raw file holds fewer samples than the sort.')
            if tuple(day.analysis_window) == (0, day.n_samples):
                analysis_window = (0, rows)
            n_samples, clock = rows, dict(day.clock, n_samples_from='raw_size')
        raw = {'path': str(path), 'n_channels': n, 'dtype': dtype.str, 'sample_rate': day.sample_rate,
               'uv_per_bit': float(uv_per_bit), 'gzip': gzip}
        return dataclasses.replace(day, n_samples=n_samples, analysis_window=analysis_window,
                                   clock=clock, raw=raw)


READER = BinaryReader()
