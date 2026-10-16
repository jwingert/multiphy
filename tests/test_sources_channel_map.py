# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: phy's channel_map.npy survives into DaySource and binary refuses a non-identity map (multiphy project folder Task 7, fix round 1) | date: 2026-09-25]
"""A phy ``channel_map.npy`` reaches ``DaySource``; ``binary`` refuses raw columns that are not the geometry rows."""
from __future__ import annotations

import dataclasses
import gzip
import shutil

import numpy as np
import pytest

from multiphy.sources.base import clear_readers_for_tests, read_day
from multiphy.sources.binary import READER as BINARY
from multiphy.sources.phy import READER as PHY

from .synthetic_day import write_day

PERMUTED = np.array([1, 0, 2, 3, 4, 5, 6, 7], dtype=np.int32)
NOT_IDENTITY = "binary: the sort's channel map is not the identity; geometry row i must be raw column i."


@pytest.fixture(autouse=True)
def fresh_registry():
    clear_readers_for_tests()
    yield
    clear_readers_for_tests()


@pytest.fixture
def sd(tmp_path):
    return write_day(tmp_path, 'D1')


@pytest.fixture
def options(sd):
    return {'day': 'D1', 'sort_dir': str(sd.sort_dir), 'raw_path': str(sd.raw_path), 'uv_per_bit': 0.195}


def _sort_and_raw(options):
    files = {'sort': PHY.files(options), 'raw': BINARY.files(options)}
    return read_day({'sort': 'phy', 'raw': 'binary'}, files, options)


def test_no_channel_map_file(sd, options):
    assert 'channel_map' not in PHY.files(options)
    assert PHY.read(PHY.files(options), options).channel_map is None
    assert _sort_and_raw(options).n_samples == 600000


def test_identity_channel_map_accepted(sd, options):
    np.save(sd.sort_dir / 'channel_map.npy', np.arange(8, dtype=np.int32)[:, None])  # KiloSort writes (n, 1)
    assert PHY.files(options)['channel_map'] == sd.sort_dir / 'channel_map.npy'
    day = _sort_and_raw(options)
    assert day.channel_map.dtype == np.int64
    assert day.channel_map.tolist() == list(range(8))
    assert day.n_samples == 600000 and day.clock['n_samples_from'] == 'raw_size'


def test_permuted_channel_map_kept_without_raw(sd, options):
    np.save(sd.sort_dir / 'channel_map.npy', PERMUTED)
    day = PHY.read(PHY.files(options), options)  # the templates tier needs no raw columns
    assert day.channel_map.tolist() == PERMUTED.tolist()
    assert np.allclose(day.templates['D1:0'], sd.templates[0], atol=1e-4)


@pytest.mark.parametrize('gzip_raw', [False, True])
def test_permuted_channel_map_refused_with_raw(sd, options, gzip_raw):
    np.save(sd.sort_dir / 'channel_map.npy', PERMUTED)
    day = PHY.read(PHY.files(options), options)
    if gzip_raw:  # a gzip file needs a sample count that is not the last spike
        target = sd.root / 'raw.bin.gz'
        with open(sd.raw_path, 'rb') as source, gzip.open(target, 'wb', compresslevel=1) as dest:
            shutil.copyfileobj(source, dest)
        options = dict(options, raw_path=str(target))
        day = dataclasses.replace(day, n_samples=600000, clock=dict(day.clock, n_samples_from='clock'))
    with pytest.raises(ValueError) as exc:
        BINARY.read(BINARY.files(options), options, day)
    assert str(exc.value) == NOT_IDENTITY


@pytest.mark.parametrize('bad, message', [
    (np.arange(7), 'phy: channel_map.npy holds 7 channels; channel_positions.npy has 8.'),
    (np.arange(8, dtype=float), 'phy: channel_map.npy must hold integer channel indices.'),
])
def test_bad_channel_map_refused(sd, options, bad, message):
    np.save(sd.sort_dir / 'channel_map.npy', bad)
    with pytest.raises(ValueError) as exc:
        PHY.read(PHY.files(options), options)
    assert str(exc.value) == message
# [AGENT EDIT END]
