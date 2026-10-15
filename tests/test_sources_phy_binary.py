# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the phy and binary readers read a KiloSort or phy folder and its raw file (multiphy project folder Task 7) | date: 2026-09-25]
"""The phy reader reads a synthetic phy day; binary takes the sample count from the raw file and refuses what it cannot read."""
from __future__ import annotations

import dataclasses
import gzip
import re
import shutil

import numpy as np
import pytest

from nems_lbhb.unit_match.sources import base
from nems_lbhb.unit_match.sources.base import clear_readers_for_tests, read_day, reader_ids
from nems_lbhb.unit_match.sources.binary import READER as BINARY
from nems_lbhb.unit_match.sources.phy import READER as PHY

from .synthetic_day import write_day


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


def _phy(options):
    return PHY.read(PHY.files(options), options)


def _sort_and_raw(options):
    files = {'sort': PHY.files(options), 'raw': BINARY.files(options)}
    return read_day({'sort': 'phy', 'raw': 'binary'}, files, options)


def _gzip_copy(sd, name='raw.bin.gz'):
    target = sd.root / name
    with open(sd.raw_path, 'rb') as source, gzip.open(target, 'wb', compresslevel=1) as dest:
        shutil.copyfileobj(source, dest)
    return target


def test_phy_reads_synthetic_day(sd, options):
    day = _phy(options)
    assert sorted(day.spikes) == sd.units
    assert day.clock['n_samples_from'] == 'last_spike'
    assert day.provenance['su_labels_from'] == 'cluster_group.tsv'

    last_spike = int(np.load(sd.sort_dir / 'spike_times.npy').max())  # the mua cluster counts too
    assert day.n_samples == last_spike + 1
    assert day.analysis_window == (0, last_spike + 1)
    assert day.clock == {'sample_index_origin': 'zero_based_voltage_row',
                         'method': 'phy spike_times.npy sample indices', 'n_samples_from': 'last_spike'}
    for unit in sd.units:
        assert day.spikes[unit].dtype == np.int64
        assert np.array_equal(day.spikes[unit], sd.spikes[unit])
    assert day.su_type == {unit: 'SU' for unit in sd.units}
    assert day.sample_rate == 30000.0
    assert np.array_equal(day.geometry, sd.geometry)
    assert day.provenance == {
        'reader': 'phy@1', 'su_labels_from': 'cluster_group.tsv',
        'params': {'dat_path': str(sd.raw_path), 'n_channels_dat': 8, 'dtype': 'int16', 'offset': 0,
                   'sample_rate': 30000.0, 'hp_filtered': False}}
    assert day.raw is None and len(day.trials) == 0


def test_phy_kslabel_fallback(sd, options):
    (sd.sort_dir / 'cluster_group.tsv').unlink()
    assert PHY.files(options)['cluster_labels'].name == 'cluster_KSLabel.tsv'
    day = _phy(options)
    assert day.provenance['su_labels_from'] == 'cluster_KSLabel.tsv'
    assert sorted(day.spikes) == sd.units


def test_phy_templates_unwhitened(sd, options):
    day = _phy(options)
    assert np.allclose(day.templates['D1:0'], sd.templates[0], atol=1e-4) is True
    for k, unit in enumerate(sd.units):
        assert day.templates[unit].dtype == np.float32
        assert day.templates[unit].shape == (82, 8)
        assert np.allclose(day.templates[unit], sd.templates[k], atol=1e-4)

    # A cluster whose spikes carry two templates gets their spike-count-weighted mean.
    clusters = np.load(sd.sort_dir / 'spike_clusters.npy')
    spike_templates = np.load(sd.sort_dir / 'spike_templates.npy')
    carried = np.flatnonzero(clusters == 1)[:50]
    spike_templates[carried] = 4
    np.save(sd.sort_dir / 'spike_templates.npy', spike_templates)
    expected = (150 * sd.templates[1] + 50 * sd.templates[4]) / 200
    assert np.allclose(_phy(options).templates['D1:1'], expected, atol=1e-4)


def test_params_not_executed(sd, options):
    params = sd.sort_dir / 'params.py'
    params.write_text(params.read_text() + 'import os\n')
    with pytest.raises(ValueError, match='literal assignments'):
        _phy(options)
    with pytest.raises(ValueError, match='literal assignments'):
        PHY.raw_proposal(options)


def test_raw_proposal(sd, options):
    assert PHY.raw_proposal(options) == {'path': str(sd.raw_path), 'dtype': '<i2', 'offset': 0,
                                         'n_channels': 8, 'sample_rate': 30000.0}

    # A list names its first entry, and a relative path is relative to sort_dir.
    params = sd.sort_dir / 'params.py'
    lines = [line for line in params.read_text().splitlines() if not line.startswith('dat_path')]
    params.write_text('\n'.join(["dat_path = ['../raw.bin', 'other.bin']"] + lines) + '\n')
    assert PHY.raw_proposal(options)['path'] == str(sd.sort_dir / '..' / 'raw.bin')
    params.write_text('\n'.join(lines) + '\n')
    assert PHY.raw_proposal(options) is None


def test_binary_sets_count_from_size(sd, options):
    day = _sort_and_raw(options)
    assert day.n_samples == 600000
    assert day.clock['n_samples_from'] == 'raw_size'
    assert day.analysis_window == (0, 600000)
    assert day.raw == {'path': str(sd.raw_path), 'n_channels': 8, 'dtype': '<i2', 'sample_rate': 30000.0,
                       'uv_per_bit': 0.195, 'gzip': False}

    # A .gz file keeps a sample count that the sort reader took from a clock.
    sort_day = _phy(options)
    clocked = dataclasses.replace(sort_day, n_samples=600000, analysis_window=(0, 600000),
                                  clock=dict(sort_day.clock, n_samples_from='clock'))
    gz_options = dict(options, raw_path=str(_gzip_copy(sd)))
    gz_day = BINARY.read(BINARY.files(gz_options), gz_options, clocked)
    assert gz_day.n_samples == 600000 and gz_day.clock['n_samples_from'] == 'clock'
    assert gz_day.raw['gzip'] is True and gz_day.raw['path'] == gz_options['raw_path']


def _temp_wh(sd, options):
    target = sd.root / 'temp_wh.dat'
    shutil.copyfile(sd.raw_path, target)
    options['raw_path'] = str(target)


def _no_uv_per_bit(sd, options):
    del options['uv_per_bit']


def _byte_offset(sd, options):
    options['raw_offset'] = 16


def _channel_count(sd, options):
    options['raw_n_channels'] = 9


def _gzip_last_spike(sd, options):
    options['raw_path'] = str(_gzip_copy(sd))


def _partial_row(sd, options):
    target = sd.root / 'partial.bin'
    target.write_bytes(sd.raw_path.read_bytes() + b'\0')
    options['raw_path'] = str(target)


def _short_file(sd, options):
    target = sd.root / 'short.bin'
    target.write_bytes(sd.raw_path.read_bytes()[:1000 * 8 * 2])
    options['raw_path'] = str(target)


REFUSALS = [
    (_temp_wh, "binary: temp_wh.dat is KiloSort's whitened copy, not the acquired stream."),
    (_no_uv_per_bit, 'binary: uv_per_bit is required.'),
    (_byte_offset, 'binary: a byte offset is not supported.'),
    (_channel_count, 'binary: 9 raw channels differ from 8 geometry channels.'),
    (_gzip_last_spike, 'binary: a gzip raw file needs the sample count from its sort reader.'),
    (_partial_row, 'binary: the raw file size is not a whole number of samples.'),
    (_short_file, 'binary: the raw file holds fewer samples than the sort.'),
]


@pytest.mark.parametrize('setup, message', REFUSALS, ids=[setup.__name__.lstrip('_') for setup, _ in REFUSALS])
def test_binary_refusals(sd, options, setup, message):
    setup(sd, options)
    with pytest.raises(ValueError, match=re.escape(message)) as info:
        _sort_and_raw(options)
    assert str(info.value) == message


def test_only_public_readers_without_plugins(monkeypatch):
    monkeypatch.setattr(base, '_entry_points', lambda: ())
    assert reader_ids() == ('binary', 'events_csv', 'phy')
# [AGENT EDIT END]
