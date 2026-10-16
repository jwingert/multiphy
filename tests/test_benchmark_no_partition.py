"""Halves archives without a partition: new archives omit it, old ones keep it."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from multiphy.benchmark import build_dataset, load_dataset, save_dataset
from multiphy.evaluation import _labels
from tests.test_benchmark import source


def _build(recordings):
    return build_dataset(recordings, dataset_id='d', extraction_options={'before': 2, 'after': 3})


def test_build_without_partition():
    ds = _build([source(partition=None)])
    assert len(ds.labels) == 4
    assert 'partition' not in ds.labels.columns
    assert 'partition' not in ds.metadata['sources'][0]


def test_mixed_partition_refused():
    with pytest.raises(ValueError, match='Every source needs a partition'):
        _build([source(session='s1'), source(session='s2', partition=None)])


def test_save_load_without_partition(tmp_path):
    ds = _build([source(partition=None)])
    path = save_dataset(ds, tmp_path / 'halves', cache_root=tmp_path)
    loaded = load_dataset(path)
    assert len(loaded.units) == 4
    assert 'partition' not in loaded.labels.columns


def test_labels_without_partition():
    ds = _build([source(partition=None)])
    _, _, partition_info = _labels(SimpleNamespace(units=ds.units), ds.labels, None, ('well',))
    assert partition_info['partition_counts'] == {}
    assert partition_info['n_source_recordings'] == 1


def test_labels_expected_partition_refused():
    ds = _build([source(partition=None)])
    with pytest.raises(ValueError, match='no partition column'):
        _labels(SimpleNamespace(units=ds.units), ds.labels, 'test', ('well',))
