"""The synthetic phy day holds a consistent sort, a raw file and inputs for the old path."""
from __future__ import annotations

import json

import numpy as np

from multiphy.benchmark import load_dataset
from multiphy.extract_benchmark import extract_manifest
from multiphy.multiday_reuse import reuse_day_summary

from .synthetic_day import OPTIONS, write_day, write_legacy_inputs


def _old_path(sd, tmp_path):
    manifest = write_legacy_inputs(sd, tmp_path / 'legacy', dataset_id='legacy', well='G')
    archive = extract_manifest(manifest, root=tmp_path / 'cache')
    return manifest, archive, load_dataset(archive)


def test_phy_folder_consistent(tmp_path):
    sd = write_day(tmp_path, 'D1')
    assert len(np.load(sd.sort_dir / 'spike_times.npy')) == 750
    assert len(np.load(sd.sort_dir / 'spike_clusters.npy')) == 750
    assert sd.units == ['D1:0', 'D1:1', 'D1:2']


def test_raw_file_size(tmp_path):
    sd = write_day(tmp_path, 'D1')
    assert sd.raw_path.stat().st_size == 9600000


def test_old_path_extracts_synthetic_day(tmp_path):
    sd = write_day(tmp_path, 'D1')
    manifest, archive, ds = _old_path(sd, tmp_path)
    spec = json.loads(manifest.read_text())['recordings'][0]
    summary = reuse_day_summary(archive, manifest, spec, session_date='2026-01-01',
                                analysis_window=(0, 597000), extraction_options=OPTIONS)
    assert len(ds.labels) == 6
    assert len(summary['table']) == 3


def test_weak_unit_old_path_counts(tmp_path):
    sd = write_day(tmp_path, 'D1', weak_unit=True)
    _, _, ds = _old_path(sd, tmp_path)
    assert len(ds.labels) == 7
    assert [(e['source_unit_id'], e['split_index'], e['reason']) for e in ds.exclusions] == [
        ('D1:4', 1, 'Too few spikes in temporal split.')]
