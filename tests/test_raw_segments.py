# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a day's raw segments read as one file, a segment of the wrong length stops the pass, and raw-role segment files are recorded unhashed (Plan C Task 2) | date: 2026-10-01]
"""A day's raw segments stream as one file on one clock.

``SnippetReader`` and ``FilteredSnippetReader`` take a list of ``[path, n_samples]`` segments
in place of one path and read the same snippets as from one file of their concatenation. A
segment that holds another number of samples than declared stops the pass, naming the
segment. ``add_day`` records every key ``raw`` or ``raw_<k>`` under the ``raw`` role, unhashed.
"""
from __future__ import annotations

import gzip
import hashlib
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from multiphy.extract.raw_pass import FilteredSnippetReader, SnippetReader
from multiphy.project.days import add_day
from multiphy.project.folder import Project
from multiphy.sources.base import clear_readers_for_tests, register_reader

N_CHANNELS = 4
SPIKES = np.array([200, 1300], dtype=np.int64)
PREPROCESSING = {'profile': 'shared_unwhitened_v1'}
SAMPLE_RATE = 1000.0


@pytest.fixture
def raw(tmp_path):
    """int16 data of 1800 samples x 4 channels, as gz segments of 1000 + 800 and one flat file."""
    data = np.random.default_rng(0).integers(-2000, 2000, size=(1800, N_CHANNELS)).astype('<i2')
    gz1, gz2, flat = tmp_path / 'seg1.dat.gz', tmp_path / 'seg2.dat.gz', tmp_path / 'all.dat'
    with gzip.open(gz1, 'wb') as stream:
        stream.write(data[:1000].tobytes())
    with gzip.open(gz2, 'wb') as stream:
        stream.write(data[1000:].tobytes())
    data.tofile(flat)
    return data, gz1, gz2, flat


def test_segments_read_like_one_file(raw, tmp_path):
    data, gz1, gz2, flat = raw
    segments = [[gz1, 1000], [gz2, 800]]
    reader = SnippetReader(segments, SPIKES, n_channels=N_CHANNELS, n_samples=1800, directory=tmp_path / 'plain')
    try:
        assert np.array_equal(reader(159, 241), data[159:241])
        assert np.array_equal(reader(1259, 1341), data[1259:1341])
    finally:
        reader.close()

    kwargs = dict(n_channels=N_CHANNELS, n_samples=1800, sample_rate=SAMPLE_RATE, preprocessing=PREPROCESSING)
    by_segments = FilteredSnippetReader(segments, SPIKES, directory=tmp_path / 'segments', **kwargs)
    by_file = FilteredSnippetReader(flat, SPIKES, directory=tmp_path / 'flat', **kwargs)
    try:
        for start in (159, 1259):
            assert np.array_equal(by_segments(start, start + 82), by_file(start, start + 82)) is True
    finally:
        by_segments.close()
        by_file.close()


def test_segment_length_mismatch_refused(raw, tmp_path):
    _, gz1, gz2, _ = raw
    with pytest.raises(ValueError) as caught:
        SnippetReader([[gz1, 1001], [gz2, 799]], SPIKES, n_channels=N_CHANNELS, n_samples=1800,
                      directory=tmp_path / 'snips')
    assert str(gz1) in str(caught.value)


class SegmentedReader:
    """A sort-and-raw reader whose raw voltage is two segment files."""
    id = 'test.segmented'
    version = 1
    roles = frozenset({'sort', 'raw'})

    def files(self, options: dict) -> dict:
        folder = Path(options['folder'])
        return {'raw_1': folder / 'a.dat.gz', 'raw_2': folder / 'b.dat.gz', 'sort': folder / 'sort.npz'}

    def read(self, files: dict, options: dict, day: Optional[object] = None):
        raise AssertionError('add_day reads no data')


def test_segment_files_recorded_as_raw_unhashed(tmp_path):
    folder = tmp_path / 'in'
    folder.mkdir()
    for name in ('a.dat.gz', 'b.dat.gz', 'sort.npz'):
        (folder / name).write_bytes(name.encode('utf-8'))
    clear_readers_for_tests()
    try:
        register_reader(SegmentedReader())
        project = Project.create(tmp_path / 'lab', 'P')
        add_day(project, 'D1', readers={'sort': 'test.segmented', 'raw': 'test.segmented'},
                reader_options={'folder': str(folder)}, date='2026-01-05', date_source='user')
    finally:
        clear_readers_for_tests()
    files = project.read()['days']['D1']['files']
    assert sorted(files) == ['raw_1', 'raw_2', 'sort']
    for key in ('raw_1', 'raw_2'):
        assert (files[key]['role'], files[key]['sha256'], files[key]['sha256_from']) == ('raw', None, None)
    assert files['sort']['role'] == 'sort'
    assert files['sort']['sha256'] == hashlib.sha256(b'sort.npz').hexdigest()
# [AGENT EDIT END]
