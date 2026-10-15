# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a .gz raw file is hashed during the extraction pass, and its recorded sha256 is checked or stored (multiphy project folder Task 12) | date: 2026-09-25]
"""A ``.gz`` raw file's sha256 comes from the extraction pass, at no extra read.

The raw pass hashes the stored (compressed) bytes as ``gzip`` reads them; ``extract_day``
records that digest with ``sha256_from`` ``'extract'``, or refuses a day whose recorded digest
differs. A flat raw file is never hashed, and hashing never changes a snippet.
"""
from __future__ import annotations

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - watch the raw file's opens, reads and seeks and every sha256 input during extraction (multiphy project folder Task 12) | date: 2026-09-25]
import builtins
from contextlib import contextmanager
import dataclasses
import hashlib
import io
import os
from types import SimpleNamespace
# [AGENT EDIT END]

import numpy as np
import pytest

from nems_lbhb.unit_match.extract import pipeline
from nems_lbhb.unit_match.extract.pipeline import extract_day
from nems_lbhb.unit_match.extract.raw_pass import FilteredSnippetReader, HashingReader, plan_snippets
from nems_lbhb.unit_match.project.days import add_day
from nems_lbhb.unit_match.project.folder import Project
from nems_lbhb.unit_match.project.identity import SourceChanged, sha256_file
from nems_lbhb.unit_match.sources.base import clear_readers_for_tests, register_reader
from nems_lbhb.unit_match.sources.phy import PhyReader
from nems_lbhb.unit_match.waveform_preprocessing import SharedWaveformPreprocessor

from .synthetic_day import N_CHANNELS, OPTIONS, SAMPLE_RATE, write_day

N_SAMPLES = 600000
GZ_READERS = {'sort': 'phy_clocked', 'events': 'events_csv', 'raw': 'binary'}
DAT_READERS = {'sort': 'phy', 'events': 'events_csv', 'raw': 'binary'}
WINDOWS = np.array([[0, 300000], [300000, 597000]], dtype=np.int64)


class ClockedPhyReader(PhyReader):
    """``phy`` with the sample count from ``options['n_samples']``, as a gzip raw file needs."""
    id = 'phy_clocked'

    def read(self, files, options, day=None):
        source = super().read(files, options, day)
        return dataclasses.replace(source, n_samples=int(options['n_samples']),
                                   clock=dict(source.clock, n_samples_from='clock'))


@pytest.fixture(autouse=True)
def clocked_reader():
    clear_readers_for_tests()
    register_reader(ClockedPhyReader())
    yield
    clear_readers_for_tests()


def _project(tmp_path, sd, readers: dict, **options) -> Project:
    project = Project.create(tmp_path / 'lab', 'P', group='G')
    reader_options = {'day': 'D1', 'sort_dir': str(sd.sort_dir), 'events_path': str(sd.events_path),
                      'uv_per_bit': sd.uv_per_bit}
    reader_options.update(options)
    add_day(project, 'D1', readers=dict(readers), reader_options=reader_options, date=sd.date, date_source='user')
    return project


def _gz_project(tmp_path):
    sd = write_day(tmp_path / 'data', 'D1', gzip_raw=True)
    return sd, _project(tmp_path, sd, GZ_READERS, raw_path=str(sd.gz_path), n_samples=N_SAMPLES)


def _extract(project, tmp_path) -> dict:
    return extract_day(project, 'D1', scratch_root=tmp_path / 'scratch')


def _raw_file(project) -> dict:
    return project.read()['days']['D1']['files']['raw']


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - the tests see every open, read and seek of the raw file and every sha256 input, so a second traversal or a hashed flat file fails them (multiphy project folder Task 12) | date: 2026-09-25]
class _WatchedFile:
    """A file opened on the watched path: every byte read and every seek is counted."""
    def __init__(self, stream, watch):
        self._stream, self._watch = stream, watch

    def _counted(self, data):
        self._watch.bytes_read += len(data)
        if data:
            self._watch.chunks.append(bytes(data))
        return data

    def read(self, n=-1):
        return self._counted(self._stream.read(n))

    def read1(self, n=-1):
        return self._counted(self._stream.read1(n))

    def readinto(self, buffer):
        n = self._stream.readinto(buffer)
        self._counted(memoryview(buffer)[:n])
        return n

    def seek(self, offset, whence=0):
        self._watch.seeks.append((offset, whence))
        return self._stream.seek(offset, whence)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._stream.close()

    def __getattr__(self, name):
        return getattr(self._stream, name)


class _WatchedDigest:
    """A sha256 whose every input is kept in ``hashed``."""
    def __init__(self, digest, hashed):
        self._digest, self._hashed = digest, hashed

    def update(self, data):
        self._hashed.add(bytes(data))
        self._digest.update(data)

    def __getattr__(self, name):
        return getattr(self._digest, name)


@contextmanager
def _watch_raw(monkeypatch, raw_path):
    """While the block runs, count each open, read and seek of ``raw_path`` and keep every sha256 input."""
    watch = SimpleNamespace(opens=0, bytes_read=0, seeks=[], chunks=[], hashed=set())
    target = os.path.realpath(raw_path)
    real_open, real_sha256 = builtins.open, hashlib.sha256

    def watched_open(file, *args, **kwargs):
        stream = real_open(file, *args, **kwargs)
        if isinstance(file, (str, bytes, os.PathLike)) and os.path.realpath(os.fsdecode(file)) == target:
            watch.opens += 1
            return _WatchedFile(stream, watch)
        return stream

    def watched_sha256(*args, **kwargs):
        data = kwargs.pop('string', args[0] if args else b'')
        digest = _WatchedDigest(real_sha256(**kwargs), watch.hashed)
        digest.update(data)
        return digest

    with monkeypatch.context() as patch:
        for module in (builtins, io):  # pathlib opens through io.open
            patch.setattr(module, 'open', watched_open)
        patch.setattr(hashlib, 'sha256', watched_sha256)
        yield watch


def test_gz_raw_hashed_during_pass(tmp_path, monkeypatch):
    sd, project = _gz_project(tmp_path)
    assert _raw_file(project)['sha256'] is None
    expected = sha256_file(sd.gz_path)  # the test's own read, before the watch starts
    with _watch_raw(monkeypatch, sd.gz_path) as watch:
        _extract(project, tmp_path)
    raw = _raw_file(project)
    assert raw['sha256'] == expected
    assert raw['sha256_from'] == 'extract'
    # One traversal: extraction opened the stored file once and read each of its bytes once, in order.
    opens, seeks, bytes_read = watch.opens, watch.seeks, watch.bytes_read
    assert (opens, seeks, bytes_read) == (1, [], sd.gz_path.stat().st_size)
# [AGENT EDIT END]
    # The other files keep the hashes add_day recorded.
    files = project.read()['days']['D1']['files']
    assert {files[key]['sha256_from'] for key in files if key != 'raw'} == {'init'}


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - a flat raw file is read once by the pass and none of its bytes reach a sha256 (multiphy project folder Task 12) | date: 2026-09-25]
def test_dat_raw_not_hashed(tmp_path, monkeypatch):
    sd = write_day(tmp_path / 'data', 'D1')
    project = _project(tmp_path, sd, DAT_READERS, raw_path=str(sd.raw_path))
    with _watch_raw(monkeypatch, sd.raw_path) as watch:
        _extract(project, tmp_path)
    raw = _raw_file(project)
    assert raw['sha256'] is None
    assert raw['sha256_from'] is None
    # The pass opened the flat file once, and no byte it read reached a sha256.
    opens, bytes_read = watch.opens, watch.bytes_read
    bytes_hashed = sum(len(chunk) for chunk in watch.chunks if chunk in watch.hashed)
    assert opens == 1 and bytes_read > 0
    assert bytes_hashed == 0
# [AGENT EDIT END]
    assert project.read()['days']['D1']['features']['full'] is not None


def test_hashing_leaves_snippets_unchanged(tmp_path):
    sd = write_day(tmp_path / 'data', 'D1', gzip_raw=True)
    samples = plan_snippets(sd.spikes, WINDOWS, **OPTIONS)
    assert len(samples)

    def reader(name, hash_raw):
        return FilteredSnippetReader(sd.gz_path, samples, n_channels=N_CHANNELS, n_samples=sd.n_samples,
                                     directory=tmp_path / name, sample_rate=SAMPLE_RATE,
                                     preprocessing=SharedWaveformPreprocessor(SAMPLE_RATE, N_CHANNELS).metadata(),
                                     voltage_scale=sd.uv_per_bit, hash_raw=hash_raw)

    with_hash, without_hash = reader('with', True), reader('without', False)
    try:
        before, after = OPTIONS['before'], OPTIONS['after']
        equal = [np.array_equal(with_hash(int(s) - before, int(s) + after),
                                without_hash(int(s) - before, int(s) + after)) for s in samples]
        assert all(equal) and len(equal) == len(samples)
        assert with_hash.raw_sha256 == sha256_file(sd.gz_path)
        assert without_hash.raw_sha256 is None
    finally:
        with_hash.close()
        without_hash.close()


def test_nonsequential_read_disables_hash():
    # Sequential reads, and a seek to the current position, keep the digest.
    sequential = HashingReader(io.BytesIO(b'abcdef'), hashlib.sha256())
    assert sequential.read(2) == b'ab'
    assert sequential.hexdigest() == hashlib.sha256(b'ab').hexdigest()
    sequential.seek(2)
    assert sequential.read() == b'cdef'
    assert sequential.hexdigest() == hashlib.sha256(b'abcdef').hexdigest()

    reader = HashingReader(io.BytesIO(b'abcdef'), hashlib.sha256())
    assert reader.read(2) == b'ab'
    reader.seek(0)
    assert reader.read() == b'abcdef'
    assert reader.hexdigest() is None

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - a forward, relative or end-relative seek that moves also ends the digest (multiphy project folder Task 12) | date: 2026-09-25]
    for offset, whence in ((4, 0), (1, 1), (-1, 2)):
        reader = HashingReader(io.BytesIO(b'abcdef'), hashlib.sha256())
        assert reader.read(2) == b'ab'
        reader.seek(offset, whence)
        reader.read()
        assert reader.hexdigest() is None, (offset, whence)
    # [AGENT EDIT END]


def _set_raw_sha256(project, sha256, sha256_from=None) -> None:
    def change(manifest):
        manifest['days']['D1']['files']['raw'].update(sha256=sha256, sha256_from=sha256_from)

    project.update(change)


def test_changed_gz_refused(tmp_path, monkeypatch):
    sd, project = _gz_project(tmp_path)
    _set_raw_sha256(project, '0' * 64)
    with pytest.raises(SourceChanged, match='raw source changed'):
        _extract(project, tmp_path)
    assert project.read()['days']['D1']['features'] == {'halves': None, 'full': None}
    assert _raw_file(project)['sha256'] == '0' * 64
    day_dir = project.day_dir('D1')
    assert not (day_dir / 'features').exists()
    assert not [path for path in day_dir.iterdir() if path.name.startswith('.features-')]

    # A sha256 recorded after the pass, while the full day is built, is checked again in the record.
    _set_raw_sha256(project, None)
    real_save_full_day = pipeline.save_full_day

    def save_full_day(*args, **kwargs):
        _set_raw_sha256(project, '1' * 64, 'verify')
        return real_save_full_day(*args, **kwargs)

    monkeypatch.setattr(pipeline, 'save_full_day', save_full_day)
    with pytest.raises(SourceChanged, match='raw source changed'):
        _extract(project, tmp_path)
    assert project.read()['days']['D1']['features'] == {'halves': None, 'full': None}
    assert _raw_file(project)['sha256'] == '1' * 64
    assert (day_dir / 'features').is_dir()  # refused at the record, after the publish: left unrecorded
    monkeypatch.undo()

    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fix round 1 - Do 5: a recorded sha256 that matches the pass is stored with sha256_from 'extract' (multiphy project folder Task 12) | date: 2026-09-25]
    # A recorded sha256 that matches the pass is stored with sha256_from 'extract'.
    _set_raw_sha256(project, sha256_file(sd.gz_path), 'verify')
    _extract(project, tmp_path)
    raw = _raw_file(project)
    assert (raw['sha256'], raw['sha256_from']) == (sha256_file(sd.gz_path), 'extract')
    # [AGENT EDIT END]
    assert project.read()['days']['D1']['features']['full'] is not None
# [AGENT EDIT END]
