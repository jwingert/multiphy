# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: file identities catch changed files and retargeted links (multiphy project folder Task 5) | date: 2026-09-25]
"""A recorded file identity catches a changed or missing file and a link re-pointed after recording."""
from __future__ import annotations

import os
import shutil

import pytest

from nems_lbhb.unit_match.project.identity import (FileIdentity, SourceChanged, SourceMissing, record, retargeted,
                                                    sha256_file, verify)


def _write(path, data=b'0123456789abcdef' * 64):
    path.write_bytes(data)
    return path


def _flip_one_byte_keep_mtime(path):
    """Change one byte in place; size and ``mtime_ns`` stay as they were."""
    before = os.stat(path)
    data = bytearray(path.read_bytes())
    data[5] ^= 0xFF
    path.write_bytes(bytes(data))
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    after = os.stat(path)
    assert (after.st_size, after.st_mtime_ns) == (before.st_size, before.st_mtime_ns)


def test_record_resolves_link(tmp_path):
    f = _write(tmp_path / 'f')
    l = tmp_path / 'l'
    l.symlink_to(f)
    identity = record('sort', str(l))
    assert identity.given == os.path.abspath(l)
    assert identity.resolved == os.path.realpath(f)
    assert identity.sha256 == sha256_file(f)
    assert identity.sha256_from == 'init'
    assert identity.size_bytes == os.stat(f).st_size
    assert identity.mtime_ns == os.stat(f).st_mtime_ns


def test_rehash_catches_same_size_edit(tmp_path):
    path = _write(tmp_path / 'spike_times.npy')
    identity = record('sort', str(path))
    _flip_one_byte_keep_mtime(path)
    with pytest.raises(SourceChanged, match='sort source changed'):
        verify(identity, rehash=True)


def test_no_rehash_checks_size_and_mtime(tmp_path):
    path = _write(tmp_path / 'spike_times.npy')
    identity = record('sort', str(path))
    _flip_one_byte_keep_mtime(path)
    assert verify(identity, rehash=False) is None
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: final review fix round 1: without a re-hash, verify still checks size and mtime_ns, each on its own | date: 2026-09-25]
    before = os.stat(path)
    # mtime_ns alone: the same bytes, touched a second later.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns + 10 ** 9))
    with pytest.raises(SourceChanged, match='sort source changed'):
        verify(identity, rehash=False)
    # size alone: one byte appended, mtime_ns restored.
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert verify(identity, rehash=False) is None
    with open(path, 'ab') as stream:
        stream.write(b'x')
    os.utime(path, ns=(before.st_atime_ns, before.st_mtime_ns))
    assert os.stat(path).st_mtime_ns == identity.mtime_ns and os.stat(path).st_size == identity.size_bytes + 1
    with pytest.raises(SourceChanged, match='sort source changed'):
        verify(identity, rehash=False)
    # [AGENT EDIT END]


def test_missing_source(tmp_path):
    path = _write(tmp_path / 'events.csv')
    identity = record('events', str(path))
    path.unlink()
    with pytest.raises(SourceMissing, match='events source missing'):
        verify(identity, rehash=False)
    with pytest.raises(SourceMissing, match='events source missing'):
        record('events', str(path))


def test_retargeted_link_keeps_recorded_file(tmp_path):
    f = _write(tmp_path / 'f')
    l = tmp_path / 'l'
    l.symlink_to(f)
    identity = record('sort', str(l))
    assert retargeted(identity) is False
    copy = tmp_path / 'copy'
    shutil.copyfile(f, copy)
    # the copy's mtime differs from the recorded one, so a verify that read the link would fail
    os.utime(copy, ns=(identity.mtime_ns, identity.mtime_ns + 10 ** 9))
    l.unlink()
    l.symlink_to(copy)
    assert retargeted(identity) is True
    assert verify(identity, rehash=True) is None


def test_identity_roundtrip(tmp_path):
    for identity in (record('events', str(_write(tmp_path / 'events.csv'))),
                     record('raw', str(_write(tmp_path / 'raw.bin')), hash_contents=False)):
        assert FileIdentity.from_dict(identity.to_dict()) == identity
    assert identity.sha256 is None and identity.sha256_from is None
    assert list(identity.to_dict()) == ['role', 'given', 'resolved', 'size_bytes', 'mtime_ns', 'sha256',
                                        'sha256_from']
# [AGENT EDIT END]
