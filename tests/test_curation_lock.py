# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: test module docstring: the history is replaced whole on each write (run-curation Task 2, D14) | date: 2026-10-04]
"""One writer per run: the history is replaced whole under a lock; a refused or failed action writes nothing more."""
# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: history file, lock and controller tests (run-curation Task 2) | date: 2026-10-04]
from __future__ import annotations

import csv
import io
import json
import os
import socket
import subprocess
import sys
from types import SimpleNamespace

import pytest

from multiphy import curation
from multiphy.clusters import cluster_id

T = cluster_id(['D1:0', 'D2:0'])


def _result():
    return SimpleNamespace(tracks=[['D1:0', 'D2:0']], unmatched=['D1:1'])


def _dir(tmp_path):
    return tmp_path / 'curation' / 'runs' / 'r1'


def _open(tmp_path):
    return curation.Curation.open(_dir(tmp_path), _result())


def _data_lines(d):
    path = d / 'history.csv'
    if not path.exists():
        return 0
    return len(path.read_text().splitlines()) - 1


def _write_lock(d, pid, host, token):
    d.mkdir(parents=True, exist_ok=True)
    (d / 'history.lock').write_text(json.dumps({'pid': pid, 'host': host, 'token': token, 'at': 'x'}))


def _lock(d):
    return json.loads((d / 'history.lock').read_text())


def test_history_roundtrip(tmp_path):
    a = _open(tmp_path)
    a.do('accept', cluster_id=T)
    a.close()
    b = _open(tmp_path)
    assert b.state.tracks[T]['status'] == 'accepted'
    d = _dir(tmp_path)
    assert _data_lines(d) == 1
    header = (d / 'history.csv').read_text().splitlines()[0].split(',')
    assert header == ['op_id', 'at', 'curator', 'operation', 'cluster_id', 'parent_ids', 'members',
                      'target_op', 'cellid_a', 'cellid_b', 'verdict', 'notes']


def test_second_open_is_read_only(tmp_path):
    a = _open(tmp_path)
    b = _open(tmp_path)
    assert a.read_only is False
    assert b.read_only is True
    with pytest.raises(PermissionError):
        b.do('accept', cluster_id=T)
    assert _data_lines(_dir(tmp_path)) == 0


def test_stale_local_lock_cleared(tmp_path):
    proc = subprocess.Popen([sys.executable, '-c', 'pass'])  # subprocess.run keeps no pid
    proc.wait()  # finished and reaped: its pid is gone
    _write_lock(_dir(tmp_path), proc.pid, socket.gethostname(), 'any')
    a = _open(tmp_path)
    assert a.read_only is False
    assert _lock(_dir(tmp_path))['token'] == a.token


def test_remote_lock_take_over(tmp_path):
    d = _dir(tmp_path)
    _write_lock(d, 1, 'elsewhere', 't')
    a = _open(tmp_path)
    assert a.read_only is True
    assert a.remote_lock is True
    a.take_over()
    assert a.read_only is False
    lock = _lock(d)
    assert (lock['pid'], lock['host'], lock['token']) == (os.getpid(), socket.gethostname(), a.token)


def test_local_live_lock_has_no_take_over(tmp_path):
    _open(tmp_path)
    b = _open(tmp_path)
    assert b.read_only is True
    with pytest.raises(PermissionError):
        b.take_over()


def test_write_after_take_over_refused(tmp_path):
    d = _dir(tmp_path)
    a = _open(tmp_path)
    _write_lock(d, 1, 'elsewhere', 't')
    with pytest.raises(PermissionError, match='taken over'):
        a.do('accept', cluster_id=T)
    assert a.read_only is True
    assert a.taken_over is True
    assert _data_lines(d) == 0


def test_refused_action_writes_nothing(tmp_path):
    a = _open(tmp_path)
    with pytest.raises(ValueError):
        a.do('split', cluster_id=T, members=['D1:0', 'D2:0'])
    assert not (_dir(tmp_path) / 'history.csv').exists()


def test_read_only_close_keeps_lock(tmp_path):
    _open(tmp_path)
    b = _open(tmp_path)
    b.close()
    assert (_dir(tmp_path) / 'history.lock').exists() is True


def test_close_removes_own_lock_only(tmp_path):
    d = _dir(tmp_path)
    a = _open(tmp_path)
    _write_lock(d, 1, 'elsewhere', 't')
    a.close()
    assert (d / 'history.lock').exists() is True


def test_closed_controller_cannot_write(tmp_path):
    d = _dir(tmp_path)
    a = _open(tmp_path)
    a.close()
    b = _open(tmp_path)
    assert b.read_only is False
    before = (d / 'history.lock').read_text()
    with pytest.raises(PermissionError):
        a.do('accept', cluster_id=T)
    assert (d / 'history.lock').read_text() == before


def test_take_over_refused_when_now_local(tmp_path):
    d = _dir(tmp_path)
    _write_lock(d, 1, 'elsewhere', 't')
    b = _open(tmp_path)
    assert b.read_only is True and b.remote_lock is True
    _write_lock(d, os.getpid(), socket.gethostname(), 't')
    with pytest.raises(PermissionError):
        b.take_over()
    assert _lock(d)['token'] == 't'


def test_open_failure_releases_lock(tmp_path):
    d = _dir(tmp_path)
    d.mkdir(parents=True)
    (d / 'history.csv').write_text('a,b\n')
    with pytest.raises(ValueError):
        _open(tmp_path)
    assert not (d / 'history.lock').exists()


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an existing history with no header is refused, never read as a fresh run (run-curation Task 2, fix round 1 after atomic rewrite) | date: 2026-10-04]
def test_empty_history_file_refused(tmp_path):
    d = _dir(tmp_path)
    d.mkdir(parents=True)
    (d / 'history.csv').write_bytes(b'')
    with pytest.raises(ValueError, match='unknown history columns'):
        _open(tmp_path)
    assert not (d / 'history.lock').exists()
    assert (d / 'history.csv').read_bytes() == b''
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an open that fails after another writer took the lock leaves that writer's lock in place (run-curation Task 2, fix round 2) | date: 2026-10-04]
def test_open_failure_keeps_successor_lock(tmp_path, monkeypatch):
    d = _dir(tmp_path)

    def taken_then_fail(curation_dir):
        _write_lock(d, 1, 'other-host', 'successor')
        raise ValueError('bad history')

    monkeypatch.setattr(curation, 'read_history', taken_then_fail)
    with pytest.raises(ValueError, match='bad history'):
        _open(tmp_path)
    assert _lock(d)['token'] == 'successor'
# [AGENT EDIT END]


# [AGENT EDIT END]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed rewrite leaves the history unchanged and no temp file; the rewrite keeps the file mode; a last row without a newline is kept (run-curation Task 2, D14) | date: 2026-10-04]
def test_write_failure_turns_read_only(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    a = _open(tmp_path)
    a.do('accept', cluster_id=T)
    before = (d / 'history.csv').read_bytes()

    def fail(fd):
        raise OSError('disk')

    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError):
        a.do('accept', cluster_id=cluster_id(['D1:1']))
    assert a.read_only is True
    assert a.write_failed == 'disk'
    assert (d / 'history.csv').read_bytes() == before
    assert list(d.glob('*.tmp')) == []
    with pytest.raises(PermissionError):
        a.do('accept', cluster_id=cluster_id(['D1:1']))


def test_write_keeps_file_mode(tmp_path):
    d = _dir(tmp_path)
    a = _open(tmp_path)
    a.do('accept', cluster_id=T)
    os.chmod(d / 'history.csv', 0o664)
    a.do('accept', cluster_id=cluster_id(['D1:1']))
    assert os.stat(d / 'history.csv').st_mode & 0o777 == 0o664
    assert _data_lines(d) == 2


def test_last_row_without_newline_kept(tmp_path):
    d = _dir(tmp_path)
    d.mkdir(parents=True)
    row = curation.make_row('accept', cluster_id=T)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=curation.COLUMNS, lineterminator='\n')
    writer.writeheader()
    writer.writerow(row)
    (d / 'history.csv').write_text(buf.getvalue().rstrip('\n'))
    assert [r['op_id'] for r in curation.read_history(d)] == [row['op_id']]
    a = _open(tmp_path)
    a.do('accept', cluster_id=cluster_id(['D1:1']))
    assert _data_lines(d) == 2
    assert a.state.tracks[T]['status'] == 'accepted'
    assert a.state.tracks[cluster_id(['D1:1'])]['status'] == 'accepted'
    assert a.state.skipped == []
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a write-failed controller keeps the lock; a failed lock write leaves no lock (run-curation Task 2, fix round 1) | date: 2026-10-04]
def test_write_failed_close_keeps_lock(tmp_path, monkeypatch):
    a = _open(tmp_path)

    def fail(fd):
        raise OSError('disk')

    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError):
        a.do('accept', cluster_id=T)
    a.close()
    assert (_dir(tmp_path) / 'history.lock').exists() is True


def test_lock_write_failure_leaves_no_lock(tmp_path, monkeypatch):
    def fail(fd, data):
        raise OSError('disk full')

    monkeypatch.setattr(os, 'write', fail)
    with pytest.raises(OSError):
        _open(tmp_path)
    assert not (_dir(tmp_path) / 'history.lock').exists()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a lock whose descriptor fails to close is removed (run-curation Task 2, review r1) | date: 2026-10-04]
def test_lock_close_failure_leaves_no_lock(tmp_path, monkeypatch):
    real_close = os.close
    calls = []

    def fail(fd):
        calls.append(fd)
        real_close(fd)
        raise OSError('close')

    monkeypatch.setattr(os, 'close', fail)
    with pytest.raises(OSError):
        _open(tmp_path)
    monkeypatch.undo()
    assert len(calls) == 1
    assert not (_dir(tmp_path) / 'history.lock').exists()
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed acquisition removes only the file it created; a successor's lock stays (run-curation Task 2, review r2) | date: 2026-10-04]
def test_failed_acquisition_keeps_successor_lock(tmp_path, monkeypatch):
    lock = _dir(tmp_path) / 'history.lock'
    successor = json.dumps({'pid': 1, 'host': 'elsewhere', 'token': 'succ'})
    real_write = os.write

    def short(fd, data):
        os.remove(lock)  # another controller takes over the incomplete lock
        lock.write_text(successor)
        return real_write(fd, data[:1])

    monkeypatch.setattr(os, 'write', short)
    with pytest.raises(OSError):
        _open(tmp_path)
    monkeypatch.undo()
    assert lock.read_text() == successor
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a multi-line row is refused (run-curation Task 2, fix round 2) | date: 2026-10-04]
def test_multiline_row_refused(tmp_path):
    a = _open(tmp_path)
    with pytest.raises(ValueError):
        a.do('split', cluster_id=T, members='[\n"D1:0"\n]')
    assert not (_dir(tmp_path) / 'history.csv').exists()
    assert a.read_only is False

# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a failed re-read after a write turns read-only; a bad line never swallows the next (run-curation Task 2, D14) | date: 2026-10-04]
def test_reread_failure_after_write_goes_read_only(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    a = _open(tmp_path)

    def fail(curation_dir):
        raise OSError('reread')

    monkeypatch.setattr(curation, 'read_history', fail)
    with pytest.raises(OSError):
        a.do('accept', cluster_id=T)
    assert a.read_only is True
    assert a.write_failed
    assert _data_lines(d) == 1


def test_bad_quoted_line_does_not_swallow_next(tmp_path):
    d = _dir(tmp_path)
    d.mkdir(parents=True)
    row = curation.make_row('accept', cluster_id=T)
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=curation.COLUMNS, lineterminator='\n')
    writer.writeheader()
    header = buf.getvalue()
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=curation.COLUMNS, lineterminator='\n').writerow(row)
    (d / 'history.csv').write_text(header + '"bad\n' + buf.getvalue())
    state = curation.replay(_result().tracks, _result().unmatched, curation.read_history(d))
    assert state.tracks[T]['status'] == 'accepted'
    assert len([s for s in state.skipped if s[1].startswith('malformed')]) == 1
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a row replay would skip as malformed is refused before it is written (run-curation Task 2, fix round 3) | date: 2026-10-04]
def test_row_replay_would_skip_is_refused(tmp_path):
    a = _open(tmp_path)
    with pytest.raises(ValueError):
        a.do('accept', cluster_id=T, parent_ids='not JSON')
    assert not (_dir(tmp_path) / 'history.csv').exists()
    assert a.state.tracks[T]['status'] != 'accepted'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a row the history reader cannot read back is refused before it is written (run-curation Task 2, fix round 4) | date: 2026-10-04]
def test_row_reader_cannot_read_is_refused(tmp_path):
    a = _open(tmp_path)
    with pytest.raises(ValueError):
        a.do('accept', cluster_id=T, notes='x' * (csv.field_size_limit() + 1))
    assert not (_dir(tmp_path) / 'history.csv').exists()
    assert a.state.tracks[T]['status'] != 'accepted'
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a take-over whose re-read or replay fails stays read-only and removes the lock it wrote (run-curation Task 2, D34) | date: 2026-10-04]
def test_take_over_replay_failure_stays_read_only(tmp_path, monkeypatch):
    d = _dir(tmp_path)
    _write_lock(d, 1, 'elsewhere', 't')
    a = _open(tmp_path)
    assert a.read_only is True and a.remote_lock is True

    def fail(curation_dir):
        raise OSError('reread')

    monkeypatch.setattr(curation, 'read_history', fail)
    with pytest.raises(OSError):
        a.take_over()
    assert a.read_only is True
    assert a.write_failed
    assert not (d / 'history.lock').exists()
# [AGENT EDIT END]
