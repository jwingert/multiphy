"""Curation of one across-day run: one controller writes one history, replaced whole on each write (D14).

Each across-day run has one history of curator actions. Nothing in it is edited in place:
an action, an undo and a redo are each one appended row. The tracks a curator sees are never
stored; they are rebuilt on every open by replaying the history over the run's ``result.json``
tracks and unmatched units. A row that is malformed or no longer applies to that result is
skipped and listed, never deleted.
"""
from __future__ import annotations

import copy
import dataclasses
import getpass
import json
import time
import uuid
from typing import Optional, Sequence

from multiphy import store
from multiphy.clusters import cluster_id

COLUMNS = ('op_id', 'at', 'curator', 'operation', 'cluster_id', 'parent_ids', 'members',
           'target_op', 'cellid_a', 'cellid_b', 'verdict', 'notes')
ACTIONS = ('accept', 'reject', 'reopen', 'split', 'remove_member', 'merge', 'verdict')
OPERATIONS = ACTIONS + ('undo', 'redo')
LIST_COLUMNS = ('parent_ids', 'members')


@dataclasses.dataclass
class CurationState:
    tracks: dict
    rejected: dict
    verdicts: dict
    accepted_by: dict
    skipped: list
    undo_target: Optional[str]
    redo_target: Optional[str]


def _status(members) -> str:
    return 'proposed' if len(members) >= 2 else 'unmatched'


def _add(state: CurationState, members) -> str:
    """Hold a new undecided track; an id that was rejected is no longer rejected."""
    members = sorted(members)
    tid = cluster_id(members)
    state.tracks[tid] = {'members': members, 'status': _status(members)}
    state.rejected.pop(tid, None)
    state.accepted_by.pop(tid, None)
    return tid


def _remove(state: CurationState, tid: str) -> list:
    state.accepted_by.pop(tid, None)
    return state.tracks.pop(tid)['members']


def initial_state(tracks: Sequence[Sequence[str]], unmatched: Sequence[str]) -> CurationState:
    state = CurationState(tracks={}, rejected={}, verdicts={}, accepted_by={}, skipped=[],
                          undo_target=None, redo_target=None)
    for t in tracks:
        _add(state, [str(m) for m in t])
    for u in unmatched:
        _add(state, [str(u)])
    return state


def make_row(operation: str, **fields) -> dict:
    if operation not in OPERATIONS:
        raise ValueError(f'operation must be one of {OPERATIONS}, got {operation!r}')
    unknown = set(fields) - set(COLUMNS)
    if unknown:
        raise ValueError(f'unknown history columns: {sorted(unknown)}')
    row = {c: '' for c in COLUMNS}
    for key, value in fields.items():
        if key in LIST_COLUMNS and isinstance(value, (list, tuple)):
            value = json.dumps([str(v) for v in value])
        row[key] = '' if value is None else str(value)
    row['operation'] = operation
    row['notes'] = row['notes'].replace('\r\n', ' ').replace('\r', ' ').replace('\n', ' ')
    if operation == 'verdict':
        if row['cellid_a'] == row['cellid_b']:
            raise ValueError('a unit cannot be paired with itself')
        if row['verdict'] not in store.VERDICTS:
            raise ValueError(f"verdict must be one of {store.VERDICTS}, got {row['verdict']!r}")
        row['cellid_a'], row['cellid_b'] = store.canonical(row['cellid_a'], row['cellid_b'])
    # Wall-clock resolution can be coarser than consecutive actions (notably on
    # Windows). IDs must remain distinct even when their timestamps are equal.
    row['op_id'] = 'op_' + uuid.uuid4().hex
    row['at'] = time.strftime('%Y-%m-%d %H:%M:%S')
    row['curator'] = getpass.getuser()
    return row


def _list_cell(value: str) -> list:
    """A list cell: '' is no list, else a JSON list of strings."""
    if value == '':
        return []
    parsed = json.loads(value)
    if not isinstance(parsed, list) or not all(isinstance(v, str) for v in parsed):
        raise ValueError(f'{value!r} is not a JSON list of strings')
    return parsed


def _split(state: CurationState, tid: str, part: list) -> None:
    members = _remove(state, tid)
    _add(state, part)
    _add(state, [m for m in members if m not in part])


def apply(state: CurationState, row: dict) -> CurationState:
    """Return the state after one action row, or raise ValueError(<reason>); the input is untouched."""
    op = row['operation']
    if op not in ACTIONS:
        raise ValueError(f'{op!r} is not an action')
    new = copy.deepcopy(state)
    t = row['cluster_id']
    if op == 'verdict':
        units = {m for v in new.tracks.values() for m in v['members']}
        units.update(m for ms in new.rejected.values() for m in ms)
        for cellid in (row['cellid_a'], row['cellid_b']):
            if cellid not in units:
                raise ValueError(f'{cellid} is not a unit of this run')
        new.verdicts[store.canonical(row['cellid_a'], row['cellid_b'])] = {
            'verdict': row['verdict'], 'notes': row['notes'], 'curator': row['curator'], 'at': row['at']}
        return new
    if op == 'merge':
        parents = _list_cell(row['parent_ids'])
        if len(set(parents)) < 2 or len(set(parents)) != len(parents):
            raise ValueError('a merge needs 2 or more distinct tracks')
        for p in parents:
            if p not in new.tracks:
                raise ValueError(f'{p} is not a current track')
        union = [m for p in parents for m in _remove(new, p)]
        tid = _add(new, union)
        new.tracks[tid]['status'] = 'proposed'
        return new
    if op == 'reopen' and t not in new.tracks and t in new.rejected:
        members = new.rejected[t]
        for m in members:
            single = new.tracks.get(cluster_id([m]))
            if single is None or single['status'] != 'unmatched':
                raise ValueError('a unit of this track has joined another track')
        for m in members:
            _remove(new, cluster_id([m]))
        del new.rejected[t]
        _add(new, members)
        return new
    if t not in new.tracks:
        raise ValueError(f'{t} is not a current track')
    track = new.tracks[t]
    members = track['members']
    if op == 'accept':
        if track['status'] not in ('proposed', 'unmatched'):
            raise ValueError(f'{t} is already accepted')
        track['status'] = 'accepted'
        new.accepted_by[t] = {'at': row['at'], 'curator': row['curator']}
    elif op == 'reject':
        if len(members) < 2:
            raise ValueError('a single unit cannot be rejected')
        _remove(new, t)
        new.rejected[t] = list(members)
        for m in members:
            _add(new, [m])
    elif op == 'reopen':
        if track['status'] != 'accepted':
            raise ValueError(f'{t} is not accepted or rejected')
        track['status'] = _status(members)
        new.accepted_by.pop(t, None)
    elif op == 'split':
        part = _list_cell(row['members'])
        if not part or len(set(part)) != len(part) or not set(part) < set(members):
            raise ValueError('a split needs a non-empty proper subset of the track')
        _split(new, t, part)
    elif op == 'remove_member':
        part = _list_cell(row['members'])
        if len(part) != 1 or part[0] not in members:
            raise ValueError('remove_member needs one member of the track')
        if len(members) < 2:
            raise ValueError('a single unit has no member to remove')
        _split(new, t, part)
    return new


def _malformed(row, seen: set) -> Optional[str]:
    if not isinstance(row, dict) or set(row) != set(COLUMNS) or len(row) != len(COLUMNS):
        return 'columns are not the history columns'
    if any(v is None for v in row.values()):
        return 'a cell is missing or extra'
    if row['operation'] not in OPERATIONS:
        return f"unknown operation {row['operation']!r}"
    if not row['op_id']:
        return 'empty op_id'
    if row['op_id'] in seen:
        return f"repeated op_id {row['op_id']}"
    if row['operation'] == 'verdict' and row['verdict'] not in store.VERDICTS:
        return f"unknown verdict {row['verdict']!r}"
    for col in LIST_COLUMNS:
        try:
            _list_cell(row[col])
        except (ValueError, TypeError):
            return f'{col} is not a JSON list of strings'
    return None


def replay(tracks: Sequence[Sequence[str]], unmatched: Sequence[str], rows: Sequence[dict]) -> CurationState:
    """Fold the history over the result's tracks; malformed, out-of-order and inapplicable rows are skipped."""
    skipped = []
    seen = set()
    done, redo = [], []
    for row in rows:
        what = _malformed(row, seen)
        op_id = row.get('op_id', '') if isinstance(row, dict) else ''
        if isinstance(op_id, str) and op_id:
            seen.add(op_id)  # an op_id is spent once seen, even on a malformed row
        if what is not None:
            skipped.append((op_id, f'malformed: {what}'))
            continue
        op = row['operation']
        if op in ACTIONS:
            done.append(row)
            redo.clear()
        elif op == 'undo' and done and row['target_op'] == done[-1]['op_id']:
            redo.append(done.pop())
        elif op == 'redo' and redo and row['target_op'] == redo[-1]['op_id']:
            done.append(redo.pop())
        else:
            skipped.append((row['op_id'], 'out of order'))
    state = initial_state(tracks, unmatched)
    for row in done:
        try:
            state = apply(state, row)
        except ValueError as exc:
            state.skipped.append((row['op_id'], str(exc)))
    state.skipped = skipped + state.skipped
    state.undo_target = done[-1]['op_id'] if done else None
    state.redo_target = redo[-1]['op_id'] if redo else None
    return state
import csv
import io
import datetime
import os
import secrets
import shutil
import socket
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from multiphy.contract.results import MatchResult

HISTORY = 'history.csv'
LOCK = 'history.lock'
_UNREADABLE = {'pid': None, 'host': None, 'token': None}


def read_history(curation_dir: Path) -> list:
    """The history rows as string dicts; malformed rows are kept for replay to skip and list."""
    path = Path(curation_dir) / HISTORY
    if not path.exists():
        return []
    text = path.read_bytes().decode('utf-8')
    lines = text.split('\n')
    if lines[-1] == '':
        lines.pop()  # the final newline ends the last line
    if not lines or tuple(_parse_line(lines[0]) or ()) != COLUMNS:
        raise ValueError(f'{path}: unknown history columns')
    rows = []
    for line in lines[1:]:
        fields = _parse_line(line)
        if fields is not None and len(fields) == len(COLUMNS):
            rows.append(dict(zip(COLUMNS, fields)))
        else:  # replay skips it as malformed
            rows.append({**{c: '' for c in COLUMNS}, 'notes': line})
    return rows


def _parse_line(line: str) -> Optional[list]:
    try:
        return next(csv.reader([line]), [])
    except csv.Error:
        return None


def _history_line(row: dict) -> str:
    """The row as one history line; ValueError when read_history would not read it back unchanged."""
    cells = [row[col] for col in COLUMNS]
    for col, cell in zip(COLUMNS, cells):
        if '\n' in cell or '\r' in cell:
            raise ValueError(f'{col} holds a line break; a history row must be one line')
    buf = io.StringIO()
    csv.writer(buf, lineterminator='\n').writerow(cells)
    line = buf.getvalue()
    if _parse_line(line[:-1]) != cells:
        raise ValueError('the history reader cannot read this row back (a cell may exceed the csv field size limit)')
    return line


def _read_lock(path: Path) -> Optional[dict]:
    """The decoded lock: None when there is none, all-None fields when it cannot be read (treated as remote)."""
    try:
        text = Path(path).read_text()
    except FileNotFoundError:
        return None
    except UnicodeDecodeError:
        return dict(_UNREADABLE)
    try:
        lock = json.loads(text)
    except ValueError:
        return dict(_UNREADABLE)
    if not isinstance(lock, dict):
        return dict(_UNREADABLE)
    pid, host, token = lock.get('pid'), lock.get('host'), lock.get('token')
    if (not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0
            or not isinstance(host, str) or not host or not isinstance(token, str) or not token):
        return dict(_UNREADABLE)
    return lock


def _windows_pid_alive(pid: int) -> bool:
    """Query a process handle without sending a signal or requiring terminate rights."""
    import ctypes
    from ctypes import wintypes

    if not 0 < pid <= 0xffffffff:
        return False
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = (wintypes.HANDLE, wintypes.DWORD)
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    handle = kernel.OpenProcess(0x00100000, False, pid)  # SYNCHRONIZE only
    if not handle:
        # ERROR_INVALID_PARAMETER means there is no such PID. Access denied or
        # any unexpected failure must preserve the existing writer's lock.
        return ctypes.get_last_error() != 87
    try:
        # A process is signaled only after exit. Query errors conservatively
        # count as alive; a zero timeout never blocks the UI.
        return kernel.WaitForSingleObject(handle, 0) != 0  # WAIT_OBJECT_0
    finally:
        kernel.CloseHandle(handle)


def _pid_alive(pid: int) -> bool:
    if os.name == 'nt':
        # os.kill(pid, 0) is not a portable liveness probe on Windows.
        return _windows_pid_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _local_live(lock: Optional[dict]) -> bool:
    return lock is not None and lock['host'] == socket.gethostname() and _pid_alive(lock['pid'])


class Curation:
    """The one controller of a run's history: rewrites it under the lock and replays the result."""

    def __init__(self, curation_dir: Path, result):
        self.curation_dir = Path(curation_dir)
        self.result = result
        self.state: Optional[CurationState] = None
        self.token = secrets.token_hex(8)  # each controller has its own (D34)
        self.read_only = False
        self.lock_owner: Optional[dict] = None
        self.remote_lock = False
        self.taken_over = False
        self.write_failed: Optional[str] = None
        self.closed = False

    @property
    def _lock_path(self) -> Path:
        return self.curation_dir / LOCK

    @property
    def _history_path(self) -> Path:
        return self.curation_dir / HISTORY

    def _lock_json(self) -> str:
        return json.dumps({'pid': os.getpid(), 'host': socket.gethostname(), 'token': self.token,
                           'at': datetime.datetime.now().isoformat(timespec='seconds')})

    def _create_lock(self) -> bool:
        try:
            fd = os.open(self._lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            return False
        created = None
        try:
            try:
                created = os.fstat(fd)
                data = self._lock_json().encode()
                if os.write(fd, data) != len(data):
                    raise OSError('short write of the lock file')
            finally:
                os.close(fd)
        except BaseException:
            try:
                now = os.stat(self._lock_path)
                if created is not None and (now.st_dev, now.st_ino) == (created.st_dev, created.st_ino):
                    os.remove(self._lock_path)  # the file this call created
            except OSError:
                pass  # gone, replaced or unreadable: nothing of this call's is left to remove
            raise
        return True

    def _acquire(self) -> None:
        if self._create_lock():
            return
        owner = _read_lock(self._lock_path)
        if owner is None or (owner['host'] == socket.gethostname() and not _pid_alive(owner['pid'])):
            if owner is not None:
                try:
                    os.remove(self._lock_path)  # a dead local writer's lock
                except FileNotFoundError:
                    pass
            if self._create_lock():  # try once more
                return
            owner = _read_lock(self._lock_path)
        owner = owner if owner is not None else dict(_UNREADABLE)
        self.read_only = True
        self.lock_owner = owner
        self.remote_lock = owner['host'] != socket.gethostname()

    def _replay(self) -> CurationState:
        self.state = replay(self.result.tracks, self.result.unmatched, read_history(self.curation_dir))
        return self.state

    @classmethod
    def open(cls, curation_dir: Path, result: 'MatchResult') -> 'Curation':
        self = cls(curation_dir, result)
        self.curation_dir.mkdir(parents=True, exist_ok=True)
        self._acquire()
        try:
            self._replay()
        except BaseException:
            if not self.read_only:
                try:
                    if self._owns():
                        os.remove(self._lock_path)  # the lock this call wrote (D35)
                except OSError:
                    pass  # a lock that cannot be checked or removed is left; the opening error stands
            raise
        return self

    def _owns(self) -> bool:
        if self.closed:
            return False
        lock = _read_lock(self._lock_path)
        return lock is not None and lock['token'] == self.token

    def _check_writable(self) -> None:
        if self.closed or self.read_only:
            raise PermissionError('read-only')
        if not self._owns():
            self.read_only = True
            self.taken_over = True
            raise PermissionError('taken over')

    def _write(self, row: dict) -> CurationState:
        """Replace the history with its current text plus one new line; a failure turns this controller read-only."""
        line = _history_line(row)  # refuses a row the reader cannot read back
        path = self._history_path
        tmp = self.curation_dir / f'{HISTORY}.{self.token}.tmp'
        try:
            try:
                text = path.read_bytes().decode('utf-8')
                old = True
            except FileNotFoundError:
                text, old = '', False
            if not old:
                text = ','.join(COLUMNS) + '\n'
            elif not text.endswith('\n'):
                text += '\n'
            with open(tmp, 'w', newline='', encoding='utf-8') as f:
                f.write(text + line)
                f.flush()
                os.fsync(f.fileno())
            if old:
                shutil.copymode(path, tmp)  # group access never narrows (D40)
            os.replace(tmp, path)
            return self._replay()
        except BaseException as error:
            try:
                os.remove(tmp)
            except OSError:
                pass  # already renamed, never created, or cannot be removed
            self._fail(str(error) or type(error).__name__)
            raise  # no retry (D35)

    def _fail(self, reason: str) -> None:
        """Turn read-only after a failed write; one re-read and replay, else the previous state stays."""
        self.read_only = True
        self.write_failed = reason
        try:
            self._replay()
        except Exception:
            pass

    def do(self, operation: str, **fields) -> CurationState:
        self._check_writable()
        row = make_row(operation, **fields)
        what = _malformed(row, set())
        if what is not None:
            raise ValueError(what)
        _history_line(row)
        apply(self.state, row)  # a ValueError refuses the action and nothing is written
        return self._write(row)

    def _step(self, target: Optional[str], operation: str, nothing: str) -> CurationState:
        if target is None:
            raise ValueError(nothing)
        self._check_writable()
        return self._write(make_row(operation, target_op=target))

    def undo(self) -> CurationState:
        return self._step(self.state.undo_target, 'undo', 'nothing to undo')

    def redo(self) -> CurationState:
        return self._step(self.state.redo_target, 'redo', 'nothing to redo')

    def take_over(self) -> None:
        refusal = 'only a lock from another host can be taken over'
        if not (self.read_only and self.remote_lock and not self.closed):
            raise PermissionError(refusal)
        owner = _read_lock(self._lock_path)
        if _local_live(owner):
            self.lock_owner = owner
            self.remote_lock = False
            raise PermissionError(refusal)
        tmp = self.curation_dir / f'{LOCK}.{self.token}.tmp'
        tmp.write_text(self._lock_json())
        os.replace(tmp, self._lock_path)
        try:
            self._replay()
        except BaseException as error:
            self.write_failed = str(error) or type(error).__name__
            try:
                if self._owns():
                    os.remove(self._lock_path)  # only while it still holds this token
            except OSError:
                pass  # a lock that cannot be checked or removed is left; the replay error stands
            raise
        self.read_only = False
        self.taken_over = False
        self.lock_owner = None
        self.remote_lock = False

    def close(self) -> None:
        if not self.read_only and self._owns():
            try:
                os.remove(self._lock_path)
            except FileNotFoundError:
                pass
        self.closed = True
import pandas as pd

EXPORT_COLUMNS = ('track_id', 'cellid', 'siteid', 'n_units', 'n_days', 'accepted_at', 'curator',
                  'backend_track_id', 'review_status', 'backend_id', 'backend_run_id')


def accepted_rows(state: CurationState, sites: dict, *, result: Optional[MatchResult] = None) -> pd.DataFrame:
    """Accepted members with current grouping and optional immutable backend provenance.

    ``result`` is the original run result, never the edited tracks. Its unmatched
    units have an empty ``backend_track_id`` even after manual linking. Without a
    result all backend fields are empty. A missing site raises KeyError.
    """
    backend_tracks = ({str(m): cluster_id(track) for track in result.tracks for m in track}
                      if result is not None else {})
    backend_id = getattr(result, 'backend_id', '') or ''
    backend_run_id = getattr(result, 'run_id', '') or ''
    rows = []
    for tid, track in state.tracks.items():
        if track['status'] != 'accepted':
            continue
        members = list(track['members'])
        member_sites = [sites[m] for m in members]
        accepted = state.accepted_by[tid]
        for cellid, siteid in zip(members, member_sites):
            rows.append({'track_id': tid, 'cellid': cellid, 'siteid': siteid, 'n_units': len(members),
                         'n_days': len(set(member_sites)), 'accepted_at': accepted['at'],
                         'curator': accepted['curator'], 'backend_track_id': backend_tracks.get(cellid, ''),
                         'review_status': 'accepted', 'backend_id': backend_id, 'backend_run_id': backend_run_id})
    table = pd.DataFrame(rows, columns=list(EXPORT_COLUMNS))
    return table.sort_values(['track_id', 'cellid'], kind='stable').reset_index(drop=True)


def write_export(state: CurationState, sites: dict, curation_dir: Path, out: Optional[Path] = None, *,
                 result: Optional[MatchResult] = None) -> Path:
    """Write the accepted tracks to a new CSV; an existing ``out`` is refused, a failed write leaves no file (D36)."""
    table = accepted_rows(state, sites, result=result)  # every row is built before any file exists
    if out is None:
        out = Path(curation_dir) / 'exports' / f'accepted_tracks_{time.strftime("%Y%m%d-%H%M%S")}.csv'
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    stream = open(out, 'x', newline='', encoding='utf-8')  # FileExistsError: never overwritten
    try:
        with stream:
            table.to_csv(stream, index=False, columns=list(EXPORT_COLUMNS))
    except BaseException:
        try:
            os.remove(out)  # the file this call created
        except FileNotFoundError:
            pass  # already gone: nothing is left
        # any other removal error propagates: its filename names the file left behind, and the
        # write error that started the cleanup is its __context__
        raise
    return out


def export_run(project_path: Path, run: str, out: Optional[Path] = None) -> Path:
    """Export the accepted tracks of the project's across-day run ``run``; the history is read without the lock."""
    from multiphy import backends  # imported here: it loads every backend
    from multiphy.project.folder import RUNS_DIR
    from multiphy.run import catalog

    if '/' in run or '\\' in run or run in ('', '.', '..'):
        raise ValueError('RUN is a run folder name')
    project_path = Path(project_path)
    entry = catalog.read_run(project_path / RUNS_DIR / run)  # only across-day runs live in the project's runs/
    if entry is None:
        raise ValueError(f'{run} is not a completed run')
    from multiphy.run.run_dir import read_json_object
    try:
        mode = read_json_object(project_path / RUNS_DIR / run / 'selection.json').get('mode')
    except Exception as error:
        raise ValueError(f'{run} is not an across-day run') from error
    if mode != 'across_days':
        raise ValueError(f'{run} is not an across-day run')
    result = backends.load_result(entry.result_path)
    sites = dict(zip(result.units.cellid, result.units.session_id))  # a project day's siteid is its session_id
    curation_dir = project_path / 'curation' / 'runs' / run
    state = replay(result.tracks, result.unmatched, read_history(curation_dir))
    return write_export(state, sites, curation_dir, out, result=result)
