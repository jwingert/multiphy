# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: list an archive's saved runs (method · date) for the viewers' run selector | date: 2026-09-23]
"""The saved runs of one archive, cheap enough to list on open.

A run folder holds ``run_manifest.json`` and ``result.json``. Listing reads only the
manifest and the first few kilobytes of the result, where ``backend_id``, ``version`` and
``run_id`` sit first; the full result is loaded and validated only when a run is chosen.
Real-day result files reach 20 MB, so loading every one up front would take seconds.

Two layouts exist on disk and both are scanned: ``<archive>/runs/<run>`` beside a
``dataset`` folder, and collection runs that sit directly beside the collection folder.
A run is listed only when its manifest names this dataset and does not contradict the
identity it was given (archive hash, partition, input manifest hash), its result is
format 2, and its backend id is registered.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: no id map; a run is listed only under a registered id (Task 1) | date: 2026-09-27]
from ..contract import names, registry
from .run_dir import load_result, read_json_object
# [AGENT EDIT END]

MANIFEST = 'run_manifest.json'
RESULT = 'result.json'
_HEAD_BYTES = 4096
_HEAD = re.compile(r'"result"\s*:\s*\{\s*"backend_id"\s*:\s*"([^"]*)"\s*,\s*'
                   r'"version"\s*:\s*"([^"]*)"\s*,\s*"run_id"\s*:\s*"([^"]*)"')
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: format 1 runs are hidden from the listing (Task 1) | date: 2026-09-27]
_SCHEMA = re.compile(r'"schema_version"\s*:\s*(\d+)')
# [AGENT EDIT END]


@dataclass(frozen=True)
class RunEntry:
    result_path: Path
    backend_id: str
    run_id: str
    finished: datetime
    manifest: dict = field(compare=False, repr=False)

    @property
    def folder(self):
        return self.result_path.parent


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the head also gives the schema version, so format 1 runs are skipped unread (Task 1) | date: 2026-09-27]
def _ids(result_path):
    """(schema_version, backend_id, run_id) from the result's head, else from a full load."""
    with result_path.open('rb') as stream:
        head = stream.read(_HEAD_BYTES).decode('utf-8', errors='replace')
    schema = _SCHEMA.search(head)
    schema = int(schema.group(1)) if schema else None
    if schema is not None and schema != 2:
        return schema, None, None
    match = _HEAD.search(head)
    if match and schema is not None:
        return schema, match.group(1), match.group(3)
    result = load_result(result_path)  # refuses anything but format 2
    return 2, result.backend_id, result.run_id
# [AGENT EDIT END]


def _finished(manifest, result_path):
    for key in ('finished_at', 'completed_at', 'started_at'):
        value = manifest.get(key)
        if isinstance(value, str):
            try:
                return datetime.fromisoformat(value).astimezone()
            except ValueError:
                continue
    return datetime.fromtimestamp(result_path.stat().st_mtime).astimezone()


def read_run(folder):
    """The listing entry for one completed run folder, else None."""
    folder = Path(folder)
    result_path = folder / RESULT
    if not (folder / MANIFEST).is_file() or not result_path.is_file():
        return None
    try:
        manifest = read_json_object(folder / MANIFEST)
        if manifest.get('status', 'completed') != 'completed':
            return None
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: only format 2 runs under a registered id are listed (Task 1) | date: 2026-09-27]
        schema, backend_id, run_id = _ids(result_path)
        if schema != 2:
            return None
        registry.get(backend_id)
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return RunEntry(result_path, backend_id, run_id, _finished(manifest, result_path), manifest)
    # [AGENT EDIT END]


def _candidates(dataset_dir, extra_dirs):
    if dataset_dir is not None:
        yield from (dataset_dir.parent / 'runs').glob('*')
        yield from dataset_dir.parent.iterdir()
    for extra in extra_dirs:
        extra = Path(extra).expanduser()
        if extra.is_dir():
            yield from extra.iterdir()


def find_runs(dataset_dir, *, dataset_id, identity=None, extra_dirs=()):
    """Completed runs of ``dataset_id``, newest first.

    ``identity`` maps manifest keys to the values this archive has; a run whose manifest
    records a different value is someone else's run. A run that does not record the key
    is kept, since older runs predate some keys; the full load on selection still
    validates it against the archive.
    """
    identity = {k: v for k, v in (identity or {}).items() if v is not None}
    if dataset_dir is not None:
        dataset_dir = Path(dataset_dir).expanduser().resolve()
    # the dataset folder is never one of its own runs, however it is reached
    seen, entries = {dataset_dir}, []
    for folder in _candidates(dataset_dir, extra_dirs):
        if not folder.is_dir():
            continue
        key = folder.resolve()
        if key in seen:
            continue
        seen.add(key)
        entry = read_run(folder)
        if entry is None or entry.manifest.get('dataset_id') != dataset_id:
            continue
        if any(k in entry.manifest and entry.manifest[k] != v for k, v in identity.items()):
            continue
        entries.append(entry)
    return sorted(entries, key=lambda e: e.finished, reverse=True)


def run_labels(entries):
    """``method · MM-DD``; where labels collide, the clock time, and where that still collides,
    the folder name in its place (runs launched minutes apart are told apart by name)."""
    labels = [f'{names.backend_short(e.backend_id)} · {e.finished:%m-%d}' for e in entries]
    for fmt in (lambda e: f'{names.backend_short(e.backend_id)} · {e.finished:%m-%d %H:%M}',
                lambda e: f'{names.backend_short(e.backend_id)} · {e.finished:%m-%d} · {e.folder.name}'):
        repeated = {x for x in labels if labels.count(x) > 1}
        labels = [fmt(e) if x in repeated else x for e, x in zip(entries, labels)]
    return labels


def run_tooltip(entry):
    return (f'{names.backend_label(entry.backend_id)}\nrun {entry.run_id}\n'
            f'{entry.finished:%Y-%m-%d %H:%M}\n{entry.folder}')
# [AGENT EDIT END]
