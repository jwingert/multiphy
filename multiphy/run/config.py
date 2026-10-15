# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: cache locations moved under run/, with MULTIPHY_CACHE_ROOT override | date: 2026-09-22]
"""Cache locations without importing recording loaders or the database."""
from __future__ import annotations
from datetime import datetime, timezone
import os
from pathlib import Path
import uuid


CACHE_ROOT = '/auto/data/nems_db/unit_match'   # lab default until the repository split; override with MULTIPHY_CACHE_ROOT


def cache_root(root=None):
    """Explicit root, else MULTIPHY_CACHE_ROOT, else the lab default."""
    if root is None:
        root = os.environ.get('MULTIPHY_CACHE_ROOT') or CACHE_ROOT
    return Path(root).expanduser().resolve()


def extraction_scratch_root(scratch_root=None, *, root=None):
    """Resolve temporary snippet storage; default stays in the matching cache.

    An explicit location must be absolute and outside this code repository.
    Only a uniquely named child is created and cleaned by the extractor.
    This option does not change the location of permanent datasets.
    """
    if scratch_root is None:
        return cache_root(root) / '_runtime'
    candidate = Path(scratch_root).expanduser()
    if not candidate.is_absolute():
        raise ValueError('Extraction scratch_root must be an absolute directory path.')
    candidate = candidate.resolve()
    repository = Path(__file__).resolve().parents[3]
    if candidate == repository or repository in candidate.parents:
        raise ValueError('Temporary voltage snippets must stay outside the code repository.')
    return candidate


def resolve_cache_path(path, *, root=None):
    """Reject generated-output paths outside the cache, including symlink escapes."""
    base = cache_root(root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    candidate = candidate.resolve()
    try:
        candidate.relative_to(base)
    except ValueError as exc:
        raise ValueError(f'Generated artifacts must stay under {base}: {candidate}') from exc
    return candidate


def new_run_path(dataset_id, *, kind='runs', root=None):
    """Return a unique candidate path; the producer creates it without overwrite."""
    dataset_id = str(dataset_id)
    if not dataset_id or dataset_id in ('.', '..') or any(c in dataset_id for c in '/\\\0'):
        raise ValueError('dataset_id must be a nonempty single directory name')
    if kind not in ('runs', 'training'):
        raise ValueError('kind must be runs or training')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    return resolve_cache_path(Path('benchmarks') / dataset_id / kind /
                              f'{stamp}_{uuid.uuid4().hex}', root=root)
# [AGENT EDIT END]
