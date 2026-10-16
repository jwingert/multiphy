"""A project folder: ``<root>/<name>/`` with ``project.json``, ``days/<day>/`` and ``logs/``.

``update`` is the only writer of ``project.json``: it takes ``project.lock`` (created
exclusively), applies one change to a freshly read manifest, writes it atomically and
releases the lock. A lock left by a dead process stays until a person removes it.
"""
from __future__ import annotations

import os
from pathlib import Path
import re
import socket
from typing import Callable, Optional

from .manifest import new_manifest, read_manifest, write_manifest

RESERVED_NAMES = ('collections', 'reports', 'logs', '_runtime', 'dependencies', 'viewer', 'benchmarks', 'multiday',
                  'curated')
NAME_PATTERN = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
MANIFEST_FILE = 'project.json'
LOCK_FILE = 'project.lock'
RUNS_DIR = 'runs'


class ProjectLocked(RuntimeError):
    """``project.lock`` exists: another process is updating ``project.json``, or one died doing so."""


class Project:
    """One project under ``root``; ``path`` is ``root / name``."""

    def __init__(self, root: Path, name: str):
        self.root = Path(root).expanduser().resolve()
        self.name = name
        self.path = self.root / name

    def __repr__(self) -> str:
        return f'Project({str(self.root)!r}, {self.name!r})'

    @classmethod
    def create(cls, root: Path, name: str, *, group: Optional[str] = None) -> Project:
        if name in RESERVED_NAMES:
            raise ValueError(f'{name} is a reserved name.')
        if not NAME_PATTERN.match(name):
            raise ValueError(f'Project name {name} must match [A-Za-z0-9][A-Za-z0-9_.-]*.')
        project = cls(root, name)
        try:
            project.path.mkdir(parents=True)
        except FileExistsError:
            raise FileExistsError(f'Project {name} already exists.') from None
        (project.path / 'days').mkdir()
        (project.path / 'logs').mkdir()
        write_manifest(project.path / MANIFEST_FILE, new_manifest(name, group=group or name))
        return project

    @classmethod
    def open(cls, root: Path, project: str) -> Project:
        project = cls(root, cls._name_under(root, project))
        if not (project.path / MANIFEST_FILE).is_file():
            raise FileNotFoundError(f'No project {project.name} under {project.root}.')
        return project

    @classmethod
    def ensure(cls, root: Path, project: str) -> Project:
        name = cls._name_under(root, project)
        if not cls(root, name).path.exists():
            return cls.create(root, name)
        return cls.open(root, name)

    @staticmethod
    def _name_under(root: Path, project: str) -> str:
        """``project`` as a name, or the name of the folder it points to directly under ``root``."""
        text = os.fspath(project)
        if NAME_PATTERN.match(text):
            return text
        root = Path(root).expanduser().resolve()
        candidate = Path(text).expanduser()
        if NAME_PATTERN.match(candidate.name) and candidate.parent.resolve() == root:
            return candidate.name
        raise ValueError(f'{project} is not a folder directly under {root}.')

    def day_dir(self, day: str) -> Path:
        return self.path / 'days' / day

    def read(self) -> dict:
        return read_manifest(self.path / MANIFEST_FILE)

    def update(self, change: Callable[[dict], None]) -> dict:
        """Apply ``change`` to the current manifest under ``project.lock``; return the written manifest."""
        lock = self.path / LOCK_FILE
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                holder = lock.read_text(encoding='utf-8').strip() or 'a process that has not written its name yet'
            except OSError:
                holder = 'an unreadable lock'
            raise ProjectLocked(f'project.json is locked by {holder}; remove {lock} if that process is gone.') from None
        try:
            try:
                os.write(handle, f'{socket.gethostname()} {os.getpid()}'.encode('utf-8'))
            finally:
                os.close(handle)
            manifest = self.read()
            change(manifest)
            write_manifest(self.path / MANIFEST_FILE, manifest)
        finally:
            lock.unlink(missing_ok=True)
        return manifest

    def log(self, line: str) -> None:
        with (self.path / 'logs' / 'launcher.log').open('a', encoding='utf-8') as stream:
            stream.write(line + '\n')
