# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: record and verify the identity of each day file, and detect retargeted links (multiphy project folder Task 5) | date: 2026-09-25]
"""File identities: which file a day was built from, and whether it is still that file.

``record`` stores the path as given (made absolute) and the file it resolved to, with that
file's size, ``mtime_ns`` and, optionally, its sha256. ``verify`` checks the resolved file only,
so a link re-pointed after recording keeps reading the recorded file; ``retargeted`` reports
that the link now points elsewhere. multiphy never creates a link.
"""
from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Optional

CHUNK_BYTES = 8 * 1024 * 1024


class SourceMissing(FileNotFoundError):
    """A recorded file no longer exists at its resolved path."""


class SourceChanged(ValueError):
    """A recorded file's size, ``mtime_ns`` or contents differ from its identity."""


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    role: str
    given: str
    resolved: str
    size_bytes: int
    mtime_ns: int
    sha256: Optional[str]
    sha256_from: Optional[str]

    def to_dict(self) -> dict:
        return {field.name: getattr(self, field.name) for field in dataclasses.fields(self)}

    @classmethod
    def from_dict(cls, data: dict) -> FileIdentity:
        return cls(**{field.name: data[field.name] for field in dataclasses.fields(cls)})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK_BYTES), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _stat(role: str, resolved: str) -> os.stat_result:
    try:
        return os.stat(resolved)
    except (FileNotFoundError, NotADirectoryError):
        raise SourceMissing(f'{role} source missing: {resolved}') from None


def record(role: str, given: str, *, hash_contents: bool = True) -> FileIdentity:
    """The identity of the file ``given`` resolves to now; ``hash_contents=False`` skips the sha256."""
    given = os.path.abspath(given)
    resolved = os.path.realpath(given)
    status = _stat(role, resolved)
    sha256 = sha256_file(resolved) if hash_contents else None
    return FileIdentity(role=role, given=given, resolved=resolved, size_bytes=status.st_size,
                        mtime_ns=status.st_mtime_ns, sha256=sha256, sha256_from='init' if hash_contents else None)


def verify(identity: FileIdentity, *, rehash: bool) -> None:
    """Raise unless the resolved file still has the recorded size, ``mtime_ns`` and, with ``rehash``, sha256."""
    status = _stat(identity.role, identity.resolved)
    changed = SourceChanged(f'{identity.role} source changed: {identity.resolved}')
    if status.st_size != identity.size_bytes or status.st_mtime_ns != identity.mtime_ns:
        raise changed
    if rehash and identity.sha256 is not None:
        try:
            digest = sha256_file(identity.resolved)
        except (FileNotFoundError, NotADirectoryError):
            raise SourceMissing(f'{identity.role} source missing: {identity.resolved}') from None
        if digest != identity.sha256:
            raise changed


def retargeted(identity: FileIdentity) -> bool:
    """True when the recorded path now resolves to a file other than the recorded one."""
    return os.path.realpath(identity.given) != identity.resolved
# [AGENT EDIT END]
