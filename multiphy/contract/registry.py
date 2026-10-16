# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: one backend registry with entry-point discovery | date: 2026-09-22]
# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: docstring states when entry points load and what a failure does (whole-branch review fix 2) | date: 2026-09-23]
"""Where backends are found.

Built-ins register on import of ``multiphy.backends``. Third-party packages
declare ``[project.entry-points."multiphy.backends"] name = "pkg.module:factory"`` where
``factory()`` returns a Backend. Entry points are loaded once, on the first listing or
the first miss; one that fails to load is recorded in ``ENTRY_POINT_ERRORS`` and skipped.
Availability is never probed here, so listing backends imports no upstream package.
"""
# [AGENT EDIT END]
from __future__ import annotations
import importlib.metadata

ENTRY_POINT_GROUP = 'multiphy.backends'
_BACKENDS: dict = {}
_ENTRY_POINTS_LOADED = False
_BUILTINS_LOADED = False


def register(backend) -> None:
    spec = backend.spec
    existing = _BACKENDS.get(spec.id)
    if existing is not None:
        if existing.spec.builtin:
            raise ValueError(f'Cannot replace built-in backend {spec.id!r}')
        raise ValueError(f'Backend {spec.id!r} is already registered')
    _BACKENDS[spec.id] = backend


# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: a failing plugin is recorded and skipped; loaded only once the loop completes (whole-branch review fix 2) | date: 2026-09-23]
ENTRY_POINT_ERRORS: dict = {}  # entry-point name -> why it could not be loaded


def load_entry_points() -> None:
    """Register every ``multiphy.backends`` entry point once; a failing one is recorded, not raised."""
    global _ENTRY_POINTS_LOADED
    if _ENTRY_POINTS_LOADED:
        return
    eps = importlib.metadata.entry_points()
    group = eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, 'select') else eps.get(ENTRY_POINT_GROUP, ())
    for ep in group:
        try:
            backend = ep.load()()
            if backend.spec.id not in _BACKENDS:
                register(backend)
        except Exception as exc:  # one broken plugin must not hide the others or the built-ins
            ENTRY_POINT_ERRORS[ep.name] = f'{type(exc).__name__}: {exc}'
    _ENTRY_POINTS_LOADED = True


def registered_ids() -> frozenset:
    """Ids registered so far; never loads built-ins or entry points."""
    return frozenset(_BACKENDS)
# [AGENT EDIT END]


def _ensure_builtins():
    """Register the built-in backends once per reset; never swallows an ImportError."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import multiphy.backends as builtins  # registers on first import
    builtins.register_builtins()  # re-registers after clear_for_tests(); skips ids already present


def get(backend_id: str):
    _ensure_builtins()
    if backend_id not in _BACKENDS:
        load_entry_points()
    if backend_id not in _BACKENDS:
        raise KeyError(f'Unknown backend {backend_id!r}; known: {sorted(_BACKENDS)}')
    return _BACKENDS[backend_id]


def list_specs() -> tuple:
    _ensure_builtins()
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: installed plugins are listed too, after the built-ins (whole-branch review fix 2) | date: 2026-09-23]
    load_entry_points()
    # [AGENT EDIT END]
    return tuple(b.spec for b in _BACKENDS.values())


def clear_for_tests() -> None:
    """Empty the registry; tests re-register what they need. Never called by product code."""
    global _ENTRY_POINTS_LOADED, _BUILTINS_LOADED
    _BACKENDS.clear()
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: entry-point load errors reset with the registry (whole-branch review fix 2) | date: 2026-09-23]
    ENTRY_POINT_ERRORS.clear()
    # [AGENT EDIT END]
    _ENTRY_POINTS_LOADED = False
    _BUILTINS_LOADED = False
# [AGENT EDIT END]
