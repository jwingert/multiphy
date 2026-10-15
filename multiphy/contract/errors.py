# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: backend contract error types | date: 2026-09-22]
"""Errors raised at the contract boundary.

BackendUnavailable: the method cannot run here (package, checkpoint or interpreter missing).
MissingInputs: the input lacks something the backend's spec requires; raised before any run.
ContractViolation: a backend or caller broke a rule the contract states.
"""
from __future__ import annotations


class BackendUnavailable(RuntimeError):
    """The requested method needs an installed runner, checkpoint or environment."""


class MissingInputs(ValueError):
    """spec.requires is not covered by MatchingInput.available()."""

    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: optional detail says why an input is absent (whole-branch review fix 5) | date: 2026-09-23]
    def __init__(self, backend_id: str, group: str, missing, detail: str | None = None):
        self.backend_id, self.group, self.missing = backend_id, str(group), frozenset(missing)
        self.detail = detail
        names = ', '.join(sorted(str(getattr(m, 'value', m)) for m in self.missing))
        super().__init__(f'Backend {backend_id!r} cannot run on group {group!r}: missing inputs {names}'
                         + (f'; {detail}' if detail else ''))
    # [AGENT EDIT END]


class ContractViolation(ValueError):
    """A result, option set or unit set does not satisfy the declared contract."""
# [AGENT EDIT END]
