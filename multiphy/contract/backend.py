# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: declared backend spec and protocol | date: 2026-09-22]
"""What a backend is.

A backend is an object with a ``spec`` (data: id, version, what it requires, what it
can emit, which options it takes), an ``available()`` probe and a ``run()``. The spec
is data so the runner can check inputs before running and so the CLI, the Qt dialog
and the subprocess launcher can all be generated from the option schema.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from .inputs import Input, MatchingInput
from .provenance import Provenance
from .results import MatchResult, ScoreKind, TrackKind

OPTION_TYPES = ('float', 'int', 'str', 'bool', 'path', 'json', 'choice')


@dataclass(frozen=True)
class Option:
    name: str
    type: str
    default: object = None
    help: str = ''
    choices: tuple = ()
    required: bool = False

    def __post_init__(self):
        if self.type not in OPTION_TYPES:
            raise ValueError(f'Option {self.name!r}: type must be one of {OPTION_TYPES}')
        if self.type == 'choice' and not self.choices:
            raise ValueError(f'Option {self.name!r}: choice options need choices')
        if self.type == 'choice' and self.default is not None and self.default not in self.choices:
            raise ValueError(f'Option {self.name!r}: default not in choices')


@dataclass(frozen=True)
class Provides:
    score_kinds: frozenset
    track_kinds: frozenset
    similarity: bool = False
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a spec declares that its runs save day offsets (drift-view Task 2) | date: 2026-10-08]
    # A backend that sets day_offsets saves a `registration` in each group's provenance.
    day_offsets: bool = False
    # [AGENT EDIT END]


@dataclass(frozen=True)
class Availability:
    ok: bool
    version: str | None
    reason: str | None


@dataclass(frozen=True)
class BackendSpec:
    id: str
    version: str
    requires: frozenset
    optional: frozenset
    provides: Provides
    options: tuple
    builtin: bool = False
    # [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: a spec declares whether it may run on real-day archives; benchmark baselines opt out (whole-branch review fix 4) | date: 2026-09-23]
    real_days: bool = True
    # [AGENT EDIT END]

    def __post_init__(self):
        if not self.id or not self.version:
            raise ValueError('BackendSpec needs an id and a version')
        if Input.UNITS not in self.requires:
            raise ValueError('Every backend requires Input.UNITS')
        if self.requires & self.optional:
            raise ValueError('An input cannot be both required and optional')
        names = [o.name for o in self.options]
        if len(names) != len(set(names)):
            raise ValueError(f'BackendSpec {self.id!r}: duplicate option names')

    def option(self, name) -> Option:
        for o in self.options:
            if o.name == name:
                return o
        raise KeyError(name)

    def defaults(self) -> dict:
        return {o.name: o.default for o in self.options}


@dataclass(frozen=True)
class RunContext:
    run_id: str
    work_dir: Path
    provenance: Provenance = field(default_factory=Provenance)
    seed: int = 0


class Backend(Protocol):
    spec: BackendSpec

    def available(self) -> Availability: ...

    def run(self, inp: MatchingInput, options: dict, ctx: RunContext) -> MatchResult: ...


COMMON_TRACKER_OPTIONS = (
    Option('threshold', 'float', .9, 'Minimum score for a candidate link'),
    Option('contradiction_threshold', 'float', .1, 'A measured score below this between two groups blocks their merge'),
    Option('ambiguity_margin', 'float', .02, 'A rival within this margin makes a candidate ambiguous'),
)
NATIVE_TRACKING_OPTIONS = (
    Option('tracking', 'choice', 'native_intermediate', 'Grouping rule',
           choices=('native_intermediate', 'native_conservative', 'native_liberal', 'common_constrained')),
    Option('native_threshold', 'float', None, 'Published-rule match threshold; upstream default when omitted'),
)
# [AGENT EDIT END]
