# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: argv for another interpreter from the option schema | date: 2026-09-22]
"""Build the argument vector that runs a backend in another Python interpreter.

The viewer uses this so no optional scientific package is imported into the GUI
process. Options come from a backend's spec through ``options.to_argv``; nothing here
knows a backend by name.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: typed signature of build_selection_command (Plan B Task 2) | date: 2026-09-28]
from typing import Optional
# [AGENT EDIT END]

from ..contract import registry
from . import options as opt


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: build_command forwards the cache root (Plan B Task 2) | date: 2026-09-28]
def build_command(input_path, backend_id, options, *, output, partition=None, group_column=None, interpreter=None,
                  root=None) -> tuple:
# [AGENT EDIT END]
    source = Path(input_path).expanduser().resolve()
    if not (source / 'manifest.json').is_file():
        raise ValueError(f'Select an archived dataset containing manifest.json: {source}')
    spec = registry.get(backend_id).spec
    resolved = opt.validate(spec, options)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one interpreter check for both command builders (Plan B Task 2) | date: 2026-09-28]
    program = _program(interpreter)
    # [AGENT EDIT END]
    argv = ['-u', '-m', 'multiphy.run', 'run-backend', '--input', str(source),
            '--backend', backend_id, '--output', str(Path(output).expanduser().absolute())]
    if partition:
        argv += ['--partition', partition]
    if group_column:
        argv += ['--group-column', group_column]
    argv += opt.to_argv(spec, resolved)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: build_command forwards the cache root (Plan B Task 2) | date: 2026-09-28]
    if root is not None:
        argv += ['--root', str(root)]
    # [AGENT EDIT END]
    return str(program), argv
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: argv that runs a backend on a selection of project days (Plan B Task 2) | date: 2026-09-28]


def _program(interpreter) -> Path:
    # Keep a venv's executable path: resolving its symlink selects the base interpreter.
    program = Path(interpreter or sys.executable).expanduser().absolute()
    if not program.is_file() or not os.access(program, os.X_OK):
        raise ValueError(f'Choose an existing executable Python interpreter: {program}')
    return program


def build_selection_command(project_path: Path, days: tuple, backend_id: str, options: dict, *, output: Path,
                            interpreter: Optional[str] = None) -> tuple:
    """``(program, argv)`` that runs ``backend_id`` on ``days`` of the project at ``project_path``."""
    spec = registry.get(backend_id).spec
    program = _program(interpreter)
    argv = (['-u', '-m', 'multiphy.run', 'run-backend', '--project', str(project_path)]
            + (['--day', days[0]] if len(days) == 1 else ['--days', *days])
            + ['--backend', backend_id, '--output', str(output)]
            + opt.to_argv(spec, opt.validate(spec, options)))
    return str(program), argv
# [AGENT EDIT END]
# [AGENT EDIT END]
