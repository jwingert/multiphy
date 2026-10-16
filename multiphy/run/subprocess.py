"""Build the argument vector that runs a backend in another Python interpreter.

The viewer uses this so no optional scientific package is imported into the GUI
process. Options come from a backend's spec through ``options.to_argv``; nothing here
knows a backend by name.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
from typing import Optional

from ..contract import registry
from . import options as opt


def build_command(input_path, backend_id, options, *, output, partition=None, group_column=None, interpreter=None,
                  root=None) -> tuple:
    source = Path(input_path).expanduser().resolve()
    if not (source / 'manifest.json').is_file():
        raise ValueError(f'Select an archived dataset containing manifest.json: {source}')
    spec = registry.get(backend_id).spec
    resolved = opt.validate(spec, options)
    program = _program(interpreter)
    argv = ['-u', '-m', 'multiphy.run', 'run-backend', '--input', str(source),
            '--backend', backend_id, '--output', str(Path(output).expanduser().absolute())]
    if partition:
        argv += ['--partition', partition]
    if group_column:
        argv += ['--group-column', group_column]
    argv += opt.to_argv(spec, resolved)
    if root is not None:
        argv += ['--root', str(root)]
    return str(program), argv


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
