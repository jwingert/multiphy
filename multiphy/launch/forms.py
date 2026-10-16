"""The launcher window's forms, read from the CLI's own parser (D3).

``commands_of`` lists the parser's commands, ``form_fields`` turns one command's arguments into
``Field`` rows, and ``command_argv`` turns a form's values back into the argv the CLI would have been
typed with. ``BOUND`` names, per command, the arguments the window fills from its own selection
(project, checked days, backend) rather than from a form field. No Qt, no lab module.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Optional

from ..contract import registry
from ..contract.inputs import Input
from ..project.folder import Project
from ..project.selection import Selection, selection_inputs
from ..run.project_run import check_real_day_run
from ..contract.inputs import RunEvidence

BOUND = {'init': frozenset(), 'extract': frozenset({'project', 'days'}), 'status': frozenset({'project'}),
         'run': frozenset({'project', 'backend', 'day', 'days'}), 'export': frozenset({'project'})}

_BROWSE = {'DIR': 'dir', 'FILE': 'open', 'CSV': 'open', 'EXE': 'open', 'PATH': 'save'}
_NO_FIELD = 'has no form field; run it from the command line'


@dataclass(frozen=True)
class Field:
    """One form row: an argument of a command's parser."""
    dest: str
    flag: Optional[str]
    label: str
    help: str
    kind: str
    choices: tuple
    required: bool
    browse: Optional[str]
    placeholder: str


def commands_of(parser: argparse.ArgumentParser) -> dict:
    """``{name: subparser}`` of the parser's one subparsers action, in its order."""
    actions = [a for a in parser._actions if isinstance(a, argparse._SubParsersAction)]
    return dict(actions[0].choices)


def _spec_option(spec, dest: str):
    """The option of ``spec`` named ``dest``; ``None`` without a spec or such an option."""
    if spec is None:
        return None
    return next((o for o in spec.options if o.name == dest), None)


def _unsupported_help(action) -> Optional[str]:
    """The help of an action that has no form field (D35); ``None`` for a supported action."""
    kind = type(action)
    if kind in (argparse._StoreTrueAction, argparse._StoreFalseAction):
        return None
    if kind is argparse._StoreAction:
        if action.nargs is None:
            return None
        if action.nargs in ('+', '*'):
            if not action.option_strings:
                return f'a positional list {_NO_FIELD}'
            if action.choices is not None:
                return f'a multiple choice {_NO_FIELD}'
            return None
        return f'nargs {action.nargs!r} {_NO_FIELD}'
    return f'{kind.__name__} {_NO_FIELD}'


def form_fields(subparser: argparse.ArgumentParser, bound: frozenset, spec=None) -> list:
    """One ``Field`` per argument of ``subparser``, in order, except ``--help`` and the ``bound`` dests.
    A ``--<name>`` / ``--no-<name>`` pair on one dest is one ``'switch'`` field (D36)."""
    fields, store_true = [], {}  # store_true: dest -> index of its --<name> field
    for action in subparser._actions:
        if isinstance(action, argparse._HelpAction) or action.dest in bound:
            continue
        flag = max(action.option_strings, key=len) if action.option_strings else None
        if type(action) is argparse._StoreFalseAction:  # exact type: a custom subclass stays unsupported (D35)
            index = store_true.pop(action.dest, None)
            if index is not None:  # D36: the --no-<name> half of a bool option
                first = fields[index]
                fields[index] = Field(first.dest, first.flag, first.label, first.help, 'switch',
                                      (first.flag, flag), first.required, None, first.placeholder)
                continue
        label = action.dest.replace('_', ' ') if flag is None else flag.lstrip('-').replace('-', ' ')
        option = _spec_option(spec, action.dest)
        required = bool(action.required) or bool(option is not None and option.required)
        unsupported = _unsupported_help(action)
        if unsupported is not None:
            kind, help_text = 'unsupported', unsupported
        else:
            help_text = action.help or ''
            if action.nargs == 0:
                kind = 'flag'
            elif action.choices is not None:
                kind = 'choice'
            elif action.nargs in ('+', '*'):
                kind = 'list'
            else:
                kind = 'text'
        choices = tuple(str(c) for c in action.choices or ())
        browse = _BROWSE.get(action.metavar) if kind == 'text' and isinstance(action.metavar, str) else None
        if option is not None and option.default is not None:
            placeholder = str(option.default)
        elif kind != 'flag' and action.default is not None and action.default is not argparse.SUPPRESS:
            placeholder = str(action.default)
        else:
            placeholder = ''
        if type(action) is argparse._StoreTrueAction:
            store_true[action.dest] = len(fields)
        fields.append(Field(action.dest, flag, label, help_text, kind, choices, required, browse, placeholder))
    return fields


def has_value(field: Field, value) -> bool:
    """Whether ``value`` fills ``field`` (D37)."""
    if field.kind == 'flag':
        return bool(value)
    if field.kind in ('choice', 'switch'):
        return value in field.choices
    if field.kind == 'list':
        return bool(str(value or '').split())
    if field.kind == 'text':
        return value not in (None, '')
    return False


def command_argv(command: str, fields: list, values: dict, *, project: Optional[str] = None,
                 days: tuple = (), backend: Optional[str] = None) -> list:
    """The argv a person would type for ``command`` with these form ``values``, the window's project,
    checked ``days`` and ``backend`` filling the command's ``BOUND`` arguments. ``ValueError`` names
    the first field the form cannot send, then the first required field left empty."""
    for field in fields:
        if field.kind == 'unsupported':
            raise ValueError(f'{field.flag or field.label} runs only from the command line')
    for field in fields:
        if field.required and not has_value(field, values.get(field.dest)):
            raise ValueError(f'{field.label} is required')
    bound = BOUND.get(command, frozenset())
    argv = [command]
    if 'project' in bound:
        if project in (None, ''):
            raise ValueError('choose a project')
        argv.append(project)
    for field in fields:
        if field.flag is None:
            value = values.get(field.dest, '')
            if value not in (None, ''):
                argv.append(value)
    if command == 'extract' and days:
        argv += ['--day', *days]
    elif command == 'run':
        if backend is None:
            raise ValueError('choose a backend')
        argv += ['--backend', backend]
        if len(days) == 1:
            argv += ['--day', days[0]]
        elif len(days) >= 2:
            argv += ['--days', *days]
        else:
            raise ValueError('check one or more days')
    for field in fields:
        if field.flag is None:
            continue
        value = values.get(field.dest)
        if field.kind == 'flag':
            if value:
                argv.append(field.flag)
        elif field.kind == 'switch':
            if value in field.choices:
                argv.append(value)
        elif field.kind in ('text', 'choice'):
            if value not in (None, ''):
                argv += [field.flag, value]
        elif field.kind == 'list':
            items = str(value or '').split()
            if items:
                argv += [field.flag, *items]
    return argv


def backend_refusal(project: Project, days: tuple, backend_id: str, *, interpreter: Optional[str]) -> str:
    """``''`` when ``backend_id`` can run on ``days`` of ``project``, else the first refusal (D5): missing
    inputs (or the ``Selection`` refusal), then the real-day check, then an unavailable backend with no
    verified ``interpreter``. Other exceptions propagate; the window greys the item with them (D38)."""
    backend = registry.get(backend_id)
    spec = backend.spec
    try:
        missing = spec.requires - selection_inputs(project, tuple(days)) - {Input.CHECKPOINT}
    except ValueError as error:
        return str(error)
    if missing:
        return 'missing inputs: ' + ', '.join(sorted(i.value for i in missing))
    try:
        check_real_day_run(Selection(project, tuple(days)), spec)
    except ValueError as error:
        return str(error)
    availability = backend.available()
    if not availability.ok and interpreter is None:
        return f'unavailable here: {availability.reason}; no verified interpreter'
    return ''


EVENT_CHOICES = {'feature_stream': frozenset({'stimulus'})}  # option -> the choices a run without events cannot measure


def selection_evidence(project: Project, days: tuple, backend_id: str, *, min_repeats: int) -> Optional[RunEvidence]:
    """The evidence ``days`` of ``project`` measured (``Selection.evidence``), as ``launch run`` decides it;
    ``None`` when ``backend_id`` reads no timing. ``Selection`` and ``run_evidence`` errors propagate; the
    window shows them as one ``error`` tag (D55)."""
    spec = registry.get(backend_id).spec
    if Input.TIMING not in spec.requires | spec.optional:
        return None
    return Selection(project, tuple(days)).evidence(min_repeats)
