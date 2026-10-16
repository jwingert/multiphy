"""Turn a backend's option schema into argparse arguments, a validated dict, or argv.

Every caller that takes options from a person (CLI) or hands them to another
process (subprocess launcher) goes through here, so a backend's options are declared
once, on its spec, and nowhere else.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from ..contract.errors import ContractViolation


def _flag(name):
    return '--' + name.replace('_', '-')


def load_json_option(value):
    """A mapping, a path to a JSON object, or inline JSON text."""
    if value is None or isinstance(value, dict):
        return value
    text = str(value)
    candidate = Path(text).expanduser()
    if candidate.is_file():
        text = candidate.read_text(encoding='utf-8')
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f'not a JSON object or a path to one: {exc}') from exc
    if not isinstance(loaded, dict):
        raise ValueError('JSON option must be an object')
    return loaded


def _coerce(option, value):
    if value is None:
        return None
    t = option.type
    try:
        if t == 'float':
            return float(value)
        if t == 'int':
            if isinstance(value, float) and not value.is_integer():
                raise ValueError('not an integer')
            return int(value)
        if t == 'bool':
            if isinstance(value, str):
                if value.lower() in ('1', 'true', 'yes'):
                    return True
                if value.lower() in ('0', 'false', 'no'):
                    return False
                raise ValueError('not a boolean')
            return bool(value)
        if t == 'str':
            return str(value)
        if t == 'choice':
            if value not in option.choices:
                raise ValueError(f'must be one of {option.choices}')
            return value
        if t == 'json':
            return load_json_option(value)
        if t == 'path':
            p = Path(str(value)).expanduser().resolve()
            if not p.is_file():
                raise ValueError(f'file does not exist: {p}')
            return str(p)
    except (TypeError, ValueError) as exc:
        raise ContractViolation(f'option {option.name!r}: {exc}') from exc
    raise ContractViolation(f'option {option.name!r}: unknown type {t!r}')


def validate(spec, options) -> dict:
    """Fill defaults, coerce types and reject unknown or missing required options."""
    options = dict(options or {})
    known = {o.name for o in spec.options}
    unknown = set(options) - known
    if unknown:
        raise ContractViolation(f'backend {spec.id!r} does not take options {sorted(unknown)}')
    out = {}
    for o in spec.options:
        value = options.get(o.name, o.default)
        if value is None and o.required:
            raise ContractViolation(f'option {o.name!r} is required for backend {spec.id!r}')
        out[o.name] = _coerce(o, value)
    return out


def add_arguments(parser: argparse.ArgumentParser, spec) -> None:
    for o in spec.options:
        kwargs = dict(dest=o.name, help=o.help or None, default=argparse.SUPPRESS)
        if o.type == 'bool':
            parser.add_argument(_flag(o.name), action='store_true', **kwargs)
            parser.add_argument(_flag('no_' + o.name), action='store_false', dest=o.name, default=argparse.SUPPRESS)
        elif o.type == 'float':
            parser.add_argument(_flag(o.name), type=float, **kwargs)
        elif o.type == 'int':
            parser.add_argument(_flag(o.name), type=int, **kwargs)
        elif o.type == 'choice':
            parser.add_argument(_flag(o.name), choices=o.choices, **kwargs)
        elif o.type == 'path':
            parser.add_argument(_flag(o.name), type=str, metavar='FILE', **kwargs)
        else:
            parser.add_argument(_flag(o.name), type=str, **kwargs)


def from_namespace(spec, namespace) -> dict:
    return {o.name: getattr(namespace, o.name) for o in spec.options if hasattr(namespace, o.name)}


def to_argv(spec, options) -> list:
    """Argument vector for a validated option dict; never shell text."""
    argv = []
    for o in spec.options:
        value = options.get(o.name)
        if value is None:
            continue
        if o.type == 'bool':
            argv.append(_flag(o.name) if value else _flag('no_' + o.name))
        elif o.type == 'json':
            argv += [_flag(o.name), json.dumps(value, sort_keys=True)]
        else:
            argv += [_flag(o.name), str(value)]
    return argv
