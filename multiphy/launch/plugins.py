"""Launcher plugins: modules that add source readers and launcher commands.

A plugin is a module with ``register(api)``, where ``api`` is a ``PluginApi``. ``load_plugins(root)``
imports the modules named by ``<root>/multiphy.json`` (``{"plugins": ["pkg.module", …]}``), then
calls every ``multiphy.plugins`` entry point, each naming a ``register`` function. A module named
twice registers once. Any import or register error raises ``ValueError('plugin <name>: …')``.
This module imports no lab module.
"""
from __future__ import annotations

import importlib
import importlib.metadata
import json
from pathlib import Path
from typing import Callable

from ..sources import base

CONFIG_FILE = 'multiphy.json'
ENTRY_POINT_GROUP = 'multiphy.plugins'
BUILTIN_COMMANDS = ('init', 'extract', 'status', 'run', 'export', 'gui')


class PluginApi:
    """What a plugin's ``register(api)`` adds to: readers, and launcher commands in ``commands``
    (name -> ``(configure, handler, help)``)."""

    def __init__(self):
        self.commands: dict = {}

    def add_reader(self, reader) -> None:
        """Register ``reader``; the same object already registered under its id is kept."""
        base._load_builtins()
        if base._READERS.get(reader.id) is reader:  # load_plugins ran before in this process
            return
        base.register_reader(reader)

    def add_command(self, name: str, configure: Callable, handler: Callable, help: str = '') -> None:
        """Add launcher command ``name``. ``configure(parser: argparse.ArgumentParser) -> None`` adds its
        options after the positional ``project``; ``handler(args: argparse.Namespace, root: Path) -> list``
        returns the lines to print."""
        if name in BUILTIN_COMMANDS or name in self.commands:
            raise ValueError(f'plugin command {name} is already defined')
        self.commands[name] = (configure, handler, help)


def _entry_points():
    eps = importlib.metadata.entry_points()
    return eps.select(group=ENTRY_POINT_GROUP) if hasattr(eps, 'select') else eps.get(ENTRY_POINT_GROUP, ())


def _config_modules(root: Path) -> list:
    """The module names ``<root>/multiphy.json`` lists; ``[]`` without the file or the key."""
    path = Path(root) / CONFIG_FILE
    if not path.exists():
        return []
    try:
        config = json.loads(path.read_text(encoding='utf-8'))
    except ValueError:  # JSONDecodeError and UnicodeDecodeError: not the configuration shape
        config = None
    names = config.get('plugins', []) if isinstance(config, dict) else None
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError(f'{path}: "plugins" must be a list of module names')
    return names


def load_plugins(root: Path) -> PluginApi:
    """One ``PluginApi`` after every plugin of ``root``'s ``multiphy.json`` and every
    ``multiphy.plugins`` entry point has registered."""
    api = PluginApi()
    registered = set()
    for name in _config_modules(root):
        if name in registered:
            continue
        registered.add(name)
        try:
            importlib.import_module(name).register(api)
        except Exception as exc:
            raise ValueError(f'plugin {name}: {type(exc).__name__}: {exc}') from exc
    for ep in _entry_points():
        module = ep.value.split(':', 1)[0].strip()
        if module in registered:
            continue
        registered.add(module)
        try:
            ep.load()(api)
        except Exception as exc:
            raise ValueError(f'plugin {ep.name}: {type(exc).__name__}: {exc}') from exc
    return api
