"""Launcher plugins: ``<root>/multiphy.json`` names modules whose ``register(api)`` adds readers and commands."""
import json
import sys

from multiphy.launch import plugins
from multiphy.launch.__main__ import main
from multiphy.launch.plugins import load_plugins
from multiphy.sources.base import clear_readers_for_tests, get_reader

EXAMPLE = 'tests.example_plugin'


def _write_config(root):
    (root / 'multiphy.json').write_text(json.dumps({'plugins': [EXAMPLE]}), encoding='utf-8')


def test_config_plugin_adds_command(tmp_path, capsys):
    _write_config(tmp_path)
    try:
        code = main(['--root', str(tmp_path), 'hello', 'P', '--word', 'x'])
    finally:
        clear_readers_for_tests()
    assert code == 0
    assert capsys.readouterr().out == 'hello P x\n'


def test_config_plugin_registers_reader(tmp_path):
    _write_config(tmp_path)
    try:
        load_plugins(tmp_path)
        assert get_reader('example.echo').id == 'example.echo'
    finally:
        clear_readers_for_tests()


def test_no_config_loads_no_plugin(tmp_path, monkeypatch):
    monkeypatch.setattr(plugins, '_entry_points', lambda: ())
    monkeypatch.delitem(sys.modules, EXAMPLE, raising=False)
    code = main(['--root', str(tmp_path), 'hello', 'P'])
    assert code == 1
    assert EXAMPLE not in sys.modules

