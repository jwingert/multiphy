"""``launch.forms``: form fields read from the CLI parser, and the argv a filled form sends."""
import argparse
import inspect
import json
import typing
from types import SimpleNamespace
from typing import Optional

import pytest

from multiphy.contract import registry
from multiphy.contract.backend import Option
from multiphy.launch.__main__ import _parse, build_parser
from multiphy.launch.forms import (BOUND, Field, command_argv, commands_of, form_fields,
                                               has_value)
from multiphy.launch.plugins import load_plugins
from multiphy.contract.inputs import RunEvidence
from multiphy.launch.forms import EVENT_CHOICES, selection_evidence
from multiphy.project.folder import Project
from multiphy.run import options as opt
from multiphy.sources.base import clear_readers_for_tests


def _tuples(fields):
    return [(f.dest, f.label, f.kind, f.browse) for f in fields]


def _run_fields(backend_id):
    spec = registry.get(backend_id).spec
    sub = commands_of(build_parser(None, spec))['run']
    return sub, form_fields(sub, BOUND['run'], spec)


def test_form_fields_from_parser():
    commands = commands_of(build_parser())
    assert list(commands)[:5] == ['init', 'extract', 'status', 'run', 'export']
    fields = {name: form_fields(commands[name], BOUND[name]) for name in ('init', 'extract', 'status', 'export')}
    assert _tuples(fields['init']) == [
        ('project', 'project', 'text', None), ('from_dir', 'from', 'text', 'dir'), ('day', 'day', 'text', None),
        ('sort', 'sort', 'text', 'dir'), ('date', 'date', 'text', None), ('events', 'events', 'text', 'open'),
        ('raw', 'raw', 'text', 'open'), ('uv_per_bit', 'uv per bit', 'text', None),
        ('probe', 'probe', 'text', None)]
    assert _tuples(fields['extract']) == [('tier', 'tier', 'choice', None), ('scratch', 'scratch', 'text', 'dir')]
    assert _tuples(fields['status']) == [('verify', 'verify', 'flag', None)]
    assert _tuples(fields['export']) == [('run', 'run', 'text', None), ('out', 'out', 'text', 'save')]
    required = [(name, f.dest) for name, rows in fields.items() for f in rows if f.required]
    assert required == [('init', 'project'), ('export', 'run')]


def test_backend_fields_and_placeholders():
    _sub, fields = _run_fields('functional_likelihood')
    assert _tuples(fields)[0] == ('python', 'python', 'text', 'open')
    by_dest = {f.dest: f for f in fields}
    assert (by_dest['threshold'].kind, by_dest['threshold'].placeholder) == ('text', '0.9')
    registration = by_dest['registration']
    assert registration.kind == 'choice'
    assert registration.choices == ('matches', 'profile', 'none')
    assert registration.placeholder == 'matches'
    sub, fields = _run_fields('deepunitmatch')
    by_dest = {f.dest: f for f in fields}
    assert by_dest['checkpoint'].browse == 'open'
    assert by_dest['checkpoint'].required is True
    assert by_dest['checkpoint_probe'].required is True
    assert by_dest['device'].required is False
    assert by_dest['params'].placeholder == ''
    assert '--checkpoint FILE' in sub.format_help()


def test_bool_option_switch():
    parser = argparse.ArgumentParser()
    spec = SimpleNamespace(options=(Option('verbose', 'bool', True, 'Print more'),))
    opt.add_arguments(parser, spec)
    fields = form_fields(parser, frozenset(), spec)
    assert [(f.dest, f.label, f.kind) for f in fields] == [('verbose', 'verbose', 'switch')]
    assert fields[0].choices == ('--verbose', '--no-verbose')
    assert fields[0].placeholder == 'True'
    assert command_argv('x', fields, {'verbose': '--no-verbose'})[-1:] == ['--no-verbose']
    assert command_argv('x', fields, {'verbose': ''}) == ['x']


def test_unsupported_actions_tagged():
    parser = argparse.ArgumentParser()
    parser.add_argument('project')
    parser.add_argument('--tag', action='append')
    parser.add_argument('--verbose', action='count')
    parser.add_argument('--pair', nargs=2)
    parser.add_argument('--mode', action='store_const', const='x')
    parser.add_argument('--quiet', action='store_false')
    parser.add_argument('--maybe', nargs='?')
    parser.add_argument('files', nargs='+')
    parser.add_argument('--pick', choices=['a', 'b'], nargs='+')
    parser.add_argument('--n', type=int, choices=[1, 2])
    parser.add_argument('--names', nargs='+')
    fields = form_fields(parser, frozenset())
    assert [f.kind for f in fields] == ['text', 'unsupported', 'unsupported', 'unsupported', 'unsupported', 'flag',
                                        'unsupported', 'unsupported', 'unsupported', 'choice', 'list']
    by_dest = {f.dest: f for f in fields}
    assert by_dest['tag'].help == '_AppendAction has no form field; run it from the command line'
    assert by_dest['maybe'].help == "nargs '?' has no form field; run it from the command line"
    assert by_dest['files'].help == 'a positional list has no form field; run it from the command line'
    assert by_dest['pick'].help == 'a multiple choice has no form field; run it from the command line'
    assert by_dest['n'].choices == ('1', '2')
    with pytest.raises(ValueError) as caught:
        command_argv('x', fields, {'project': 'P'})
    assert str(caught.value) == '--tag runs only from the command line'
    export = commands_of(build_parser())['export']
    export_fields = form_fields(export, BOUND['export'])
    with pytest.raises(ValueError) as caught:
        command_argv('export', export_fields, {}, project='P')
    assert str(caught.value) == 'run is required'
    class _CustomTrue(argparse._StoreTrueAction):
        pass

    custom = argparse.ArgumentParser()
    custom.add_argument('--on', dest='on', action=_CustomTrue)
    custom.add_argument('--no-on', dest='on', action='store_false')
    custom_fields = form_fields(custom, frozenset())
    assert [(f.dest, f.kind) for f in custom_fields] == [('on', 'unsupported'), ('on', 'flag')]
    assert custom_fields[0].help == '_CustomTrue has no form field; run it from the command line'
    with pytest.raises(ValueError) as caught:
        command_argv('x', custom_fields, {'on': '--no-on'})
    assert str(caught.value) == '--on runs only from the command line'


def test_required_flag_needs_check():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true', required=True)
    parser.add_argument('--mode', choices=['a', 'b'], required=True)
    parser.add_argument('--names', nargs='+')
    fields = form_fields(parser, frozenset())
    with pytest.raises(ValueError) as caught:
        command_argv('y', fields[:2], {'force': False, 'mode': 'a'})
    assert str(caught.value) == 'force is required'
    with pytest.raises(ValueError) as caught:
        command_argv('y', fields[:2], {'force': True, 'mode': 'c'})
    assert str(caught.value) == 'mode is required'
    assert command_argv('y', fields[:2], {'force': True, 'mode': 'a'}) == ['y', '--force', '--mode', 'a']
    names = fields[2]
    assert names.kind == 'list'
    assert has_value(names, '  ') is False
    assert has_value(names, 'a b') is True


def test_browse_by_metavar_exact():
    parser = argparse.ArgumentParser()
    for index, metavar in enumerate(('DIR', 'FILE', 'CSV', 'EXE', 'PATH', 'Dir', 'DIRS', None)):
        parser.add_argument(f'--opt{index}', metavar=metavar)
    parser.add_argument('--many', nargs='+', metavar='FILE')
    fields = form_fields(parser, frozenset())
    assert [f.browse for f in fields[:8]] == ['dir', 'open', 'open', 'open', 'save', None, None, None]
    assert (fields[8].kind, fields[8].browse) == ('list', None)


def _same_parse(form_argv, typed_argv, commands=None):
    assert form_argv == typed_argv
    assert vars(_parse(form_argv, commands)) == vars(_parse(typed_argv, commands))


def test_form_argv_parses_like_typed(tmp_path):
    commands = commands_of(build_parser())
    init = form_fields(commands['init'], BOUND['init'])
    _same_parse(command_argv('init', init, {'project': 'P', 'from_dir': '/x', 'uv_per_bit': '0.195'}),
                ['init', 'P', '--from', '/x', '--uv-per-bit', '0.195'])
    extract = form_fields(commands['extract'], BOUND['extract'])
    _same_parse(command_argv('extract', extract, {}, project='P', days=('D2',)), ['extract', 'P', '--day', 'D2'])
    _sub, run = _run_fields('functional_likelihood')
    values = {'threshold': '0.8', 'registration': 'profile', 'python': ''}
    typed = ['run', 'P', '--backend', 'functional_likelihood', '--days', 'D1', 'D2', '--registration', 'profile',
             '--threshold', '0.8']
    form = command_argv('run', run, values, project='P', days=('D1', 'D2'), backend='functional_likelihood')
    _same_parse(form, typed)
    assert _parse(form).options == {'registration': 'profile', 'threshold': 0.8}
    one_day = command_argv('run', run, values, project='P', days=('D1',), backend='functional_likelihood')
    _same_parse(one_day, ['run', 'P', '--backend', 'functional_likelihood', '--day', 'D1', '--registration',
                          'profile', '--threshold', '0.8'])
    status = form_fields(commands['status'], BOUND['status'])
    _same_parse(command_argv('status', status, {'verify': True}, project='P'), ['status', 'P', '--verify'])
    (tmp_path / 'multiphy.json').write_text(json.dumps({'plugins': ['tests.example_plugin']}),
                                            encoding='utf-8')
    try:
        plugin_commands = load_plugins(tmp_path).commands
    finally:
        clear_readers_for_tests()
    hello = form_fields(commands_of(build_parser(plugin_commands))['hello'], BOUND.get('hello', frozenset()))
    _same_parse(command_argv('hello', hello, {'project': 'P', 'word': 'hi'}), ['hello', 'P', '--word', 'hi'],
                plugin_commands)


def test_forms_signatures():
    assert typing.get_type_hints(build_parser) == {'plugin_commands': Optional[dict],
                                                   'return': argparse.ArgumentParser}
    assert typing.get_type_hints(commands_of) == {'parser': argparse.ArgumentParser, 'return': dict}
    assert typing.get_type_hints(form_fields) == {'subparser': argparse.ArgumentParser, 'bound': frozenset,
                                                  'return': list}
    assert typing.get_type_hints(command_argv) == {'command': str, 'fields': list, 'values': dict,
                                                   'project': Optional[str], 'days': tuple,
                                                   'backend': Optional[str], 'return': list}
    assert typing.get_type_hints(has_value) == {'field': Field, 'return': bool}
    assert typing.get_type_hints(Field) == {'dest': str, 'flag': Optional[str], 'label': str, 'help': str,
                                            'kind': str, 'choices': tuple, 'required': bool,
                                            'browse': Optional[str], 'placeholder': str}


def test_selection_evidence_signature():
    parameters = inspect.signature(selection_evidence).parameters
    assert list(parameters) == ['project', 'days', 'backend_id', 'min_repeats']
    assert parameters['min_repeats'].kind is inspect.Parameter.KEYWORD_ONLY
    assert typing.get_type_hints(selection_evidence) == {'project': Project, 'days': tuple, 'backend_id': str,
                                                         'min_repeats': int, 'return': Optional[RunEvidence]}
    assert EVENT_CHOICES == {'feature_stream': frozenset({'stimulus'})}
