import argparse
import json
import pytest

from multiphy.contract import registry
from multiphy.contract.backend import BackendSpec, Option, Provides
from multiphy.contract.errors import ContractViolation
from multiphy.contract.inputs import Input
from multiphy.contract.results import ScoreKind, TrackKind
from multiphy.run import options as opt

SPEC = BackendSpec('t', '1', frozenset({Input.UNITS}), frozenset(),
                   Provides(frozenset({ScoreKind.NONE}), frozenset({TrackKind.COMMON})),
                   (Option('threshold', 'float', .9), Option('n', 'int', 2), Option('flag', 'bool', False),
                    Option('mode', 'choice', 'a', choices=('a', 'b')), Option('settings', 'json', None),
                    Option('model', 'path', None), Option('name', 'str', None, required=True)))


def test_validate_fills_defaults_coerces_and_names_bad_option():
    out = opt.validate(SPEC, {'name': 'x', 'threshold': '0.5'})
    assert out['threshold'] == .5 and out['n'] == 2 and out['mode'] == 'a' and out['flag'] is False
    with pytest.raises(ContractViolation, match='threshold'):
        opt.validate(SPEC, {'name': 'x', 'threshold': 'abc'})
    with pytest.raises(ContractViolation, match='mode'):
        opt.validate(SPEC, {'name': 'x', 'mode': 'c'})
    with pytest.raises(ContractViolation, match='name'):
        opt.validate(SPEC, {})
    with pytest.raises(ContractViolation, match='bogus'):
        opt.validate(SPEC, {'name': 'x', 'bogus': 1})


def test_json_option_accepts_path_or_inline(tmp_path):
    p = tmp_path / 's.json'; p.write_text('{"a": 1}')
    assert opt.validate(SPEC, {'name': 'x', 'settings': str(p)})['settings'] == {'a': 1}
    assert opt.validate(SPEC, {'name': 'x', 'settings': '{"b": 2}'})['settings'] == {'b': 2}
    with pytest.raises(ContractViolation, match='settings'):
        opt.validate(SPEC, {'name': 'x', 'settings': '[1]'})


def test_path_option_must_exist(tmp_path):
    with pytest.raises(ContractViolation, match='model'):
        opt.validate(SPEC, {'name': 'x', 'model': str(tmp_path / 'missing.json')})


def roundtrip(spec, values):
    parser = argparse.ArgumentParser()
    opt.add_arguments(parser, spec)
    argv = opt.to_argv(spec, values)
    return opt.validate(spec, opt.from_namespace(spec, parser.parse_args(argv)))


def test_argv_roundtrip_synthetic(tmp_path):
    p = tmp_path / 'm.json'; p.write_text('{}')
    values = opt.validate(SPEC, {'name': 'x y', 'threshold': .7, 'n': 3, 'flag': True, 'mode': 'b',
                                 'settings': {'k': [1, 2]}, 'model': str(p)})
    assert roundtrip(SPEC, values) == values


@pytest.mark.parametrize('spec', registry.list_specs(), ids=lambda s: s.id)
def test_argv_roundtrip_every_builtin(spec, tmp_path):
    values = spec.defaults()
    for o in spec.options:
        if o.required and values.get(o.name) is None:
            f = tmp_path / o.name; f.write_text('x'); values[o.name] = str(f) if o.type == 'path' else 'np1'
    values = opt.validate(spec, values)
    assert roundtrip(spec, values) == values


def test_cli_wrong_type_fails_at_parse():
    parser = argparse.ArgumentParser()
    opt.add_arguments(parser, SPEC)
    with pytest.raises(SystemExit):
        parser.parse_args(['--name', 'x', '--threshold', 'abc'])
