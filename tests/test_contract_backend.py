import pytest

from multiphy.contract import registry
from multiphy.contract.backend import (Availability, BackendSpec, Option, Provides,
                                                   COMMON_TRACKER_OPTIONS, NATIVE_TRACKING_OPTIONS)
from multiphy.contract.inputs import Input
from multiphy.contract.results import ScoreKind, TrackKind
from multiphy.contract import names


class Fake:
    spec = BackendSpec('fake', '0', frozenset({Input.UNITS}), frozenset(),
                       Provides(frozenset({ScoreKind.NONE}), frozenset({TrackKind.COMMON})), (Option('k', 'int', 1),))

    def available(self):
        return Availability(True, '0', None)

    def run(self, inp, options, ctx):
        raise NotImplementedError


@pytest.fixture(autouse=True)
def isolated_registry():
    registry.clear_for_tests()
    yield
    registry.clear_for_tests()


def test_register_get_list_and_duplicate_rejection():
    registry.register(Fake())
    assert registry.get('fake').spec.id == 'fake'
    assert 'fake' in [s.id for s in registry.list_specs()]
    with pytest.raises(ValueError, match='fake'):
        registry.register(Fake())


def test_builtin_cannot_be_replaced():
    class B(Fake):
        spec = BackendSpec('b', '0', frozenset({Input.UNITS}), frozenset(), Fake.spec.provides, (), builtin=True)
    registry.register(B())
    with pytest.raises(ValueError, match='built-in'):
        registry.register(B())


def test_unknown_backend_names_known_ones():
    registry.register(Fake())
    with pytest.raises(KeyError, match='fake'):
        registry.get('nope')


def test_spec_rejects_bad_option_and_unit_requirement():
    with pytest.raises(ValueError, match='UNITS'):
        BackendSpec('x', '0', frozenset(), frozenset(), Fake.spec.provides, ())
    with pytest.raises(ValueError, match='choices'):
        BackendSpec('x', '0', frozenset({Input.UNITS}), frozenset(), Fake.spec.provides, (Option('t', 'choice'),))
    with pytest.raises(ValueError, match='duplicate'):
        BackendSpec('x', '0', frozenset({Input.UNITS}), frozenset(), Fake.spec.provides, (Option('t', 'int'), Option('t', 'int')))


def test_shared_option_groups():
    assert [o.name for o in COMMON_TRACKER_OPTIONS] == ['threshold', 'contradiction_threshold', 'ambiguity_margin']
    assert [o.name for o in NATIVE_TRACKING_OPTIONS] == ['tracking', 'native_threshold']
    assert dict((o.name, o.default) for o in COMMON_TRACKER_OPTIONS) == {'threshold': .9, 'contradiction_threshold': .1, 'ambiguity_margin': .02}


def test_names_cover_every_builtin_and_fail_loudly():
    for key in ('functional_likelihood', 'unitmatch', 'dant', 'deepunitmatch'):
        assert names.display('backend', key)
    for old in ('legacy', 'constrained', 'legacy_fresh'):
        with pytest.raises(KeyError):
            names.display('backend', old)
    assert names.display('tracking', 'native_intermediate').startswith('Published rule')
    assert names.display('evidence', 'combined')
    with pytest.raises(KeyError):
        names.display('backend', 'current_local')
    spec = registry.get('functional_likelihood').spec
    assert names.backend_label(spec.id) == 'Functional likelihood — firing, waveform and optional event-response likelihood ratios'
class _EntryPoint:
    def __init__(self, name, target):
        self.name, self._target = name, target

    def load(self):
        if isinstance(self._target, Exception):
            raise self._target
        return self._target


class _EntryPoints(dict):
    """Both shapes of ``importlib.metadata.entry_points()``: a group mapping (3.9) and ``select``."""
    def select(self, group):
        return self.get(group, ())


def _install_entry_points(monkeypatch, *eps):
    import importlib.metadata
    monkeypatch.setattr(importlib.metadata, 'entry_points',
                        lambda: _EntryPoints({registry.ENTRY_POINT_GROUP: list(eps)}))


def test_entry_point_backend_is_listed_runnable_and_on_the_cli(monkeypatch, capsys):
    from .example_backend import RandomBaselineBackend
    from multiphy.run.__main__ import main
    _install_entry_points(monkeypatch, _EntryPoint('random_baseline', lambda: RandomBaselineBackend()))
    assert 'random_baseline' in [s.id for s in registry.list_specs()]
    assert 'functional_likelihood' in [s.id for s in registry.list_specs()]  # built-ins still come first
    assert registry.get('random_baseline').spec.id == 'random_baseline'
    assert main(['list-backends']) == 0
    assert 'random_baseline' in capsys.readouterr().out


def test_failing_entry_point_is_recorded_and_the_others_still_load(monkeypatch):
    from .example_backend import RandomBaselineBackend
    _install_entry_points(monkeypatch, _EntryPoint('broken', ImportError('no module named plugin_pkg')),
                          _EntryPoint('random_baseline', lambda: RandomBaselineBackend()))
    ids = [s.id for s in registry.list_specs()]
    assert 'random_baseline' in ids and 'broken' not in ids
    assert 'no module named plugin_pkg' in registry.ENTRY_POINT_ERRORS['broken']
    registry.clear_for_tests()
    assert registry.ENTRY_POINT_ERRORS == {}
def test_real_days_opt_outs():
    assert Fake.spec.real_days is True
    specs = registry.list_specs()
    assert {s.id for s in specs if not s.real_days} == set()
