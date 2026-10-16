# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: conformance suite every registered backend must pass, plus the grep gate and pilot reproduction (Task 13) | date: 2026-09-22]
"""Every registered backend must satisfy the contract on a synthetic input built from its own spec."""
import subprocess
import sys
import numpy as np
import pandas as pd
import pytest

from multiphy.contract import registry
from multiphy.contract.backend import RunContext
from multiphy.contract.inputs import CheckpointRef, Input, MatchingInput, RawWaveformBundle
from multiphy.contract.results import single_units


def synthetic_input(spec, tmp_path, n_per_session=3, sessions=('A', 'B')):
    ids = tuple(f'{s}-{i}' for s in sessions for i in range(1, n_per_session + 1))
    units = pd.DataFrame({'cellid': ids, 'session_id': [i.split('-')[0] for i in ids], 'su_type': ['SU'] * len(ids)})
    units.loc[len(units) - 1, 'su_type'] = 'MU'
    rng = np.random.default_rng(0)
    need = spec.requires | spec.optional
    pairs = pd.DataFrame([(a, b, rng.uniform()) for a in ids if a.startswith('A') for b in ids if b.startswith('B')],
                         columns=['cellid_a', 'cellid_b', 'p_global']) if Input.CANDIDATE_PAIRS in need else None
    kwargs = dict(pairs=pairs, group='g')
    if Input.CHECKPOINT in need:
        ck = tmp_path / 'ckpt'; ck.write_bytes(b'0'); kwargs['checkpoint'] = CheckpointRef(str(ck), 'np1')
    if Input.WAVEFORM_HALVES in need:
        geom = np.column_stack([np.zeros(384), np.arange(384) * 10.])
        bundle = RawWaveformBundle(ids, rng.normal(size=(len(ids), 82, 384, 2)), tuple(units.session_id),
                                   {s: geom for s in sessions}, tuple(np.sort(rng.uniform(0, 100, 50)) for _ in ids),
                                   {'raw_source': 'synthetic', 'split_method': 'halves', 'probe_type': 'np1'},
                                   peth=rng.normal(size=(len(ids), 6)) if Input.FINGERPRINT in need else None,
                                   peth_bin_keys=tuple(f'k{i}' for i in range(6)) if Input.FINGERPRINT in need else (),
                                   waveform_spike_counts=np.full((len(ids), 2), 25))
        return MatchingInput.from_bundle(units, bundle, **kwargs).validate()
    return MatchingInput(units, candidate_pairs=pairs, group='g').validate()


def _assert_conforms(backend, tmp_path):
    spec = backend.spec
    inp = synthetic_input(spec, tmp_path)
    assert spec.requires <= inp.available()
    result = backend.run(inp, spec.defaults(), RunContext('conf', tmp_path / 'work', inp.provenance)).validate(spec)
    expected, excluded = single_units(inp.units)
    assert set(result.units.cellid) == set(expected.cellid) and set(result.excluded_units) == set(excluded)
    assert result.run_id == 'conf'
    used = result.provenance.get('used_optional_inputs')
    assert used is not None and set(used) <= {i.value for i in spec.optional}


@pytest.mark.parametrize('spec', registry.list_specs(), ids=lambda s: s.id)
def test_backend_conforms(spec, tmp_path):
    backend = registry.get(spec.id)
    avail = backend.available()
    if not avail.ok:
        pytest.skip(f'{spec.id} unavailable: {avail.reason}')
    if Input.FEATURES in spec.requires or Input.TIMING in spec.requires:
        pytest.skip(f'{spec.id} needs archived features; covered by tests/test_functional_likelihood.py')  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: renamed test module (Task 3) | date: 2026-09-27]
    _assert_conforms(backend, tmp_path)


@pytest.fixture
def registered_example_backend():
    """Register docs/adding_a_backend.md's worked example for this one test only.

    ``registry.list_specs()`` is evaluated once, at collection time, to build
    ``test_backend_conforms``'s parametrize list; registering here (inside a test-scoped
    fixture) cannot and must not retroactively add to that list or leak into any other
    test module's own ``registry.list_specs()`` parametrize (``test_run_options.py`` has
    one too). So the example backend gets its own dedicated conformance test below,
    registered only for its duration and cleared immediately after.
    """
    from .example_backend import RandomBaselineBackend
    registry.register(RandomBaselineBackend())
    try:
        yield registry.get('random_baseline')
    finally:
        registry.clear_for_tests()


def test_example_backend_conforms(registered_example_backend, tmp_path):
    """The ``docs/adding_a_backend.md`` worked example must satisfy the same contract."""
    _assert_conforms(registered_example_backend, tmp_path)


def test_no_backend_ids_outside_backends_package():
    """A backend id must never be compared outside ``backends/``: new callers should hold a
    ``BackendSpec`` or the registry, not branch on a string. No file is exempt.
    """
    # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: multiday_runner no longer compares an id (Task 2) | date: 2026-09-27]
    pattern = r"== ?'(dant|unitmatch|deepunitmatch|functional_likelihood)'|in \('(dant|unitmatch|deepunitmatch|functional_likelihood)'"  # [AGENT EDIT | agent: claude-opus-5-5 | user: wingertj | reason: the registered ids (Task 3) | date: 2026-09-27]
    out = subprocess.run(['grep', '-rnE', pattern, 'multiphy', '--include=*.py',
                          '--exclude-dir=backends'],
                         capture_output=True, text=True)
    assert out.stdout == '', out.stdout

# [AGENT EDIT END]
