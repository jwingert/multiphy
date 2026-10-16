import numpy as np
import pytest

from multiphy.contract import errors
from multiphy.contract.provenance import Provenance, json_safe, payload_digest


def test_json_safe_normalizes_numpy_and_rejects_executables():
    assert json_safe({'a': np.float64(1.5), 'b': np.array([1, 2]), 'c': float('nan')}) == {'a': 1.5, 'b': [1, 2], 'c': None}
    with pytest.raises(ValueError):
        json_safe({'f': lambda: 0})
    with pytest.raises(ValueError):
        json_safe({1: 'non-string key'})


def test_json_safe_preserves_bool_and_handles_none():
    assert json_safe({'t': True}) == {'t': True}
    assert json_safe(True) is True
    assert json_safe(None) is None
    assert json_safe(False) is False


def test_provenance_roundtrip_keeps_unknown_keys_in_extra():
    p = Provenance.from_json({'dataset_id': 'd', 'group': 'g', 'assignment': 'native_intermediate', 'threshold': .5})
    assert p.dataset_id == 'd' and p.group == 'g'
    assert p.extra == {'assignment': 'native_intermediate', 'threshold': .5}
    assert Provenance.from_json(p.to_json()) == p
    assert 'extra' not in p.to_json() and p.to_json()['assignment'] == 'native_intermediate'


def test_provenance_from_json_none():
    assert Provenance.from_json(None) == Provenance()


def test_provenance_get_reads_typed_and_extra_fields():
    p = Provenance(dataset_id='d', extra={'x': 1})
    assert p.get('dataset_id') == 'd'
    assert p.get('group', 'dflt') == 'dflt'
    assert p.get('x') == 1
    assert p.get('missing') is None


def test_provenance_merged_merges_extra_dicts():
    p = Provenance(dataset_id='d', extra={'a': 1})
    q = p.merged(group='g', extra={'b': 2})
    assert p.extra == {'a': 1} and q.extra == {'a': 1, 'b': 2} and q.dataset_id == 'd' and q.group == 'g'


def test_payload_digest_is_order_independent():
    assert payload_digest({'a': 1, 'b': [2]}) == payload_digest({'b': [2], 'a': 1})


def test_missing_inputs_message_names_backend_group_and_inputs():
    err = errors.MissingInputs('dant', 'well16', frozenset({'spike_times', 'geometry'}))
    assert err.backend_id == 'dant' and err.group == 'well16'
    assert 'dant' in str(err) and 'well16' in str(err) and 'geometry' in str(err) and 'spike_times' in str(err)
    assert isinstance(err, ValueError)
    assert issubclass(errors.BackendUnavailable, RuntimeError)
    assert issubclass(errors.ContractViolation, ValueError)
