"""``backend_refusal`` gives ``''`` for a backend that can run on the checked days, else the first refusal (D5)."""
from __future__ import annotations

from pathlib import Path
import typing
from typing import Optional

import pytest

from multiphy.contract import registry
from multiphy.contract.backend import Availability
from multiphy.contract.inputs import Input
from multiphy.launch import forms
from multiphy.launch.forms import backend_refusal
from multiphy.project.folder import Project

from .synthetic_day import extracted_project

FL = 'functional_likelihood'


def _open(project_dir: Path) -> Project:
    return Project.open(project_dir.parent, project_dir.name)


@pytest.fixture(scope='module')
def disjoint(tmp_path_factory) -> Project:
    """D1 with conditions A and B, D2 with C and D: no condition on every day."""
    return _open(extracted_project(tmp_path_factory.mktemp('disjoint'), conditions={'D2': ('C', 'D')}))


@pytest.fixture(scope='module')
def shared(tmp_path_factory) -> Project:
    """D1 and D2 with the default conditions A and B on both days."""
    return _open(extracted_project(tmp_path_factory.mktemp('shared')))


def test_backend_refusal_reasons(disjoint, shared, monkeypatch):
    assert backend_refusal(disjoint, ('D1', 'D2'), FL, interpreter=None) == ''
    assert backend_refusal(disjoint, ('D2',), FL, interpreter=None) == ''
    monkeypatch.setattr(registry.get('dant'), 'available', lambda: Availability(False, None, 'missing package'))
    assert backend_refusal(shared, ('D1', 'D2'), 'dant', interpreter=None) == (
        'unavailable here: missing package; no verified interpreter')
    assert backend_refusal(shared, ('D1', 'D2'), 'dant', interpreter='/lab/python') == ''


def test_backend_refusal_missing_inputs(monkeypatch):
    monkeypatch.setattr(forms, 'selection_inputs', lambda project, days: frozenset({Input.UNITS}))
    assert backend_refusal(None, ('D1', 'D2'), FL, interpreter=None) == 'missing inputs: features, geometry'
    assert backend_refusal(None, ('D1', 'D2'), 'deepunitmatch', interpreter=None) == (
        'missing inputs: geometry, waveform_halves')  # its checkpoint comes from an option


def test_backend_refusal_signature():
    assert typing.get_type_hints(backend_refusal) == {
        'project': Project, 'days': tuple, 'backend_id': str, 'interpreter': Optional[str], 'return': str}
