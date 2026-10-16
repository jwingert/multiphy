# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one project folder and one project.json per project (multiphy project folder Task 4) | date: 2026-09-25]
"""A project folder holds one project.json, validated on every read and write and replaced atomically under project.lock."""
from __future__ import annotations

import os

import pytest

from multiphy.project.folder import Project, ProjectLocked
from multiphy.project.manifest import new_day


def test_create_writes_manifest(tmp_path):
    project = Project.create(tmp_path, 'P')
    assert project.read()['kind'] == 'multiphy_project'
    assert project.read()['identity']['group'] == 'P'


def test_reserved_name_refused(tmp_path):
    with pytest.raises(ValueError, match='reserved'):
        Project.create(tmp_path, 'collections')
    assert not (tmp_path / 'collections').exists()


def test_bad_name_refused(tmp_path):
    with pytest.raises(ValueError, match='must match'):
        Project.create(tmp_path, '-x')
    assert not (tmp_path / '-x').exists()


def test_update_lock_held(tmp_path):
    project = Project.create(tmp_path, 'P')
    lock = project.path / 'project.lock'
    lock.write_text('h 1')
    with pytest.raises(ProjectLocked, match='h 1'):
        project.update(lambda manifest: None)
    assert lock.exists()


def _day(date):
    return new_day(date, 'user', {'sort': 'phy@1'}, {'day': 'D'}, {'params': {'role': 'sort'}})


def test_duplicate_dates_refused(tmp_path):
    project = Project.create(tmp_path, 'P')

    def add_two_days(manifest):
        manifest['days']['D2'] = _day('2026-01-01')
        manifest['days']['D1'] = _day('2026-01-01')

    with pytest.raises(ValueError, match='Days D1 and D2 share the date 2026-01-01'):
        project.update(add_two_days)
    assert project.read()['days'] == {}
    assert not (project.path / 'project.lock').exists()


def test_write_is_atomic(tmp_path, monkeypatch):
    project = Project.create(tmp_path, 'P')
    before = (project.path / 'project.json').read_bytes()

    def refuse(*args, **kwargs):
        raise OSError('replace refused')

    monkeypatch.setattr(os, 'replace', refuse)
    with pytest.raises(OSError, match='replace refused'):
        project.update(lambda manifest: manifest['identity'].update(subject='S'))
    assert (project.path / 'project.json').read_bytes() == before
    assert not list(project.path.glob('.project-*'))
    assert not (project.path / 'project.lock').exists()


def test_open_by_path(tmp_path):
    Project.create(tmp_path, 'P')
    assert Project.open(tmp_path, str(tmp_path / 'P')).name == 'P'
    with pytest.raises(ValueError, match='not a folder directly under'):
        Project.open(tmp_path, '/elsewhere/P')
# [AGENT EDIT END]
