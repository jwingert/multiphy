"""The project-folder guide's claims about conditions, ``init`` and runs hold in the code."""
from __future__ import annotations

import builtins
import io
import os
import re
from pathlib import Path

from multiphy.launch import commands
from multiphy.project.folder import Project
from multiphy.project.selection import Selection

from .synthetic_day import extracted_project, write_day

DOCS = Path(__file__).resolve().parents[1] / 'multiphy' / 'docs'


def test_condition_labels_are_stripped(tmp_path):
    project_dir = extracted_project(tmp_path, ('D1',), conditions={'D1': (' A ', 'B')})
    project = Project.open(project_dir.parent, project_dir.name)

    assert Selection(project, ('D1',)).conditions()['D1'] == frozenset({'A', 'B'})


def test_init_does_not_open_raw(tmp_path, monkeypatch):
    sd = write_day(tmp_path / 'in', 'D1')
    raw = os.path.realpath(sd.raw_path)
    attempts = []

    def refusing(real_open):
        def guarded(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)) and os.path.realpath(os.fsdecode(file)) == raw:
                attempts.append(file)
                raise AssertionError(f'init opened the raw file {file}')
            return real_open(file, *args, **kwargs)
        return guarded

    monkeypatch.setattr(builtins, 'open', refusing(builtins.open))
    monkeypatch.setattr(io, 'open', refusing(io.open))
    monkeypatch.setattr(os, 'open', refusing(os.open))
    sd.raw_path.chmod(0)
    try:
        commands.init_day(tmp_path / 'lab', 'P', day='D1', sort=str(sd.sort_dir), date=sd.date,
                          events=str(sd.events_path), raw=str(sd.raw_path), uv_per_bit=sd.uv_per_bit)
    finally:
        sd.raw_path.chmod(0o644)
        monkeypatch.undo()

    assert attempts == []
    record = Project.open(tmp_path / 'lab', 'P').read()['days']['D1']
    assert record['files']['raw']['sha256'] is None


def test_docs_name_project_runs():
    text = (DOCS / 'project_folder.md').read_text(encoding='utf-8')
    assert '### run' in text
    assert 'stripped of surrounding spaces' in text


def test_matching_guide_names_no_launch_dialog():
    text = re.sub(r'<!--.*?-->', '', (DOCS.parent / 'MATCHING_INTERFACE.md').read_text(encoding='utf-8'), flags=re.S)
    for phrase in ('dialog', 'Cancel', 'progress pane', 'visible cutoff', 'Select '):
        assert phrase not in text, phrase
    for flag in ('--threshold', '--native-threshold', '--ambiguity-margin', '--contradiction-threshold', '--python'):
        assert flag in text, flag


def test_matching_guide_states_python_precedence():
    text = re.sub(r'<!--.*?-->', '', (DOCS.parent / 'MATCHING_INTERFACE.md').read_text(encoding='utf-8'), flags=re.S)
    text = ' '.join(text.split())
    assert 'With `--python EXE`, `launch run` runs the backend in that interpreter' in text
    assert 'in its own interpreter when the backend is available there. Otherwise it runs it in `--python EXE`' not in text
