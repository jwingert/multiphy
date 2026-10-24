"""Native Windows publication rules, simulated without changing global sys.platform."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from multiphy import benchmark_collection as collection
from multiphy.benchmark import load_dataset
from tests.test_collection import archive


def _windows_directory_moves(monkeypatch):
    real_rename, real_replace = Path.rename, Path.replace

    def rename(source, target):
        if Path(target).exists():
            raise FileExistsError(target)
        return real_rename(source, target)

    def replace(source, target):
        if source.is_dir() and Path(target).exists():
            raise PermissionError('Windows cannot replace an existing directory')
        return real_replace(source, target)

    monkeypatch.setattr(collection, 'sys', SimpleNamespace(platform='win32'), raising=False)
    monkeypatch.setattr(Path, 'rename', rename)
    monkeypatch.setattr(Path, 'replace', replace)
    return rename


def test_collection_windows_publication_does_not_create_placeholder(tmp_path, monkeypatch):
    source = archive(tmp_path, 'source', 'animal:1', 'test')
    _windows_directory_moves(monkeypatch)
    target = collection.create_collection([source], tmp_path / 'collection',
                                          dataset_id='collection', root=tmp_path)
    assert load_dataset(target).metadata['dataset_id'] == 'collection'
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    assert set(before) == {'manifest.json', 'METHODS.md'}
    assert not list(tmp_path.glob('.collection-*'))
    with pytest.raises(FileExistsError):
        collection.create_collection([source], target, dataset_id='collection', root=tmp_path)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert not list(tmp_path.glob('.collection-*'))


@pytest.mark.parametrize('native', [False, pytest.param(True, marks=pytest.mark.skipif(
    sys.platform != 'win32', reason='Native Windows directory rename semantics'))])
@pytest.mark.parametrize('completed', [False, True])
def test_collection_windows_publication_preserves_racing_writer(tmp_path, monkeypatch, completed, native):
    source = archive(tmp_path, 'source', 'animal:1', 'test')
    rename = Path.rename if native else _windows_directory_moves(monkeypatch)
    target = tmp_path / 'collection'
    winner = {'manifest.json': b'{"winner": true}', 'METHODS.md': b'Another completed output'} if completed else {}

    def publish_after_other_writer(stage, destination):
        assert Path(destination) == target
        assert not target.exists()
        target.mkdir()
        for name, data in winner.items():
            (target / name).write_bytes(data)
        return rename(stage, destination)

    monkeypatch.setattr(Path, 'rename', publish_after_other_writer)
    with pytest.raises(FileExistsError):
        collection.create_collection([source], target, dataset_id='collection', root=tmp_path)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == winner
    assert not list(tmp_path.glob('.collection-*'))


def test_collection_windows_publication_cleans_stage_on_refusal(tmp_path, monkeypatch):
    source = archive(tmp_path, 'source', 'animal:1', 'test')
    _windows_directory_moves(monkeypatch)
    target = tmp_path / 'collection'
    other_stage = tmp_path / '.collection-other-writer'
    other_stage.mkdir()
    (other_stage / 'keep.txt').write_bytes(b'Another writer owns this stage')

    def refuse(stage, destination):
        assert stage.is_dir() and not Path(destination).exists()
        raise PermissionError('simulated publication refusal')

    monkeypatch.setattr(Path, 'rename', refuse)
    with pytest.raises(PermissionError, match='simulated publication refusal'):
        collection.create_collection([source], target, dataset_id='collection', root=tmp_path)
    assert not target.exists()
    assert list(tmp_path.glob('.collection-*')) == [other_stage]
    assert (other_stage / 'keep.txt').read_bytes() == b'Another writer owns this stage'
