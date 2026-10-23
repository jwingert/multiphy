"""Methods documents must work even when the OS default cannot encode arrows."""
from pathlib import Path

from .test_benchmark import dataset
from multiphy.benchmark import save_dataset, load_dataset


def test_benchmark_methods_with_legacy_default_encoding(tmp_path, monkeypatch):
    original = Path.write_text

    def legacy_write(path, data, encoding=None, errors=None):
        return original(path, data, encoding=encoding or 'cp1252', errors=errors)

    monkeypatch.setattr(Path, 'write_text', legacy_write)
    target = save_dataset(dataset(), tmp_path / 'archive', cache_root=tmp_path)
    assert '\u2190' in (target / 'METHODS.md').read_text(encoding='utf-8')
    load_dataset(target)
