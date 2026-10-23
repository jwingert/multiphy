"""Report artifacts retain Unicode independently of the host text encoding."""
import ast
import io
import json
from pathlib import Path

import pytest

from multiphy import diagnostic_report as diagnostic
from multiphy import drift_validation as drift
from multiphy.benchmark import load_dataset
from multiphy.benchmark_collection import create_collection

from .test_collection import archive
from .test_diagnostic_report import fixture as diagnostic_fixture
from .test_drift_validation import _dataset


@pytest.fixture
def cp1252_default(monkeypatch, tmp_path):
    """Emulate Windows' legacy default without changing explicit encodings."""
    real_open = Path.open

    def open_cp1252(path, mode='r', buffering=-1, encoding=None, errors=None, newline=None):
        if 'b' not in mode and encoding in (None, 'locale'):
            encoding = 'cp1252'
        return real_open(path, mode, buffering, encoding, errors, newline)

    monkeypatch.setattr(Path, 'open', open_cp1252)
    # Python 3.10+ resolves the default before calling Path.open. Also simulate
    # cp1252 when the interpreter itself was started in UTF-8 mode.
    if hasattr(io, 'text_encoding'):
        real_text_encoding = io.text_encoding
        monkeypatch.setattr(io, 'text_encoding',
                            lambda encoding, stacklevel=2: 'cp1252' if encoding is None
                            else real_text_encoding(encoding, stacklevel))
    probe = tmp_path / 'encoding-probe.txt'
    with pytest.raises(UnicodeEncodeError):
        probe.write_text('“recording” ← −')
    probe.write_text('“recording” ← −', encoding='utf-8')
    with pytest.raises(UnicodeDecodeError):
        probe.read_text()


def test_collection_reports_are_utf8_with_cp1252_default(tmp_path, cp1252_default):
    source = archive(tmp_path, 'source', 'animal:1', 'test')
    name = 'collection-测试'
    target = create_collection([source], tmp_path / name, dataset_id=name, root=tmp_path)

    methods = (target / 'METHODS.md').read_text(encoding='utf-8')
    assert '← existing benchmark manifests' in methods
    assert name in methods
    assert json.loads((target / 'manifest.json').read_text(encoding='utf-8'))['dataset_id'] == name
    assert load_dataset(target).metadata['dataset_id'] == name


def test_diagnostic_reports_are_utf8_with_cp1252_default(tmp_path, monkeypatch, cp1252_default):
    dataset, result = diagnostic_fixture()
    dataset.metadata['dataset_id'] = 'diagnostic-测试'
    dataset.labels['source_unit_id'] = dataset.labels.source_unit_id.map({'a': '神经元a', 'b': '神经元b'})
    names = dict(diagnostic.NAMES, title='Waveforms − 波形')
    monkeypatch.setattr(diagnostic, 'NAMES', names)
    monkeypatch.setattr(diagnostic, 'load_dataset', lambda _: dataset)
    monkeypatch.setattr(diagnostic, 'load_result', lambda _: result)
    monkeypatch.setattr(diagnostic, '_plot_misses', lambda *args: None)
    target = diagnostic.write_report('dataset', 'result', output_dir='diagnostic-测试', root=tmp_path)

    report = (target / 'report.md').read_text(encoding='utf-8')
    assert 'Waveforms − 波形' in report and '−1 indicating opposite patterns' in report
    assert 'diagnostic-测试' in (target / 'METHODS.md').read_text(encoding='utf-8')
    assert '神经元b' in (target / 'missed_waveforms_legend.md').read_text(encoding='utf-8')
    # Generated Python follows Python 3's UTF-8 source convention as well.
    source = ast.parse((target / 'names.py').read_bytes())
    assignment = next(node for node in source.body if isinstance(node, ast.Assign))
    assert ast.literal_eval(assignment.value) == names


def test_drift_reports_are_utf8_with_cp1252_default(tmp_path, monkeypatch, cp1252_default):
    name = 'synthetic-测试'
    dataset = _dataset(name=name, n_neurons=16, seed=0)
    monkeypatch.setattr(drift, '_drift_figure', lambda *args: None)
    monkeypatch.setattr(drift, '_calibration_figure', lambda *args: None)
    target = tmp_path / 'drift'
    summary = drift.validate_datasets([dataset], target, shifts_um=(0,),
                                      backends=('functional_likelihood',), registrations=('matches',))

    assert summary.status.tolist() == ['ok']
    for filename in ('drift_recovery_legend.md', 'calibration_legend.md', 'METHODS.md'):
        text = (target / filename).read_text(encoding='utf-8')
        assert name in text
    assert '−' in (target / 'METHODS.md').read_text(encoding='utf-8')
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['inputs'][0]['name'] == name
