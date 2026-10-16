# [AGENT EDIT START | agent: codex | user: wingertj | reason: independently verify streaming actual-day extraction and raw-summary reuse without snippet files | date: 2026-09-18]
import builtins
import copy
import json
from pathlib import Path

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: numpy and pandas were used only by the dropped multiday tests (Plan B Task 8) | date: 2026-09-28]
# [AGENT EDIT END]
import pytest

from tests.test_multiday_reuse import source, reused
from multiphy.benchmark import load_dataset, save_dataset
from multiphy.extract_benchmark import extract_manifest
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: multiday.py is retired; no multiday import (Plan B Task 8) | date: 2026-09-28]
# [AGENT EDIT END]
from multiphy.waveform_preprocessing import SharedWaveformPreprocessor


def shared_source(tmp_path, session='DAY001a'):
    manifest, spec, options, _, _ = source(tmp_path, session=session)
    spec['waveform_preprocessing'] = SharedWaveformPreprocessor(1000, 4).metadata()
    contents = json.loads(manifest.read_text())
    contents['recordings'] = [spec]
    manifest.write_text(json.dumps(contents))
    return manifest, spec, options


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the multiday streaming tests and their value-comparison helper left with multiday.py (Plan B Task 8) | date: 2026-09-28]
# [AGENT EDIT END]
@pytest.mark.parametrize('storage', ['snippets', 'streaming_summary'])
def test_reuse_accepts_both_operational_storage_modes_and_still_checks_transforms(tmp_path, monkeypatch, storage):
    manifest, spec, options = shared_source(tmp_path)
    archive = extract_manifest(manifest, root=tmp_path / 'cache', waveform_storage=storage)
    original_open = builtins.open
    raw_path = (tmp_path / spec['raw_voltage']).resolve()

    def no_voltage_open(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            assert Path(file).resolve() != raw_path, 'Summary reuse must not read raw voltage'
        return original_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, 'open', no_voltage_open)
    summary = reused(archive, manifest, spec, options)
    assert len(summary['features']) == 2
    dataset = load_dataset(archive)
    for change in ({'padding_seconds': .1}, {'unknown_filter': 'different'}, {'output_dtype': 'float64'}):
        changed = copy.deepcopy(dataset)
        changed.metadata['sources'][0]['provenance']['waveform_preprocessing'].update(change)
        location = save_dataset(changed, tmp_path / 'cache' / next(iter(change)), cache_root=tmp_path / 'cache')
        with pytest.raises(ValueError, match='preprocessing differs'):
            reused(location, manifest, spec, options)
# [AGENT EDIT END]
