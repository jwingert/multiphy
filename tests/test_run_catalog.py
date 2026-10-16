import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

from multiphy.backends import result_from_pairs
from multiphy.run import catalog, run_dir


def _result(run_id, backend='unitmatch'):
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'match_score': [.95]})
    return result_from_pairs(backend, '1', units, pairs, run_id=run_id, provenance={'dataset_id': 'd'})


def _run(folder, run_id, *, backend='unitmatch', result=True, **manifest):
    folder.mkdir(parents=True)
    if result:
        run_dir.save_result(_result(run_id, backend), folder / 'result.json')
    run_dir.write_json(folder / 'run_manifest.json', {'status': 'completed', 'dataset_id': 'd', **manifest})
    return folder


def test_lists_this_archives_completed_runs_newest_first(tmp_path):
    dataset = tmp_path / 'archive' / 'dataset'
    dataset.mkdir(parents=True)
    runs = tmp_path / 'archive' / 'runs'
    _run(runs / 'old', 'old', finished_at='2026-09-18T10:00:00+00:00', archive_sha256='abc')
    _run(runs / 'new', 'new', backend='unitmatch', completed_at='2026-09-19T10:00:00+00:00')
    _run(runs / 'failed', 'failed', status='failed', finished_at='2026-09-20T10:00:00+00:00')
    _run(runs / 'no_result', 'no_result', result=False, finished_at='2026-09-20T10:00:00+00:00')
    _run(runs / 'foreign', 'foreign', dataset_id='other', finished_at='2026-09-20T10:00:00+00:00')
    _run(runs / 'other_archive', 'other_archive', archive_sha256='zzz', finished_at='2026-09-20T10:00:00+00:00')
    entries = catalog.find_runs(dataset, dataset_id='d', identity={'archive_sha256': 'abc', 'partition': None})
    assert [e.run_id for e in entries] == ['new', 'old']
    assert entries[0].backend_id == 'unitmatch' and entries[0].folder == runs / 'new'
    assert entries[1].finished == datetime(2026, 9, 18, 10, tzinfo=timezone.utc)


def test_start_time_then_file_time_date_the_run(tmp_path):
    started = _run(tmp_path / 'started', 'started', started_at='2026-09-17T08:00:00+00:00')
    undated = _run(tmp_path / 'undated', 'undated')
    assert catalog.read_run(started).finished == datetime(2026, 9, 17, 8, tzinfo=timezone.utc)
    mtime = datetime.fromtimestamp((undated / 'result.json').stat().st_mtime, timezone.utc)
    assert abs(catalog.read_run(undated).finished - mtime) < timedelta(seconds=1)


def test_collection_runs_beside_the_dataset_and_extra_dirs_are_listed_once(tmp_path):
    collection = tmp_path / 'heldout'
    dataset = _run(collection / 'collection_001', 'the_dataset_folder_itself')
    _run(collection / 'local_001', 'local_001', input_manifest_sha256='m')
    _run(collection / 'other_input', 'other_input', input_manifest_sha256='x')
    launched = _run(tmp_path / 'cache' / 'runs' / 'launched', 'launched')
    entries = catalog.find_runs(dataset, dataset_id='d', identity={'input_manifest_sha256': 'm'},
                                extra_dirs=[collection, launched.parent, tmp_path / 'missing'])
    assert sorted(e.run_id for e in entries) == ['launched', 'local_001']
    assert [e.run_id for e in catalog.find_runs(None, dataset_id='d', extra_dirs=[launched.parent])] == ['launched']


def test_an_unexpected_head_falls_back_to_a_full_load(tmp_path):
    folder = _run(tmp_path / 'sorted', 'sorted')
    document = json.loads((folder / 'result.json').read_text())
    (folder / 'result.json').write_text(json.dumps(document, sort_keys=True))
    assert catalog.read_run(folder).run_id == 'sorted'


def test_catalog_hides_format_1_and_unknown_ids(tmp_path):
    format_1 = _run(tmp_path / 'format_1', 'format_1')
    document = json.loads((format_1 / 'result.json').read_text())
    document['schema_version'] = 1
    (format_1 / 'result.json').write_text(json.dumps(document, indent=2))
    unknown = _run(tmp_path / 'unknown', 'unknown', backend='legacy')
    assert json.loads((unknown / 'result.json').read_text())['schema_version'] == 2
    assert catalog.read_run(format_1) is None
    assert catalog.read_run(unknown) is None


def test_listing_keeps_exactly_the_registered_format_2_runs(tmp_path):
    dataset = tmp_path / 'archive' / 'dataset'
    dataset.mkdir(parents=True)
    runs = tmp_path / 'archive' / 'runs'
    _run(runs / 'dant', 'dant', backend='dant')
    _run(runs / 'unitmatch', 'unitmatch', backend='unitmatch')
    _run(runs / 'constrained', 'constrained', backend='constrained')
    _run(runs / 'legacy_fresh', 'legacy_fresh', backend='legacy_fresh')
    _run(runs / 'unknown', 'unknown', backend='legacy')
    _run(runs / 'alias', 'alias', backend='current_local')
    for name, sort_keys in (('format_1', False), ('format_1_sorted', True)):
        folder = _run(runs / name, name)
        document = json.loads((folder / 'result.json').read_text())
        document['schema_version'] = 1
        (folder / 'result.json').write_text(json.dumps(document, indent=2, sort_keys=sort_keys))
    entries = catalog.find_runs(dataset, dataset_id='d')
    assert sorted(e.run_id for e in entries) == ['dant', 'unitmatch']


def test_unreadable_manifest_or_result_is_skipped(tmp_path):
    broken = _run(tmp_path / 'broken_manifest', 'x')
    (broken / 'run_manifest.json').write_text('{not json')
    garbage = _run(tmp_path / 'garbage', 'y', result=False)
    (garbage / 'result.json').write_text('[]')
    assert catalog.read_run(broken) is None and catalog.read_run(garbage) is None


def _entry(backend, when, folder='f'):
    return catalog.RunEntry(Path('/runs') / folder / 'result.json', backend, folder, when, {})


def test_labels_show_method_and_date_adding_time_then_folder_only_on_collision():
    day = datetime(2026, 9, 18, 14, 13)
    entries = [_entry('unitmatch', day, 'a'), _entry('unitmatch', day.replace(hour=9, minute=5), 'b'),
               _entry('dant', day, 'c'), _entry('dant', day - timedelta(days=1), 'd'),
               _entry('functional_likelihood', day, 'e'), _entry('functional_likelihood', day, 'g')]
    assert catalog.run_labels(entries) == [
        'UnitMatch · 09-18 14:13', 'UnitMatch · 09-18 09:05', 'DANT · 09-18', 'DANT · 09-17',
        'Functional likelihood · 09-18 · e', 'Functional likelihood · 09-18 · g']
    assert catalog.run_labels([_entry('plugin_backend', day)]) == ['plugin_backend · 09-18']
    tip = catalog.run_tooltip(entries[0])
    assert 'UnitMatch' in tip and '2026-09-18 14:13' in tip and '/runs/a' in tip
