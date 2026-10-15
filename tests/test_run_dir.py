# [AGENT EDIT START | agent: claude-fable-5-1 | user: wingertj | reason: TDD coverage for cache config env var and result format 2 with v1 reader | date: 2026-09-22]
import json
import numpy as np
import pandas as pd
import pytest

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: fixture built with result_from_pairs (Task 3 Do 10) | date: 2026-09-27]
from nems_lbhb.unit_match.backends import result_from_pairs
from nems_lbhb.unit_match.contract.results import TrackKind
# [AGENT EDIT END]
from nems_lbhb.unit_match.run import config, run_dir


def test_cache_root_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv('MULTIPHY_CACHE_ROOT', str(tmp_path))
    assert config.cache_root() == tmp_path.resolve()
    monkeypatch.delenv('MULTIPHY_CACHE_ROOT')
    assert str(config.cache_root()) == config.CACHE_ROOT
    assert config.cache_root(tmp_path / 'x') == (tmp_path / 'x').resolve()


def test_resolve_rejects_escape(tmp_path):
    with pytest.raises(ValueError):
        config.resolve_cache_path('../outside', root=tmp_path)


def _result():
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a fabricated unitmatch result is built with result_from_pairs (Task 3 Do 10; review r3) | date: 2026-09-27]
    units = pd.DataFrame({'cellid': ['A-1', 'B-1'], 'session_id': ['A', 'B'], 'su_type': ['SU', 'SU']})
    pairs = pd.DataFrame({'cellid_a': ['A-1'], 'cellid_b': ['B-1'], 'match_score': [.95]})
    return result_from_pairs('unitmatch', '1', units, pairs, run_id='r1', provenance={'dataset_id': 'd'})
    # [AGENT EDIT END]


def test_v2_roundtrip_and_no_overwrite(tmp_path):
    p = tmp_path / 'result.json'
    run_dir.save_result(_result(), p)
    doc = json.loads(p.read_text())
    assert doc['schema_version'] == 2 and doc['result']['score']['kind'] == 'uncalibrated_score'
    back = run_dir.load_result(p)
    assert back.digest() == _result().digest() and back.track_kind is TrackKind.COMMON
    with pytest.raises(FileExistsError):
        run_dir.save_result(_result(), p)


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: format 1 is refused, not mapped (Task 1) | date: 2026-09-27]
def test_load_result_refuses_format_1(tmp_path):
    p = tmp_path / 'result.json'
    run_dir.save_result(_result(), p)
    doc = json.loads(p.read_text())
    doc['schema_version'] = 1
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='unsupported matching result format or schema version'):
        run_dir.load_result(p)
# [AGENT EDIT END]


def test_corrupt_digest_rejected(tmp_path):
    p = tmp_path / 'result.json'
    run_dir.save_result(_result(), p)
    doc = json.loads(p.read_text()); doc['result']['run_id'] = 'tampered'
    p.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match='digest'):
        run_dir.load_result(p)

# [AGENT EDIT END]
