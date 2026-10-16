import pandas as pd
import pytest

from multiphy.contract import registry
from multiphy.contract.results import ScoreKind, TrackKind
from multiphy.backends.local import tracker


def units(*ids):
    return pd.DataFrame({'cellid': ids, 'session_id': [i.split('-')[0] for i in ids], 'su_type': ['SU'] * len(ids)})


def pairs(*edges):
    return pd.DataFrame(edges, columns=['cellid_a', 'cellid_b', 'p_global'])


def test_builtins_are_registered_with_declared_requirements():
    specs = {s.id: s for s in registry.list_specs()}
    assert 'functional_likelihood' in specs and not {'legacy', 'constrained', 'legacy_fresh'} & set(specs)
    assert specs['functional_likelihood'].provides.track_kinds == {TrackKind.COMMON}
    assert specs['functional_likelihood'].builtin


def test_common_tracker_through_result_from_pairs():
    u = units('A-1', 'B-1', 'C-1')
    p = pairs(('A-1', 'B-1', .99), ('B-1', 'C-1', .97), ('A-1', 'C-1', .01))
    r = tracker.result_from_pairs('unitmatch', '1', u, p, score_column='p_global', run_id='run1')
    assert r.tracks == (('A-1', 'B-1'),) and r.unmatched == ('C-1',)
    assert r.pairs.iloc[1].decision == 'measured_contradiction'
    assert r.score.kind is ScoreKind.UNCALIBRATED and r.score.column == 'p_global'
    assert r.run_id == 'run1' and r.tracking_rule == 'mutual_margin_best_first'


def test_legacy_backend_gone():
    with pytest.raises(KeyError):
        registry.get('legacy')


def test_run_backend_unknown_alias():
    from multiphy import backends as b
    with pytest.raises(KeyError):
        b.run_backend('current_local', units('A-1', 'B-1'), pairs(('A-1', 'B-1', .99)))
