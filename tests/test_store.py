import pandas as pd

from multiphy import store


def test_chains_from_verdict_rows():
    rows = pd.DataFrame([
        dict(cellid_a='D1:0', cellid_b='D2:0', siteid_a='D1', siteid_b='D2', verdict='accept'),
        dict(cellid_a='D2:0', cellid_b='D3:0', siteid_a='D2', siteid_b='D3', verdict='accept'),
        dict(cellid_a='D1:1', cellid_b='D2:1', siteid_a='D1', siteid_b='D2', verdict='reject'),
    ])
    _, summary = store.chains(rows)
    assert len(summary) == 1
    assert summary.iloc[0].n_units == 3
    assert summary.iloc[0].n_sites == 3
