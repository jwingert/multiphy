"""
Pair identifiers, chains and triangle checks over verdict rows.

Verdicts are EDGES, one row per curated candidate pair, and chains across three
or more days are derived from them rather than stored. Storing chains would hide
disagreement: if day 1 to day 2 and day 2 to day 3 are both accepted but day 1
to day 3 is rejected, that triangle is information the curator should see, not
something to resolve silently. `chains()` computes connected components and
`inconsistencies()` reports the triangles.

Verdicts are stored in the run's `history.csv` (`curation.py`), not one CSV per
animal; this module only reads verdict rows handed to it.
"""
import hashlib

import pandas as pd

from multiphy import evidence as ev

VERDICTS = ('accept', 'reject', 'unsure')
# the per-animal table API (STORE_ROOT, store_path, load, record, upsert,
# decoy_error_rate and its column lists) is removed: curation.py is the one writer

def canonical(cellid_a, cellid_b):
    """Order a pair so one physical pairing has exactly one representation."""
    return tuple(sorted([str(cellid_a), str(cellid_b)]))


def match_id(cellid_a, cellid_b, pipeline_version=ev.PIPELINE_VERSION):
    a, b = canonical(cellid_a, cellid_b)
    h = hashlib.sha1(f'{a}|{b}|{pipeline_version}'.encode()).hexdigest()[:12]
    return f'um_{h}'


# ------------------------------------------------------------------- chains
def _row_sites(rows):
    """{cellid: siteid} from pair rows; every row, whatever its verdict, must name both of its sites."""
    def refuse(missing):
        ids = ', '.join(f'{a}/{b}' for a, b in zip(list(missing.cellid_a)[:5], list(missing.cellid_b)[:5]))
        raise ValueError(f'pair rows need siteid_a and siteid_b · missing for: {ids}')

    # an absent column refuses even for a table with no rows
    if 'siteid_a' not in rows or 'siteid_b' not in rows:
        refuse(rows)

    def blank(v):
        return v.isna() | (v.astype(str).str.strip() == '')
    missing = rows[blank(rows.siteid_a) | blank(rows.siteid_b)]
    if len(missing):
        refuse(missing)
    sites = dict(zip(rows.cellid_a.astype(str), rows.siteid_a.astype(str)))
    sites.update(zip(rows.cellid_b.astype(str), rows.siteid_b.astype(str)))
    return sites


def chains(df, verdict='accept'):
    """Connected components of accepted edges: a neuron followed across days.

    Returns the table with a `chain_id`, plus a summary of how many days each
    chain spans.
    """
    acc = df[df.verdict == verdict]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    sites = _row_sites(df)
    for _, r in acc.iterrows():
        union(r.cellid_a, r.cellid_b)
    comp = {c: find(c) for c in parent}
    label = {root: f'chain_{i:04d}' for i, root in enumerate(sorted(set(comp.values())))}
    members = {}
    for c, root in comp.items():
        members.setdefault(label[root], []).append(c)
    SUMMARY_COLS = ['chain_id', 'n_units', 'n_sites', 'members']
    summary = pd.DataFrame([
        dict(chain_id=k, n_units=len(v),
             n_sites=len({sites[str(m)] for m in v}),
             members=','.join(sorted(v)))
        for k, v in members.items()], columns=SUMMARY_COLS)
    # a table of rejections has no chains, and an empty frame built from an
    # empty list has no columns to sort by
    summary = summary.sort_values('n_units', ascending=False)
    out = df.copy()
    out['chain_id'] = out.cellid_a.map(lambda c: label.get(comp.get(c), ''))
    return out, summary


def inconsistencies(df):
    """Accepted A-B and B-C whose A-C was rejected, or was never offered.

    Not an error to fix automatically; a disagreement to show the curator.
    """
    acc = df[df.verdict == 'accept']
    rej = set()
    for _, r in df[df.verdict == 'reject'].iterrows():
        rej.add(canonical(r.cellid_a, r.cellid_b))
    sites = _row_sites(df)
    adj = {}
    for _, r in acc.iterrows():
        adj.setdefault(r.cellid_a, set()).add(r.cellid_b)
        adj.setdefault(r.cellid_b, set()).add(r.cellid_a)
    out = []
    for b, nbrs in adj.items():
        for a in nbrs:
            for c in nbrs:
                if a >= c:
                    continue
                if sites[str(a)] == sites[str(c)]:
                    continue                       # same day, not a triangle
                pair = canonical(a, c)
                if pair in rej:
                    out.append(dict(via=b, cellid_a=pair[0], cellid_b=pair[1],
                                    problem='closing edge rejected'))
                elif c not in adj.get(a, set()):
                    out.append(dict(via=b, cellid_a=pair[0], cellid_b=pair[1],
                                    problem='closing edge not accepted'))
    return pd.DataFrame(out).drop_duplicates() if out else pd.DataFrame(
        columns=['via', 'cellid_a', 'cellid_b', 'problem'])

