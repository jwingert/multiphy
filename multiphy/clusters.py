# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the module keeps identifiers, day lookup, scoring and track description; curation history lives in curation.py (run-curation Task 9) | date: 2026-10-05]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names proposed links as a review reason (native-same-day Task 8 fix round 1) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a proposed link's partner may be on the same day or another day; tag wraps the whole docstring (native-same-day Task 8, final review fix round 1) | date: 2026-10-12]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: an accepted or rejected track leaves the review queue for every flag (D35) (native-same-day Task 12) | date: 2026-10-12]
"""
Tracked units: clusters of cellids that are one neuron followed across days.

Tracks are proposed by the matcher: each across-day run's `result.json` holds
the tracks it found. This module names and describes them -- a deterministic
`cluster_id` from membership, each unit's day from its siteid, the pair score a
track was judged on, and the two flags that send a track to review
(`needs_review`):

  same-day collision   the track claims one neuron was recorded twice on one
                       day. Ambiguous on purpose: it is either an over-merge
                       across days, or KiloSort over-splitting one neuron into
                       two clusters that day. Only a person can tell, and the
                       second case is information about the sort.
  weak internal link   two members whose own pair scores below the threshold,
                       so they are joined only through a third unit.

`proposed_links` lists each unit's proposed links (the partner may be on the
same day or another day) the backend saved; the
Tracked units panel flags and queues those too (a single unmatched unit
included). An accepted or rejected track leaves the review queue for every
flag; the flags stay. Reopening it, or merging it into a new `proposed` track,
queues it again; a unit a rejected track left unmatched stays out of the queue.

Curation history (accepts, rejects, splits, merges, undo and redo) lives in
`curation.py`, one `history.csv` per run; this module writes nothing.
"""
# [AGENT EDIT END]
# [AGENT EDIT END]
# [AGENT EDIT END]
import hashlib
import itertools

import numpy as np
import pandas as pd

from nems_lbhb.unit_match import evidence as ev

DEFAULT_THRESHOLD = 0.90
# the ledger API (LEDGER_ROOT, ledger_path, load_ledger, append, current,
# assignment, overlaps, reconcile, run_root, seed, propose, split, merge,
# remove_member, set_status) is removed: curation.py is the one writer
# [AGENT EDIT END]

# [AGENT EDIT START | agent: claude-opus-5 | user: wingertj | reason: cluster layer for cross-day tracking | date: 2026-09-15]
# --------------------------------------------------------------- identifiers
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: docstring names history.csv, not the removed ledger (run-curation final review 2) | date: 2026-10-06]
def cluster_id(members, pipeline_version=ev.PIPELINE_VERSION):
    """Deterministic id from membership, so re-proposing is stable.

    An edit changes the membership and therefore the id; the cluster_id and
    parent_ids columns of the run's history.csv record which cluster it came
    from, which is what keeps the history legible.
    """
    key = '|'.join(sorted(str(m) for m in members)) + f'|{pipeline_version}'
    return 'uc_' + hashlib.sha1(key.encode()).hexdigest()[:12]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a unit's day is its unit-table siteid, never a cellid prefix (Plan B Task 4) | date: 2026-09-28]
def unit_sites(units: pd.DataFrame) -> dict:
    """{cellid: siteid} from a unit table; every unit must name its siteid."""
    ids = [str(c) for c in units['cellid']]
    if 'siteid' not in units:
        # an absent column refuses even for a table with no units
        raise ValueError(f"unit table needs a siteid for every unit · missing: {', '.join(ids[:5])}")
    missing = [c for c, v in zip(ids, units['siteid']) if pd.isna(v) or not str(v).strip()]
    if missing:
        raise ValueError(f"unit table needs a siteid for every unit · missing: {', '.join(missing[:5])}")
    return dict(zip(ids, (str(v) for v in units['siteid'])))


def unit_label(cellid: str, siteid: str) -> str:
    """The unit's name within its day: the cellid after its siteid and a '-' or ':'."""
    cellid, siteid = str(cellid), str(siteid)
    if cellid.startswith(siteid) and cellid[len(siteid):len(siteid) + 1] in ('-', ':'):
        return cellid[len(siteid) + 1:]
    return cellid
# [AGENT EDIT END]


# ------------------------------------------------------------------ proposal
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the comment names the event-response stream (launcher-gui Task 10) | date: 2026-10-07]
# Clusters follow the combined probability, which weighs the event-response and the
# physiology streams together. A cache built before the streams existed has
# only the event-response one; both are a probability that the pair is the same
# neuron, so the threshold means the same thing either way.
# [AGENT EDIT END]
# [AGENT EDIT START | agent: codex | user: wingertj | reason: compare matching backends with isolated run curation | date: 2026-09-17]
SCORE_COLUMNS = ('match_score', 'p_global', 'posterior')
# [AGENT EDIT END]


def score_column(pairs):
    # [AGENT EDIT START | agent: codex | user: wingertj | reason: never substitute cached scores for a backend with no confidence output | date: 2026-09-17]
    if 'match_score' in getattr(pairs, 'columns', ()):
        return 'match_score'
    # [AGENT EDIT END]
    for c in SCORE_COLUMNS:
        if c in getattr(pairs, 'columns', ()):
            if pd.to_numeric(pairs[c], errors='coerce').notna().any():
                return c
    return SCORE_COLUMNS[-1]


def posterior_lookup(pairs):
    """Symmetric {(cellid_a, cellid_b): probability} over in-gate pairs."""
    out = {}
    col = score_column(pairs)
    for r in pairs.itertuples():
        p = getattr(r, col, np.nan)
        if p is None or not np.isfinite(p):
            continue
        out[(r.cellid_a, r.cellid_b)] = float(p)
        out[(r.cellid_b, r.cellid_a)] = float(p)
    return out



# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: days are counted from each member's siteid (Plan B Task 4) | date: 2026-09-28]
def describe(members, lookup, sites, threshold=DEFAULT_THRESHOLD):
    """Size, day coverage and the two flags that drive the review queue."""
    members = sorted(members)
    missing = [m for m in members if m not in sites]
    if missing:
        raise ValueError(f'no siteid for {missing}')
    counts = pd.Series([sites[m] for m in members], dtype=object).value_counts()
# [AGENT EDIT END]
    internal = [lookup.get((a, b)) for a, b in itertools.combinations(members, 2)]
    internal = [v for v in internal if v is not None and np.isfinite(v)]
    worst = float(min(internal)) if internal else np.nan
    return dict(
        n_members=len(members), n_days=int(counts.size),
        same_day_collision=bool((counts > 1).any()),
        max_per_day=int(counts.max()) if len(counts) else 0,
        weak_link=bool(internal and worst < threshold),
        min_internal_posterior=worst,
        n_internal_measured=len(internal),
        n_internal_possible=len(members) * (len(members) - 1) // 2,
    )


def needs_review(info):
    return bool(info['same_day_collision'] or info['weak_link'])


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: each unit's proposed same-day link partners, for the Tracked units flag (native-same-day Task 8) | date: 2026-10-11]
def proposed_links(pairs: pd.DataFrame) -> dict:
    """``{cellid: sorted partner cellids}`` over the ``proposed_link`` rows, both directions; ``{}`` without ``decision``."""
    if 'decision' not in getattr(pairs, 'columns', ()):
        return {}
    links = {}
    hit = pairs[pairs['decision'] == 'proposed_link']
    for a, b in zip(hit['cellid_a'].astype(str), hit['cellid_b'].astype(str)):
        links.setdefault(a, set()).add(b)
        links.setdefault(b, set()).add(a)
    return {c: sorted(p) for c, p in links.items()}
# [AGENT EDIT END]

# [AGENT EDIT END]
