# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: one native group resolver for UnitMatch, DeepUnitMatch and DANT (native-same-day Task 1) | date: 2026-10-11]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the docstring names the order of the label_pairs rules (native-same-day Task 12) | date: 2026-10-12]
"""Native group resolver: one unit per session in each native track.

A native backend labels units with a group id; a ``None`` label puts a unit in no group. Groups
run in sorted label order and a group's members keep input order. In a group, each session
holding one member keeps it. A session holding two or more members keeps the member with the
highest mean of the finite scores ``score[u, j]`` over the group's members ``j`` on other
sessions; a member with no finite score ranks below every finite mean (``-inf``). A tie
(exact ``==`` on that mean) goes to the larger ``n_spikes`` (NaN ranks lowest), then to the
smaller cellid. Two or more kept members form one track, in input order; one kept member forms
no track. Every within-group pair that is not two members of one track, same-session or not,
is a proposed link.

``label_pairs`` sets each row by the first rule that applies: both units in one track →
``selected`` true with the kept decision; a proposed pair → ``proposed_link``; one session →
``same_session``; else the backend's unmatched decision.
"""
# [AGENT EDIT END]
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd

from ..contract.results import Decision


@dataclass(frozen=True)
class NativeGroups:
    tracks: tuple
    proposed: frozenset


def _rank_key(mean, n_spikes, cellid):
    """Sort key where the smallest key wins: highest mean, then most spikes, then smallest cellid."""
    spikes = float(n_spikes)
    spikes = -np.inf if np.isnan(spikes) else spikes
    return (-mean, -spikes, cellid)


def resolve_groups(cellids: Sequence[str], sessions: Sequence[str], labels: Sequence[Optional[int]],
                   score: np.ndarray, n_spikes: Sequence[float]) -> NativeGroups:
    n = len(cellids)
    score = np.asarray(score, dtype=float)
    if (len(sessions) != n or len(labels) != n or len(n_spikes) != n
            or score.shape != (n, n)):
        raise ValueError('resolve_groups inputs disagree in length')
    members = {}
    for i, lab in enumerate(labels):
        if lab is None:
            continue
        members.setdefault(lab, []).append(i)
    tracks, proposed = [], set()
    for lab in sorted(members):
        group = members[lab]
        by_session = {}
        for i in group:
            by_session.setdefault(sessions[i], []).append(i)
        kept = set()
        for sess, idx in by_session.items():
            if len(idx) == 1:
                kept.add(idx[0])
                continue
            others = [j for j in group if sessions[j] != sess]
            keys = []
            for u in idx:
                vals = score[u, others] if others else np.array([])
                vals = vals[np.isfinite(vals)]
                mean = float(np.mean(vals)) if vals.size else -np.inf
                keys.append((_rank_key(mean, n_spikes[u], cellids[u]), u))
            kept.add(min(keys)[1])
        track = tuple(cellids[i] for i in group if i in kept)
        in_track = set(track) if len(track) >= 2 else set()
        if len(track) >= 2:
            tracks.append(track)
        for x in range(len(group)):
            for y in range(x + 1, len(group)):
                a, b = cellids[group[x]], cellids[group[y]]
                if not (a in in_track and b in in_track):
                    proposed.add(frozenset((a, b)))
    return NativeGroups(tuple(tracks), frozenset(proposed))


def label_pairs(pairs: pd.DataFrame, sessions: dict, groups: NativeGroups, kept: str,
                unmatched: str) -> pd.DataFrame:
    """Copy ``pairs`` with ``selected`` and ``decision`` set row by row; rows and order unchanged."""
    track_of = {c: t for t, track in enumerate(groups.tracks) for c in track}
    selected, decision = [], []
    for a, b in zip(pairs['cellid_a'], pairs['cellid_b']):
        if a in track_of and track_of[a] == track_of.get(b):
            selected.append(True); decision.append(kept)
        elif frozenset((a, b)) in groups.proposed:
            selected.append(False); decision.append(Decision.PROPOSED_LINK.value)
        elif sessions[a] == sessions[b]:
            selected.append(False); decision.append(Decision.SAME_SESSION.value)
        else:
            selected.append(False); decision.append(unmatched)
    out = pairs.copy()
    out['selected'] = pd.Series(selected, index=out.index, dtype=object)
    out['decision'] = pd.Series(decision, index=out.index, dtype=object)
    return out
# [AGENT EDIT END]
