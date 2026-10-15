# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the halves rule: a day's analysis window cut into two contiguous blocks at a trial start or the sample midpoint (multiphy project folder Task 9) | date: 2026-09-25]
"""The halves rule: where a day's analysis window is cut into its two halves.

With trials, the boundary is the start of the middle trial (``'trial_start'``), so no trial
crosses it; without, it is the sample midpoint (``'sample_midpoint'``). The two halves are
contiguous blocks of time, ``[start, boundary)`` and ``[boundary, stop)``. The fingerprint
and waveform splits inside each half are left as they are.
"""
from __future__ import annotations

import numpy as np


def halves(trials: np.ndarray, analysis_window: tuple) -> dict:
    """Return ``{'rule', 'boundary', 'windows'}`` for ``analysis_window`` ``(start, stop)``.

    ``trials`` is ``(n, 2)`` ``[start, stop)`` samples, possibly empty; they are sorted by
    start before the middle one is taken. ``windows`` is the ``(2, 2)`` int64 array
    ``[[start, boundary], [boundary, stop]]``. A boundary at or outside either end of the
    window leaves a half empty and raises ``ValueError``.
    """
    start, stop = (int(value) for value in analysis_window)
    trials = np.asarray(trials, dtype=np.int64).reshape(-1, 2)
    if len(trials):
        trials = trials[np.argsort(trials[:, 0], kind='stable')]
        boundary, rule = int(trials[len(trials) // 2, 0]), 'trial_start'
    else:
        boundary, rule = (start + stop) // 2, 'sample_midpoint'
    windows = np.array([[start, boundary], [boundary, stop]], dtype=np.int64)
    if boundary <= start or boundary >= stop:
        raise ValueError('A half of the day is empty.')
    return {'rule': rule, 'boundary': boundary, 'windows': windows}
# [AGENT EDIT END]
