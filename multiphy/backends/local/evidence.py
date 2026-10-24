"""
Matching logic. Imports no UI toolkit and no recording loader, so it can be
tested without a display and without touching the data tree.

Design, in the order the criteria apply:

  GATE on electrode proximity. Position narrows the field and contributes
  nothing to the score, because when the tissue moves the footprint moves with
  it, and waveform shape is not distinctive enough to compensate: neurons
  recorded simultaneously, and so certainly different, have peak-channel
  waveforms about 0.91 similar whether they sit 20 um or 200 um apart.

  REQUIRE agreement on properties of the cell rather than of its position --
  firing rate, the inter-spike-interval histogram, and spike width as a veto.
  Thresholds are set to retain 90% of same-neuron pairs, calibrated from
  within-session split halves.

  SCORE whatever survives by the correlation between the two days' event-response
  fingerprints, and convert that to a probability against two reference
  distributions: the fingerprint's own reliability for a neuron measured twice,
  and the best correlation among simultaneously recorded neighbours for a neuron
  that is not the same.

The reference for "same" is within-day reliability, which is the most a neuron
can correlate with itself. Real day-to-day response change only lowers it, so
the resulting probability is conservative.

"""
from dataclasses import dataclass, asdict, field

import numpy as np

PIPELINE_VERSION = '1.0'
UM_PER_CHAN = 10.0


@dataclass
class Criteria:
    """Thresholds, all calibrated to retain 90% of same-neuron pairs."""
    gate_um: float = 40.0
    max_dlog_rate: float = 0.293      # spontaneous rate, |delta log|
    min_isi_cosine: float = 0.978
    max_dsw_ms: float = 0.15          # spike-width veto, never a deciding factor
    min_reliability: float = 0.30     # a fingerprint below this is not matchable
    rate_measure: str = 'spont'       # 'spont' | 'session' | 'shared'
    use: tuple = ('prox', 'rate', 'isi', 'sw')

    def to_dict(self):
        d = asdict(self)
        d['use'] = list(self.use)
        d['pipeline_version'] = PIPELINE_VERSION
        return d


# ------------------------------------------------------------- small helpers
def _aligned(a, b):
    """Both as float arrays, or None when they cannot be compared.

    Two curves of different lengths are not a shorter and a longer version of
    one thing: a 41-tone tuning curve against a 46-tone one, or a 7-second
    event response against a 20-second one, describe different measurements. Saying so
    with NaN is right; broadcasting them was a crash, and truncating them
    would have been a wrong number.
    """
    a, b = np.asarray(a, float).ravel(), np.asarray(b, float).ravel()
    if a.shape != b.shape or a.size == 0:
        return None
    return a, b


def cosine(a, b):
    ab = _aligned(a, b)
    if ab is None:
        return np.nan
    a, b = ab
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a, b = a[m], b[m]
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 1e-12 and nb > 1e-12 else np.nan


def pearson(a, b):
    ab = _aligned(a, b)
    if ab is None:
        return np.nan
    a, b = ab
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return np.nan
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def _num(x):
    """A stored value as a float, or NaN. Absent evidence is missing, not fatal.

    These helpers are handed values straight out of the feature tables, where a
    key can be absent: a unit can carry a fingerprint and no rate, because the
    peristimulus histogram is keyed by the rasterised channel list and the rate
    by the spike-time dictionary, and those two need not agree. np.isfinite(None)
    raises, so a single such unit used to take down a whole well.
    """
    try:
        return float(x) if x is not None else np.nan
    except (TypeError, ValueError):
        return np.nan


def dlog(a, b, floor=0.05):
    a, b = _num(a), _num(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.nan
    return float(abs(np.log(a + floor) - np.log(b + floor)))


def spearman_brown(r):
    """Reliability of a full-length curve from its split-half correlation."""
    r = _num(r)
    if not np.isfinite(r):
        return np.nan
    return float(2 * np.clip(r, -0.999, 0.999) / (1 + np.clip(r, -0.999, 0.999)))


def corr_matrix(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    A = np.nan_to_num(A - np.nanmean(A, 1, keepdims=True))
    B = np.nan_to_num(B - np.nanmean(B, 1, keepdims=True))
    A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    return A @ B.T


# ------------------------------------------------------------ depth alignment
def depth_profile(depths, amps, lo=0.0, hi=3840.0, bin_um=UM_PER_CHAN, sigma_um=60.0):
    """Amplitude-weighted density of units along the probe, over [lo, hi).

    The extent is a parameter rather than the length of one probe because depth
    is a position on the shank, and which part of the shank was recorded varies:
    a fixed 0-3840 grid silently piled every unit deeper than that into its last
    bin, which flattened the profile and made the alignment below unmeasurable.
    """
    n = max(int(round((hi - lo) / bin_um)), 1)
    p = np.zeros(n)
    for d, a in zip(depths, amps):
        if not np.isfinite(d):
            continue
        c = int(np.clip(round((d - lo) / bin_um), 0, n - 1))
        p[c] += np.log1p(max(a, 0.0))
    s = sigma_um / bin_um
    k = np.exp(-0.5 * (np.arange(-4 * s, 4 * s + 1) / s) ** 2)
    return np.convolve(p, k / k.sum(), mode='same')


def estimate_offset(depths_a, amps_a, depths_b, amps_b, max_shift=40,
                    bin_um=UM_PER_CHAN):
    """Rigid depth offset to apply to B, in microns.

    Identity-free: it uses only where activity sits on the shank, so it cannot
    manufacture a unit match. Its own noise floor is about 20 um, measured by
    running it on same-day recordings where the true offset is zero.

    Both days are binned on ONE grid spanning what either of them recorded, so
    two days that sampled different stretches of the probe are still compared
    over the stretch they share.
    """
    da = np.asarray(depths_a, float); db = np.asarray(depths_b, float)
    fa, fb = da[np.isfinite(da)], db[np.isfinite(db)]
    if not len(fa) or not len(fb):
        return 0.0, 0.0
    pad = max_shift * bin_um
    lo = float(min(fa.min(), fb.min())) - pad
    hi = float(max(fa.max(), fb.max())) + pad
    a = depth_profile(depths_a, amps_a, lo, hi, bin_um)
    b = depth_profile(depths_b, amps_b, lo, hi, bin_um)
    need = max(int(0.4 * len(a)), 20)
    shifts = np.arange(-max_shift, max_shift + 1)
    cc = np.full(len(shifts), np.nan)
    for i, s in enumerate(shifts):
        x, y = (a[s:], b[:len(b) - s]) if s >= 0 else (a[:len(a) + s], b[-s:])
        if len(x) < need or x.std() < 1e-9 or y.std() < 1e-9:
            continue
        cc[i] = np.corrcoef(x, y)[0, 1]
    if np.all(np.isnan(cc)):
        return 0.0, 0.0
    best = int(np.nanargmax(cc))
    return float(shifts[best] * bin_um), float(cc[best])


def register_depths(measurements, days=None):
    """One depth offset per day for a whole well, solved from all pairwise fits.

    Fitting each day pair independently over-parameterises the problem: ten days
    give forty-five pairwise offsets but only nine free parameters, so the
    estimates contradict each other. Measured on Prince well 3, the closure error
    between a direct offset and the two-step route through a third day had a
    median of 60 um and a 90th percentile of 250 um, against a 40 um gate.

    Here every measurement is treated as a statement about a difference,
    z[b] - z[a] ~= m_ab, weighted by how good that alignment looked, and the
    per-day offsets are solved jointly by weighted least squares. Offsets then
    compose exactly, and a long-interval pair inherits its alignment through the
    chain of short-interval pairs, which are the reliable ones -- a direct fit
    between days a fortnight apart is the case that saturates the search range.

    `measurements` is a list of (day_a, day_b, offset_um, weight). Each connected
    group of days is solved separately with its own gauge, since there is nothing
    tying two groups together. Returns (offsets, residuals) where `offsets` maps a
    day to microns relative to its group's mean.
    """
    meas = [m for m in measurements if np.isfinite(m[2]) and np.isfinite(m[3])]
    if days is None:
        days = sorted({d for m in meas for d in m[:2]})
    days = list(days)
    if not meas or len(days) < 2:
        return {d: 0.0 for d in days}, {}

    # connected groups: days linked by at least one usable measurement
    parent = {d: d for d in days}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _, w in meas:
        if w > 0 and a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    groups = {}
    for d in days:
        groups.setdefault(find(d), []).append(d)

    offsets, residuals = {}, {}
    for members in groups.values():
        if len(members) == 1:
            offsets[members[0]] = 0.0
            continue
        idx = {d: i for i, d in enumerate(members)}
        rows, rhs, wts = [], [], []
        for a, b, off, w in meas:
            if a not in idx or b not in idx or w <= 0:
                continue
            r = np.zeros(len(members))
            r[idx[b]] = 1.0
            r[idx[a]] = -1.0
            rows.append(r); rhs.append(off); wts.append(w)
        if not rows:
            for d in members:
                offsets[d] = 0.0
            continue
        A = np.asarray(rows); y = np.asarray(rhs); w = np.sqrt(np.asarray(wts))
        # pin the group mean so the arbitrary additive constant is fixed
        A = np.vstack([A * w[:, None], np.ones(len(members))])
        y = np.concatenate([y * w, [0.0]])
        z, *_ = np.linalg.lstsq(A, y, rcond=None)
        for d in members:
            offsets[d] = float(z[idx[d]])
        for a, b, off, _ in meas:
            if a in idx and b in idx:
                residuals[(a, b)] = float(abs((offsets[b] - offsets[a]) - off))
    return offsets, residuals


# --------------------------------------------------------------- pair scoring
def pair_evidence(ua, ub, offset_um=0.0, shared_stims=None):
    """Every evidence value for one candidate pair. Units are dicts of features."""
    e = dict(
        depth_a=ua['depth'], depth_b=ub['depth'],
        depth_delta_um=abs(ua['depth'] - (ub['depth'] + offset_um)),
        depth_offset_um=offset_um,
        fingerprint_r=pearson(ua.get('psth'), ub.get('psth'))
        if ua.get('psth') is not None and ub.get('psth') is not None else np.nan,
        reliability_a=ua.get('reliability', np.nan),
        reliability_b=ub.get('reliability', np.nan),
        tuning_r=pearson(ua.get('ftc'), ub.get('ftc'))
        if ua.get('ftc') is not None and ub.get('ftc') is not None else np.nan,
        spont_a=ua.get('spont', np.nan), spont_b=ub.get('spont', np.nan),
        d_log_spont=dlog(ua.get('spont', np.nan), ub.get('spont', np.nan)),
        d_log_rate_session=dlog(ua.get('rate_session', np.nan), ub.get('rate_session', np.nan)),
        isi_cosine=cosine(ua.get('isi'), ub.get('isi')),
        acg_cosine=cosine(ua.get('acg'), ub.get('acg')),
        spike_width_a=ua.get('sw_ms', np.nan), spike_width_b=ub.get('sw_ms', np.nan),
        d_spike_width_ms=abs(ua.get('sw_ms', np.nan) - ub.get('sw_ms', np.nan)),
        footprint_r=pearson(ua.get('ptp'), ub.get('ptp')),
        amp_a=ua.get('amp', np.nan), amp_b=ub.get('amp', np.nan),
    )
    # driven rate restricted to stimuli played on both days
    ps_a, ps_b = ua.get('per_stim') or {}, ub.get('per_stim') or {}
    shared = sorted(set(ps_a) & set(ps_b)) if shared_stims is None else list(shared_stims)
    if shared:
        va = np.nanmean([ps_a.get(s, np.nan) for s in shared])
        vb = np.nanmean([ps_b.get(s, np.nan) for s in shared])
        e['d_log_rate_shared'] = dlog(va, vb)
    else:
        e['d_log_rate_shared'] = np.nan
    return e


def rate_delta(e, measure):
    return {'spont': e['d_log_spont'], 'session': e['d_log_rate_session'],
            'shared': e['d_log_rate_shared']}.get(measure, e['d_log_spont'])


def gates_passed(e, crit: Criteria):
    """Which criteria this pair satisfies. Missing evidence fails its gate."""
    out = {}
    if 'prox' in crit.use:
        out['prox'] = bool(e['depth_delta_um'] <= crit.gate_um)
    if 'rate' in crit.use:
        v = rate_delta(e, crit.rate_measure)
        v = _num(v)
        out['rate'] = bool(np.isfinite(v) and v <= crit.max_dlog_rate)
    if 'isi' in crit.use:
        v = e['isi_cosine']
        v = _num(v)
        out['isi'] = bool(np.isfinite(v) and v >= crit.min_isi_cosine)
    if 'sw' in crit.use:
        v = e['d_spike_width_ms']
        v = _num(v)
        out['sw'] = bool(np.isfinite(v) and v <= crit.max_dsw_ms)
    return out


# ------------------------------------------------------------ the probability
# Two streams of evidence. Each property contributes a likelihood ratio built
# from its own same-neuron and different-neuron references, so a value that
# merely misses a threshold is penalised rather than discarded.
#
# `sign` is +1 when a larger value means better agreement.
STREAMS = {
    'stim': [('fingerprint_r', +1)],
    'phys': [('d_log_spont', -1), ('isi_cosine', +1), ('acg_cosine', +1)],
}
ALL_PROPERTIES = [p for v in STREAMS.values() for p, _ in v]


def run_streams(events: bool, *, waveform: bool) -> dict:
    """The streams a run can fit, in ``Ensemble`` order: ``stim``, ``phys``, ``wave``.

    ``stim`` (the event-response fingerprint) only when the run has events; ``phys`` always, with
    the whole-session rate only when the run has events; ``wave`` (local waveform, local footprint,
    spike width) only when its units carry waveform halves. A stream the run did not measure is
    left out rather than scored as neutral.
    """
    streams = {}
    if events:
        streams['stim'] = [('fingerprint_r', +1)]
    streams['phys'] = [('d_log_spont', -1), ('isi_cosine', +1), ('acg_cosine', +1)]
    if events:
        streams['phys'].append(('d_log_rate_session', -1))
    if waveform:
        streams['wave'] = [('waveform_r_local', +1), ('footprint_r_local', +1), ('d_spike_width_ms', -1)]
    return streams


def _reference_properties(streams: dict) -> list:
    """``ALL_PROPERTIES`` plus every property of ``streams``, once each: what the references hold."""
    return list(dict.fromkeys([*ALL_PROPERTIES, *(p for stream in streams.values() for p, _ in stream)]))


class LogLR:
    """log f_same(x) / f_diff(x) for one property, from its two references."""

    def __init__(self, same, diff, bw=0.3, clip=5.0):
        from scipy.stats import gaussian_kde
        same = np.asarray(same, float); same = same[np.isfinite(same)]
        diff = np.asarray(diff, float); diff = diff[np.isfinite(diff)]
        self.clip = clip
        self.ok = len(same) >= 8 and len(diff) >= 8 and same.std() > 1e-9 and diff.std() > 1e-9
        if self.ok:
            self._s = gaussian_kde(same, bw_method=bw)
            self._d = gaussian_kde(diff, bw_method=bw)

    def __call__(self, x):
        x = np.atleast_1d(np.asarray(x, float))
        out = np.zeros(x.shape)                  # missing evidence is neutral
        if not self.ok:
            return out
        m = np.isfinite(x)
        if m.any():
            ls = np.maximum(self._s(x[m]), 1e-12)
            ld = np.maximum(self._d(x[m]), 1e-12)
            out[m] = np.clip(np.log(ls) - np.log(ld), -self.clip, self.clip)
        return out


def _logistic_fit(X, y, l2=1.0, *, diagnostics=None):
    """Weights for combining stream scores, fitted on the labelled references.

    Rate, interval histogram and autocorrelogram are not independent -- rate
    sets the interval scale and the autocorrelogram restates the short
    intervals -- so summing their likelihood ratios is overconfident. Fitting
    the stream weights on labelled data, and then calibrating the combined
    score against the same references, absorbs that rather than trusting the
    product. Optional ``diagnostics`` records the optimizer outcome without
    changing its inputs, returned parameters or convergence policy.
    """
    from scipy.optimize import minimize
    X = np.asarray(X, float); y = np.asarray(y, float)
    A = np.hstack([X, np.ones((len(X), 1))])

    def nll(w):
        z = np.clip(A @ w, -30, 30)
        return float(np.sum(np.log1p(np.exp(-y * z))) + l2 * np.sum(w[:-1] ** 2))

    r = minimize(nll, np.zeros(A.shape[1]), method='L-BFGS-B')
    if diagnostics is not None:
        diagnostics.update(attempted=True, method='L-BFGS-B')
        for name, field, cast in (('success', 'success', bool), ('status', 'status', int),
                                  ('message', 'message', str), ('iterations', 'nit', int)):
            value = getattr(r, field, None)
            if value is not None:
                diagnostics[name] = cast(value)
        objective = getattr(r, 'fun', None)
        if objective is not None:
            diagnostics['objective'] = float(objective) if np.isfinite(objective) else None
    return r.x


class Ensemble:
    """One score per stream, and a global score combining them.

    ``streams`` maps a stream name to its ``(property, sign)`` list (``run_streams``); ``None``
    means ``STREAMS``. Weights are fitted from the reference data, one per stream in dict order,
    so the balance between event response, physiology and waveform is measured rather than
    asserted. A stream left out of ``streams`` is not scored at all.
    ``fit_diagnostics`` records convergence separately from ``fitted`` without changing score availability.
    """

    def __init__(self, same_ref, diff_ref, streams=None):
        self.streams = STREAMS if streams is None else streams
        self.lr = {}
        for prop in dict.fromkeys(p for st in self.streams.values() for p, _ in st):
            sgn = [s for st in self.streams.values() for p, s in st if p == prop][0]
            self.lr[prop] = LogLR(sgn * np.asarray(same_ref.get(prop, []), float),
                                  sgn * np.asarray(diff_ref.get(prop, []), float))
        self.weights = {stream: 1.0 for stream in self.streams}
        self.bias = 0.0
        self.fitted = False
        self.fit_diagnostics = dict(attempted=False)
        self._fit(same_ref, diff_ref)

    def stream_score(self, values, stream):
        tot = np.zeros(np.shape(np.atleast_1d(
            values.get(self.streams[stream][0][0], np.nan))))
        for prop, sgn in self.streams[stream]:
            tot = tot + self.lr[prop](sgn * np.asarray(values.get(prop, np.nan), float))
        return tot

    def _fit(self, same_ref, diff_ref):
        properties = _reference_properties(self.streams)
        rows, labels = [], []
        for ref, lab in ((same_ref, 1.0), (diff_ref, -1.0)):
            n = max((len(np.atleast_1d(ref.get(p, []))) for p in properties),
                    default=0)
            if not n:
                continue
            vals = {p: np.asarray(ref.get(p, [np.nan] * n), float) for p in properties}
            if any(len(vals[p]) != n for p in properties):
                self.fit_diagnostics['reason'] = 'references are not row-aligned'
                return                       # references must be row-aligned
            s = np.column_stack([self.stream_score(vals, stream) for stream in self.streams])
            rows.append(s)
            labels.append(np.full(len(s), lab))
        if len(rows) != 2:
            self.fit_diagnostics['reason'] = 'both reference classes are required'
            return
        X = np.vstack(rows); y = np.concatenate(labels)
        keep = np.isfinite(X).all(1)
        if keep.sum() < 20:
            self.fit_diagnostics['reason'] = 'fewer than 20 finite reference rows'
            return
        w = _logistic_fit(X[keep], y[keep], diagnostics=self.fit_diagnostics)
        # a stream may be uninformative but must never argue against a match:
        # a negative weight would make disagreement on it count as evidence FOR
        # the same neuron, so clamp at zero rather than trusting the sign
        self.weights = {stream: float(max(w[i], 0.0)) for i, stream in enumerate(self.streams)}
        self.bias = float(w[-1])
        self.fitted = True

    def score(self, values):
        """``s_<stream>`` for every stream, in dict order, then ``s_global``."""
        scores = {'s_' + stream: self.stream_score(values, stream) for stream in self.streams}
        g = sum(self.weights[stream] * scores['s_' + stream] for stream in self.streams) + self.bias
        return dict(scores, s_global=g)

    def reference_scores(self, ref):
        properties = _reference_properties(self.streams)
        n = max((len(np.atleast_1d(ref.get(p, []))) for p in properties), default=0)
        vals = {p: np.asarray(ref.get(p, [np.nan] * n), float) for p in properties}
        return self.score(vals)


class Mixture:
    """P(same neuron | fingerprint correlation).

    The observed distribution of best-candidate correlations is modelled as a
    fraction pi drawn from the same-neuron reference and the rest from the
    different-neuron reference, fitted by maximum likelihood.
    """

    def __init__(self, same_ref, diff_ref, bw=0.25):
        from scipy.stats import gaussian_kde
        same_ref = np.asarray(same_ref, float); same_ref = same_ref[np.isfinite(same_ref)]
        diff_ref = np.asarray(diff_ref, float); diff_ref = diff_ref[np.isfinite(diff_ref)]
        self.ok = len(same_ref) >= 5 and len(diff_ref) >= 5
        if self.ok:
            self._same = gaussian_kde(same_ref, bw_method=bw)
            self._diff = gaussian_kde(diff_ref, bw_method=bw)
        self.pi = np.nan
        self.pi_lo = np.nan
        self.pi_hi = np.nan

    def fit(self, observed):
        from scipy.optimize import minimize_scalar
        obs = np.asarray(observed, float); obs = obs[np.isfinite(obs)]
        if not self.ok or len(obs) < 8:
            return self
        ls = np.maximum(self._same(obs), 1e-12)
        ld = np.maximum(self._diff(obs), 1e-12)

        def nll(p):
            p = np.clip(p, 1e-6, 1 - 1e-6)
            return -np.sum(np.log(p * ls + (1 - p) * ld))

        self.pi = float(minimize_scalar(nll, bounds=(1e-6, 1 - 1e-6), method='bounded').x)
        grid = np.linspace(1e-6, 1 - 1e-6, 400)
        inside = np.array([nll(p) for p in grid]) <= nll(self.pi) + 1.92
        if inside.any():
            self.pi_lo, self.pi_hi = float(grid[inside].min()), float(grid[inside].max())
        return self

    def posterior(self, r):
        r = np.atleast_1d(np.asarray(r, float))
        if not self.ok or not np.isfinite(self.pi):
            return np.full(r.shape, np.nan)
        ls = np.maximum(self._same(r), 1e-12)
        ld = np.maximum(self._diff(r), 1e-12)
        return self.pi * ls / (self.pi * ls + (1 - self.pi) * ld)

    def threshold_for(self, target=0.8, lo=-0.5, hi=1.0):
        """Smallest correlation whose posterior reaches `target`."""
        grid = np.linspace(lo, hi, 600)
        p = self.posterior(grid)
        ok = grid[np.isfinite(p) & (p >= target)]
        return float(ok.min()) if len(ok) else np.nan
