"""The calibrated pair classifier.

``CalibratedPairModel`` is a regularized feature classifier with an independent
calibration partition.
"""
from __future__ import annotations

import numpy as np

from .evidence import ALL_PROPERTIES, _logistic_fit


class CalibratedPairModel:
    """Regularized feature classifier with an independent calibration partition.

    Both inputs contain ordinary candidate PAIRS, not per-unit maxima. Labels
    are 0/1 in ``same_unit``. Group IDs identify the complete source recording
    or well; all split pseudo-sessions from that source must share the group.
    The calibration partition must represent the intended candidate population,
    including absent-counterpart units, rather than an artificially balanced
    case/control sample. A third, untouched partition is needed for evaluation.

    Calibration is specific to that reference population. Within-sort temporal
    split calibration does not establish cross-day or reinsertion calibration.
    Features may include stimulus fingerprints, physiology, and waveforms;
    missing values are imputed using fitting data only and have explicit flags.
    """

    def __init__(self, properties=None, l2=1.0):
        self.properties = tuple(properties or ALL_PROPERTIES)
        if not self.properties or len(set(self.properties)) != len(self.properties):
            raise ValueError('Provide distinct feature names.')
        reserved = {'same_unit', 'cellid', 'cellid_a', 'cellid_b', 'source_unit',
                    'source_unit_id', 'source_session', 'session_id', 'siteid',
                    'well', 'partition'}
        if any(not isinstance(p, str) or not p.strip() for p in self.properties) or reserved.intersection(self.properties):
            raise ValueError('Reference labels, source identities, and partition metadata cannot be model features.')
        if not np.isfinite(l2) or l2 <= 0:
            raise ValueError('l2 must be positive.')
        self.l2 = float(l2)
        self.fitted = False

    def _matrix(self, table):
        if any(p not in table for p in self.properties):
            raise ValueError('Every configured feature column must be present; use NaN for unavailable evidence.')
        out = np.column_stack([np.asarray(table[p], dtype=float) for p in self.properties])
        if out.ndim != 2:
            raise ValueError('Feature columns must be one-dimensional.')
        return out

    @staticmethod
    def _labels(table, n):
        y = np.asarray(table['same_unit'], dtype=float)
        if y.shape != (n,) or not np.isfinite(y).all() or set(y) != {0.0, 1.0}:
            raise ValueError('Each fitting partition must contain both 0 and 1 labels.')
        if n < 20:
            raise ValueError('At least 20 candidate pairs per fitting partition are required.')
        return y

    def _transform(self, x):
        missing = ~np.isfinite(x)
        scaled = (np.where(missing, self.center_, x) - self.center_) / self.scale_
        return np.column_stack((scaled, missing.astype(float)))

    def fit(self, training, calibration, *, training_groups, calibration_groups,
            reference, label_column='same_unit'):
        from scipy.optimize import minimize
        from scipy.special import expit
        self.fitted = False
        if label_column != 'same_unit':
            raise ValueError('The explicit reference label column must be named same_unit.')
        x, c = self._matrix(training), self._matrix(calibration)
        y, cy = self._labels(training, len(x)), self._labels(calibration, len(c))
        tg, cg = np.asarray(training_groups, dtype=object), np.asarray(calibration_groups, dtype=object)
        if tg.shape != (len(x),) or cg.shape != (len(c),):
            raise ValueError('Supply one source-recording/well group per pair.')
        if any(g is None or not str(g).strip() or str(g).lower() == 'nan' for g in list(tg) + list(cg)):
            raise ValueError('Source-group IDs must not be missing.')
        if set(map(str, tg)) & set(map(str, cg)):
            raise ValueError('Training and calibration source groups overlap; split all parts together.')
        if not isinstance(reference, str) or not reference.strip():
            raise ValueError('Name the reference population and preprocessing used for calibration.')
        finite = np.isfinite(x)
        if not finite.any(1).all() or not np.isfinite(c).any(1).all():
            raise ValueError('Unmeasurable pairs must be recorded as unavailable, not fitted as evidence.')
        self.center_ = np.array([np.median(x[finite[:, j], j]) if finite[:, j].any() else 0.
                                 for j in range(x.shape[1])])
        self.scale_ = np.array([np.std(x[finite[:, j], j]) if finite[:, j].any() else 1.
                                for j in range(x.shape[1])])
        self.scale_[self.scale_ < 1e-9] = 1.
        self.weights_ = _logistic_fit(self._transform(x), y * 2 - 1, l2=self.l2)
        if not np.isfinite(self.weights_).all():
            raise ValueError('Feature classifier fitting failed.')
        design = self._transform(c)
        score = design @ self.weights_[:-1] + self.weights_[-1]

        def objective(ab):
            z = ab[0] * score + ab[1]
            loss = np.sum(np.logaddexp(0., z) - cy * z) + self.l2 * ab[0] ** 2
            residual = expit(z) - cy
            grad = np.array([residual @ score + 2 * self.l2 * ab[0], residual.sum()])
            return float(loss), grad

        opt = minimize(objective, [1., 0.], jac=True, method='L-BFGS-B',
                       bounds=[(0., None), (None, None)])
        if not opt.success or not np.isfinite(opt.x).all():
            raise ValueError(f'Independent calibration failed: {opt.message}')
        self.calibration_ = opt.x
        self.reference = reference
        self.training_groups_ = tuple(sorted(set(map(str, tg))))
        self.calibration_groups_ = tuple(sorted(set(map(str, cg))))
        self.n_training_, self.n_calibration_ = len(x), len(c)
        self.calibration_prevalence_ = float(cy.mean())
        self.fitted = True
        return self

    def predict_proba(self, pairs):
        """Return NaN for pairs with no measured feature rather than match them."""
        from scipy.special import expit
        if not self.fitted:
            raise ValueError('Fit on disjoint training/calibration source groups first.')
        x = self._matrix(pairs)
        z = self._transform(x) @ self.weights_[:-1] + self.weights_[-1]
        out = expit(self.calibration_[0] * z + self.calibration_[1])
        out[~np.isfinite(x).any(axis=1)] = np.nan
        return out

    def to_dict(self):
        """JSON-compatible model and provenance; no Python pickle required."""
        if not self.fitted:
            raise ValueError('Cannot export an unfitted model.')
        return dict(schema_version=1, properties=list(self.properties), l2=self.l2,
                    center=self.center_.tolist(), scale=self.scale_.tolist(),
                    weights=self.weights_.tolist(), calibration=self.calibration_.tolist(),
                    reference=self.reference, training_groups=list(self.training_groups_),
                    calibration_groups=list(self.calibration_groups_),
                    n_training=self.n_training_, n_calibration=self.n_calibration_,
                    calibration_prevalence=self.calibration_prevalence_)

    @classmethod
    def from_dict(cls, spec):
        if spec.get('schema_version') != 1:
            raise ValueError('Unsupported calibrated pair model schema.')
        obj = cls(spec['properties'], spec['l2'])
        n = len(obj.properties)
        for key, shape in (('center', (n,)), ('scale', (n,)),
                           ('weights', (2 * n + 1,)), ('calibration', (2,))):
            value = np.asarray(spec[key], dtype=float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f'Invalid serialized {key}.')
            setattr(obj, key + '_', value)
        if (obj.scale_ <= 0).any() or obj.calibration_[0] < 0:
            raise ValueError('Invalid model scale or calibration slope.')
        obj.reference = spec['reference']
        if not isinstance(obj.reference, str) or not obj.reference.strip():
            raise ValueError('Invalid calibration reference.')
        for key in ('training_groups', 'calibration_groups'):
            groups = spec[key]
            if (not isinstance(groups, (list, tuple)) or not groups or
                    any(not isinstance(g, str) or not g.strip() or g.lower() == 'nan' for g in groups)):
                raise ValueError('Invalid calibration source-group provenance.')
            setattr(obj, key + '_', tuple(groups))
        if set(obj.training_groups_) & set(obj.calibration_groups_):
            raise ValueError('Invalid calibration provenance.')
        for key in ('n_training', 'n_calibration'):
            value = spec[key]
            if not isinstance(value, (int, float)) or not np.isfinite(value) or int(value) != value or value < 20:
                raise ValueError('Invalid calibration sample counts.')
            setattr(obj, key + '_', int(value))
        obj.calibration_prevalence_ = float(spec['calibration_prevalence'])
        if not np.isfinite(obj.calibration_prevalence_) or not 0 < obj.calibration_prevalence_ < 1:
            raise ValueError('Calibration must include both reference classes.')
        obj.fitted = True
        return obj
