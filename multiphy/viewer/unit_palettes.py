"""The unit colour palettes that ``Settings`` ▸ ``Unit colours`` offers, the active palette (set by the app's
entry points and the menu) and the Day ramp. Imports no Qt."""
import numpy as np

IDS = ('pastel', 'tol', 'maxdist', 'okabe', 'tab20', 'ramp')
LABELS = {'pastel': 'Graphite pastel', 'tol': 'Tol light +', 'maxdist': 'Max-distance',
          'okabe': 'Okabe–Ito + tints', 'tab20': 'Tableau 20', 'ramp': 'Day ramp'}
DEFAULT = 'pastel'
FIXED = {
    'pastel': ('#E2E8F0', '#F472B6', '#FEF08A', '#C4B5FD', '#FCD34D', '#86EFAC', '#FDA4AF', '#5EEAD4',
               '#BEF264', '#FB923C', '#C084FC', '#22D3EE', '#7DD3FC', '#A5B4FC', '#60A5FA'),
    'tol': ('#77AADD', '#EE8866', '#EEDD88', '#FFAABB', '#99DDFF', '#44BB99', '#BBCC33', '#AAAA00',
            '#DDDDDD', '#AA4499', '#CC3311', '#009988', '#0077BB', '#EE3377', '#228833'),
    'maxdist': ('#4BFFFF', '#E14B96', '#F0A5C3', '#3C962D', '#D26978', '#2DD2FF', '#2D96B4', '#78FFD2',
                '#E1F000', '#96FF96', '#00C3B4', '#FF3C3C', '#FF4BD2', '#69B496', '#2D9696'),
    'okabe': ('#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00', '#CC79A7', '#DDDDDD',
              '#ABDAF4', '#F2CF80', '#80CEB9', '#EAAE80', '#F8F2A0', '#E6BCD3', '#80B8D8'),
    'tab20': ('#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD', '#8C564B', '#E377C2', '#7F7F7F',
              '#BCBD22', '#17BECF', '#AEC7E8', '#FFBB78', '#98DF8A', '#FF9896', '#C5B0D5'),
}
_palette = DEFAULT        # the active palette id

WHITE = np.array([0.95047, 1.0, 1.08883])                      # D65, 2-degree observer
RGB_FROM_XYZ = np.linalg.inv(np.array([[0.412453, 0.357580, 0.180423],
                                       [0.212671, 0.715160, 0.072169],
                                       [0.019334, 0.119193, 0.950227]]))


def set_palette(name: str) -> None:
    """Make ``name`` (one of ``IDS``, exact case) the active palette; any other value raises ``ValueError``
    and leaves the active palette unchanged."""
    global _palette
    if name not in IDS:
        raise ValueError(f'unknown unit palette {name!r}')
    _palette = name


def palette_name() -> str:
    """The active palette id."""
    return _palette


def by_day() -> bool:
    """True when the active palette is the Day ramp, which colours units by recording day."""
    return _palette == 'ramp'


def day_ramp(n: int) -> list:
    """``n`` hex colours along the Day ramp: L* 52 → 92, chroma 45, hue 280° sweeping 180°, D65 white. One
    colour is the first step; ``n <= 0`` gives ``[]``."""
    if n <= 0:
        return []
    t = np.arange(n) / (n - 1) if n > 1 else np.zeros(1)
    hue = np.deg2rad(280.0 + 180.0 * t)
    L, a, b = 52.0 + 40.0 * t, 45.0 * np.cos(hue), 45.0 * np.sin(hue)
    fy = (L + 16.0) / 116.0
    f = np.stack([a / 500.0 + fy, fy, np.clip(fy - b / 200.0, 0, None)], axis=1)
    xyz = np.where(f > 0.2068966, f ** 3, (f - 16.0 / 116.0) / 7.787) * WHITE
    lin = xyz @ RGB_FROM_XYZ.T
    srgb = np.where(lin > 0.0031308, 1.055 * np.clip(lin, 0, None) ** (1 / 2.4) - 0.055, 12.92 * lin)
    srgb = np.clip(srgb, 0, 1)
    return ['#' + ''.join(f'{int(round(float(v) * 255)):02X}' for v in c) for c in srgb]


def colours(n: int) -> list:
    """``n`` unit colours from the active palette: the Day ramp for ``'ramp'``, else the fixed palette, which
    repeats from its first colour past the 15th unit. ``n <= 0`` gives ``[]``."""
    if n <= 0:
        return []
    if _palette == 'ramp':
        return day_ramp(n)
    return [FIXED[_palette][i % 15] for i in range(n)]


def swatch(name: str) -> list:
    """The eight colours of a palette's swatch strip; a name not in ``IDS`` raises ``ValueError``."""
    if name not in IDS:
        raise ValueError(f'unknown unit palette {name!r}')
    if name == 'ramp':
        return day_ramp(8)
    return list(FIXED[name][:8])
