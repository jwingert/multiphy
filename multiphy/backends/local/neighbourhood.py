"""A unit's channel neighbourhood: its peak channel and the channels its spike reaches.

Each channel's amplitude is its largest absolute voltage inside ``spike_window`` (UnitMatch's
``waveidx``). The peak channel has the largest amplitude. The neighbourhood is every channel
closer than ``CANDIDATE_RADIUS_UM`` to the peak channel whose amplitude is at least
``THRESHOLD_FRACTION`` of the peak channel's, plus the ``MIN_CHANNELS`` channels nearest the
peak (ties in distance broken by channel order), so a compact unit still gives a correlation.
Each channel's own maximum, not one shared sample, keeps channels whose spike shape shifts in
time across the probe. ``pair_channels`` unions two units' neighbourhoods in one frame.
"""
from __future__ import annotations

import numpy as np

CANDIDATE_RADIUS_UM = 150.0  # UnitMatch channel_radius
THRESHOLD_FRACTION = 0.10  # a tenfold fall from the peak channel's amplitude
MIN_CHANNELS = 5  # ev.pearson needs 5 finite values for a local correlation


def unit_neighbourhood(waveform: np.ndarray, geometry: np.ndarray) -> dict:
    """``{'peak_um', 'channels'}`` of one unit's (sample, channel) waveform on (channel, 2) x, y µm geometry.

    ``peak_um`` is the peak channel's (x, y); ``channels`` is a boolean mask over ``geometry``'s
    channels. Raises ``ValueError`` on mismatched shapes or no finite value in the spike window.
    """
    waveform = np.asarray(waveform, dtype=float)
    geometry = np.asarray(geometry, dtype=float)
    if waveform.ndim != 2 or geometry.shape != (waveform.shape[1], 2):
        raise ValueError(f'waveform {waveform.shape} must be (sample, channel) and geometry '
                         f'{geometry.shape} must be (channel, 2)')
    window = np.abs(waveform[spike_window(waveform.shape[0])])
    if not np.isfinite(window).any():
        raise ValueError(f'waveform {waveform.shape} has no finite value in its spike window')
    amp = np.fmax.reduce(window, axis=0)  # ignores NaN samples; an all-NaN channel stays NaN, never selected
    peak = int(np.nanargmax(amp))
    peak_um = geometry[peak].copy()
    dist = np.linalg.norm(geometry - peak_um, axis=1)
    channels = (amp >= THRESHOLD_FRACTION * amp[peak]) & (dist < CANDIDATE_RADIUS_UM)
    channels[np.argsort(dist, kind='stable')[:MIN_CHANNELS]] = True
    return {'peak_um': peak_um, 'channels': channels}


def pair_channels(neighbourhood_a: dict, neighbourhood_b: dict, geometry_a: np.ndarray, geometry_b: np.ndarray,
                  offset_um: float) -> np.ndarray:
    """Boolean mask over ``geometry_a``'s channels (unit A's frame): in either unit's neighbourhood.

    B's mask moves into A's frame by ``align_channels`` of its 0/1 values at ``offset_um``
    (added to B's depths, as for B's waveform) and is kept where the interpolated value is at
    least 0.5; an A channel outside B's depth range is not B's.
    """
    from .examples import align_channels  # examples imports this module

    mask_a = np.asarray(neighbourhood_a['channels'], dtype=bool)
    mask_b = np.asarray(neighbourhood_b['channels'], dtype=bool)
    if mask_a.shape != (len(geometry_a),) or mask_b.shape != (len(geometry_b),):
        raise ValueError(f'channel masks {mask_a.shape} and {mask_b.shape} do not follow geometries of '
                         f'{len(geometry_a)} and {len(geometry_b)} channels')
    moved = align_channels(mask_b[None, :].astype(float), geometry_b, geometry_a, offset_um)[0]
    return mask_a | (np.nan_to_num(moved, nan=0.0) >= 0.5)


def spike_window(n_samples: int) -> np.ndarray:
    """UnitMatch's ``waveidx``: samples ``n_samples // 2 - 8`` to ``n_samples // 2 + 14``, inside the snippet.

    Indices outside ``[0, n_samples)`` are dropped, never clamped, so no sample counts twice.
    """
    window = np.arange(n_samples // 2 - 8, n_samples // 2 + 15)
    return window[(window >= 0) & (window < n_samples)]
