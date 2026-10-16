# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: threshold neighbourhood pinned by hand-counted channel masks (threshold neighbourhood) | date: 2026-10-04]
from __future__ import annotations

import numpy as np
import pytest

from multiphy.backends.local.neighbourhood import unit_neighbourhood

T = np.arange(82)
COLUMN = np.array([(0.0, 20.0 * k) for k in range(40)])  # one column, 20 µm pitch; channel k at depth 20·k


def _trough(sample: int) -> np.ndarray:
    return np.exp(-0.5 * ((T - sample) / 1.5) ** 2)  # exactly 1 at its sample


def _unit(amplitudes: dict, trough_sample=None) -> np.ndarray:
    """(82, 40) waveform: channel c is -amplitudes[c] times a trough at sample 41 (or trough_sample[c])."""
    waveform = np.zeros((82, len(COLUMN)))
    for channel, amplitude in amplitudes.items():
        waveform[:, channel] = -amplitude * _trough((trough_sample or {}).get(channel, 41))
    return waveform


def test_threshold_mask_hand_counted():
    # peak channel 10 at 100 µV; 10% = 10 µV. Channel 7 sits exactly at 10 (kept), 13 just under (dropped,
    # a hole inside the footprint), 14 above (kept), 30 is 400 µm away (beyond 150 µm, dropped)
    amplitudes = {c: 1.0 for c in range(40)}
    amplitudes.update({10: 100.0, 9: 40.0, 11: 40.0, 8: 15.0, 12: 15.0, 7: 10.0, 13: 9.99, 14: 12.0, 30: 50.0})
    result = unit_neighbourhood(_unit(amplitudes), COLUMN)
    assert set(result) == {'peak_um', 'channels'}
    assert result['channels'].dtype == bool and result['channels'].shape == (40,)
    assert np.flatnonzero(result['channels']).tolist() == [7, 8, 9, 10, 11, 12, 14]
    np.testing.assert_array_equal(result['peak_um'], (0.0, 200.0))


def test_five_nearest_channels_always_kept():
    # a compact unit: only the peak channel passes 10%, so the 5 nearest (0, 20, 20, 40, 40 µm) are added
    amplitudes = {c: 1.0 for c in range(40)}
    amplitudes[10] = 100.0
    assert np.flatnonzero(unit_neighbourhood(_unit(amplitudes), COLUMN)['channels']).tolist() == [8, 9, 10, 11, 12]


def test_peak_and_amplitudes_read_the_spike_window():
    # channel 13 (60 µm) troughs at sample 50, ~0 at the peak's sample 41 but inside the window 33..55: kept.
    # channel 14 troughs at sample 70 and channel 30 at sample 5, both outside the window: neither counts,
    # though channel 30's -500 µV is the snippet's largest value
    amplitudes = {10: 100.0, 13: 60.0, 14: 80.0, 30: 500.0}
    result = unit_neighbourhood(_unit(amplitudes, {13: 50, 14: 70, 30: 5}), COLUMN)
    np.testing.assert_array_equal(result['peak_um'], (0.0, 200.0))
    assert np.flatnonzero(result['channels']).tolist() == [8, 9, 10, 11, 12, 13]
# [AGENT EDIT END]


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: pair channel union of threshold masks, B moved into A's frame, hand-counted (threshold neighbourhood) | date: 2026-10-04]
from multiphy.backends.local.examples import pair_properties  # noqa: E402
from multiphy.backends.local.neighbourhood import pair_channels, spike_window  # noqa: E402


def _mask(channels) -> dict:
    mask = np.zeros(len(COLUMN), dtype=bool)
    mask[list(channels)] = True
    return {'peak_um': np.zeros(2), 'channels': mask}


def test_pair_channels_union_same_frame():
    mask = pair_channels(_mask([3, 4, 5]), _mask([5, 6, 9]), COLUMN, COLUMN, 0.0)
    assert np.flatnonzero(mask).tolist() == [3, 4, 5, 6, 9]


@pytest.mark.parametrize('offset_um, expected', [
    # A channel at depth y reads B at y - offset: B's 100..140 µm land on A's 300..340 µm (channels 15-17)
    (200.0, [3, 15, 16, 17]),
    # half a channel off: A's 300 µm and 360 µm read B at 90 and 150 µm, halfway onto B's mask (0.5, kept)
    (210.0, [3, 15, 16, 17, 18]),
])
def test_pair_channels_moves_b_into_a_frame(offset_um, expected):
    mask = pair_channels(_mask([3]), _mask([5, 6, 7]), COLUMN, COLUMN, offset_um)
    assert np.flatnonzero(mask).tolist() == expected


def test_spike_window_matches_unitmatch():
    assert spike_window(82).tolist() == list(range(33, 56))


def _local_unit(seed: int) -> dict:
    geometry = np.array([(x, 20.0 * k) for k in range(40) for x in (0.0, 32.0)])
    dist = np.linalg.norm(geometry - np.array([0.0, 100.0]), axis=1)
    near = dist <= 200
    waveform = np.empty((82, len(geometry)))
    waveform[:, near] = _trough(41)[:, None] * -100 * np.exp(-0.03 * dist[near])[None, :]
    n_far = int((~near).sum())
    rng = np.random.default_rng(seed)
    scale = rng.uniform(0, 40, n_far)
    noise = rng.uniform(-1, 1, (82, n_far)) * scale
    waveform[:, ~near] = noise
    half = {'fingerprints': {'s': np.arange(6.0)}, 'spont': 1.0, 'rate': 2.0, 'per_stim': {'s': 2.0},
            'isi': np.ones(4), 'acg': np.ones(4)}
    return dict(splits=[(half, half)], waveform_halves=np.stack([waveform, waveform], axis=-1),
                sw_halves=(0.3, 0.3), geometry=geometry, neighbourhood=unit_neighbourhood(waveform, geometry))


def test_local_r_ignores_far_channels():
    result = pair_properties(_local_unit(0), _local_unit(1), ['s'], 0, offset_um=0)
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the peak-to-peak correlations are read by their footprint names (event-optional Task 1) | date: 2026-10-09]
    assert result['waveform_r_local'] == pytest.approx(1.0)
    assert result['footprint_r_local'] == pytest.approx(1.0)
    assert result['waveform_r'] < 0.9
    assert result['footprint_r'] < 0.9
    # [AGENT EDIT END]
# [AGENT EDIT END]
