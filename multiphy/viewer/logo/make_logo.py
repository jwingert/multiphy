# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the multiphy logo generator: one unit's peak-channel waveform tracked across three days (logo) | date: 2026-10-13]
"""Write the multiphy logo: one unit's waveform, tracked across three days.

The waveform is the mean waveform of unit ``PRN067a-338-1`` on its peak channel (channel index 337, the channel
with the largest peak-to-peak), read once from the ``waveform`` feature of project
``Prince_w16_BNT_e2e_binaural`` and kept here as numbers (``SAMPLES``), so drawing the logo reads no data.

Day 1 is drawn at full opacity; day 2 is shifted down and left at 60%; day 3 is shifted further left and back
up at 32% (``DAYS``). White links join the unit's centre from day to day, and a white ring marks each centre,
as the drift view rings a selected unit. The ground is a rounded Graphite tile. The colours are read from the
Graphite tokens (``display.PALETTES``, ``unit_palettes.FIXED``), so a token change shows up as committed SVGs
that no longer match.

``logo.svg`` (strokes at weight 1) is the README header, the About picture and the 128 and 256 px icons;
``icon_<px>.svg`` draws the same mark with heavier strokes for the smaller icons (``WEIGHTS``). Run
``python multiphy/viewer/logo/make_logo.py`` after changing anything here (it needs no Qt); a test
checks that the committed SVGs equal what ``build`` returns.
"""
from pathlib import Path

import numpy as np

from multiphy.viewer import display, unit_palettes

HERE = Path(__file__).resolve().parent
# PRN067a-338-1, channel index 337: the stored samples from 12 before the trough to the end, rounded to 0.01
SAMPLES = (
    -2.64, -5.51, -5.10, -5.41, -2.45, -3.92, -2.97, -0.11, 14.93, 10.27, -71.20, -184.78, -218.61, -178.88,
    -130.21, -90.17, -60.46, -36.17, -19.80, -4.16, 9.63, 24.13, 38.07, 53.48, 66.02, 77.55, 87.70, 97.64,
    101.19, 103.66, 99.63, 94.64, 85.97, 77.25, 68.25, 59.31, 51.20, 43.14, 38.10, 30.17, 25.57, 21.45, 16.16,
    11.48, 10.00, 6.22, 2.01, -1.10, -1.57, -3.81, -4.90, -5.36, -6.85,
)
PINK = unit_palettes.FIXED['pastel'][1]      # Graphite pastel, the second unit's colour
WHITE = display.PALETTES['graphite']['text']  # Graphite text, the same white as its plot line
GROUND = display.PALETTES['graphite']['ground']
DAYS = ((0.0, 0.0, 1.0), (-0.6, -0.55, 0.6), (-1.2, -0.25, 0.32))  # per day: x shift, y shift, opacity
SPAN = (0.46, 0.5)  # half the waveform's width and its peak-to-peak height, in layout units
UNIT = 100.0        # SVG units per layout unit
STROKES = {'wave': 11.0, 'link': 7.5, 'ring_radius': 13.0, 'ring': 6.0}  # SVG units at weight 1
WEIGHTS = {64: 1.3, 48: 1.45, 32: 1.7, 24: 2.0, 16: 2.3}  # px -> the stroke weight of icon_<px>.svg
MARGIN = 0.08       # the tile's margin on each side, a share of the drawn extent
CORNER = 0.18       # the tile's corner radius, a share of its side
LOGO = 'logo.svg'


def waveform() -> tuple:
    """``(x, y)`` of the drawn trace: ``SAMPLES`` smoothed by a 7-tap Gaussian (sigma 1.2 samples, edges
    repeated), shifted to start at 0 and scaled to a peak-to-peak of 1; ``x`` runs from -1 to 1."""
    kernel = np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2)
    y = np.convolve(np.pad(np.asarray(SAMPLES), 3, mode='edge'), kernel / kernel.sum(), mode='valid')
    y = (y - y[0]) / np.ptp(y)
    return np.linspace(-1, 1, y.size), y


def _point(x: float, y: float) -> str:
    return f'{round(x * UNIT, 1) + 0.0:.1f},{round(-y * UNIT, 1) + 0.0:.1f}'  # + 0.0 turns -0.0 into 0.0


def mark(weight: float = 1.0) -> str:
    """The logo as SVG text, its strokes ``weight`` times ``STROKES``, on a square tile fitted to the drawing."""
    wave_w, link_w = STROKES['wave'] * weight, STROKES['link'] * weight
    ring_r, ring_w = STROKES['ring_radius'] * weight, STROKES['ring'] * weight
    x, y = waveform()
    xs = [dx + s * SPAN[0] for dx, _, _ in DAYS for s in (x[0], x[-1])]
    ys = [dy + v * SPAN[1] for _, dy, _ in DAYS for v in (y.min(), y.max())]
    half = max(wave_w, ring_w + 2 * ring_r) / 2 / UNIT
    x0, x1, y0, y1 = min(xs) - half, max(xs) + half, min(ys) - half, max(ys) + half
    width, height = (x1 - x0) * UNIT, (y1 - y0) * UNIT
    side = max(width, height) * (1 + 2 * MARGIN)
    left, top = x0 * UNIT - (side - width) / 2, -y1 * UNIT - (side - height) / 2
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{left:.1f} {top:.1f} {side:.1f} {side:.1f}" '
           'role="img">',
           '<title>multiphy</title>',
           f'<rect x="{left:.1f}" y="{top:.1f}" width="{side:.1f}" height="{side:.1f}" rx="{side * CORNER:.1f}" '
           f'fill="{GROUND}"/>',
           f'<polyline points="{" ".join(_point(dx, dy) for dx, dy, _ in DAYS)}" fill="none" stroke="{WHITE}" '
           f'stroke-width="{link_w:.2f}" stroke-linecap="round" stroke-linejoin="round"/>']
    for dx, dy, opacity in reversed(DAYS):  # day 1 on top
        trace = ' '.join(_point(dx + a * SPAN[0], dy + b * SPAN[1]) for a, b in zip(x, y))
        out.append(f'<polyline points="{trace}" fill="none" stroke="{PINK}" stroke-width="{wave_w:.2f}" '
                   f'stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"/>')
        cx, cy = _point(dx, dy).split(',')
        out.append(f'<circle cx="{cx}" cy="{cy}" r="{ring_r:.2f}" fill="none" stroke="{WHITE}" '
                   f'stroke-width="{ring_w:.2f}"/>')
    out.append('</svg>')
    return '\n'.join(out) + '\n'


def build() -> dict:
    """``{file name: SVG text}``: ``LOGO`` at weight 1, then one ``icon_<px>.svg`` per ``WEIGHTS`` entry."""
    files = {LOGO: mark(1.0)}
    for px, weight in WEIGHTS.items():
        files[f'icon_{px}.svg'] = mark(weight)
    return files


def main() -> None:
    """Write every ``build`` file next to this module and print its path."""
    for name, text in build().items():
        (HERE / name).write_text(text)
        print(HERE / name)


if __name__ == '__main__':
    main()
# [AGENT EDIT END]
