# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the unit palettes: exact fixed palettes, the Day ramp, colours cycling by the active palette, the swatch strip and the signatures (graphite-palettes Task 1) | date: 2026-10-11]
"""The unit colour palettes and the Day ramp. Every oracle is a literal hex from the plan, never computed here."""
import typing

import pytest

from nems_lbhb.unit_match.viewer import unit_palettes


@pytest.fixture(autouse=True)
def _default_palette(monkeypatch):
    monkeypatch.setattr(unit_palettes, '_palette', 'pastel')


def test_fixed_palettes_exact():
    assert unit_palettes.IDS == ('pastel', 'tol', 'maxdist', 'okabe', 'tab20', 'ramp')
    assert unit_palettes.LABELS == {'pastel': 'Graphite pastel', 'tol': 'Tol light +', 'maxdist': 'Max-distance',
                                    'okabe': 'Okabe–Ito + tints', 'tab20': 'Tableau 20', 'ramp': 'Day ramp'}
    assert unit_palettes.DEFAULT == 'pastel'
    assert unit_palettes.FIXED == {
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
    assert list(unit_palettes.FIXED) == ['pastel', 'tol', 'maxdist', 'okabe', 'tab20']


def test_day_ramp_matches_board():
    assert unit_palettes.day_ramp(15) == [
        '#4E7CC8', '#767DCB', '#977DCB', '#B47EC7', '#CD80C1', '#E183B8', '#F288AE', '#FF8FA4', '#FF999A',
        '#FFA491', '#FFB18B', '#FFBF87', '#FFCE86', '#FFDC8A', '#F7EA92']
    assert unit_palettes.day_ramp(10) == [
        '#4E7CC8', '#897DCB', '#B77EC6', '#DB81BB', '#F589AC', '#FF979C', '#FFA98F', '#FFBE87', '#FFD487',
        '#F7EA92']
    assert unit_palettes.day_ramp(3) == ['#4E7CC8', '#FF8FA4', '#F7EA92']
    assert unit_palettes.day_ramp(1) == ['#4E7CC8']
    assert unit_palettes.day_ramp(0) == []
    assert unit_palettes.day_ramp(-1) == []


def test_colours_cycle_by_selection():
    assert unit_palettes.colours(2) == ['#E2E8F0', '#F472B6']
    assert unit_palettes.colours(0) == []
    unit_palettes.set_palette('tol')
    assert unit_palettes.colours(17)[15:] == ['#77AADD', '#EE8866']
    assert unit_palettes.by_day() is False
    unit_palettes.set_palette('ramp')
    assert unit_palettes.colours(3) == ['#4E7CC8', '#FF8FA4', '#F7EA92']
    assert unit_palettes.by_day() is True
    with pytest.raises(ValueError, match="unknown unit palette 'Tol'"):
        unit_palettes.set_palette('Tol')
    assert unit_palettes.palette_name() == 'ramp'


def test_swatch_eight():
    assert unit_palettes.swatch('okabe') == ['#E69F00', '#56B4E9', '#009E73', '#F0E442', '#0072B2', '#D55E00',
                                             '#CC79A7', '#DDDDDD']
    assert unit_palettes.swatch('ramp') == ['#4E7CC8', '#977DCB', '#CD80C1', '#F288AE', '#FF999A', '#FFB18B',
                                            '#FFCE86', '#F7EA92']
    with pytest.raises(ValueError):
        unit_palettes.swatch('purple')


def test_unit_palettes_signatures():
    hints = {name: typing.get_type_hints(getattr(unit_palettes, name))
             for name in ('set_palette', 'palette_name', 'by_day', 'day_ramp', 'colours', 'swatch')}
    assert hints == {
        'set_palette': {'name': str, 'return': type(None)},
        'palette_name': {'return': str},
        'by_day': {'return': bool},
        'day_ramp': {'n': int, 'return': list},
        'colours': {'n': int, 'return': list},
        'swatch': {'name': str, 'return': list},
    }
# [AGENT EDIT END]
