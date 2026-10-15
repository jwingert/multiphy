# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the generic layer names the fingerprint an event response, not a sound; stored keys keep their names (launcher-gui Task 10) | date: 2026-10-07]
"""The generic layer's words: an event response and events, never a sound."""
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from PySide6 import QtWidgets

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams test builds its result with saved weights (event-optional Task 12) | date: 2026-10-10]
from nems_lbhb.unit_match import backends
# [AGENT EDIT END]
from nems_lbhb.unit_match.contract import names
from nems_lbhb.unit_match.contract.names import backend_label
from nems_lbhb.unit_match.viewer import display
from nems_lbhb.unit_match.viewer.app import App
from nems_lbhb.unit_match.viewer.display import headline_probability, probability
from nems_lbhb.unit_match.viewer.panels.plots.fingerprint import FingerprintPanel
from nems_lbhb.unit_match.viewer.panels.plots.mixture import MixturePanel

from .viewer_fakes import FAKE_MODE, DictData

# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the label names firing, waveform and optional event-response likelihood ratios (event-optional Task 13, D38) | date: 2026-10-10]
LABEL = 'Functional likelihood — firing, waveform and optional event-response likelihood ratios'
# [AGENT EDIT END]
PKG = Path(__file__).resolve().parents[2] / 'nems_lbhb' / 'unit_match'
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the day table's text is checked too (native-same-day Task 9) | date: 2026-10-11]
GENERIC = ['contract/names.py', 'viewer/display.py', 'viewer/app.py',
           'viewer/panels/plots/mixture.py', 'viewer/panels/plots/fingerprint.py',
           'viewer/panels/tables/evidence.py', 'viewer/panels/tables/rivals.py',
           'viewer/panels/tables/browse.py', 'viewer/panels/tables/days.py', 'backends/local/__init__.py',
           'backends/local/functional_likelihood.py', 'backends/local/pair_logistic.py',
           'backends/local/evidence.py', 'clusters.py', '__init__.py', 'docs/viewer.md',
           'docs/contract.md', 'docs/adding_a_backend.md']
# [AGENT EDIT END]


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_backend_label_says_event_response():
    assert names.NAMES['backend']['functional_likelihood'].label == LABEL
    assert backend_label('functional_likelihood') == LABEL
    dant = names.NAMES['definitions']['dant'].label
    assert 'sound' not in dant
    assert dant.endswith('event-relative time bins.')


# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the rivals table alone carries the evt column (native-same-day Task 10 fix round 1) | date: 2026-10-11]
def test_viewer_strings_say_events(qt, tmp_path):
    assert display.P_EVENT == 'p_stim'
    assert not hasattr(display, 'P_SOUND')
    assert headline_probability({'posterior': 0.7}) == (0.7, 'P(same neuron), events only')
    assert probability({'p_stim': 0.8}, display.P_EVENT) == 0.8
    assert FingerprintPanel.panel_title == 'Fingerprint, shared conditions'
    assert FingerprintPanel.ALL == '— all conditions —'
    assert MixturePanel.ORDER[0] == ('stim', 'event-response fingerprint', 's_stim', 'p_stim')

    window = App(FAKE_MODE, layout_file=tmp_path / 'l.json')
    for header in (window.rivals.table.horizontalHeaderItem(2),):
        assert header.text() == 'evt'
        assert header.toolTip() == 'event-response fingerprint only'
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams line names the streams the pair's group saved weights for, so the result saves its group's event and physiology weights and run_evidence (event-optional Task 12) | date: 2026-10-10]
    units = pd.DataFrame({'cellid': ['a-01', 'b-01'], 'siteid': ['a', 'b'], 'su_type': 'SU'})
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'], 'recording_group': ['G'], 'p_global': [0.9],
                          'p_stim': [0.8], 'p_phys': [0.6]})
    provenance = {'mixture_references': {'G': {'weight_stim': 0.5, 'weight_phys': 0.5}},
                  'run_evidence': {'G': {'events': True, 'spont': 'baseline', 'shared': {}, 'reasons': []}}}
    result = backends.result_from_pairs('dict_data', '1', units, pairs, score_column='p_global', run_id='dict_data',
                                        provenance=provenance)
    window.set_data(DictData(units, {}, {}, dict(zip(units.cellid, units.siteid)), result=result))
    # [AGENT EDIT END]
    window.select_pair('a-01', 'b-01')
    assert window.evidence.streams.text() == 'events 80% · physiology 60%'
    window.close()
# [AGENT EDIT END]


def test_no_sound_in_generic_layer():
    sound = re.compile(r'(?i)\bsounds?\b')
    for rel in GENERIC:
        for n, line in enumerate((PKG / rel).read_text().splitlines(), 1):
            if '[AGENT EDIT' in line:
                continue
            assert not sound.search(line), f'{rel}:{n}: {line}'
    readme = (PKG / 'README.md').read_text()
    # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the README names firing, waveform and optional event-response likelihood ratios (event-optional Task 13) | date: 2026-10-10]
    for new in ('scores firing, waveform and optional event-response likelihood ratios',
                # [AGENT EDIT END]
                '| event response | fingerprint correlation |',
                'The event response comes out heavier wherever it has been fitted',
                'A pair the event response convinces on but the',
                '| Fingerprint | the event response, all shared conditions or one at full width |',
                'Matching on event responses selects for similar event responses.',
                '## Choosing the sounds'):
        assert new in readme, new
    for old in ('scores sound and firing', '| sound | fingerprint correlation |',
                'Matching on sound responses'):
        assert old not in readme, old
# [AGENT EDIT END]
