"""The generic layer's words: an event response and events, never a sound."""
import os
import re
from pathlib import Path

import pandas as pd
import pytest
from PySide6 import QtWidgets

from multiphy import backends
from multiphy.contract import names
from multiphy.contract.names import backend_label
from multiphy.viewer import display
from multiphy.viewer.app import App
from multiphy.viewer.display import headline_probability, probability
from multiphy.viewer.panels.plots.fingerprint import FingerprintPanel
from multiphy.viewer.panels.plots.mixture import MixturePanel

from .viewer_fakes import FAKE_MODE, DictData

LABEL = 'Functional likelihood — firing, waveform and optional event-response likelihood ratios'
PKG = Path(__file__).resolve().parents[1] / 'multiphy'
GENERIC = ['contract/names.py', 'viewer/display.py', 'viewer/app.py',
           'viewer/panels/plots/mixture.py', 'viewer/panels/plots/fingerprint.py',
           'viewer/panels/tables/evidence.py', 'viewer/panels/tables/rivals.py',
           'viewer/panels/tables/browse.py', 'viewer/panels/tables/days.py', 'backends/local/__init__.py',
           'backends/local/functional_likelihood.py', 'backends/local/pair_logistic.py',
           'backends/local/evidence.py', 'clusters.py', '__init__.py', 'docs/viewer.md',
           'docs/contract.md', 'docs/adding_a_backend.md']


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
    units = pd.DataFrame({'cellid': ['a-01', 'b-01'], 'siteid': ['a', 'b'], 'su_type': 'SU'})
    pairs = pd.DataFrame({'cellid_a': ['a-01'], 'cellid_b': ['b-01'], 'recording_group': ['G'], 'p_global': [0.9],
                          'p_stim': [0.8], 'p_phys': [0.6]})
    provenance = {'mixture_references': {'G': {'weight_stim': 0.5, 'weight_phys': 0.5}},
                  'run_evidence': {'G': {'events': True, 'spont': 'baseline', 'shared': {}, 'reasons': []}}}
    result = backends.result_from_pairs('dict_data', '1', units, pairs, score_column='p_global', run_id='dict_data',
                                        provenance=provenance)
    window.set_data(DictData(units, {}, {}, dict(zip(units.cellid, units.siteid)), result=result))
    window.select_pair('a-01', 'b-01')
    assert window.evidence.streams.text() == 'events 80% · physiology 60%'
    window.close()


def test_no_sound_in_generic_layer():
    sound = re.compile(r'(?i)\bsounds?\b')
    for rel in GENERIC:
        for n, line in enumerate((PKG / rel).read_text().splitlines(), 1):
            assert not sound.search(line), f'{rel}:{n}: {line}'
    readme = (PKG / 'README.md').read_text()
    for new in ('scores firing, waveform and optional event-response likelihood ratios',
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
