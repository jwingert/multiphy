"""A merged run's group runs carry its fitted mixtures to the viewers; its registration offsets stay in the run."""
import copy
import dataclasses
import json
import os

import numpy as np
import pytest

from multiphy.backends.local import functional_likelihood as fl
from multiphy.contract.provenance import Provenance, payload_digest
from multiphy.viewer.app import open_run
from multiphy.project.folder import Project
from multiphy.project.selection import Selection
from multiphy.run.project_run import run_selection
from multiphy.viewer import data
from PySide6 import QtWidgets

from .synthetic_day import GEOMETRY, extracted_project


def _group_run(group='P', session_a='D1', session_b='D2', offsets=None):
    provenance = {
        'session_pairs': [{'recording_group': group, 'session_a': session_a, 'session_b': session_b,
                           'streams': {'stim': {'observed': [0.5, 0.7, 0.9], 'pi': 0.25},
                                       'phys': {'observed': [0.1], 'unavailable': 'mixture not fitted'}}}],
        'mixture_references': {group: {'weight_stim': 0.6, 'weight_phys': 0.4, 'bias': 0.0,
                                       'streams': {'stim': {'same': [1.0] * 5, 'diff': [0.0] * 5},
                                                   'phys': {'same': [2.0] * 5, 'diff': [3.0] * 5}}}}}
    if offsets is not None:
        provenance['registration'] = {'method': 'matches', 'offsets_um': offsets}
    return {'run_id': 'r-' + group, 'provenance': provenance}


def test_mixture_day_pairs_reads_group_runs():
    provenance = Provenance(extra={'group_runs': {'P': _group_run()}})
    assert data.mixture_day_pairs(provenance) == {frozenset({'D1', 'D2'}): {
        'streams': {'stim': {'same': [1.0] * 5, 'diff': [0.0] * 5, 'observed': [0.5, 0.7, 0.9], 'pi': 0.25}},
        'weight_stim': 0.6, 'weight_phys': 0.4}}


def test_mixture_day_pairs_ignores_top_level():
    top = _group_run()['provenance']
    provenance = Provenance(extra={'mixture_references': top['mixture_references'],
                                   'session_pairs': top['session_pairs']})
    assert data.mixture_day_pairs(provenance) == {}


def test_apply_run_provenance_keeps_build_streams():
    provenance = Provenance(extra={'group_runs': {
        'P': _group_run('P', offsets={'D1': 0.0, 'D2': 12.5}),
        'Q': _group_run('Q', 'D3', 'D4')}})
    built = {'siteid_a': 'D3', 'siteid_b': 'D4', 'streams': {'stim': {'pi': 0.9}}, 'weight_stim': 0.1}
    day_pairs = [{'siteid_a': 'D2', 'siteid_b': 'D1', 'days_apart': 1}, copy.deepcopy(built)]
    out = data.apply_run_provenance(day_pairs, provenance)
    assert out[1] == built
    assert out[0]['streams']['stim']['pi'] == 0.25
    assert out[0]['weight_phys'] == 0.4


@pytest.fixture(scope='module')
def qt():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_mixture_panel_draws_the_fitted_model(qt):
    """The drawn P(same) curve is the posterior of the Mixture the backend fits, bandwidth included."""
    import numpy as np
    import pandas as pd
    from multiphy import evidence as ev
    rng = np.random.default_rng(0)
    same, diff = rng.normal(1.0, 0.4, 40), rng.normal(-0.5, 0.6, 60)
    model = ev.Mixture(same, diff).fit(np.concatenate([same[:10], diff[:30]]))
    from multiphy.viewer.panels.plots.mixture import MixturePanel
    from multiphy.viewer.state import PairState
    from .viewer_fakes import DictData
    state = PairState()
    state.set_data(DictData(['D1:0', 'D2:0'], {}, {}, {'D1:0': 'D1', 'D2:0': 'D2'}))
    state.day_pair = {'siteid_a': 'D1', 'siteid_b': 'D2', 'days_apart': 1, 'weight_stim': 1.0, 'weight_phys': 0.0,
                      'streams': {'stim': {'same': list(same), 'diff': list(diff), 'observed': [0.1], 'pi': model.pi}}}
    state.row = pd.Series({'s_stim': 0.2, 'p_stim': float(model.posterior(0.2)[0])})
    panel = MixturePanel(state)
    panel.refresh()
    assert len(panel.axes[0].lines) > 0
    assert [t.get_text() for t in panel.axes[1].texts] == ['not fitted']
    (twin,) = [ax for ax in panel.fig.axes if ax not in panel.axes]
    grid, drawn = twin.lines[0].get_data()
    inside = np.isfinite(drawn)
    np.testing.assert_allclose(drawn[inside], 100 * model.posterior(grid[inside]), rtol=1e-9, atol=1e-9)
    qt.processEvents()          # the panel's queued redraw runs while its canvas still exists
    panel.close()


def test_days_viewer_draws_measured_positions(qt, tmp_path, monkeypatch, capsys):
    # Three good units a day give at most 6 same-day neighbour pairs; the functional-likelihood
    # backend needs 30 to fit references, so on the synthetic days it registers nothing. The
    # backend's own result gains the offsets and fitted mixture of a registered group before the
    # runner saves it; the runner writes result.json once and open_run reads it untouched.
    real = fl.run_functional_likelihood

    def registered(*args, **kwargs):
        result = real(*args, **kwargs)
        return dataclasses.replace(result, provenance=result.provenance.merged(
            extra=_group_run('P', offsets={'D1': 0.0, 'D2': 12.5})['provenance']))

    monkeypatch.setattr(fl, 'run_functional_likelihood', registered)
    folder = extracted_project(tmp_path, weak_units=('D1',))
    run = run_selection(Selection(Project.open(folder.parent, folder.name), ('D1', 'D2')), 'functional_likelihood')
    saved = json.loads((run / 'result.json').read_text())
    assert saved['sha256'] == payload_digest(saved['result'])
    (group,) = saved['result']['provenance']['group_runs'].values()
    offsets = group['provenance']['registration']['offsets_um']
    assert offsets == {'D1': 0.0, 'D2': 12.5}
    probe_depths = GEOMETRY[:, 1].copy()
    window = open_run(run)
    data = window.state.data
    assert not hasattr(data, 'day_offset')
    assert data.day_pairs()[0]['streams']['stim']['pi'] == 0.25
    probe = window._docks['probe'].widget()
    assert [b.text() for b in probe.findChildren(QtWidgets.QCheckBox)] == ['whole probe']
    # A selection without a candidate row (an unmatched unit, a longer track) shows the mixture tag.
    method = window._docks['mixture'].widget()
    cid = next(c for c, site in window.state.sites.items() if site == 'D2')
    window.state.set_selection([cid])
    assert window.state.row is None
    assert [t.get_text() for t in method.axes[0].texts] == ['mixture: not fitted']
    footprint = window._docks['footprint'].widget()
    footprint.refresh()
    ax = footprint.axes[0]
    measured = np.asarray(data.geometry('D2'), float)[:, 1]
    assert len(measured) == 8
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), measured)
    np.testing.assert_array_equal(measured, probe_depths)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), probe_depths)
    assert ax.get_title() == 'Measured coordinates'
    assert 'Traceback' not in capsys.readouterr().err
    # Leave no queued layout or redraw work behind for later event-loop timing tests.
    window.close()
    window.deleteLater()
    qt.processEvents()


