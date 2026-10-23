"""Scalar evidence, the criterion outcomes, and the verdict controls."""
import numpy as np

from multiphy import store as st
from multiphy.viewer import display
from multiphy.viewer import theme
from multiphy.viewer import keys
from multiphy.viewer.display import (STREAM_WORDS, diagnostic_gate_outcome, headline_probability,
                                                 pct, probability, shows_scores)
from multiphy.viewer.panels.tables.base import _table
from multiphy.viewer.state import Actions, PairState
from multiphy.viewer.qt import QtWidgets


class EvidencePanel(QtWidgets.QWidget):
    """Scalar evidence, the criterion outcomes, and the verdict controls."""
    panel_id, panel_title = 'evidence', 'Pair evidence'

    FIELDS = display.EVIDENCE_FIELDS

    def __init__(self, state: PairState, actions: Actions, parent=None):
        super().__init__(parent)
        self.state, self.actions = state, actions
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.posterior = QtWidgets.QLabel('')
        self.posterior.setStyleSheet('font-size:19px; font-weight:700;')
        lay.addWidget(self.posterior)
        self.streams = QtWidgets.QLabel('')
        self.streams.setProperty('role', 'hint')
        self.streams.setStyleSheet('font-size:12px;')
        self.streams.setToolTip(
            'The event-response stream is the fingerprint correlation. The physiology '
            'stream is spontaneous rate, interval histogram and autocorrelogram, plus '
            'session rate in event runs. '
            'The waveform stream is waveform and footprint correlation and spike width. '
            'The line lists the streams the pair\'s recording group saved a weight for; '
            'the combined probability weighs them by a fit to the reference pairs.')
        lay.addWidget(self.streams)
        self.table = _table(['evidence', 'value', 'criterion'], [220, 66, 70])
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionsClickable(False)
        self.table.setRowCount(len(self.FIELDS))
        lay.addWidget(self.table, 1)
        self.pair_review = QtWidgets.QWidget()
        review_layout = QtWidgets.QVBoxLayout(self.pair_review)
        review_layout.setContentsMargins(0, 0, 0, 0)
        hint = QtWidgets.QLabel('Advanced pair review: independent of track approval and export.')
        hint.setWordWrap(True)
        hint.setProperty('role', 'hint')
        review_layout.addWidget(hint)
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText('notes for this pair')
        self.notes.setMaximumHeight(56)
        review_layout.addWidget(self.notes)
        brow = QtWidgets.QHBoxLayout()
        self.verdict_buttons = []
        for v in st.VERDICTS:
            b = QtWidgets.QPushButton()
            b.setProperty('verdict', v)
            b.setStyleSheet('padding:7px 10px;')
            b.clicked.connect(lambda _=False, vv=v: self.actions.set_verdict(vv))
            brow.addWidget(b)
            self.verdict_buttons.append(b)
        review_layout.addLayout(brow)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        self.status.setProperty('role', 'hint')
        review_layout.addWidget(self.status)
        lay.addWidget(self.pair_review)
        self.pair_review.hide()
        self.bind_keys()
        keys.notifier().changed.connect(self.bind_keys)
        state.changed.connect(self.refresh)

    def bind_keys(self):
        """Name pair decisions explicitly and show their actual, possibly customized shortcuts."""
        bound = keys.bindings()
        labels = {'accept': 'Pair same', 'reject': 'Pair different', 'unsure': 'Pair unsure'}
        for button in self.verdict_buttons:
            verdict = button.property('verdict')
            key = bound[verdict]
            button.setText(labels[verdict] + (f' ({key})' if key else ''))
            button.setToolTip('Save a pair verdict and advance to the next candidate. '
                              'Does not change track membership or export eligibility.')

    def refresh(self):
        r = self.state.row
        if r is None:
            self.table.clearContents()
            self.posterior.setText('No pair selected')
            self.streams.clear()
            self.notes.clear()
            self.status.clear()
            return
        for i, (key, label, gate) in enumerate(self.FIELDS):
            v = r.get(key, np.nan)
            self.table.setItem(i, 0, QtWidgets.QTableWidgetItem(label))
            txt = '--' if v is None or (isinstance(v, float) and not np.isfinite(v)) else f'{v:.3f}'
            self.table.setItem(i, 1, QtWidgets.QTableWidgetItem(txt))
            if gate is None:
                self.table.setItem(i, 2, QtWidgets.QTableWidgetItem(''))
            else:
                ok = diagnostic_gate_outcome(r, gate)
                it = QtWidgets.QTableWidgetItem('not evaluated' if ok is None else 'pass' if ok else 'fail')
                it.setToolTip('No criterion result was computed for this measurement.' if ok is None
                              else 'Saved diagnostic criterion result; this need not be an automatic matching input.')
                theme.tone(it, 'muted' if ok is None else 'accept' if ok else 'reject')
                self.table.setItem(i, 2, it)
        p, label = headline_probability(r)
        shown = pct(p) if r.get('score_kind') == 'calibrated_probability' else (f'{p:.3f}' if np.isfinite(p) else '--')
        self.posterior.setText(f'{label} = {shown}')
        self.posterior.setProperty('tone', 'accept' if np.isfinite(p) and p >= .8 else
                                   'reject' if np.isfinite(p) and p < .5 else 'unsure')
        if not shows_scores(r):
            if not r.get('score_display_label'):   # DANT keeps its native-membership label
                self.posterior.setText('no matching scores')
            self.posterior.setProperty('tone', 'muted')
            theme.repolish(self.posterior)
            self.streams.clear()
            return
        theme.repolish(self.posterior)
        data = self.state.data
        streams = display.group_streams(data.result.provenance, r.get('recording_group')) if data is not None else []
        self.streams.setText(' · '.join(f'{STREAM_WORDS[s]} {pct(probability(r, "p_" + s))}' for s in streams))
