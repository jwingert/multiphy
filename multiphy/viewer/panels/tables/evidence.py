# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the scalar evidence and verdict controls, from gui.py:1350-1471; the verdict buttons call actions.set_verdict and the fields come from display.EVIDENCE_FIELDS; fix round 2: the constructor declares state: PairState and actions: Actions (viewer-split Task 6, fix round 2) | date: 2026-10-06]
"""Scalar evidence, the criterion outcomes, and the verdict controls."""
import numpy as np

from multiphy import store as st
from multiphy.viewer import display
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: labels, buttons and cells carry theme properties and tokens instead of colours (launcher-gui Task 8) | date: 2026-10-07]
from multiphy.viewer import theme
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: P_SOUND renamed P_EVENT (launcher-gui Task 10) | date: 2026-10-07]
# [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams line reads each stream's word and probability column from display (event-optional Task 12) | date: 2026-10-10]
from multiphy.viewer.display import (STREAM_WORDS, diagnostic_gate_outcome, headline_probability,
                                                 pct, probability, shows_scores)
# [AGENT EDIT END]
# [AGENT EDIT END]
from multiphy.viewer.panels.tables.base import _table
from multiphy.viewer.state import Actions, PairState
from multiphy.viewer.qt import QtWidgets
# [AGENT EDIT END]


class EvidencePanel(QtWidgets.QWidget):
    """Scalar evidence, the criterion outcomes, and the verdict controls."""
    panel_id, panel_title = 'evidence', 'Evidence and verdict'

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
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a hint label the app style sheet colours (launcher-gui Task 8) | date: 2026-10-07]
        self.streams.setProperty('role', 'hint')
        self.streams.setStyleSheet('font-size:12px;')
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams tooltip names the waveform stream and says the line lists the streams the pair's group saved weights for (event-optional Task 12) | date: 2026-10-10]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams tooltip names the event-response stream (launcher-gui Task 10) | date: 2026-10-07]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the physiology tooltip names the session rate event runs add to the stream (event-optional Task 12, final review fix round 1) | date: 2026-10-10]
        self.streams.setToolTip(
            'The event-response stream is the fingerprint correlation. The physiology '
            'stream is spontaneous rate, interval histogram and autocorrelogram, plus '
            'session rate in event runs. '
            'The waveform stream is waveform and footprint correlation and spike width. '
            'The line lists the streams the pair\'s recording group saved a weight for; '
            'the combined probability weighs them by a fit to the reference pairs.')
        # [AGENT EDIT END]
        # [AGENT EDIT END]
        # [AGENT EDIT END]
        lay.addWidget(self.streams)
        self.table = _table(['evidence', 'value', 'criterion'], [220, 66, 70])
        self.table.horizontalHeader().setSectionResizeMode(2, QtWidgets.QHeaderView.ResizeToContents)
        self.table.setSortingEnabled(False)
        self.table.horizontalHeader().setSectionsClickable(False)
        self.table.setRowCount(len(self.FIELDS))
        lay.addWidget(self.table, 1)
        self.notes = QtWidgets.QPlainTextEdit()
        self.notes.setPlaceholderText('notes for this pair')
        self.notes.setMaximumHeight(56)
        lay.addWidget(self.notes)
        brow = QtWidgets.QHBoxLayout()
        self.verdict_buttons = []
        for v in st.VERDICTS:
            b = QtWidgets.QPushButton(f'{v.capitalize()}  ({v[0].upper()})')
            # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the verdict button is outlined in its status colour by the app style sheet (launcher-gui Task 8) | date: 2026-10-07]
            b.setProperty('verdict', v)
            b.setStyleSheet('padding:7px 10px;')
            # [AGENT EDIT END]
            b.clicked.connect(lambda _=False, vv=v: self.actions.set_verdict(vv))
            brow.addWidget(b)
            self.verdict_buttons.append(b)
        lay.addLayout(brow)
        self.status = QtWidgets.QLabel('')
        self.status.setWordWrap(True)
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: a hint label the app style sheet colours (launcher-gui Task 8) | date: 2026-10-07]
        self.status.setProperty('role', 'hint')
        # [AGENT EDIT END]
        lay.addWidget(self.status)
        state.changed.connect(self.refresh)

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
                # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the gate cell carries its outcome's theme token (launcher-gui Task 8) | date: 2026-10-07]
                theme.tone(it, 'muted' if ok is None else 'accept' if ok else 'reject')
                # [AGENT EDIT END]
                self.table.setItem(i, 2, it)
        p, label = headline_probability(r)
        shown = pct(p) if r.get('score_kind') == 'calibrated_probability' else (f'{p:.3f}' if np.isfinite(p) else '--')
        self.posterior.setText(f'{label} = {shown}')
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the headline's tone property picks its status colour from the app style sheet (launcher-gui Task 8) | date: 2026-10-07]
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
        # [AGENT EDIT END]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams line names the streams the row's group saved weights for, not the columns present (event-optional Task 12) | date: 2026-10-10]
        # [AGENT EDIT START | agent: claude-opus-5-5 | user: wingertj | reason: the streams line reads events, not sound (launcher-gui Task 10) | date: 2026-10-07]
        data = self.state.data
        streams = display.group_streams(data.result.provenance, r.get('recording_group')) if data is not None else []
        self.streams.setText(' · '.join(f'{STREAM_WORDS[s]} {pct(probability(r, "p_" + s))}' for s in streams))
        # [AGENT EDIT END]
        # [AGENT EDIT END]
# [AGENT EDIT END]
