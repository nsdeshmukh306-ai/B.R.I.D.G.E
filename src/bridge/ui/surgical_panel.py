"""The SURGICAL CASE panel: the case, the live count board and the alert log.

Read-only mirror of `JarvisAssistant.snapshot()` plus the handful of buttons a
circulating nurse needs when their hands are free. Everything here is a
convenience: every action is reachable by voice, because in a sterile field the
voice channel is the only one available.
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

STATUS_STYLE = {
    "ok": "#5ae68c",
    "short": "#ff5a5a",
    "over": "#ffb43c",
    "pending": "#a8b0bb",
}
LEVEL_STYLE = {"critical": "#ff5a5a", "caution": "#ffb43c", "info": "#a8b0bb"}


class SurgicalPanel(QGroupBox):
    """Panel for the case autopilot and the surgical count."""

    alert_signal = Signal(object)   # marshals monitor alerts onto the Qt thread

    def __init__(self, core, parent: Optional[QWidget] = None):
        super().__init__("SURGICAL CASE", parent)
        self.core = core
        self._last_signature: tuple = ()
        self._build()
        self.alert_signal.connect(self._append_alert)

    # -- layout ---------------------------------------------------------------------------
    def _build(self) -> None:
        lay = QVBoxLayout(self)

        row = QHBoxLayout()
        self.cb_set = QComboBox()
        assistant = getattr(self.core, "assistant", None)
        if assistant is not None:
            for sid, name in assistant.sets.names():
                self.cb_set.addItem(name, sid)
        row.addWidget(self.cb_set, 1)
        self.btn_start = QPushButton("START CASE")
        self.btn_start.setObjectName("primary")
        self.btn_start.clicked.connect(self._start_case)
        row.addWidget(self.btn_start)
        lay.addLayout(row)

        self.lbl_phase = QLabel("No case running.")
        self.lbl_phase.setWordWrap(True)
        self.lbl_phase.setObjectName("mono")
        lay.addWidget(self.lbl_phase)

        self.lbl_item = QLabel("")
        self.lbl_item.setWordWrap(True)
        lay.addWidget(self.lbl_item)

        row = QHBoxLayout()
        for text, phrase in (("Confirm", "confirmed"), ("Skip", "skip"), ("Back", "back"),
                             ("Next phase", "next phase")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, p=phrase: self._say(p))
            row.addWidget(b)
        lay.addLayout(row)

        row = QHBoxLayout()
        for text, phrase in (("Count in", "start the count"), ("As per sheet", "count as per the sheet"),
                             ("Closing", "closing count"), ("Final", "final count")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, p=phrase: self._say(p))
            row.addWidget(b)
        lay.addLayout(row)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["item", "count", "seen"])
        self.table.verticalHeader().setVisible(False)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.setMinimumHeight(150)
        lay.addWidget(self.table, 1)

        row = QHBoxLayout()
        self.chk_proactive = QCheckBox("Speak alerts")
        self.chk_proactive.setChecked(True)
        self.chk_proactive.toggled.connect(self._toggle_proactive)
        row.addWidget(self.chk_proactive)
        self.chk_board = QCheckBox("Project count board")
        self.chk_board.setChecked(True)
        self.chk_board.toggled.connect(self._toggle_board)
        row.addWidget(self.chk_board)
        lay.addLayout(row)

        row = QHBoxLayout()
        self.btn_missing = QPushButton("What's missing")
        self.btn_missing.clicked.connect(lambda: self._say("what's missing"))
        row.addWidget(self.btn_missing)
        self.btn_close = QPushButton("CLOSE CASE")
        self.btn_close.setObjectName("danger")
        self.btn_close.clicked.connect(lambda: self._say("close the case"))
        row.addWidget(self.btn_close)
        lay.addLayout(row)

        self.lbl_alerts = QLabel("")
        self.lbl_alerts.setWordWrap(True)
        self.lbl_alerts.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.lbl_alerts)
        self._alerts: list[str] = []

    # -- actions ---------------------------------------------------------------------------
    def _say(self, phrase: str) -> None:
        """Every button goes through the same spoken entry point the voice uses."""
        reply = self.core.handle_spoken(phrase)
        if reply:
            self.lbl_item.setText(reply)
        self.refresh(force=True)

    def _start_case(self) -> None:
        name = self.cb_set.currentText()
        self._say(f"start a case for a {name}")

    def _toggle_proactive(self, on: bool) -> None:
        if self.core.assistant is not None:
            self.core.assistant.proactive = on

    def _toggle_board(self, on: bool) -> None:
        if self.core.assistant is not None:
            self.core.assistant.set_board_visible(on)

    def on_alert(self, alert) -> None:
        """Called from the camera thread — hop to the Qt thread via the signal."""
        self.alert_signal.emit(alert)

    def _append_alert(self, alert) -> None:
        colour = LEVEL_STYLE.get(getattr(alert, "level", "info"), "#a8b0bb")
        self._alerts.append(f'<span style="color:{colour}">■ {alert.text}</span>')
        self._alerts = self._alerts[-4:]
        self.lbl_alerts.setText("<br>".join(reversed(self._alerts)))

    # -- refresh ----------------------------------------------------------------------------
    def refresh(self, force: bool = False) -> None:
        a = getattr(self.core, "assistant", None)
        if a is None:
            self.setEnabled(False)
            return
        snap = a.snapshot()
        signature = (snap["phase"], snap["counted"], snap["expected"], snap["verdict"],
                     snap["checklist_item"], len(snap["board"]), snap["entities"])
        if signature == self._last_signature and not force:
            return
        self._last_signature = signature

        head = snap["phase_title"]
        if snap["set"]:
            head += f" · {snap['set']}"
        if snap["expected"]:
            head += f" · {snap['counted']}/{snap['expected']} ({snap['verdict']})"
        head += f" · {snap['entities']} objects seen, {snap['labelled']} named"
        self.lbl_phase.setText(head)
        if snap["checklist_item"]:
            self.lbl_item.setText(snap["checklist_item"])

        board = snap["board"]
        self.table.setRowCount(len(board))
        for r, (_cat, name, counted, baseline, status) in enumerate(board):
            cells = [name, f"{counted}/{baseline}" if baseline else str(counted),
                     str(next((l.observed for l in a.counts.lines if l.name == name), 0))]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c and status in STATUS_STYLE:
                    from PySide6.QtGui import QColor

                    item.setForeground(QColor(STATUS_STYLE[status]))
                self.table.setItem(r, c, item)
