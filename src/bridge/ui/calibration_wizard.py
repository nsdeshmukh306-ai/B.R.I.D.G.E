"""Auto-calibration wizard: intro -> detecting -> result / failure."""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget,
)

from bridge.spatial.calibration import CalibrationResult
from bridge.ui.widgets import ImageView, Worker, run_in_background


class CalibrationWizard(QDialog):
    """`run_calibration(progress_cb) -> CalibrationResult` is executed on a worker thread."""

    saved = Signal()

    def __init__(self, run_calibration: Callable[[Callable[[str, float], None]], CalibrationResult],
                 save_profile: Callable[[], bool], camera_name: str, display_name: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("BRIDGE — Auto Calibration")
        self.setModal(True)
        self.resize(640, 480)
        self._run = run_calibration
        self._save = save_profile
        self.result: Optional[CalibrationResult] = None
        self._worker: Optional[Worker] = None
        self.camera_name, self.display_name = camera_name, display_name

        self.stack = QStackedWidget()
        root = QVBoxLayout(self)
        root.addWidget(self.stack)
        self.stack.addWidget(self._page_intro())
        self.stack.addWidget(self._page_running())
        self.stack.addWidget(self._page_result())

    # -- pages ---------------------------------------------------------------------------------
    def _page_intro(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        t = QLabel("AUTO CALIBRATION")
        t.setObjectName("title")
        lay.addWidget(t)
        lay.addWidget(QLabel("Place the camera and projector so they both see the same surface.\n\n"
                             "BRIDGE will project a marker pattern, detect it with the camera and compute the\n"
                             "camera → projector mapping. Keep hands and objects away from the projected markers."))
        lay.addStretch()
        b = QPushButton("START")
        b.setObjectName("primary")
        b.clicked.connect(self._start)
        lay.addWidget(b)
        return w

    def _page_running(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_running = QLabel("Detecting surface...")
        self.lbl_running.setObjectName("title")
        lay.addWidget(self.lbl_running)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        lay.addWidget(self.progress)
        lay.addStretch()
        return w

    def _page_result(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        self.lbl_result_title = QLabel("Calibration complete")
        self.lbl_result_title.setObjectName("title")
        lay.addWidget(self.lbl_result_title)
        self.lbl_result = QLabel("")
        self.lbl_result.setObjectName("mono")
        self.lbl_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        lay.addWidget(self.lbl_result)
        self.debug_view = ImageView("(camera view of pattern)")
        lay.addWidget(self.debug_view, 1)
        row = QHBoxLayout()
        self.btn_save = QPushButton("SAVE")
        self.btn_save.setObjectName("primary")
        self.btn_save.clicked.connect(self._on_save)
        self.btn_retry = QPushButton("RETRY")
        self.btn_retry.clicked.connect(self._start)
        self.btn_close = QPushButton("CLOSE")
        self.btn_close.clicked.connect(self.reject)
        row.addWidget(self.btn_save)
        row.addWidget(self.btn_retry)
        row.addWidget(self.btn_close)
        lay.addLayout(row)
        return w

    # -- flow --------------------------------------------------------------------------------------
    def _start(self) -> None:
        self.stack.setCurrentIndex(1)
        self.progress.setValue(0)
        self.lbl_running.setText("Detecting surface...")
        worker = Worker(self._run_with_progress)
        worker.signals.progress.connect(self._on_progress)
        worker.signals.finished.connect(self._on_done)
        worker.signals.error.connect(lambda e: self._on_done(CalibrationResult(success=False, message=e)))
        self._worker = worker
        from PySide6.QtCore import QThreadPool

        QThreadPool.globalInstance().start(worker)

    def _run_with_progress(self) -> CalibrationResult:
        assert self._worker is not None
        sig = self._worker.signals
        return self._run(lambda msg, frac: sig.progress.emit(msg, frac))

    def _on_progress(self, msg: str, frac: float) -> None:
        self.lbl_running.setText(msg)
        self.progress.setValue(int(frac * 100))

    def _on_done(self, result: CalibrationResult) -> None:
        self.result = result
        self.stack.setCurrentIndex(2)
        if result.success and result.validation is not None:
            v = result.validation
            self.lbl_result_title.setText("Calibration complete")
            self.lbl_result_title.setStyleSheet("color: #3ddc97;")
            self.lbl_result.setText(
                f"Accuracy: {v.mean_error_px:.1f} px (max {v.max_error_px:.1f} px, n={v.n_points})\n"
                f"Workspace detected\nProjector: {self.display_name}\nCamera: {self.camera_name}\nAttempts: {result.attempts}")
            self.btn_save.setEnabled(True)
        else:
            self.lbl_result_title.setText("Calibration failed.")
            self.lbl_result_title.setStyleSheet("color: #ff6b6b;")
            self.lbl_result.setText(result.message)
            self.btn_save.setEnabled(False)
        frame = result.debug_frames.get("pattern")
        self.debug_view.set_frame(frame)

    def _on_save(self) -> None:
        if self._save():
            self.saved.emit()
            self.accept()
        else:
            self.lbl_result.setText(self.lbl_result.text() + "\n\nCould not save profile (camera/display not selected).")
