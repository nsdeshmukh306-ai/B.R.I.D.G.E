"""Reusable Qt widgets: image view, log panel, worker helpers, theme."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QSizePolicy, QWidget

from bridge.spatial.geometry import Point

STYLESHEET = """
QMainWindow, QDialog, QWizard { background: #14171c; }
QWidget { color: #e6e9ef; font-size: 13px; }
QGroupBox { border: 1px solid #2a2f38; border-radius: 8px; margin-top: 14px; padding: 8px 8px 6px 8px; font-weight: 600; color: #9aa3b2; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QPushButton { background: #232833; border: 1px solid #333a47; border-radius: 6px; padding: 7px 12px; }
QPushButton:hover { background: #2c3340; }
QPushButton:disabled { color: #6b7280; }
QPushButton#primary { background: #106563; border-color: #1c8d8a; font-weight: 700; }
QPushButton#primary:hover { background: #1a7f7c; }
QPushButton#accent { background: #163C71; border-color: #2a5aa8; font-weight: 700; }
QPushButton#accent:hover { background: #1d4d8f; }
QPushButton#danger { background: #4a1d20; border-color: #B32328; }
QComboBox, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox { background: #0f1115; border: 1px solid #333a47; border-radius: 6px; padding: 5px; }
QLabel#status_ok { color: #3ddc97; font-weight: 600; }
QLabel#status_warn { color: #E29119; font-weight: 600; }
QLabel#status_err { color: #ff6b6b; font-weight: 600; }
QLabel#title { font-size: 20px; font-weight: 800; color: #ffffff; }
QLabel#subtitle { color: #9aa3b2; }
QLabel#mono { font-family: Menlo, Consolas, monospace; color: #c8d0dc; }
QRadioButton, QCheckBox { padding: 2px; color: #e6e9ef; }
QRadioButton::indicator { width: 14px; height: 14px; }
QProgressBar { border: 1px solid #333a47; border-radius: 6px; background: #0f1115; text-align: center; }
QProgressBar::chunk { background: #106563; border-radius: 5px; }
"""


def ndarray_to_pixmap(bgr: np.ndarray) -> QPixmap:
    h, w = bgr.shape[:2]
    if bgr.ndim == 2:
        img = QImage(np.ascontiguousarray(bgr).data, w, h, w, QImage.Format.Format_Grayscale8)
    else:
        rgb = np.ascontiguousarray(bgr[:, :, ::-1])
        img = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
    return QPixmap.fromImage(img.copy())


class ImageView(QLabel):
    """Shows a BGR frame scaled to fit; reports clicks/drags in *image* coordinates."""

    clicked = Signal(object)   # Point in image coords
    dragged = Signal(object)   # Point in image coords
    released = Signal(object)

    def __init__(self, placeholder: str = "No signal", parent: QWidget | None = None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("background: #0b0d10; border: 1px solid #2a2f38; border-radius: 8px; color: #6b7280;")
        self.setText(placeholder)
        self._pix: Optional[QPixmap] = None
        self._img_size = (0, 0)
        self._dragging = False

    def set_frame(self, bgr: np.ndarray | None) -> None:
        if bgr is None:
            self._pix = None
            self.update()
            return
        self._img_size = (bgr.shape[1], bgr.shape[0])
        self._pix = ndarray_to_pixmap(bgr)
        self.update()

    def _draw_rect(self) -> tuple[int, int, int, int]:
        if self._pix is None:
            return (0, 0, 0, 0)
        w, h = self.width(), self.height()
        iw, ih = self._pix.width(), self._pix.height()
        scale = min(w / iw, h / ih)
        dw, dh = int(iw * scale), int(ih * scale)
        return ((w - dw) // 2, (h - dh) // 2, dw, dh)

    def paintEvent(self, event) -> None:  # noqa: N802
        if self._pix is None:
            super().paintEvent(event)
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        x, y, w, h = self._draw_rect()
        p.drawPixmap(x, y, w, h, self._pix)
        p.end()

    def _to_image(self, pos) -> Optional[Point]:
        if self._pix is None:
            return None
        x, y, w, h = self._draw_rect()
        if w == 0 or h == 0:
            return None
        ix = (pos.x() - x) / w * self._img_size[0]
        iy = (pos.y() - y) / h * self._img_size[1]
        if 0 <= ix <= self._img_size[0] and 0 <= iy <= self._img_size[1]:
            return Point(x=ix, y=iy)
        return None

    def mousePressEvent(self, e) -> None:  # noqa: N802
        p = self._to_image(e.position())
        if p is not None:
            self._dragging = True
            self.clicked.emit(p)

    def mouseMoveEvent(self, e) -> None:  # noqa: N802
        if self._dragging:
            p = self._to_image(e.position())
            if p is not None:
                self.dragged.emit(p)

    def mouseReleaseEvent(self, e) -> None:  # noqa: N802
        if self._dragging:
            self._dragging = False
            p = self._to_image(e.position())
            self.released.emit(p)


class LogPanel(QPlainTextEdit, logging.Handler):
    """A logging.Handler that appends records to a read-only text box (thread-safe via signal)."""

    class _Bridge(QObject):
        line = Signal(str)

    def __init__(self, parent: QWidget | None = None, max_lines: int = 400):
        QPlainTextEdit.__init__(self, parent)
        logging.Handler.__init__(self)
        self.setReadOnly(True)
        self.setMaximumBlockCount(max_lines)
        self.setStyleSheet("font-family: Menlo, Consolas, monospace; font-size: 11px;")
        self._bridge = self._Bridge()
        self._bridge.line.connect(self.appendPlainText)
        self.setFormatter(logging.Formatter("%(asctime)s %(levelname)-5s %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:  # logging.Handler API
        try:
            self._bridge.line.emit(self.format(record))
        except RuntimeError:
            pass

    def close(self) -> None:  # called by logging.shutdown at exit; the C++ widget may be gone already
        try:
            logging.Handler.close(self)
        except RuntimeError:
            pass


class WorkerSignals(QObject):
    finished = Signal(object)
    error = Signal(str)
    progress = Signal(str, float)


class Worker(QRunnable):
    """Run a blocking function on the global thread pool and deliver the result via signals."""

    def __init__(self, fn: Callable, *args, **kwargs):
        super().__init__()
        self.fn, self.args, self.kwargs = fn, args, kwargs
        self.signals = WorkerSignals()
        self.setAutoDelete(True)

    @Slot()
    def run(self) -> None:
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:  # noqa: BLE001
            logging.getLogger("bridge.ui").exception("worker failed")
            self.signals.error.emit(str(e))
            return
        self.signals.finished.emit(result)


def run_in_background(fn: Callable, on_done: Callable[[object], None], on_error: Callable[[str], None] | None = None,
                      progress: Callable[[str, float], None] | None = None, *args, **kwargs) -> Worker:
    w = Worker(fn, *args, **kwargs)
    w.signals.finished.connect(on_done)
    if on_error:
        w.signals.error.connect(on_error)
    if progress:
        w.signals.progress.connect(progress)
    QThreadPool.globalInstance().start(w)
    return w
