"""Reusable Qt widgets: image view, log panel, worker helpers, theme."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal, Slot
from PySide6.QtGui import QImage, QPainter, QPixmap
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QSizePolicy, QWidget

from bridge.spatial.geometry import Point

# --------------------------------------------------------------------------------------------
# Clinical theme.
#
# BRIDGE's control panel is read at a glance, often from a stride away and under bright
# theatre lighting, sometimes by someone whose hands are occupied and who cannot linger on a
# small control. That is a different design brief from a desktop dashboard, so the defaults
# below follow the conventions used across clinical and OR-adjacent software rather than
# generic app styling:
#
#   - Severity has exactly three colours, applied consistently everywhere in the app, matching
#     the red/amber/green convention IEC 60601-1-8 (medical electrical equipment alarm systems)
#     and most clinical monitoring UIs use for critical/caution/normal indication.
#   - Body text is 15px (~12pt at typical desktop DPI) minimum, per common medical-device UI
#     guidance that critical information stay legible from about a metre away.
#   - Interactive controls keep a minimum touch target close to 40px (~10mm) high, the spacing
#     guidance cited to avoid mis-hits under time pressure or with gloved/imprecise input.
#   - Every focusable control gets a visible focus ring; colour is never the only signal
#     (status labels pair colour with a text state, never colour alone).
# --------------------------------------------------------------------------------------------

BG = "#11141a"
PANEL_BG = "#181c24"
FIELD_BG = "#0d0f13"
BORDER = "#2e3542"
TEXT = "#eef1f6"
TEXT_MUTED = "#a7b0bf"
TEXT_DIM = "#7c8494"

TEAL = "#106563"
TEAL_LIGHT = "#1c8d8a"
GOLD = "#E29119"
NAVY = "#163C71"
MAROON = "#B32328"

# Severity palette -- IEC 60601-1-8-style red/amber/(green|cyan), applied identically to every
# status label, alert banner and table cell in the app. Never introduce a fourth colour for
# severity; add a new SEVERITY_* constant here instead so every surface picks it up at once.
SEVERITY_OK = "#3ddc97"        # normal / reconciled / connected
SEVERITY_CAUTION = "#f2a63d"   # needs attention, not yet urgent
SEVERITY_CRITICAL = "#ff5c5c"  # needs a response now
SEVERITY_NEUTRAL = "#a7b0bf"   # no status to report yet (pending / idle)

STYLESHEET = f"""
QMainWindow, QDialog, QWizard {{ background: {BG}; }}
QWidget {{ color: {TEXT}; font-size: 15px; background: {BG}; }}
QScrollArea, QScrollArea > QWidget > QWidget {{ background: {BG}; border: none; }}
QGroupBox {{
    background: {PANEL_BG}; border: 1px solid {BORDER}; border-radius: 10px;
    margin-top: 16px; padding: 12px 10px 10px 10px; font-weight: 700; font-size: 13px;
    color: {TEXT_MUTED}; letter-spacing: 0.5px;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QPushButton {{
    background: #262c38; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 10px 16px; min-height: 20px; font-size: 14px; font-weight: 600;
}}
QPushButton:hover {{ background: #2f3646; border-color: #454e60; }}
QPushButton:pressed {{ background: #20252f; }}
QPushButton:disabled {{ color: {TEXT_DIM}; background: #1c2028; border-color: #262b35; }}
QPushButton:focus {{ border: 2px solid {TEAL_LIGHT}; padding: 9px 15px; }}
QPushButton:checkable:checked {{ background: {TEAL}; border-color: {TEAL_LIGHT}; }}
QPushButton#primary {{ background: {TEAL}; border-color: {TEAL_LIGHT}; color: #ffffff; font-weight: 800; }}
QPushButton#primary:hover {{ background: #147a77; }}
QPushButton#accent {{ background: {NAVY}; border-color: #2a5aa8; color: #ffffff; font-weight: 800; }}
QPushButton#accent:hover {{ background: #1d4d8f; }}
QPushButton#danger {{ background: #3a1517; border-color: {MAROON}; color: #ffd7d7; font-weight: 800; }}
QPushButton#danger:hover {{ background: #4a1d20; }}
QComboBox, QLineEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background: {FIELD_BG}; border: 1px solid {BORDER}; border-radius: 8px;
    padding: 8px; min-height: 18px; font-size: 14px; selection-background-color: {TEAL};
}}
QComboBox:focus, QLineEdit:focus, QPlainTextEdit:focus {{ border: 2px solid {TEAL_LIGHT}; }}
QComboBox::drop-down {{ border: none; width: 24px; }}
QLabel#status_ok {{ color: {SEVERITY_OK}; font-weight: 700; }}
QLabel#status_warn {{ color: {SEVERITY_CAUTION}; font-weight: 700; }}
QLabel#status_err {{ color: {SEVERITY_CRITICAL}; font-weight: 700; }}
QLabel#title {{ font-size: 24px; font-weight: 800; color: #ffffff; letter-spacing: 0.5px; }}
QLabel#subtitle {{ color: {TEXT_MUTED}; font-size: 13px; }}
QLabel#mono {{ font-family: "Cascadia Code", Consolas, Menlo, monospace; font-size: 14px; color: #cfd7e3; }}
QLabel#section_head {{ font-size: 15px; font-weight: 700; color: {TEXT}; padding-top: 4px; }}
QRadioButton, QCheckBox {{ padding: 4px 2px; font-size: 14px; spacing: 8px; }}
QRadioButton::indicator, QCheckBox::indicator {{ width: 20px; height: 20px; }}
QTableWidget {{
    background: {FIELD_BG}; alternate-background-color: #12151b; gridline-color: {BORDER};
    border: 1px solid {BORDER}; border-radius: 8px; font-size: 14px;
}}
QHeaderView::section {{
    background: #1a1f28; color: {TEXT_MUTED}; padding: 8px; border: none;
    border-bottom: 1px solid {BORDER}; font-weight: 700; font-size: 12px;
}}
QTableWidget::item {{ padding: 6px; }}
QProgressBar {{ border: 1px solid {BORDER}; border-radius: 8px; background: {FIELD_BG}; text-align: center; min-height: 18px; }}
QProgressBar::chunk {{ background: {TEAL}; border-radius: 7px; }}
QScrollBar:vertical {{ background: transparent; width: 12px; }}
QScrollBar::handle:vertical {{ background: #3a4150; border-radius: 6px; min-height: 30px; }}
QScrollBar::handle:vertical:hover {{ background: #4a5364; }}
QToolTip {{ background: #232935; color: {TEXT}; border: 1px solid {BORDER}; padding: 6px; font-size: 13px; }}
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
        self.setStyleSheet(f"background: {FIELD_BG}; border: 1px solid {BORDER}; border-radius: 10px; "
                           f"color: {TEXT_DIM}; font-size: 14px;")
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
        self.setStyleSheet(f"background: {FIELD_BG}; border: 1px solid {BORDER}; border-radius: 8px; "
                           f"font-family: Consolas, Menlo, monospace; font-size: 13px; color: #b6bfcc;")
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
