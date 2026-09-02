"""Borderless fullscreen projection window. Shows only the rendered canvas."""
from __future__ import annotations

import logging
from typing import Callable, Optional

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QGuiApplication, QImage, QPainter
from PySide6.QtWidgets import QWidget

from bridge.projector.manager import DisplayDevice
from bridge.render.renderer import ProjectionRenderer

log = logging.getLogger("bridge.projector")


def ndarray_to_qimage(bgr: np.ndarray) -> QImage:
    h, w = bgr.shape[:2]
    rgb = np.ascontiguousarray(bgr[:, :, ::-1])
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


class ProjectionWindow(QWidget):
    """Displays frames from a ProjectionRenderer (or any override image) fullscreen."""

    def __init__(self, renderer: ProjectionRenderer, fps: int = 60):
        super().__init__(None, Qt.WindowType.Window | Qt.WindowType.FramelessWindowHint)
        self.setWindowTitle("BRIDGE Projection")
        self.setCursor(Qt.CursorShape.BlankCursor)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
        self.renderer = renderer
        self._override: Optional[np.ndarray] = None
        self._image: Optional[QImage] = None
        self.display: Optional[DisplayDevice] = None
        self._timer = QTimer(self)
        self._timer.setInterval(max(1, int(1000 / max(1, fps))))
        self._timer.timeout.connect(self._tick)
        self.frame_listeners: list[Callable[[np.ndarray], None]] = []

    # -- lifecycle ------------------------------------------------------------------------
    def show_on(self, display: DisplayDevice) -> None:
        self.display = display
        self.renderer.set_size(display.width, display.height)
        screens = QGuiApplication.screens()
        target = None
        for s in screens:
            if s.name() == display.name or f":{s.name()}" in display.id:
                target = s
                break
        if target is not None:
            self.setScreen(target)
            g = target.geometry()
            self.setGeometry(g)
        else:
            self.setGeometry(display.x, display.y, display.width, display.height)
        self.showFullScreen()
        self._timer.start()
        log.info("Projection window opened on %s", display.name)

    def show_windowed(self, width: int, height: int) -> None:
        """Simulation/debug helper: same content in a normal window."""
        self.renderer.set_size(width, height)
        self.setWindowFlags(Qt.WindowType.Window)
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self.resize(width, height)
        self.show()
        self._timer.start()

    def close_projection(self) -> None:
        self._timer.stop()
        self.close()

    # -- content -----------------------------------------------------------------------
    def set_override(self, image: Optional[np.ndarray]) -> None:
        """Temporarily show a fixed image (e.g. calibration pattern) instead of the scene."""
        self._override = image
        self._tick()
        # Force a synchronous repaint so a worker thread waiting on us sees the change on screen.
        self.repaint()

    @Slot(object)
    def set_override_slot(self, image: object) -> None:
        self.set_override(image)  # type: ignore[arg-type]

    def current_frame(self) -> np.ndarray:
        return self._override if self._override is not None else self.renderer.render()

    def _tick(self) -> None:
        frame = self.current_frame()
        self._image = ndarray_to_qimage(frame)
        for fn in list(self.frame_listeners):
            try:
                fn(frame)
            except Exception:  # noqa: BLE001
                log.exception("projection frame listener failed")
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        if self._image is None:
            painter.fillRect(self.rect(), Qt.GlobalColor.white)
        else:
            painter.drawImage(self.rect(), self._image)
        painter.end()

    def keyPressEvent(self, event) -> None:  # noqa: N802
        if event.key() == Qt.Key.Key_Escape:
            self.close_projection()
