"""Display enumeration. A projector is just another OS display."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("bridge.projector")


@dataclass
class DisplayDevice:
    id: str
    name: str
    width: int
    height: int
    x: int
    y: int
    is_primary: bool
    scale: float = 1.0

    @property
    def label(self) -> str:
        role = "primary" if self.is_primary else "secondary"
        return f"{self.name} ({self.width}x{self.height}, {role})"


def _enumerate_qt() -> list[DisplayDevice]:
    from PySide6.QtGui import QGuiApplication

    app = QGuiApplication.instance()
    if app is None:
        return []
    out: list[DisplayDevice] = []
    primary = QGuiApplication.primaryScreen()
    for i, s in enumerate(QGuiApplication.screens()):
        g = s.geometry()
        dpr = float(s.devicePixelRatio())
        out.append(DisplayDevice(
            id=f"display:{i}:{s.name()}", name=s.name() or f"Display {i + 1}",
            width=int(g.width() * dpr), height=int(g.height() * dpr), x=g.x(), y=g.y(),
            is_primary=(s is primary), scale=dpr,
        ))
    return out


def _enumerate_screeninfo() -> list[DisplayDevice]:
    try:
        from screeninfo import get_monitors
    except Exception:  # noqa: BLE001
        return []
    out: list[DisplayDevice] = []
    try:
        for i, m in enumerate(get_monitors()):
            out.append(DisplayDevice(id=f"display:{i}:{m.name}", name=m.name or f"Display {i + 1}",
                                     width=m.width, height=m.height, x=m.x, y=m.y, is_primary=bool(m.is_primary)))
    except Exception:  # noqa: BLE001
        return []
    return out


def enumerate_displays() -> list[DisplayDevice]:
    displays = _enumerate_qt() or _enumerate_screeninfo()
    log.info("Enumerated %d display(s)", len(displays))
    return displays


class DisplayManager:
    def __init__(self) -> None:
        self.available: list[DisplayDevice] = []
        self.selected: Optional[DisplayDevice] = None

    def refresh(self) -> list[DisplayDevice]:
        try:
            self.available = enumerate_displays()
        except Exception:  # noqa: BLE001
            log.exception("Display enumeration failed")
            self.available = []
        return self.available

    def select(self, display_id: str) -> Optional[DisplayDevice]:
        for d in self.available:
            if d.id == display_id:
                self.selected = d
                log.info("Display connected name=%s res=%dx%d", d.name, d.width, d.height)
                return d
        return None

    def suggest_projector(self) -> Optional[DisplayDevice]:
        """Prefer a secondary display (that is where a projector usually lands)."""
        for d in self.available:
            if not d.is_primary:
                return d
        return self.available[0] if self.available else None
