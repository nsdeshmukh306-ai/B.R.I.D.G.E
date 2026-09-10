"""Persistent overlays: the count board and the gaps on the tray.

These live in their own scene groups (`hud`, `gaps`, `alert`) so they coexist
with whatever the CommandExecutor is doing in `target` / `zone` / `message`.
Nothing here tracks or animates — overlays are rebuilt when the underlying
numbers change, which is a few times a minute, not sixty times a second.

The count board is deliberately drawn at the edge of the canvas: it belongs on
the drape or the table skirt, not on top of the instruments.
"""
from __future__ import annotations

import logging
from typing import Iterable, Literal, Optional, Sequence

from bridge.render.primitives import ACCENT, ALERT, SUCCESS, TEXT, WARNING, Style
from bridge.render.renderer import ProjectionRenderer
from bridge.spatial.geometry import Point
from bridge.spatial.homography import CoordinateMapper

log = logging.getLogger("bridge.render.overlay")

GROUP_HUD = "hud"
GROUP_GAPS = "gaps"
GROUP_ALERT = "alert"

Corner = Literal["top_left", "top_right", "bottom_left", "bottom_right"]

STATUS_COLOR = {"ok": SUCCESS, "short": ALERT, "over": WARNING, "pending": TEXT}


class OverlayLayer:
    """Draws non-tracking overlays into a renderer, in projector pixels."""

    def __init__(self, renderer: ProjectionRenderer, corner: Corner = "top_left",
                 line_height: int = 34, margin: int = 40, max_rows: int = 12):
        self.renderer = renderer
        self.corner = corner
        self.line_height = line_height
        self.margin = margin
        self.max_rows = max_rows
        self._hud_signature: tuple = ()

    # -- count board -------------------------------------------------------------------
    def show_board(self, title: str, rows: Sequence[tuple[str, str, int, int, str]],
                   footer: str = "", force: bool = False) -> bool:
        """Draw the count board. Returns True if the canvas actually changed.

        `rows` are (category, name, counted, baseline, status) — exactly what
        `CountSession.board()` produces.
        """
        signature = (title, footer, tuple(rows[: self.max_rows]))
        if signature == self._hud_signature and not force:
            return False
        self._hud_signature = signature
        self.renderer.scene.remove_group(GROUP_HUD)
        if not title and not rows:
            return True
        x, y, step = self._origin(len(rows[: self.max_rows]) + 2)
        self.renderer.label(Point(x=x, y=y), title, Style(color=ACCENT, glow=False), GROUP_HUD)
        y += step
        shown = 0
        for _cat, name, counted, baseline, status in rows:
            if shown >= self.max_rows:
                break
            colour = STATUS_COLOR.get(status, TEXT)
            text = f"{name}  {counted}/{baseline}" if baseline else f"{name}  {counted}"
            self.renderer.label(Point(x=x, y=y), text, Style(color=colour, glow=False), GROUP_HUD)
            y += step
            shown += 1
        if len(rows) > self.max_rows:
            self.renderer.label(Point(x=x, y=y), f"+{len(rows) - self.max_rows} more",
                                Style(color=TEXT, glow=False), GROUP_HUD)
            y += step
        if footer:
            self.renderer.label(Point(x=x, y=y), footer, Style(color=TEXT, glow=False), GROUP_HUD)
        return True

    def hide_board(self) -> None:
        self._hud_signature = ()
        self.renderer.scene.remove_group(GROUP_HUD)

    # -- gaps on the tray ----------------------------------------------------------------
    def show_gaps(self, polygons_camera: Iterable[Sequence[Point]], mapper: Optional[CoordinateMapper],
                  labels: Sequence[str] = (), colour=WARNING) -> int:
        """Outline where missing items should be. Input polygons are camera-space."""
        self.renderer.scene.remove_group(GROUP_GAPS)
        if mapper is None:
            return 0
        n = 0
        labels = list(labels)
        for i, poly in enumerate(polygons_camera):
            try:
                pts = [mapper.camera_to_projector(p) for p in poly]
            except Exception:  # noqa: BLE001 - a degenerate homography must not kill rendering
                log.exception("gap projection failed")
                continue
            if len(pts) < 3:
                continue
            self.renderer.target_zone(pts, Style(color=colour, thickness=3, dashed=True,
                                                 animation="flow", glow=True), GROUP_GAPS)
            if i < len(labels) and labels[i]:
                top = min(pts, key=lambda p: p.y)
                self.renderer.label(Point(x=top.x, y=max(top.y - 26, 20)), labels[i],
                                    Style(color=colour, glow=False), GROUP_GAPS)
            n += 1
        return n

    def hide_gaps(self) -> None:
        self.renderer.scene.remove_group(GROUP_GAPS)

    # -- banner alert ---------------------------------------------------------------------
    def show_alert(self, text: str, level: str = "caution") -> None:
        """A short banner across the canvas for a critical or caution alert."""
        self.renderer.scene.remove_group(GROUP_ALERT)
        colour = ALERT if level == "critical" else WARNING
        self.renderer.message(text, style=Style(color=colour, glow=False), group=GROUP_ALERT)

    def hide_alert(self) -> None:
        self.renderer.scene.remove_group(GROUP_ALERT)

    def clear(self) -> None:
        self.hide_board()
        self.hide_gaps()
        self.hide_alert()

    # -- internals -------------------------------------------------------------------------
    def _origin(self, rows: int) -> tuple[float, float, float]:
        w, h = self.renderer.width, self.renderer.height
        step = self.line_height
        block = rows * step
        x = self.margin if self.corner.endswith("left") else max(self.margin, w * 0.62)
        y = self.margin + step if self.corner.startswith("top") else max(self.margin, h - self.margin - block)
        return float(x), float(y), float(step)
