"""ProjectionRenderer: rasterizes a Scene into a BGR frame at projector resolution.

Independent of camera, AI and Qt. Uses OpenCV with anti-aliasing. The
ProjectionWindow blits the resulting image; the simulator shows it in a panel.
"""
from __future__ import annotations

import math
import time
from typing import Optional

import cv2
import numpy as np

from bridge.render import animation
from bridge.render.primitives import (
    Arrow, Circle, Dot, Label, Message, Outline, Path, Primitive, Style, TargetZone,
)
from bridge.render.scene import Scene
from bridge.spatial.geometry import Point

BACKGROUNDS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "transparent": (40, 40, 40),  # simulation only: dark grey stands in for "off"
}


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


class ProjectionRenderer:
    def __init__(self, width: int, height: int, background: str = "white",
                 custom_rgb: tuple[int, int, int] = (255, 255, 255), scale: float = 1.0):
        self.width = int(width)
        self.height = int(height)
        self.background = background
        self.custom_rgb = custom_rgb
        self.scale = scale  # global scale for thickness/fonts when projector is tiny/huge
        self.scene = Scene()
        self._t0 = time.perf_counter()
        self.frames_rendered = 0
        self.last_frame: Optional[np.ndarray] = None
        self.last_frame_version = -1

    # ---- public API mirroring the spec -------------------------------------------------
    def circle(self, center: Point, radius: float, style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Circle(center=center, radius=radius, style=style or Style(), group=group))

    def outline(self, points: list[Point], style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Outline(points=points, style=style or Style(), group=group))

    def arrow(self, start: Point, end: Point, style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Arrow(start=start, end=end, style=style or Style(), group=group))

    def point(self, center: Point, style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Dot(center=center, style=style or Style(fill=True), group=group))

    def label(self, position: Point, text: str, style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Label(position=position, text=text, style=style or Style(), group=group))

    def target_zone(self, polygon: list[Point], style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(TargetZone(points=polygon, style=style or Style(color=(40, 200, 90), dashed=True), group=group))

    def path(self, points: list[Point], style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Path(points=points, style=style or Style(animation="flow"), group=group))

    def message(self, text: str, position: Point | None = None, style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Message(text=text, position=position, style=style or Style(color=(30, 30, 30)), group=group))

    def clear(self) -> None:
        self.scene.clear()

    def set_size(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)

    # ---- rasterization ---------------------------------------------------------------
    def background_bgr(self) -> tuple[int, int, int]:
        if self.background == "custom":
            return _bgr(self.custom_rgb)
        return _bgr(BACKGROUNDS.get(self.background, (255, 255, 255)))

    def render(self, t: Optional[float] = None) -> np.ndarray:
        t = (time.perf_counter() - self._t0) if t is None else t
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        img[:] = self.background_bgr()
        for prim in self.scene.items():
            self._draw(img, prim, t)
        self.frames_rendered += 1
        self.last_frame = img
        self.last_frame_version = self.scene.version
        return img

    def render_mask(self, t: Optional[float] = None, dilate_px: int = 0) -> np.ndarray:
        """Binary mask (uint8 0/255) of pixels currently covered by graphics.

        Used to suppress BRIDGE's own projected graphics in the camera image so
        the tracker does not lock onto the light it is projecting.
        """
        frame = self.last_frame if (t is None and self.last_frame is not None and self.last_frame_version == self.scene.version
                                    and self.last_frame.shape[:2] == (self.height, self.width)) else self.render(t)
        bg = np.empty_like(frame)
        bg[:] = self.background_bgr()
        diff = cv2.cvtColor(cv2.absdiff(frame, bg), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask

    def _thick(self, style: Style) -> int:
        return max(1, int(round(style.thickness * self.scale)))

    def _draw(self, img: np.ndarray, prim: Primitive, t: float) -> None:
        st = prim.style
        if st.animation == "blink" and not animation.blink_visible(t):
            return
        layer = img if st.opacity >= 0.999 else img.copy()
        color = _bgr(st.color)
        th = self._thick(st)
        if isinstance(prim, Circle):
            r = prim.radius * (animation.pulse_scale(t) if st.animation == "pulse" else 1.0)
            c = prim.center.as_int()
            if st.fill:
                cv2.circle(layer, c, int(r), color, -1, cv2.LINE_AA)
            elif st.dashed:
                self._dashed_circle(layer, c, r, color, th)
            else:
                cv2.circle(layer, c, max(1, int(r)), color, th, cv2.LINE_AA)
        elif isinstance(prim, Dot):
            r = prim.radius * (animation.pulse_scale(t, amplitude=0.3) if st.animation == "pulse" else 1.0)
            cv2.circle(layer, prim.center.as_int(), max(1, int(r)), color, -1, cv2.LINE_AA)
        elif isinstance(prim, Arrow):
            self._arrow(layer, prim, color, th, t)
        elif isinstance(prim, Outline):
            self._polyline(layer, prim.points, color, th, closed=True, dashed=st.dashed)
        elif isinstance(prim, TargetZone):
            pts = np.array([p.as_int() for p in prim.points], np.int32)
            if st.fill:
                cv2.fillPoly(layer, [pts], color, cv2.LINE_AA)
            self._polyline(layer, prim.points, color, th, closed=True, dashed=True,
                           phase=animation.flow_offset(t) if st.animation in ("flow", "pulse") else 0.0)
        elif isinstance(prim, Path):
            phase = animation.flow_offset(t) if st.animation == "flow" else 0.0
            self._polyline(layer, prim.points, color, th, closed=False, dashed=True, phase=phase)
            if len(prim.points) >= 2:
                self._arrow_head(layer, prim.points[-2], prim.points[-1], color, th)
        elif isinstance(prim, Label):
            self._label(layer, prim, color)
        elif isinstance(prim, Message):
            self._message(layer, prim, color)
        if layer is not img:
            cv2.addWeighted(layer, st.opacity, img, 1 - st.opacity, 0, dst=img)

    def _arrow(self, img: np.ndarray, a: Arrow, color, th: int, t: float) -> None:
        s, e = a.start, a.end
        if a.style.animation == "pulse":
            k = 0.06 * math.sin(2 * math.pi * t / 1.0)
            d = e - s
            s = Point(x=s.x + d.x * k, y=s.y + d.y * k)
        cv2.line(img, s.as_int(), e.as_int(), color, th, cv2.LINE_AA)
        self._arrow_head(img, s, e, color, th)

    def _arrow_head(self, img, s: Point, e: Point, color, th: int) -> None:
        ang = math.atan2(e.y - s.y, e.x - s.x)
        size = max(12.0, 5.0 * th) * self.scale
        for da in (math.pi * 5 / 6, -math.pi * 5 / 6):
            p = Point(x=e.x + size * math.cos(ang + da), y=e.y + size * math.sin(ang + da))
            cv2.line(img, e.as_int(), p.as_int(), color, th, cv2.LINE_AA)

    def _polyline(self, img, points: list[Point], color, th: int, closed: bool, dashed: bool, phase: float = 0.0) -> None:
        if len(points) < 2:
            return
        pts = list(points) + ([points[0]] if closed else [])
        if not dashed:
            arr = np.array([p.as_int() for p in pts], np.int32)
            cv2.polylines(img, [arr], False, color, th, cv2.LINE_AA)
            return
        dash, gap = 30.0 * self.scale, 18.0 * self.scale
        period = dash + gap
        carry = -phase
        for a, b in zip(pts[:-1], pts[1:]):
            seg = a.distance_to(b)
            if seg == 0:
                continue
            ux, uy = (b.x - a.x) / seg, (b.y - a.y) / seg
            pos = carry
            while pos < seg:
                d0 = max(pos, 0.0)
                d1 = min(pos + dash, seg)
                if d1 > d0:
                    p0 = (int(a.x + ux * d0), int(a.y + uy * d0))
                    p1 = (int(a.x + ux * d1), int(a.y + uy * d1))
                    cv2.line(img, p0, p1, color, th, cv2.LINE_AA)
                pos += period
            carry = pos - seg

    def _dashed_circle(self, img, c: tuple[int, int], r: float, color, th: int) -> None:
        n = max(8, int(2 * math.pi * r / 40))
        for i in range(n):
            a0 = 360.0 * i / n
            cv2.ellipse(img, c, (int(r), int(r)), 0, a0, a0 + 360.0 / n * 0.6, color, th, cv2.LINE_AA)

    def _label(self, img, lab: Label, color) -> None:
        fs = lab.font_scale * self.scale
        th = max(1, int(2 * self.scale))
        (tw, tht), base = cv2.getTextSize(lab.text, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        x, y = lab.position.as_int()
        x = int(min(max(x, 4), self.width - tw - 4))
        y = int(min(max(y, tht + 4), self.height - 4))
        if lab.background:
            cv2.rectangle(img, (x - 6, y - tht - 6), (x + tw + 6, y + base + 4), color, -1, cv2.LINE_AA)
            txt_color = (255, 255, 255)
        else:
            txt_color = color
        cv2.putText(img, lab.text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, txt_color, th, cv2.LINE_AA)

    def _message(self, img, m: Message, color) -> None:
        fs = 1.2 * self.scale
        th = max(1, int(2 * self.scale))
        lines = m.text.split("\n")
        sizes = [cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for ln in lines]
        line_h = max(s[1] for s in sizes) + int(16 * self.scale)
        total_h = line_h * len(lines)
        if m.position is None:
            cx, cy = self.width // 2, self.height // 2 - total_h // 2
        else:
            cx, cy = m.position.as_int()
        for i, (ln, (tw, _)) in enumerate(zip(lines, sizes)):
            x = int(cx - tw / 2) if m.position is None else cx
            y = cy + i * line_h + line_h
            cv2.putText(img, ln, (x, y), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th, cv2.LINE_AA)


def calibration_pattern(width: int, height: int, points: list[Point], radius: int = 28,
                        background: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """High-contrast marker pattern: white filled discs on a dark background."""
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = background
    for p in points:
        cv2.circle(img, p.as_int(), radius, (255, 255, 255), -1, cv2.LINE_AA)
    return img


def solid(width: int, height: int, bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.empty((height, width, 3), dtype=np.uint8)
    img[:] = bgr
    return img
