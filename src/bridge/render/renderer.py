"""ProjectionRenderer: rasterizes a Scene into a BGR frame at projector resolution.

Independent of camera, AI and Qt. Uses OpenCV with anti-aliasing. The
ProjectionWindow blits the resulting image; the simulator shows it in a panel.

Visual language (designed for a BLACK canvas, where the projector emits no light
except for the graphics): thin crisp strokes over a soft additive glow, reticle
rings and corner brackets for targets, pill-shaped labels, tapered arrows.
"""
from __future__ import annotations

import logging
import math
import time
from typing import Optional

import cv2
import numpy as np

from bridge.render import animation
from bridge.render.primitives import (
    ACCENT, SUCCESS, TEXT, Arrow, Circle, Dot, Label, Message, Outline, Path, Primitive, Style, TargetZone,
)
from bridge.render.scene import Scene
from bridge.spatial.geometry import Point

BACKGROUNDS = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "transparent": (0, 0, 0),  # a projector shows nothing for black: closest thing to transparent
}
FONT = cv2.FONT_HERSHEY_DUPLEX
log = logging.getLogger("bridge.render")


def _bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return (int(rgb[2]), int(rgb[1]), int(rgb[0]))


def _scale_color(bgr: tuple[int, int, int], k: float) -> tuple[int, int, int]:
    return tuple(int(min(255, v * k)) for v in bgr)


def rounded_rect_points(x: float, y: float, w: float, h: float, r: float, n: int = 6) -> np.ndarray:
    """Polygon approximating a rounded rectangle (int32, for polylines/fillPoly)."""
    r = max(0.0, min(r, w / 2, h / 2))
    pts = []
    corners = [(x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90), (x + r, y + r, 180)]
    for cx, cy, a0 in corners:
        for i in range(n + 1):
            a = math.radians(a0 + 90 * i / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.int32)


class ProjectionRenderer:
    def __init__(self, width: int, height: int, background: str = "black",
                 custom_rgb: tuple[int, int, int] = (0, 0, 0), scale: float = 1.0):
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
        # Called with the render timestamp just before each frame is rasterized, so motion
        # can be evaluated at display rate (60 fps) rather than at camera rate.
        self.pre_render_hooks: list = []

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
        return self.scene.add(TargetZone(points=polygon, style=style or Style(color=SUCCESS, dashed=True), group=group))

    def path(self, points: list[Point], style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Path(points=points, style=style or Style(animation="flow"), group=group))

    def message(self, text: str, position: Point | None = None, style: Style | None = None, group: str | None = None) -> str:
        return self.scene.add(Message(text=text, position=position, style=style or Style(color=TEXT, glow=False), group=group))

    def clear(self) -> None:
        self.scene.clear()

    def set_size(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)
        # Keep strokes visually similar across projector resolutions (reference: 1080p).
        self.scale = max(0.6, min(3.0, min(self.width, self.height) / 1080.0))

    # ---- rasterization ---------------------------------------------------------------
    def background_bgr(self) -> tuple[int, int, int]:
        if self.background == "custom":
            return _bgr(self.custom_rgb)
        return _bgr(BACKGROUNDS.get(self.background, (0, 0, 0)))

    def _blank(self) -> np.ndarray:
        """Background-filled frame. (Tuple broadcast into a 4K array is ~50x slower than fill.)"""
        bgr = self.background_bgr()
        img = np.empty((self.height, self.width, 3), dtype=np.uint8)
        if bgr[0] == bgr[1] == bgr[2]:
            img.fill(bgr[0])
        else:
            img[:] = np.array(bgr, dtype=np.uint8)
        return img

    @property
    def dark(self) -> bool:
        return sum(self.background_bgr()) < 3 * 128

    def render(self, t: Optional[float] = None) -> np.ndarray:
        t = (time.perf_counter() - self._t0) if t is None else t
        for hook in list(self.pre_render_hooks):
            try:
                hook(t)
            except Exception:  # noqa: BLE001 - a bad animator must never stop the projector
                log.exception("pre-render hook failed")
        img = self._blank()
        items = self.scene.items()
        if items and self.dark:
            self._render_glow(img, items, t)
        for prim in items:
            self._draw(img, prim, t)
        self.frames_rendered += 1
        self.last_frame = img
        self.last_frame_version = self.scene.version
        return img

    def _render_glow(self, img: np.ndarray, items: list[Primitive], t: float) -> None:
        """Additive halo: draw glowing primitives thick on a black layer, blur, add. Limited to
        the bounding region of the graphics so 4K projectors stay cheap."""
        glow_items = [p for p in items if p.style.glow and not isinstance(p, (Message, Label))]
        if not glow_items:
            return
        x0, y0, x1, y1 = self._extent(glow_items)
        pad = int(40 * self.scale)
        x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
        x1, y1 = min(self.width, x1 + pad), min(self.height, y1 + pad)
        if x1 <= x0 or y1 <= y0:
            return
        layer = np.zeros((y1 - y0, x1 - x0, 3), np.uint8)
        offset = Point(x=-x0, y=-y0)
        for p in glow_items:
            self._draw(layer, self._shift(p, offset), t, glow_pass=True)
        k = int(max(9, 10 * self.scale)) | 1
        layer = cv2.GaussianBlur(layer, (k, k), 0)
        roi = img[y0:y1, x0:x1]
        cv2.add(roi, layer, dst=roi)

    @staticmethod
    def _shift(p: Primitive, d: Point) -> Primitive:
        if isinstance(p, (Circle, Dot)):
            return p.model_copy(update={"center": p.center + d})
        if isinstance(p, Arrow):
            return p.model_copy(update={"start": p.start + d, "end": p.end + d})
        if isinstance(p, (Outline, TargetZone, Path)):
            return p.model_copy(update={"points": [q + d for q in p.points]})
        if isinstance(p, Label):
            return p.model_copy(update={"position": p.position + d})
        return p

    def _extent(self, items: list[Primitive]) -> tuple[int, int, int, int]:
        xs, ys = [], []
        for p in items:
            if isinstance(p, (Circle, Dot)):
                r = p.radius * 1.3
                xs += [p.center.x - r, p.center.x + r]
                ys += [p.center.y - r, p.center.y + r]
            elif isinstance(p, Arrow):
                xs += [p.start.x, p.end.x]
                ys += [p.start.y, p.end.y]
            elif isinstance(p, (Outline, TargetZone, Path)):
                xs += [q.x for q in p.points]
                ys += [q.y for q in p.points]
        if not xs:
            return (0, 0, 0, 0)
        return (int(min(xs)), int(min(ys)), int(max(xs)) + 1, int(max(ys)) + 1)

    def render_mask(self, t: Optional[float] = None, dilate_px: int = 0) -> np.ndarray:
        """Binary mask (uint8 0/255) of pixels currently covered by graphics.

        Used to suppress BRIDGE's own projected graphics in the camera image so
        the tracker does not lock onto the light it is projecting.
        """
        frame = self.last_frame if (t is None and self.last_frame is not None and self.last_frame_version == self.scene.version
                                    and self.last_frame.shape[:2] == (self.height, self.width)) else self.render(t)
        bgr = self.background_bgr()
        if bgr == (0, 0, 0):
            diff = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            diff = cv2.cvtColor(cv2.absdiff(frame, self._blank()), cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(diff, 8, 255, cv2.THRESH_BINARY)
        if dilate_px > 0:
            k = 2 * dilate_px + 1
            mask = cv2.dilate(mask, np.ones((k, k), np.uint8))
        return mask

    # ---- drawing ------------------------------------------------------------------------
    def _thick(self, style: Style, glow_pass: bool = False) -> int:
        t = max(1, int(round(style.thickness * self.scale)))
        return t * 3 if glow_pass else t

    def _color(self, style: Style, glow_pass: bool) -> tuple[int, int, int]:
        c = _bgr(style.color)
        return _scale_color(c, 0.45) if glow_pass else c

    def _draw(self, img: np.ndarray, prim: Primitive, t: float, glow_pass: bool = False) -> None:
        st = prim.style
        if st.animation == "blink" and not animation.blink_visible(t):
            return
        layer = img if (st.opacity >= 0.999 or glow_pass) else img.copy()
        color = self._color(st, glow_pass)
        th = self._thick(st, glow_pass)
        if isinstance(prim, Circle):
            self._circle(layer, prim, color, th, t, glow_pass)
        elif isinstance(prim, Dot):
            r = prim.radius * self.scale * (animation.pulse_scale(t, amplitude=0.3) if st.animation == "pulse" else 1.0)
            cv2.circle(layer, prim.center.as_int(), max(1, int(r + (th if glow_pass else 0))), color, -1, cv2.LINE_AA)
            if not glow_pass and self.dark:
                cv2.circle(layer, prim.center.as_int(), max(1, int(r * 0.45)), (255, 255, 255), -1, cv2.LINE_AA)
        elif isinstance(prim, Arrow):
            self._arrow(layer, prim, color, th, t)
        elif isinstance(prim, Outline):
            if st.variant in ("auto", "bracket"):
                self._brackets(layer, prim.points, color, th)
            else:
                self._polyline(layer, prim.points, color, th, closed=True, dashed=st.dashed)
        elif isinstance(prim, TargetZone):
            self._zone(layer, prim, color, th, t, glow_pass)
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

    def _circle(self, img, c: Circle, color, th: int, t: float, glow_pass: bool) -> None:
        st = c.style
        r = c.radius * (animation.pulse_scale(t, amplitude=0.08) if st.animation == "pulse" else 1.0)
        centre = c.center.as_int()
        if st.fill:
            cv2.circle(img, centre, int(r), color, -1, cv2.LINE_AA)
            return
        if st.dashed:
            self._dashed_circle(img, centre, r, color, th)
            return
        cv2.circle(img, centre, max(1, int(r)), color, th, cv2.LINE_AA)
        if st.variant in ("auto", "reticle") and not glow_pass:
            # four short ticks outside the ring; they rotate slowly when pulsing
            rot = (t * 25.0) % 360 if st.animation == "pulse" else 0.0
            tick = max(8.0, r * 0.18)
            for k in range(4):
                a = math.radians(45 + 90 * k + rot)
                p0 = (int(centre[0] + (r + th * 1.5) * math.cos(a)), int(centre[1] + (r + th * 1.5) * math.sin(a)))
                p1 = (int(centre[0] + (r + th * 1.5 + tick) * math.cos(a)), int(centre[1] + (r + th * 1.5 + tick) * math.sin(a)))
                cv2.line(img, p0, p1, color, th, cv2.LINE_AA)

    def _brackets(self, img, points: list[Point], color, th: int) -> None:
        """Corner brackets around the polygon's bounding box: reads as 'selected' without boxing the object in."""
        if len(points) < 2:
            return
        xs, ys = [p.x for p in points], [p.y for p in points]
        x0, y0, x1, y1 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        L = int(max(12, min(x1 - x0, y1 - y0) * 0.22))
        for (cx, cy, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x1, y1, -1, -1), (x0, y1, 1, -1)):
            cv2.line(img, (cx, cy), (cx + sx * L, cy), color, th, cv2.LINE_AA)
            cv2.line(img, (cx, cy), (cx, cy + sy * L), color, th, cv2.LINE_AA)

    def _zone(self, img, z: TargetZone, color, th: int, t: float, glow_pass: bool) -> None:
        pts = np.array([p.as_int() for p in z.points], np.int32)
        if not glow_pass:
            overlay = img.copy()
            cv2.fillPoly(overlay, [pts], color, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.12 if self.dark else 0.08, img, 0.88 if self.dark else 0.92, 0, dst=img)
        phase = animation.flow_offset(t) if z.style.animation in ("flow", "pulse") else 0.0
        self._polyline(img, z.points, color, th, closed=True, dashed=True, phase=phase)

    def _arrow(self, img: np.ndarray, a: Arrow, color, th: int, t: float) -> None:
        s, e = a.start, a.end
        if a.style.animation == "pulse":
            k = 0.05 * math.sin(2 * math.pi * t / 1.0)
            d = e - s
            s = Point(x=s.x + d.x * k, y=s.y + d.y * k)
        # tapered shaft: a thin quad widening toward the tail
        ang = math.atan2(e.y - s.y, e.x - s.x)
        nx, ny = -math.sin(ang), math.cos(ang)
        head = max(14.0, 6.0 * th)
        tip = Point(x=e.x - head * 0.8 * math.cos(ang), y=e.y - head * 0.8 * math.sin(ang))
        w_tail, w_tip = th * 1.6, th * 0.5
        quad = np.array([
            (s.x + nx * w_tail, s.y + ny * w_tail), (tip.x + nx * w_tip, tip.y + ny * w_tip),
            (tip.x - nx * w_tip, tip.y - ny * w_tip), (s.x - nx * w_tail, s.y - ny * w_tail),
        ], np.int32)
        cv2.fillPoly(img, [quad], color, cv2.LINE_AA)
        self._arrow_head(img, s, e, color, th, filled=True)

    def _arrow_head(self, img, s: Point, e: Point, color, th: int, filled: bool = True) -> None:
        ang = math.atan2(e.y - s.y, e.x - s.x)
        size = max(14.0, 6.0 * th)
        p1 = (e.x + size * math.cos(ang + math.pi * 5 / 6), e.y + size * math.sin(ang + math.pi * 5 / 6))
        p2 = (e.x + size * math.cos(ang - math.pi * 5 / 6), e.y + size * math.sin(ang - math.pi * 5 / 6))
        tri = np.array([e.as_int(), (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1]))], np.int32)
        if filled:
            cv2.fillPoly(img, [tri], color, cv2.LINE_AA)
        else:
            cv2.polylines(img, [tri], True, color, th, cv2.LINE_AA)

    def _polyline(self, img, points: list[Point], color, th: int, closed: bool, dashed: bool, phase: float = 0.0) -> None:
        if len(points) < 2:
            return
        pts = list(points) + ([points[0]] if closed else [])
        if not dashed:
            arr = np.array([p.as_int() for p in pts], np.int32)
            cv2.polylines(img, [arr], False, color, th, cv2.LINE_AA)
            return
        dash, gap = 26.0 * self.scale, 14.0 * self.scale
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
        fs = 0.62 * lab.font_scale * self.scale
        th = max(1, int(round(1.2 * self.scale)))
        (tw, tht), base = cv2.getTextSize(lab.text, FONT, fs, th)
        pad_x, pad_y = int(12 * self.scale), int(8 * self.scale)
        x, y = lab.position.as_int()
        x = int(min(max(x, pad_x), self.width - tw - pad_x - 2))
        y = int(min(max(y, tht + pad_y + 2), self.height - base - pad_y - 2))
        if lab.background:
            pill = rounded_rect_points(x - pad_x, y - tht - pad_y, tw + 2 * pad_x, tht + base + 2 * pad_y, (tht + 2 * pad_y) / 2)
            cv2.fillPoly(img, [pill], color, cv2.LINE_AA)
            txt_color = (10, 12, 14)  # dark text on the accent pill
        else:
            txt_color = color
        cv2.putText(img, lab.text, (x, y), FONT, fs, txt_color, th, cv2.LINE_AA)

    def _message(self, img, m: Message, color) -> None:
        fs = 0.95 * self.scale
        th = max(1, int(round(1.5 * self.scale)))
        lines = m.text.split("\n")
        sizes = [cv2.getTextSize(ln, FONT, fs, th)[0] for ln in lines]
        line_h = max(s[1] for s in sizes) + int(18 * self.scale)
        total_h = line_h * len(lines)
        if m.position is None:
            cx, cy = self.width // 2, self.height // 2 - total_h // 2
        else:
            cx, cy = m.position.as_int()
        for i, (ln, (tw, _)) in enumerate(zip(lines, sizes)):
            x = int(cx - tw / 2) if m.position is None else cx
            y = cy + i * line_h + line_h
            cv2.putText(img, ln, (x, y), FONT, fs, color, th, cv2.LINE_AA)
        if m.position is None and lines:
            # thin accent rule under the message
            w0 = max(s[0] for s in sizes)
            y = cy + total_h + int(10 * self.scale)
            cv2.line(img, (cx - w0 // 2, y), (cx + w0 // 2, y), _bgr(ACCENT), max(1, th), cv2.LINE_AA)


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
