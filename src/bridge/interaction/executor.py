"""CommandExecutor: command -> resolve target -> track -> map -> render.

This is the only place camera coordinates become projector coordinates for
user-facing graphics. It refuses to project spatial content without a valid
calibration.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np

from bridge.interaction.commands import (
    ClearProjection, Command, DrawPath, HighlightObject, LabelObject, PointToObject, ShowMessage, ShowTargetZone, TargetSpec,
)
from bridge.interaction.resolver import Resolution, TargetResolver
from bridge.render.primitives import SUCCESS, TEXT, Style
from bridge.render.renderer import ProjectionRenderer
from bridge.spatial.geometry import BoundingBox, Point
from bridge.spatial.homography import CoordinateMapper
from bridge.vision.detection import Detection
from bridge.vision.suppression import projector_mask_to_camera, suppress
from bridge.vision.tracking import MultiTracker, TrackingState

log = logging.getLogger("bridge.executor")

GROUP_TARGET = "target"
GROUP_MESSAGE = "message"
GROUP_ZONE = "zone"


@dataclass
class ExecutionResult:
    ok: bool
    message: str = ""
    strategy: str = ""
    n_targets: int = 0
    projector_points: list[Point] = field(default_factory=list)


@dataclass
class ActiveTarget:
    command: Command  # type: ignore[valid-type]
    tracker: MultiTracker
    style: Style
    label_text: str
    kind: str  # highlight | point | label | zone | path
    destination_camera: Optional[BoundingBox] = None


class CommandExecutor:
    def __init__(self, renderer: ProjectionRenderer, resolver: TargetResolver | None = None,
                 tracking_backend: str = "csrt", lost_after_frames: int = 15,
                 accent_rgb: tuple[int, int, int] = (0, 150, 255), line_width: int = 4, target_style: str = "pulse",
                 on_status: Callable[[str], None] | None = None):
        self.renderer = renderer
        self.resolver = resolver or TargetResolver()
        self.mapper: Optional[CoordinateMapper] = None
        self.tracking_backend = tracking_backend
        self.lost_after = lost_after_frames
        self.accent = accent_rgb
        self.line_width = line_width
        self.default_animation = "pulse" if target_style == "pulse" else "none"
        self.on_status = on_status or (lambda s: None)
        self.active: Optional[ActiveTarget] = None
        self._lock = threading.RLock()
        self._message_expiry: Optional[float] = None
        self.last_states: list[TrackingState] = []
        self.on_target_lost: Callable[[], None] | None = None
        self.suppress_self_projection = True
        self._mask_cache: tuple[int, Optional[np.ndarray]] = (-1, None)

    def clean_frame(self, frame: np.ndarray) -> np.ndarray:
        """Camera frame with BRIDGE's own projected graphics suppressed."""
        if not self.suppress_self_projection or self.mapper is None or len(self.renderer.scene) == 0:
            return frame
        h, w = frame.shape[:2]
        version = self.renderer.scene.version
        cached_version, cached = self._mask_cache
        if cached is None or cached_version != version or cached.shape[:2] != (h, w):
            mask_proj = self.renderer.render_mask(dilate_px=self.line_width)
            cached = projector_mask_to_camera(mask_proj, self.mapper.H, (w, h), dilate_px=max(4, self.line_width))
            self._mask_cache = (version, cached)
        return suppress(frame, cached)

    # -- calibration -----------------------------------------------------------------------
    def set_mapper(self, mapper: Optional[CoordinateMapper], keep_scene: bool = False) -> None:
        with self._lock:
            self.mapper = mapper
            self._mask_cache = (-1, None)
            if mapper is None:
                self.clear()
            elif keep_scene and self.active is not None:
                self._render_states([s for s in self.last_states if not s.is_lost])

    @property
    def calibrated(self) -> bool:
        return self.mapper is not None

    # -- public -----------------------------------------------------------------------------
    def execute(self, cmd: Command, frame: np.ndarray, tracked: list[TrackingState] | None = None) -> ExecutionResult:  # type: ignore[valid-type]
        with self._lock:
            if isinstance(cmd, ClearProjection):
                self.clear()
                return ExecutionResult(True, "Projection cleared", "clear")
            if isinstance(cmd, ShowMessage):
                self._show_message(cmd.text, cmd.duration_s)
                return ExecutionResult(True, cmd.text, "message")
            if not self.calibrated:
                self._show_message("Spatial calibration required.", 6)
                log.warning("Refusing spatial command without calibration")
                return ExecutionResult(False, "Spatial calibration required.", "uncalibrated")
            if isinstance(cmd, DrawPath):
                return self._execute_path(cmd, frame, tracked)
            spec: TargetSpec = cmd.target
            frame = self.clean_frame(frame)
            res = self.resolver.resolve(spec, frame, tracked or self.last_states)
            if not res.found:
                self.clear_targets()
                self._show_message(f"Cannot find {spec.label or 'target'}.\nPlease clarify.", 5)
                return ExecutionResult(False, f"Target '{spec.label}' not found", res.strategy)
            self._start_tracking(cmd, res, frame)
            states = self.active.tracker.update(frame) if self.active else []
            self._render_states(states)
            pts = [self.mapper.camera_to_projector(s.center) for s in states] if self.mapper else []
            for s in states:
                log.info("Target: %s location=(%.0f,%.0f) projection=%s", s.label, s.center.x, s.center.y, cmd.command)
            return ExecutionResult(True, f"{cmd.command}: {spec.label}", res.strategy, len(states), pts)

    def update(self, frame: np.ndarray) -> list[TrackingState]:
        """Called every camera frame: advance tracking and refresh projected graphics."""
        with self._lock:
            if self._message_expiry and time.time() > self._message_expiry:
                self.renderer.scene.remove_group(GROUP_MESSAGE)
                self._message_expiry = None
            if self.active is None:
                return []
            states = self.active.tracker.update(self.clean_frame(frame))
            self.last_states = states
            live = [s for s in states if not s.is_lost]
            if not live and states:
                log.warning("Tracking lost: clearing stale projection")
                self.renderer.scene.remove_group(GROUP_TARGET)
                self.renderer.scene.remove_group(GROUP_ZONE)
                self._show_message("Target lost.", 4)
                self.active = None
                self.on_status("Target lost.")
                if self.on_target_lost:
                    self.on_target_lost()
                return states
            self._render_states(live)
            return states

    def clear_targets(self) -> None:
        with self._lock:
            if self.active:
                self.active.tracker.stop_all()
            self.active = None
            self.last_states = []
            self.renderer.scene.remove_group(GROUP_TARGET)
            self.renderer.scene.remove_group(GROUP_ZONE)

    def clear(self) -> None:
        with self._lock:
            self.clear_targets()
            self.renderer.clear()
            self._message_expiry = None

    # -- internals -------------------------------------------------------------------------
    def _style(self, cmd: Command, dashed: bool = False, fill: bool = False) -> Style:  # type: ignore[valid-type]
        cs = getattr(cmd, "style", None)
        color = (cs.color if cs and cs.color else self.accent)
        anim = cs.animation if cs else self.default_animation
        if anim == "pulse" and self.default_animation == "none":
            anim = "none"
        return Style(color=color, thickness=self.line_width, animation=anim, dashed=dashed, fill=fill)

    def _start_tracking(self, cmd: Command, res: Resolution, frame: np.ndarray) -> None:  # type: ignore[valid-type]
        self.clear_targets()
        self.renderer.scene.remove_group(GROUP_MESSAGE)
        tracker = MultiTracker(self.tracking_backend, self.lost_after)
        tracker.start(res.detections, frame, color_hint=res.color_hint)
        kind = {"highlight_object": "highlight", "point_to_object": "point", "label_object": "label",
                "show_target_zone": "zone"}.get(cmd.command, "highlight")
        label_text = getattr(cmd, "label_text", "") or getattr(cmd, "text", "")
        self.active = ActiveTarget(cmd, tracker, self._style(cmd), label_text, kind)

    def _render_states(self, states: list[TrackingState]) -> None:
        if self.active is None or self.mapper is None:
            return
        scene = self.renderer.scene
        scene.remove_group(GROUP_TARGET)
        scene.remove_group(GROUP_ZONE)
        a = self.active
        for s in states:
            self._draw_target(a, s)
        self.renderer.scene.version += 1

    def _draw_target(self, a: ActiveTarget, s: TrackingState) -> None:
        m = self.mapper
        assert m is not None
        center = m.camera_to_projector(s.center)
        radius = max(18.0, m.camera_radius_to_projector(s.center, s.bbox.radius) * 1.15)
        st = a.style
        shape = getattr(getattr(a.command, "style", None), "shape", "circle")
        if a.kind == "highlight":
            if shape == "outline":
                self.renderer.outline(m.camera_bbox_to_projector_polygon(s.bbox).points, st, GROUP_TARGET)
            elif shape == "point":
                self.renderer.point(center, Style(color=st.color, fill=True, animation=st.animation), GROUP_TARGET)
            else:
                self.renderer.circle(center, radius, st, GROUP_TARGET)
            if a.label_text:
                self.renderer.label(Point(x=center.x - radius * 0.6, y=center.y - radius - 18), a.label_text,
                                    Style(color=st.color), GROUP_TARGET)
        elif a.kind == "point":
            # Arrow from an offset above/left of the object down to its edge.
            offset = Point(x=center.x - radius * 2.2, y=center.y - radius * 2.2)
            offset = Point(x=min(max(offset.x, 20), self.renderer.width - 20), y=min(max(offset.y, 20), self.renderer.height - 20))
            d = center - offset
            n = max(1e-6, (d.x ** 2 + d.y ** 2) ** 0.5)
            end = Point(x=center.x - d.x / n * radius, y=center.y - d.y / n * radius)
            self.renderer.arrow(offset, end, Style(color=st.color, thickness=self.line_width, animation="pulse" if st.animation == "pulse" else "none"), GROUP_TARGET)
            self.renderer.point(center, Style(color=st.color, fill=True), GROUP_TARGET)
        elif a.kind == "label":
            self.renderer.point(center, Style(color=st.color, fill=True), GROUP_TARGET)
            self.renderer.label(Point(x=center.x + radius * 0.5, y=center.y - radius * 0.5), a.label_text or s.label,
                                Style(color=st.color), GROUP_TARGET)
        elif a.kind == "zone":
            poly = m.camera_bbox_to_projector_polygon(self._inflate(s.bbox, 1.3)).points
            self.renderer.target_zone(poly, Style(color=SUCCESS, thickness=self.line_width, dashed=True, animation="flow"), GROUP_ZONE)
            self.renderer.label(Point(x=center.x - radius * 0.5, y=center.y), a.label_text or "Place here", Style(color=SUCCESS), GROUP_ZONE)

    @staticmethod
    def _inflate(b: BoundingBox, k: float) -> BoundingBox:
        cw, ch = b.w * k, b.h * k
        c = b.center
        return BoundingBox(x=c.x - cw / 2, y=c.y - ch / 2, w=cw, h=ch)

    def _execute_path(self, cmd: DrawPath, frame: np.ndarray, tracked: list[TrackingState] | None) -> ExecutionResult:
        frame = self.clean_frame(frame)
        src = self.resolver.resolve(cmd.source, frame, tracked)
        dst = self.resolver.resolve(cmd.destination, frame, tracked)
        if not src.found or not dst.found:
            missing = cmd.source.label if not src.found else cmd.destination.label
            self._show_message(f"Cannot find {missing}.\nPlease clarify.", 5)
            return ExecutionResult(False, f"Path endpoint '{missing}' not found", "path")
        assert self.mapper is not None
        self.clear_targets()
        a = self.mapper.camera_to_projector(src.detections[0].center)
        b = self.mapper.camera_to_projector(dst.detections[0].center)
        mid = Point(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2 - abs(b.x - a.x) * 0.15)
        self.renderer.path([a, mid, b], Style(color=self.accent, thickness=self.line_width, animation="flow"), GROUP_TARGET)
        self.renderer.circle(a, 30, Style(color=self.accent, thickness=self.line_width), GROUP_TARGET)
        zone = self.mapper.camera_bbox_to_projector_polygon(self._inflate(dst.detections[0].bbox, 1.3)).points
        self.renderer.target_zone(zone, Style(color=SUCCESS, thickness=self.line_width, dashed=True), GROUP_ZONE)
        log.info("Projection action: draw_path %s -> %s", cmd.source.label, cmd.destination.label)
        return ExecutionResult(True, f"Path {cmd.source.label} -> {cmd.destination.label}", "path", 2, [a, b])

    def _show_message(self, text: str, duration_s: float) -> None:
        self.renderer.scene.remove_group(GROUP_MESSAGE)
        self.renderer.message(text, style=Style(color=TEXT if self.renderer.dark else (30, 30, 30), glow=False), group=GROUP_MESSAGE)
        self._message_expiry = time.time() + duration_s if duration_s > 0 else None
        self.on_status(text.replace("\n", " "))
