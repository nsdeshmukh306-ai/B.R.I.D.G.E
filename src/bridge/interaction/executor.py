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

import cv2
import numpy as np

from bridge.interaction.commands import (
    ClearProjection, Command, DrawPath, HighlightObject, LabelObject, PointToObject, ShowMessage, ShowTargetZone, TargetSpec,
)
from bridge.interaction.resolver import Resolution, TargetResolver
from bridge.render.motion import TargetMotion
from bridge.render.primitives import ALERT, SUCCESS, TEXT, WARNING, Style
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
    motions: dict[str, TargetMotion] = field(default_factory=dict)   # object id -> smoothed motion
    visuals: dict[str, dict[str, str]] = field(default_factory=dict)  # object id -> role -> primitive id
    labels: dict[str, str] = field(default_factory=dict)             # object id -> tracked label


class CommandExecutor:
    def __init__(self, renderer: ProjectionRenderer, resolver: TargetResolver | None = None,
                 tracking_backend: str = "csrt", lost_after_frames: int = 15,
                 accent_rgb: tuple[int, int, int] = (0, 220, 200), line_width: int = 3, target_style: str = "pulse",
                 on_status: Callable[[str], None] | None = None, process_width: int = 640,
                 smoothing: bool = True, lead_ms: int = 60):
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
        # Latency: detection and tracking run on a downscaled copy of the camera frame.
        self.process_width = int(process_width)
        self.smoothing = smoothing
        self.lead_s = max(0.0, lead_ms / 1000.0)
        self.proc_scale = 1.0            # processing_px = camera_px * proc_scale
        self.last_update_ms = 0.0        # per-frame processing cost, for diagnostics
        self.renderer.pre_render_hooks.append(self._animate)

    # -- frame preparation ------------------------------------------------------------------
    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        """Downscale to the processing width and record the scale factor."""
        h, w = frame.shape[:2]
        if self.process_width and w > self.process_width:
            self.proc_scale = self.process_width / w
            return cv2.resize(frame, (self.process_width, max(1, int(round(h * self.proc_scale)))),
                              interpolation=cv2.INTER_AREA)
        self.proc_scale = 1.0
        return frame

    def clean_frame(self, frame: np.ndarray, scale: float | None = None) -> np.ndarray:
        """Camera frame with BRIDGE's own projected graphics suppressed.

        `scale` is the processing scale of `frame` (processing_px = camera_px * scale);
        the projector mask is warped straight into that space.
        """
        if not self.suppress_self_projection or self.mapper is None or len(self.renderer.scene) == 0:
            return frame
        scale = self.proc_scale if scale is None else scale
        h, w = frame.shape[:2]
        version = self.renderer.scene.version
        cached_version, cached = self._mask_cache
        if cached is None or cached_version != version or cached.shape[:2] != (h, w):
            mask_proj = self.renderer.render_mask(dilate_px=self.line_width)
            # H maps camera px -> projector px; this frame is camera px * scale.
            h_proc = self.mapper.H @ np.diag([1.0 / scale, 1.0 / scale, 1.0])
            cached = projector_mask_to_camera(mask_proj, h_proc, (w, h), dilate_px=max(3, int(self.line_width * scale)))
            self._mask_cache = (version, cached)
        return suppress(frame, cached)

    @staticmethod
    def _scale_state(st: TrackingState, k: float) -> TrackingState:
        if k == 1.0:
            return st
        b = st.bbox
        return st.model_copy(update={
            "center": Point(x=st.center.x * k, y=st.center.y * k),
            "bbox": BoundingBox(x=b.x * k, y=b.y * k, w=b.w * k, h=b.h * k),
        })

    @staticmethod
    def _scale_bbox(b: BoundingBox, k: float) -> BoundingBox:
        return b if k == 1.0 else BoundingBox(x=b.x * k, y=b.y * k, w=b.w * k, h=b.h * k)

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
            proc = self.clean_frame(self.prepare_frame(frame))
            k = self.proc_scale
            spec_proc = spec.model_copy(update={"boxes_camera": [self._scale_bbox(b, k) for b in spec.boxes_camera]})
            tracked_proc = [self._scale_state(t, k) for t in (tracked or self.last_states)]
            res = self.resolver.resolve(spec_proc, proc, tracked_proc)
            if not res.found:
                self.clear_targets()
                self._show_message(f"Cannot find {spec.label or 'target'}.\nPlease clarify.", 5)
                return ExecutionResult(False, f"Target '{spec.label}' not found", res.strategy)
            self._start_tracking(cmd, res, proc)
            states_proc = self.active.tracker.update(proc) if self.active else []
            states = [self._scale_state(st, 1.0 / k) for st in states_proc]
            self.last_states = states
            self._render_states(states)
            pts = [self.mapper.camera_to_projector(s.center) for s in states] if self.mapper else []
            for s in states:
                log.info("Target: %s location=(%.0f,%.0f) projection=%s", s.label, s.center.x, s.center.y, cmd.command)
            return ExecutionResult(True, f"{cmd.command}: {spec.label}", res.strategy, len(states), pts)

    def update(self, frame: np.ndarray) -> list[TrackingState]:
        """Called every camera frame: advance tracking and refresh the motion targets.

        The graphics themselves are redrawn by `_animate` at projector frame rate, so a slow
        camera makes the reticle less *accurate*, never less *smooth*.
        """
        t0 = time.perf_counter()
        with self._lock:
            if self._message_expiry and time.time() > self._message_expiry:
                self.renderer.scene.remove_group(GROUP_MESSAGE)
                self._message_expiry = None
            if self.active is None:
                return []
            proc = self.clean_frame(self.prepare_frame(frame))
            k = self.proc_scale
            states = [self._scale_state(st, 1.0 / k) for st in self.active.tracker.update(proc)]
            self.last_states = states
            self.last_update_ms = (time.perf_counter() - t0) * 1000
            live = [s for s in states if not s.is_lost]
            if not live and states:
                log.warning("Tracking lost: clearing stale projection")
                for m in self.active.motions.values():
                    m.release()
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

    # -- motion + drawing ------------------------------------------------------------------
    def _status_of(self, s: TrackingState) -> str:
        if s.is_lost:
            return "lost"
        if s.frames_since_seen > 2 or s.confidence < 0.5:
            return "searching"
        return "locked"

    def _colour_for(self, base: tuple[int, int, int], status: str) -> tuple[int, int, int]:
        """Colour carries tracking state: teal = locked, amber = re-acquiring, red = lost."""
        return {"locked": base, "searching": WARNING, "lost": ALERT, "caution": WARNING}.get(status, base)

    def _render_states(self, states: list[TrackingState]) -> None:
        """Feed new measurements into the motion models and (re)build primitives if the set changed."""
        if self.active is None or self.mapper is None:
            return
        a = self.active
        now = time.perf_counter()
        ids = []
        for s in states:
            ids.append(s.object_id)
            a.labels[s.object_id] = s.label
            status = self._status_of(s)
            m = a.motions.get(s.object_id)
            if m is None:
                a.motions[s.object_id] = TargetMotion(
                    x=s.center.x, y=s.center.y, w=s.bbox.w, h=s.bbox.h, t0=now,
                    lead_s=self.lead_s if self.smoothing else 0.0,
                    tau_pos=0.075 if self.smoothing else 0.0001, status=status, confidence=s.confidence)
            else:
                m.observe(s.center.x, s.center.y, s.bbox.w, s.bbox.h, now, status, s.confidence)
        for gone in [k for k in a.motions if k not in ids]:
            a.motions.pop(gone, None)
            a.visuals.pop(gone, None)
        if set(a.visuals) != set(ids):
            self.renderer.scene.remove_group(GROUP_TARGET)
            self.renderer.scene.remove_group(GROUP_ZONE)
            a.visuals = {oid: self._build_primitives(a, oid) for oid in ids}
        self._animate(now, force=True)

    def _build_primitives(self, a: ActiveTarget, oid: str) -> dict[str, str]:
        """Create the primitives for one target once; `_animate` moves them every frame."""
        st = a.style
        shape = getattr(getattr(a.command, "style", None), "shape", "circle")
        z = Point(x=0.0, y=0.0)
        roles: dict[str, str] = {}
        if a.kind == "highlight":
            if shape == "outline":
                roles["main"] = self.renderer.outline([z, Point(x=1, y=0), Point(x=1, y=1), Point(x=0, y=1)], st.model_copy(), GROUP_TARGET)
            elif shape == "point":
                roles["main"] = self.renderer.point(z, Style(color=st.color, fill=True, animation=st.animation), GROUP_TARGET)
            else:
                roles["main"] = self.renderer.circle(z, 10.0, st.model_copy(), GROUP_TARGET)
            if a.label_text:
                roles["label"] = self.renderer.label(z, a.label_text, Style(color=st.color), GROUP_TARGET)
        elif a.kind == "point":
            roles["arrow"] = self.renderer.arrow(z, Point(x=1, y=1), Style(color=st.color, thickness=self.line_width,
                                                                          animation="pulse" if st.animation == "pulse" else "none"), GROUP_TARGET)
            roles["dot"] = self.renderer.point(z, Style(color=st.color, fill=True), GROUP_TARGET)
        elif a.kind == "label":
            roles["dot"] = self.renderer.point(z, Style(color=st.color, fill=True), GROUP_TARGET)
            roles["label"] = self.renderer.label(z, a.label_text or a.labels.get(oid, ""), Style(color=st.color), GROUP_TARGET)
        elif a.kind == "zone":
            roles["zone"] = self.renderer.target_zone([z, Point(x=1, y=0), Point(x=1, y=1), Point(x=0, y=1)],
                                                      Style(color=SUCCESS, thickness=self.line_width, dashed=True, animation="flow"), GROUP_ZONE)
            roles["label"] = self.renderer.label(z, a.label_text or "Place here", Style(color=SUCCESS), GROUP_ZONE)
        return roles

    def _animate(self, t: float | None = None, force: bool = False) -> None:
        """Pre-render hook: place every target's primitives from its smoothed motion model.

        Runs at projector frame rate, so the graphic glides between camera measurements and is
        extrapolated forward by the pipeline latency instead of trailing the object.
        """
        a = self.active
        if a is None or self.mapper is None or not a.visuals:
            return
        if not self._lock.acquire(blocking=force):
            return  # a tracker update is mid-flight; skip this frame rather than stall the projector
        try:
            if self.active is not a or self.mapper is None:
                return
            now = time.perf_counter()
            m = self.mapper
            for oid, roles in a.visuals.items():
                motion = a.motions.get(oid)
                if motion is None:
                    continue
                cx, cy, bw, bh = motion.sample(now)
                cam_c = Point(x=cx, y=cy)
                bbox = BoundingBox(x=cx - bw / 2, y=cy - bh / 2, w=bw, h=bh)
                centre = m.camera_to_projector(cam_c)
                radius = max(18.0, m.camera_radius_to_projector(cam_c, bbox.radius) * 1.15)
                grow = motion.radius_scale(now)
                alpha = motion.opacity(now)
                colour = self._colour_for(a.style.color, motion.status)
                self._place(a, roles, centre, radius, grow, alpha, colour, bbox, m)
            self.renderer.scene.version += 1
        finally:
            self._lock.release()

    def _place(self, a: ActiveTarget, roles: dict[str, str], centre: Point, radius: float, grow: float,
               alpha: float, colour: tuple[int, int, int], bbox: BoundingBox, m: CoordinateMapper) -> None:
        scene = self.renderer.scene
        for role, pid in roles.items():
            prim = scene.get(pid)
            if prim is None:
                continue
            prim.style.color = colour
            prim.style.opacity = alpha
            if role == "main":
                if hasattr(prim, "radius"):
                    prim.center = centre
                    prim.radius = radius * grow
                elif hasattr(prim, "points"):
                    prim.points = m.camera_bbox_to_projector_polygon(bbox).points
                else:
                    prim.center = centre
            elif role == "dot":
                prim.center = centre
            elif role == "arrow":
                offset = Point(x=centre.x - radius * 2.2, y=centre.y - radius * 2.2)
                offset = Point(x=min(max(offset.x, 20), self.renderer.width - 20),
                               y=min(max(offset.y, 20), self.renderer.height - 20))
                d = centre - offset
                n = max(1e-6, (d.x ** 2 + d.y ** 2) ** 0.5)
                prim.start = offset
                prim.end = Point(x=centre.x - d.x / n * radius * grow, y=centre.y - d.y / n * radius * grow)
            elif role == "zone":
                prim.points = m.camera_bbox_to_projector_polygon(self._inflate(bbox, 1.3)).points
                prim.style.color = SUCCESS
            elif role == "label":
                if a.kind == "highlight":
                    prim.position = Point(x=centre.x - radius * 0.6, y=centre.y - radius * grow - 18)
                elif a.kind == "zone":
                    prim.position = Point(x=centre.x - radius * 0.5, y=centre.y)
                    prim.style.color = SUCCESS
                else:
                    prim.position = Point(x=centre.x + radius * 0.5, y=centre.y - radius * 0.5)

    @staticmethod
    def _inflate(b: BoundingBox, k: float) -> BoundingBox:
        cw, ch = b.w * k, b.h * k
        c = b.center
        return BoundingBox(x=c.x - cw / 2, y=c.y - ch / 2, w=cw, h=ch)

    def _execute_path(self, cmd: DrawPath, frame: np.ndarray, tracked: list[TrackingState] | None) -> ExecutionResult:
        proc = self.clean_frame(self.prepare_frame(frame))
        k = self.proc_scale
        src_spec = cmd.source.model_copy(update={"boxes_camera": [self._scale_bbox(b, k) for b in cmd.source.boxes_camera]})
        dst_spec = cmd.destination.model_copy(update={"boxes_camera": [self._scale_bbox(b, k) for b in cmd.destination.boxes_camera]})
        tracked_proc = [self._scale_state(t, k) for t in (tracked or [])]
        src = self.resolver.resolve(src_spec, proc, tracked_proc)
        dst = self.resolver.resolve(dst_spec, proc, tracked_proc)
        if not src.found or not dst.found:
            missing = cmd.source.label if not src.found else cmd.destination.label
            self._show_message(f"Cannot find {missing}.\nPlease clarify.", 5)
            return ExecutionResult(False, f"Path endpoint '{missing}' not found", "path")
        assert self.mapper is not None
        self.clear_targets()
        inv = 1.0 / k
        src_c = Point(x=src.detections[0].center.x * inv, y=src.detections[0].center.y * inv)
        dst_box = self._scale_bbox(dst.detections[0].bbox, inv)
        a = self.mapper.camera_to_projector(src_c)
        b = self.mapper.camera_to_projector(dst_box.center)
        mid = Point(x=(a.x + b.x) / 2, y=(a.y + b.y) / 2 - abs(b.x - a.x) * 0.15)
        self.renderer.path([a, mid, b], Style(color=self.accent, thickness=self.line_width, animation="flow"), GROUP_TARGET)
        self.renderer.circle(a, 30, Style(color=self.accent, thickness=self.line_width), GROUP_TARGET)
        zone = self.mapper.camera_bbox_to_projector_polygon(self._inflate(dst_box, 1.3)).points
        self.renderer.target_zone(zone, Style(color=SUCCESS, thickness=self.line_width, dashed=True), GROUP_ZONE)
        log.info("Projection action: draw_path %s -> %s", cmd.source.label, cmd.destination.label)
        return ExecutionResult(True, f"Path {cmd.source.label} -> {cmd.destination.label}", "path", 2, [a, b])

    def _show_message(self, text: str, duration_s: float) -> None:
        self.renderer.scene.remove_group(GROUP_MESSAGE)
        self.renderer.message(text, style=Style(color=TEXT if self.renderer.dark else (30, 30, 30), glow=False), group=GROUP_MESSAGE)
        self._message_expiry = time.time() + duration_s if duration_s > 0 else None
        self.on_status(text.replace("\n", " "))
