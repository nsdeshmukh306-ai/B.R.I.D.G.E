"""BridgeCore: the headless application. The Qt UI is a thin shell around this.

Wires: camera -> (calibration | tracking) -> mapper -> renderer -> projector,
and user query -> AI provider -> command planner -> executor.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bridge.ai.base import AIError, AIProvider
from bridge.ai.mock import build_provider
from bridge.ai.schemas import ActionPlan, TargetIdentification
from bridge.app.config import ConfigManager
from bridge.app.events import EventBus, Topic
from bridge.app.state import CalibrationStatus, StateManager
from bridge.camera.device import CameraInfo
from bridge.camera.manager import CameraManager
from bridge.interaction.commands import Command, CommandPlanner
from bridge.interaction.executor import CommandExecutor, ExecutionResult
from bridge.interaction.resolver import TargetResolver
from bridge.projector.manager import DisplayDevice, DisplayManager
from bridge.render.renderer import ProjectionRenderer
from bridge.spatial.calibration import CalibrationEngine, CalibrationResult, ProjectorLink, build_method
from bridge.spatial.homography import CoordinateMapper
from bridge.spatial.profile import ProfileStore

log = logging.getLogger("bridge.core")


class _CameraLink:
    def __init__(self, cam: CameraManager):
        self.cam = cam

    def capture(self, settle_s: float = 0.25) -> Optional[np.ndarray]:
        seq = self.cam.frame_seq
        if settle_s > 0:
            time.sleep(settle_s)  # let the projector/camera exposure settle
        # Only accept a frame captured after the projected image changed.
        return self.cam.wait_for_fresh_frame(seq, skip=2)

    def size(self) -> tuple[int, int]:
        return self.cam.resolution


class BridgeCore:
    def __init__(self, config: ConfigManager, bus: EventBus | None = None):
        self.config = config
        self.settings = config.settings
        self.bus = bus or EventBus()
        self.state = StateManager(mode="physical")  # becomes "simulation" only via enter_simulation()
        self.cameras = CameraManager(self.bus)
        self.displays = DisplayManager()
        self.store = ProfileStore(Path(self.settings.profiles_dir))
        c = self.settings.calibration
        self.calibration = CalibrationEngine(
            build_method(c.method, marker_radius=c.marker_radius_px, margin_fraction=c.margin_fraction,
                         validation_threshold_px=c.validation_threshold_px, min_confidence=c.detection_min_confidence,
                         max_retries=c.max_retries),
            self.store,
        )
        r = self.settings.render
        self.renderer = ProjectionRenderer(1280, 720, r.background, r.custom_background_rgb)
        self.executor = CommandExecutor(self.renderer, TargetResolver(), self.settings.tracking.backend,
                                        self.settings.tracking.lost_after_frames, r.accent_rgb, r.line_width, r.target_style,
                                        on_status=lambda s: self.bus.publish(Topic.STATUS_MESSAGE, text=s, level="info"))
        self.executor.on_target_lost = lambda: self.bus.publish(Topic.TRACKING_LOST)
        self.planner = CommandPlanner(self.settings.ai.min_confidence)
        self.ai: Optional[AIProvider] = None
        self.projector_link: Optional[ProjectorLink] = None
        self.projector_display: Optional[DisplayDevice] = None
        self.simulation = None  # (world, projector, camera) when in simulation mode
        self._ai_lock = threading.Lock()
        self._calibrating = threading.Event()
        self.last_identification: Optional[TargetIdentification] = None
        self.last_command: Optional[Command] = None  # type: ignore[valid-type]
        self._frame_unsub: Optional[Callable[[], None]] = None
        self._frame_unsub = self.cameras.add_frame_listener(self._on_frame)
        self._frame_count = 0
        self.bus.subscribe(Topic.CAMERA_CONNECTED, self._on_camera_connected)

    # ---- hardware -------------------------------------------------------------------------
    def refresh_devices(self) -> tuple[list[CameraInfo], list[DisplayDevice]]:
        cams = self.cameras.refresh() if self.state.mode == "physical" else []
        disps = self.displays.refresh()
        return cams, disps

    def select_camera(self, info: CameraInfo) -> bool:
        cs = self.settings.camera
        ok = self.cameras.open(info, cs.width, cs.height, cs.fps)
        if ok:
            self.state.diagnostics.camera_connected = True
            self.state.diagnostics.camera_name = info.name
            w, h = self.cameras.resolution
            self.state.diagnostics.camera_resolution = f"{w}x{h}"
            self._invalidate_if_hardware_changed()
        else:
            self.state.diagnostics.camera_connected = False
        return ok

    def select_display(self, display: DisplayDevice) -> None:
        old = self.projector_display
        self.projector_display = display
        self.displays.selected = display
        self.state.diagnostics.display_name = display.name
        self.state.diagnostics.display_resolution = f"{display.width}x{display.height}"
        self.renderer.set_size(display.width, display.height)
        if old is not None and (old.id != display.id or (old.width, old.height) != (display.width, display.height)):
            self.invalidate_calibration("Projector/display changed. Calibration invalidated."
                                        if old.id != display.id else "Display resolution changed. Calibration invalidated.")
        self.bus.publish(Topic.DISPLAY_CONNECTED, display_id=display.id)
        log.info("Display connected name=%s res=%dx%d", display.name, display.width, display.height)

    def attach_projector(self, link: ProjectorLink) -> None:
        """Attach whatever shows images on the projector (ProjectionWindow adapter or simulator)."""
        self.projector_link = link
        w, h = link.size()
        self.renderer.set_size(w, h)

    def _on_camera_connected(self, ev) -> None:
        self.state.diagnostics.camera_connected = True
        self.state.diagnostics.camera_resolution = f"{ev.payload.get('width')}x{ev.payload.get('height')}"

    # ---- simulation ----------------------------------------------------------------------
    def enter_simulation(self, seed: int = 0) -> None:
        from bridge.simulation.world import make_default_simulation

        self.state.mode = "simulation"
        world, projector, camera = make_default_simulation(seed)
        self.simulation = (world, projector, camera)
        self.cameras.use_source(camera, "sim:camera")
        self.state.diagnostics.camera_name = "Simulated camera"
        self.state.diagnostics.camera_connected = True
        w, h = camera.resolution
        self.state.diagnostics.camera_resolution = f"{w}x{h}"
        disp = DisplayDevice(id="sim:projector", name="Simulated projector", width=projector.width, height=projector.height,
                             x=0, y=0, is_primary=False)
        self.select_display(disp)
        self.attach_projector(projector)
        self.renderer.background = "white" if self.settings.render.background == "transparent" else self.settings.render.background
        self.set_ai_provider("mock")
        self.invalidate_calibration("Entered simulation mode")
        log.info("Simulation mode started")

    def leave_simulation(self) -> None:
        self.cameras.close()
        self.simulation = None
        self.state.mode = "physical"
        self.projector_link = None
        self.invalidate_calibration("Left simulation mode")

    # ---- AI ----------------------------------------------------------------------------
    def set_ai_provider(self, kind: str | None = None) -> tuple[bool, str]:
        kind = kind or self.settings.ai.provider
        if self.state.mode == "physical" and kind == "mock":
            log.warning("Mock AI provider requested in physical mode; allowed only for testing")
        try:
            sim = (self.simulation[0], self.simulation[2]) if (self.simulation and kind == "mock") else None
            self.ai = build_provider(kind, self.config.secrets.gemini_api_key, self.settings.ai.model,
                                     self.settings.ai.timeout_s, simulation=sim)
        except (AIError, ValueError) as e:
            self.ai = None
            self.state.diagnostics.ai_provider = kind
            self.state.diagnostics.ai_status = f"Unavailable: {e}"
            log.error("AI provider unavailable: %s", e)
            return False, str(e)
        self.state.diagnostics.ai_provider = self.ai.status.provider
        self.state.diagnostics.ai_status = "Ready"
        return True, "ok"

    def _sync_ai_status(self) -> None:
        if self.ai is None:
            return
        st = self.ai.status
        self.state.diagnostics.ai_status = st.text
        self.state.diagnostics.ai_last_request_ts = st.last_request_ts
        self.state.diagnostics.ai_last_latency_s = st.last_latency_s

    def ask(self, query: str) -> ExecutionResult:
        """Natural-language entry point. Blocking (call from a worker thread in the UI)."""
        query = query.strip()
        if not query:
            return ExecutionResult(False, "Empty command")
        if self.ai is None:
            ok, msg = self.set_ai_provider()
            if not ok:
                return ExecutionResult(False, f"AI unavailable: {msg}")
        # A *fresh* frame, so it reflects what the projector is showing right now.
        frame = self.cameras.wait_for_fresh_frame(self.cameras.frame_seq, skip=1, timeout_s=2.0)
        if frame is None:
            return ExecutionResult(False, "No camera frame available")
        assert self.ai is not None
        with self._ai_lock:
            self.bus.publish(Topic.AI_REQUEST, query=query)
            known = [s.label for s in self.executor.last_states if not s.is_lost]
            try:
                ident = self.ai.identify_target(frame, query, known or None)
            except AIError as e:
                self._sync_ai_status()
                self.bus.publish(Topic.AI_ERROR, error=str(e))
                self.executor.execute(self.planner.plan(TargetIdentification(intent="show_message", message="AI request failed."), 1, 1), frame)
                return ExecutionResult(False, f"AI error: {e}")
            self._sync_ai_status()
        self.last_identification = ident
        log.info("AI command: %s target=%s confidence=%.2f boxes=%d", ident.intent, ident.target, ident.confidence, len(ident.boxes))
        self.bus.publish(Topic.AI_RESPONSE, identification=ident)
        h, w = frame.shape[:2]
        cmd = self.planner.plan(ident, w, h)
        return self.execute(cmd, frame)

    def execute(self, cmd: Command, frame: np.ndarray | None = None) -> ExecutionResult:  # type: ignore[valid-type]
        frame = frame if frame is not None else self.cameras.wait_for_frame(2.0)
        if frame is None:
            return ExecutionResult(False, "No camera frame available")
        self.last_command = cmd
        res = self.executor.execute(cmd, frame)
        self.bus.publish(Topic.COMMAND_EXECUTED, command=cmd.command, result=res)
        self._update_tracking_diag()
        return res

    def next_step(self, task_context: str = "Assembly: pick up the tools in the right order.") -> ExecutionResult:
        """Assembly assistance: ask the AI what to do next and project it."""
        if self.ai is None and not self.set_ai_provider()[0]:
            return ExecutionResult(False, "AI unavailable")
        frame = self.cameras.wait_for_frame(2.0)
        if frame is None:
            return ExecutionResult(False, "No camera frame available")
        assert self.ai is not None
        try:
            plan: ActionPlan = self.ai.plan_action(frame, task_context)
        except AIError as e:
            return ExecutionResult(False, f"AI error: {e}")
        finally:
            self._sync_ai_status()
        if plan.done:
            return self.execute(self.planner.plan(TargetIdentification(intent="show_message", message=plan.instruction), 1, 1), frame)
        ident = TargetIdentification(intent="next_step", target=plan.target, visual_description=plan.visual_description,
                                     confidence=plan.confidence, boxes=plan.boxes, message=plan.instruction)
        h, w = frame.shape[:2]
        return self.execute(self.planner.plan(ident, w, h), frame)

    # ---- calibration ------------------------------------------------------------------------
    def can_calibrate(self) -> tuple[bool, str]:
        if not self.cameras.is_open:
            return False, "Connect a camera first."
        if self.projector_link is None:
            return False, "Open the projection window first."
        return True, "ok"

    def calibrate(self, progress: Callable[[str, float], None] | None = None) -> CalibrationResult:
        ok, why = self.can_calibrate()
        if not ok:
            return CalibrationResult(success=False, message=why)
        if self._calibrating.is_set():
            return CalibrationResult(success=False, message="Calibration already running")
        self._calibrating.set()
        try:
            self.executor.set_mapper(None)
            self.state.set_calibration(CalibrationStatus.CALIBRATING)
            self.bus.publish(Topic.CALIBRATION_STARTED)
            assert self.projector_link is not None
            res = self.calibration.run(self.projector_link, _CameraLink(self.cameras), progress)
            self.cameras.wait_for_fresh_frame(self.cameras.frame_seq, skip=2, timeout_s=1.0)  # white canvas visible again
            if res.success and res.homography is not None:
                self.executor.set_mapper(CoordinateMapper(res.homography))
                self.state.set_calibration(CalibrationStatus.VALID, res.validation.mean_error_px if res.validation else None)
                self.bus.publish(Topic.CALIBRATION_COMPLETE, result=res)
            else:
                self.state.set_calibration(CalibrationStatus.INVALID)
                self.bus.publish(Topic.CALIBRATION_FAILED, result=res)
            return res
        finally:
            self._calibrating.clear()

    def save_profile(self, name: str = "default") -> bool:
        if self.projector_display is None or self.cameras.active_id is None:
            return False
        prof = self.calibration.save(name, self.cameras.active_id, self.cameras.resolution, self.projector_display.id,
                                     (self.projector_display.width, self.projector_display.height), self.state.surface_type,
                                     self.state.mode)
        return prof is not None

    def try_load_profile(self) -> bool:
        if self.projector_display is None or self.cameras.active_id is None:
            return False
        prof = self.calibration.load_matching(self.cameras.active_id, self.cameras.resolution, self.projector_display.id,
                                              (self.projector_display.width, self.projector_display.height))
        if prof is None:
            self.state.set_calibration(CalibrationStatus.NOT_CALIBRATED)
            return False
        self.executor.set_mapper(CoordinateMapper(prof.homography))
        self.state.surface_type = prof.surface_type
        self.state.set_calibration(CalibrationStatus.VALID, prof.validation.mean_error_px)
        self.bus.publish(Topic.CALIBRATION_COMPLETE, result=self.calibration.result, loaded=True)
        return True

    def invalidate_calibration(self, reason: str) -> None:
        self.calibration.invalidate(reason)
        self.executor.set_mapper(None)
        self.state.set_calibration(CalibrationStatus.NOT_CALIBRATED)
        self.bus.publish(Topic.CALIBRATION_INVALIDATED, reason=reason)
        self.bus.publish(Topic.STATUS_MESSAGE, text=reason, level="warning")

    def _invalidate_if_hardware_changed(self) -> None:
        prof = self.calibration.profile
        if prof is None:
            if self.calibration.is_valid:
                self.invalidate_calibration("Camera changed. Calibration invalidated.")
            return
        ok, reason = prof.matches_hardware(self.cameras.active_id, self.cameras.resolution,
                                           self.projector_display.id if self.projector_display else None,
                                           (self.projector_display.width, self.projector_display.height) if self.projector_display else None)
        if not ok:
            self.invalidate_calibration(reason)

    # ---- frame loop ---------------------------------------------------------------------------
    def _on_frame(self, frame: np.ndarray) -> None:
        self._frame_count += 1
        self.state.diagnostics.camera_fps = self.cameras.fps_counter.fps
        if self._calibrating.is_set():
            return
        if self.simulation is not None and self.projector_link is not None:
            # In simulation nothing else pumps the renderer into the virtual projector
            # (the ProjectionWindow does this in physical mode).
            self.projector_link.show_image(self.renderer.render())
        if not self.settings.tracking.enabled:
            return
        states = self.executor.update(frame)
        if states:
            self.bus.publish(Topic.TRACKING_UPDATE, states=states)
        if self._frame_count % 5 == 0:
            self._update_tracking_diag()

    def _update_tracking_diag(self) -> None:
        live = [s for s in self.executor.last_states if not s.is_lost]
        d = self.state.diagnostics
        if live:
            d.tracking_target = ", ".join(sorted({s.label for s in live}))
            d.tracking_confidence = float(np.mean([s.confidence for s in live]))
        else:
            d.tracking_target = "--"
            d.tracking_confidence = None

    def shutdown(self) -> None:
        self.cameras.close()
        if self.ai:
            self.ai.close()
        log.info("BRIDGE core shut down")
