"""Calibration engine with a pluggable CalibrationMethod interface.

V1 implements PlanarMarkerCalibration (4- or 9-point projected discs). The
interface leaves room for ChArUco, ArUco, Gray-code and non-planar methods.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Protocol

import numpy as np

from bridge.spatial.geometry import Point, points_to_array
from bridge.spatial.homography import HomographyError, Matrix3x3, compute_homography
from bridge.spatial.markers import DetectedMarker, detect_bright_markers, match_markers_to_layout
from bridge.spatial.profile import CalibrationProfile, ProfileStore
from bridge.spatial.validation import CalibrationValidator, ValidationResult

log = logging.getLogger("bridge.calibration")


class ProjectorLink(Protocol):
    """What the calibration method needs from the projector: show an image, know its size."""

    def show_image(self, image: np.ndarray) -> None: ...
    def size(self) -> tuple[int, int]: ...


class CameraLink(Protocol):
    def capture(self, settle_s: float = 0.25) -> Optional[np.ndarray]: ...
    def size(self) -> tuple[int, int]: ...


@dataclass
class CalibrationResult:
    success: bool
    homography: Optional[Matrix3x3] = None
    validation: Optional[ValidationResult] = None
    camera_points: list[Point] = field(default_factory=list)
    projector_points: list[Point] = field(default_factory=list)
    message: str = ""
    attempts: int = 0
    debug_frames: dict[str, np.ndarray] = field(default_factory=dict)

    def status_text(self) -> str:
        if not self.success or self.validation is None:
            return f"Calibration failed.\n{self.message}"
        return (f"Calibration complete\n\nReprojection error: {self.validation.mean_error_px:.1f} px\n"
                f"Max error: {self.validation.max_error_px:.1f} px\nStatus: {'VALID' if self.validation.valid else 'INVALID'}")


FAILURE_HINTS = (
    "Possible causes:\n"
    "- camera cannot see projected pattern\n"
    "- projector is too bright/dim\n"
    "- surface is outside camera view\n"
    "- projector image is clipped\n"
    "- object is blocking the pattern"
)


class CalibrationMethod(ABC):
    name: str = "abstract"

    @abstractmethod
    def calibrate(self, projector: ProjectorLink, camera: CameraLink,
                  progress: Callable[[str, float], None] | None = None) -> CalibrationResult: ...

    @abstractmethod
    def validate(self, projector: ProjectorLink, camera: CameraLink, H: np.ndarray) -> ValidationResult: ...

    def save(self, result: CalibrationResult, store: ProfileStore, **meta) -> CalibrationProfile:
        assert result.homography is not None and result.validation is not None
        prof = CalibrationProfile(method=self.name, homography=result.homography, validation=result.validation,
                                  workspace_camera=result.camera_points, workspace_projector=result.projector_points, **meta)
        store.save(prof)
        return prof

    def load(self, store: ProfileStore, name: str) -> Optional[CalibrationProfile]:
        return store.load(name)


def marker_layout(width: int, height: int, n: int = 4, margin_fraction: float = 0.12) -> list[Point]:
    """Projector-space marker positions: 4 corners, or a 3x3 grid for n=9."""
    mx, my = width * margin_fraction, height * margin_fraction
    if n == 4:
        return [Point(x=mx, y=my), Point(x=width - mx, y=my), Point(x=width - mx, y=height - my), Point(x=mx, y=height - my)]
    if n == 9:
        xs = [mx, width / 2, width - mx]
        ys = [my, height / 2, height - my]
        return [Point(x=x, y=y) for y in ys for x in xs]
    raise ValueError("n must be 4 or 9")


class PlanarMarkerCalibration(CalibrationMethod):
    """Project bright discs one-at-a-time and all-at-once; detect them; fit a homography.

    Projecting markers *individually* (in addition to the full pattern) gives an
    unambiguous projector->camera correspondence that does not depend on
    ordering heuristics, at the cost of a few extra frames.
    """

    name = "planar_4point"

    def __init__(self, n_points: int = 4, marker_radius: int = 28, margin_fraction: float = 0.12,
                 validation_threshold_px: float = 10.0, min_confidence: float = 0.6, max_retries: int = 3,
                 settle_s: float = 0.3):
        self.n_points = n_points
        self.name = "planar_4point" if n_points == 4 else "planar_9point"
        self.marker_radius = marker_radius
        self.margin_fraction = margin_fraction
        self.validator = CalibrationValidator(validation_threshold_px)
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.settle_s = settle_s

    # --- capture helpers ------------------------------------------------------------------
    def _capture_marker(self, projector: ProjectorLink, camera: CameraLink, pw: int, ph: int, p: Point,
                        reference: np.ndarray) -> Optional[DetectedMarker]:
        from bridge.render.renderer import calibration_pattern

        projector.show_image(calibration_pattern(pw, ph, [p], self.marker_radius))
        frame = camera.capture(self.settle_s)
        if frame is None:
            return None
        dets = detect_bright_markers(frame, reference, expected_count=1)
        if not dets or dets[0].confidence < self.min_confidence * 0.5:
            return None
        return dets[0]

    def _collect(self, projector: ProjectorLink, camera: CameraLink, layout: list[Point],
                 progress: Callable[[str, float], None] | None) -> tuple[list[Point], list[Point], dict]:
        from bridge.render.renderer import calibration_pattern, solid

        pw, ph = projector.size()
        debug: dict[str, np.ndarray] = {}
        projector.show_image(solid(pw, ph, (0, 0, 0)))
        reference = camera.capture(self.settle_s * 2)
        if reference is None:
            raise RuntimeError("camera returned no frame")
        debug["reference"] = reference
        cam_pts: list[Point] = []
        proj_pts: list[Point] = []
        for i, p in enumerate(layout):
            if progress:
                progress(f"Detecting marker {i + 1}/{len(layout)}...", (i + 1) / (len(layout) + 1))
            det = self._capture_marker(projector, camera, pw, ph, p, reference)
            if det is not None:
                cam_pts.append(det.center)
                proj_pts.append(p)
        # Also project the full pattern for a consistency check / debug image.
        projector.show_image(calibration_pattern(pw, ph, layout, self.marker_radius))
        full = camera.capture(self.settle_s)
        if full is not None:
            debug["pattern"] = full
            if len(cam_pts) < len(layout):
                # Fallback: all-at-once detection with geometric matching.
                dets = detect_bright_markers(full, reference, expected_count=len(layout))
                matched = match_markers_to_layout(dets, layout, camera.size())
                if len(matched) == len(layout):
                    cam_pts = [m.center for _, m in matched]
                    proj_pts = [layout[i] for i, _ in matched]
        return cam_pts, proj_pts, debug

    # --- public ------------------------------------------------------------------------
    def calibrate(self, projector: ProjectorLink, camera: CameraLink,
                  progress: Callable[[str, float], None] | None = None) -> CalibrationResult:
        from bridge.render.renderer import solid

        pw, ph = projector.size()
        layout = marker_layout(pw, ph, self.n_points, self.margin_fraction)
        last_msg = ""
        debug: dict = {}
        for attempt in range(1, self.max_retries + 1):
            log.info("Calibration started attempt=%d method=%s", attempt, self.name)
            if progress:
                progress(f"Detecting surface... (attempt {attempt})", 0.05)
            try:
                cam_pts, proj_pts, debug = self._collect(projector, camera, layout, progress)
            except RuntimeError as e:
                last_msg = str(e)
                continue
            if len(cam_pts) < 4:
                last_msg = f"Only {len(cam_pts)} of {len(layout)} markers detected."
                log.warning("Calibration attempt %d failed: %s", attempt, last_msg)
                continue
            try:
                H, mask = compute_homography(cam_pts, proj_pts)
            except HomographyError as e:
                last_msg = str(e)
                continue
            fit = self.validator.validate(H, points_to_array(cam_pts), points_to_array(proj_pts), mask)
            if progress:
                progress("Validating...", 0.9)
            val = self.validate(projector, camera, H) if self.n_points == 4 else fit
            # Use the independent validation when available, otherwise the fit residuals.
            if not np.isfinite(val.mean_error_px):
                val = fit
            projector.show_image(solid(pw, ph, (255, 255, 255)))
            result = CalibrationResult(success=val.valid, homography=Matrix3x3.from_numpy(H), validation=val,
                                       camera_points=cam_pts, projector_points=proj_pts, attempts=attempt,
                                       debug_frames=debug,
                                       message="" if val.valid else f"Reprojection error too high ({val.mean_error_px:.1f} px).")
            log.info("Calibration complete error=%.2fpx valid=%s", val.mean_error_px, val.valid)
            if val.valid:
                return result
            last_msg = result.message
        projector.show_image(solid(pw, ph, (255, 255, 255)))
        log.error("Calibration failed after %d attempts: %s", self.max_retries, last_msg)
        return CalibrationResult(success=False, message=f"{last_msg}\n\n{FAILURE_HINTS}", attempts=self.max_retries,
                                 debug_frames=debug)

    def validate(self, projector: ProjectorLink, camera: CameraLink, H: np.ndarray) -> ValidationResult:
        """Project independent validation points (a 3x3 grid inset differently) and measure."""
        from bridge.render.renderer import solid

        pw, ph = projector.size()
        pts = marker_layout(pw, ph, 9, margin_fraction=0.25)
        projector.show_image(solid(pw, ph, (0, 0, 0)))
        reference = camera.capture(self.settle_s)
        cam_pts, proj_pts = [], []
        if reference is not None:
            for p in pts:
                det = self._capture_marker(projector, camera, pw, ph, p, reference)
                if det is not None:
                    cam_pts.append(det.center)
                    proj_pts.append(p)
        if len(cam_pts) < 4:
            return ValidationResult(mean_error_px=float("inf"), max_error_px=float("inf"), median_error_px=float("inf"),
                                    n_points=len(cam_pts), threshold_px=self.validator.threshold_px, valid=False, inlier_ratio=0.0)
        return self.validator.validate(H, points_to_array(cam_pts), points_to_array(proj_pts))


def build_method(name: str, **kw) -> CalibrationMethod:
    if name in ("planar_4point", "planar"):
        return PlanarMarkerCalibration(n_points=4, **kw)
    if name == "planar_9point":
        return PlanarMarkerCalibration(n_points=9, **kw)
    raise ValueError(f"Unknown calibration method '{name}'. Available: planar_4point, planar_9point")


class CalibrationEngine:
    """Glue: runs a CalibrationMethod, keeps the result, persists profiles, invalidates on change."""

    def __init__(self, method: CalibrationMethod, store: ProfileStore):
        self.method = method
        self.store = store
        self.result: Optional[CalibrationResult] = None
        self.profile: Optional[CalibrationProfile] = None

    def run(self, projector: ProjectorLink, camera: CameraLink,
            progress: Callable[[str, float], None] | None = None) -> CalibrationResult:
        t0 = time.time()
        self.result = self.method.calibrate(projector, camera, progress)
        log.info("Calibration run finished in %.1fs success=%s", time.time() - t0, self.result.success)
        return self.result

    def save(self, name: str, camera_id: str, camera_res: tuple[int, int], display_id: str,
             display_res: tuple[int, int], surface_type: str, mode: str = "physical") -> Optional[CalibrationProfile]:
        if self.result is None or not self.result.success:
            return None
        self.profile = self.method.save(self.result, self.store, name=name, camera_id=camera_id, camera_resolution=camera_res,
                                        display_id=display_id, display_resolution=display_res, surface_type=surface_type, mode=mode)
        return self.profile

    def load_matching(self, camera_id: str | None, camera_res: tuple[int, int] | None,
                      display_id: str | None, display_res: tuple[int, int] | None) -> Optional[CalibrationProfile]:
        self.profile = self.store.find_matching(camera_id, camera_res, display_id, display_res)
        if self.profile is not None:
            self.result = CalibrationResult(success=True, homography=self.profile.homography, validation=self.profile.validation,
                                            camera_points=self.profile.workspace_camera, projector_points=self.profile.workspace_projector)
            log.info("Loaded matching profile '%s' (error=%.2fpx)", self.profile.name, self.profile.validation_error)
        return self.profile

    def invalidate(self, reason: str) -> None:
        if self.result is not None or self.profile is not None:
            log.warning("Calibration invalidated: %s", reason)
        self.result = None
        self.profile = None

    @property
    def is_valid(self) -> bool:
        return self.result is not None and self.result.success and self.result.homography is not None

    @property
    def homography(self) -> Optional[Matrix3x3]:
        return self.result.homography if self.is_valid and self.result else None
