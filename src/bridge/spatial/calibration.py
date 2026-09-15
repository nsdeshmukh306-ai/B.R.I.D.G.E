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

import cv2
import numpy as np

from bridge.spatial.geometry import Point, points_to_array
from bridge.spatial.homography import HomographyError, Matrix3x3, apply_homography, compute_homography
from bridge.spatial.markers import DetectedMarker, detect_bright_markers, detect_projector_footprint
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
    """Coarse-to-fine planar calibration with projected discs.

    1. Project black, then white: the lit quadrilateral is the projector's
       footprint in the camera. Its corners give a *coarse* homography and a
       region of interest, so marker search is confined to the surface.
    2. Project one disc at a time at known projector positions. Each disc is
       searched near where the coarse homography predicts it, so ordering is
       never ambiguous and stray reflections elsewhere are ignored.
    3. Fit the homography, then project an independent grid and measure the
       reprojection error. A calibration is only VALID if that independent
       validation succeeded - fit residuals are never used as a substitute.

    Marker size scales with the projector resolution and grows on each retry.
    """

    name = "planar_4point"

    def __init__(self, n_points: int = 4, marker_radius: int = 28, margin_fraction: float = 0.12,
                 validation_threshold_px: float = 10.0, min_confidence: float = 0.6, max_retries: int = 3,
                 settle_s: float = 0.4):
        self.n_points = n_points
        self.name = "planar_4point" if n_points == 4 else "planar_9point"
        self.marker_radius = marker_radius
        self.margin_fraction = margin_fraction
        self.validator = CalibrationValidator(validation_threshold_px)
        self.min_confidence = min_confidence
        self.max_retries = max_retries
        self.settle_s = settle_s
        self._radius_scale = 1.0
        self._footprint_mask: Optional[np.ndarray] = None
        self._footprint_corners: Optional[np.ndarray] = None
        self._H_coarse: Optional[np.ndarray] = None
        self._last_validation_pts: Optional[tuple[list[Point], list[Point]]] = None

    # --- geometry helpers ----------------------------------------------------------------
    def _radius(self, pw: int, ph: int) -> int:
        """Disc radius in projector pixels: at least ~3% of the short edge, scaled per retry."""
        base = max(self.marker_radius, int(0.03 * min(pw, ph)))
        return int(base * self._radius_scale)

    def _search_radius(self, cam_size: tuple[int, int]) -> float:
        if self._footprint_corners is not None:
            c = self._footprint_corners
            diag = float(np.linalg.norm(c[2] - c[0]))
            return max(12.0, 0.18 * diag)
        return 0.2 * max(cam_size)

    def _expected(self, p: Point) -> Optional[Point]:
        if self._H_coarse is None:
            return None
        try:
            H_inv = np.linalg.inv(self._H_coarse)  # camera->projector, so invert to predict camera position
        except np.linalg.LinAlgError:
            return None
        out = apply_homography(H_inv, [p])[0]
        return Point(x=float(out[0]), y=float(out[1]))

    # --- capture helpers ------------------------------------------------------------------
    def _capture_footprint(self, projector: ProjectorLink, camera: CameraLink, pw: int, ph: int) -> tuple[np.ndarray, dict]:
        """Black reference + white frame -> footprint mask, corners and coarse homography."""
        from bridge.render.renderer import solid

        debug: dict[str, np.ndarray] = {}
        projector.show_image(solid(pw, ph, (0, 0, 0)))
        reference = camera.capture(self.settle_s * 2)
        if reference is None:
            raise RuntimeError("camera returned no frame")
        projector.show_image(solid(pw, ph, (255, 255, 255)))
        white = camera.capture(self.settle_s * 2)
        debug["reference"] = reference
        self._footprint_mask, self._footprint_corners, self._H_coarse = None, None, None
        if white is not None:
            debug["white"] = white
            corners, mask = detect_projector_footprint(white, reference)
            if mask is not None:
                self._footprint_mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
            if corners is not None:
                try:
                    proj_corners = np.array([[0, 0], [pw, 0], [pw, ph], [0, ph]], dtype=np.float64)
                    self._H_coarse, _ = compute_homography(corners, proj_corners)
                    self._footprint_corners = corners
                    log.info("Projector footprint found in camera: %s", corners.astype(int).tolist())
                except HomographyError:
                    self._H_coarse = None
            if mask is None:
                log.warning("Projector footprint not visible to the camera (white frame gave no response)")
        return reference, debug

    def _capture_marker(self, projector: ProjectorLink, camera: CameraLink, pw: int, ph: int, p: Point,
                        reference: np.ndarray) -> Optional[DetectedMarker]:
        from bridge.render.renderer import calibration_pattern

        projector.show_image(calibration_pattern(pw, ph, [p], self._radius(pw, ph)))
        frame = camera.capture(self.settle_s)
        if frame is None:
            return None
        dets = detect_bright_markers(frame, reference, roi_mask=self._footprint_mask)
        if not dets:
            return None
        expected = self._expected(p)
        if expected is not None:
            r = self._search_radius(camera.size())
            near = [d for d in dets if d.center.distance_to(expected) <= r]
            if near:
                near.sort(key=lambda d: (d.center.distance_to(expected) / r) - d.confidence)
                return near[0]
            # The coarse (footprint-corner) homography that produced `expected` is a 4-point
            # fit and can be locally inaccurate, especially when the footprint quad is small
            # or heavily skewed in the camera view — the real marker then lands outside the
            # predicted radius even though it was detected. Returning None here (as before)
            # silently drops a real detection and was the main source of "0 markers found"
            # despite a clean footprint. `dets` is still constrained to the footprint ROI, and
            # exactly one disc is lit at a time, so falling back to the single brightest blob
            # in that ROI is a much smaller risk than the false positives the expected-position
            # filter exists to reject (which come from reflections *outside* the footprint).
            best = dets[0]
            if best.confidence >= self.min_confidence:
                log.debug("marker at %s: %d blobs but none within %.0fpx of expected %s; using largest in-ROI blob instead",
                          p.as_int(), len(dets), r, expected.as_int())
                return best
            return None
        best = dets[0]
        return best if best.confidence >= self.min_confidence * 0.3 else None

    def _collect(self, projector: ProjectorLink, camera: CameraLink, layout: list[Point],
                 progress: Callable[[str, float], None] | None) -> tuple[list[Point], list[Point], dict]:
        from bridge.render.renderer import calibration_pattern

        pw, ph = projector.size()
        if progress:
            progress("Locating projector footprint...", 0.08)
        reference, debug = self._capture_footprint(projector, camera, pw, ph)
        cam_pts: list[Point] = []
        proj_pts: list[Point] = []
        for i, p in enumerate(layout):
            if progress:
                progress(f"Detecting marker {i + 1}/{len(layout)}...", 0.1 + 0.6 * (i + 1) / len(layout))
            det = self._capture_marker(projector, camera, pw, ph, p, reference)
            if det is not None:
                cam_pts.append(det.center)
                proj_pts.append(p)
        # Debug image: camera view of the full pattern with detections annotated.
        projector.show_image(calibration_pattern(pw, ph, layout, self._radius(pw, ph)))
        full = camera.capture(self.settle_s)
        if full is not None:
            debug["pattern"] = self._annotate(full, cam_pts, layout)
        # Fallback: if markers were lost but the footprint quad is clean, its corners are
        # exact correspondences to the projector image corners (as long as the image is not clipped).
        if len(cam_pts) < 4 and self._footprint_corners is not None:
            log.warning("Only %d markers detected; adding projector footprint corners as correspondences", len(cam_pts))
            for (cx, cy), (px, py) in zip(self._footprint_corners, [(0, 0), (pw, 0), (pw, ph), (0, ph)]):
                cam_pts.append(Point(x=float(cx), y=float(cy)))
                proj_pts.append(Point(x=float(px), y=float(py)))
        return cam_pts, proj_pts, debug

    def _annotate(self, frame: np.ndarray, cam_pts: list[Point], layout: list[Point]) -> np.ndarray:
        img = frame.copy()
        if self._footprint_corners is not None:
            cv2.polylines(img, [self._footprint_corners.astype(np.int32)], True, (0, 200, 255), 1, cv2.LINE_AA)
        for p in layout:
            e = self._expected(p)
            if e is not None:
                cv2.drawMarker(img, e.as_int(), (0, 165, 255), cv2.MARKER_CROSS, 10, 1, cv2.LINE_AA)
        for c in cam_pts:
            cv2.circle(img, c.as_int(), 7, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img, f"{len(cam_pts)} marker(s) detected", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
        return img

    # --- public ------------------------------------------------------------------------
    def calibrate(self, projector: ProjectorLink, camera: CameraLink,
                  progress: Callable[[str, float], None] | None = None) -> CalibrationResult:
        from bridge.render.renderer import solid

        pw, ph = projector.size()
        layout = marker_layout(pw, ph, self.n_points, self.margin_fraction)
        last_msg = ""
        debug: dict = {}
        for attempt in range(1, self.max_retries + 1):
            self._radius_scale = 1.0 + 0.7 * (attempt - 1)  # bigger discs on each retry
            log.info("Calibration started attempt=%d method=%s marker_radius=%dpx", attempt, self.name, self._radius(pw, ph))
            if progress:
                progress(f"Detecting surface... (attempt {attempt})", 0.05)
            try:
                cam_pts, proj_pts, debug = self._collect(projector, camera, layout, progress)
            except RuntimeError as e:
                last_msg = str(e)
                continue
            n_markers = sum(1 for p in proj_pts if p in layout)
            if len(cam_pts) < 4:
                last_msg = (f"Only {n_markers} of {len(layout)} markers detected"
                            + (" and the projector footprint was not visible." if self._footprint_mask is None else "."))
                log.warning("Calibration attempt %d failed: %s", attempt, last_msg)
                continue
            try:
                H, mask = compute_homography(cam_pts, proj_pts)
            except HomographyError as e:
                last_msg = str(e)
                continue
            if progress:
                progress("Validating...", 0.85)
            self._H_coarse = H  # validation markers are searched near their predicted positions
            val = self.validate(projector, camera, H)
            projector.show_image(solid(pw, ph, (255, 255, 255)))
            if not np.isfinite(val.mean_error_px):
                last_msg = f"Validation markers not detected ({val.n_points} of 9 found); calibration not trusted."
                log.warning("Calibration attempt %d: %s", attempt, last_msg)
                continue
            # Refinement: the independent validation points are unbiased extra correspondences.
            # Re-fit on everything (least squares, RANSAC outlier rejection) for the final H; the
            # reported accuracy stays the honest pre-refinement measurement on unseen points.
            if val.valid and self._last_validation_pts is not None:
                vc, vp = self._last_validation_pts
                try:
                    H_ref, _ = compute_homography(cam_pts + vc, proj_pts + vp, ransac_threshold=3.0)
                    if np.all(np.isfinite(H_ref)):
                        H = H_ref
                        log.info("Homography refined on %d correspondences", len(cam_pts) + len(vc))
                except HomographyError:
                    pass
            result = CalibrationResult(success=val.valid, homography=Matrix3x3.from_numpy(H), validation=val,
                                       camera_points=cam_pts, projector_points=proj_pts, attempts=attempt,
                                       debug_frames=debug,
                                       message="" if val.valid else f"Reprojection error too high ({val.mean_error_px:.1f} px).")
            log.info("Calibration complete error=%.2fpx max=%.2fpx n=%d valid=%s", val.mean_error_px, val.max_error_px,
                     val.n_points, val.valid)
            if val.valid:
                return result
            last_msg = result.message
        projector.show_image(solid(pw, ph, (255, 255, 255)))
        log.error("Calibration failed after %d attempts: %s", self.max_retries, last_msg)
        return CalibrationResult(success=False, message=f"{last_msg}\n\n{FAILURE_HINTS}", attempts=self.max_retries,
                                 debug_frames=debug)

    def validate(self, projector: ProjectorLink, camera: CameraLink, H: np.ndarray) -> ValidationResult:
        """Project an independent 3x3 grid (inset 25%) and measure reprojection error against H."""
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
        self._last_validation_pts = (cam_pts, proj_pts) if len(cam_pts) >= 4 else None
        if len(cam_pts) < 4:
            return ValidationResult(mean_error_px=float("inf"), max_error_px=float("inf"), median_error_px=float("inf"),
                                    n_points=len(cam_pts), threshold_px=self.validator.threshold_px, valid=False,
                                    inlier_ratio=0.0, independent=True)
        res = self.validator.validate(H, points_to_array(cam_pts), points_to_array(proj_pts))
        res.independent = True
        return res


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
