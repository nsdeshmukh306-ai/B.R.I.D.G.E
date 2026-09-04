"""Calibration validation: measurable, not assumed."""
from __future__ import annotations

import numpy as np
from pydantic import BaseModel

from bridge.spatial.homography import reprojection_errors


class ValidationResult(BaseModel):
    """Reprojection error of a calibration.

    Errors are reported in CAMERA pixels (mean/max/median_error_px), which is what
    physical registration depends on: one camera pixel may cover many projector
    pixels when a 4K projector is watched by a small webcam, so projector-space
    numbers alone are misleading. Projector-space errors are kept for reference.
    """

    mean_error_px: float
    max_error_px: float
    median_error_px: float
    n_points: int
    threshold_px: float
    valid: bool
    inlier_ratio: float = 1.0
    independent: bool = False  # True only when measured on points NOT used for the fit
    mean_error_proj_px: float | None = None
    max_error_proj_px: float | None = None

    def summary(self) -> str:
        proj = f" (projector-space mean={self.mean_error_proj_px:.1f}px)" if self.mean_error_proj_px is not None else ""
        return (f"mean={self.mean_error_px:.2f}px max={self.max_error_px:.2f}px n={self.n_points} "
                f"threshold={self.threshold_px:.1f}px{proj} -> {'VALID' if self.valid else 'INVALID'}")


class CalibrationValidator:
    def __init__(self, threshold_px: float = 10.0, min_points: int = 4):
        self.threshold_px = threshold_px
        self.min_points = min_points

    def validate(self, H: np.ndarray, camera_pts: np.ndarray, projector_pts: np.ndarray,
                 inlier_mask: np.ndarray | None = None) -> ValidationResult:
        camera_pts = np.asarray(camera_pts, dtype=np.float64).reshape(-1, 2)
        projector_pts = np.asarray(projector_pts, dtype=np.float64).reshape(-1, 2)
        n = len(camera_pts)
        if n < self.min_points or n != len(projector_pts):
            return ValidationResult(mean_error_px=float("inf"), max_error_px=float("inf"), median_error_px=float("inf"),
                                    n_points=n, threshold_px=self.threshold_px, valid=False, inlier_ratio=0.0)
        errs_proj = reprojection_errors(H, camera_pts, projector_pts)
        try:
            errs_cam = reprojection_errors(np.linalg.inv(H), projector_pts, camera_pts)
        except np.linalg.LinAlgError:
            errs_cam = np.full(n, np.inf)
        ratio = float(np.mean(inlier_mask.ravel() > 0)) if inlier_mask is not None else 1.0
        mean, mx, med = float(np.mean(errs_cam)), float(np.max(errs_cam)), float(np.median(errs_cam))
        valid = bool(np.isfinite(mean) and mean <= self.threshold_px and mx <= self.threshold_px * 2.5 and ratio >= 0.75)
        return ValidationResult(mean_error_px=mean, max_error_px=mx, median_error_px=med, n_points=n,
                                threshold_px=self.threshold_px, valid=valid, inlier_ratio=ratio,
                                mean_error_proj_px=float(np.mean(errs_proj)), max_error_proj_px=float(np.max(errs_proj)))
