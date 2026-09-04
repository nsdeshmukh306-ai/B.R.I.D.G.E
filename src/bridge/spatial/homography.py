"""Planar homography and coordinate mapping between camera and projector space."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from pydantic import BaseModel, field_validator

from bridge.spatial.geometry import BoundingBox, Point, Polygon, array_to_points, points_to_array


class Matrix3x3(BaseModel):
    rows: list[list[float]]

    @field_validator("rows")
    @classmethod
    def _shape(cls, v: list[list[float]]) -> list[list[float]]:
        if len(v) != 3 or any(len(r) != 3 for r in v):
            raise ValueError("matrix must be 3x3")
        return v

    def to_numpy(self) -> np.ndarray:
        return np.array(self.rows, dtype=np.float64)

    @classmethod
    def from_numpy(cls, m: np.ndarray) -> "Matrix3x3":
        m = np.asarray(m, dtype=np.float64)
        if m.shape != (3, 3):
            raise ValueError("matrix must be 3x3")
        return cls(rows=m.tolist())

    @classmethod
    def identity(cls) -> "Matrix3x3":
        return cls.from_numpy(np.eye(3))


class HomographyError(RuntimeError):
    pass


def compute_homography(
    src: Sequence[Point] | np.ndarray,
    dst: Sequence[Point] | np.ndarray,
    method: int = cv2.RANSAC,
    ransac_threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute H such that dst ~ H @ src. Returns (H, inlier_mask)."""
    src_arr = src if isinstance(src, np.ndarray) else points_to_array(src)
    dst_arr = dst if isinstance(dst, np.ndarray) else points_to_array(dst)
    if len(src_arr) < 4 or len(dst_arr) < 4:
        raise HomographyError("at least 4 point pairs are required")
    if len(src_arr) != len(dst_arr):
        raise HomographyError("point lists must have the same length")
    if len(src_arr) == 4:
        method = 0  # exact solve; RANSAC needs >4 to be meaningful
    H, mask = cv2.findHomography(src_arr.astype(np.float32), dst_arr.astype(np.float32), method, ransac_threshold)
    if H is None or not np.all(np.isfinite(H)):
        raise HomographyError("homography estimation failed (degenerate points?)")
    if np.linalg.cond(H) > 1e8 or abs(np.linalg.det(H)) < 1e-12:
        raise HomographyError("homography is degenerate (collinear or coincident points)")
    mask = np.ones((len(src_arr), 1), dtype=np.uint8) if mask is None else mask
    return H, mask


def apply_homography(H: np.ndarray, points: Sequence[Point] | np.ndarray) -> np.ndarray:
    arr = points if isinstance(points, np.ndarray) else points_to_array(points)
    if arr.size == 0:
        return np.zeros((0, 2))
    hom = np.hstack([arr, np.ones((len(arr), 1))])
    out = (H @ hom.T).T
    w = out[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return out[:, :2] / w


def reprojection_errors(H: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    proj = apply_homography(H, src)
    return np.linalg.norm(proj - dst, axis=1)


class CoordinateMapper:
    """Maps camera-space geometry to projector space (and back) with one homography."""

    def __init__(self, homography: Matrix3x3 | np.ndarray, trim: tuple[float, float] = (0.0, 0.0)):
        H = homography.to_numpy() if isinstance(homography, Matrix3x3) else np.asarray(homography, dtype=np.float64)
        # Registration trim: a small user-set translation in projector pixels, folded into H so
        # every mapping (points, boxes, masks, inverse) stays consistent.
        self.trim = (float(trim[0]), float(trim[1]))
        T = np.array([[1, 0, self.trim[0]], [0, 1, self.trim[1]], [0, 0, 1]], dtype=np.float64)
        self.H_raw = H
        self.H = T @ H
        try:
            self.H_inv = np.linalg.inv(self.H)
        except np.linalg.LinAlgError as e:
            raise HomographyError("homography is singular") from e

    def with_trim(self, dx: float, dy: float) -> "CoordinateMapper":
        return CoordinateMapper(self.H_raw, (dx, dy))

    @classmethod
    def identity(cls) -> "CoordinateMapper":
        return cls(np.eye(3))

    def camera_to_projector(self, p: Point) -> Point:
        out = apply_homography(self.H, [p])[0]
        return Point(x=float(out[0]), y=float(out[1]))

    def projector_to_camera(self, p: Point) -> Point:
        out = apply_homography(self.H_inv, [p])[0]
        return Point(x=float(out[0]), y=float(out[1]))

    def camera_points_to_projector(self, pts: Sequence[Point]) -> list[Point]:
        return array_to_points(apply_homography(self.H, pts))

    def camera_bbox_to_projector_polygon(self, bbox: BoundingBox) -> Polygon:
        return Polygon(points=self.camera_points_to_projector(bbox.corners()))

    def camera_bbox_to_projector_bbox(self, bbox: BoundingBox) -> BoundingBox:
        return self.camera_bbox_to_projector_polygon(bbox).bounding_box()

    def camera_radius_to_projector(self, center: Point, radius: float) -> float:
        """Approximate local scale: map a small circle and take its mean radius."""
        c = self.camera_to_projector(center)
        ring = [Point(x=center.x + radius, y=center.y), Point(x=center.x, y=center.y + radius),
                Point(x=center.x - radius, y=center.y), Point(x=center.x, y=center.y - radius)]
        mapped = self.camera_points_to_projector(ring)
        return float(np.mean([c.distance_to(m) for m in mapped]))
