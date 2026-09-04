"""Detection of projected calibration markers (bright discs) in camera frames.

Uses difference imaging (pattern frame minus dark reference frame) so ambient
lighting and objects on the surface are suppressed, then blob analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from bridge.spatial.geometry import Point

log = logging.getLogger("bridge.calibration")


@dataclass
class DetectedMarker:
    center: Point
    area: float
    circularity: float
    confidence: float


def _difference(frame: np.ndarray, reference: np.ndarray | None) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if reference is None:
        return gray
    ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
    return cv2.subtract(gray, ref)


def detect_projector_footprint(
    white_frame: np.ndarray, black_frame: np.ndarray, min_area_fraction: float = 0.01
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Find the illuminated quadrilateral (projector image) in the camera view.

    Returns (corners, mask): corners is a 4x2 float array ordered TL, TR, BR, BL
    (None if no clean quad was found), mask is the filled footprint region
    (uint8 0/255; None if nothing lit up).
    """
    diff = cv2.GaussianBlur(_difference(white_frame, black_frame), (7, 7), 0)
    thr, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr < 8:
        return None, None
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    c = max(contours, key=cv2.contourArea)
    h, w = mask.shape[:2]
    if cv2.contourArea(c) < min_area_fraction * w * h:
        return None, None
    region = np.zeros_like(mask)
    cv2.drawContours(region, [c], -1, 255, -1)
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    corners = None
    for eps in (0.02, 0.04, 0.06, 0.08):
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4:
            corners = order_quad(approx.reshape(4, 2).astype(np.float64))
            break
    return corners, region


def order_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 points as TL, TR, BR, BL."""
    s = pts.sum(axis=1)
    d = pts[:, 0] - pts[:, 1]
    tl, br = pts[np.argmin(s)], pts[np.argmax(s)]
    tr, bl = pts[np.argmax(d)], pts[np.argmin(d)]
    return np.array([tl, tr, br, bl], dtype=np.float64)


def detect_bright_markers(
    frame: np.ndarray,
    reference: np.ndarray | None = None,
    expected_count: int | None = None,
    min_area: float = 8.0,
    min_circularity: float = 0.35,
    roi_mask: np.ndarray | None = None,
) -> list[DetectedMarker]:
    diff = _difference(frame, reference)
    if roi_mask is not None:
        diff = cv2.bitwise_and(diff, diff, mask=roi_mask)
    diff = cv2.GaussianBlur(diff, (3, 3), 0)
    peak = int(diff.max()) if diff.size else 0
    if peak < 12:  # nothing meaningfully bright
        return []
    # Threshold relative to the brightest response: robust to small, dim, blurred discs.
    thr = max(10, int(peak * 0.45))
    _, mask = cv2.threshold(diff, thr, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out: list[DetectedMarker] = []
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        perim = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (perim * perim) if perim > 0 else 0.0
        if circ < min_circularity:
            continue
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
        # Refine centre with intensity-weighted centroid inside the contour.
        x, y, w, h = cv2.boundingRect(c)
        roi = diff[y:y + h, x:x + w].astype(np.float64)
        contour_mask = np.zeros_like(roi)
        cv2.drawContours(contour_mask, [c - [x, y]], -1, 1.0, -1)
        weights = roi * contour_mask
        if weights.sum() > 0:
            ys, xs = np.mgrid[0:h, 0:w]
            cx = x + float((xs * weights).sum() / weights.sum())
            cy = y + float((ys * weights).sum() / weights.sum())
        brightness = float(diff[int(cy), int(cx)]) / 255.0 if 0 <= int(cy) < diff.shape[0] and 0 <= int(cx) < diff.shape[1] else 0.5
        conf = float(min(1.0, circ) * min(1.0, area / (min_area * 3)) * (0.5 + 0.5 * brightness))
        out.append(DetectedMarker(center=Point(x=cx, y=cy), area=float(area), circularity=float(circ), confidence=conf))
    out.sort(key=lambda d: d.area, reverse=True)
    if expected_count is not None and len(out) > expected_count:
        out = out[:expected_count]
    return out


def match_markers_to_layout(
    detected: list[DetectedMarker], layout: list[Point], frame_size: tuple[int, int]
) -> list[tuple[int, DetectedMarker]]:
    """Match detected blobs to the projected layout by normalized relative position.

    Works because projector->camera is a homography (order-preserving for a
    convex layout): we sort by angle around the centroid on both sides.
    """
    if len(detected) < len(layout):
        return []

    def angular_order(points: list[Point]) -> list[int]:
        cx = sum(p.x for p in points) / len(points)
        cy = sum(p.y for p in points) / len(points)
        return sorted(range(len(points)), key=lambda i: np.arctan2(points[i].y - cy, points[i].x - cx))

    det_pts = [d.center for d in detected[: len(layout)]]
    lay_order = angular_order(layout)
    det_order = angular_order(det_pts)
    # Align the two cyclic orders by minimizing normalized position distance across rotations.
    lw = max(p.x for p in layout) - min(p.x for p in layout) or 1
    lh = max(p.y for p in layout) - min(p.y for p in layout) or 1
    dw = max(p.x for p in det_pts) - min(p.x for p in det_pts) or 1
    dh = max(p.y for p in det_pts) - min(p.y for p in det_pts) or 1
    lx0, ly0 = min(p.x for p in layout), min(p.y for p in layout)
    dx0, dy0 = min(p.x for p in det_pts), min(p.y for p in det_pts)
    best, best_cost = None, float("inf")
    n = len(layout)
    for rot in range(n):
        cost = 0.0
        pairs = []
        for k in range(n):
            li = lay_order[k]
            di = det_order[(k + rot) % n]
            lp, dp = layout[li], det_pts[di]
            cost += ((lp.x - lx0) / lw - (dp.x - dx0) / dw) ** 2 + ((lp.y - ly0) / lh - (dp.y - dy0) / dh) ** 2
            pairs.append((li, detected[di]))
        if cost < best_cost:
            best, best_cost = pairs, cost
    return sorted(best or [], key=lambda t: t[0])
