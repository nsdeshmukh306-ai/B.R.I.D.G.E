"""Detection of projected calibration markers (bright discs) in camera frames.

Uses difference imaging (pattern frame minus dark reference frame) so ambient
lighting and objects on the surface are suppressed, then blob analysis.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

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


def detect_bright_markers(
    frame: np.ndarray,
    reference: np.ndarray | None = None,
    expected_count: int | None = None,
    min_area: float = 30.0,
    min_circularity: float = 0.5,
) -> list[DetectedMarker]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
    if reference is not None:
        ref = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY) if reference.ndim == 3 else reference
        diff = cv2.subtract(gray, ref)
    else:
        diff = gray
    diff = cv2.GaussianBlur(diff, (5, 5), 0)
    # Otsu on the difference image; fall back to a fixed threshold when the image is flat.
    thr_val, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if thr_val < 15:  # nothing meaningfully bright
        _, mask = cv2.threshold(diff, 60, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
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
        roi_mask = np.zeros_like(roi)
        cv2.drawContours(roi_mask, [c - [x, y]], -1, 1.0, -1)
        weights = roi * roi_mask
        if weights.sum() > 0:
            ys, xs = np.mgrid[0:h, 0:w]
            cx = x + float((xs * weights).sum() / weights.sum())
            cy = y + float((ys * weights).sum() / weights.sum())
        conf = float(min(1.0, circ) * min(1.0, area / (min_area * 4)))
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
