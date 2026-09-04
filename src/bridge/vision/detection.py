"""Object detection interface and local (non-AI) detectors.

Gemini answers WHAT; these answer WHERE, cheaply and locally.
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel, Field

from bridge.spatial.geometry import BoundingBox, Point

log = logging.getLogger("bridge.vision")


class Detection(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: BoundingBox
    center: Point
    source: str = "local"
    description: str = ""

    @classmethod
    def from_bbox(cls, label: str, bbox: BoundingBox, confidence: float, source: str, description: str = "") -> "Detection":
        return cls(label=label, confidence=confidence, bbox=bbox, center=bbox.center, source=source, description=description)


class ObjectDetector(ABC):
    name: str = "detector"

    @abstractmethod
    def detect(self, frame: np.ndarray) -> list[Detection]: ...


class ContourDetector(ObjectDetector):
    """Finds salient foreground blobs on a plain (light) surface.

    This is the MVP local detector: good enough for tools on a table, and
    used for re-detection when tracking is lost and for matching Gemini's
    visual descriptions by colour.
    """

    name = "contour"

    def __init__(self, min_area: float = 250.0, max_area_fraction: float = 0.35, blur: int = 5, reject_shadows: bool = False):
        self.min_area, self.max_area_fraction, self.blur = min_area, max_area_fraction, blur
        self.reject_shadows = reject_shadows

    def foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        # Surface is assumed to be the dominant brightness; objects deviate from it.
        # Background estimate at 1/4 scale (median of a large window) keeps this ~20 ms at 1080p.
        small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
        bg = cv2.resize(cv2.medianBlur(small, 21), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
        diff = cv2.absdiff(gray, bg)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        _, m1 = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        _, m2 = cv2.threshold(sat, 70, 255, cv2.THRESH_BINARY)
        mask = cv2.bitwise_or(m1, m2)
        if self.reject_shadows:
            # A projector at an angle casts a hard shadow beside every object. Shadows are
            # moderately darker than the surface and colourless; genuinely dark objects are
            # much darker. Remove the shadow band so it cannot pull an object's centre.
            ratio = gray.astype(np.float32) / np.maximum(bg.astype(np.float32), 1.0)
            shadow = ((ratio > 0.35) & (ratio < 0.85) & (sat < 45)).astype(np.uint8) * 255
            shadow = cv2.morphologyEx(shadow, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(shadow))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        return mask

    def detect(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        mask = self.foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        out: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area_fraction * w * h:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            if x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1:
                continue  # touching the border: probably the table edge / background
            bbox = BoundingBox(x=x, y=y, w=bw, h=bh)
            roi = frame[y:y + bh, x:x + bw]
            mroi = mask[y:y + bh, x:x + bw]
            mean_bgr = cv2.mean(roi, mask=mroi)[:3]
            out.append(Detection.from_bbox("object", bbox, min(1.0, area / (self.min_area * 8)), self.name,
                                           description=describe_color(mean_bgr)))
        out.sort(key=lambda d: d.bbox.area, reverse=True)
        return out


COLOR_NAMES = {
    "red": ((0, 10), (170, 180)), "orange": ((10, 22),), "yellow": ((22, 35),), "green": ((35, 85),),
    "cyan": ((85, 100),), "blue": ((100, 130),), "purple": ((130, 150),), "pink": ((150, 170),),
}


def describe_color(bgr: tuple[float, float, float]) -> str:
    b, g, r = bgr
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 60:
        return "black"
    if s < 45:
        return "white" if v > 190 else "grey"
    for name, ranges in COLOR_NAMES.items():
        for lo, hi in ranges:
            if lo <= h < hi:
                return name
    return "unknown"


def color_score(frame: np.ndarray, bbox: BoundingBox, wanted: str) -> float:
    """Fraction of pixels in bbox matching a colour name (0..1)."""
    x, y, w, h = bbox.as_xywh_int()
    roi = frame[max(0, y):y + h, max(0, x):x + w]
    if roi.size == 0:
        return 0.0
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hh, ss, vv = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    if wanted == "black":
        m = vv < 60
    elif wanted == "white":
        m = (ss < 45) & (vv > 190)
    elif wanted in ("grey", "gray", "silver"):
        m = (ss < 45) & (vv >= 60) & (vv <= 190)
    elif wanted in COLOR_NAMES:
        m = np.zeros(hh.shape, bool)
        for lo, hi in COLOR_NAMES[wanted]:
            m |= (hh >= lo) & (hh < hi)
        m &= (ss >= 45) & (vv >= 60)
    else:
        return 0.0
    return float(m.mean())


class ColorDetector(ObjectDetector):
    """Detect blobs of a named colour (used to ground descriptions like 'the red object')."""

    name = "color"

    def __init__(self, color: str, min_area: float = 200.0):
        self.color, self.min_area = color, min_area

    def detect(self, frame: np.ndarray) -> list[Detection]:
        base = ContourDetector(min_area=self.min_area).detect(frame)
        out = []
        for d in base:
            s = color_score(frame, d.bbox, self.color)
            if s > 0.25:
                out.append(d.model_copy(update={"label": f"{self.color} object", "confidence": min(1.0, s), "source": self.name}))
        return out


def match_description(frame: np.ndarray, candidates: list[Detection], description: str) -> Optional[Detection]:
    """Pick the candidate whose colour best matches a visual description (e.g. 'red-handled')."""
    words = description.lower().replace("-", " ").split()
    colors = [w for w in words if w in COLOR_NAMES or w in ("black", "white", "grey", "gray", "silver")]
    if not colors or not candidates:
        return None
    best, best_s = None, 0.0
    for d in candidates:
        s = max(color_score(frame, d.bbox, c) for c in colors)
        if s > best_s:
            best, best_s = d, s
    if best is None or best_s < 0.2:
        return None
    return best.model_copy(update={"confidence": min(1.0, best_s + 0.3), "source": "description-match"})
