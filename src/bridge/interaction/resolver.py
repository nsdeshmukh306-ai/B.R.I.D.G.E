"""Target resolution: turn a TargetSpec into camera-space detections.

Priority (from the spec):
 1. an existing locally tracked object with the requested label
 2. a recent AI bounding box
 3. AI visual description + local colour/contour matching
 4. (caller) ask the AI again
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from bridge.interaction.commands import TargetSpec
from bridge.vision.detection import ContourDetector, Detection, match_description
from bridge.vision.tracking import TrackingState

log = logging.getLogger("bridge.resolver")


@dataclass
class Resolution:
    detections: list[Detection] = field(default_factory=list)
    strategy: str = "none"
    color_hint: Optional[str] = None

    @property
    def found(self) -> bool:
        return bool(self.detections)


COLOR_WORDS = ("red", "blue", "green", "black", "white", "grey", "gray", "yellow", "orange", "silver", "purple", "pink", "cyan")


def color_hint_from(text: str) -> Optional[str]:
    words = text.lower().replace("-", " ").split()
    for w in words:
        if w in COLOR_WORDS:
            return "grey" if w in ("gray", "silver") else w
    return None


class TargetResolver:
    def __init__(self, detector: ContourDetector | None = None, snap_iou: float = 0.45):
        self.detector = detector or ContourDetector()
        self.snap_iou = snap_iou

    def resolve(self, spec: TargetSpec, frame: np.ndarray, tracked: list[TrackingState] | None = None) -> Resolution:
        hint = color_hint_from(spec.description) or color_hint_from(spec.label)
        # 1. already tracked
        if tracked:
            same = [t for t in tracked if not t.is_lost and t.label == spec.label and spec.label]
            if same and not spec.boxes_camera:
                dets = [Detection.from_bbox(t.label, t.bbox, t.confidence, "tracker") for t in same]
                return Resolution(dets if spec.all_instances else dets[:1], "tracked", hint)
        # 2. AI boxes, snapped to local contours when a contour overlaps well (tightens loose boxes)
        if spec.boxes_camera:
            local = self.detector.detect(frame)
            dets: list[Detection] = []
            for b in spec.boxes_camera:
                best, best_iou = None, 0.0
                for d in local:
                    iou = d.bbox.iou(b)
                    if iou > best_iou:
                        best, best_iou = d, iou
                if best is not None and best_iou >= self.snap_iou and best.bbox.area <= b.area * 1.5:
                    dets.append(best.model_copy(update={"label": spec.label or best.label, "source": "ai+local", "confidence": 0.9}))
                else:
                    dets.append(Detection.from_bbox(spec.label or "object", b, 0.75, "ai"))
            log.info("Target resolved via AI boxes n=%d", len(dets))
            return Resolution(dets if spec.all_instances else dets[:1], "ai_boxes", hint)
        # 3. description matching against local contours
        if spec.description or hint:
            local = self.detector.detect(frame)
            m = match_description(frame, local, spec.description or spec.label)
            if m is not None:
                log.info("Target resolved via description match (%s)", spec.description)
                return Resolution([m.model_copy(update={"label": spec.label or m.label})], "description", hint)
        return Resolution([], "none", hint)
