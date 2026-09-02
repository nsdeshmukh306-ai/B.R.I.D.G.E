"""Local object tracking so projected graphics follow objects without AI calls."""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel

from bridge.spatial.geometry import BoundingBox, Point
from bridge.vision.detection import ContourDetector, Detection, color_score

log = logging.getLogger("bridge.tracking")


class TrackingState(BaseModel):
    object_id: str
    label: str
    center: Point
    bbox: BoundingBox
    confidence: float
    is_lost: bool = False
    frames_since_seen: int = 0
    backend: str = ""


def _make_cv_tracker(backend: str):
    legacy = getattr(cv2, "legacy", None)
    table = {
        "csrt": [getattr(cv2, "TrackerCSRT_create", None), getattr(legacy, "TrackerCSRT_create", None) if legacy else None],
        "kcf": [getattr(cv2, "TrackerKCF_create", None), getattr(legacy, "TrackerKCF_create", None) if legacy else None],
        "mosse": [getattr(legacy, "TrackerMOSSE_create", None) if legacy else None],
        "mil": [getattr(cv2, "TrackerMIL_create", None)],
    }
    # MIL is only used when explicitly requested: it is ~10x slower than CSRT/KCF.
    for ctor in table.get(backend, []) + (table["kcf"] if backend != "mil" else []):
        if ctor is not None:
            try:
                return ctor()
            except Exception:  # noqa: BLE001
                continue
    return None


class ObjectTracker:
    """Tracks one object. OpenCV tracker with re-detection by colour/contour when lost."""

    def __init__(self, backend: str = "csrt", lost_after_frames: int = 15, min_confidence: float = 0.3):
        self.backend_name = backend
        self.lost_after = lost_after_frames
        self.min_confidence = min_confidence
        self._tracker = None
        self._lock = threading.Lock()
        self.state: Optional[TrackingState] = None
        self._object_id = ""
        self._label = ""
        self._color: Optional[str] = None
        self._template_area = 0.0
        self._redetector = ContourDetector()
        self.last_update_ms = 0.0

    @property
    def active(self) -> bool:
        return self.state is not None

    def start(self, detection: Detection, frame: np.ndarray, object_id: str | None = None, color_hint: str | None = None) -> TrackingState:
        with self._lock:
            self._object_id = object_id or f"{detection.label}-{int(time.time() * 1000) % 100000}"
            self._label = detection.label
            self._color = color_hint
            bbox = detection.bbox.clip(frame.shape[1], frame.shape[0])
            self._template_area = bbox.area
            self._tracker = _make_cv_tracker(self.backend_name)
            backend = self.backend_name
            if self.backend_name == "color":
                self._tracker = None  # pure local re-detection each frame (fast, colour/contour based)
            elif self._tracker is not None and bbox.w >= 4 and bbox.h >= 4:
                ok = self._tracker.init(frame, bbox.as_xywh_int())
                if ok is False:  # OpenCV >= 4.5 returns None on success
                    self._tracker = None
            else:
                self._tracker = None
            if self._tracker is None:
                backend = "color"
            self.state = TrackingState(object_id=self._object_id, label=self._label, center=bbox.center, bbox=bbox,
                                       confidence=detection.confidence, backend=backend)
            log.info("Tracking started object=%s label=%s bbox=%s backend=%s", self._object_id, self._label,
                     bbox.as_xywh_int(), backend)
            return self.state

    def _redetect(self, frame: np.ndarray, near: Point, max_dist: float) -> Optional[BoundingBox]:
        cands = self._redetector.detect(frame)
        best, best_score = None, 0.0
        for d in cands:
            dist = d.center.distance_to(near)
            if dist > max_dist:
                continue
            area_ratio = min(d.bbox.area, self._template_area) / max(d.bbox.area, self._template_area, 1)
            if area_ratio < 0.3:
                continue
            score = area_ratio * (1 - dist / max_dist)
            if self._color:
                cs = color_score(frame, d.bbox, self._color)
                if cs < 0.15:
                    continue  # wrong colour: not our object
                score *= 0.3 + cs
            if score > best_score:
                best, best_score = d.bbox, score
        return best

    def update(self, frame: np.ndarray) -> Optional[TrackingState]:
        t0 = time.perf_counter()
        with self._lock:
            if self.state is None:
                return None
            st = self.state
            ok, bbox = False, None
            if self._tracker is not None:
                try:
                    ok, box = self._tracker.update(frame)
                except cv2.error:
                    ok = False
                if ok:
                    x, y, w, h = box
                    bbox = BoundingBox(x=float(x), y=float(y), w=float(w), h=float(h)).clip(frame.shape[1], frame.shape[0])
                    if bbox.area < 0.2 * self._template_area or bbox.area > 5 * self._template_area:
                        ok = False
                if ok and self._color and color_score(frame, bbox, self._color) < 0.05:
                    ok = False  # drifted onto the background
            if not ok:
                max_dist = max(frame.shape[:2]) * (0.15 + 0.05 * st.frames_since_seen)
                bbox = self._redetect(frame, st.center, max_dist)
                if bbox is not None:
                    ok = True
                    if self.backend_name != "color":
                        self._tracker = _make_cv_tracker(self.backend_name)
                        if self._tracker is not None and self._tracker.init(frame, bbox.as_xywh_int()) is False:
                            self._tracker = None
            if ok and bbox is not None:
                conf = min(1.0, 0.6 + 0.4 * (1 - min(st.frames_since_seen, 10) / 10))
                self.state = st.model_copy(update={"center": bbox.center, "bbox": bbox, "confidence": conf,
                                                   "is_lost": False, "frames_since_seen": 0})
            else:
                n = st.frames_since_seen + 1
                lost = n >= self.lost_after
                if lost and not st.is_lost:
                    log.warning("Tracking lost object=%s", self._object_id)
                self.state = st.model_copy(update={"confidence": max(0.0, st.confidence - 0.1), "is_lost": lost,
                                                   "frames_since_seen": n})
            self.last_update_ms = (time.perf_counter() - t0) * 1000
            return self.state

    def stop(self) -> None:
        with self._lock:
            if self.state is not None:
                log.info("Tracking stopped object=%s", self._object_id)
            self._tracker = None
            self.state = None


class MultiTracker:
    """Tracks several objects at once (e.g. 'all the screws')."""

    def __init__(self, backend: str = "csrt", lost_after_frames: int = 15):
        self.backend, self.lost_after = backend, lost_after_frames
        self.trackers: dict[str, ObjectTracker] = {}

    def start(self, detections: list[Detection], frame: np.ndarray, color_hint: str | None = None) -> list[TrackingState]:
        self.stop_all()
        states = []
        for i, d in enumerate(detections):
            t = ObjectTracker(self.backend, self.lost_after)
            states.append(t.start(d, frame, object_id=f"{d.label}-{i}", color_hint=color_hint))
            self.trackers[states[-1].object_id] = t
        return states

    def update(self, frame: np.ndarray) -> list[TrackingState]:
        out = []
        for t in self.trackers.values():
            s = t.update(frame)
            if s is not None:
                out.append(s)
        return out

    def stop_all(self) -> None:
        for t in self.trackers.values():
            t.stop()
        self.trackers.clear()

    @property
    def active(self) -> bool:
        return bool(self.trackers)
