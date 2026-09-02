"""Optional hand tracking via MediaPipe (used for future action verification).

Imports MediaPipe lazily; if it is not installed, `HandTracker.available` is
False and `update()` returns an empty list. Nothing else depends on it.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from bridge.spatial.geometry import Point

log = logging.getLogger("bridge.vision.hands")


class HandTracker:
    def __init__(self, max_hands: int = 2):
        self.available = False
        self._hands = None
        try:
            import mediapipe as mp  # type: ignore

            solutions = getattr(mp, "solutions", None)
            if solutions is not None and hasattr(solutions, "hands"):
                self._hands = solutions.hands.Hands(max_num_hands=max_hands, min_detection_confidence=0.5)
                self.available = True
        except Exception as e:  # noqa: BLE001
            log.info("MediaPipe hands unavailable: %s", e)

    def update(self, frame_bgr: np.ndarray) -> list[list[Point]]:
        """Returns a list of hands; each hand is 21 landmark points in pixel coords."""
        if not self.available or self._hands is None:
            return []
        import cv2

        h, w = frame_bgr.shape[:2]
        res = self._hands.process(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        out: list[list[Point]] = []
        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                out.append([Point(x=lm.x * w, y=lm.y * h) for lm in hand.landmark])
        return out

    def index_fingertip(self, frame_bgr: np.ndarray) -> Optional[Point]:
        hands = self.update(frame_bgr)
        return hands[0][8] if hands else None

    def close(self) -> None:
        if self._hands is not None:
            self._hands.close()
