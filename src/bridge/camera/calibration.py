"""Camera intrinsics placeholder-free helper: lens undistortion support.

V1 does not require intrinsic calibration (a planar homography absorbs a
perspective camera looking at a plane). This module lets a user supply
intrinsics for optional undistortion without changing the rest of the pipeline.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np
from pydantic import BaseModel


class CameraIntrinsics(BaseModel):
    fx: float
    fy: float
    cx: float
    cy: float
    dist: list[float] = []

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], dtype=np.float64)


class Undistorter:
    def __init__(self, intrinsics: Optional[CameraIntrinsics]):
        self.intrinsics = intrinsics

    def apply(self, frame: np.ndarray) -> np.ndarray:
        if self.intrinsics is None or not self.intrinsics.dist:
            return frame
        K = self.intrinsics.matrix()
        d = np.array(self.intrinsics.dist, dtype=np.float64)
        return cv2.undistort(frame, K, d)
