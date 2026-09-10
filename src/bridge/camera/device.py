"""Camera abstraction. No vendor-specific code lives here."""
from __future__ import annotations

import logging
import platform
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger("bridge.camera")


@dataclass
class CameraInfo:
    id: str
    name: str
    index: int
    width: int = 0
    height: int = 0
    fps: float = 0.0
    backend: str = "opencv"


class FrameSource(ABC):
    """Anything that yields BGR frames (real camera or simulator)."""

    @abstractmethod
    def read_frame(self) -> Optional[np.ndarray]: ...

    @property
    @abstractmethod
    def resolution(self) -> tuple[int, int]: ...

    @property
    def is_open(self) -> bool:
        return True

    def release(self) -> None:
        return None


class CameraDevice(FrameSource):
    """A camera opened through OpenCV's VideoCapture."""

    def __init__(self, info: CameraInfo, width: int | None = None, height: int | None = None, fps: float | None = None,
                 auto_exposure: bool = True, exposure: float | None = None):
        self.info = info
        self.auto_exposure, self.exposure = auto_exposure, exposure
        self.id = info.id
        self.name = info.name
        self.width = width or info.width or 1280
        self.height = height or info.height or 720
        self.fps = fps or info.fps or 30.0
        self._cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()
        self._last_frame_ts = 0.0
        self.failed_reads = 0

    @staticmethod
    def _preferred_backend() -> int:
        sysname = platform.system()
        if sysname == "Windows":
            return cv2.CAP_DSHOW
        if sysname == "Darwin":
            return cv2.CAP_AVFOUNDATION
        return cv2.CAP_V4L2

    def open(self) -> bool:
        with self._lock:
            cap = cv2.VideoCapture(self.info.index, self._preferred_backend())
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(self.info.index)
            if not cap.isOpened():
                log.error("Camera %s (%s) could not be opened", self.name, self.info.index)
                self._cap = None
                return False
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.fps)
            # Latency: never queue frames behind the consumer (Windows backends buffer several by default).
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            try:
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))  # MJPG keeps 30 fps at 720p on USB2 webcams
            except Exception:  # noqa: BLE001
                pass
            if not self.auto_exposure:
                # 0.25 = manual on V4L2/most DSHOW drivers, 0.75 = auto. A fixed exposure keeps the
                # frame rate up in a dark room (auto-exposure otherwise drops to ~8 fps) and stops
                # the projected graphics from re-triggering exposure changes.
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
                if self.exposure is not None:
                    cap.set(cv2.CAP_PROP_EXPOSURE, float(self.exposure))
            else:
                cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
            self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or self.width
            self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or self.height
            self.fps = float(cap.get(cv2.CAP_PROP_FPS)) or self.fps
            self._cap = cap
            log.info("Camera connected name=%s res=%dx%d fps=%.1f", self.name, self.width, self.height, self.fps)
            return True

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def resolution(self) -> tuple[int, int]:
        return (self.width, self.height)

    def set_resolution(self, width: int, height: int) -> tuple[int, int]:
        with self._lock:
            if self._cap is not None:
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or width
                self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or height
            else:
                self.width, self.height = width, height
        return (self.width, self.height)

    def read_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._cap is None:
                return None
            ok, frame = self._cap.read()
        if not ok or frame is None:
            self.failed_reads += 1
            if self.failed_reads in (1, 30, 300):
                log.warning("Camera %s returned no frame (%d failures)", self.name, self.failed_reads)
            return None
        self.failed_reads = 0
        self._last_frame_ts = time.time()
        return frame

    def release(self) -> None:
        with self._lock:
            if self._cap is not None:
                self._cap.release()
                self._cap = None
                log.info("Camera released name=%s", self.name)
