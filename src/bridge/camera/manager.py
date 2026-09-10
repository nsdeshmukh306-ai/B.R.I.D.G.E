"""Camera enumeration and lifecycle management."""
from __future__ import annotations

import logging
import os
import platform
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np

from bridge.app.diagnostics import FPSCounter
from bridge.app.events import EventBus, Topic
from bridge.camera.device import CameraDevice, CameraInfo, FrameSource

log = logging.getLogger("bridge.camera")

COMMON_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]


def _linux_camera_names() -> dict[int, str]:
    names: dict[int, str] = {}
    base = "/sys/class/video4linux"
    if not os.path.isdir(base):
        return names
    for entry in os.listdir(base):
        if not entry.startswith("video"):
            continue
        try:
            idx = int(entry[5:])
            with open(os.path.join(base, entry, "name")) as f:
                names[idx] = f.read().strip()
        except (ValueError, OSError):
            continue
    return names


def enumerate_cameras(max_index: int = 8, probe_timeout_s: float = 2.0) -> list[CameraInfo]:
    """Probe camera indexes 0..max_index-1. Returns only cameras that open."""
    found: list[CameraInfo] = []
    sysname = platform.system()
    linux_names = _linux_camera_names() if sysname == "Linux" else {}
    if sysname == "Linux" and not linux_names:
        return found  # no /dev/video* devices: skip probing entirely
    candidates = sorted(linux_names) if linux_names else range(max_index)
    prev = os.environ.get("OPENCV_LOG_LEVEL")
    os.environ["OPENCV_LOG_LEVEL"] = "OFF"
    try:
        for idx in candidates:
            start = time.time()
            cap = cv2.VideoCapture(idx)
            ok = cap.isOpened()
            if ok:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = float(cap.get(cv2.CAP_PROP_FPS))
                # Some virtual/metadata nodes open but never deliver frames.
                grabbed = cap.grab() if (time.time() - start) < probe_timeout_s else False
                cap.release()
                if not grabbed:
                    continue
                name = linux_names.get(idx, f"Camera {idx}")
                found.append(CameraInfo(id=f"cam:{idx}:{name}", name=name, index=idx, width=w, height=h, fps=fps))
            else:
                cap.release()
                if not linux_names and idx > 1 and not found:
                    break  # stop scanning early when nothing found on 0/1
    finally:
        if prev is None:
            os.environ.pop("OPENCV_LOG_LEVEL", None)
        else:
            os.environ["OPENCV_LOG_LEVEL"] = prev
    log.info("Enumerated %d camera(s)", len(found))
    return found


class CameraManager:
    """Owns the active FrameSource and a background capture thread."""

    def __init__(self, bus: EventBus):
        self.bus = bus
        self.available: list[CameraInfo] = []
        self.active: Optional[FrameSource] = None
        self.active_id: Optional[str] = None
        self._thread: Optional[threading.Thread] = None
        self._running = threading.Event()
        self._latest: Optional[np.ndarray] = None
        self._latest_lock = threading.Lock()
        self.frame_seq = 0
        self.fps_counter = FPSCounter()
        self._frame_listeners: list[Callable[[np.ndarray], None]] = []

    def refresh(self) -> list[CameraInfo]:
        try:
            self.available = enumerate_cameras()
        except Exception:  # noqa: BLE001
            log.exception("Camera enumeration failed")
            self.available = []
        return self.available

    def open(self, info: CameraInfo, width: int | None = None, height: int | None = None, fps: float | None = None,
             auto_exposure: bool = True, exposure: float | None = None) -> bool:
        self.close()
        dev = CameraDevice(info, width, height, fps, auto_exposure, exposure)
        if not dev.open():
            self.bus.publish(Topic.STATUS_MESSAGE, text=f"Camera '{info.name}' unavailable", level="error")
            return False
        self._start_source(dev, info.id)
        return True

    def use_source(self, source: FrameSource, source_id: str) -> None:
        """Attach a non-hardware frame source (simulation)."""
        self.close()
        self._start_source(source, source_id)

    def _start_source(self, source: FrameSource, source_id: str) -> None:
        self.active = source
        self.active_id = source_id
        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()
        w, h = source.resolution
        self.bus.publish(Topic.CAMERA_CONNECTED, source_id=source_id, width=w, height=h)

    def _loop(self) -> None:
        assert self.active is not None
        while self._running.is_set():
            frame = self.active.read_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            with self._latest_lock:
                self._latest = frame
                self.frame_seq += 1
            self.fps_counter.tick()
            for fn in list(self._frame_listeners):
                try:
                    fn(frame)
                except Exception:  # noqa: BLE001
                    log.exception("frame listener failed")

    def add_frame_listener(self, fn: Callable[[np.ndarray], None]) -> Callable[[], None]:
        self._frame_listeners.append(fn)
        return lambda: self._frame_listeners.remove(fn) if fn in self._frame_listeners else None

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._latest_lock:
            return None if self._latest is None else self._latest.copy()

    def wait_for_frame(self, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            f = self.latest_frame()
            if f is not None:
                return f
            time.sleep(0.02)
        return None

    def wait_for_fresh_frame(self, after_seq: int, skip: int = 2, timeout_s: float = 2.0) -> Optional[np.ndarray]:
        """Return a frame captured at least `skip` frames after `after_seq` (i.e. after a change)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._latest_lock:
                if self._latest is not None and self.frame_seq >= after_seq + skip:
                    return self._latest.copy()
            time.sleep(0.005)
        return self.latest_frame()

    @property
    def resolution(self) -> tuple[int, int]:
        return self.active.resolution if self.active else (0, 0)

    @property
    def is_open(self) -> bool:
        return self.active is not None and self.active.is_open

    def close(self) -> None:
        self._running.clear()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None
        if self.active is not None:
            self.active.release()
            self.bus.publish(Topic.CAMERA_DISCONNECTED, source_id=self.active_id)
        self.active = None
        self.active_id = None
        with self._latest_lock:
            self._latest = None
