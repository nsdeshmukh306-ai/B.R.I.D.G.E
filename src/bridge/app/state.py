"""Central application state (observable through the event bus)."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class CalibrationStatus(str, Enum):
    NOT_CALIBRATED = "Not calibrated"
    CALIBRATING = "Calibrating"
    VALID = "VALID"
    INVALID = "INVALID"


class AppPhase(str, Enum):
    WELCOME = "welcome"
    SETUP = "setup"
    CALIBRATION_REQUIRED = "calibration_required"
    READY = "ready"


@dataclass
class DiagnosticsSnapshot:
    camera_connected: bool = False
    camera_name: str = "--"
    camera_resolution: str = "--"
    camera_fps: float = 0.0
    display_name: str = "--"
    display_resolution: str = "--"
    calibration: str = CalibrationStatus.NOT_CALIBRATED.value
    calibration_mean_error: Optional[float] = None
    tracking_target: str = "--"
    tracking_confidence: Optional[float] = None
    ai_provider: str = "--"
    ai_status: str = "--"
    ai_last_request_ts: Optional[float] = None
    ai_last_latency_s: Optional[float] = None

    def ai_seconds_since(self) -> Optional[float]:
        if self.ai_last_request_ts is None:
            return None
        return time.time() - self.ai_last_request_ts


@dataclass
class StateManager:
    phase: AppPhase = AppPhase.WELCOME
    mode: str = "physical"
    surface_type: str = "table"
    calibration_status: CalibrationStatus = CalibrationStatus.NOT_CALIBRATED
    diagnostics: DiagnosticsSnapshot = field(default_factory=DiagnosticsSnapshot)
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def set_calibration(self, status: CalibrationStatus, mean_error: float | None = None) -> None:
        with self._lock:
            self.calibration_status = status
            self.diagnostics.calibration = status.value
            self.diagnostics.calibration_mean_error = mean_error
            self.phase = AppPhase.READY if status is CalibrationStatus.VALID else AppPhase.CALIBRATION_REQUIRED

    @property
    def is_ready(self) -> bool:
        return self.calibration_status is CalibrationStatus.VALID
