"""Calibration profile persistence (JSON files under profiles/)."""
from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from bridge.spatial.geometry import Point
from bridge.spatial.homography import Matrix3x3
from bridge.spatial.validation import ValidationResult
from bridge.spatial.workspace import SurfaceType

log = logging.getLogger("bridge.profile")


class CalibrationProfile(BaseModel):
    name: str = "default"
    created_at: float = Field(default_factory=time.time)
    method: str = "planar_4point"
    camera_id: str
    camera_resolution: tuple[int, int]
    display_id: str
    display_resolution: tuple[int, int]
    surface_type: SurfaceType = "table"
    homography: Matrix3x3
    workspace_camera: list[Point] = Field(default_factory=list)
    workspace_projector: list[Point] = Field(default_factory=list)
    validation: ValidationResult
    mode: str = "physical"

    @property
    def validation_error(self) -> float:
        return self.validation.mean_error_px

    def matches_hardware(self, camera_id: str | None, camera_res: tuple[int, int] | None,
                         display_id: str | None, display_res: tuple[int, int] | None) -> tuple[bool, str]:
        """Return (ok, reason). Any hardware/resolution mismatch invalidates the profile."""
        if camera_id is not None and camera_id != self.camera_id:
            return False, "Camera changed. Calibration invalidated."
        if camera_res is not None and tuple(camera_res) != tuple(self.camera_resolution):
            return False, "Camera resolution changed. Calibration invalidated."
        if display_id is not None and display_id != self.display_id:
            return False, "Projector/display changed. Calibration invalidated."
        if display_res is not None and tuple(display_res) != tuple(self.display_resolution):
            return False, "Display resolution changed. Calibration invalidated."
        return True, "ok"


def _slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", name).strip("_") or "profile"


class ProfileStore:
    def __init__(self, directory: Path = Path("profiles")):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def path_for(self, name: str) -> Path:
        return self.dir / f"{_slug(name)}.json"

    def save(self, profile: CalibrationProfile) -> Path:
        p = self.path_for(profile.name)
        p.write_text(json.dumps(profile.model_dump(mode="json"), indent=2))
        log.info("Profile saved %s (error=%.2fpx)", p, profile.validation_error)
        return p

    def load(self, name: str) -> Optional[CalibrationProfile]:
        p = self.path_for(name)
        if not p.exists():
            return None
        try:
            return CalibrationProfile.model_validate_json(p.read_text())
        except Exception:  # noqa: BLE001
            log.exception("Corrupt profile %s", p)
            return None

    def list(self) -> list[CalibrationProfile]:
        out = []
        for p in sorted(self.dir.glob("*.json")):
            try:
                out.append(CalibrationProfile.model_validate_json(p.read_text()))
            except Exception:  # noqa: BLE001
                log.warning("Skipping unreadable profile %s", p)
        return out

    def delete(self, name: str) -> bool:
        p = self.path_for(name)
        if p.exists():
            p.unlink()
            return True
        return False

    def find_matching(self, camera_id: str | None, camera_res: tuple[int, int] | None,
                      display_id: str | None, display_res: tuple[int, int] | None) -> Optional[CalibrationProfile]:
        best: Optional[CalibrationProfile] = None
        for prof in self.list():
            ok, _ = prof.matches_hardware(camera_id, camera_res, display_id, display_res)
            trusted = prof.validation.valid and prof.validation.independent and prof.validation.n_points >= 4
            if ok and trusted and (best is None or prof.created_at > best.created_at):
                best = prof
        return best
