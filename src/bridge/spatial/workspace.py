"""Physical workspace model (planar in V1)."""
from __future__ import annotations

import uuid
from typing import Literal, Optional

from pydantic import BaseModel, Field

from bridge.spatial.geometry import Point
from bridge.spatial.homography import Matrix3x3

SurfaceType = Literal["table", "wall", "floor", "custom"]

SURFACE_PRESETS: dict[str, dict] = {
    "table": {"label": "Table", "description": "Horizontal planar surface, projector above/angled", "orientation": "horizontal"},
    "wall": {"label": "Wall", "description": "Vertical planar surface, projector facing it", "orientation": "vertical"},
    "floor": {"label": "Floor", "description": "Horizontal planar surface, projector overhead", "orientation": "horizontal"},
    "custom": {"label": "Custom", "description": "Any planar surface", "orientation": "unknown"},
}


class Workspace(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str = "Workspace"
    surface_type: SurfaceType = "table"
    polygon_camera: list[Point] = Field(default_factory=list)
    polygon_projector: list[Point] = Field(default_factory=list)
    homography: Optional[Matrix3x3] = None

    @property
    def is_defined(self) -> bool:
        return self.homography is not None and len(self.polygon_projector) >= 3
