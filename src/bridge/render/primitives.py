"""Spatial rendering primitives. All coordinates are PROJECTOR pixels."""
from __future__ import annotations

import uuid
from typing import Literal, Optional, Union

from pydantic import BaseModel, Field

from bridge.spatial.geometry import Point

RGB = tuple[int, int, int]


ACCENT: RGB = (0, 220, 200)      # teal-cyan: primary highlight on the black canvas
SUCCESS: RGB = (90, 230, 140)    # target zones / confirmations
WARNING: RGB = (255, 180, 60)    # caution / uncertainty
ALERT: RGB = (255, 90, 90)       # lost target / stop
TEXT: RGB = (240, 244, 248)      # message text on black


class Style(BaseModel):
    color: RGB = ACCENT
    thickness: int = 3
    dashed: bool = False
    fill: bool = False
    opacity: float = Field(default=1.0, ge=0.0, le=1.0)
    animation: Literal["none", "pulse", "flow", "blink"] = "none"
    glow: bool = True                          # soft additive halo (dark backgrounds only)
    variant: Literal["auto", "ring", "reticle", "bracket"] = "auto"


class _Base(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:10])
    style: Style = Field(default_factory=Style)
    group: Optional[str] = None  # e.g. object id, so a group can be moved/cleared together


class Circle(_Base):
    kind: Literal["circle"] = "circle"
    center: Point
    radius: float = Field(gt=0)


class Outline(_Base):
    kind: Literal["outline"] = "outline"
    points: list[Point]


class Arrow(_Base):
    kind: Literal["arrow"] = "arrow"
    start: Point
    end: Point


class Dot(_Base):
    kind: Literal["point"] = "point"
    center: Point
    radius: float = 8


class Label(_Base):
    kind: Literal["label"] = "label"
    position: Point
    text: str
    font_scale: float = 0.9
    background: bool = True


class TargetZone(_Base):
    kind: Literal["target_zone"] = "target_zone"
    points: list[Point]  # polygon


class Path(_Base):
    kind: Literal["path"] = "path"
    points: list[Point]


class Message(_Base):
    kind: Literal["message"] = "message"
    text: str
    position: Optional[Point] = None  # None = centered


Primitive = Union[Circle, Outline, Arrow, Dot, Label, TargetZone, Path, Message]
