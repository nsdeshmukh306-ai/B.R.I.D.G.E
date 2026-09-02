"""Pydantic schemas for structured AI responses. Every AI output is validated here.

The model never returns code; it returns a controlled vocabulary of intents
and (optionally) normalized bounding boxes. BRIDGE executes the action.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

from bridge.spatial.geometry import BoundingBox

Intent = Literal[
    "find_object",        # locate and highlight one object
    "find_all",           # locate every instance of a label
    "highlight_object",
    "point_to_object",
    "label_object",
    "show_target_zone",   # where to put something
    "draw_path",          # movement from A to B
    "show_message",
    "clear_projection",
    "describe_scene",
    "next_step",          # assembly guidance: what to pick up first / next
    "unknown",
]

Shape = Literal["circle", "outline", "point"]
Animation = Literal["none", "pulse", "blink"]


class NormalizedBox(BaseModel):
    """[ymin, xmin, ymax, xmax] on a 0..1000 scale, as Gemini produces."""

    box_2d: list[float] = Field(min_length=4, max_length=4)
    label: str = ""

    @field_validator("box_2d")
    @classmethod
    def _range(cls, v: list[float]) -> list[float]:
        if any(x < 0 or x > 1000 for x in v):
            raise ValueError("box_2d values must be within 0..1000")
        y1, x1, y2, x2 = v
        if y2 < y1 or x2 < x1:
            raise ValueError("box_2d must be [ymin, xmin, ymax, xmax]")
        return v

    def to_pixels(self, width: int, height: int) -> BoundingBox:
        return BoundingBox.from_normalized(self.box_2d, width, height, "ymin_xmin_ymax_xmax", scale=1000.0)


class SceneObject(BaseModel):
    label: str
    visual_description: str = ""
    box: Optional[NormalizedBox] = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class SceneUnderstanding(BaseModel):
    objects: list[SceneObject] = Field(default_factory=list)
    summary: str = ""


class TargetIdentification(BaseModel):
    """Answer to: which object in the image does the user mean?"""

    intent: Intent = "find_object"
    target: str = ""
    visual_description: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    boxes: list[NormalizedBox] = Field(default_factory=list)
    message: str = ""  # a short human-readable message, e.g. reason for uncertainty
    style: Shape = "circle"
    animation: Animation = "pulse"
    secondary_target: str = ""  # e.g. destination for draw_path / show_target_zone
    secondary_boxes: list[NormalizedBox] = Field(default_factory=list)


class ActionPlan(BaseModel):
    """Assembly / task guidance: next thing to do."""

    instruction: str
    target: str
    visual_description: str = ""
    boxes: list[NormalizedBox] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False
