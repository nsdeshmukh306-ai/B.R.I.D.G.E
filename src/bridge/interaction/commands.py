"""Controlled spatial command vocabulary. AI output is converted to one of these
(validated by Pydantic); nothing else can reach the renderer.
"""
from __future__ import annotations

from typing import Annotated, Literal, Optional, Union

from pydantic import BaseModel, Field, TypeAdapter, ValidationError

from bridge.ai.schemas import TargetIdentification
from bridge.spatial.geometry import BoundingBox


class CommandStyle(BaseModel):
    shape: Literal["circle", "outline", "point"] = "circle"
    animation: Literal["none", "pulse", "blink"] = "pulse"
    color: Optional[tuple[int, int, int]] = None


class TargetSpec(BaseModel):
    """What to find: a label, a description, and optionally camera-space boxes from the AI."""

    label: str = ""
    description: str = ""
    boxes_camera: list[BoundingBox] = Field(default_factory=list)
    all_instances: bool = False


class HighlightObject(BaseModel):
    command: Literal["highlight_object"] = "highlight_object"
    target: TargetSpec
    style: CommandStyle = Field(default_factory=CommandStyle)
    label_text: str = ""


class PointToObject(BaseModel):
    command: Literal["point_to_object"] = "point_to_object"
    target: TargetSpec
    style: CommandStyle = Field(default_factory=lambda: CommandStyle(shape="point", animation="pulse"))


class LabelObject(BaseModel):
    command: Literal["label_object"] = "label_object"
    target: TargetSpec
    text: str = ""


class ShowTargetZone(BaseModel):
    command: Literal["show_target_zone"] = "show_target_zone"
    target: TargetSpec  # the zone location (an object or region)
    text: str = "Place here"


class DrawPath(BaseModel):
    command: Literal["draw_path"] = "draw_path"
    source: TargetSpec
    destination: TargetSpec


class ShowMessage(BaseModel):
    command: Literal["show_message"] = "show_message"
    text: str
    duration_s: float = 5.0


class ClearProjection(BaseModel):
    command: Literal["clear_projection"] = "clear_projection"


Command = Annotated[
    Union[HighlightObject, PointToObject, LabelObject, ShowTargetZone, DrawPath, ShowMessage, ClearProjection],
    Field(discriminator="command"),
]
_adapter: TypeAdapter = TypeAdapter(Command)


def parse_command(data: dict) -> Command:  # type: ignore[valid-type]
    """Validate a raw dict (e.g. from JSON) into a Command. Raises ValueError if invalid."""
    try:
        return _adapter.validate_python(data)
    except ValidationError as e:
        raise ValueError(f"invalid command: {e.errors()[:2]}") from e


class CommandPlanner:
    """Converts a validated AI TargetIdentification into a Command."""

    def __init__(self, min_confidence: float = 0.5):
        self.min_confidence = min_confidence

    def plan(self, ident: TargetIdentification, frame_w: int, frame_h: int, label_override: str | None = None) -> Command:  # type: ignore[valid-type]
        cmd = self._plan(ident, frame_w, frame_h)
        if label_override and hasattr(cmd, "label_text"):
            cmd.label_text = label_override
        return cmd

    def _plan(self, ident: TargetIdentification, frame_w: int, frame_h: int) -> Command:  # type: ignore[valid-type]
        if ident.intent == "clear_projection":
            return ClearProjection()
        if ident.intent in ("show_message", "describe_scene", "unknown"):
            return ShowMessage(text=ident.message or "I did not understand that request.")
        if ident.confidence < self.min_confidence:
            msg = "Target uncertain.\nPlease clarify."
            if ident.message:
                msg += f"\n{ident.message}"
            return ShowMessage(text=msg)
        spec = TargetSpec(label=ident.target, description=ident.visual_description,
                          boxes_camera=[b.to_pixels(frame_w, frame_h) for b in ident.boxes],
                          all_instances=ident.intent == "find_all")
        style = CommandStyle(shape=ident.style, animation=ident.animation)
        if ident.intent == "point_to_object":
            return PointToObject(target=spec)
        if ident.intent == "label_object":
            return LabelObject(target=spec, text=ident.target)
        if ident.intent == "show_target_zone":
            return ShowTargetZone(target=spec, text=ident.message or "Place here")
        if ident.intent == "draw_path":
            dest = TargetSpec(label=ident.secondary_target, boxes_camera=[b.to_pixels(frame_w, frame_h) for b in ident.secondary_boxes])
            return DrawPath(source=spec, destination=dest)
        if ident.intent == "next_step":
            if not ident.target and not ident.boxes:
                return ShowMessage(text=ident.message or "Nothing to do next.")
            return HighlightObject(target=spec, style=CommandStyle(shape="circle", animation="pulse"), label_text=ident.message or ident.target)
        return HighlightObject(target=spec, style=style, label_text=ident.target if ident.intent in ("find_object", "highlight_object", "find_all") else "")
