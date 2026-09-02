"""Core geometric types. Physical-world coordinates are first-class data."""
from __future__ import annotations

import math
from typing import Iterable, Sequence

from pydantic import BaseModel, Field, field_validator


class Point(BaseModel):
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)

    def as_int(self) -> tuple[int, int]:
        return (int(round(self.x)), int(round(self.y)))

    def distance_to(self, other: "Point") -> float:
        return math.hypot(self.x - other.x, self.y - other.y)

    def __add__(self, other: "Point") -> "Point":
        return Point(x=self.x + other.x, y=self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(x=self.x - other.x, y=self.y - other.y)

    def scale(self, sx: float, sy: float | None = None) -> "Point":
        sy = sx if sy is None else sy
        return Point(x=self.x * sx, y=self.y * sy)


class BoundingBox(BaseModel):
    """Axis-aligned box in pixel coordinates (x, y = top-left)."""

    x: float
    y: float
    w: float = Field(ge=0)
    h: float = Field(ge=0)

    @property
    def center(self) -> Point:
        return Point(x=self.x + self.w / 2, y=self.y + self.h / 2)

    @property
    def x2(self) -> float:
        return self.x + self.w

    @property
    def y2(self) -> float:
        return self.y + self.h

    @property
    def area(self) -> float:
        return self.w * self.h

    @property
    def radius(self) -> float:
        """Radius of a circle that comfortably encloses the box."""
        return math.hypot(self.w, self.h) / 2

    def corners(self) -> list[Point]:
        return [
            Point(x=self.x, y=self.y),
            Point(x=self.x2, y=self.y),
            Point(x=self.x2, y=self.y2),
            Point(x=self.x, y=self.y2),
        ]

    def as_xywh_int(self) -> tuple[int, int, int, int]:
        return (int(round(self.x)), int(round(self.y)), int(round(self.w)), int(round(self.h)))

    def iou(self, other: "BoundingBox") -> float:
        ix1, iy1 = max(self.x, other.x), max(self.y, other.y)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def clip(self, width: float, height: float) -> "BoundingBox":
        x1 = min(max(self.x, 0.0), width)
        y1 = min(max(self.y, 0.0), height)
        x2 = min(max(self.x2, 0.0), width)
        y2 = min(max(self.y2, 0.0), height)
        return BoundingBox(x=x1, y=y1, w=max(0.0, x2 - x1), h=max(0.0, y2 - y1))

    @classmethod
    def from_xyxy(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        return cls(x=min(x1, x2), y=min(y1, y2), w=abs(x2 - x1), h=abs(y2 - y1))

    @classmethod
    def from_normalized(
        cls, box: Sequence[float], width: int, height: int, order: str = "ymin_xmin_ymax_xmax", scale: float = 1.0
    ) -> "BoundingBox":
        """Convert a normalized box to pixel coordinates.

        Gemini returns boxes as [ymin, xmin, ymax, xmax] on a 0..1000 scale.
        `scale` is the normalization range (1.0 or 1000.0).
        """
        if len(box) != 4:
            raise ValueError("box must have 4 values")
        vals = [float(v) / scale for v in box]
        if order == "ymin_xmin_ymax_xmax":
            y1, x1, y2, x2 = vals
        elif order == "xmin_ymin_xmax_ymax":
            x1, y1, x2, y2 = vals
        else:
            raise ValueError(f"unknown box order {order}")
        return cls.from_xyxy(x1 * width, y1 * height, x2 * width, y2 * height).clip(width, height)


class Polygon(BaseModel):
    points: list[Point]

    @field_validator("points")
    @classmethod
    def _at_least_three(cls, v: list[Point]) -> list[Point]:
        if len(v) < 3:
            raise ValueError("polygon needs at least 3 points")
        return v

    @property
    def centroid(self) -> Point:
        n = len(self.points)
        return Point(x=sum(p.x for p in self.points) / n, y=sum(p.y for p in self.points) / n)

    def bounding_box(self) -> BoundingBox:
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return BoundingBox.from_xyxy(min(xs), min(ys), max(xs), max(ys))

    def contains(self, p: Point) -> bool:
        inside = False
        pts = self.points
        j = len(pts) - 1
        for i in range(len(pts)):
            xi, yi, xj, yj = pts[i].x, pts[i].y, pts[j].x, pts[j].y
            if (yi > p.y) != (yj > p.y) and p.x < (xj - xi) * (p.y - yi) / (yj - yi + 1e-12) + xi:
                inside = not inside
            j = i
        return inside


def points_to_array(points: Iterable[Point]):
    import numpy as np

    return np.array([[p.x, p.y] for p in points], dtype=np.float64)


def array_to_points(arr) -> list[Point]:
    return [Point(x=float(r[0]), y=float(r[1])) for r in arr]
