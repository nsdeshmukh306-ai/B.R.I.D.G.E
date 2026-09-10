"""Tray geography: where things live, and where a missing thing *should* be.

A count tells you a clamp is missing. A projector can do better than say so — it
can put an amber square on the empty spot the clamp came from. That needs a
memory of positions, which is what `TrayLayout` is: a snapshot of where each
counted item sat when the tray was laid out, taken at the initial count.

`Zone` is a named polygon in camera space (the tray itself, the neutral/hands-free
zone for sharps, the back table). Zones make spatial rules expressible: "a sharp
outside the neutral zone for more than 20 seconds" is a monitor rule, not a
special case in the renderer.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Iterable, Literal, Optional

from bridge.perception.scene_graph import SceneEntity, SceneGraph
from bridge.spatial.geometry import BoundingBox, Point, Polygon

log = logging.getLogger("bridge.surgical.tray")

ZoneKind = Literal["tray", "neutral", "back_table", "waste", "field", "custom"]


@dataclass
class Zone:
    name: str
    polygon: Polygon
    kind: ZoneKind = "custom"

    def contains_point(self, p: Point) -> bool:
        return self.polygon.contains(p)

    def contains(self, box: BoundingBox) -> bool:
        return self.polygon.contains(box.center)

    @property
    def center(self) -> Point:
        return self.polygon.centroid

    @classmethod
    def from_box(cls, name: str, box: BoundingBox, kind: ZoneKind = "custom") -> "Zone":
        return cls(name=name, polygon=Polygon(points=box.corners()), kind=kind)

    @classmethod
    def rect(cls, name: str, x: float, y: float, w: float, h: float, kind: ZoneKind = "custom") -> "Zone":
        return cls.from_box(name, BoundingBox(x=x, y=y, w=w, h=h), kind)


@dataclass
class Slot:
    """A remembered resting place for one instance of a counted item."""

    item: str
    position: Point
    radius: float = 70.0
    entity_id: str = ""
    recorded_ts: float = field(default_factory=time.time)

    def occupied_by(self, entities: Iterable[SceneEntity]) -> Optional[SceneEntity]:
        best, best_d = None, self.radius
        for e in entities:
            d = e.center.distance_to(self.position)
            if d <= best_d:
                best, best_d = e, d
        return best

    def outline(self, pad: float = 1.25) -> list[Point]:
        r = self.radius * pad
        p = self.position
        return [Point(x=p.x - r, y=p.y - r), Point(x=p.x + r, y=p.y - r),
                Point(x=p.x + r, y=p.y + r), Point(x=p.x - r, y=p.y + r)]


class TrayLayout:
    """Remembers where each counted item was at layout time, in camera pixels.

    Invalidated by calibration changes the same way everything else is: the
    positions are camera-space, so moving the camera makes them meaningless.
    Callers should `clear()` when calibration is invalidated.
    """

    def __init__(self) -> None:
        self.slots: list[Slot] = []
        self.zones: dict[str, Zone] = {}
        self.captured_ts: float = 0.0

    # -- zones ------------------------------------------------------------------------
    def set_zone(self, zone: Zone) -> None:
        self.zones[zone.name] = zone

    def zone(self, name: str) -> Optional[Zone]:
        return self.zones.get(name)

    def zone_of(self, entity: SceneEntity) -> Optional[Zone]:
        for z in self.zones.values():
            if z.contains(entity.bbox):
                return z
        return None

    def outside(self, entity: SceneEntity, zone_name: str) -> bool:
        z = self.zones.get(zone_name)
        return z is not None and not z.contains(entity.bbox)

    # -- layout capture ----------------------------------------------------------------
    def capture(self, scene: SceneGraph, name_for=None) -> int:
        """Snapshot the current positions of every visible entity as slots.

        `name_for(entity) -> str | None` maps an entity to the count-sheet item it
        belongs to; entities it returns None for are skipped.
        """
        self.slots.clear()
        for ent in scene.present():
            item = name_for(ent) if name_for is not None else (ent.label or None)
            if not item:
                continue
            self.slots.append(Slot(item=item, position=ent.center,
                                   radius=max(60.0, ent.bbox.radius * 1.5), entity_id=ent.id))
        self.captured_ts = time.time()
        log.info("Tray layout captured: %d slots", len(self.slots))
        return len(self.slots)

    def clear(self) -> None:
        self.slots.clear()
        self.captured_ts = 0.0

    @property
    def captured(self) -> bool:
        return bool(self.slots)

    # -- queries ------------------------------------------------------------------------
    def slots_for(self, item: str) -> list[Slot]:
        from bridge.perception.scene_graph import label_similarity

        return [s for s in self.slots if label_similarity(s.item, item) >= 0.5]

    def empty_slots(self, scene: SceneGraph, item: str = "") -> list[Slot]:
        """Slots whose object is no longer there — where to project the gap."""
        present = scene.present()
        pool = self.slots_for(item) if item else self.slots
        return [s for s in pool if s.occupied_by(present) is None]

    def occupied_slots(self, scene: SceneGraph, item: str = "") -> list[Slot]:
        present = scene.present()
        pool = self.slots_for(item) if item else self.slots
        return [s for s in pool if s.occupied_by(present) is not None]

    def item_names(self) -> list[str]:
        seen, out = set(), []
        for s in self.slots:
            if s.item not in seen:
                seen.add(s.item)
                out.append(s.item)
        return out


def infer_tray_zone(scene: SceneGraph, pad: float = 0.06) -> Optional[Zone]:
    """Guess the tray region as the bounding box of everything currently on the surface.

    Good enough to answer "is this item on the tray or on the field?" without
    asking the user to draw anything; the user can always set an explicit zone.
    """
    ents = scene.present()
    if len(ents) < 3:
        return None
    xs = [e.bbox.x for e in ents] + [e.bbox.x2 for e in ents]
    ys = [e.bbox.y for e in ents] + [e.bbox.y2 for e in ents]
    x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return Zone.rect("tray", x1 - w * pad, y1 - h * pad, w * (1 + 2 * pad), h * (1 + 2 * pad), kind="tray")
