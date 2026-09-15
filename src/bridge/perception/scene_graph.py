"""SceneGraph: BRIDGE's persistent model of what is on the surface.

Why this exists
---------------
Without it every "project the scalpel" costs a Gemini round-trip (0.6-2 s).
With it, the AI is asked *occasionally* what things are (slow, semantic) while
OpenCV answers *where they are* on every frame (fast, geometric). A spoken
command then resolves out of local memory in microseconds and the reticle
appears immediately.

Design
------
* An **entity** is a thing BRIDGE believes is on the surface: a stable id, a
  label, aliases, the last camera-space box, when it was last seen, and whether
  it is present right now.
* Every camera frame contributes *observations* (contour detections). They are
  associated to existing entities by overlap + proximity + colour, so an entity
  survives being nudged, briefly occluded by a hand, or re-detected slightly
  differently.
* Every AI scene pass contributes *labels*. Labels are attached to whichever
  entity the AI box lands on, so semantic identity accumulates on top of the
  geometric identity instead of replacing it.
* Presence decays: an entity not observed for `absent_after_s` becomes absent
  (but is remembered, which is exactly what "what's missing?" needs).

Everything here is camera-space pixels and pure Python/NumPy: no Qt, no AI, no
rendering. That makes it trivially testable against the simulator.
"""
from __future__ import annotations

import itertools
import logging
import math
import re
import threading
import time
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np

from bridge.spatial.geometry import BoundingBox, Point
from bridge.vision.detection import Detection, describe_color

log = logging.getLogger("bridge.scene")

_ids = itertools.count(1)

# Words that carry no identity on their own; ignored when matching a spoken label.
_STOP = {"the", "a", "an", "that", "this", "one", "please", "my", "your", "of", "on", "in", "to", "for"}

# Nouns that are already plural in form and must not be "singularised" — half the
# instrument tray is one of these ("pass me the forcep" is not a sentence).
_INVARIANT = {"scissors", "forceps", "pliers", "tongs", "glasses", "gauze",
              "sharps", "series", "species", "bs"}


def normalise_label(text: str) -> str:
    """'the 15 blade scalpel!' -> '15 blade scalpel'. Singularises trailing plurals."""
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower())
    words = [w for w in t.split() if w and w not in _STOP]
    out = []
    for w in words:
        if w in _INVARIANT:
            pass
        elif len(w) > 3 and w.endswith("ies"):
            w = w[:-3] + "y"
        elif len(w) > 3 and w.endswith("ses"):
            w = w[:-2]
        elif len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
            w = w[:-1]
        out.append(w)
    return " ".join(out)


def label_similarity(a: str, b: str) -> float:
    """0..1 overlap of two normalised labels. 'artery clamp' vs 'clamp' -> 0.5."""
    wa, wb = set(normalise_label(a).split()), set(normalise_label(b).split())
    if not wa or not wb:
        return 0.0
    if wa == wb:
        return 1.0
    inter = wa & wb
    if not inter:
        return 0.0
    return len(inter) / max(len(wa), len(wb))


def label_refines(spoken: str, name: str) -> bool:
    """True when one label is the other with extra qualifying words.

    'clamp' refines 'artery clamp' (a shorter way of saying the same thing) and
    'curved artery clamp' refines 'artery clamp' (the same thing, said more
    precisely). 'bulldog clamp' does **not** refine 'artery clamp' — they share a
    noun but disagree on the qualifier, and treating them as the same line is how
    a count goes wrong.

    The head noun (last word) must agree, or a one-word name would swallow every
    phrase containing it: 'suture scissors' is not a kind of 'suture'.
    """
    a, b = normalise_label(spoken).split(), normalise_label(name).split()
    if not a or not b or a[-1] != b[-1]:
        return False
    return set(a) <= set(b) or set(b) <= set(a)


@dataclass
class SceneEntity:
    """One physical thing BRIDGE believes is (or was) on the surface."""

    id: str
    bbox: BoundingBox
    label: str = ""
    aliases: set[str] = field(default_factory=set)
    color: str = ""
    confidence: float = 0.5
    label_confidence: float = 0.0
    source: str = "local"            # local | ai | specialist-local | ai+local | specialist-local+local | manual
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    last_moved: float = field(default_factory=time.time)
    present: bool = True
    observations: int = 1
    sticky: bool = False             # never forgotten (e.g. a counted instrument)
    tags: set[str] = field(default_factory=set)   # "sharp", "sponge", "instrument", ...

    @property
    def center(self) -> Point:
        return self.bbox.center

    @property
    def age_s(self) -> float:
        return time.time() - self.first_seen

    def seconds_since_seen(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.last_seen

    def seconds_still(self, now: float | None = None) -> float:
        return (now if now is not None else time.time()) - self.last_moved

    def matches(self, text: str, threshold: float = 0.5) -> float:
        """How well a spoken phrase names this entity (0..1)."""
        best = label_similarity(text, self.label)
        for a in self.aliases:
            best = max(best, label_similarity(text, a))
        if self.color and normalise_label(self.color) in normalise_label(text).split():
            best = max(best, best + 0.15 if best else 0.35)
        return best if best >= threshold else 0.0

    def describe(self) -> str:
        name = self.label or self.color or "object"
        return f"{name}" if not self.color or self.color in name else f"{self.color} {name}"

    def to_detection(self) -> Detection:
        return Detection.from_bbox(self.label or "object", self.bbox, self.confidence, f"scene:{self.source}",
                                   description=self.describe())


@dataclass
class SceneDelta:
    """What changed in one update — the input to the proactive monitor."""

    appeared: list[SceneEntity] = field(default_factory=list)
    disappeared: list[SceneEntity] = field(default_factory=list)
    moved: list[SceneEntity] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    @property
    def any(self) -> bool:
        return bool(self.appeared or self.disappeared or self.moved)


class SceneGraph:
    """Thread-safe registry of scene entities.

    `observe()` is called from the camera thread on every frame; `apply_ai()`
    from an AI worker every few seconds; `find()` from the voice thread.
    """

    def __init__(self, absent_after_s: float = 1.2, forget_after_s: float = 900.0,
                 move_threshold_px: float = 12.0, match_iou: float = 0.25,
                 max_entities: int = 120):
        self.absent_after_s = absent_after_s
        self.forget_after_s = forget_after_s
        self.move_threshold_px = move_threshold_px
        self.match_iou = match_iou
        self.max_entities = max_entities
        self._entities: dict[str, SceneEntity] = {}
        self._lock = threading.RLock()
        self.version = 0
        self.frame_size: tuple[int, int] = (0, 0)
        self.last_ai_ts: float = 0.0

    # -- read ---------------------------------------------------------------------------
    def __len__(self) -> int:
        with self._lock:
            return len(self._entities)

    @property
    def entities(self) -> list[SceneEntity]:
        with self._lock:
            return list(self._entities.values())

    def present(self) -> list[SceneEntity]:
        with self._lock:
            return [e for e in self._entities.values() if e.present]

    def absent(self) -> list[SceneEntity]:
        with self._lock:
            return [e for e in self._entities.values() if not e.present]

    def get(self, entity_id: str) -> Optional[SceneEntity]:
        with self._lock:
            return self._entities.get(entity_id)

    def labelled(self) -> list[SceneEntity]:
        with self._lock:
            return [e for e in self._entities.values() if e.label]

    def unresolved_label_ids(self, min_confidence: float = 0.5) -> set[str]:
        """IDs of *present* entities that are unlabelled, or labelled with low confidence.

        Local contour detection is noisy (a shadow, a tray edge, a wrinkle in gauze
        can all produce a stable-looking entity that is not a real countable object),
        so some entities may never earn a confident label no matter how many times a
        vision pass looks at them. The caller (JarvisAssistant) is expected to compare
        this set to the one from its *previous* check: a call is worth spending only
        when this set contains an id that was not already unresolved last time —
        otherwise it is the same handful of not-a-real-object blobs asking to be told
        the same "no" again.
        """
        with self._lock:
            return {e.id for e in self._entities.values()
                   if e.present and (not e.label or e.label_confidence < min_confidence)}

    def needs_ai_label(self, min_confidence: float = 0.5) -> bool:
        """True when at least one present entity is unlabelled or low-confidence.

        A simple existence check — useful on its own, but JarvisAssistant's background
        loop uses `unresolved_label_ids` instead so it can tell a *new* unresolved
        entity apart from one that was already asked about and stayed unresolved.
        """
        return bool(self.unresolved_label_ids(min_confidence))

    def find(self, text: str, present_only: bool = True, limit: int = 0,
             threshold: float = 0.5) -> list[SceneEntity]:
        """Resolve a spoken phrase to entities, best first. No AI, no network."""
        with self._lock:
            scored = []
            for e in self._entities.values():
                if present_only and not e.present:
                    continue
                s = e.matches(text, threshold)
                if s > 0:
                    # prefer confidently labelled, recently seen things
                    scored.append((s + 0.1 * e.label_confidence - 0.02 * min(e.seconds_since_seen(), 5), e))
            scored.sort(key=lambda p: -p[0])
            out = [e for _, e in scored]
            return out[:limit] if limit else out

    def find_one(self, text: str, present_only: bool = True) -> Optional[SceneEntity]:
        hits = self.find(text, present_only, limit=1)
        return hits[0] if hits else None

    def counts_by_label(self, present_only: bool = True) -> dict[str, int]:
        out: dict[str, int] = {}
        with self._lock:
            for e in self._entities.values():
                if present_only and not e.present:
                    continue
                key = normalise_label(e.label) or "unidentified"
                out[key] = out.get(key, 0) + 1
        return out

    def summary(self, limit: int = 8) -> str:
        c = self.counts_by_label()
        if not c:
            return "I can't see anything on the surface yet."
        parts = [f"{n} {k}" if n > 1 else k for k, n in sorted(c.items(), key=lambda p: -p[1])[:limit]]
        return ", ".join(parts)

    # -- write --------------------------------------------------------------------------
    def observe(self, detections: Iterable[Detection], frame: np.ndarray | None = None,
                now: float | None = None) -> SceneDelta:
        """Fold one frame's local detections into the graph. Cheap; called per frame."""
        now = now if now is not None else time.time()
        dets = [d for d in detections]
        if frame is not None:
            self.frame_size = (frame.shape[1], frame.shape[0])
        delta = SceneDelta(ts=now)
        with self._lock:
            unmatched = dict(self._entities)
            for d in dets:
                ent = self._best_match(d, unmatched)
                if ent is None:
                    ent = self._add(d, frame, now)
                    delta.appeared.append(ent)
                else:
                    unmatched.pop(ent.id, None)
                    if ent.center.distance_to(d.bbox.center) > self.move_threshold_px:
                        ent.last_moved = now
                        delta.moved.append(ent)
                    ent.bbox = d.bbox
                    ent.last_seen = now
                    ent.observations += 1
                    ent.confidence = max(ent.confidence, d.confidence)
                    if not ent.present:
                        ent.present = True
                        delta.appeared.append(ent)
                    if frame is not None and ent.observations % 15 == 0:
                        ent.color = self._colour_of(frame, ent.bbox) or ent.color
            # presence decay for everything not seen this frame
            for ent in list(unmatched.values()):
                if ent.present and ent.seconds_since_seen(now) > self.absent_after_s:
                    ent.present = False
                    delta.disappeared.append(ent)
                if (not ent.sticky and not ent.present
                        and ent.seconds_since_seen(now) > self.forget_after_s):
                    self._entities.pop(ent.id, None)
            self._prune()
            if delta.any:
                self.version += 1
        return delta

    def apply_ai(self, labelled_boxes: Iterable[tuple[str, BoundingBox, float, str]],
                 now: float | None = None, source: str = "ai") -> int:
        """Attach model-supplied labels to existing entities (or create them).

        `labelled_boxes` items are (label, bbox_camera, confidence, description).
        `source` records which labeller this came from ("ai" for Gemini,
        "specialist-local" for the local instrument recognizer, etc.) — it is
        advisory metadata only; the merge and matching logic is identical for
        every source, and local CV still owns position regardless of who
        supplied the label. Returns the number of entities labelled.
        """
        now = now if now is not None else time.time()
        n = 0
        with self._lock:
            self.last_ai_ts = now
            for label, box, conf, desc in labelled_boxes:
                if not label:
                    continue
                ent = self._best_match(Detection.from_bbox(label, box, conf, source), self._entities)
                if ent is None:
                    ent = SceneEntity(id=f"e{next(_ids)}", bbox=box, label=label, confidence=conf,
                                      label_confidence=conf, source=source, first_seen=now, last_seen=now,
                                      last_moved=now)
                    self._entities[ent.id] = ent
                else:
                    if ent.label and label_similarity(ent.label, label) < 0.5:
                        ent.aliases.add(ent.label)
                    if conf >= ent.label_confidence:
                        ent.label, ent.label_confidence = label, conf
                    else:
                        ent.aliases.add(label)
                    ent.source = f"{source}+local" if ent.source == "local" else ent.source
                if desc:
                    ent.aliases.add(desc[:60])
                n += 1
            self._prune()
            self.version += 1
        return n

    def name_entity(self, entity_id: str, label: str, tags: Iterable[str] = (), sticky: bool = False) -> bool:
        """Manual naming (from a count sheet or the user: 'call that the mayo scissors')."""
        with self._lock:
            ent = self._entities.get(entity_id)
            if ent is None:
                return False
            if ent.label and label_similarity(ent.label, label) < 0.6:
                ent.aliases.add(ent.label)
            ent.label, ent.label_confidence, ent.source = label, 1.0, "manual"
            ent.tags |= set(tags)
            ent.sticky = ent.sticky or sticky
            self.version += 1
            return True

    def forget(self, entity_id: str) -> bool:
        with self._lock:
            self.version += 1
            return self._entities.pop(entity_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._entities.clear()
            self.version += 1

    # -- internals ----------------------------------------------------------------------
    def _add(self, d: Detection, frame: np.ndarray | None, now: float) -> SceneEntity:
        ent = SceneEntity(id=f"e{next(_ids)}", bbox=d.bbox, label=d.label if d.source != "contour" else "",
                          confidence=d.confidence, source="local", first_seen=now, last_seen=now, last_moved=now)
        if frame is not None:
            ent.color = self._colour_of(frame, d.bbox) or ""
        self._entities[ent.id] = ent
        return ent

    def _best_match(self, d: Detection, pool: dict[str, SceneEntity]) -> Optional[SceneEntity]:
        """Associate a detection with an entity: overlap first, then proximity + size."""
        best, best_score = None, 0.0
        r = max(d.bbox.radius, 1.0)
        for ent in pool.values():
            iou = ent.bbox.iou(d.bbox)
            dist = ent.center.distance_to(d.bbox.center)
            size_ratio = min(ent.bbox.area, d.bbox.area) / max(ent.bbox.area, d.bbox.area, 1.0)
            score = 0.0
            if iou >= self.match_iou:
                score = 1.0 + iou
            elif dist < 1.5 * r and size_ratio > 0.35:
                score = 0.5 + (1.5 * r - dist) / (3.0 * r)
            if score <= 0:
                continue
            if d.label and ent.label:
                score += 0.3 * label_similarity(d.label, ent.label)
            score -= 0.1 * min(ent.seconds_since_seen(), 3.0)
            if score > best_score:
                best, best_score = ent, score
        return best

    @staticmethod
    def _colour_of(frame: np.ndarray, box: BoundingBox) -> str:
        b = box.clip(frame.shape[1], frame.shape[0])
        x, y, w, h = b.as_xywh_int()
        if w < 2 or h < 2:
            return ""
        roi = frame[y:y + h, x:x + w]
        if roi.size == 0:
            return ""
        return describe_color(tuple(float(v) for v in roi.reshape(-1, 3).mean(axis=0)))

    def _prune(self) -> None:
        if len(self._entities) <= self.max_entities:
            return
        # drop the least useful first: absent, unlabelled, least observed, oldest sighting
        victims = sorted(
            self._entities.values(),
            key=lambda e: (e.sticky, e.present, bool(e.label), e.observations, e.last_seen),
        )
        for e in victims[: len(self._entities) - self.max_entities]:
            if e.sticky:
                continue
            self._entities.pop(e.id, None)


def stable_gap(entity: SceneEntity, radius_px: float = 90.0) -> list[Point]:
    """A square outline where an absent entity used to be — used to project the gap."""
    c = entity.center
    r = max(radius_px, entity.bbox.radius * 1.4)
    return [Point(x=c.x - r, y=c.y - r), Point(x=c.x + r, y=c.y - r),
            Point(x=c.x + r, y=c.y + r), Point(x=c.x - r, y=c.y + r)]


def bbox_distance(a: BoundingBox, b: BoundingBox) -> float:
    return math.hypot(a.center.x - b.center.x, a.center.y - b.center.y)
