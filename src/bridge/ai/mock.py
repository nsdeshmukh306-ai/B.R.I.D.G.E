"""Mock / simulation AI provider. NOT used in the physical production path.

Two flavours:
- ScriptedMockProvider: returns canned responses (unit tests).
- SimulationOracleProvider: answers from the VirtualWorld's ground truth with
  a simple intent parser, so the whole pipeline can be exercised without a
  network connection or API key. Boxes are returned in Gemini's normalized
  format so the same target-resolution code path is used.
"""
from __future__ import annotations

import re
import time
from typing import Callable, Optional

import numpy as np

from bridge.ai.base import AIProvider
from bridge.ai.schemas import ActionPlan, NormalizedBox, SceneObject, SceneUnderstanding, TargetIdentification

COLOR_WORDS = ("red", "blue", "green", "black", "white", "grey", "gray", "yellow", "orange", "silver", "purple", "pink")


def parse_intent(query: str) -> tuple[str, str, str, str]:
    """Return (intent, target, style, secondary). Tiny rule-based parser for simulation/tests."""
    q = query.lower().strip().rstrip("?.!")
    secondary = ""
    m = re.search(r"(?:from|move) (?:the )?(.+?) to (?:the )?(.+)$", q)
    if m:
        return "draw_path", m.group(1).strip(), "circle", m.group(2).strip()
    if any(k in q for k in ("clear", "remove projection", "reset")):
        return "clear_projection", "", "circle", ""
    if any(k in q for k in ("pick up first", "what should i do", "next step", "what next", "first")):
        return "next_step", "", "circle", ""
    if q.startswith(("what is on", "describe", "what do you see")):
        return "describe_scene", "", "circle", ""
    if "where should i put" in q or "where to put" in q or "where does" in q:
        return "show_target_zone", _strip_target(q.split("put", 1)[-1] if "put" in q else q), "outline", ""
    intent = "find_object"
    if q.startswith("point to") or q.startswith("point at"):
        intent = "point_to_object"
    elif q.startswith("label") or q.startswith("name"):
        intent = "label_object"
    elif q.startswith(("highlight", "circle", "mark", "show me the")):
        intent = "highlight_object"
    target = _strip_target(q)
    plural = target.endswith("s") and not target.endswith("ss") and target not in PLURAL_ONLY
    if intent == "find_object" and (re.search(r"\b(all|every|objects|tools)\b", q) or plural):
        intent = "find_all"
    if plural:
        target = target[:-1]
    return intent, target, "circle", secondary


PLURAL_ONLY = {"scissors", "pliers", "tweezers", "glasses", "tongs"}


def _strip_target(q: str) -> str:
    q = re.sub(r"\s*\([^)]*\)", "", q).strip()  # drop parenthetical hints
    q = re.sub(r"^(where is|where are|where's|find|locate|highlight|circle|mark|point to|point at|label|name|show me|show)\s+", "", q)
    q = re.sub(r"^(the|a|an|all|every|all the|all of the)\s+", "", q)
    q = re.sub(r"\s+(please|now|for me)$", "", q)
    return q.strip()


class ScriptedMockProvider(AIProvider):
    name = "mock"

    def __init__(self, responder: Optional[Callable[[str], TargetIdentification]] = None, latency_s: float = 0.0):
        super().__init__()
        self.responder = responder
        self.latency_s = latency_s
        self.calls: list[str] = []

    def identify_target(self, frame: np.ndarray, query: str, known_labels: list[str] | None = None) -> TargetIdentification:
        self.calls.append(query)
        t0 = time.perf_counter()
        if self.latency_s:
            time.sleep(self.latency_s)
        if self.responder:
            out = self.responder(query)
        else:
            intent, target, style, secondary = parse_intent(query)
            out = TargetIdentification(intent=intent, target=target, confidence=0.0, message="mock: no vision", style=style,
                                       secondary_target=secondary)
        self.status.record("identify_target", time.perf_counter() - t0, True)
        return out

    def understand_scene(self, frame: np.ndarray) -> SceneUnderstanding:
        self.status.record("understand_scene", 0.0, True)
        return SceneUnderstanding(summary="mock scene")

    def plan_action(self, frame: np.ndarray, task_context: str) -> ActionPlan:
        self.status.record("plan_action", 0.0, True)
        return ActionPlan(instruction="mock", target="", confidence=0.0, done=True)


class SimulationOracleProvider(AIProvider):
    """Answers from simulator ground truth. Clearly a simulation aid, never for hardware mode."""

    name = "simulation-oracle"

    def __init__(self, world, camera, latency_s: float = 0.15, pickup_order: list[str] | None = None):
        super().__init__()
        self.world, self.camera = world, camera
        self.latency_s = latency_s
        self.pickup_order = pickup_order or ["gloves", "tourniquet", "alcohol swab", "needle", "syringe", "lavender cap tube",
                                             "gauze", "screwdriver", "screw", "pen", "scissors", "phone"]
        self.status.provider = "Simulation oracle (mock)"

    def _box(self, obj) -> NormalizedBox:
        b = self.camera.object_bbox_in_camera(obj)
        w, h = self.camera.resolution
        pad = 0.02
        y1, x1 = max(0, b.y / h - pad) * 1000, max(0, b.x / w - pad) * 1000
        y2, x2 = min(1, b.y2 / h + pad) * 1000, min(1, b.x2 / w + pad) * 1000
        return NormalizedBox(box_2d=[y1, x1, y2, x2], label=obj.label)

    def _matches(self, obj, target: str) -> bool:
        t = target.lower()
        if not t:
            return False
        words = t.split()
        if obj.label == t or obj.label in words or (obj.label + "s") in words:
            return True
        colours = [c for c in COLOR_WORDS if c in t]
        if colours and ("object" in t or "thing" in t or "one" in t):
            return any(c in obj.description for c in colours)
        desc_words = set(obj.description.lower().replace("-", " ").split())
        return any(w in desc_words for w in words if len(w) > 3)

    def identify_target(self, frame: np.ndarray, query: str, known_labels: list[str] | None = None) -> TargetIdentification:
        t0 = time.perf_counter()
        time.sleep(self.latency_s)
        intent, target, style, secondary = parse_intent(query)
        if intent in ("clear_projection", "describe_scene", "next_step"):
            out = TargetIdentification(intent=intent, target=target, confidence=1.0, message="")
        else:
            matches = [o for o in self.world.objects if self._matches(o, target)]
            if intent != "find_all":
                matches = matches[:1]
            sec = [o for o in self.world.objects if self._matches(o, secondary)][:1] if secondary else []
            conf = 0.95 if matches else 0.1
            out = TargetIdentification(
                intent=intent, target=target, visual_description=matches[0].description if matches else "",
                confidence=conf, boxes=[self._box(o) for o in matches], style=style,
                message="" if matches else f"I cannot see a {target} on the surface.",
                secondary_target=secondary, secondary_boxes=[self._box(o) for o in sec],
            )
        self.status.record("identify_target", time.perf_counter() - t0, True)
        return out

    def understand_scene(self, frame: np.ndarray) -> SceneUnderstanding:
        time.sleep(self.latency_s)
        objs = [SceneObject(label=o.label, visual_description=o.description, box=self._box(o), confidence=0.95) for o in self.world.objects]
        self.status.record("understand_scene", self.latency_s, True)
        return SceneUnderstanding(objects=objs, summary=", ".join(o.label for o in self.world.objects))

    def plan_action(self, frame: np.ndarray, task_context: str) -> ActionPlan:
        time.sleep(self.latency_s)
        for label in self.pickup_order:
            obj = next((o for o in self.world.objects if o.label == label), None)
            if obj is not None:
                self.status.record("plan_action", self.latency_s, True)
                return ActionPlan(instruction=f"Pick up the {label}", target=label, visual_description=obj.description,
                                  boxes=[self._box(obj)], confidence=0.9)
        self.status.record("plan_action", self.latency_s, True)
        return ActionPlan(instruction="All steps complete", target="", confidence=1.0, done=True)


def build_provider(kind: str, api_key: str | None, model: str, timeout_s: float = 30.0, simulation=None) -> AIProvider:
    if kind == "gemini":
        from bridge.ai.gemini import GeminiProvider

        return GeminiProvider(api_key, model, timeout_s)
    if kind == "mock":
        if simulation is not None:
            world, camera = simulation
            return SimulationOracleProvider(world, camera)
        return ScriptedMockProvider()
    raise ValueError(f"unknown AI provider '{kind}'")
