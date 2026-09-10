"""Proactive safety monitor: the part of BRIDGE that speaks first.

Everything else in BRIDGE is request-response. This module watches the scene and
the count sheet and raises an `Alert` when something is worth interrupting for.

Three design rules keep it from becoming an alarm that people learn to ignore:

1. **Every rule has a cooldown and a key.** The same condition is announced once,
   then not again until it clears and re-occurs (or the cooldown expires).
2. **Levels mean something.** `info` is never spoken unasked; `caution` is spoken
   once; `critical` is spoken, projected in red and written into the case record.
3. **A rule states what it observed, never what to do clinically.** "Two lap pads
   are not on the tray" is an observation. "You have a retained sponge" is a
   diagnosis, and BRIDGE does not make it.

Vision is fallible: a hand over the tray hides instruments. So conditions that
depend on *absence* require the absence to persist, and their wording says what
BRIDGE can actually see ("I cannot see", not "it is gone").
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

from bridge.perception.scene_graph import SceneDelta, SceneGraph
from bridge.surgical.counts import CountSession
from bridge.surgical.trays import TrayLayout

log = logging.getLogger("bridge.surgical.monitor")

Level = Literal["info", "caution", "critical"]
LEVEL_RANK: dict[Level, int] = {"info": 0, "caution": 1, "critical": 2}


@dataclass
class Alert:
    key: str
    level: Level
    text: str                       # spoken verbatim
    project: bool = False           # also draw on the surface
    entity_ids: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    @property
    def speak(self) -> bool:
        return self.level in ("caution", "critical")


@dataclass
class _Condition:
    """How long a condition has continuously held, so alerts need persistence."""

    since: float
    announced_at: float = 0.0


class SafetyMonitor:
    """Evaluated on scene updates (a few times a second at most, not per frame)."""

    def __init__(self, *, sharp_grace_s: float = 20.0, field_item_grace_s: float = 45.0,
                 absence_grace_s: float = 6.0, cooldown_s: float = 45.0,
                 min_interval_s: float = 3.0, enabled: bool = True):
        self.sharp_grace_s = sharp_grace_s
        self.field_item_grace_s = field_item_grace_s
        self.absence_grace_s = absence_grace_s
        self.cooldown_s = cooldown_s
        self.min_interval_s = min_interval_s
        self.enabled = enabled
        self._conditions: dict[str, _Condition] = {}
        self._last_alert_ts = 0.0
        self.raised: list[Alert] = []
        self.muted: set[str] = set()

    # -- public ----------------------------------------------------------------------
    def mute(self, key: str) -> None:
        self.muted.add(key)

    def unmute(self, key: str = "") -> None:
        if key:
            self.muted.discard(key)
        else:
            self.muted.clear()

    def reset(self) -> None:
        self._conditions.clear()
        self.raised.clear()
        self._last_alert_ts = 0.0

    def evaluate(self, scene: SceneGraph, counts: Optional[CountSession] = None,
                 phase: str = "idle", layout: Optional[TrayLayout] = None,
                 delta: Optional[SceneDelta] = None, now: float | None = None) -> list[Alert]:
        """Run every rule; return only the alerts that are due to be announced."""
        if not self.enabled:
            return []
        now = now if now is not None else time.time()
        candidates: list[Alert] = []
        self._touched: set[str] = set()   # conditions seen this pass, alerting or still in grace
        candidates += self._rule_tray_disagreement(scene, counts, phase, now)
        candidates += self._rule_sharps_off_tray(scene, layout, phase, now)
        candidates += self._rule_sponges_out(scene, counts, phase, now)
        candidates += self._rule_item_left_on_field(scene, layout, phase, now)
        candidates += self._rule_unlisted_object(scene, counts, phase, delta, now)
        candidates += self._rule_view_lost(scene, phase, now, delta)
        return self._gate(candidates, now)

    # -- gating ----------------------------------------------------------------------
    def _gate(self, candidates: list[Alert], now: float) -> list[Alert]:
        """Cooldown, dedupe, mute and rate-limit. Critical alerts bypass the rate limit."""
        out: list[Alert] = []
        for a in sorted(candidates, key=lambda x: -LEVEL_RANK[x.level]):
            if a.key in self.muted:
                continue
            cond = self._conditions.setdefault(a.key, _Condition(since=now))
            if cond.announced_at and now - cond.announced_at < self.cooldown_s:
                continue
            if a.level != "critical" and now - self._last_alert_ts < self.min_interval_s:
                continue
            cond.announced_at = now
            self._last_alert_ts = now
            self.raised.append(a)
            self.raised = self.raised[-200:]
            log.info("ALERT [%s] %s: %s", a.level, a.key, a.text)
            out.append(a)
        # A condition that stopped holding is forgotten, so it fires again if it
        # recurs. Conditions still inside their grace period were touched during
        # this pass and must survive, or the grace period could never elapse.
        live = {a.key for a in candidates} | getattr(self, "_touched", set())
        for key in list(self._conditions):
            if key not in live:
                self._conditions.pop(key, None)
        return out

    def _held_for(self, key: str, now: float, seconds: float) -> bool:
        """True once this condition has held continuously for `seconds`."""
        cond = self._conditions.setdefault(key, _Condition(since=now))
        if hasattr(self, "_touched"):
            self._touched.add(key)
        return now - cond.since >= seconds

    # -- rules -----------------------------------------------------------------------
    def _rule_tray_disagreement(self, scene: SceneGraph, counts: Optional[CountSession],
                                phase: str, now: float) -> list[Alert]:
        """Before the field is open, the tray should match the initial count."""
        if counts is None or phase not in ("count_in", "procedure"):
            return []
        if not scene.labelled():
            return []
        out = []
        for line, observed, expected in counts.visual_disagreements():
            if line.category not in ("sponge", "sharp"):
                continue
            missing = expected - observed
            if missing <= 0:
                continue
            key = f"tray_short:{line.name}"
            if not self._held_for(key, now, self.absence_grace_s):
                continue
            out.append(Alert(
                key=key, level="caution", project=True,
                text=(f"I can only see {observed} of {expected} {line.name} on the tray. "
                      "Confirm the rest are in use or recount."),
            ))
        return out

    def _rule_sharps_off_tray(self, scene: SceneGraph, layout: Optional[TrayLayout],
                              phase: str, now: float) -> list[Alert]:
        """A sharp resting outside the tray and the neutral zone is a needlestick risk."""
        if layout is None or phase not in ("procedure", "count_closing", "count_final"):
            return []
        if not layout.zone("tray") and not layout.zone("neutral"):
            return []
        out = []
        for ent in scene.present():
            if "sharp" not in ent.tags and not _looks_sharp(ent.label, ent.aliases):
                continue
            in_tray = layout.zone("tray") is not None and layout.zone("tray").contains(ent.bbox)  # type: ignore[union-attr]
            in_neutral = layout.zone("neutral") is not None and layout.zone("neutral").contains(ent.bbox)  # type: ignore[union-attr]
            if in_tray or in_neutral:
                continue
            key = f"sharp_loose:{ent.id}"
            if not self._held_for(key, now, self.sharp_grace_s):
                continue
            name = ent.label or "a sharp"
            out.append(Alert(key=key, level="caution", project=True, entity_ids=[ent.id],
                             text=f"{name} is outside the tray and the neutral zone. Return it before it is lost."))
        return out

    def _rule_sponges_out(self, scene: SceneGraph, counts: Optional[CountSession],
                          phase: str, now: float) -> list[Alert]:
        """At closing, every sponge should be back and countable."""
        if counts is None or phase not in ("count_closing", "count_final"):
            return []
        out = []
        for line in counts.sponges + counts.sharps:
            base = line.baseline
            if not base:
                continue
            if line.observed >= base:
                continue
            key = f"closing_short:{line.name}"
            if not self._held_for(key, now, self.absence_grace_s):
                continue
            short = base - line.observed
            out.append(Alert(
                key=key, level="critical", project=True,
                text=(f"{short} {line.name} not visible on the tray, {line.observed} of {base}. "
                      "Account for it before closing."),
            ))
        return out

    def _rule_item_left_on_field(self, scene: SceneGraph, layout: Optional[TrayLayout],
                                 phase: str, now: float) -> list[Alert]:
        """Something sitting motionless off the tray for a long time during the case."""
        tray = layout.zone("tray") if layout else None
        if tray is None or phase != "procedure":
            return []
        out = []
        for ent in scene.present():
            if tray.contains(ent.bbox):
                continue
            if ent.seconds_still(now) < self.field_item_grace_s:
                continue
            key = f"left_out:{ent.id}"
            if not self._held_for(key, now, self.field_item_grace_s):
                continue
            out.append(Alert(key=key, level="caution", project=True, entity_ids=[ent.id],
                             text=f"{ent.describe()} has been off the tray for a while. Still needed?"))
        return out

    def _rule_unlisted_object(self, scene: SceneGraph, counts: Optional[CountSession],
                              phase: str, delta: Optional[SceneDelta], now: float) -> list[Alert]:
        """Something opened onto the field that is not on the count sheet."""
        if counts is None or delta is None or phase != "procedure":
            return []
        out = []
        for ent in delta.appeared:
            if not ent.label:
                continue
            if counts.line(ent.label) is not None:
                continue
            key = f"unlisted:{ent.id}"
            out.append(Alert(key=key, level="caution", entity_ids=[ent.id],
                             text=(f"{ent.describe()} is new and not on the count sheet. "
                                   f"Say 'add {ent.label}' if it goes on the field.")))
        return out

    def _rule_view_lost(self, scene: SceneGraph, phase: str, now: float,
                        delta: Optional[SceneDelta]) -> list[Alert]:
        """Everything vanished at once: the camera is blocked, not the tray emptied."""
        if phase in ("idle", "closed") or delta is None:
            return []
        present = len(scene.present())
        if present > 2 or len(delta.disappeared) < 4:
            return []
        key = "view_blocked"
        if not self._held_for(key, now, self.absence_grace_s):
            return []
        return [Alert(key=key, level="caution",
                      text="I have lost sight of the tray. Something is blocking the camera.")]


def _looks_sharp(label: str, aliases) -> bool:
    from bridge.surgical.sets import SHARP_WORDS

    text = " ".join([label or "", *[a for a in (aliases or [])]]).lower()
    return any(w in text for w in SHARP_WORDS)
