"""Surgical count engine — the deepest workflow in BRIDGE.

The rule the whole module exists to enforce is arithmetic, not clinical:

    final count  ==  initial count  +  everything added during the case

If that does not hold, an item is unaccounted for and the team follows its
discrepancy protocol (recount, search the field and drapes, radiograph). BRIDGE
computes the arithmetic, keeps the tally hands-free, shows it on the tray and
says the number out loud. It never says a count is "fine" on its own authority
and never authorises closure: `Reconciliation.verdict` is a statement about the
numbers, and the spoken text always attributes the decision to the team.

What is *observed* by the camera is advisory and clearly separated from what is
*counted* by a person. `CountLine.counted_*` fields are human counts; `observed`
is what BRIDGE currently sees on the tray. A camera cannot see inside a wound,
so vision never substitutes for the count — it only flags disagreement early.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field

from bridge.perception.scene_graph import (
    SceneEntity,
    SceneGraph,
    label_refines,
    label_similarity,
    normalise_label,
)
from bridge.surgical.sets import CATEGORY_ORDER, Category, InstrumentSet, SetItem, classify

log = logging.getLogger("bridge.surgical.count")

CountPhase = Literal["initial", "added", "closing", "final"]
PHASES: tuple[CountPhase, ...] = ("initial", "added", "closing", "final")

LineStatus = Literal["pending", "ok", "short", "over"]
Verdict = Literal["pending", "reconciled", "discrepancy"]


class CountLine(BaseModel):
    """One countable item class across the whole case."""

    name: str
    category: Category = "instrument"
    aliases: list[str] = Field(default_factory=list)
    expected: int = 0                 # from the set definition (a starting hint only)
    counted_initial: Optional[int] = None
    counted_added: int = 0
    counted_closing: Optional[int] = None
    counted_final: Optional[int] = None
    observed: int = 0                 # what the camera can currently see on the tray
    radiopaque: bool = False
    note: str = ""

    # -- derived ------------------------------------------------------------------------
    @property
    def baseline(self) -> int:
        """What must be accounted for at the end."""
        base = self.counted_initial if self.counted_initial is not None else self.expected
        return base + self.counted_added

    def counted_at(self, phase: CountPhase) -> Optional[int]:
        return {"initial": self.counted_initial, "added": self.counted_added,
                "closing": self.counted_closing, "final": self.counted_final}[phase]

    def status_at(self, phase: CountPhase) -> LineStatus:
        got = self.counted_at(phase)
        if phase == "initial":
            if got is None:
                return "pending"
            if not self.expected:
                return "ok"
            return "ok" if got == self.expected else ("short" if got < self.expected else "over")
        if phase == "added":
            return "ok"
        if got is None:
            return "pending"
        return "ok" if got == self.baseline else ("short" if got < self.baseline else "over")

    def delta_at(self, phase: CountPhase) -> int:
        got = self.counted_at(phase)
        if got is None:
            return 0
        return got - (self.expected if phase == "initial" else self.baseline)

    def matches(self, spoken: str, threshold: float = 0.6) -> float:
        """How well a spoken phrase names this count line.

        Stricter than scene-graph matching: on a count sheet, "bulldog clamp"
        landing on the "artery clamp" line would silently corrupt the arithmetic.
        A partial match only counts when one name refines the other.
        """
        best = 0.0
        for name in (self.name, *self.aliases):
            s = label_similarity(spoken, name)
            if s >= 1.0:
                return 1.0
            if s > 0 and label_refines(spoken, name):
                best = max(best, max(s, 0.85))
            elif s >= threshold:
                best = max(best, s)
        return best

    @classmethod
    def from_set_item(cls, item: SetItem) -> "CountLine":
        return cls(name=item.name, category=item.kind, aliases=list(item.aliases),
                   expected=item.quantity, radiopaque=item.radiopaque)


@dataclass
class Reconciliation:
    """The arithmetic result of one count phase."""

    phase: CountPhase
    lines: list[CountLine] = field(default_factory=list)
    ts: float = field(default_factory=time.time)

    @property
    def short(self) -> list[CountLine]:
        return [l for l in self.lines if l.status_at(self.phase) == "short"]

    @property
    def over(self) -> list[CountLine]:
        return [l for l in self.lines if l.status_at(self.phase) == "over"]

    @property
    def pending(self) -> list[CountLine]:
        return [l for l in self.lines if l.status_at(self.phase) == "pending"]

    @property
    def discrepancies(self) -> list[CountLine]:
        return self.short + self.over

    @property
    def verdict(self) -> Verdict:
        if self.pending:
            return "pending"
        return "discrepancy" if self.discrepancies else "reconciled"

    @property
    def totals(self) -> tuple[int, int]:
        """(counted, expected) across all lines for this phase."""
        counted = sum(l.counted_at(self.phase) or 0 for l in self.lines)
        expected = sum(l.expected if self.phase == "initial" else l.baseline for l in self.lines)
        return counted, expected

    def spoken(self) -> str:
        """One or two sentences for the voice channel. Never authorises closure."""
        label = {"initial": "Initial count", "added": "Additions", "closing": "Closing count",
                 "final": "Final count"}[self.phase]
        if self.pending and self.verdict == "pending":
            names = ", ".join(l.name for l in self.pending[:4])
            return f"{label} incomplete. Still to count: {names}."
        counted, expected = self.totals
        if self.verdict == "reconciled":
            return (f"{label} reconciles. {counted} of {expected} items accounted for. "
                    "Confirm with your own count sheet.")
        bits = []
        for l in self.short:
            bits.append(f"{abs(l.delta_at(self.phase))} {l.name} short")
        for l in self.over:
            bits.append(f"{l.delta_at(self.phase)} extra {l.name}")
        detail = "; ".join(bits[:4])
        return (f"Count discrepancy. {detail}. Do not close. "
                "Recount, search the field and drapes, and follow your discrepancy protocol.")

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "ts": datetime.fromtimestamp(self.ts, timezone.utc).isoformat(),
            "verdict": self.verdict,
            "counted": self.totals[0],
            "expected": self.totals[1],
            "lines": [
                {"name": l.name, "category": l.category, "counted": l.counted_at(self.phase),
                 "baseline": l.expected if self.phase == "initial" else l.baseline,
                 "status": l.status_at(self.phase), "observed": l.observed}
                for l in self.lines
            ],
        }


class CountSession:
    """A whole case's counts. Thread-safe: voice, camera and UI all touch it."""

    def __init__(self, instrument_set: InstrumentSet | None = None, case_id: str = ""):
        self.set = instrument_set
        self.case_id = case_id or datetime.now().strftime("case-%Y%m%d-%H%M%S")
        self.lines: list[CountLine] = [CountLine.from_set_item(i) for i in (instrument_set.items if instrument_set else [])]
        self.phase: CountPhase = "initial"
        self.history: list[Reconciliation] = []
        self.events: list[dict] = []
        self.started_ts = time.time()
        self.closed = False
        self._lock = threading.RLock()

    # -- lookup ---------------------------------------------------------------------------
    def line(self, spoken: str, create: bool = False) -> Optional[CountLine]:
        with self._lock:
            best, score = None, 0.0
            for l in self.lines:
                s = l.matches(spoken)
                if s > score:
                    best, score = l, s
            if best is not None:
                return best
            if create and spoken.strip():
                l = CountLine(name=normalise_label(spoken) or spoken.strip(), category=classify(spoken))
                self.lines.append(l)
                self._log("line_added", name=l.name, category=l.category)
                return l
            return None

    def by_category(self, category: Category) -> list[CountLine]:
        return [l for l in self.lines if l.category == category]

    @property
    def sharps(self) -> list[CountLine]:
        return self.by_category("sharp")

    @property
    def sponges(self) -> list[CountLine]:
        return self.by_category("sponge")

    # -- recording ------------------------------------------------------------------------
    def set_phase(self, phase: CountPhase) -> str:
        with self._lock:
            self.phase = phase
            self._log("phase", phase=phase)
            return phase

    def record(self, item: str, quantity: int, phase: CountPhase | None = None,
               create: bool = True) -> tuple[Optional[CountLine], str]:
        """Record a human count. Returns (line, spoken acknowledgement)."""
        with self._lock:
            if self.closed:
                return None, "This case is already closed. Start a new count."
            phase = phase or self.phase
            l = self.line(item, create=create)
            if l is None:
                return None, f"I don't have {item} on the count sheet. Say 'add {item}' first."
            if quantity < 0:
                return l, "A count cannot be negative."
            if phase == "initial":
                l.counted_initial = quantity
            elif phase == "added":
                l.counted_added += quantity
            elif phase == "closing":
                l.counted_closing = quantity
            else:
                l.counted_final = quantity
            self._log("count", item=l.name, phase=phase, quantity=quantity)
            return l, self._ack(l, phase, quantity)

    def add_item(self, item: str, quantity: int = 1) -> tuple[CountLine, str]:
        """Something opened onto the field mid-case — the classic count-discrepancy source."""
        with self._lock:
            l = self.line(item, create=True)
            assert l is not None
            l.counted_added += quantity
            self._log("added", item=l.name, quantity=quantity)
            unit = l.name if quantity == 1 else f"{l.name}s"
            return l, (f"{quantity} {unit} added. {l.name} now expects {l.baseline} at the final count.")

    def bulk_record(self, pairs: Iterable[tuple[str, int]], phase: CountPhase | None = None) -> list[CountLine]:
        out = []
        for name, qty in pairs:
            l, _ = self.record(name, qty, phase)
            if l is not None:
                out.append(l)
        return out

    def accept_expected(self, phase: CountPhase = "initial") -> Reconciliation:
        """'Count as per the sheet' — record every line at its expected quantity."""
        with self._lock:
            for l in self.lines:
                if phase == "initial":
                    l.counted_initial = l.expected
                elif phase == "closing":
                    l.counted_closing = l.baseline
                elif phase == "final":
                    l.counted_final = l.baseline
            self._log("accept_expected", phase=phase)
            return self.reconcile(phase)

    # -- vision (advisory only) -------------------------------------------------------------
    def observe(self, scene: SceneGraph) -> dict[str, int]:
        """Tally what the camera can see per line. Advisory: never a substitute for a count."""
        with self._lock:
            tally = {l.name: 0 for l in self.lines}
            for ent in scene.present():
                l = self._line_for_entity(ent)
                if l is not None:
                    tally[l.name] += 1
            for l in self.lines:
                l.observed = tally.get(l.name, 0)
            return tally

    def _line_for_entity(self, ent: SceneEntity) -> Optional[CountLine]:
        text = " ".join(filter(None, [ent.label, *sorted(ent.aliases)]))
        best, score = None, 0.0
        for l in self.lines:
            s = l.matches(text, threshold=0.45)
            if s > score:
                best, score = l, s
        return best

    def visual_disagreements(self, tolerance: int = 0) -> list[tuple[CountLine, int, int]]:
        """Lines where what BRIDGE sees on the tray differs from what should be there.

        Returns (line, observed, expected_on_tray). Only meaningful before the
        field is opened; items in use are legitimately off the tray.
        """
        out = []
        for l in self.lines:
            base = l.counted_initial if l.counted_initial is not None else l.expected
            if base and abs(l.observed - base) > tolerance:
                out.append((l, l.observed, base))
        return out

    # -- reconciliation --------------------------------------------------------------------
    def reconcile(self, phase: CountPhase | None = None) -> Reconciliation:
        with self._lock:
            r = Reconciliation(phase=phase or self.phase, lines=list(self.lines))
            self.history.append(r)
            self.history = self.history[-40:]
            self._log("reconcile", phase=r.phase, verdict=r.verdict,
                      counted=r.totals[0], expected=r.totals[1])
            if r.verdict == "discrepancy":
                log.warning("COUNT DISCREPANCY phase=%s short=%s over=%s", r.phase,
                            [l.name for l in r.short], [l.name for l in r.over])
            return r

    def unaccounted(self) -> list[tuple[CountLine, int]]:
        """Items whose final count is below baseline — potential retained items."""
        out = []
        for l in self.lines:
            if l.counted_final is not None and l.counted_final < l.baseline:
                out.append((l, l.baseline - l.counted_final))
        return out

    @property
    def retained_risk(self) -> bool:
        return any(cat in ("sponge", "sharp") for l, _ in self.unaccounted() for cat in (l.category,))

    def close(self) -> Reconciliation:
        with self._lock:
            r = self.reconcile("final")
            self.closed = True
            self._log("closed", verdict=r.verdict)
            return r

    # -- reporting -------------------------------------------------------------------------
    def board(self) -> list[tuple[str, str, int, int, LineStatus]]:
        """Rows for the projected HUD: (category, name, counted-now, baseline, status)."""
        rows = []
        for cat in CATEGORY_ORDER:
            for l in sorted(self.by_category(cat), key=lambda x: x.name):
                got = l.counted_at(self.phase)
                rows.append((cat, l.name, got if got is not None else l.observed,
                             l.expected if self.phase == "initial" else l.baseline,
                             l.status_at(self.phase)))
        return rows

    def summary_line(self) -> str:
        counted, expected = Reconciliation(phase=self.phase, lines=self.lines).totals
        return f"{self.phase.upper()}  {counted}/{expected}"

    def _ack(self, line: CountLine, phase: CountPhase, quantity: int) -> str:
        if phase in ("initial", "added"):
            return f"{quantity} {line.name}. Recorded."
        base = line.baseline
        if quantity == base:
            return f"{line.name}: {quantity} of {base}. Correct."
        diff = quantity - base
        if diff < 0:
            return (f"{line.name}: {quantity} of {base}. {abs(diff)} missing. "
                    "Recount and search the field before closing.")
        return f"{line.name}: {quantity} of {base}. {diff} more than expected. Recheck the sheet."

    def _log(self, kind: str, **payload) -> None:
        self.events.append({"ts": time.time(), "kind": kind, **payload})
        self.events = self.events[-500:]

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "set": self.set.id if self.set else None,
            "set_name": self.set.name if self.set else None,
            "started": datetime.fromtimestamp(self.started_ts, timezone.utc).isoformat(),
            "phase": self.phase,
            "closed": self.closed,
            "lines": [l.model_dump() for l in self.lines],
            "reconciliations": [r.to_dict() for r in self.history],
            "events": self.events,
        }

    def save(self, directory: Path | str = Path("case_records")) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.case_id}-counts.json"
        p.write_text(json.dumps(self.to_dict(), indent=2))
        log.info("Count record written to %s", p)
        return p


def parse_spoken_count(text: str) -> Optional[tuple[str, int]]:
    """'four artery clamps' / 'artery clamps four' / 'lap pads: 10' -> ('artery clamp', 4)."""
    import re

    words = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
             "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
             "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
             "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
             "a": 1, "an": 1, "no": 0, "none": 0}
    t = re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).strip()
    if not t:
        return None
    tokens = t.split()
    qty, idx = None, -1
    for i, tok in enumerate(tokens):
        if tok.isdigit():
            qty, idx = int(tok), i
            break
        if tok in words:
            qty, idx = words[tok], i
            break
    if qty is None:
        return None
    name = " ".join(tokens[:idx] + tokens[idx + 1:]).strip()
    name = normalise_label(name)
    return (name, qty) if name else None
