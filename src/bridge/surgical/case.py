"""Case autopilot: the spine of a surgical case, hands-free.

A case walks through phases in order, each with a small set of things that must
be confirmed out loud:

    briefing -> sign_in -> timeout -> count_in -> procedure
             -> count_closing -> count_final -> sign_out -> debrief -> closed

The checklist content follows the shape of the WHO Surgical Safety Checklist
(sign in before anaesthesia, time out before incision, sign out before the
patient leaves theatre). It is *data*, editable per hospital, and BRIDGE reads
items aloud and records who confirmed what — it does not decide anything
clinical and does not gate the operation. If the team says "skip", BRIDGE
records the skip and moves on; refusing to proceed would be unsafe in its own
right.

Everything is recorded into a `CaseRecord` written as JSON at sign-out, so the
case has an auditable trail of counts, confirmations, skips and alerts.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Literal, Optional

from pydantic import BaseModel, Field

from bridge.surgical.counts import CountSession, Reconciliation
from bridge.surgical.sets import InstrumentSet

log = logging.getLogger("bridge.surgical.case")

Phase = Literal[
    "idle", "briefing", "sign_in", "timeout", "count_in", "procedure",
    "count_closing", "count_final", "sign_out", "debrief", "closed",
]

PHASE_ORDER: tuple[Phase, ...] = (
    "idle", "briefing", "sign_in", "timeout", "count_in", "procedure",
    "count_closing", "count_final", "sign_out", "debrief", "closed",
)

PHASE_TITLE: dict[Phase, str] = {
    "idle": "No case", "briefing": "Team briefing", "sign_in": "Sign in",
    "timeout": "Time out", "count_in": "Initial count", "procedure": "Procedure",
    "count_closing": "Closing count", "count_final": "Final count",
    "sign_out": "Sign out", "debrief": "Debrief", "closed": "Case closed",
}


class ChecklistItem(BaseModel):
    key: str
    prompt: str                      # spoken and projected
    expects: Literal["confirm", "value", "observe"] = "confirm"
    critical: bool = False           # a skip on these is called out in the record
    note: str = ""


class ChecklistPhase(BaseModel):
    phase: Phase
    intro: str = ""
    items: list[ChecklistItem] = Field(default_factory=list)


def _c(key: str, prompt: str, expects: str = "confirm", critical: bool = False) -> ChecklistItem:
    return ChecklistItem(key=key, prompt=prompt, expects=expects, critical=critical)  # type: ignore[arg-type]


DEFAULT_CHECKLIST: list[ChecklistPhase] = [
    ChecklistPhase(
        phase="briefing", intro="Team briefing.",
        items=[
            _c("introductions", "Everyone state your name and role."),
            _c("procedure_named", "Confirm the planned procedure.", "value", critical=True),
            _c("duration", "Expected duration and any critical steps.", "value"),
            _c("concerns", "Any anaesthetic, equipment or patient concerns."),
        ],
    ),
    ChecklistPhase(
        phase="sign_in", intro="Sign in, before anaesthesia.",
        items=[
            _c("identity", "Patient identity, procedure and consent confirmed.", "confirm", critical=True),
            _c("site_marked", "Surgical site marked, or not applicable.", "confirm", critical=True),
            _c("anaesthesia_check", "Anaesthesia machine and medication check complete.", "confirm", critical=True),
            _c("pulse_oximeter", "Pulse oximeter on the patient and working."),
            _c("allergy", "Any known allergy.", "value", critical=True),
            _c("airway_risk", "Difficult airway or aspiration risk, and equipment ready."),
            _c("blood_loss_risk", "Risk of blood loss over 500 millilitres, and access planned."),
        ],
    ),
    ChecklistPhase(
        phase="timeout", intro="Time out, before skin incision. Everyone stop.",
        items=[
            _c("team_confirm", "All team members have introduced themselves by name and role."),
            _c("patient_procedure_site", "Surgeon, anaesthetist and nurse confirm patient, site and procedure.",
               "confirm", critical=True),
            _c("critical_steps", "Surgeon states critical or unexpected steps and expected duration.", "value"),
            _c("anaesthetic_concerns", "Anaesthetist states any patient-specific concerns.", "value"),
            _c("sterility", "Nursing team confirms sterility and equipment issues.", "confirm", critical=True),
            _c("antibiotic", "Antibiotic prophylaxis given in the last sixty minutes, or not indicated.",
               "confirm", critical=True),
            _c("imaging", "Essential imaging displayed, or not required."),
        ],
    ),
    ChecklistPhase(
        phase="sign_out", intro="Sign out, before the patient leaves theatre.",
        items=[
            _c("procedure_recorded", "Nurse confirms the name of the procedure recorded.", "value"),
            _c("counts_correct", "Instrument, sponge and sharp counts are correct.", "confirm", critical=True),
            _c("specimen", "Specimen labelled, including patient name.", "confirm", critical=True),
            _c("equipment_problems", "Any equipment problems to be addressed.", "value"),
            _c("recovery_concerns", "Key concerns for recovery and management of this patient.", "value"),
        ],
    ),
    ChecklistPhase(
        phase="debrief", intro="Debrief.",
        items=[
            _c("what_went_well", "What went well.", "value"),
            _c("what_to_improve", "What could be improved.", "value"),
        ],
    ),
]


@dataclass
class Confirmation:
    phase: Phase
    key: str
    prompt: str
    response: str
    skipped: bool = False
    ts: float = field(default_factory=time.time)


class CaseSession:
    """Drives a case through its phases. All state changes are logged."""

    def __init__(self, checklist: list[ChecklistPhase] | None = None,
                 instrument_set: InstrumentSet | None = None,
                 case_id: str = "", procedure_name: str = ""):
        self.checklist = {p.phase: p for p in (checklist or DEFAULT_CHECKLIST)}
        self.case_id = case_id or datetime.now().strftime("case-%Y%m%d-%H%M%S")
        self.procedure_name = procedure_name
        self.counts = CountSession(instrument_set, case_id=self.case_id)
        self.phase: Phase = "idle"
        self.item_index = 0
        self.confirmations: list[Confirmation] = []
        self.alerts: list[dict] = []
        self.timeline: list[dict] = []
        self.started_ts = 0.0
        self.ended_ts = 0.0
        self.on_phase: Callable[[Phase], None] = lambda p: None
        self._lock = threading.RLock()

    # -- state ------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.phase not in ("idle", "closed")

    @property
    def current_phase_items(self) -> list[ChecklistItem]:
        p = self.checklist.get(self.phase)
        return p.items if p else []

    @property
    def current_item(self) -> Optional[ChecklistItem]:
        items = self.current_phase_items
        return items[self.item_index] if 0 <= self.item_index < len(items) else None

    @property
    def in_checklist(self) -> bool:
        return self.current_item is not None

    def progress(self) -> tuple[int, int]:
        return self.item_index, len(self.current_phase_items)

    # -- lifecycle ---------------------------------------------------------------------
    def start(self, procedure_name: str = "", instrument_set: InstrumentSet | None = None) -> str:
        with self._lock:
            self.procedure_name = procedure_name or self.procedure_name
            if instrument_set is not None:
                self.counts = CountSession(instrument_set, case_id=self.case_id)
            self.started_ts = time.time()
            self._event("case_started", procedure=self.procedure_name,
                        set=self.counts.set.id if self.counts.set else None)
            return self._enter("briefing")

    def next_phase(self) -> str:
        with self._lock:
            i = PHASE_ORDER.index(self.phase)
            nxt = PHASE_ORDER[min(i + 1, len(PHASE_ORDER) - 1)]
            return self._enter(nxt)

    def go_to(self, phase: Phase) -> str:
        with self._lock:
            return self._enter(phase)

    def _enter(self, phase: Phase) -> str:
        self.phase = phase
        self.item_index = 0
        self._event("phase", phase=phase)
        log.info("Case %s entered phase %s", self.case_id, phase)
        try:
            self.on_phase(phase)
        except Exception:  # noqa: BLE001
            log.exception("case phase callback failed")
        if phase == "count_in":
            self.counts.set_phase("initial")
        elif phase == "procedure":
            self.counts.set_phase("added")
        elif phase == "count_closing":
            self.counts.set_phase("closing")
        elif phase == "count_final":
            self.counts.set_phase("final")
        return self.announce()

    def announce(self) -> str:
        """The sentence to speak on entering the current phase / item."""
        p = self.checklist.get(self.phase)
        title = PHASE_TITLE[self.phase]
        if self.phase == "count_in":
            n = len(self.counts.lines)
            src = self.counts.set.name if self.counts.set else "no set loaded"
            return (f"Initial count. {src}, {n} item lines. Count aloud and I will record. "
                    "Say 'count as per the sheet' if the tray matches exactly.")
        if self.phase == "procedure":
            return ("Counts recorded. I am watching the tray. Tell me anything added to the field, "
                    "and ask me to project any instrument.")
        if self.phase == "count_closing":
            return "Closing count. Call each item and I will check it against the baseline."
        if self.phase == "count_final":
            return "Final count. Call each item; I will read back the arithmetic."
        if self.phase == "closed":
            return "Case closed. The record is saved."
        if p is None:
            return f"{title}."
        item = self.current_item
        intro = p.intro or f"{title}."
        return f"{intro} {item.prompt}" if item else intro

    # -- checklist -----------------------------------------------------------------------
    def confirm(self, response: str = "yes") -> str:
        """Confirm the current checklist item and advance."""
        with self._lock:
            item = self.current_item
            if item is None:
                return self.next_phase()
            self.confirmations.append(Confirmation(self.phase, item.key, item.prompt, response))
            self._event("confirmed", phase=self.phase, key=item.key, response=response[:120])
            self.item_index += 1
            nxt = self.current_item
            if nxt is not None:
                return nxt.prompt
            return f"{PHASE_TITLE[self.phase]} complete. " + self.next_phase()

    def skip(self, reason: str = "") -> str:
        with self._lock:
            item = self.current_item
            if item is None:
                return self.next_phase()
            self.confirmations.append(Confirmation(self.phase, item.key, item.prompt, reason or "skipped", skipped=True))
            self._event("skipped", phase=self.phase, key=item.key, critical=item.critical, reason=reason[:120])
            if item.critical:
                log.warning("Critical checklist item skipped: %s/%s", self.phase, item.key)
            self.item_index += 1
            nxt = self.current_item
            note = "Recorded as not done. " if item.critical else ""
            if nxt is not None:
                return note + nxt.prompt
            return note + f"{PHASE_TITLE[self.phase]} complete. " + self.next_phase()

    def repeat(self) -> str:
        item = self.current_item
        return item.prompt if item else self.announce()

    def back(self) -> str:
        with self._lock:
            if self.item_index > 0:
                self.item_index -= 1
                if self.confirmations:
                    self.confirmations.pop()
                return self.repeat()
            i = PHASE_ORDER.index(self.phase)
            return self._enter(PHASE_ORDER[max(i - 1, 1)])

    # -- record ---------------------------------------------------------------------------
    def record_alert(self, level: str, key: str, text: str) -> None:
        self.alerts.append({"ts": time.time(), "level": level, "key": key, "text": text})
        self.alerts = self.alerts[-300:]

    def _event(self, kind: str, **payload) -> None:
        self.timeline.append({"ts": time.time(), "kind": kind, **payload})
        self.timeline = self.timeline[-1000:]

    @property
    def critical_skips(self) -> list[Confirmation]:
        return [c for c in self.confirmations if c.skipped and self._is_critical(c)]

    def _is_critical(self, c: Confirmation) -> bool:
        p = self.checklist.get(c.phase)
        return bool(p and any(i.key == c.key and i.critical for i in p.items))

    def close(self) -> tuple[Reconciliation, str]:
        """End the case: final reconciliation + the sentence to speak."""
        with self._lock:
            rec = self.counts.close()
            self.ended_ts = time.time()
            self._enter("closed")
            skips = self.critical_skips
            parts = [rec.spoken()]
            if skips:
                parts.append(f"{len(skips)} critical checklist item"
                             f"{'s were' if len(skips) > 1 else ' was'} not confirmed and is in the record.")
            return rec, " ".join(parts)

    def to_dict(self) -> dict:
        def iso(ts: float) -> str:
            return datetime.fromtimestamp(ts, timezone.utc).isoformat() if ts else ""

        return {
            "case_id": self.case_id,
            "procedure": self.procedure_name,
            "started": iso(self.started_ts),
            "ended": iso(self.ended_ts),
            "phase": self.phase,
            "checklist": [
                {"phase": c.phase, "key": c.key, "prompt": c.prompt, "response": c.response,
                 "skipped": c.skipped, "ts": iso(c.ts)}
                for c in self.confirmations
            ],
            "critical_skips": [c.key for c in self.critical_skips],
            "counts": self.counts.to_dict(),
            "alerts": self.alerts,
            "timeline": self.timeline,
            "disclaimer": ("BRIDGE is an assistive spatial and documentation aid. Counts, checklist "
                           "confirmations and clinical decisions are the responsibility of the operating team."),
        }

    def save(self, directory: Path | str = Path("case_records")) -> Path:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{self.case_id}.json"
        p.write_text(json.dumps(self.to_dict(), indent=2))
        log.info("Case record written to %s", p)
        return p

    def report_text(self) -> str:
        """A short human-readable summary for the log panel and the chat."""
        lines = [f"CASE {self.case_id}", f"Procedure: {self.procedure_name or 'not recorded'}"]
        if self.counts.set:
            lines.append(f"Set: {self.counts.set.name}")
        for r in self.counts.history[-4:]:
            counted, expected = r.totals
            lines.append(f"{r.phase:>8} count: {counted}/{expected} — {r.verdict}")
        if self.critical_skips:
            lines.append("Critical items not confirmed: " + ", ".join(c.key for c in self.critical_skips))
        if self.alerts:
            lines.append(f"Alerts raised: {len(self.alerts)}")
        return "\n".join(lines)


def load_checklist(path: Path | str) -> list[ChecklistPhase]:
    """Load a hospital's own checklist JSON: [{phase, intro, items:[{key, prompt, ...}]}]."""
    data = json.loads(Path(path).read_text())
    return [ChecklistPhase.model_validate(p) for p in data]
