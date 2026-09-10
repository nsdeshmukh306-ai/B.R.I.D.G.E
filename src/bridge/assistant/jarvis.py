"""JarvisAssistant — the always-on surgical assistant.

This is the layer that makes BRIDGE feel like an assistant rather than a
command line. It owns the persistent state of a case and decides, for every
spoken sentence, whether it can answer locally in microseconds or needs the
vision model.

Routing, in order:

    1. reference resolution      "the other one"  ->  "the other metzenbaum scissors"
    2. local intent grammar      project / count / case / status        (no network)
    3. scene-graph lookup        already know where it is?              (no network)
    4. procedure control words   the existing ProcedureGuide
    5. Gemini                    genuinely open-ended requests only

and *underneath* all of that, a monitor loop fed by the camera that can speak
first when the numbers stop adding up.

Boundaries that do not move:

* Dosing, diagnosis and treatment questions are refused in
  `healthcare.safety.check_request` before this class sees them, and refused
  again here for anything that reaches the AI path.
* The count is the team's, not BRIDGE's. Every count reply attributes the
  decision to the team and BRIDGE never says a count is correct on the strength
  of what the camera sees.
* Vision is advisory. Wording is always "I can see" / "I cannot see", never
  "it is there" / "it is gone".
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from bridge.assistant.announce import Announcer, Priority
from bridge.assistant.conversation import Conversation, extract_subject
from bridge.assistant.intents import HELP_TEXT, Intent, parse
from bridge.interaction.commands import (
    ClearProjection,
    CommandStyle,
    HighlightObject,
    LabelObject,
    PointToObject,
    ShowMessage,
    TargetSpec,
)
from bridge.perception.scene_graph import SceneDelta, SceneGraph, normalise_label, stable_gap
from bridge.render.overlay import OverlayLayer
from bridge.render.primitives import ACCENT, ALERT, WARNING
from bridge.spatial.geometry import BoundingBox, Point
from bridge.surgical.case import PHASE_TITLE, CaseSession
from bridge.surgical.counts import CountSession
from bridge.surgical.monitor import Alert, SafetyMonitor
from bridge.surgical.sets import SetLibrary
from bridge.surgical.trays import TrayLayout, Zone, infer_tray_zone
from bridge.vision.detection import ContourDetector, Detection

log = logging.getLogger("bridge.jarvis")

COUNT_PHASES = {"count_in": "initial", "count_closing": "closing", "count_final": "final"}


class JarvisAssistant:
    """Owns scene memory, the case, the counts and the proactive monitor."""

    def __init__(self, core, *, scan_hz: float = 5.0, ai_label_interval_s: float = 12.0,
                 monitor_interval_s: float = 1.0, proactive: bool = True,
                 records_dir: Path | str = Path("case_records")):
        self.core = core
        self.scene = SceneGraph()
        self.conversation = Conversation()
        self.sets = SetLibrary()
        self.case = CaseSession()
        self.layout = TrayLayout()
        self.monitor = SafetyMonitor(enabled=proactive)
        self.detector = ContourDetector()
        self.overlay = OverlayLayer(core.renderer)
        self.announcer = Announcer(core.speak, stop_speaking=getattr(core, "stop_speaking", None))
        self.records_dir = Path(records_dir)
        self.scan_interval_s = 1.0 / max(scan_hz, 0.5)
        self.ai_label_interval_s = ai_label_interval_s
        self.monitor_interval_s = monitor_interval_s
        self.proactive = proactive
        self.show_board = True
        self._last_scan = 0.0
        self._last_monitor = 0.0
        self._last_ai_label = 0.0
        self._ai_labelling = threading.Event()
        self._lock = threading.RLock()
        self.on_alert: Callable[[Alert], None] = lambda a: None
        self.on_state: Callable[[], None] = lambda: None
        self.last_alert: Optional[Alert] = None

    # ================================================================= lifecycle
    def start(self) -> None:
        self.announcer.start()

    def stop(self) -> None:
        self.announcer.stop()

    @property
    def counts(self) -> CountSession:
        return self.case.counts

    @property
    def phase(self) -> str:
        return self.case.phase

    def say(self, text: str, priority: Priority = "info") -> None:
        self.announcer.say(text, priority)

    # ================================================================= camera loop
    def on_frame(self, frame: np.ndarray) -> None:
        """Called from the camera thread. Cheap and throttled; never blocks."""
        now = time.time()
        if now - self._last_scan < self.scan_interval_s:
            return
        self._last_scan = now
        try:
            delta = self._scan(frame, now)
        except Exception:  # noqa: BLE001 - perception must never kill the camera thread
            log.exception("scene scan failed")
            return
        if now - self._last_monitor >= self.monitor_interval_s:
            self._last_monitor = now
            try:
                self._run_monitor(delta, now)
            except Exception:  # noqa: BLE001
                log.exception("monitor failed")
        if (self.core.ai is not None and self.case.active
                and now - self._last_ai_label >= self.ai_label_interval_s):
            self._last_ai_label = now
            self._label_scene_async(frame)

    def _scan(self, frame: np.ndarray, now: float) -> SceneDelta:
        clean = self.core.executor.clean_frame(frame)   # remove BRIDGE's own projection
        dets = self.detector.detect(clean)
        delta = self.scene.observe(dets, clean, now)
        if self.case.active:
            self.counts.observe(self.scene)
            if self.show_board:
                self._refresh_board()
        return delta

    def _run_monitor(self, delta: SceneDelta, now: float) -> None:
        alerts = self.monitor.evaluate(self.scene, self.counts if self.case.active else None,
                                       self.phase, self.layout, delta, now)
        for a in alerts:
            self._raise(a)

    def _raise(self, alert: Alert) -> None:
        self.last_alert = alert
        self.case.record_alert(alert.level, alert.key, alert.text)
        if alert.speak and self.proactive:
            self.announcer.say(alert.text, "critical" if alert.level == "critical" else "caution")
        if alert.project and alert.level == "critical":
            self.overlay.show_alert(alert.text, alert.level)
        try:
            self.on_alert(alert)
        except Exception:  # noqa: BLE001
            log.exception("alert handler failed")

    def _label_scene_async(self, frame: np.ndarray) -> None:
        """Ask the model what things are, in the background. Positions stay local."""
        if self._ai_labelling.is_set() or self.core.ai is None:
            return
        self._ai_labelling.set()
        snapshot = frame.copy()

        def work() -> None:
            try:
                scene = self.core.ai.understand_scene(snapshot)
                h, w = snapshot.shape[:2]
                boxes = [(o.label, o.box.to_pixels(w, h), o.confidence, o.visual_description)
                         for o in scene.objects if o.box is not None and o.label]
                if boxes:
                    n = self.scene.apply_ai(boxes)
                    log.info("Scene labelled by AI: %d objects", n)
                    self.on_state()
            except Exception as e:  # noqa: BLE001 - AI is optional for perception
                log.info("Background scene labelling skipped: %s", e)
            finally:
                self._ai_labelling.clear()

        threading.Thread(target=work, name="jarvis-scene-label", daemon=True).start()

    def refresh_labels_now(self) -> bool:
        frame = self.core.cameras.latest_frame()
        if frame is None or self.core.ai is None:
            return False
        self._label_scene_async(frame)
        return True

    # ================================================================= spoken entry point
    def handle(self, text: str) -> str:
        """Route one utterance. Returns the sentence to speak back."""
        raw = (text or "").strip()
        if not raw:
            return ""
        with self._lock:
            # The safety gate runs before anything else, including the local intent
            # grammar. Otherwise a pattern like "how many mg should I give" would be
            # answered as a count question and never reach the policy at all.
            declined = self._safety_gate(raw)
            if declined is not None:
                self.conversation.remember(raw, declined, intent="declined")
                return declined
            # A running procedure owns "next", "back" and "repeat" before the case
            # does — and on the raw words, because control words are literal and
            # must not be rewritten by reference resolution.
            if self.core.guide.active:
                reply = self.core.guide.handle_control_word(raw)
                if reply is not None:
                    self.conversation.remember(raw, reply, intent="procedure")
                    return reply
            resolved, another = self.conversation.resolve(raw)
            intent = parse(resolved, case_phase=self.phase,
                           counting=self.phase in COUNT_PHASES, in_checklist=self.case.in_checklist)
            log.info("Intent %s target=%r qty=%s (from %r)", intent.kind, intent.target, intent.quantity, raw)
            reply = self._dispatch(intent, resolved, another)
            subject = intent.target or extract_subject(resolved)
            self.conversation.case_context = self._case_context()
            self.conversation.remember(raw, reply, subject=subject if intent.kind == "locate" else "",
                                       intent=intent.kind)
            self.on_state()
            return reply

    def _safety_gate(self, text: str) -> Optional[str]:
        """Refuse dosing/diagnosis requests in code, before any interpretation.

        Returns the refusal to speak, or None to carry on. `core.ask` applies the
        same check again on the AI path — two independent gates, deliberately.
        """
        if self.core.settings.domain != "healthcare":
            return None
        from bridge.healthcare.safety import check_request

        verdict = check_request(text)
        if verdict.allowed:
            return None
        log.warning("Request declined by safety policy: %r", text)
        frame = self.core.cameras.latest_frame()
        if frame is not None:
            self.core.executor.execute(
                ShowMessage(text="Not a decision I can make.\nCheck the protocol or prescriber.", duration_s=6), frame)
        return verdict.reason

    def _dispatch(self, intent: Intent, text: str, another: bool) -> str:
        k = intent.kind
        if k == "locate":
            return self.locate(intent.target, another=another, action=intent.slots.get("action", "highlight"))
        if k == "clear":
            self.overlay.hide_gaps()
            self.overlay.hide_alert()
            self.core.execute(ClearProjection())
            return "Cleared."
        if k == "scene_describe":
            return self.describe_scene()
        if k == "whats_missing":
            return self.whats_missing()
        if k == "where_did_it_go":
            return self.where_did_it_go(intent.target)
        if k == "help":
            return HELP_TEXT
        if k == "mute":
            self.proactive = False
            self.announcer.clear()
            return "Alerts muted. I will still answer you. Say alerts on to bring them back."
        if k == "unmute":
            self.proactive = True
            return "Alerts back on."
        if k == "stop_talking":
            self.announcer.interrupt()
            return ""
        # ---- case ----
        if k == "case_start":
            return self.start_case(intent.target)
        if k == "case_confirm":
            return self.case.confirm(intent.text)
        if k == "case_skip":
            return self.case.skip()
        if k == "case_next":
            return self.case.go_to(intent.phase) if intent.phase else self.case.next_phase()  # type: ignore[arg-type]
        if k == "case_back":
            return self.case.back()
        if k == "case_repeat":
            return self.case.repeat() if self.case.active else (self.conversation.last_reply() or "Nothing to repeat.")
        if k == "case_status":
            return self.status()
        if k == "case_close":
            return self.close_case()
        # ---- counts ----
        if k == "count_begin":
            return self.begin_count(intent.slots.get("set", ""))
        if k == "count_phase":
            return self.count_phase(intent.phase)
        if k == "count_record":
            return self.record_count(intent.target, intent.quantity or 0)
        if k == "count_add":
            return self.add_to_field(intent.target, intent.quantity or 1)
        if k == "count_line_add":
            line, _ = self.counts.add_item(intent.target, 0)
            return f"{line.name} added to the count sheet. How many?"
        if k == "count_accept":
            return self.accept_sheet()
        if k == "count_status":
            return self.count_status(intent.target)
        # ---- fallthrough ----
        return self._fallback(text)

    # Phrases that predate the intent grammar and still have to work.
    _WHAT_NEXT = ("what next", "what's next", "whats next", "what should i do next",
                  "what should i pick up first", "next step", "what do i do next")
    _LIST_PROCEDURES = ("list procedures", "what procedures do you have", "which procedures")
    _CALIBRATE = ("calibrate", "run calibration", "start calibration")

    def _fallback(self, text: str) -> str:
        """Procedure control words, the legacy phrases, then the AI."""
        reply = self.core.guide.handle_control_word(text)
        if reply is not None:
            return reply
        t = text.lower().strip().rstrip("?.! ")
        if t in self._WHAT_NEXT:
            return self.core.guide.next() if self.core.guide.active else self.core.spoken_reply(self.core.next_step())
        if t in self._LIST_PROCEDURES:
            return "Available procedures: " + ", ".join(n for _, n in self.core.procedures.names()) + "."
        if t in self._CALIBRATE:
            return "Please press Auto Calibrate on the control panel; I cannot calibrate hands-free yet."
        if t in ("sets", "list sets", "which instrument sets", "what sets do you have"):
            return "Instrument sets: " + ", ".join(n for _, n in self.sets.names()) + "."
        if self.core.ai is None:
            ok, _ = self.core.set_ai_provider()
            if not ok:
                return ("I did not recognise that, and the AI service is not available. "
                        "Say help for the commands I know offline.")
        return self.core.spoken_reply(self.core.ask(text))

    # ================================================================= locating
    def locate(self, target: str, another: bool = False, action: str = "highlight") -> str:
        """Project a target. Scene graph first (instant), Gemini only if unknown.

        `action` picks the graphic: a reticle for "where is it", an arrow for
        "hand me that", a text label for "label it".
        """
        target = (target or "").strip()
        if not target:
            return "What should I project?"
        frame = self.core.cameras.latest_frame()
        if frame is None:
            return "I have no camera image right now."
        if not self.core.executor.calibrated:
            return "I need a spatial calibration before I can project onto the surface."
        skip = set(self.conversation.mentioned) if another else set()
        hits = [e for e in self.scene.find(target) if e.id not in skip]
        if not hits and another:
            hits = self.scene.find(target)          # only one exists; show it again
        if hits:
            ent = hits[0]
            name = ent.label or target
            spec = TargetSpec(label=name, description=ent.describe(), boxes_camera=[ent.bbox])
            if action == "point":
                cmd = PointToObject(target=spec)
                verb = "Pointing to"
            elif action == "label":
                cmd = LabelObject(target=spec, text=name)
                verb = "Labelled"
            else:
                cmd = HighlightObject(target=spec, style=CommandStyle(shape="circle", animation="pulse"),
                                      label_text=name)
                verb = "Found"
            res = self.core.execute(cmd, frame)
            self.conversation.remember(f"locate {target}", subject=target, entity_ids=[ent.id])
            if res.ok:
                extra = f" {len(hits) - 1} more on the tray." if len(hits) > 1 else ""
                cautions = " ".join(self._cautions(name))
                return f"{verb} {name}.{extra} {cautions}".strip()
            log.info("Local projection failed (%s); falling back to the model", res.message)
        # not in local memory — ask the model, which also teaches the scene graph
        query = {"point": f"point to the {target}", "label": f"label the {target}"}.get(
            action, f"highlight the {target}")
        reply = self.core.spoken_reply(self.core.ask(query))
        self._learn_from_last_identification(target)
        return reply

    def _learn_from_last_identification(self, target: str) -> None:
        """Fold whatever the model just found into the scene graph, so next time is instant."""
        ident = self.core.last_identification
        frame = self.core.cameras.latest_frame()
        if ident is None or frame is None or not ident.boxes:
            return
        h, w = frame.shape[:2]
        self.scene.apply_ai([(ident.target or target, b.to_pixels(w, h), ident.confidence,
                              ident.visual_description) for b in ident.boxes])

    def _cautions(self, label: str) -> list[str]:
        from bridge.healthcare.safety import cautions_for_target

        return cautions_for_target(label) if self.core.settings.domain == "healthcare" else []

    # ================================================================= scene questions
    def describe_scene(self) -> str:
        n = len(self.scene.present())
        if not n:
            return "I cannot see anything on the surface."
        known = self.scene.summary()
        unlabelled = sum(1 for e in self.scene.present() if not e.label)
        tail = f" {unlabelled} I cannot name yet." if unlabelled else ""
        return f"On the surface I can see {known}.{tail}"

    def whats_missing(self) -> str:
        """Project the empty slots and say what is not visible."""
        if not self.case.active:
            return "No case is running, so I have nothing to compare the tray against."
        gone: list[tuple[str, list[Point]]] = []
        for line, observed, expected in self.counts.visual_disagreements():
            if observed >= expected:
                continue
            slots = self.layout.empty_slots(self.scene, line.name)
            for s in slots[: expected - observed]:
                gone.append((line.name, s.outline()))
        if not gone:
            for ent in self.scene.absent():
                if ent.sticky or ent.label:
                    gone.append((ent.label or "item", stable_gap(ent)))
        if not gone:
            self.overlay.hide_gaps()
            return "Everything I have a position for is still on the tray. That is not a count."
        self.overlay.show_gaps([p for _, p in gone], self.core.executor.mapper,
                               labels=[n for n, _ in gone], colour=WARNING)
        names: dict[str, int] = {}
        for n, _ in gone:
            names[n] = names.get(n, 0) + 1
        listed = ", ".join(f"{v} {k}" if v > 1 else k for k, v in names.items())
        return (f"I cannot see {listed} on the tray. I have marked where each one was. "
                "Confirm they are in use, or recount.")

    def where_did_it_go(self, target: str) -> str:
        ent = self.scene.find_one(target, present_only=False)
        if ent is None:
            return f"I have no memory of {target} on this surface."
        if ent.present:
            return self.locate(target)
        secs = int(ent.seconds_since_seen())
        self.overlay.show_gaps([stable_gap(ent)], self.core.executor.mapper,
                               labels=[ent.label or target], colour=WARNING)
        when = f"{secs} seconds ago" if secs < 90 else f"{secs // 60} minutes ago"
        return f"I last saw {ent.describe()} {when}. I have marked the spot."

    # ================================================================= case control
    def start_case(self, spoken: str = "") -> str:
        with self._lock:
            iset = self.sets.find(spoken) if spoken else None
            self.case = CaseSession(instrument_set=iset, procedure_name=spoken)
            self.case.on_phase = lambda p: self.on_state()
            self.monitor.reset()
            self.scene.clear()
            self.layout.clear()
            self.overlay.clear()
            self.conversation.clear()
            opening = self.case.start(spoken, iset)
            set_note = (f"Using the {iset.name}, {iset.total} items." if iset
                        else "No instrument set matched, so I will build the count sheet from what you call out.")
            self.announcer.say("Case started. " + set_note, "reply")
            return opening

    def close_case(self) -> str:
        with self._lock:
            if not self.case.active:
                return "No case is running."
            rec, spoken = self.case.close()
            self.overlay.clear()
            try:
                path = self.case.save(self.records_dir)
                where = f" Record saved as {path.name}."
            except OSError as e:
                log.warning("Could not write the case record: %s", e)
                where = " I could not write the case record to disk."
            if rec.verdict == "discrepancy":
                self.announcer.say(spoken, "critical")
            return spoken + where

    def status(self) -> str:
        if not self.case.active:
            return "No case is running. Say start a case to begin."
        title = PHASE_TITLE[self.phase]
        bits = [f"{title}."]
        item = self.case.current_item
        if item is not None:
            i, n = self.case.progress()
            bits.append(f"Item {i + 1} of {n}: {item.prompt}")
        if self.phase in COUNT_PHASES or self.phase == "procedure":
            counted, expected = self.counts.reconcile(self.counts.phase).totals
            bits.append(f"Count sheet: {counted} of {expected} accounted for.")
        return " ".join(bits)

    # ================================================================= counts
    def begin_count(self, set_spoken: str = "") -> str:
        with self._lock:
            if set_spoken:
                iset = self.sets.find(set_spoken)
                if iset is not None:
                    self.case.counts = CountSession(iset, case_id=self.case.case_id)
            if not self.case.active:
                self.case.start(self.case.procedure_name)
            self.case.go_to("count_in")
            self._capture_layout()
            n = len(self.counts.lines)
            src = self.counts.set.name if self.counts.set else "an empty sheet"
            self._refresh_board(force=True)
            return (f"Initial count, {src}, {n} lines. Call each item and its number. "
                    "Say count as per the sheet if the tray matches exactly. "
                    "Your count sheet is the record; I am recording alongside it.")

    def count_phase(self, phase: str) -> str:
        with self._lock:
            target = "count_closing" if phase == "closing" else "count_final"
            if not self.case.active:
                return "No case is running."
            self.case.go_to(target)  # type: ignore[arg-type]
            self._refresh_board(force=True)
            return self.case.announce()

    def record_count(self, item: str, quantity: int) -> str:
        with self._lock:
            line, ack = self.counts.record(item, quantity)
            self._refresh_board(force=True)
            if line is None:
                return ack
            if line.status_at(self.counts.phase) in ("short", "over") and self.counts.phase in ("closing", "final"):
                self.announcer.say(ack, "critical")
            return ack

    def add_to_field(self, item: str, quantity: int = 1) -> str:
        with self._lock:
            line, ack = self.counts.add_item(item, quantity)
            self._refresh_board(force=True)
            return ack

    def accept_sheet(self) -> str:
        with self._lock:
            phase = COUNT_PHASES.get(self.phase, "initial")
            rec = self.counts.accept_expected(phase)  # type: ignore[arg-type]
            self._capture_layout()
            self._refresh_board(force=True)
            return rec.spoken()

    def count_status(self, item: str = "") -> str:
        if not self.case.active:
            return "No case is running, so there is no count sheet."
        if item:
            line = self.counts.line(item)
            if line is None:
                return f"{item} is not on the count sheet."
            got = line.counted_at(self.counts.phase)
            seen = f" I can see {line.observed} on the tray." if line.observed else ""
            if got is None:
                return f"{line.name}: not counted yet this phase, baseline {line.baseline}.{seen}"
            return f"{line.name}: {got} of {line.baseline}, {line.status_at(self.counts.phase)}.{seen}"
        return self.counts.reconcile(self.counts.phase).spoken()

    def reconcile_now(self) -> str:
        rec = self.counts.reconcile()
        self._refresh_board(force=True)
        if rec.verdict == "discrepancy":
            self.announcer.say(rec.spoken(), "critical")
        return rec.spoken()

    # ================================================================= tray / zones
    def _capture_layout(self) -> int:
        def name_for(ent):
            line = self.counts._line_for_entity(ent)  # noqa: SLF001 - same package boundary
            return line.name if line else (ent.label or None)

        n = self.layout.capture(self.scene, name_for)
        tray = infer_tray_zone(self.scene)
        if tray is not None:
            self.layout.set_zone(tray)
        return n

    def set_zone_from_box(self, name: str, box: BoundingBox, kind: str = "custom") -> None:
        self.layout.set_zone(Zone.from_box(name, box, kind))  # type: ignore[arg-type]

    def _case_context(self) -> str:
        if not self.case.active:
            return ""
        bits = [PHASE_TITLE[self.phase].lower()]
        if self.case.procedure_name:
            bits.insert(0, self.case.procedure_name)
        return ", ".join(bits)

    # ================================================================= HUD
    def _refresh_board(self, force: bool = False) -> None:
        if not self.show_board or not self.case.active:
            self.overlay.hide_board()
            return
        rows = self.counts.board()
        title = f"{PHASE_TITLE[self.phase].upper()}   {self.counts.summary_line()}"
        footer = "team count is the record"
        self.overlay.show_board(title, rows, footer, force=force)

    def set_board_visible(self, visible: bool) -> None:
        self.show_board = visible
        if visible:
            self._refresh_board(force=True)
        else:
            self.overlay.hide_board()

    # ================================================================= reporting
    def report(self) -> str:
        return self.case.report_text() if self.case.active or self.case.confirmations else "No case recorded yet."

    def snapshot(self) -> dict:
        """Everything the UI panel needs, in one thread-safe read."""
        counted, expected = self.counts.reconcile(self.counts.phase).totals if self.case.active else (0, 0)
        return {
            "phase": self.phase,
            "phase_title": PHASE_TITLE[self.phase],
            "case_id": self.case.case_id,
            "procedure": self.case.procedure_name,
            "set": self.counts.set.name if self.counts.set else "",
            "counted": counted,
            "expected": expected,
            "verdict": self.counts.reconcile(self.counts.phase).verdict if self.case.active else "pending",
            "entities": len(self.scene.present()),
            "labelled": len(self.scene.labelled()),
            "alerts": len(self.monitor.raised),
            "proactive": self.proactive,
            "checklist_item": self.case.current_item.prompt if self.case.current_item else "",
            "board": self.counts.board() if self.case.active else [],
        }
