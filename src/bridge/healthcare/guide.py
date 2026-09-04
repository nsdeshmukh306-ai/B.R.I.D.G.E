"""ProcedureGuide: drives a Procedure through the task state machine, projecting
each step's target and speaking the instruction. Voice words: next, back,
repeat, stop. Steps with verification=object_removed advance automatically
when the tracker reports the highlighted item gone.
"""
from __future__ import annotations

import logging
import re
import threading
from typing import Callable, Optional

from bridge.ai.schemas import TargetIdentification
from bridge.healthcare.procedures import Procedure, ProcedureLibrary, ProcedureStep
from bridge.interaction.commands import ShowMessage
from bridge.interaction.tasks import TaskRunner, TaskState

log = logging.getLogger("bridge.healthcare.guide")

NEXT_WORDS = ("next", "done", "counted", "got it", "okay next", "continue", "ready")
BACK_WORDS = ("back", "previous", "go back")
REPEAT_WORDS = ("repeat", "again", "say again", "what was that")
STOP_WORDS = ("stop procedure", "stop the procedure", "cancel procedure", "end procedure", "abort")


class ProcedureGuide:
    def __init__(self, core, library: ProcedureLibrary | None = None,
                 speak: Callable[[str], None] | None = None):
        self.core = core
        self.library = library or ProcedureLibrary()
        self.speak = speak or (lambda s: None)
        self.procedure: Optional[Procedure] = None
        self.runner: Optional[TaskRunner] = None
        self._lock = threading.RLock()
        self.on_step: Callable[[Optional[ProcedureStep], int, int], None] = lambda step, i, n: None
        core.bus.subscribe(__import__("bridge.app.events", fromlist=["Topic"]).Topic.TRACKING_LOST, self._on_tracking_lost)

    # -- state ------------------------------------------------------------------------
    @property
    def active(self) -> bool:
        return self.runner is not None and self.runner.state not in (TaskState.IDLE, TaskState.DONE)

    @property
    def step_index(self) -> int:
        return self.runner.step_index if self.runner else -1

    @property
    def current_step(self) -> Optional[ProcedureStep]:
        if self.procedure is None or self.runner is None:
            return None
        i = self.runner.step_index
        return self.procedure.steps[i] if 0 <= i < len(self.procedure.steps) else None

    # -- control ----------------------------------------------------------------------
    def start(self, procedure: Procedure | str) -> str:
        with self._lock:
            proc = procedure if isinstance(procedure, Procedure) else (self.library.get(procedure) or self.library.find(procedure))
            if proc is None:
                return f"I don't have a procedure called {procedure}."
            self.stop(quiet=True)
            self.procedure = proc
            self.runner = TaskRunner(proc.to_task())
            self.runner.start()
            log.info("Procedure started: %s (%d steps)", proc.name, len(proc.steps))
            intro = proc.intro or f"Starting {proc.name}."
            spoken = self._present_step()
            return f"{intro} {spoken}".strip()

    def next(self) -> str:
        with self._lock:
            if not self.active:
                return "No procedure is running."
            assert self.runner is not None
            if self.runner.state is TaskState.UNDERSTAND:
                self.runner.located()  # step was never located (message-only) – move on
            nxt = self.runner.action_observed(True)
            if nxt is None:
                outro = self.procedure.outro if self.procedure else "Procedure complete."
                self.core.execute(ShowMessage(text=outro, duration_s=8))
                self.on_step(None, len(self.procedure.steps) if self.procedure else 0, len(self.procedure.steps) if self.procedure else 0)
                return outro
            return self._present_step()

    def back(self) -> str:
        with self._lock:
            if not self.active or self.runner is None:
                return "No procedure is running."
            if self.runner.step_index <= 0:
                return self._present_step()
            # rewind: restart the runner at the previous index
            proc = self.procedure
            assert proc is not None
            idx = self.runner.step_index - 1
            self.runner = TaskRunner(proc.to_task())
            self.runner.start()
            for _ in range(idx):
                self.runner.located()
                self.runner.action_observed(True)
            return self._present_step()

    def repeat(self) -> str:
        with self._lock:
            return self._present_step() if self.active else "No procedure is running."

    def stop(self, quiet: bool = False) -> str:
        with self._lock:
            was_active = self.active
            if self.runner is not None:
                self.runner.abort()
            self.runner, self.procedure = None, None
            self.core.executor.clear()
            if not quiet:
                self.on_step(None, 0, 0)
            return "Procedure stopped." if was_active else "No procedure is running."

    def handle_control_word(self, text: str) -> Optional[str]:
        """If `text` is a procedure control phrase, act on it and return the reply; else None."""
        t = text.lower().strip().rstrip(".!?")
        m = re.match(r"^(?:start|begin|open|run)\s+(?:the\s+)?(.+?)(?:\s+procedure|\s+checklist)?$", t)
        if m and (self.library.find(m.group(1)) is not None):
            return self.start(self.library.find(m.group(1)))  # type: ignore[arg-type]
        if any(t == w or t.startswith(w + " ") for w in STOP_WORDS) or (t == "stop" and self.active):
            return self.stop()
        if not self.active:
            return None
        if t in NEXT_WORDS or t.startswith("next"):
            return self.next()
        if t in BACK_WORDS:
            return self.back()
        if t in REPEAT_WORDS:
            return self.repeat()
        return None

    # -- internals ----------------------------------------------------------------------
    def _present_step(self) -> str:
        step = self.current_step
        proc = self.procedure
        if step is None or proc is None or self.runner is None:
            return ""
        i, n = self.runner.step_index, len(proc.steps)
        self.on_step(step, i, n)
        header = f"Step {i + 1} of {n}"
        if not step.target_object or step.projection_action == "message":
            self.core.execute(ShowMessage(text=f"{header}\n{step.instruction}", duration_s=0))
            self._mark_located()
            return step.instruction
        # Locate the target through the normal AI -> resolver -> tracker path.
        intent = {"highlight_object": "highlight_object", "point_to_object": "point_to_object",
                  "show_target_zone": "show_target_zone", "draw_path": "highlight_object"}[step.projection_action]
        query = f"{'Highlight' if intent == 'highlight_object' else 'Point to' if intent == 'point_to_object' else 'Show where to put'} the {step.target_object}"
        if step.visual_hint:
            query += f" ({step.visual_hint})"
        result = self.core.ask(query, label_override=f"{i + 1}/{n} {step.target_object}")
        self._mark_located()
        found = result.ok and result.n_targets > 0
        spoken = f"{header}. {step.instruction}"
        if not found:
            spoken += f" I could not see the {step.target_object}; say next when you have it."
        if step.caution and step.caution.lower().rstrip(".") not in step.instruction.lower():
            spoken += f" {step.caution}"
        return spoken

    def _mark_located(self) -> None:
        if self.runner is not None and self.runner.state is TaskState.UNDERSTAND:
            self.runner.located()

    def _on_tracking_lost(self, _ev) -> None:
        step = self.current_step
        if step is None or self.runner is None:
            return
        if step.verification == "object_removed" and step.expected_action == "pick_up" and self.runner.state is TaskState.WAIT_FOR_ACTION:
            log.info("Step verified by removal: %s", step.target_object)
            reply = self.next()
            if reply:
                self.speak(reply)
