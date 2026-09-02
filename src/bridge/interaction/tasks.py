"""Task state machine for guided assembly / training (architecture for V2 workflows)."""
from __future__ import annotations

import logging
import time
from enum import Enum
from typing import Callable, Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("bridge.tasks")


class TaskStep(BaseModel):
    instruction: str
    target_object: str
    expected_action: Literal["pick_up", "place", "connect", "remove", "observe"] = "pick_up"
    projection_action: Literal["highlight_object", "point_to_object", "show_target_zone", "draw_path"] = "highlight_object"
    verification_method: Literal["none", "object_removed", "object_present", "manual"] = "none"


class Task(BaseModel):
    id: str
    name: str
    steps: list[TaskStep] = Field(default_factory=list)


class TaskState(str, Enum):
    IDLE = "IDLE"
    UNDERSTAND = "UNDERSTAND"
    LOCATE = "LOCATE"
    PROJECT = "PROJECT"
    WAIT_FOR_ACTION = "WAIT_FOR_ACTION"
    VERIFY = "VERIFY"
    NEXT_STEP = "NEXT_STEP"
    DONE = "DONE"


TRANSITIONS: dict[TaskState, set[TaskState]] = {
    TaskState.IDLE: {TaskState.UNDERSTAND},
    TaskState.UNDERSTAND: {TaskState.LOCATE, TaskState.IDLE, TaskState.DONE},
    TaskState.LOCATE: {TaskState.PROJECT, TaskState.UNDERSTAND},
    TaskState.PROJECT: {TaskState.WAIT_FOR_ACTION},
    TaskState.WAIT_FOR_ACTION: {TaskState.VERIFY, TaskState.IDLE},
    TaskState.VERIFY: {TaskState.NEXT_STEP, TaskState.WAIT_FOR_ACTION, TaskState.LOCATE},
    TaskState.NEXT_STEP: {TaskState.UNDERSTAND, TaskState.DONE},
    TaskState.DONE: {TaskState.IDLE},
}


class TaskRunner:
    """Drives a Task through the state machine. Verification hooks are injected."""

    def __init__(self, task: Task, on_state: Callable[[TaskState, Optional[TaskStep]], None] | None = None):
        self.task = task
        self.state = TaskState.IDLE
        self.step_index = -1
        self.on_state = on_state
        self.history: list[tuple[float, TaskState]] = []

    @property
    def current_step(self) -> Optional[TaskStep]:
        if 0 <= self.step_index < len(self.task.steps):
            return self.task.steps[self.step_index]
        return None

    def transition(self, new: TaskState) -> None:
        if new not in TRANSITIONS[self.state]:
            raise ValueError(f"illegal transition {self.state.value} -> {new.value}")
        log.info("Task %s: %s -> %s", self.task.id, self.state.value, new.value)
        self.state = new
        self.history.append((time.time(), new))
        if self.on_state:
            self.on_state(new, self.current_step)

    def start(self) -> Optional[TaskStep]:
        self.step_index = 0
        self.transition(TaskState.UNDERSTAND)
        if not self.task.steps:
            self.transition(TaskState.DONE)
            return None
        return self.current_step

    def located(self) -> None:
        self.transition(TaskState.LOCATE)
        self.transition(TaskState.PROJECT)
        self.transition(TaskState.WAIT_FOR_ACTION)

    def action_observed(self, verified: bool) -> Optional[TaskStep]:
        self.transition(TaskState.VERIFY)
        if not verified:
            self.transition(TaskState.WAIT_FOR_ACTION)
            return self.current_step
        self.transition(TaskState.NEXT_STEP)
        self.step_index += 1
        if self.step_index >= len(self.task.steps):
            self.transition(TaskState.DONE)
            return None
        self.transition(TaskState.UNDERSTAND)
        return self.current_step

    def abort(self) -> None:
        if self.state in (TaskState.UNDERSTAND, TaskState.WAIT_FOR_ACTION, TaskState.DONE):
            self.transition(TaskState.IDLE)
        else:
            self.state = TaskState.IDLE
