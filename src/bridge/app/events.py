"""Thread-safe publish/subscribe event bus decoupling subsystems."""
from __future__ import annotations

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

log = logging.getLogger("bridge.events")


class Topic(str, Enum):
    CAMERA_CONNECTED = "camera.connected"
    CAMERA_DISCONNECTED = "camera.disconnected"
    CAMERA_FRAME = "camera.frame"
    DISPLAY_CONNECTED = "display.connected"
    DISPLAY_CHANGED = "display.changed"
    CALIBRATION_STARTED = "calibration.started"
    CALIBRATION_PROGRESS = "calibration.progress"
    CALIBRATION_COMPLETE = "calibration.complete"
    CALIBRATION_FAILED = "calibration.failed"
    CALIBRATION_INVALIDATED = "calibration.invalidated"
    TRACKING_UPDATE = "tracking.update"
    TRACKING_LOST = "tracking.lost"
    AI_REQUEST = "ai.request"
    AI_RESPONSE = "ai.response"
    AI_ERROR = "ai.error"
    COMMAND_EXECUTED = "command.executed"
    STATUS_MESSAGE = "status.message"
    SCENE_UPDATED = "scene.updated"
    VOICE_EVENT = "voice.event"
    PROCEDURE_STEP = "procedure.step"


@dataclass
class Event:
    topic: Topic
    payload: dict[str, Any] = field(default_factory=dict)


Handler = Callable[[Event], None]


class EventBus:
    def __init__(self) -> None:
        self._subs: dict[Topic, list[Handler]] = defaultdict(list)
        self._lock = threading.RLock()

    def subscribe(self, topic: Topic, handler: Handler) -> Callable[[], None]:
        with self._lock:
            self._subs[topic].append(handler)

        def unsubscribe() -> None:
            with self._lock:
                if handler in self._subs[topic]:
                    self._subs[topic].remove(handler)

        return unsubscribe

    def publish(self, topic: Topic, **payload: Any) -> None:
        with self._lock:
            handlers = list(self._subs.get(topic, []))
        ev = Event(topic, payload)
        for h in handlers:
            try:
                h(ev)
            except Exception:  # noqa: BLE001 - a bad subscriber must not break the bus
                log.exception("Event handler failed for %s", topic.value)
