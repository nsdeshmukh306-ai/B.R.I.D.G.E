"""Announcer: a speech queue that knows what is worth interrupting for.

Proactive alerts and conversational replies compete for one voice. Without a
policy the result is BRIDGE talking over itself, or a critical count alert
queued behind a description of the tray.

Policy:
* **critical** preempts — the current utterance is cut off and the queue is
  cleared ahead of it.
* **reply** (an answer to something the user just said) goes to the front of the
  normal queue: the user is waiting.
* **caution** waits its turn and is dropped if it is older than `stale_after_s`
  when it reaches the front — a warning about a state that has since changed is
  worse than silence.
* Identical text within `dedupe_s` is dropped outright.

The queue runs on its own thread so a slow TTS engine never blocks the camera,
the voice loop or the UI.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

log = logging.getLogger("bridge.assistant.announce")

Priority = Literal["critical", "reply", "caution", "info"]
_RANK: dict[Priority, int] = {"critical": 0, "reply": 1, "caution": 2, "info": 3}


@dataclass(order=True)
class _Item:
    rank: int
    seq: int
    text: str = field(compare=False)
    priority: Priority = field(default="info", compare=False)
    ts: float = field(default_factory=time.time, compare=False)


class Announcer:
    """Priority speech queue on top of any object with `.speak(text)`."""

    def __init__(self, speak: Callable[[str], None], *, stale_after_s: float = 12.0,
                 dedupe_s: float = 20.0, stop_speaking: Optional[Callable[[], None]] = None,
                 max_queue: int = 24):
        self._speak = speak
        self._stop_speaking = stop_speaking
        self.stale_after_s = stale_after_s
        self.dedupe_s = dedupe_s
        self.max_queue = max_queue
        self._q: "queue.PriorityQueue[_Item]" = queue.PriorityQueue()
        self._recent: dict[str, float] = {}
        self._seq = 0
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.spoken: list[tuple[float, Priority, str]] = []   # transcript, for the UI and tests
        self.on_spoken: Callable[[str, Priority], None] = lambda text, prio: None

    # -- lifecycle ---------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="bridge-announcer", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._q.put(_Item(-1, -1, "", "critical"))   # wake the loop
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- enqueue -----------------------------------------------------------------------
    def say(self, text: str, priority: Priority = "info") -> bool:
        """Queue an utterance. Returns False if it was dropped (duplicate/full)."""
        text = (text or "").strip()
        if not text:
            return False
        now = time.time()
        with self._lock:
            last = self._recent.get(text)
            if last is not None and now - last < self.dedupe_s and priority != "critical":
                return False
            self._recent[text] = now
            if len(self._recent) > 128:
                self._recent = {k: v for k, v in self._recent.items() if now - v < self.dedupe_s}
            if self._q.qsize() >= self.max_queue and priority in ("info", "caution"):
                return False
            self._seq += 1
            item = _Item(_RANK[priority], self._seq, text, priority, now)
        if priority == "critical":
            self.interrupt()
        self._q.put(item)
        if not self.running:
            # Nothing is draining the queue (not started, or the thread died), so
            # speak inline rather than losing the utterance. Priority ordering only
            # holds across items that are queued together — see drain().
            self._drain_sync()
        return True

    def drain(self) -> None:
        """Speak everything queued, in priority order. For synchronous use."""
        self._drain_sync()

    def interrupt(self) -> None:
        """Cut off the current utterance and drop everything non-critical queued."""
        if self._stop_speaking is not None:
            try:
                self._stop_speaking()
            except Exception:  # noqa: BLE001
                log.exception("stop_speaking failed")
        kept: list[_Item] = []
        while True:
            try:
                it = self._q.get_nowait()
            except queue.Empty:
                break
            if it.priority == "critical" and it.text:
                kept.append(it)
        for it in kept:
            self._q.put(it)

    def clear(self) -> None:
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                break

    @property
    def pending(self) -> int:
        return self._q.qsize()

    # -- draining ----------------------------------------------------------------------
    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                continue
            if self._stop.is_set() or not item.text:
                break
            self._emit(item)

    def _drain_sync(self) -> None:
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                return
            if item.text:
                self._emit(item)

    def _emit(self, item: _Item) -> None:
        age = time.time() - item.ts
        if item.priority in ("caution", "info") and age > self.stale_after_s:
            log.debug("Dropped stale announcement (%.1fs): %s", age, item.text[:60])
            return
        self.spoken.append((item.ts, item.priority, item.text))
        self.spoken = self.spoken[-200:]
        try:
            self._speak(item.text)
        except Exception:  # noqa: BLE001 - a broken TTS must not kill the queue
            log.exception("speak failed")
        try:
            self.on_spoken(item.text, item.priority)
        except Exception:  # noqa: BLE001
            log.exception("on_spoken handler failed")
