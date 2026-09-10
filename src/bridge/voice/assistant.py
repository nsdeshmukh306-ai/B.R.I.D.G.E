"""VoiceAssistant: microphone -> VAD -> STT -> wake word -> command handler -> TTS.

Runs on its own thread. The command handler is injected (BridgeCore.handle_spoken)
and returns the sentence to speak back. Capture pauses while BRIDGE is speaking
so it never transcribes its own voice.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from bridge.ai.base import AIError
from bridge.voice.audio import AudioSource, UtteranceCapture, Utterance
from bridge.voice.stt import SpeechToText, Transcript
from bridge.voice.tts import TextToSpeech

log = logging.getLogger("bridge.voice")


class VoiceState(str, Enum):
    OFF = "off"
    LISTENING = "listening"
    HEARD = "heard"          # utterance captured, transcribing
    THINKING = "thinking"    # command being executed
    SPEAKING = "speaking"
    ERROR = "error"


@dataclass
class VoiceEvent:
    state: VoiceState
    transcript: Optional[Transcript] = None
    response: str = ""
    detail: str = ""
    ts: float = field(default_factory=time.time)


CommandHandler = Callable[[str], str]  # spoken English command -> sentence to say back


class VoiceAssistant:
    def __init__(self, source_factory: Callable[[], AudioSource], stt: SpeechToText, tts: TextToSpeech,
                 handler: CommandHandler, wake_word: str = "bridge", require_wake_word: bool = False,
                 on_event: Callable[[VoiceEvent], None] | None = None, capture: UtteranceCapture | None = None,
                 min_confidence: float = 0.3, barge_in: bool = True, speak_reply: bool = True):
        self._source_factory = source_factory
        self.stt, self.tts, self.handler = stt, tts, handler
        self.wake_word = wake_word.lower().strip()
        self.require_wake_word = require_wake_word
        self.barge_in = barge_in
        self.speak_reply = speak_reply      # False when an Announcer owns the voice channel
        self._speaking_text = ""
        self.barge_ins = 0
        self.on_event = on_event or (lambda e: None)
        self.capture = capture or UtteranceCapture()
        self.min_confidence = min_confidence
        self.state = VoiceState.OFF
        self._stop = threading.Event()
        self._paused = threading.Event()  # set while speaking
        self._thread: Optional[threading.Thread] = None
        self._source: Optional[AudioSource] = None
        self.history: list[VoiceEvent] = []
        self.tts.on_speaking = self._on_speaking

    # -- lifecycle ------------------------------------------------------------------------
    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._source = self._source_factory()  # raises RuntimeError if no microphone
        self._thread = threading.Thread(target=self._loop, name="voice-assistant", daemon=True)
        self._thread.start()
        self._set(VoiceState.LISTENING)

    def stop(self) -> None:
        self._stop.set()
        if self._source is not None:
            self._source.close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None
        self._set(VoiceState.OFF)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # -- internals ------------------------------------------------------------------------
    def _set(self, state: VoiceState, **kw) -> None:
        self.state = state
        ev = VoiceEvent(state, **kw)
        self.history.append(ev)
        self.history = self.history[-100:]
        try:
            self.on_event(ev)
        except Exception:  # noqa: BLE001
            log.exception("voice event handler failed")

    def _on_speaking(self, speaking: bool) -> None:
        if speaking:
            self._paused.set()
            self._set(VoiceState.SPEAKING)
        else:
            self._paused.clear()
            if self.running:
                self._set(VoiceState.LISTENING)

    def _loop(self) -> None:
        assert self._source is not None
        while not self._stop.is_set():
            utt = self.capture.next_utterance(self._source, self._stop)
            if utt is None:
                if self._stop.is_set() or getattr(self._source, "exhausted", False):
                    break
                continue
            if self._paused.is_set() and not self.barge_in:
                continue  # our own voice coming back through the microphone
            self.process_utterance(utt)

    def process_utterance(self, utt: Utterance) -> Optional[str]:
        """Transcribe + route one utterance; returns the spoken response (also used by push-to-talk)."""
        self._set(VoiceState.HEARD, detail=f"{utt.duration_s:.1f}s")
        try:
            tr = self.stt.transcribe(utt.wav_bytes())
        except AIError as e:
            self._set(VoiceState.ERROR, detail=str(e))
            self._set(VoiceState.LISTENING)
            return None
        if not tr.is_speech or not tr.command_text or tr.confidence < self.min_confidence:
            log.info("Ignored utterance (no speech / low confidence): %r", tr.text)
            self._set(VoiceState.LISTENING, transcript=tr)
            return None
        text = tr.command_text
        if self._paused.is_set():
            # BRIDGE is talking. Either this is our own voice echoing back, or the
            # user is cutting in. Echo is recognisable: it repeats what we are saying.
            if is_echo(text, self._speaking_text):
                log.debug("Ignored echo of our own speech: %r", text)
                return None
            log.info("Barge-in: %r", text)
            self.barge_ins += 1
            self.tts.stop()
            self._paused.clear()
        if self.require_wake_word:
            stripped = strip_wake_word(text, self.wake_word)
            if stripped is None:
                log.info("Ignored (no wake word '%s'): %r", self.wake_word, text)
                self._set(VoiceState.LISTENING, transcript=tr)
                return None
            text = stripped
        else:
            text = strip_wake_word(text, self.wake_word) or text
        self._set(VoiceState.THINKING, transcript=tr)
        try:
            response = self.handler(text)
        except Exception as e:  # noqa: BLE001
            log.exception("voice command failed")
            response = "Sorry, that command failed."
            self._set(VoiceState.ERROR, transcript=tr, detail=str(e))
        self._set(VoiceState.LISTENING, transcript=tr, response=response)
        if response and self.speak_reply:
            self.tts.speak(response)
        return response


def is_echo(heard: str, spoken: str, threshold: float = 0.5) -> bool:
    """True when a captured utterance is mostly BRIDGE's own words coming back.

    Cheap acoustic echo rejection: compare word sets. A real interruption ("stop",
    "no, the other one") shares almost nothing with the sentence being spoken,
    while a microphone picking up the speaker repeats it nearly verbatim.
    """
    if not spoken or not heard:
        return False
    h = {w for w in re.sub(r"[^a-z0-9 ]+", " ", heard.lower()).split() if len(w) > 2}
    s = {w for w in re.sub(r"[^a-z0-9 ]+", " ", spoken.lower()).split() if len(w) > 2}
    if not h or not s:
        return False
    return len(h & s) / len(h) >= threshold


def strip_wake_word(text: str, wake_word: str) -> Optional[str]:
    """Return the command without a leading wake word, or None if the wake word is absent."""
    if not wake_word:
        return text
    pattern = r"^\s*(?:hey|ok|okay|hi)?\s*" + re.escape(wake_word) + r"[\s,.:!?]*"
    m = re.match(pattern, text, re.IGNORECASE)
    if m is None:
        return None
    return text[m.end():].strip()
