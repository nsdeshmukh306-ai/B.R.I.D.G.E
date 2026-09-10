"""Text-to-speech (pluggable). Pyttsx3TTS uses the OS voice (SAPI5 on Windows) offline."""
from __future__ import annotations

import logging
import queue
import threading
from abc import ABC, abstractmethod
from typing import Callable, Optional

log = logging.getLogger("bridge.voice.tts")


class TextToSpeech(ABC):
    name = "tts"

    def __init__(self) -> None:
        self.on_speaking: Optional[Callable[[bool], None]] = None  # True when speech starts, False when done

    @abstractmethod
    def speak(self, text: str) -> None: ...

    def stop(self) -> None:
        return None

    def close(self) -> None:
        return None


class Pyttsx3TTS(TextToSpeech):
    """Speaks from a dedicated thread (pyttsx3 engines are not thread-safe)."""

    name = "pyttsx3"

    def __init__(self, rate: int = 175, volume: float = 1.0, voice_hint: str | None = None):
        super().__init__()
        try:
            import pyttsx3  # noqa: F401
        except ImportError as e:
            raise RuntimeError("pyttsx3 is not installed (pip install pyttsx3)") from e
        self.rate, self.volume, self.voice_hint = rate, volume, voice_hint
        self._q: queue.Queue[Optional[str]] = queue.Queue()
        self._ready = threading.Event()
        self._error: Optional[str] = None
        self._engine = None          # set by the worker thread; used by stop() for barge-in
        self.speaking = False
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()
        self._ready.wait(5.0)
        if self._error:
            raise RuntimeError(self._error)

    def _run(self) -> None:
        import pyttsx3

        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)
            if self.voice_hint:
                for v in engine.getProperty("voices"):
                    if self.voice_hint.lower() in (v.name or "").lower():
                        engine.setProperty("voice", v.id)
                        break
        except Exception as e:  # noqa: BLE001
            self._error = f"TTS engine failed to start: {e}"
            self._ready.set()
            return
        self._engine = engine
        self._ready.set()
        while True:
            text = self._q.get()
            if text is None:
                break
            try:
                self.speaking = True
                if self.on_speaking:
                    self.on_speaking(True)
                engine.say(text)
                engine.runAndWait()
            except Exception as e:  # noqa: BLE001
                log.warning("TTS failed: %s", e)
            finally:
                self.speaking = False
                if self.on_speaking:
                    self.on_speaking(False)

    def speak(self, text: str) -> None:
        if text.strip():
            self._q.put(text.strip())

    def stop(self) -> None:
        """Barge-in: drop everything queued and cut off the current utterance."""
        while True:
            try:
                item = self._q.get_nowait()
            except queue.Empty:
                break
            if item is None:            # keep a pending shutdown request
                self._q.put(None)
                break
        eng = self._engine
        if eng is not None:
            try:
                eng.stop()
            except Exception as e:  # noqa: BLE001 - some drivers refuse stop() mid-utterance
                log.debug("TTS stop ignored: %s", e)

    def close(self) -> None:
        self._q.put(None)


class SilentTTS(TextToSpeech):
    """Used when speech output is disabled or unavailable; records what would have been said."""

    name = "silent"

    def __init__(self) -> None:
        super().__init__()
        self.spoken: list[str] = []
        self.stops = 0
        self.speaking = False

    def speak(self, text: str) -> None:
        if text.strip():
            self.spoken.append(text.strip())
            log.info("TTS (silent): %s", text.strip())
            if self.on_speaking:
                self.on_speaking(True)
                self.on_speaking(False)

    def stop(self) -> None:
        self.stops += 1


def build_tts(enabled: bool, rate: int = 175, voice_hint: str | None = None) -> TextToSpeech:
    if not enabled:
        return SilentTTS()
    try:
        return Pyttsx3TTS(rate=rate, voice_hint=voice_hint)
    except RuntimeError as e:
        log.warning("Speech output unavailable, continuing silently: %s", e)
        return SilentTTS()
