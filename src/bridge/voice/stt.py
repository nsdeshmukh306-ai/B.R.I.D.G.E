"""Speech-to-text providers (pluggable).

- GeminiSTT: sends the utterance as audio to Gemini and gets structured JSON back
  (verbatim text + English rendering + language). No extra install; multilingual
  (English, Hindi, Marathi, ...). Needs GEMINI_API_KEY.
- WhisperSTT: local faster-whisper if installed (offline).
- MockSTT: canned transcripts for tests.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Optional

from pydantic import BaseModel, Field

from bridge.ai.base import AIError, StructuredResponseParser

log = logging.getLogger("bridge.voice.stt")


class Transcript(BaseModel):
    text: str = ""            # what was said, verbatim (original language)
    text_en: str = ""         # English rendering used for command routing
    language: str = "en"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_speech: bool = True    # False when the audio contained no intelligible speech

    @property
    def command_text(self) -> str:
        return (self.text_en or self.text).strip()


class SpeechToText(ABC):
    name = "stt"

    def __init__(self) -> None:
        self.last_latency_s: Optional[float] = None
        self.last_error: str = ""

    @abstractmethod
    def transcribe(self, wav_bytes: bytes) -> Transcript: ...


STT_PROMPT = (
    "Transcribe the speech in this audio. It is a spoken command to a healthcare spatial assistant "
    "(examples: 'where is the syringe', 'highlight the lavender tube', 'start instrument count', 'next step'). "
    "Return JSON: text (verbatim, original language), text_en (faithful English rendering; identical to text if "
    "already English), language (BCP-47 code such as en, hi, mr), confidence (0-1), is_speech (false if there is no "
    "intelligible speech, e.g. only noise). Do not add words that were not spoken."
)


class GeminiSTT(SpeechToText):
    name = "gemini"

    def __init__(self, api_key: Optional[str], model: str = "gemini-2.5-flash", timeout_s: float = 20.0):
        super().__init__()
        if not api_key:
            raise AIError("GEMINI_API_KEY is not set (needed for Gemini speech recognition).")
        from google import genai
        from google.genai import types

        self._types = types
        self.model = model
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))

    def transcribe(self, wav_bytes: bytes) -> Transcript:
        types = self._types
        t0 = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"), STT_PROMPT],
                config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=Transcript,
                                                   temperature=0.0),
            )
            tr = StructuredResponseParser.parse(resp.text or "", Transcript)
        except AIError:
            self.last_error = "bad transcription response"
            raise
        except Exception as e:  # noqa: BLE001
            self.last_error = str(e)
            raise AIError(f"speech recognition failed: {e}") from e
        self.last_latency_s = time.perf_counter() - t0
        log.info("STT ok latency=%.2fs lang=%s text=%r", self.last_latency_s, tr.language, tr.text)
        return tr


class WhisperSTT(SpeechToText):
    """Offline recognition with faster-whisper (pip install faster-whisper). English output via translate."""

    name = "whisper"

    def __init__(self, model_size: str = "base", device: str = "auto"):
        super().__init__()
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            raise AIError("faster-whisper is not installed (pip install faster-whisper)") from e
        self._model = WhisperModel(model_size, device=device, compute_type="int8")

    def transcribe(self, wav_bytes: bytes) -> Transcript:
        import io

        t0 = time.perf_counter()
        segments, info = self._model.transcribe(io.BytesIO(wav_bytes), beam_size=1)
        text = " ".join(s.text.strip() for s in segments).strip()
        text_en = text
        if info.language != "en" and text:
            seg_en, _ = self._model.transcribe(io.BytesIO(wav_bytes), beam_size=1, task="translate")
            text_en = " ".join(s.text.strip() for s in seg_en).strip()
        self.last_latency_s = time.perf_counter() - t0
        return Transcript(text=text, text_en=text_en, language=info.language or "en",
                          confidence=float(getattr(info, "language_probability", 0.5) or 0.5), is_speech=bool(text))


class MockSTT(SpeechToText):
    name = "mock"

    def __init__(self, transcripts: list[str] | None = None):
        super().__init__()
        self.transcripts = list(transcripts or [])
        self.calls = 0

    def transcribe(self, wav_bytes: bytes) -> Transcript:
        self.calls += 1
        if not self.transcripts:
            return Transcript(text="", is_speech=False)
        text = self.transcripts.pop(0)
        return Transcript(text=text, text_en=text, language="en", confidence=0.95, is_speech=bool(text))


def build_stt(kind: str, api_key: Optional[str], model: str) -> SpeechToText:
    if kind == "gemini":
        return GeminiSTT(api_key, model)
    if kind == "whisper":
        return WhisperSTT()
    if kind == "mock":
        return MockSTT()
    raise ValueError(f"unknown STT provider '{kind}'")
