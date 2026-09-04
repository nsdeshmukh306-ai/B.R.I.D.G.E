"""Microphone capture and voice activity detection.

`AudioSource` yields 16 kHz mono int16 chunks. `MicrophoneSource` wraps
sounddevice (PortAudio); `ArraySource` replays a numpy buffer for tests.
`UtteranceCapture` uses an energy VAD with hangover to cut one utterance.
"""
from __future__ import annotations

import io
import logging
import queue
import threading
import time
import wave
from dataclasses import dataclass
from typing import Iterator, Optional, Protocol

import numpy as np

log = logging.getLogger("bridge.voice.audio")

SAMPLE_RATE = 16000
CHUNK_MS = 30


class AudioSource(Protocol):
    sample_rate: int

    def chunks(self) -> Iterator[np.ndarray]: ...
    def close(self) -> None: ...


def list_input_devices() -> list[tuple[int, str]]:
    try:
        import sounddevice as sd

        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append((i, d["name"]))
        return out
    except Exception as e:  # noqa: BLE001
        log.info("No audio input devices available: %s", e)
        return []


class MicrophoneSource:
    """Live microphone via sounddevice. Raises RuntimeError if audio is unavailable."""

    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS):
        try:
            import sounddevice as sd
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Microphone unavailable (sounddevice/PortAudio): {e}") from e
        self.sample_rate = sample_rate
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._closed = threading.Event()
        frames = int(sample_rate * chunk_ms / 1000)

        def cb(indata, _frames, _time, status):  # noqa: ANN001
            if status:
                log.debug("audio status: %s", status)
            try:
                self._q.put_nowait(indata[:, 0].copy())
            except queue.Full:
                pass

        try:
            self._stream = sd.InputStream(samplerate=sample_rate, channels=1, dtype="int16", blocksize=frames,
                                          device=device, callback=cb)
            self._stream.start()
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"Could not open microphone: {e}") from e
        log.info("Microphone opened device=%s rate=%d", device, sample_rate)

    def chunks(self) -> Iterator[np.ndarray]:
        while not self._closed.is_set():
            try:
                yield self._q.get(timeout=0.2)
            except queue.Empty:
                continue

    def close(self) -> None:
        self._closed.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:  # noqa: BLE001
            pass


class ArraySource:
    """Replays a buffer as chunks (tests / recorded files). Optionally paces to real time."""

    def __init__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS, realtime: bool = False):
        self.samples = samples.astype(np.int16)
        self.sample_rate = sample_rate
        self._n = int(sample_rate * chunk_ms / 1000)
        self._realtime = realtime
        self._closed = False
        self._pos = 0
        self.exhausted = False

    def chunks(self) -> Iterator[np.ndarray]:
        """Resumes from where the previous consumer stopped (like a live stream would)."""
        while self._pos < len(self.samples) and not self._closed:
            if self._realtime:
                time.sleep(self._n / self.sample_rate)
            chunk = self.samples[self._pos:self._pos + self._n]
            self._pos += self._n
            yield chunk
        self.exhausted = True

    def close(self) -> None:
        self._closed = True


def rms(chunk: np.ndarray) -> float:
    if chunk.size == 0:
        return 0.0
    x = chunk.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x)))


@dataclass
class Utterance:
    samples: np.ndarray
    sample_rate: int
    started_at: float
    duration_s: float
    peak_rms: float

    def wav_bytes(self) -> bytes:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(self.sample_rate)
            w.writeframes(self.samples.astype(np.int16).tobytes())
        return buf.getvalue()


class UtteranceCapture:
    """Energy VAD: speech starts when RMS exceeds threshold (adaptive to noise floor) and
    ends after `silence_ms` below it. Pre-roll keeps the first syllable."""

    def __init__(self, threshold: float = 0.012, silence_ms: int = 800, min_speech_ms: int = 250,
                 max_utterance_s: float = 12.0, preroll_ms: int = 300, chunk_ms: int = CHUNK_MS):
        self.threshold = threshold
        self.silence_chunks = max(1, silence_ms // chunk_ms)
        self.min_speech_chunks = max(1, min_speech_ms // chunk_ms)
        self.max_chunks = int(max_utterance_s * 1000 // chunk_ms)
        self.preroll = max(1, preroll_ms // chunk_ms)
        self.noise_floor = 0.004
        self.level = 0.0  # last RMS, for UI meters

    def _speech_threshold(self) -> float:
        return max(self.threshold, self.noise_floor * 3.0)

    def next_utterance(self, source: AudioSource, stop: Optional[threading.Event] = None) -> Optional[Utterance]:
        pre: list[np.ndarray] = []
        buf: list[np.ndarray] = []
        in_speech, silent, voiced, peak = False, 0, 0, 0.0
        t_start = 0.0
        for chunk in source.chunks():
            if stop is not None and stop.is_set():
                return None
            e = rms(chunk)
            self.level = e
            if not in_speech:
                # track the noise floor slowly while idle
                self.noise_floor = 0.95 * self.noise_floor + 0.05 * e
                pre.append(chunk)
                pre = pre[-self.preroll:]
                if e > self._speech_threshold():
                    in_speech, t_start = True, time.time()
                    buf = list(pre)
                    voiced, silent, peak = 1, 0, e
                continue
            buf.append(chunk)
            peak = max(peak, e)
            if e > self._speech_threshold():
                voiced += 1
                silent = 0
            else:
                silent += 1
            if silent >= self.silence_chunks or len(buf) >= self.max_chunks:
                if voiced < self.min_speech_chunks:
                    in_speech, buf = False, []  # a click, not speech
                    continue
                samples = np.concatenate(buf)
                return Utterance(samples, source.sample_rate, t_start, len(samples) / source.sample_rate, peak)
        if in_speech and voiced >= self.min_speech_chunks and buf:
            samples = np.concatenate(buf)
            return Utterance(samples, source.sample_rate, t_start, len(samples) / source.sample_rate, peak)
        return None


def synth_speech_like(duration_s: float, sample_rate: int = SAMPLE_RATE, amplitude: float = 0.3, seed: int = 0) -> np.ndarray:
    """Band-limited noise burst with syllable-like envelope (tests only)."""
    rng = np.random.default_rng(seed)
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    env = 0.5 + 0.5 * np.abs(np.sin(2 * np.pi * 4 * t))
    sig = rng.normal(0, 1, n) * env * amplitude
    return np.clip(sig * 32767, -32768, 32767).astype(np.int16)
