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


def _sd():
    try:
        import sounddevice as sd
    except Exception as e:  # noqa: BLE001 - missing PortAudio, no audio subsystem, ...
        raise RuntimeError(f"Audio input unavailable ({e}). Install with: pip install sounddevice") from e
    return sd


def list_input_devices() -> list[tuple[int, str]]:
    """(index, name) of every device that can record. Empty if audio is unavailable."""
    try:
        sd = _sd()
        return [(i, d["name"]) for i, d in enumerate(sd.query_devices()) if d.get("max_input_channels", 0) > 0]
    except Exception as e:  # noqa: BLE001
        log.info("No audio input devices available: %s", e)
        return []


def default_input_device() -> Optional[int]:
    """The computer's own default recording device (what Windows calls the default input)."""
    try:
        sd = _sd()
        default = sd.default.device
        idx = default[0] if isinstance(default, (list, tuple)) else default
        if idx is not None and idx >= 0:
            info = sd.query_devices(idx)
            if info.get("max_input_channels", 0) > 0:
                return int(idx)
    except Exception as e:  # noqa: BLE001
        log.debug("No default input device: %s", e)
    devices = list_input_devices()
    return devices[0][0] if devices else None


def _resample_to(chunk: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Linear resample. Speech recognition is unaffected by the tiny interpolation error."""
    if src_rate == dst_rate or chunk.size == 0:
        return chunk
    n_out = max(1, int(round(len(chunk) * dst_rate / src_rate)))
    x_in = np.linspace(0.0, 1.0, num=len(chunk), endpoint=False)
    x_out = np.linspace(0.0, 1.0, num=n_out, endpoint=False)
    return np.interp(x_out, x_in, chunk.astype(np.float32)).astype(np.int16)


class MicrophoneSource:
    """Live microphone via sounddevice.

    With no device given it opens the computer's default recording device, so a laptop's
    built-in microphone works with nothing plugged in and nothing configured. Windows audio
    backends frequently refuse 16 kHz mono, so the stream is opened at whatever rate the
    device accepts and resampled to 16 kHz for the rest of the voice pipeline.
    """

    CANDIDATE_RATES = (16000, 48000, 44100, 32000, 22050, 8000)

    def __init__(self, device: int | str | None = None, sample_rate: int = SAMPLE_RATE, chunk_ms: int = CHUNK_MS):
        sd = _sd()
        self.sample_rate = sample_rate           # what consumers see (always 16 kHz)
        self._q: queue.Queue[np.ndarray] = queue.Queue(maxsize=200)
        self._closed = threading.Event()

        if device is None:
            device = default_input_device()
            if device is None:
                raise RuntimeError("No microphone found. Connect one, or check Windows sound settings "
                                   "(Settings > System > Sound > Input).")
        try:
            info = sd.query_devices(device)
            self.device_name = info.get("name", str(device))
            device_rate = int(info.get("default_samplerate") or 0) or None
            max_channels = max(1, int(info.get("max_input_channels", 1)))
        except Exception:  # noqa: BLE001
            self.device_name, device_rate, max_channels = str(device), None, 1

        rates = [sample_rate] + ([device_rate] if device_rate else []) + list(self.CANDIDATE_RATES)
        seen, attempts = set(), []
        errors: list[str] = []
        for rate in [r for r in rates if r and not (r in seen or seen.add(r))]:
            for channels in (1, max_channels) if max_channels > 1 else (1,):
                attempts.append((rate, channels))
                try:
                    self._open(sd, device, rate, channels, chunk_ms)
                except Exception as e:  # noqa: BLE001
                    errors.append(f"{rate} Hz/{channels}ch: {e}")
                    continue
                self.device_rate, self.channels = rate, channels
                log.info("Microphone opened: %s (device %s) at %d Hz, %d channel(s)%s",
                         self.device_name, device, rate, channels,
                         "" if rate == sample_rate else f" -> resampled to {sample_rate} Hz")
                return
        available = ", ".join(f"[{i}] {n}" for i, n in list_input_devices()) or "none"
        raise RuntimeError(f"Could not open microphone '{self.device_name}'. Tried {len(attempts)} format(s); "
                           f"last error: {errors[-1] if errors else 'unknown'}. Available inputs: {available}")

    def _open(self, sd, device, rate: int, channels: int, chunk_ms: int) -> None:
        frames = max(1, int(rate * chunk_ms / 1000))
        target = self.sample_rate

        def cb(indata, _frames, _time, status):  # noqa: ANN001
            if status:
                log.debug("audio status: %s", status)
            mono = indata[:, 0] if indata.ndim > 1 else indata
            try:
                self._q.put_nowait(_resample_to(np.ascontiguousarray(mono), rate, target))
            except queue.Full:
                pass  # consumer is behind; dropping old audio keeps latency low

        stream = sd.InputStream(samplerate=rate, channels=channels, dtype="int16", blocksize=frames,
                                device=device, callback=cb)
        stream.start()
        self._stream = stream

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
