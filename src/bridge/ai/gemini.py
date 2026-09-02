"""Gemini adapter via the official google-genai SDK. API key from environment only."""
from __future__ import annotations

import logging
import time
from typing import Optional, Type, TypeVar

import cv2
import numpy as np
from pydantic import BaseModel

from bridge.ai.base import AIError, AIProvider, StructuredResponseParser
from bridge.ai.prompts import PromptBuilder
from bridge.ai.schemas import ActionPlan, SceneUnderstanding, TargetIdentification

log = logging.getLogger("bridge.ai.gemini")
T = TypeVar("T", bound=BaseModel)

MAX_IMAGE_EDGE = 1024


def encode_frame_jpeg(frame: np.ndarray, max_edge: int = MAX_IMAGE_EDGE, quality: int = 85) -> tuple[bytes, float]:
    """Downscale for bandwidth; returns (jpeg_bytes, scale_applied)."""
    h, w = frame.shape[:2]
    scale = min(1.0, max_edge / max(h, w))
    if scale < 1.0:
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise AIError("could not encode frame")
    return buf.tobytes(), scale


class GeminiProvider(AIProvider):
    name = "gemini"

    def __init__(self, api_key: Optional[str], model: str = "gemini-2.5-flash", timeout_s: float = 30.0):
        super().__init__()
        if not api_key:
            raise AIError("GEMINI_API_KEY is not set. Put it in .env or the environment.")
        try:
            from google import genai
            from google.genai import types
        except ImportError as e:  # pragma: no cover
            raise AIError("google-genai SDK not installed (pip install google-genai)") from e
        self._types = types
        self.model = model
        self.timeout_s = timeout_s
        self._client = genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=int(timeout_s * 1000)))
        self.status.provider = f"Gemini ({model})"

    def _call(self, kind: str, frame: np.ndarray, prompt: str, schema: Type[T]) -> T:
        types = self._types
        jpeg, _ = encode_frame_jpeg(frame)
        t0 = time.perf_counter()
        try:
            resp = self._client.models.generate_content(
                model=self.model,
                contents=[types.Part.from_bytes(data=jpeg, mime_type="image/jpeg"), prompt],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=schema,
                    temperature=0.1,
                ),
            )
            text = resp.text or ""
            parsed = StructuredResponseParser.parse(text, schema)
        except AIError as e:
            self.status.record(kind, time.perf_counter() - t0, False, str(e))
            raise
        except Exception as e:  # noqa: BLE001 - network/SDK errors
            self.status.record(kind, time.perf_counter() - t0, False, str(e))
            log.error("Gemini request failed (%s): %s", kind, e)
            raise AIError(f"Gemini request failed: {e}") from e
        latency = time.perf_counter() - t0
        self.status.record(kind, latency, True)
        log.info("Gemini %s ok latency=%.2fs", kind, latency)
        return parsed

    def understand_scene(self, frame: np.ndarray) -> SceneUnderstanding:
        h, w = frame.shape[:2]
        return self._call("understand_scene", frame, PromptBuilder.understand_scene(w, h), SceneUnderstanding)

    def identify_target(self, frame: np.ndarray, query: str, known_labels: list[str] | None = None) -> TargetIdentification:
        h, w = frame.shape[:2]
        return self._call("identify_target", frame, PromptBuilder.identify_target(query, w, h, known_labels), TargetIdentification)

    def plan_action(self, frame: np.ndarray, task_context: str) -> ActionPlan:
        h, w = frame.shape[:2]
        return self._call("plan_action", frame, PromptBuilder.plan_action(task_context, w, h), ActionPlan)

    def check_connection(self) -> tuple[bool, str]:
        try:
            self._client.models.get(model=self.model)
            self.status.connected = True
            return True, "ok"
        except Exception as e:  # noqa: BLE001
            self.status.last_error = str(e)
            return False, str(e)
