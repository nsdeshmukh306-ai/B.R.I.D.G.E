"""Gemini adapter via the official google-genai SDK. API key from environment only."""
from __future__ import annotations

import logging
import re
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

# Fallback backoff windows, used when the failure doesn't tell us how long to wait.
# A quota failure usually does (Google's RetryInfo.retryDelay, parsed below); a retired
# or misconfigured model never self-heals, so it gets a long cooldown rather than being
# retried every ai_label_interval_s until someone fixes settings.yaml.
_QUOTA_COOLDOWN_DEFAULT_S = 30.0
_MODEL_UNAVAILABLE_COOLDOWN_S = 300.0
_OVERLOADED_COOLDOWN_S = 20.0
_GENERIC_ERROR_COOLDOWN_S = 10.0
# A per-day quota (free-tier RPD cap) doesn't self-heal within Google's own retryDelay —
# that field is a generic short suggestion, not tied to the actual daily reset — so
# retrying on it just repeats the same failure every ~10s and spams the UI. Back off much
# longer instead and say plainly that this is a daily cap, not a transient throttle.
_DAILY_QUOTA_COOLDOWN_S = 3600.0
_RETRY_DELAY_RE = re.compile(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s")
_PER_DAY_QUOTA_RE = re.compile(r"PerDay", re.IGNORECASE)

# USD per 1M tokens (standard, non-batch, text/image), as of the Gemini API pricing page
# in Sept 2026 (https://ai.google.dev/gemini-api/docs/pricing) — check that page before
# trusting this for anything billing-critical; Google revises these, and some already have
# a scheduled increase (noted per entry). Used only to print an estimate in the logs, never
# to enforce a budget. A model missing from this table logs token counts with no cost figure
# rather than guessing.
_MODEL_PRICING_USD_PER_1M = {
    # model prefix: (input $/1M, output $/1M)
    "gemini-3.6-flash": (0.75, 3.75),   # rises to 1.50 / 7.50 on 2027-01-01
    "gemini-3.5-flash": (1.50, 9.00),
    "gemini-2.5-flash": (0.30, 2.50),   # retired for new users; kept for stragglers on it
}


def _estimate_cost_usd(model: str, tokens_in: Optional[int], tokens_out: Optional[int]) -> Optional[float]:
    if tokens_in is None or tokens_out is None:
        return None
    for prefix, (in_rate, out_rate) in _MODEL_PRICING_USD_PER_1M.items():
        if model.startswith(prefix):
            return (tokens_in * in_rate + tokens_out * out_rate) / 1_000_000
    return None


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

    def __init__(self, api_key: Optional[str], model: str = "gemini-3.6-flash", timeout_s: float = 30.0,
                 budget_inr: Optional[float] = None, usd_to_inr: float = 88.0):
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
        self.status.model = model
        # Set on a failure that won't self-heal by retrying immediately (quota exhaustion,
        # a retired model). Calls made before this deadline are refused locally, without
        # touching the network, so a 12s background labelling loop doesn't hammer an API
        # that just told us to back off.
        self._cooldown_until: float = 0.0
        self._cooldown_message: str = ""
        # Optional soft session spend cap (see AISettings.budget_inr) — once our own cost
        # estimate for this run reaches it, stop calling Gemini entirely rather than risk
        # running past whatever credit is actually loaded on the key. Unlike the cooldowns
        # above this never auto-clears; it's meant to require a conscious restart/raise.
        self.usd_to_inr = usd_to_inr
        self._budget_inr = budget_inr if budget_inr and budget_inr > 0 else None
        self._budget_usd = (self._budget_inr / usd_to_inr) if self._budget_inr else None
        self._budget_exhausted = False

    def _classify_failure(self, e: Exception) -> tuple[str, float]:
        """Turn a raw SDK/network exception into (user-facing message, cooldown seconds)."""
        text = str(e)
        if "RESOURCE_EXHAUSTED" in text or " 429" in text or text.startswith("429"):
            if _PER_DAY_QUOTA_RE.search(text):
                return (f"Gemini daily quota exhausted for '{self.model}'; working locally and will "
                         "keep checking hourly — if this persists, check quota/billing for the API "
                         "key in Google AI Studio / Cloud Console", _DAILY_QUOTA_COOLDOWN_S)
            m = _RETRY_DELAY_RE.search(text)
            retry_s = float(m.group(1)) if m else _QUOTA_COOLDOWN_DEFAULT_S
            return (f"AI quota exceeded; retrying automatically in {retry_s:.0f}s", retry_s)
        if "NOT_FOUND" in text or " 404" in text or text.startswith("404"):
            return (f"Gemini model '{self.model}' is unavailable (retired or misconfigured); "
                     "an administrator needs to update ai.model in settings.yaml", _MODEL_UNAVAILABLE_COOLDOWN_S)
        if "UNAVAILABLE" in text or " 503" in text or text.startswith("503"):
            # Transient Google-side capacity issue, not a configuration problem — the
            # model and API key are fine, so don't tell the user to check them.
            return (f"Gemini is temporarily overloaded (high demand on {self.model}); "
                     f"retrying automatically in {_OVERLOADED_COOLDOWN_S:.0f}s", _OVERLOADED_COOLDOWN_S)
        return (f"Gemini request failed: {e}", _GENERIC_ERROR_COOLDOWN_S)

    def _budget_message(self) -> str:
        spent_inr = self.status.session_cost_usd * self.usd_to_inr
        return (f"Session AI budget reached (~₹{spent_inr:.0f} of ₹{self._budget_inr:.0f} spent this run); "
                "working locally only. Raise ai.budget_inr in settings.yaml, or restart the app, to continue.")

    def _call(self, kind: str, frame: np.ndarray, prompt: str, schema: Type[T]) -> T:
        if self._budget_usd is not None and (
            self._budget_exhausted or self.status.session_cost_usd >= self._budget_usd
        ):
            self._budget_exhausted = True
            message = self._budget_message()
            self.status.record(kind, 0.0, False, message)
            raise AIError(message)
        now = time.perf_counter()
        if now < self._cooldown_until:
            self.status.record(kind, 0.0, False, self._cooldown_message)
            raise AIError(self._cooldown_message)
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
            message, cooldown_s = self._classify_failure(e)
            self._cooldown_until = time.perf_counter() + cooldown_s
            self._cooldown_message = message
            self.status.record(kind, time.perf_counter() - t0, False, message)
            log.warning("Gemini request failed (%s): %s (cooling down %.0fs)", kind, e, cooldown_s)
            raise AIError(message) from e
        latency = time.perf_counter() - t0
        tokens_in, tokens_out, tokens_total, cost = self._usage(resp)
        self.status.record(kind, latency, True, tokens_in=tokens_in, tokens_out=tokens_out,
                           tokens_total=tokens_total, cost_usd=cost)
        if tokens_total is not None:
            cost_txt = f" cost=${cost:.5f}" if cost is not None else " cost=? (model not in local pricing table)"
            log.info("Gemini %s ok latency=%.2fs tokens_in=%s tokens_out=%s tokens_total=%s%s session_total=$%.5f",
                     kind, latency, tokens_in, tokens_out, tokens_total, cost_txt, self.status.session_cost_usd)
        else:
            log.info("Gemini %s ok latency=%.2fs (no usage metadata in response)", kind, latency)
        if self._budget_usd is not None and self.status.session_cost_usd >= self._budget_usd:
            self._budget_exhausted = True
            log.warning("Gemini session budget reached: ~$%.5f (~₹%.0f) of ₹%.0f — further calls blocked "
                        "until restart or ai.budget_inr is raised",
                        self.status.session_cost_usd, self.status.session_cost_usd * self.usd_to_inr,
                        self._budget_inr)
        return parsed

    def _usage(self, resp) -> tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
        """Best-effort token usage from the response. Never raises: a successful AI call
        must not fail just because usage accounting couldn't be read."""
        meta = getattr(resp, "usage_metadata", None)
        if meta is None:
            return None, None, None, None
        tokens_in = getattr(meta, "prompt_token_count", None)
        tokens_out = getattr(meta, "candidates_token_count", None)
        tokens_total = getattr(meta, "total_token_count", None)
        if tokens_total is None and tokens_in is not None and tokens_out is not None:
            tokens_total = tokens_in + tokens_out
        cost = _estimate_cost_usd(self.model, tokens_in, tokens_out)
        return tokens_in, tokens_out, tokens_total, cost

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
