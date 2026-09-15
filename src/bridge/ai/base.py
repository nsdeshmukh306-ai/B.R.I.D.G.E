"""AI provider interface (provider-agnostic)."""
from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Type, TypeVar

import numpy as np
from pydantic import BaseModel, ValidationError

from bridge.ai.schemas import ActionPlan, SceneUnderstanding, TargetIdentification

log = logging.getLogger("bridge.ai")
T = TypeVar("T", bound=BaseModel)


class AIError(RuntimeError):
    pass


@dataclass
class AIStatus:
    provider: str
    model: str = ""
    connected: bool = False
    last_request_ts: Optional[float] = None
    last_latency_s: Optional[float] = None
    last_error: str = ""
    requests: int = 0
    history: list[dict] = field(default_factory=list)
    # Usage of the most recent successful call — tokens are None when a provider doesn't
    # report usage metadata (e.g. the mock/simulation providers).
    last_tokens_in: Optional[int] = None
    last_tokens_out: Optional[int] = None
    last_tokens_total: Optional[int] = None
    last_cost_usd: Optional[float] = None
    # Running total of estimated cost for this process's lifetime, used to enforce an
    # optional soft session budget (see GeminiProvider's budget_inr) and shown in
    # diagnostics so a paid API key's spend is visible without leaving the app.
    session_cost_usd: float = 0.0

    def record(self, kind: str, latency: float, ok: bool, detail: str = "",
               tokens_in: Optional[int] = None, tokens_out: Optional[int] = None,
               tokens_total: Optional[int] = None, cost_usd: Optional[float] = None) -> None:
        self.last_request_ts = time.time()
        self.last_latency_s = latency
        self.requests += 1
        self.connected = ok or self.connected
        if not ok:
            self.last_error = detail
        if ok:
            self.last_tokens_in, self.last_tokens_out = tokens_in, tokens_out
            self.last_tokens_total, self.last_cost_usd = tokens_total, cost_usd
            if cost_usd is not None:
                self.session_cost_usd += cost_usd
        self.history.append({"kind": kind, "latency_s": latency, "ok": ok, "detail": detail[:200],
                             "tokens_total": tokens_total, "cost_usd": cost_usd})
        self.history = self.history[-50:]

    @property
    def text(self) -> str:
        if self.last_error and not self.connected:
            return f"Error: {self.last_error[:60]}"
        return "Connected" if self.connected else "Not yet used"


class StructuredResponseParser:
    """Extracts and validates JSON from a model response. Rejects anything off-schema."""

    _fence = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)

    @classmethod
    def parse(cls, text: str, schema: Type[T]) -> T:
        if not text or not text.strip():
            raise AIError("empty response from model")
        candidate = text.strip()
        m = cls._fence.search(candidate)
        if m:
            candidate = m.group(1).strip()
        if not candidate.startswith("{"):
            start, end = candidate.find("{"), candidate.rfind("}")
            if start == -1 or end == -1:
                raise AIError(f"no JSON object in response: {text[:120]!r}")
            candidate = candidate[start:end + 1]
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError as e:
            raise AIError(f"invalid JSON from model: {e}") from e
        if not isinstance(data, dict):
            raise AIError("model response is not a JSON object")
        try:
            return schema.model_validate(data)
        except ValidationError as e:
            raise AIError(f"model response failed schema validation: {e.errors()[:3]}") from e


class AIProvider(ABC):
    name: str = "abstract"

    def __init__(self) -> None:
        self.status = AIStatus(provider=self.name)

    @abstractmethod
    def understand_scene(self, frame: np.ndarray) -> SceneUnderstanding: ...

    @abstractmethod
    def identify_target(self, frame: np.ndarray, query: str, known_labels: list[str] | None = None) -> TargetIdentification: ...

    @abstractmethod
    def plan_action(self, frame: np.ndarray, task_context: str) -> ActionPlan: ...

    def check_connection(self) -> tuple[bool, str]:
        return True, "ok"

    def close(self) -> None:
        return None
