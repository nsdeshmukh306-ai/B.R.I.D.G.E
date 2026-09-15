"""Diagnostics: FPS counters and a text report for the diagnostics panel."""
from __future__ import annotations

import time
from collections import deque

from bridge.app.state import StateManager


class FPSCounter:
    def __init__(self, window: int = 30):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> float:
        self._times.append(time.perf_counter())
        return self.fps

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


def diagnostics_report(state: StateManager) -> str:
    d = state.diagnostics
    err = f"{d.calibration_mean_error:.1f} px" if d.calibration_mean_error is not None else "--"
    conf = f"{d.tracking_confidence:.2f}" if d.tracking_confidence is not None else "--"
    since = d.ai_seconds_since()
    last = f"{since:.1f} sec ago" if since is not None else "--"
    if d.ai_last_tokens_total is not None:
        cost = f"${d.ai_last_cost_usd:.5f}" if d.ai_last_cost_usd is not None else "?"
        usage = f"{d.ai_last_tokens_total} tokens (est. {cost})"
    else:
        usage = "--"
    spend_inr = d.ai_session_cost_usd * d.ai_usd_to_inr
    if d.ai_budget_inr:
        spend = f"~₹{spend_inr:.1f} of ₹{d.ai_budget_inr:.0f} session budget (${d.ai_session_cost_usd:.5f})"
    else:
        spend = f"~₹{spend_inr:.1f} (${d.ai_session_cost_usd:.5f}) — no budget cap set"
    return (
        "CAMERA\n"
        f"Connected: {'YES' if d.camera_connected else 'NO'}\n"
        f"Device: {d.camera_name}\n"
        f"Resolution: {d.camera_resolution}\n"
        f"FPS: {d.camera_fps:.1f}\n\n"
        "DISPLAY\n"
        f"Display: {d.display_name}\n"
        f"Resolution: {d.display_resolution}\n\n"
        "SPATIAL\n"
        f"Calibration: {d.calibration}\n"
        f"Mean error: {err}\n\n"
        "VISION\n"
        f"Tracking: {d.tracking_target}\n"
        f"Confidence: {conf}\n\n"
        "AI\n"
        f"Provider: {d.ai_provider}\n"
        f"Model: {d.ai_model}\n"
        f"Status: {d.ai_status}\n"
        f"Last request: {last}\n"
        f"Last usage: {usage}\n"
        f"Session spend: {spend}"
    )
