"""Time-based animation helpers (pure functions of time -> parameters)."""
from __future__ import annotations

import math


def pulse_scale(t: float, period_s: float = 1.2, amplitude: float = 0.18) -> float:
    """Radius multiplier oscillating around 1.0."""
    return 1.0 + amplitude * math.sin(2 * math.pi * t / period_s)


def blink_visible(t: float, period_s: float = 0.8) -> bool:
    return (t % period_s) < period_s * 0.6


def flow_offset(t: float, speed_px_s: float = 120.0, dash_len: float = 30.0) -> float:
    """Dash phase offset for a 'marching ants' path animation."""
    return (t * speed_px_s) % (dash_len * 2)
