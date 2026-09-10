"""Motion smoothing and latency compensation for projected target graphics.

The camera delivers 8–30 measurements per second; the projector redraws at 60.
Without a motion model the graphic only moves when a measurement arrives, so it
visibly steps and always trails the object by the capture+processing latency.

`TargetMotion` keeps a velocity estimate per tracked object and answers
"where should the graphic be right now?" at render time: the last measurement
extrapolated forward by the measured pipeline latency plus a small lead, then
smoothed with an exponential filter so the graphic glides instead of snapping.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

TargetStatus = Literal["locked", "searching", "caution", "lost"]

# Intro ("lock-on") animation: the reticle converges onto the object and fades in.
LOCK_ON_S = 0.38
FADE_OUT_S = 0.30


def ease_out_cubic(k: float) -> float:
    k = min(1.0, max(0.0, k))
    return 1.0 - (1.0 - k) ** 3


@dataclass
class TargetMotion:
    """Smoothed, latency-compensated position/size of one tracked object (camera pixels)."""

    x: float
    y: float
    w: float
    h: float
    t0: float = field(default_factory=time.perf_counter)      # when the target appeared
    tau_pos: float = 0.075          # position smoothing time constant (s)
    tau_vel: float = 0.18           # velocity smoothing time constant (s)
    lead_s: float = 0.06            # how far ahead to extrapolate (camera + processing latency)
    max_extrapolate_s: float = 0.25  # never predict further than this past the last measurement
    max_speed: float = 4000.0       # camera px/s, clamps nonsense velocities after a re-detect
    status: TargetStatus = "locked"
    confidence: float = 1.0

    def __post_init__(self) -> None:
        self.mx, self.my = self.x, self.y          # last measurement
        self.mw, self.mh = self.w, self.h
        self.vx = self.vy = 0.0
        self.t_meas = self.t0
        self.t_render = self.t0
        self.released_at: float | None = None

    # -- measurement ---------------------------------------------------------------------
    def observe(self, x: float, y: float, w: float, h: float, t: float | None = None,
                status: TargetStatus = "locked", confidence: float = 1.0) -> None:
        t = time.perf_counter() if t is None else t
        dt = t - self.t_meas
        if dt > 1e-4:
            vx, vy = (x - self.mx) / dt, (y - self.my) / dt
            speed = math.hypot(vx, vy)
            if speed > self.max_speed:  # jump from a re-detection, not real motion
                vx = vy = 0.0
            a = math.exp(-dt / self.tau_vel)
            self.vx = a * self.vx + (1 - a) * vx
            self.vy = a * self.vy + (1 - a) * vy
        self.mx, self.my, self.mw, self.mh = x, y, w, h
        self.t_meas = t
        self.status, self.confidence = status, confidence

    def release(self, t: float | None = None) -> None:
        """Target is going away: start the fade-out."""
        self.released_at = time.perf_counter() if t is None else t
        self.status = "lost"

    # -- render-time value ---------------------------------------------------------------
    def sample(self, t: float | None = None) -> tuple[float, float, float, float]:
        """Position and size to draw at time `t`. Call once per rendered frame."""
        t = time.perf_counter() if t is None else t
        stale = max(0.0, t - self.t_meas)
        # Velocity decays while no new measurement arrives, so a stopped object does not drift.
        decay = math.exp(-stale / 0.22)
        # Total lead = age of the measurement + pipeline latency + the lag this filter itself
        # adds (an exponential filter trails a constant-velocity target by exactly tau).
        ahead = min(stale + self.lead_s + self.tau_pos, self.max_extrapolate_s)
        tx = self.mx + self.vx * decay * ahead
        ty = self.my + self.vy * decay * ahead
        dt = max(0.0, t - self.t_render)
        a = 1.0 - math.exp(-dt / self.tau_pos) if dt > 0 else 0.0
        self.x += (tx - self.x) * a
        self.y += (ty - self.y) * a
        self.w += (self.mw - self.w) * a
        self.h += (self.mh - self.h) * a
        self.t_render = t
        return self.x, self.y, self.w, self.h

    # -- intro / outro -------------------------------------------------------------------
    def lock_on(self, t: float | None = None) -> float:
        """0 at the moment the target appeared, 1 once the reticle has settled."""
        t = time.perf_counter() if t is None else t
        return ease_out_cubic((t - self.t0) / LOCK_ON_S)

    def opacity(self, t: float | None = None) -> float:
        t = time.perf_counter() if t is None else t
        if self.released_at is not None:
            return max(0.0, 1.0 - (t - self.released_at) / FADE_OUT_S)
        return 0.15 + 0.85 * self.lock_on(t)

    def radius_scale(self, t: float | None = None) -> float:
        """Reticle converges from wide open onto the object."""
        return 1.0 + 1.35 * (1.0 - self.lock_on(t))

    @property
    def finished(self) -> bool:
        return self.released_at is not None and (time.perf_counter() - self.released_at) > FADE_OUT_S

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)
