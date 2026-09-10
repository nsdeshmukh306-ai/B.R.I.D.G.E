"""Application configuration: YAML settings file + environment secrets.

Secrets (GEMINI_API_KEY) only come from the environment / .env, never from
the YAML settings file and never from source.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, Field

DEFAULT_SETTINGS_PATH = Path("settings.yaml")


class CameraSettings(BaseModel):
    preferred_device: Optional[str] = None
    width: int = 1280
    height: int = 720
    fps: int = 30
    auto_exposure: bool = True      # set False + exposure for a stable frame rate in a dark room
    exposure: Optional[float] = None  # backend-specific units (DSHOW: -6 ≈ 1/60 s, -5 ≈ 1/30 s)


class ProjectorSettings(BaseModel):
    preferred_display: Optional[str] = None


class CalibrationSettings(BaseModel):
    method: str = "planar_9point"  # 3x3 markers + RANSAC; more accurate than 4 corners
    validation_threshold_px: float = 6.0  # camera pixels, measured on independent points
    marker_radius_px: int = 28
    margin_fraction: float = 0.12
    detection_min_confidence: float = 0.6
    max_retries: int = 3


class TrackingSettings(BaseModel):
    enabled: bool = True
    process_width: int = 640        # frames are downscaled to this width for detection/tracking (latency)
    smoothing: bool = True          # render-time motion smoothing + latency compensation
    lead_ms: int = 60               # how far ahead the projected graphic is extrapolated (camera+processing latency)
    backend: Literal["csrt", "kcf", "mosse", "mil", "color"] = "csrt"  # falls back to colour re-detection if unavailable
    lost_after_frames: int = 15
    min_confidence: float = 0.3


class AISettings(BaseModel):
    provider: Literal["gemini", "mock"] = "gemini"
    model: str = "gemini-2.5-flash"
    min_confidence: float = 0.5
    timeout_s: float = 30.0


class VoiceSettings(BaseModel):
    enabled: bool = True                 # start listening automatically when the app is READY
    stt_provider: Literal["gemini", "whisper", "mock"] = "gemini"
    tts_enabled: bool = True
    tts_rate: int = 175
    tts_voice: Optional[str] = None      # substring of an installed voice name, e.g. "Zira"
    wake_word: str = "bridge"
    require_wake_word: bool = False
    mic_device: Optional[int] = None
    vad_threshold: float = 0.012
    silence_ms: int = 800
    max_utterance_s: float = 12.0
    barge_in: bool = True                # speaking over BRIDGE cuts it off mid-sentence


class AssistantSettings(BaseModel):
    """The always-on surgical assistant layer."""

    enabled: bool = True
    proactive: bool = True               # BRIDGE may speak first when something is wrong
    scan_hz: float = 5.0                 # scene-graph updates per second (local CV, cheap)
    ai_label_interval_s: float = 12.0    # how often the model is asked to name what it sees
    monitor_interval_s: float = 1.0
    show_count_board: bool = True        # project the live count board onto the surface
    records_dir: Path = Path("case_records")


class SurgicalSettings(BaseModel):
    default_set: str = "minor"           # instrument set used when none is named
    sharp_grace_s: float = 20.0          # a sharp off the tray this long earns a warning
    field_item_grace_s: float = 45.0
    absence_grace_s: float = 6.0         # how long something must be unseen before it counts as missing
    alert_cooldown_s: float = 45.0


class RenderSettings(BaseModel):
    background: Literal["white", "black", "transparent", "custom"] = "black"  # black = projector emits only the graphics
    custom_background_rgb: tuple[int, int, int] = (0, 0, 0)
    target_style: Literal["pulse", "static"] = "pulse"
    accent_rgb: tuple[int, int, int] = (0, 220, 200)
    line_width: int = 3
    fps: int = 60


class AppSettings(BaseModel):
    mode: Literal["physical", "simulation"] = "physical"
    profiles_dir: Path = Path("profiles")
    log_dir: Path = Path("logs")
    log_level: str = "INFO"
    camera: CameraSettings = Field(default_factory=CameraSettings)
    projector: ProjectorSettings = Field(default_factory=ProjectorSettings)
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    tracking: TrackingSettings = Field(default_factory=TrackingSettings)
    ai: AISettings = Field(default_factory=AISettings)
    render: RenderSettings = Field(default_factory=RenderSettings)
    voice: VoiceSettings = Field(default_factory=VoiceSettings)
    assistant: AssistantSettings = Field(default_factory=AssistantSettings)
    surgical: SurgicalSettings = Field(default_factory=SurgicalSettings)
    domain: Literal["healthcare", "general"] = "healthcare"


class Secrets(BaseModel):
    gemini_api_key: Optional[str] = None

    @classmethod
    def from_env(cls, dotenv: Path | None = Path(".env")) -> "Secrets":
        if dotenv and dotenv.exists():
            for line in dotenv.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        key = os.environ.get("GEMINI_API_KEY") or None
        return cls(gemini_api_key=key)


class ConfigManager:
    """Loads/saves AppSettings from YAML and applies environment overrides."""

    def __init__(self, path: Path = DEFAULT_SETTINGS_PATH):
        self.path = Path(path)
        self.settings = self.load()
        self.secrets = Secrets.from_env()

    def load(self) -> AppSettings:
        if self.path.exists():
            data = yaml.safe_load(self.path.read_text()) or {}
            settings = AppSettings.model_validate(data)
        else:
            settings = AppSettings()
        if model := os.environ.get("BRIDGE_AI_MODEL"):
            settings.ai.model = model
        if level := os.environ.get("BRIDGE_LOG_LEVEL"):
            settings.log_level = level
        if mode := os.environ.get("BRIDGE_MODE"):
            if mode in ("physical", "simulation"):
                settings.mode = mode  # type: ignore[assignment]
        return settings

    def save(self) -> None:
        data = self.settings.model_dump(mode="json")
        self.path.write_text(yaml.safe_dump(data, sort_keys=False))
