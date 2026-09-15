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
    model: str = "gemini-3.6-flash"
    min_confidence: float = 0.5
    timeout_s: float = 30.0
    # Soft spend cap for this run, in INR, matching whatever credit is loaded on the
    # billing account behind GEMINI_API_KEY. Once BRIDGE's own running cost estimate
    # (see ai/gemini.py's per-request cost table) reaches this, it stops calling Gemini
    # and works local-only until the app is restarted or this is raised — a safety net,
    # not an exact bill (Google's actual charge can differ slightly from the estimate).
    # None or 0 disables the cap.
    budget_inr: Optional[float] = 500.0
    usd_to_inr: float = 88.0  # approximate manual FX rate for the cap above; update as needed


class RecognizerSettings(BaseModel):
    """A specialist, local, offline instrument-recognition model.

    Tried before Gemini for scene labelling when configured, because a model
    trained specifically on surgical instruments is more accurate here than a
    general vision-language model, and because it needs no network at all.
    Nothing is bundled with BRIDGE: weights_path and labels_path point at
    files a deployer supplies. See docs/recognition.md before pointing this
    at a downloaded model — pretrained surgical-instrument weights found
    during development carry a non-commercial licence that is very unlikely
    to be compatible with running BRIDGE as a paid product.
    """

    enabled: bool = False
    weights_path: Optional[Path] = None   # .onnx file, YOLO-family export (v5 or v8+ shape)
    labels_path: Optional[Path] = None    # one class name per line, in the model's training order
    conf_threshold: float = 0.45
    iou_threshold: float = 0.45
    input_size: int = 640


class VoiceSettings(BaseModel):
    enabled: bool = True                 # start listening automatically when the app is READY
    # Local/offline by default: no network round-trip per utterance (steadier, lower-latency
    # listening than a cloud call), no Gemini spend for STT, and it keeps working if Gemini
    # itself is rate-limited/down. Needs `pip install faster-whisper` (downloads its small
    # model from Hugging Face on first use, then runs fully offline). Set to "gemini" to go
    # back to cloud transcription (multilingual, no local install, but adds latency + cost).
    stt_provider: Literal["gemini", "whisper", "mock"] = "whisper"
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
    # How often BRIDGE *checks* whether anything on the tray still needs naming.
    # It does not call Gemini (or the local recognizer) every tick regardless — only
    # when something present is unlabelled or low-confidence (SceneGraph.needs_ai_label).
    # Once a tray is fully labelled and stable, this stops costing anything at all.
    ai_label_interval_s: float = 12.0
    # Staleness safety net: force a re-check this often even when nothing looks new,
    # in case an instrument was swapped for a similar one or an earlier label was
    # wrong. Always treated as >= ai_label_interval_s.
    ai_relabel_interval_s: float = 90.0
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
    recognizer: RecognizerSettings = Field(default_factory=RecognizerSettings)
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
        if budget_inr := os.environ.get("BRIDGE_AI_BUDGET_INR"):
            settings.ai.budget_inr = float(budget_inr)
        if level := os.environ.get("BRIDGE_LOG_LEVEL"):
            settings.log_level = level
        if mode := os.environ.get("BRIDGE_MODE"):
            if mode in ("physical", "simulation"):
                settings.mode = mode  # type: ignore[assignment]
        return settings

    def save(self) -> None:
        data = self.settings.model_dump(mode="json")
        self.path.write_text(yaml.safe_dump(data, sort_keys=False))
