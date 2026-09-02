# BRIDGE — Universal Spatial AI Projection Platform

**B**rain **R**easoning **I**ntelligence **D**irected into **G**rounded **E**nvironments

> Don't bring the human into the computer. Bring the computer's intelligence into the human's physical environment.

BRIDGE connects a commodity USB camera, a commodity projector (any OS display) and a vision-language model. It calibrates the camera↔projector relationship automatically, understands what is on a table or wall through AI, and projects spatially registered graphics — circles, arrows, labels, target zones, paths — directly onto the real objects. Ask *"Where is the screwdriver?"* and a circle appears around the physical screwdriver; move it and the circle follows.

## What BRIDGE does

```
User: "Where is the screwdriver?"
        │
        ▼
   Gemini (WHAT)  ──►  intent=find_object, target=screwdriver, box=[…]
        │
        ▼
   Target resolver ──► local contour/colour matching  (WHERE, now)
        │
        ▼
   Local tracker  ──► follows the object at camera frame-rate (no AI per frame)
        │
        ▼
   Homography     ──► camera pixels → projector pixels
        │
        ▼
   Renderer       ──► pulsing circle on the white canvas → projector → table
```

* **Hardware agnostic** — any camera OpenCV can open, any display the OS exposes. No vendor code.
* **AI-provider agnostic** — `AIProvider` interface; Gemini implemented; a clearly separated simulation/mock provider for testing.
* **Local-first** — Gemini answers *what*; OpenCV answers *where* at 20–30 FPS.
* **Measured calibration** — reprojection error is computed on independent validation points and stored; nothing is assumed.
* **Never project uncertainty as certainty** — low confidence → "Target uncertain. Please clarify."; tracking lost → stale graphics are cleared.
* **Simulation mode** — a virtual table, virtual objects (draggable), virtual camera and virtual projector run the *same* code paths, so everything can be developed and tested without hardware.

## Architecture

```
src/bridge/
├── app/          BridgeCore (headless application), ConfigManager, EventBus, StateManager, Diagnostics
├── camera/       CameraDevice / FrameSource abstraction, CameraManager (enumeration + capture thread)
├── projector/    DisplayManager (OS displays), ProjectionWindow (borderless fullscreen canvas)
├── spatial/      Point/BoundingBox/Polygon, Homography + CoordinateMapper, Workspace,
│                 CalibrationMethod (ABC) → PlanarMarkerCalibration, marker detection, CalibrationValidator,
│                 CalibrationProfile + ProfileStore (JSON persistence)
├── vision/       ObjectDetector (ContourDetector, ColorDetector), ObjectTracker / MultiTracker,
│                 self-projection suppression, optional MediaPipe HandTracker
├── ai/           AIProvider (ABC), GeminiProvider (google-genai SDK), Pydantic response schemas,
│                 PromptBuilder, StructuredResponseParser, ScriptedMock / SimulationOracle providers
├── interaction/  Controlled command vocabulary (Pydantic), CommandPlanner, TargetResolver,
│                 CommandExecutor (command → resolve → track → map → render), Task state machine
├── render/       ProjectionRenderer (OpenCV rasterizer), primitives, animation, Scene
├── simulation/   VirtualWorld, SimulatedProjector, SimulatedCamera (with ground-truth homographies)
├── ui/           PySide6 MainWindow, CalibrationWizard, SettingsDialog, widgets
└── main.py       entry point
```

The three concerns the spec insists on stay separate: **AI reasoning** (`ai/`), **perception** (`vision/`), **spatial mapping** (`spatial/`) and **rendering** (`render/`) only meet inside `interaction/executor.py`. See [docs/architecture.md](docs/architecture.md).

## Installation

Requirements: Python 3.11+, a desktop OS (Windows, macOS, Linux).

```bash
git clone <this repo> bridge && cd bridge
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Dependencies: `numpy`, `opencv-contrib-python` (CSRT/KCF trackers), `PySide6`, `pydantic`, `pyyaml`, `google-genai`, `screeninfo`; optional `mediapipe` for hand tracking (`pip install -e ".[hands]"`).

> Install **only** `opencv-contrib-python` (not `opencv-python` alongside it). Without the contrib build BRIDGE still works but falls back to the colour/contour tracker.

## Gemini API setup

```bash
cp .env.example .env
# edit .env:
GEMINI_API_KEY=your_key_here
```

The key is read from the environment / `.env` only. It is never written to `settings.yaml` and `.env` is git-ignored. The model defaults to `gemini-2.5-flash` (change in Settings or with `BRIDGE_AI_MODEL`). See [docs/gemini.md](docs/gemini.md).

## Running

```bash
bridge                     # physical mode
bridge --simulation        # hardware-free simulation
bridge --headless-selftest # calibrate + query in simulation, no window (CI smoke test)
python -m bridge.main      # same as `bridge`
```

## First run (physical hardware)

1. **Camera** — pick your webcam in the CAMERA box. Status turns green and the live view appears.
2. **Display / Projector** — the projector is just a second display; BRIDGE pre-selects the secondary one. Press **OPEN PROJECTION**: a borderless white canvas opens fullscreen on that display (Esc closes it). Press **Test graphics** to check you can see a circle, arrow and label on the surface.
3. **Surface** — Table / Wall / Custom (all planar in V1; the preset is stored in the profile).
4. **AUTO CALIBRATE** — the wizard projects bright markers one at a time on a dark background, detects them with the camera, fits a homography, then projects an independent 3×3 validation grid and measures the reprojection error. You get `Accuracy: 2.8 px` and **SAVE** writes `profiles/default.json`. Next time the same camera + display is found the profile is loaded automatically.
5. Type **"Where is the screwdriver?"** and press **EXECUTE**.

The banner switches to **BRIDGE READY** once a valid calibration is active.

## Calibration

Planar 4-point (default) or 9-point marker calibration; details, tips and the validation model are in [docs/calibration.md](docs/calibration.md). Key points:

* Camera and projector must both see the same flat surface and must not move afterwards.
* Moving either device, changing the camera, display or resolution **invalidates** the calibration (BRIDGE detects device/resolution changes and tells you).
* "Click-to-project test" lets you click any point in the camera view and see a marker projected at that physical spot — the fastest way to check registration.

## Simulation mode

Select **Simulation** in the MODE box (or start with `--simulation`). You get a virtual table with a screwdriver, phone, scissors, two screws and a pen, seen by a virtual perspective camera and lit by a virtual keystoned projector. Calibrate, ask questions, and **drag objects in the camera view** to move them — the projected circle follows. The AI provider in simulation is a clearly labelled *Simulation oracle (mock)* that answers from the virtual world's ground truth so no API key or network is needed; the rest of the pipeline (marker detection, homography, tracking, rendering, command validation) is the real production code.

## Diagnostics and logging

The right-hand panel shows camera FPS/resolution, display, calibration status and mean error, current tracked target and confidence, AI provider status and last-request latency. Logs go to the console, to `logs/bridge.log` (rotating) and to the in-app LOG panel. Set `log_level: DEBUG` in `settings.yaml` or `--log-level DEBUG`.

## Configuration

`settings.yaml` (created by the Settings dialog; all keys optional):

```yaml
mode: physical
camera: {preferred_device: null, width: 1280, height: 720, fps: 30}
projector: {preferred_display: null}
calibration: {method: planar_4point, validation_threshold_px: 10, marker_radius_px: 28, margin_fraction: 0.12, max_retries: 3}
tracking: {enabled: true, backend: csrt, lost_after_frames: 15}
ai: {provider: gemini, model: gemini-2.5-flash, min_confidence: 0.5, timeout_s: 30}
render: {background: white, target_style: pulse, accent_rgb: [0, 150, 255], line_width: 4, fps: 60}
```

## Development

```bash
pip install -e ".[dev]"
pytest                    # 74 tests: geometry, homography, calibration vs simulator ground truth,
                          # AI schema validation, command validation, rendering, tracking,
                          # full simulated demo, device failure handling, GUI smoke tests
bridge --headless-selftest
```

Tests run headless (`QT_QPA_PLATFORM=offscreen` is set in `tests/conftest.py`). Add a calibration method by subclassing `spatial.calibration.CalibrationMethod` and registering it in `build_method`; add an AI provider by subclassing `ai.base.AIProvider` and registering it in `ai.mock.build_provider`; add a tracker backend in `vision.tracking._make_cv_tracker`.

## Troubleshooting

See [docs/troubleshooting.md](docs/troubleshooting.md). The most common issues: the camera cannot see the projected markers (too bright a room, auto-exposure), the projection window opened on the wrong display, or `opencv-python` shadowing `opencv-contrib-python`.

## Known limitations

* Calibration is a **planar homography**: valid only for flat surfaces (table, wall, floor). Curved or multi-plane surfaces are out of scope for V1.
* The setup must be **stable**: moving the camera or projector invalidates calibration.
* Very dark, glossy or reflective surfaces and strong ambient light reduce marker detection quality; projector brightness and camera exposure matter.
* AI localisation is not pixel-perfect; BRIDGE tightens Gemini's boxes with local contours and then relies on local tracking. Accuracy figures are the *measured* reprojection error on validation points — no millimetre claims are made.
* Local tracking is appearance based (CSRT/KCF or colour/contour). Fast motion, occlusion by hands, or identical-looking objects can cause a lost or swapped target; BRIDGE then clears the graphic and says *Target lost.*
* "Any camera / any projector" means any device supported by the OS and OpenCV/Qt, not literally every device.
* One camera and one projector in V1.

## Roadmap

ChArUco / ArUco calibration → Gray-code structured light and dense correspondence → automatic surface detection and non-planar surfaces → better markerless tracking → hand tracking for action verification → guided assembly workflows on the existing task state machine → voice input → multiple cameras/projectors → other AI providers (OpenAI, local VLMs) → installers (Windows first).

## License

MIT.
