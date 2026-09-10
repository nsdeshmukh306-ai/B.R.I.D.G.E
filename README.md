# BRIDGE — Spatial AI for Healthcare

**B**rain **R**easoning **I**ntelligence **D**irected into **G**rounded **E**nvironments

> Don't bring the human into the computer. Bring the computer's intelligence into the human's physical environment.

BRIDGE connects a commodity USB camera, a commodity projector (any OS display) and a vision-language model. It calibrates the camera↔projector relationship automatically, understands what is on the bench through AI, and projects spatially registered graphics — reticles, arrows, labels, target zones, paths — directly onto the real items. Say *"Where is the syringe?"* and a glowing reticle lands on the physical syringe; move it and the reticle follows. Say *"Start the order of draw"* and BRIDGE highlights each blood tube in sequence while speaking the step.

It is **voice-first** (microphone → Gemini or offline Whisper → spoken replies) and built for **healthcare workspaces** — phlebotomy trays, medication preparation, instrument counts, dressing kits, PPE, specimen handling — with hard rules: BRIDGE locates items and guides protocol steps; it never diagnoses, prescribes or decides doses. See [docs/healthcare.md](docs/healthcare.md) and [docs/voice.md](docs/voice.md).

On top of that sits an **always-on surgical assistant**: it keeps a live model of everything on the tray, holds a conversation ("and the other one"), drives a whole case through a WHO-style checklist, runs the instrument and sponge count hands-free, and **speaks first** when the numbers stop adding up. See [docs/surgical.md](docs/surgical.md).

The projection canvas is **black**: the projector emits nothing except the graphics, so the room stays dark-friendly and the camera never fights a white wash.

## What BRIDGE does

```
User: "Project the scalpel."  (spoken or typed)
        │
        ▼
   Safety gate    ──►  dosing/diagnosis refused in code, before anything else
        │
        ▼
   Reference res. ──►  "the other one" -> "the other artery clamp"
        │
        ▼
   Local grammar  ──►  counts, case control, status        (microseconds)
        │
        ▼
   Scene graph    ──►  already know where the scalpel is?  (microseconds)
        │  no
        ▼
   Gemini (WHAT)  ──►  intent=find_object, target=syringe, box=[…]
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
   Renderer       ──► glowing reticle on the black canvas → projector → bench
        │
        ▼
   Voice          ──► "Found syringe."
```

* **Hardware agnostic** — any camera OpenCV can open, any display the OS exposes. No vendor code.
* **AI-provider agnostic** — `AIProvider` interface; Gemini implemented; a clearly separated simulation/mock provider for testing.
* **Local-first** — Gemini answers *what*; OpenCV answers *where* at 20–30 FPS.
* **Measured calibration** — reprojection error is computed on independent validation points and stored; nothing is assumed.
* **Never project uncertainty as certainty** — low confidence → "Target uncertain. Please clarify."; tracking lost → stale graphics are cleared.
* **Simulation mode** — a virtual clinic bench (or workshop table), draggable objects, virtual camera and projector run the *same* code paths, so everything can be developed and tested without hardware.
* **Voice** — hands-free commands and spoken replies through the computer's own microphone; wake word optional; push-to-talk for noisy rooms.
* **Real-time projection** — tracking on downscaled frames, a ≤1080p internal canvas, and a per-object motion model that smooths and predicts between camera frames, so graphics glide at projector frame rate instead of stepping with the camera.
* **Guided procedures** — spoken checklists with the current item highlighted on the bench; custom procedures as JSON.
* **Scene memory** — a persistent model of every object on the surface, so a known item is projected without an AI round-trip and "what's missing?" outlines the empty slot on the tray.
* **Surgical counts** — hands-free initial/added/closing/final counts with the arithmetic read back, a live count board projected on the drape, and a JSON case record at sign-out.
* **Speaks first** — a rule engine watching the tray raises a spoken, projected alert when a sponge or sharp cannot be accounted for, with cooldowns so it never becomes an alarm people ignore.

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
├── simulation/   VirtualWorld (clinic / workshop scenes), SimulatedProjector, SimulatedCamera (ground-truth homographies)
├── voice/        MicrophoneSource + VAD, SpeechToText (Gemini / Whisper / mock), TextToSpeech (pyttsx3),
│                 VoiceAssistant (always-on, barge-in, echo rejection)
├── perception/   SceneGraph: persistent entities, association, presence decay, AI labels on local geometry
├── assistant/    Conversation (reference resolution), local intent grammar, Announcer (priority speech),
│                 JarvisAssistant (routes every utterance; owns case, counts, monitor)
├── surgical/     instrument sets, CountSession (the arithmetic), TrayLayout + Zone, CaseSession (WHO
│                 checklist + case record), SafetyMonitor (proactive rules)
├── healthcare/   domain prompt context, safety policy, procedure library, ProcedureGuide (voice-driven steps)
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

Dependencies: `numpy`, `opencv-contrib-python` (CSRT/KCF trackers), `PySide6`, `pydantic`, `pyyaml`, `google-genai`, `screeninfo`, `sounddevice` (microphone), `pyttsx3` (speech output); optional `mediapipe` for hand tracking (`pip install -e ".[hands]"`) and `faster-whisper` for offline speech recognition (`pip install -e ".[offline-stt]"`).

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
2. **Display / Projector** — the projector is just a second display; BRIDGE pre-selects the secondary one. Press **OPEN PROJECTION**: a borderless black canvas opens fullscreen on that display (Esc closes it). Press **Test graphics** to check you can see a circle, arrow and label on the surface.
3. **Surface** — Table / Wall / Custom (all planar in V1; the preset is stored in the profile).
4. **AUTO CALIBRATE** — the wizard projects bright markers one at a time on a dark background, detects them with the camera, fits a homography, then projects an independent 3×3 validation grid and measures the reprojection error. You get `Accuracy: 2.8 px` and **SAVE** writes `profiles/default.json`. Next time the same camera + display is found the profile is loaded automatically.
5. Say **"Where is the syringe?"** (LISTEN starts automatically once calibrated) or type it and press **EXECUTE**.
   For a case: **"start a case for a minor set"** → **"start the count"** → count aloud → **"final count"** → **"close the case"**.
6. If the reticle sits slightly off the item, enable **Click-to-project test** and nudge with the arrow keys (Shift = 10 px, R = reset). The trim is saved with the profile.

The banner switches to **BRIDGE READY** once a valid calibration is active.

## Calibration

Planar 9-point (default) or 4-point marker calibration with footprint detection and an independent validation grid measured in camera pixels; details, tips and the validation model are in [docs/calibration.md](docs/calibration.md). Key points:

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
pytest                    # 191 tests: geometry, homography, calibration vs simulator ground truth (incl. 4K projector +
                          # small dark camera), AI schema validation, command validation, rendering, tracking, voice
                          # pipeline on synthetic audio, healthcare safety + procedures, scene graph, surgical count
                          # arithmetic, case autopilot, proactive monitor rules, intent grammar, reference resolution,
                          # speech priority queue, full simulated demo and case, GUI smoke
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
* Objects have height; a planar homography maps the surface, so tall items seen by an off-axis camera register slightly off their base — use the registration trim for a fixed setup.
* Voice recognition needs a reasonably quiet room or push-to-talk; the `gemini` provider sends audio clips to Google, `whisper` keeps them local.
* BRIDGE is not a medical device: it locates and guides, it never diagnoses, prescribes or decides doses.
* The surgical count is an **aid to** the team's count, never a replacement for it. A camera cannot see inside a wound or under a drape, so what BRIDGE observes is reported separately from what a person counted, and a reconciled count is never presented as permission to close.
* Scene labels come from the AI and are only as good as the view: similar instruments in a pile, heavy occlusion by hands, or a moved camera all degrade recognition. Items BRIDGE cannot name are reported as unnamed rather than guessed.

## Roadmap

Per-hospital checklist and count-sheet packs → RFID/barcode cross-check for sponges → object-removal verification with hand tracking → per-hospital procedure packs → ChArUco / ArUco calibration → Gray-code structured light and dense correspondence → automatic surface detection and non-planar surfaces → better markerless tracking → hand tracking for action verification → guided assembly workflows on the existing task state machine → voice input → multiple cameras/projectors → other AI providers (OpenAI, local VLMs) → installers (Windows first).

## License

MIT.
