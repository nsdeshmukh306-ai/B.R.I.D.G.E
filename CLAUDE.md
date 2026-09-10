# CLAUDE.md — working on BRIDGE

## What this is
Desktop (PySide6) app: commodity camera + commodity projector + Gemini → spatially registered graphics projected onto a physical planar surface. Python 3.11+, OpenCV, NumPy, Pydantic.

## Non-negotiable architecture rules
- Keep **AI reasoning** (`ai/`), **perception** (`vision/`), **spatial mapping** (`spatial/`) and **rendering** (`render/`) separate. They only meet in `interaction/executor.py`.
- Gemini answers *what*; local CV answers *where*. **Never** call the AI per frame.
- All AI output is validated by Pydantic (`ai/schemas.py`) and converted to the controlled vocabulary in `interaction/commands.py`. No model-generated code is ever executed.
- Renderer takes **projector coordinates** only. The only camera→projector conversion for user graphics happens in `CommandExecutor` via `CoordinateMapper`.
- No spatial graphics without a valid calibration; low confidence → "Target uncertain."; tracking lost → clear the graphic.
- No vendor-specific hardware code. Cameras are OpenCV indices; projectors are OS displays.
- Secrets come from the environment / `.env` only.
- Mock/simulation AI providers (`ai/mock.py`) are for tests and simulation mode only; never in the physical path.
- Healthcare safety: dosing/diagnosis requests are declined in `healthcare/safety.py` before the AI; cautions for sharps/medications are appended to spoken replies. Keep it that way.
- Voice: `BridgeCore.handle_spoken(text) -> reply` is the single entry point. It delegates to `JarvisAssistant.handle`, which routes: safety gate -> running procedure's control words -> reference resolution -> local intent grammar -> scene graph -> procedure guide -> AI. Every new capability plugs into that order; nothing bypasses the safety gate.
- The safety gate runs **twice**, on purpose: in `JarvisAssistant.handle` before the intent grammar, and again in `BridgeCore.ask` before the AI. Never remove either.
- Surgical counts are arithmetic the team owns. `CountLine.counted_*` is what a person counted; `observed` is what the camera sees. Never let vision write a count, and never phrase a reconciled count as permission to close.
- The scene graph (`perception/`) is the only place object identity persists. AI supplies labels, local CV supplies positions — never the reverse.
- Proactive alerts need a key and a cooldown, state an observation rather than a conclusion, and use "I can/cannot see" wording.

## Layout
`src/bridge/{app,camera,projector,spatial,vision,perception,ai,interaction,assistant,surgical,render,simulation,voice,healthcare,ui}`, `tests/`, `docs/`, `profiles/` (git-ignored JSON calibration profiles), `procedures/` (optional custom procedure JSON), `sets/` (optional custom instrument-set JSON), `case_records/` (git-ignored case records).

## Commands
```bash
pip install -e ".[dev]"
pytest                          # full suite, headless Qt
bridge --simulation             # GUI without hardware
bridge --headless-selftest      # CI smoke: calibrate + query in simulation
```

## Extending
- Calibration method: subclass `spatial.calibration.CalibrationMethod`, register in `build_method`.
- AI provider: subclass `ai.base.AIProvider`, register in `ai.mock.build_provider`.
- Tracker backend: add to `vision.tracking._make_cv_tracker` and the `TrackingSettings.backend` Literal.
- New spatial command: add a Pydantic model to `interaction/commands.py`, map it in `CommandPlanner.plan`, handle it in `CommandExecutor`.
- New spoken command: add a pattern + `Kind` in `assistant/intents.py`, handle it in `JarvisAssistant._dispatch`. Keep patterns conservative — `unknown` means "ask the model", never "guess".
- New proactive rule: add a `_rule_*` method to `SafetyMonitor` and call it from `evaluate`. Use `_held_for` for a grace period and give it a stable key.
- New instrument set / checklist: JSON in `sets/` matching `InstrumentSet`, or `case.load_checklist(path)`.

## Testing philosophy
The simulator (`simulation/world.py`) knows the ground-truth camera→projector homography, so calibration accuracy, target localisation and tracking are asserted numerically (`tests/test_pipeline_and_devices.py::test_full_demo_calibrate_ask_follow`). Anything that cannot be tested physically gets a deterministic software test here.
