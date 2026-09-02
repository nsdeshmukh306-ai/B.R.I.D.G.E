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

## Layout
`src/bridge/{app,camera,projector,spatial,vision,ai,interaction,render,simulation,ui}`, `tests/`, `docs/`, `profiles/` (git-ignored JSON calibration profiles).

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

## Testing philosophy
The simulator (`simulation/world.py`) knows the ground-truth camera→projector homography, so calibration accuracy, target localisation and tracking are asserted numerically (`tests/test_pipeline_and_devices.py::test_full_demo_calibrate_ask_follow`). Anything that cannot be tested physically gets a deterministic software test here.
