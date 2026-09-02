# Troubleshooting

## Devices

**No cameras found** — on Linux check `/dev/video*` and permissions (`sudo usermod -aG video $USER`); on Windows close other apps that hold the camera (Teams, Zoom); on macOS grant camera permission to the terminal/Python. Press ⟳ to re-enumerate.

**Camera opens but shows no frames / FPS 0** — some virtual cameras open but never deliver; select another index. Try a lower resolution in Settings.

**Projector not listed** — the OS must see it as an extended display (not mirrored). On Windows use *Extend* in the display settings. Press ⟳.

**Projection window opened on the laptop screen** — choose the other display in DISPLAY / PROJECTOR and press OPEN PROJECTION again. Esc closes the projection window.

**`AttributeError: module 'cv2' has no attribute 'TrackerCSRT_create'`** / tracker falls back to `color` — you have `opencv-python` installed next to `opencv-contrib-python`. `pip uninstall opencv-python opencv-python-headless` and keep only `opencv-contrib-python`.

## Calibration

**Calibration failed — only N of 4 markers detected**

* the camera cannot see the whole projected image → move/tilt the camera, or reduce projector zoom;
* room too bright → dim lights, close blinds;
* marker too small or blurred → increase `marker_radius_px`;
* camera auto-exposure reacting to the black/white flashes → lock exposure;
* a hand or object covers a corner.

**Reprojection error too high** — the surface is not flat (curved, cluttered with tall objects), the camera moved during calibration, or a reflection was detected as a marker. Clear the surface and retry; use `planar_9point`.

**Calibration invalidated: camera / display / resolution changed** — expected; BRIDGE refuses to reuse a profile for different hardware. Re-run AUTO CALIBRATE or re-select the original device.

**Projected marker is offset from where I clicked** — the camera or projector moved since calibration; recalibrate.

## AI

**AI unavailable: GEMINI_API_KEY is not set** — create `.env` from `.env.example`.

**Gemini request failed: 4xx/5xx** — check key validity, quota, network/proxy. The last error is shown in the diagnostics panel.

**Target uncertain. Please clarify.** — the model's confidence was below `ai.min_confidence`. Be more specific ("the red screwdriver"), improve lighting, or lower the threshold in Settings.

**Cannot find X** — the model named the object but no local contour matched; the surface may be too busy or the object too similar to the background. Try again after moving it to a clear area.

## Tracking

**Target lost.** — the object left the view, was covered, or moved too fast. BRIDGE clears the marker instead of leaving a wrong one. Ask again.

**Circle jitters / drifts** — the projected ring is being picked up by the tracker: keep `line_width` modest and make sure the calibration is accurate so self-projection suppression aligns. A CSRT backend is more stable than `color`.

## Performance

Camera processing target is 20–30 FPS; if lower, reduce camera resolution (Settings → 1280×720 or 640×480) or use the `kcf` tracker. The projection renders at up to 60 FPS independently of the camera.

## Logs

`logs/bridge.log` (rotating), console, and the in-app LOG panel. Run with `--log-level DEBUG` for per-frame detail.
