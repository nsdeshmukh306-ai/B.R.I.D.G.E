# Calibration

## Goal

```
camera image coordinates  --H-->  projector coordinates
```

For a planar surface a single 3×3 homography `H` maps camera pixels to projector pixels. `p_proj ~ H · [x, y, 1]ᵀ` (normalised by the third component).

## Planar marker method (V1, `planar_4point` / `planar_9point`)

1. Project a solid black frame (reference), then a solid white frame. The lit quadrilateral is the projector's **footprint** in the camera: its corners give a coarse homography and a region of interest, so reflections elsewhere in the room are ignored.
2. For each layout point (4 corners inset by `margin_fraction`, or a 3×3 grid): project one bright white disc, capture, subtract the reference and find the blob nearest to where the coarse homography predicts it. Disc radius is at least 3 % of the projector's short edge (≈65 px on a 4K projector) and grows by 70 % on each retry, so markers stay visible even when a small webcam watches a 4K projection from across a room.
3. Projecting markers *one at a time* plus the predicted position makes the correspondence unambiguous. If fewer than four discs are found but the footprint quad is clean, its corners are used as correspondences to the projector image corners.
4. `cv2.findHomography(camera_points, projector_points)` (exact solve for 4 points, RANSAC for 9). Degenerate solutions are rejected.
5. **Validation**: project an *independent* 3×3 grid (inset 25 %), detect, and compute reprojection error `|H·cam − proj|` per point. Stored as

```json
{"mean_error_px": 1.6, "max_error_px": 6.0, "median_error_px": 1.2, "n_points": 9, "threshold_px": 10.0, "valid": true}
```

Errors are measured in **camera pixels** (`mean_error_px`, `max_error_px`); projector-space values are kept alongside (`mean_error_proj_px`). One camera pixel can cover ten projector pixels when a 4K projector is watched by a 640 px webcam, so projector-space thresholds would be meaningless. `valid` requires mean ≤ threshold, max ≤ 2.5 × threshold and ≥ 75 % inliers; the threshold is `calibration.validation_threshold_px` (default 6 camera px). A calibration whose independent validation could not be measured is **never** marked valid, and profiles without an independent validation are never reloaded.

6. The projector returns to the white canvas. On failure the engine retries up to `max_retries` times and reports the causes checklist.

In simulation the method recovers the ground-truth homography to < 1 px (tested in `tests/test_calibration.py`).

## Tips for a good physical calibration

* Dim the room a little; the markers must be clearly brighter than ambient light on the surface.
* Turn off camera auto-exposure if the projected discs saturate or vanish; ~1/60 s exposure usually works.
* Make sure the whole projected image is inside the camera view and not clipped by the table edge.
* Keep hands and shiny objects away from the four corners while calibrating.
* Increase `marker_radius_px` (Settings) if the camera is far away or low resolution; decrease it if blobs merge.
* Matte, light surfaces work best. Dark or glossy surfaces reduce contrast.
* Use `planar_9point` for slightly better accuracy on larger surfaces (more points → RANSAC → outlier rejection).

## Checking registration

Enable **Click-to-project test** and click a point in the camera view: BRIDGE projects a dot + circle at that spot. If the projected dot lands where you clicked on the physical surface, registration is good. The reported accuracy is only what was measured on the validation grid — no millimetre accuracy is claimed.

## Profiles

`profiles/<name>.json` stores camera id + resolution, display id + resolution, surface type, homography, workspace polygons and the validation result. On start-up BRIDGE looks for a valid profile matching the current camera and display; any mismatch in device or resolution invalidates it with an explicit message:

```
Camera changed. Calibration invalidated.
Projector/display changed. Calibration invalidated.
Display resolution changed. Calibration invalidated.
```

Moving the camera or projector physically also invalidates the calibration — BRIDGE cannot detect that automatically in V1; re-run AUTO CALIBRATE.

## Future methods

The `CalibrationMethod` interface is designed for ChArUco/ArUco boards (robust to ambient light, gives camera intrinsics), Gray-code structured light (dense per-pixel correspondence, non-planar surfaces) and surface segmentation. They slot in via `spatial.calibration.build_method`.
