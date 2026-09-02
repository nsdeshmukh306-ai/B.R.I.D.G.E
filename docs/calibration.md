# Calibration

## Goal

```
camera image coordinates  --H-->  projector coordinates
```

For a planar surface a single 3×3 homography `H` maps camera pixels to projector pixels. `p_proj ~ H · [x, y, 1]ᵀ` (normalised by the third component).

## Planar marker method (V1, `planar_4point` / `planar_9point`)

1. Project a solid black frame; capture a **reference** image.
2. For each layout point (4 corners inset by `margin_fraction`, or a 3×3 grid): project one bright white disc (`marker_radius_px`), capture, subtract the reference, threshold (Otsu), find the most circular blob, refine its centre with an intensity-weighted centroid.
3. Projecting markers *one at a time* gives an unambiguous projector→camera correspondence. As a fallback the full pattern is projected and blobs are matched to the layout by angular order.
4. `cv2.findHomography(camera_points, projector_points)` (exact solve for 4 points, RANSAC for 9). Degenerate solutions are rejected.
5. **Validation**: project an *independent* 3×3 grid (inset 25 %), detect, and compute reprojection error `|H·cam − proj|` per point. Stored as

```json
{"mean_error_px": 1.6, "max_error_px": 6.0, "median_error_px": 1.2, "n_points": 9, "threshold_px": 10.0, "valid": true}
```

`valid` requires mean ≤ threshold, max ≤ 2.5 × threshold and ≥ 75 % inliers. The threshold is `calibration.validation_threshold_px` (default 10 px in projector space).

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
