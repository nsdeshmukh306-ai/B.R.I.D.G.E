# BRIDGE architecture

## Pipeline

```
                    USER
                     |
             natural language
                     |
                     v
              +-------------+
              |   GEMINI    |   ai/gemini.py  (AIProvider)
              | what is it? |   -> TargetIdentification (Pydantic-validated JSON)
              +------+------+
                     |
                     v
              CommandPlanner        interaction/commands.py  -> HighlightObject | PointToObject | ...
                     |
                     v
+----------+   +-------------------+   +------------+
| CAMERA   |-->| BRIDGE CORE       |-->| PROJECTOR  |
| any USB  |   |  TargetResolver   |   | any display|
| camera   |   |  ObjectTracker    |   +------+-----+
+----------+   |  CoordinateMapper |          |
               |  Renderer         |          v
               +-------------------+    PHYSICAL SURFACE
```

## Threads

| Thread | Owner | Work |
|---|---|---|
| `camera-capture` | `CameraManager` | reads frames from the `FrameSource`, publishes to listeners |
| camera listener → `BridgeCore._on_frame` | core | `CommandExecutor.update(frame)`: tracker update + re-render of target graphics; in simulation also pumps the renderer into the virtual projector |
| Qt GUI thread | `MainWindow`, `ProjectionWindow` | UI refresh at ~15 Hz; projection window repaints from the renderer at up to 60 Hz |
| worker (`QThreadPool`) | wizard / EXECUTE button | `BridgeCore.calibrate()` and `BridgeCore.ask()` (blocking AI call) |

`CommandExecutor` holds an `RLock` so a command execution and a frame update never interleave. The renderer's `Scene` is thread-safe.

## Coordinate systems

* **Camera space** — pixels of the camera frame. Detections, tracking states and AI boxes (after de-normalization) live here.
* **Projector space** — pixels of the projection canvas (= selected display resolution). All `render/` primitives live here.
* **Physical surface** — implicit: the planar surface is the image plane both devices look at. With a flat surface, camera↔projector is a single 3×3 homography `H` (`spatial/homography.py`), so no explicit world coordinates are needed in V1. The simulator does have explicit "table" coordinates, which is what makes ground-truth tests possible.

`CoordinateMapper` provides `camera_to_projector`, `projector_to_camera`, bbox→polygon and local radius scaling.

## Calibration engine

`CalibrationMethod` (ABC: `calibrate / validate / save / load`) → `PlanarMarkerCalibration`. `CalibrationEngine` runs a method, keeps the result, persists `CalibrationProfile`s (`ProfileStore`) and invalidates on device changes. The engine talks to hardware through two tiny protocols so the same code drives a real `ProjectionWindow` + `CameraManager` or the simulator:

```python
class ProjectorLink: show_image(image); size() -> (w, h)
class CameraLink:    capture(settle_s) -> frame; size() -> (w, h)
```

Future ChArUco / Gray-code methods only need to implement `CalibrationMethod`.

## Target resolution priority

1. an object already tracked with the requested label;
2. the AI's bounding boxes (snapped to a local contour when IoU ≥ 0.45 to tighten loose boxes);
3. the AI's visual description matched against local contours by colour;
4. (caller) ask the AI again.

## Self-projection suppression

The camera sees what the projector draws. Before every tracker update the executor renders a mask of the current graphics in projector space, warps it into camera space with `H⁻¹`, dilates it and replaces those pixels with a blurred background estimate (`vision/suppression.py`). Without this the tracker latches onto the projected ring around the object (measured in simulation: max following error 48 px → 25 px).

## Safety behaviour

| Condition | Behaviour |
|---|---|
| no valid calibration | spatial commands refused; canvas shows *Spatial calibration required.* |
| AI confidence < `ai.min_confidence` | *Target uncertain. Please clarify.* + the model's message |
| target not found locally | *Cannot find X. Please clarify.* |
| tracking lost for `lost_after_frames` | target graphics cleared, *Target lost.*, `TRACKING_LOST` event |
| camera / display / resolution change | calibration invalidated with an explicit message |

## Task state machine (`interaction/tasks.py`)

`IDLE → UNDERSTAND → LOCATE → PROJECT → WAIT_FOR_ACTION → VERIFY → NEXT_STEP → …/DONE`, with legal transitions enforced. `BridgeCore.next_step()` uses `AIProvider.plan_action` for single-step assembly guidance today; verification hooks (`object_removed`, hand tracking) plug into `TaskRunner.action_observed`.

## Future extension points

Multiple cameras/projectors (one `CoordinateMapper` per pair), non-planar surfaces (replace `CoordinateMapper` with a ray-mapping implementation behind the same interface), other AI providers (`AIProvider`), other trackers (`_make_cv_tracker`).
