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

## Latency and smooth motion

Three things decide whether the projection feels live:

| stage | cost (720p camera, 4K projector) | how it is kept low |
|---|---|---|
| camera capture | 8–30 fps | `CAP_PROP_BUFFERSIZE=1` so a frame is never queued behind the consumer, MJPG at 720p, optional fixed exposure (auto-exposure collapses to ~8 fps in a dark room) |
| detection + tracking | 65 ms → 15 ms | frames are downscaled to `tracking.process_width` (default 640 px) before detection/tracking; results are scaled back to camera pixels before they reach the homography |
| projector rendering | 332 ms → 8 ms | the canvas is rendered at ≤1080p and upscaled by the window (`internal_canvas_size`), and the QImage wraps the buffer instead of copying and swapping channels |
| motion between measurements | judder + trailing | `render/motion.py`: per-object velocity estimate, exponential smoothing and forward extrapolation evaluated at projector frame rate |

`TargetMotion` is the reason the graphic can look smooth on a slow camera. Every camera measurement calls `observe()`; every rendered frame calls `sample()`, which extrapolates the last measurement forward by its own age, the pipeline latency (`tracking.lead_ms`) and the smoothing filter's own lag, then eases the drawn position toward that prediction. Velocity decays when no new measurement arrives, so a stopped object never drifts. Measured on a 10 fps camera against a 300 px/s target: peak per-frame jump 13.0 → 7.5 px, mean error against ground truth 18.0 → 4.3 px.

Primitives are created once per target and *moved* each frame by `CommandExecutor._animate`, registered as a `ProjectionRenderer.pre_render_hook`. That keeps rendering independent of the camera: a slow camera makes the reticle less accurate, never less smooth.

## Colour as state

| colour | meaning |
|---|---|
| teal `ACCENT` | locked on and tracking |
| amber `WARNING` | re-acquiring, low confidence, or an uncertain answer |
| green `SUCCESS` | target zone / where to place something |
| red `ALERT` | target lost |

The reticle also animates its state: it converges from wide open and fades in over ~0.4 s when it locks on, and fades out when the target is released.

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

## The assistant layer

`BridgeCore.handle_spoken` delegates to `JarvisAssistant.handle`, which is the
single router for everything spoken or typed:

```
raw utterance
   -> healthcare safety gate            refuse dosing/diagnosis in code, before interpretation
   -> running procedure's control words on the raw words: "next" is literal, never rewritten
   -> reference resolution              "the other one" -> "the other artery clamp"
   -> local intent grammar              counts, case control, status, mute      (microseconds)
   -> scene graph                       already know where it is?               (microseconds)
   -> procedure guide + legacy phrases
   -> Gemini                            open-ended requests only
```

Only the last step touches the network, and whatever it finds is folded back into
the scene graph so the same question is answered locally next time.

| module | owns |
|---|---|
| `perception/scene_graph.py` | object identity over time: association, presence decay, AI labels attached to local geometry |
| `assistant/conversation.py` | dialogue memory and pronoun/ellipsis rewriting |
| `assistant/intents.py` | the closed vocabulary matched without the model |
| `assistant/announce.py` | one voice channel, priority queue, preemption, dedupe |
| `surgical/counts.py` | the count arithmetic (the team's numbers, kept apart from the camera's) |
| `surgical/case.py` | case phases, checklist, case record |
| `surgical/monitor.py` | rules that let BRIDGE speak first |
| `render/overlay.py` | count board, tray gaps and alert banner, in their own scene groups |

Threading: `JarvisAssistant.on_frame` runs on the camera thread, throttled to
`assistant.scan_hz`, and never blocks — AI scene labelling is dispatched to a
short-lived worker. `handle` runs on the voice or UI thread under one lock. The
`Announcer` owns its own thread so a slow TTS engine cannot stall either.

Overlays live in the `hud`, `gaps` and `alert` scene groups, disjoint from the
executor's `target`, `zone` and `message` groups, so tracking graphics and the
count board never clobber each other.
