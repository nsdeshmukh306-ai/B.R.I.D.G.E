# Gemini integration

## Setup

```
GEMINI_API_KEY=...        # .env or environment; never in source or settings.yaml
BRIDGE_AI_MODEL=...       # optional override, default gemini-2.5-flash
```

`GeminiProvider` (`ai/gemini.py`) uses the official `google-genai` SDK: `client.models.generate_content` with the camera frame as a JPEG part (downscaled to ≤ 1024 px on the long edge), `response_mime_type="application/json"` and `response_schema` set to the Pydantic model, temperature 0.1.

## Interface

```python
class AIProvider:
    understand_scene(frame) -> SceneUnderstanding
    identify_target(frame, query, known_labels) -> TargetIdentification
    plan_action(frame, task_context) -> ActionPlan
```

## Structured responses

```json
{
  "intent": "find_object",
  "target": "screwdriver",
  "visual_description": "a red-handled screwdriver",
  "confidence": 0.94,
  "boxes": [{"box_2d": [231, 268, 286, 501], "label": "screwdriver"}],
  "style": "circle",
  "animation": "pulse",
  "message": ""
}
```

* `intent` ∈ `find_object, find_all, highlight_object, point_to_object, label_object, show_target_zone, draw_path, show_message, clear_projection, describe_scene, next_step, unknown`.
* `box_2d` is `[ymin, xmin, ymax, xmax]` on a **0–1000** scale (Gemini's convention). `NormalizedBox.to_pixels(w, h)` converts to camera pixels; this is unit-tested.
* `secondary_target` / `secondary_boxes` carry the destination for `draw_path` and `show_target_zone`.

Every response is parsed by `StructuredResponseParser` (strips code fences, extracts the JSON object) and validated against the Pydantic schema. Unknown intents, out-of-range confidences, malformed boxes or non-JSON text raise `AIError` and the user sees *AI request failed.* Nothing the model returns is ever executed as code.

## From response to projection

`CommandPlanner` turns the validated `TargetIdentification` into one of the controlled commands (`interaction/commands.py`). Confidence below `ai.min_confidence` (default 0.5) becomes `ShowMessage("Target uncertain. Please clarify.")`. The `CommandExecutor` then resolves the target locally, starts tracking and renders.

## Cost and latency

One request per user command (plus one for **What next?**). Frames are never streamed to the model; local tracking keeps graphics attached to objects. Typical `gemini-2.5-flash` latency is 1–4 s; the diagnostics panel shows the last request's age.

## Mock providers (tests / simulation only)

* `ScriptedMockProvider` — canned answers for unit tests.
* `SimulationOracleProvider` — answers from the virtual world's ground truth with a tiny rule-based intent parser; used automatically in simulation mode and labelled *Simulation oracle (mock)* in the diagnostics panel.

Neither is used in physical mode unless `ai.provider: mock` is set explicitly (a warning is logged).
