# Gemini integration

## Setup

```
GEMINI_API_KEY=...        # .env or environment; never in source or settings.yaml
BRIDGE_AI_MODEL=...       # optional override, default gemini-3.6-flash
```

Google retires model IDs on its own schedule — a call can start failing with `404 NOT_FOUND`
("this model is no longer available") even though nothing in BRIDGE changed. `GeminiProvider`
treats that as a standing, non-transient failure: it stops calling the API for five minutes
(`_MODEL_UNAVAILABLE_COOLDOWN_S` in `ai/gemini.py`) rather than retrying every `ai_label_interval_s`,
and the diagnostics panel / logs say plainly that `ai.model` needs updating. Check
[ai.google.dev/gemini-api/docs/models](https://ai.google.dev/gemini-api/docs/models) for the
current model ID and either set `ai.model` in `settings.yaml` or export `BRIDGE_AI_MODEL`.

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

One request per user command (plus one for **What next?**), plus the background scene-labelling
loop in `JarvisAssistant._label_scene_async`, which only runs while the always-on assistant has an
active case. That loop is **content-gated, not a blind timer**: every `assistant.ai_label_interval_s`
(default 12s) it checks locally (no network cost) whether anything currently visible is unlabelled
or low-confidence — `SceneGraph.unresolved_label_ids()` — and only spends an actual Gemini (or local
recognizer) call when there is an entity that was *not* already unresolved after the previous pass.
Local contour detection is noisy (a shadow, a tray edge, a wrinkle in gauze can look like a stable
object), so some entities may never earn a confident label no matter how many times they're asked
about; the gate remembers which ones were already asked about and stays quiet about them rather than
re-asking every 12 seconds. Once a tray is fully named and nothing new appears, this loop stops
costing anything at all. As a safety net against a mislabel or a swapped-in instrument going
unnoticed, `assistant.ai_relabel_interval_s` (default 90s) still forces a full re-check on that
cadence regardless of whether anything looks new.

Before this gate existed, the loop fired unconditionally on every tick, which is 300 requests/hour —
enough to exhaust a free-tier key (20 requests/day) within minutes of the assistant starting. With a
paid/billed key this is no longer a hard wall, but it is still real money per request — see
`ai.budget_inr` below. Further options to cut request volume, in order of effort: enable
`recognizer.*` (see [docs/recognition.md](recognition.md)) so Gemini is only asked for objects the
local model didn't confidently name, raise `assistant.ai_label_interval_s` (checks less often), or
raise `assistant.ai_relabel_interval_s` (trusts old labels for longer before forcing a refresh).
Frames are never streamed to the model; local tracking keeps graphics attached to objects between
requests. Typical Flash-tier latency is 1–4 s; the diagnostics panel shows the last request's age.

Every successful call's estimated cost (from the per-model pricing table in `ai/gemini.py`) is
logged and accumulated into a running session total, shown in the diagnostics panel as
**Session spend**. `ai.budget_inr` (default ₹500, `ai.usd_to_inr` for the conversion) is a soft
cap: once the running total reaches it, `GeminiProvider` stops calling Gemini entirely — local
tracking/UI keep working — until the app is restarted or the cap is raised in `settings.yaml`
(or via `BRIDGE_AI_BUDGET_INR`). This is BRIDGE's own estimate, not Google's actual bill; treat it
as a safety net, not an exact figure, and check the real usage/billing dashboard for the account.

## Quota and outage handling

`GeminiProvider` classifies a failed request into several shapes and backs off instead of retrying
immediately:

* **Quota exhaustion** (`429 RESOURCE_EXHAUSTED`) — split further by which cap was hit:
  * *Per-day* (`quotaId` containing `PerDay`, e.g. the free-tier RPD cap) doesn't self-heal within
    Google's own `retryDelay` — that field is a short generic suggestion, not tied to the actual
    daily reset — so the provider backs off for a full hour instead of retrying every few seconds,
    and says plainly that this is a daily cap ("check quota/billing... if this persists").
  * Any other quota error uses Google's own `retryDelay` from the response and refuses further
    calls until it elapses (falls back to 30s if the API doesn't supply one). The user hears "AI
    quota exceeded; retrying automatically in Ns" instead of a generic failure.
  Either way, background labelling silently sits out the cooldown rather than logging an error
  every 12 seconds.
* **Model unavailable** (`404 NOT_FOUND`, e.g. a retired model ID) — this never self-heals by
  retrying, so the provider backs off for five minutes and the log names the problem plainly
  (`ai.model` needs updating), instead of re-attempting the same doomed call every cycle.
* **Overloaded** (`503 UNAVAILABLE`, Google's "high demand" response) — transient and unrelated to
  configuration, so the message says so explicitly rather than suggesting the model/API key are
  wrong; backs off 20s.
* **Session budget reached** (see Cost and latency above) — not a Google error at all; once
  BRIDGE's own running cost estimate hits `ai.budget_inr`, further calls are refused locally
  (no network round-trip) with an honest "session budget reached" message, and stay refused for
  the rest of the run.

All of these are provider-level, so every caller — `ask()`, `next_step()`, and the background
labelling loop — gets the same protection for free.

## Mock providers (tests / simulation only)

* `ScriptedMockProvider` — canned answers for unit tests.
* `SimulationOracleProvider` — answers from the virtual world's ground truth with a tiny rule-based intent parser; used automatically in simulation mode and labelled *Simulation oracle (mock)* in the diagnostics panel.

Neither is used in physical mode unless `ai.provider: mock` is set explicitly (a warning is logged).
