# Local instrument recognition

## What this is

Until now, BRIDGE has had exactly one source of *what* an object is: Gemini.
That's a reasonable default for a general-purpose assistant, but it has two
weaknesses that matter specifically for this project's SaaS pivot toward
rural hospitals and PHCs:

1. **Accuracy.** A general vision-language model is not trained on surgical
   instruments specifically. It is asked to tell a Metzenbaum scissors from a
   Mayo scissors from a set of reflective steel tools under theatre
   lighting — precisely the "open technical question" flagged in Document 1's
   validation section. A model trained on that exact problem should do
   better.
2. **Network dependence.** Gemini is a cloud API call. The whole premise of
   the SaaS pivot is that the core assistant works with no reliable internet.
   Object *naming* is the one place that premise was still compromised.

`src/bridge/vision/surgical_recognizer.py` adds a second, narrower labeller:
a local object-detection model, run entirely on-device through ONNX Runtime,
tried first whenever one is configured. It changes nothing else about how
BRIDGE works — the "what vs where" split holds exactly as before (this
supplies *what*; local CV still supplies *where*), results reach the scene
graph through the same `SceneGraph.apply_ai()` path Gemini already used, and
no model output is ever anything but a label and a box.

## What is and isn't shipped

**Shipped:** the recognizer code (`vision/surgical_recognizer.py`), the
config surface (`recognizer:` in `settings.yaml`), and the wiring into
`JarvisAssistant`. All of that is BRIDGE's own MIT-licensed code and ships
like everything else.

**Not shipped, on purpose: any model weights.** `recognizer.weights_path`
and `recognizer.labels_path` are `null` by default and point at files a
deployer supplies. This is a direct consequence of what was found while
researching which pretrained model to point this at.

## What was evaluated, and why nothing is bundled

The most relevant pretrained, downloadable option found was the **Rona
surgical instrument detector** published by DocCheck
([GitHub](https://github.com/DocCheck/Surgical-Instrument-Detector),
[Hugging Face](https://huggingface.co/DocCheck/medical-instrument-detection)):
a YOLOv5 model trained on 12 surgical instrument classes, built for
DocCheck's own open-source surgical-assistance robot. Technically it is a
good match for this recognizer's interface. Licence-wise it is not a good
match for BRIDGE as a SaaS product:

| Component | Licence | What it means for a paid SaaS deployment |
|---|---|---|
| Pretrained weights (Hugging Face) | CC-BY-NC-4.0 | **Non-commercial only.** Using these weights in a product hospitals pay for is exactly what this licence prohibits. |
| Training/inference code (GitHub) | GNU AGPLv3 | Copyleft, and specifically triggers on network use: if BRIDGE's own code incorporated AGPL code and were offered as a network service, the AGPL requires offering the complete corresponding source of the whole service to every user. That is a materially different obligation than MIT, and one a commercial vendor needs to decide on deliberately, not inherit by accident. |

Neither of those is a reason not to use this model at all — for a hackathon
demo, a thesis defence, or any non-commercial evaluation, it is a genuinely
reasonable choice, properly attributed. It is a reason the recognizer ships
with *no* weights and treats "point me at a model" as the deployer's
decision, made with the licence terms in front of them, rather than baking a
non-commercial dependency into a product being pitched as sellable software.

A few research-grade datasets (m2cai16-tool-locations, Cholec80,
CholecTrack20, all from the CAMMA lab) came up during the same research pass.
They are the datasets most published surgical-instrument detectors are
trained on, but they are distributed under research-use agreements, not
licences that clear a path to redistributing derived commercial weights
either.

## What a commercial path looks like

Three honest options, in order of effort:

1. **Use the DocCheck/Rona weights for now, non-commercially.** Fine for the
   jury demo, a thesis defence, or a pilot explicitly run as a research
   evaluation. Attribute DocCheck. Do not ship it in anything sold.
2. **License weights or a dataset commercially.** Some academic groups will
   license research data for commercial use on request; this is a
   conversation, not an engineering task.
3. **Train an in-house model.** The architecture here is deliberately
   decoupled from any specific model: `decode_yolo_onnx_output` accepts any
   YOLOv5/v8-shaped ONNX export, so a model trained from scratch (or
   fine-tuned from a permissively-licensed base such as an Apache-2.0
   checkpoint) on instrument photos BRIDGE itself is allowed to use commercially
   drops in with no code change — only new files at `recognizer.weights_path`
   and `recognizer.labels_path`.

This mirrors the honesty pattern in Document 1's validation section: what is
built is built, what is not solved is stated plainly, and nothing here
pretends licensing has been resolved when it hasn't.

## Second evaluation pass (2026-09-15): SurgVISTA, GSViT, BariatricSurgeryGPT

Niraj asked directly whether BRIDGE should train its models on three named
projects. All three turned out to be real, recent research releases — but
none of them clears the bar the DocCheck/Rona evaluation above didn't
clear either, and for three different reasons. Recorded here so this
research isn't repeated.

| Candidate | What it actually is | Why it doesn't fit here |
|---|---|---|
| [SurgVISTA](https://github.com/isyangshu/SurgVISTA) | A self-supervised **video foundation model** (spatiotemporal reconstruction pre-training) for surgical video understanding — phase recognition and instrument/verb/target *triplet recognition*, not object detection. | Two independent blockers. (1) **Licence: CC-BY-NC-ND-4.0** — non-commercial *and* no-derivatives, stricter than Rona's CC-BY-NC (a paid SaaS product can't use it, and can't fine-tune around the restriction either since derivatives are also barred). (2) **Wrong output shape.** It produces frame/clip-level embeddings and classifications for downstream fine-tuning, not the `(label, box_xyxy)` detections `surgical_recognizer.py`'s interface expects — using it would mean redesigning the recognizer around a different task, not just swapping a weights file. |
| [GSViT](https://github.com/SamuelSchmidgall/GSViT) | A video-pretrained foundation model for general surgery, evaluated on the Cholec80 **phase-recognition** task, explicitly built for real-time inference. | The GitHub repo carries **no LICENSE file at all** — under default copyright that means all rights reserved, no licence is granted for any use (commercial or non-commercial) without asking the author directly. Weights aren't published openly either; the README asks for an email request through a gated SharePoint link, i.e. research-collaboration distribution, not something a product can depend on or redistribute. Same task-shape mismatch as SurgVISTA: phase recognition, not instrument bounding boxes. |
| [BariatricSurgeryGPT](https://doi.org/10.1177/15533506251400130) | A GPT-2-based **text** LLM, fine-tuned on ~8,800 PubMed bariatric-surgery abstracts, proposed for surgical education, patient communication, and — explicitly — **"clinical decision-making"** support. | Not a vision model at all, so it has nothing to do with the instrument-recognition problem this file is about. More importantly: no weights or code are publicly released (the paper is paywalled with no stated release plan), and its own stated purpose — synthesizing evidence to support clinical decisions — is precisely the category BRIDGE's safety gate (`healthcare/safety.py`, the dual dosing/diagnosis refusal in `JarvisAssistant.handle` / `BridgeCore.ask`) exists to refuse. Even a hypothetical future release of this model would need to stay outside BRIDGE entirely, not get wired in — it's a different, and for this product a disallowed, kind of tool. |

None of the three is usable for BRIDGE as shipped, and the reasons aren't
the same reason twice: a licence that forbids commercial use and
derivatives, a repo with no licence and gated access, and a text model whose
purpose is outside the product's own safety boundary. No training run and
no integration work was done against any of them — there is nothing legal
or architecturally sound to build yet. Nothing in `vision/surgical_recognizer.py`
changed as a result of this pass; the "what a commercial path looks like"
options above still stand as the real next steps.

## How it fits the routing

```
_label_scene_async(frame):
    if local recognizer configured and loaded:
        run it on the frame                     -> apply_ai(..., source="specialist-local")
    if a Gemini provider is configured:
        run it on the frame too                 -> apply_ai(..., source="gemini")
```

Both run every `assistant.ai_label_interval_s` seconds, in the background,
never per-frame — identical cadence to the Gemini-only behaviour before this
change. The local recognizer runs first because it is faster (no network)
and, when it is configured at all, it is presumed more accurate for this
specific domain; Gemini still runs afterward so anything outside the local
model's trained classes (a specialised instrument it was never trained on)
still gets named. A site with recognizer.enabled: true and no internet at
all keeps recognizing instruments; a site with neither configured behaves
exactly as v0.4 did.

## Format expected

A YOLOv5 or YOLOv8/v11-family object detector exported to ONNX
(`yolo export format=onnx` in Ultralytics' own tooling, or the equivalent for
another training framework), plus a plain-text labels file — one class name
per line, in the exact order the model was trained on. `input_size` defaults
to 640, matching the common YOLO export; letterboxing to a square canvas is
handled internally so a differently-shaped source frame is fine.
