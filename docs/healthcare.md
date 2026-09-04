# Healthcare assistance

BRIDGE's objective is to assist with healthcare activities and operations at the point of work: the medication counter, the phlebotomy tray, the dressing trolley, the instrument table, the laboratory bench. It projects onto the real items so the assistance is exactly where the hands are, and it is driven by voice so hands stay clean.

## What it does

* **Locate** — *"Where is the light blue tube?"*: a reticle lands on the item and follows it if moved.
* **Point / zone** — *"Point to the sharps container"*, *"Where should I put the needle?"*.
* **Guided procedures** — spoken, step-by-step checklists whose current item is highlighted on the bench: order of draw, IV cannulation tray check, medication preparation (five rights), instrument count, wound dressing kit, PPE donning/doffing, specimen handling. Steps advance on *next*, and pick-up steps can auto-advance when the tracker sees the item leave the bench.
* **Next step** — *"What next?"*: the AI proposes the next item from the scene.

## What it deliberately does not do

BRIDGE is a spatial guide, not a medical device. It does not diagnose, prescribe, calculate doses, or decide whether something is safe to give. Requests of that kind are declined in code (`healthcare/safety.py`) before they reach the AI, and the AI prompt (`healthcare/prompts.py`) instructs the model never to guess an illegible label. Every medication-related answer ends with *"Check the label: drug name, strength, expiry and patient identity before use."* and every sharp with a sharps caution. Local protocol always overrides the built-in sequences.

## Procedure library

Built-ins live in `healthcare/procedures.py`; each step has an instruction (shown and spoken), a target label, a visual hint for the AI, the expected action and a caution. Hospitals can add their own as JSON in `procedures/*.json`:

```json
{
  "id": "chest_drain_tray",
  "name": "Chest drain insertion tray",
  "category": "procedure room",
  "intro": "Chest drain tray check.",
  "steps": [
    {"instruction": "Sterile gloves.", "target_object": "sterile gloves"},
    {"instruction": "Local anaesthetic. Read the label.", "target_object": "lidocaine vial", "visual_hint": "small glass vial",
     "caution": "Check the label: drug name, strength, expiry and patient identity."},
    {"instruction": "Scalpel. Sharp item.", "target_object": "scalpel"},
    {"instruction": "Tray complete.", "expected_action": "verify", "projection_action": "message"}
  ]
}
```

`projection_action` is `highlight_object` (default), `point_to_object`, `show_target_zone` or `message`; `verification` can be `object_removed` to auto-advance when the highlighted item is picked up.

## Simulation

Simulation mode starts on a virtual clinic bench (blood tubes by cap colour, syringe, needle, swab, gauze, gloves, scalpel, forceps, tourniquet, sharps container) so every procedure can be rehearsed without hardware. The simulated AI is an oracle over the virtual scene; on real hardware Gemini identifies items from the camera image.

## Recognition limits

Cap colours, packaging and label text are what the model sees. Similar syringes or tubes of the same colour will produce *"Target uncertain. Please clarify."* — say which one ("the 10 mL syringe", "the tube nearest the rack"). Items overlapping or stacked confuse both the AI box and the tracker; spread the tray out.
