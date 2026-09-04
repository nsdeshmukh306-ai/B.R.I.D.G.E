"""Procedure library: voice-guided, projected checklists for common clinical tasks.

Each step names a target object on the surface (highlighted through the normal
target-resolution pipeline), what the user is expected to do, and the sentence
BRIDGE speaks. Procedures are data (Pydantic) so hospitals can add their own as
JSON in `procedures/` without touching code. The built-ins are general,
widely published sequences (e.g. CLSI order of draw, WHO PPE donning order);
local protocol always wins and is the clinician's responsibility.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

from bridge.interaction.tasks import Task, TaskStep

log = logging.getLogger("bridge.healthcare")

ProjectionAction = Literal["highlight_object", "point_to_object", "show_target_zone", "draw_path", "message"]


class ProcedureStep(BaseModel):
    instruction: str                       # shown on the canvas and spoken
    target_object: str = ""                # label to locate ("" for message-only steps)
    visual_hint: str = ""                  # extra description for the AI ("light blue cap")
    expected_action: Literal["pick_up", "place", "connect", "remove", "observe", "verify"] = "pick_up"
    projection_action: ProjectionAction = "highlight_object"
    verification: Literal["none", "object_removed", "manual"] = "manual"
    caution: str = ""

    def to_task_step(self) -> TaskStep:
        verification = "object_removed" if self.verification == "object_removed" else ("manual" if self.verification == "manual" else "none")
        action = self.projection_action if self.projection_action != "message" else "highlight_object"
        return TaskStep(instruction=self.instruction, target_object=self.target_object, expected_action=self.expected_action,
                        projection_action=action, verification_method=verification)


class Procedure(BaseModel):
    id: str
    name: str
    category: str = "general"
    intro: str = ""
    steps: list[ProcedureStep] = Field(default_factory=list)
    outro: str = "Procedure complete."
    source_note: str = ""  # where the sequence comes from; local protocol overrides

    def to_task(self) -> Task:
        return Task(id=self.id, name=self.name, steps=[s.to_task_step() for s in self.steps])


def _s(instruction: str, target: str = "", hint: str = "", action: str = "pick_up", proj: str = "highlight_object",
       verification: str = "manual", caution: str = "") -> ProcedureStep:
    return ProcedureStep(instruction=instruction, target_object=target, visual_hint=hint, expected_action=action,  # type: ignore[arg-type]
                         projection_action=proj, verification=verification, caution=caution)  # type: ignore[arg-type]


BUILTIN_PROCEDURES: list[Procedure] = [
    Procedure(
        id="order_of_draw", name="Blood collection: order of draw", category="phlebotomy",
        intro="Venipuncture order of draw. I will highlight each tube in sequence.",
        source_note="General CLSI GP41 order; follow your laboratory's own order if it differs.",
        steps=[
            _s("Blood culture bottles first, if ordered.", "blood culture bottle", "aerobic and anaerobic culture bottles"),
            _s("Light blue cap: citrate tube for coagulation. Fill to the line.", "light blue cap tube", "light blue top blood tube"),
            _s("Red or gold cap: serum tube.", "red cap tube", "red or gold top serum tube"),
            _s("Green cap: heparin tube.", "green cap tube", "green top heparin tube"),
            _s("Lavender cap: EDTA tube for haematology. Invert gently eight times.", "lavender cap tube", "purple or lavender top EDTA tube"),
            _s("Grey cap: fluoride tube for glucose, last.", "grey cap tube", "grey top fluoride tube"),
            _s("Label every tube at the bedside before leaving the patient.", "", "", "verify", "message"),
        ],
    ),
    Procedure(
        id="iv_setup", name="IV cannulation tray check", category="nursing",
        intro="IV cannulation tray. I will point to each item; confirm it is present and in date.",
        source_note="Generic tray composition; use your unit's checklist.",
        steps=[
            _s("Gloves.", "gloves", "examination gloves"),
            _s("Tourniquet.", "tourniquet"),
            _s("Alcohol swab for skin preparation.", "alcohol swab", "small sealed swab packet"),
            _s("IV cannula. Check gauge and expiry. Sharp item.", "IV cannula", "cannula in sterile packaging", caution="Sharp item."),
            _s("Transparent dressing to secure the cannula.", "transparent dressing", "clear film dressing"),
            _s("Saline flush syringe.", "saline flush syringe", "prefilled or drawn 10 mL syringe"),
            _s("Sharps container within reach.", "sharps container", "yellow or red sharps bin", "observe", "point_to_object"),
            _s("Tray complete. Perform hand hygiene before starting.", "", "", "verify", "message"),
        ],
    ),
    Procedure(
        id="medication_prep", name="Medication preparation: five rights", category="pharmacy",
        intro="Medication preparation. I will highlight each item; you verify the five rights on the label.",
        source_note="Five rights: right patient, drug, dose, route, time. BRIDGE does not verify these; the clinician does.",
        steps=[
            _s("Prescription or MAR. Confirm patient identity and the order.", "prescription", "printed prescription or chart", "verify"),
            _s("Medication. Read the label aloud: name, strength, expiry.", "medication", "vial, ampoule or blister pack",
               caution="Check the label: drug name, strength, expiry and patient identity."),
            _s("Syringe of the right size for the volume.", "syringe", "syringe in packaging"),
            _s("Drawing-up needle or filter needle. Sharp item.", "needle", "needle in packaging", caution="Sharp item."),
            _s("Alcohol swab for the vial top.", "alcohol swab"),
            _s("Label for the prepared syringe.", "label", "blank medication label"),
            _s("Sharps container.", "sharps container", "", "observe", "point_to_object"),
        ],
    ),
    Procedure(
        id="instrument_count", name="Instrument count", category="surgical",
        intro="Instrument count. I will highlight each instrument; say next when it is counted.",
        source_note="Illustrative basic set; use the tray list for the actual set.",
        steps=[
            _s("Scalpel handle. Sharp item.", "scalpel", "", caution="Sharp item."),
            _s("Forceps.", "forceps", "toothed or non-toothed forceps"),
            _s("Scissors.", "scissors"),
            _s("Needle holder.", "needle holder"),
            _s("Artery clamps.", "artery clamp", "curved haemostat"),
            _s("Suture packs.", "suture", "suture packet"),
            _s("Gauze packs. Count the pieces.", "gauze", "gauze pack"),
            _s("Count complete. Record the count.", "", "", "verify", "message"),
        ],
    ),
    Procedure(
        id="wound_dressing", name="Wound dressing kit", category="nursing",
        intro="Dressing change. I will highlight each item in the order you will use it.",
        steps=[
            _s("Gloves.", "gloves"),
            _s("Cleaning solution.", "saline", "saline bottle or sachet"),
            _s("Gauze for cleaning.", "gauze"),
            _s("Forceps.", "forceps"),
            _s("Primary dressing.", "dressing", "sterile dressing pad"),
            _s("Tape or bandage to secure.", "tape", "adhesive tape roll"),
            _s("Waste bag for the old dressing.", "waste bag", "", "observe", "show_target_zone"),
        ],
    ),
    Procedure(
        id="ppe_donning", name="PPE donning order", category="infection control",
        intro="Putting on protective equipment, in order.",
        source_note="WHO/CDC donning order: hand hygiene, gown, mask or respirator, eye protection, gloves.",
        steps=[
            _s("Hand hygiene first. Twenty seconds.", "", "", "verify", "message"),
            _s("Gown.", "gown", "folded gown"),
            _s("Mask or respirator. Fit check.", "mask", "surgical mask or N95"),
            _s("Eye protection.", "face shield", "face shield or goggles"),
            _s("Gloves over the gown cuffs, last.", "gloves"),
        ],
    ),
    Procedure(
        id="ppe_doffing", name="PPE doffing order", category="infection control",
        intro="Removing protective equipment safely.",
        source_note="Gloves first, then gown, hand hygiene, eye protection, mask, hand hygiene.",
        steps=[
            _s("Gloves first. Peel without touching the outside.", "gloves", "", "remove"),
            _s("Gown. Roll it inward.", "gown", "", "remove"),
            _s("Hand hygiene.", "", "", "verify", "message"),
            _s("Eye protection, by the strap.", "face shield", "", "remove"),
            _s("Mask, by the ties or loops.", "mask", "", "remove"),
            _s("Hand hygiene again.", "", "", "verify", "message"),
        ],
    ),
    Procedure(
        id="sample_handling", name="Specimen handling", category="laboratory",
        intro="Specimen handling at the bench.",
        steps=[
            _s("Specimen container. Check the lid is sealed.", "specimen container", "", "verify"),
            _s("Request form. Match name and ID with the container label.", "request form", "paper form", "verify"),
            _s("Biohazard transport bag.", "biohazard bag", "", "place", "show_target_zone"),
            _s("Place the container in the bag and the form in the outer pocket.", "", "", "verify", "message"),
        ],
    ),
]


class ProcedureLibrary:
    def __init__(self, extra_dir: Path | None = Path("procedures")):
        self.procedures: dict[str, Procedure] = {p.id: p for p in BUILTIN_PROCEDURES}
        if extra_dir is not None and Path(extra_dir).is_dir():
            for f in sorted(Path(extra_dir).glob("*.json")):
                try:
                    p = Procedure.model_validate(json.loads(f.read_text()))
                    self.procedures[p.id] = p
                    log.info("Loaded custom procedure %s from %s", p.id, f)
                except Exception:  # noqa: BLE001
                    log.exception("Invalid procedure file %s", f)

    def get(self, key: str) -> Optional[Procedure]:
        return self.procedures.get(key)

    def find(self, spoken: str) -> Optional[Procedure]:
        """Fuzzy match a spoken name ('start the order of draw', 'IV tray')."""
        t = spoken.lower()
        best, best_score = None, 0
        for p in self.procedures.values():
            words = set(p.name.lower().replace(":", "").split()) | set(p.id.replace("_", " ").split())
            score = sum(1 for w in words if len(w) > 2 and w in t)
            if score > best_score:
                best, best_score = p, score
        return best if best_score >= 1 else None

    def names(self) -> list[tuple[str, str]]:
        return [(p.id, p.name) for p in self.procedures.values()]
