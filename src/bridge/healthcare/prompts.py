"""Healthcare domain context injected into every AI prompt, plus the safety policy.

BRIDGE is an assistive spatial guide for clinical *workspaces*: it locates items,
walks through procedural checklists and projects instructions onto the bench,
tray or bedside table. It is not a medical device and never diagnoses, prescribes
or decides a dose. Those rules are part of the prompt and are enforced again in
code (see `healthcare.safety`).

SURGICAL_INSTRUMENT_CONTEXT below intentionally mirrors the exact instrument
names and aliases in `surgical.sets.BUILTIN_SETS` (standard, widely published OR
tray compositions), not a separately invented vocabulary — so the vision model's
"what is this" labels line up with what the local count-matching system
(`surgical.counts.CountLine.matches`) already expects, instead of drifting apart
on synonyms.
"""
from __future__ import annotations

HEALTHCARE_CONTEXT = (
    "Domain: healthcare workspaces (hospital ward, OPD, procedure room, laboratory bench, pharmacy counter, "
    "home care). Typical items on the surface: syringes (1/2/5/10/20 mL), needles, cannulas/IV catheters, IV bags and "
    "giving sets, three-way taps, vials, ampoules, blister packs, pill bottles, pill organisers, blood collection tubes "
    "identified by cap colour (lavender/purple EDTA, red or gold serum, light blue citrate, green heparin, grey fluoride, "
    "yellow/aerobic and purple/anaerobic blood-culture bottles), tourniquet, alcohol swabs, gauze, cotton, dressings, "
    "tapes, bandages, sutures, scalpel, forceps, scissors, kidney tray, sharps container, gloves, masks, gowns, "
    "thermometer, BP cuff, pulse oximeter, stethoscope, glucometer and strips, specimen containers, labels.\n"
    "Rules: identify items by their visible features (cap colour, label text, shape, size markings). If a label is "
    "readable, quote it verbatim in visual_description. Never guess a drug name or dose that is not clearly legible; "
    "say the label must be checked. Never give diagnostic conclusions, dosing decisions or treatment advice; you "
    "locate items and guide steps of a protocol the clinician chose. Treat needles, scalpels and broken glass as sharps "
    "and mention them in message when the target is a sharp. If two items look alike (same cap colour, similar "
    "syringes), set confidence below 0.5 and ask the user to specify."
)

SURGICAL_INSTRUMENT_CONTEXT = (
    "Surgical instrument tray (operating/procedure tray, instrument count workflows): use the precise standard "
    "instrument name, never a generic word like 'clamp' or 'scissors' alone — the count sheet matches on the exact "
    "name. Distinguish by these visible cues:\n"
    "- Scissors: mayo scissors = heavy, straight-to-slightly-curved blades (suture/heavy tissue). metzenbaum "
    "scissors = slim, curved blades (delicate dissection). suture scissors = small, straight, for cutting suture "
    "only.\n"
    "- Ring-handled clamps: artery clamp/haemostat/hemostat (kelly/crile/mosquito) = smooth curved jaws, no teeth "
    "at the tip. "
    "kocher clamp (ochsner) = heavy jaws with 1x2 interlocking teeth at the tip. allis forceps = multiple fine "
    "interlocking teeth on a fenestrated tip. babcock forceps = smooth rounded fenestrated (triangular) tip, no "
    "teeth, for atraumatic tissue holding.\n"
    "- needle holder (mayo-hegar/needle driver) = short, stubby, ridged jaws on ring handles, often a ratchet — "
    "shorter and blunter-jawed than an artery clamp.\n"
    "- Non-locking forceps: toothed forceps (adson) = fine interlocking teeth at the tip (skin). tissue/dressing "
    "forceps without teeth = smooth or serrated tip, no interlock.\n"
    "- Retractors: handheld small hooked-blade retractors (langenbeck, senn) vs framed self-retaining retractors "
    "with a ratchet (balfour, bookwalter) vs a single long curved flat blade (deaver retractor).\n"
    "- towel clip (backhaus clip) = sharp crossed points that pierce drapes — a sharp, handle with the same care as "
    "a needle or blade.\n"
    "- Suction: yankauer (bulb-tipped) vs poole (perforated sleeve) suction tip.\n"
    "- bovie tip / diathermy tip = removable electrosurgical pencil electrode.\n"
    "Sharps to count precisely: scalpel blade (state the number if legible: 10/11/15), suture needle (curved, "
    "usually still in its foil pack), introducer/hypodermic needle. Sponges to count precisely: raytec sponge "
    "(small square gauze with a blue radiopaque stripe) vs laparotomy pad (large folded pad with a blue tail and "
    "ring) — state which, since staff tell them apart by the stripe/tail, not just 'gauze'. When several "
    "instruments of the same broad family are visible, name the specific type of each rather than one grouped "
    "label; if genuinely indistinguishable at this image's resolution or angle, say so explicitly in message "
    "instead of guessing a specific name."
)

MEDICATION_CAUTION = "Check the label: drug name, strength, expiry and patient identity before use."
SHARPS_CAUTION = "Sharp item. Use the sharps container after use; never recap needles."

SPOKEN_STYLE = (
    "Speak like a calm scrub nurse: short sentences, one instruction at a time, no jargon unless the user used it."
)
