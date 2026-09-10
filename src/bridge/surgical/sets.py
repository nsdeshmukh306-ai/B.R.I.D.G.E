"""Instrument sets and countable-item taxonomy.

Sets are *data*: a hospital adds `sets/<name>.json` matching `InstrumentSet`
without touching code. The built-ins are generic, widely published tray
compositions used as a starting point — the circulating nurse's own count sheet
is always authoritative, and BRIDGE says so at the start of every count.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable, Literal, Optional

from pydantic import BaseModel, Field

log = logging.getLogger("bridge.surgical.sets")

# Category drives the safety rules: sharps and sponges are the retained-item risk.
Category = Literal["instrument", "sponge", "sharp", "miscellaneous"]

CATEGORY_ORDER: tuple[Category, ...] = ("sponge", "sharp", "instrument", "miscellaneous")

# Words that identify a countable class regardless of the exact instrument name.
SHARP_WORDS = ("blade", "scalpel", "needle", "suture", "trocar", "hypodermic", "knife", "pin", "wire",
               "cannula", "lancet", "staple")
SPONGE_WORDS = ("sponge", "raytec", "lap pad", "laparotomy pad", "gauze", "swab", "peanut", "kittner",
                "cottonoid", "patty")
MISC_WORDS = ("vessel loop", "umbilical tape", "bulldog", "clip", "cap", "sheath", "guidewire",
              "catheter tip", "drain", "bovie tip", "electrode")


def classify(name: str) -> Category:
    """Categorise a countable item by its name."""
    t = (name or "").lower()
    if any(w in t for w in SPONGE_WORDS):
        return "sponge"
    if any(w in t for w in SHARP_WORDS):
        return "sharp"
    if any(w in t for w in MISC_WORDS):
        return "miscellaneous"
    return "instrument"


class SetItem(BaseModel):
    name: str
    quantity: int = Field(default=1, ge=0)
    category: Optional[Category] = None       # inferred from the name when omitted
    aliases: list[str] = Field(default_factory=list)
    visual_hint: str = ""                     # helps the AI find it ("curved, ring handles")
    radiopaque: bool = False                  # sponges/needles that show on an X-ray

    @property
    def kind(self) -> Category:
        return self.category or classify(self.name)


class InstrumentSet(BaseModel):
    id: str
    name: str
    specialty: str = "general"
    items: list[SetItem] = Field(default_factory=list)
    note: str = ""

    @property
    def total(self) -> int:
        return sum(i.quantity for i in self.items)

    def by_category(self, category: Category) -> list[SetItem]:
        return [i for i in self.items if i.kind == category]


def _i(name: str, qty: int = 1, hint: str = "", aliases: Iterable[str] = (), radiopaque: bool = False) -> SetItem:
    return SetItem(name=name, quantity=qty, visual_hint=hint, aliases=list(aliases), radiopaque=radiopaque)


BUILTIN_SETS: list[InstrumentSet] = [
    InstrumentSet(
        id="minor", name="Minor / basic set", specialty="general",
        note="Generic minor procedure tray. Reconcile against your own count sheet.",
        items=[
            _i("raytec sponge", 10, "small square gauze, blue radiopaque stripe", ("raytec", "small sponge", "4x4"), True),
            _i("laparotomy pad", 5, "large folded pad with a blue tail and ring", ("lap pad", "lap sponge", "abdominal pad"), True),
            _i("scalpel blade", 2, "small steel blade, number 10 or 15", ("blade", "knife blade"), True),
            _i("suture needle", 4, "curved needle in a foil pack", ("needle", "suture"), True),
            _i("scalpel handle", 1, "flat metal handle", ("knife handle", "bard parker")),
            _i("toothed forceps", 2, "tweezer-like, serrated tip", ("adson forceps", "pickups")),
            _i("mayo scissors", 1, "heavy straight scissors", ("straight scissors",)),
            _i("metzenbaum scissors", 1, "slim curved scissors", ("metz", "tissue scissors")),
            _i("needle holder", 2, "ring handles with a short stubby jaw", ("needle driver", "mayo hegar")),
            _i("artery clamp", 6, "curved ring-handled clamp", ("haemostat", "hemostat", "mosquito", "kelly")),
            _i("allis forceps", 2, "ring handles with toothed tips", ("allis",)),
            _i("retractor", 2, "flat hooked blade", ("langenbeck", "senn")),
            _i("towel clip", 4, "sharp crossed points", ("backhaus clip",)),
            _i("kidney tray", 1, "curved steel dish", ("kidney dish", "emesis basin")),
        ],
    ),
    InstrumentSet(
        id="laparotomy", name="Laparotomy set", specialty="general surgery",
        note="Generic major abdominal tray. Reconcile against your own count sheet.",
        items=[
            _i("laparotomy pad", 10, "large folded pad with a blue tail", ("lap pad", "lap sponge"), True),
            _i("raytec sponge", 20, "small square gauze with a blue stripe", ("raytec", "4x4"), True),
            _i("scalpel blade", 3, "steel blade, number 10, 11 or 15", ("blade",), True),
            _i("suture needle", 8, "curved needle in a foil pack", ("needle",), True),
            _i("scalpel handle", 2),
            _i("artery clamp", 12, "curved ring-handled clamp", ("haemostat", "kelly", "crile")),
            _i("kocher clamp", 4, "heavy toothed clamp", ("ochsner",)),
            _i("babcock forceps", 4, "ring handles, rounded fenestrated tip", ("babcock",)),
            _i("allis forceps", 6),
            _i("needle holder", 4, "", ("needle driver",)),
            _i("metzenbaum scissors", 2),
            _i("mayo scissors", 2),
            _i("deaver retractor", 2, "long curved flat blade"),
            _i("self-retaining retractor", 1, "framed retractor with ratchet", ("balfour", "bookwalter")),
            _i("suction tip", 2, "rigid plastic or steel suction", ("yankauer", "poole")),
            _i("bovie tip", 1, "electrosurgical pencil tip", ("diathermy tip", "electrode")),
            _i("towel clip", 6),
            _i("umbilical tape", 2, "flat white cotton tape", ("tape",)),
            _i("vessel loop", 4, "thin coloured silicone loop", ("loop",)),
        ],
    ),
    InstrumentSet(
        id="suture_tray", name="Suturing tray", specialty="minor procedures",
        note="Emergency-department style laceration tray.",
        items=[
            _i("raytec sponge", 5, "", ("gauze",), True),
            _i("scalpel blade", 1, "", ("blade",), True),
            _i("suture needle", 2, "", ("suture",), True),
            _i("needle holder", 1),
            _i("toothed forceps", 1, "", ("adson",)),
            _i("suture scissors", 1, "", ("scissors",)),
            _i("artery clamp", 2, "", ("mosquito",)),
            _i("local anaesthetic syringe", 1, "syringe with a fine needle", ("syringe",)),
        ],
    ),
    InstrumentSet(
        id="central_line", name="Central line insertion tray", specialty="critical care",
        note="Generic Seldinger central venous catheter tray.",
        items=[
            _i("raytec sponge", 4, "", ("gauze",), True),
            _i("introducer needle", 1, "long thin needle", ("needle",), True),
            _i("guidewire", 1, "coiled flexible wire", ("wire",)),
            _i("dilator", 1, "tapered plastic tube"),
            _i("central venous catheter", 1, "multi-lumen catheter with coloured hubs", ("cvc", "catheter")),
            _i("scalpel blade", 1, "", ("blade",), True),
            _i("suture needle", 2, "", ("suture",), True),
            _i("syringe", 3, "5 or 10 mL syringe"),
            _i("chlorhexidine swab", 1, "", ("skin prep", "prep stick")),
            _i("transparent dressing", 1, "", ("tegaderm",)),
        ],
    ),
]


class SetLibrary:
    def __init__(self, extra_dir: Path | None = Path("sets")):
        self.sets: dict[str, InstrumentSet] = {s.id: s for s in BUILTIN_SETS}
        if extra_dir is not None and Path(extra_dir).is_dir():
            for f in sorted(Path(extra_dir).glob("*.json")):
                try:
                    s = InstrumentSet.model_validate(json.loads(f.read_text()))
                    self.sets[s.id] = s
                    log.info("Loaded instrument set %s from %s", s.id, f)
                except Exception:  # noqa: BLE001 - a bad file must not stop the app
                    log.exception("Invalid instrument set file %s", f)

    def get(self, key: str) -> Optional[InstrumentSet]:
        return self.sets.get(key)

    def find(self, spoken: str) -> Optional[InstrumentSet]:
        t = (spoken or "").lower()
        best, best_score = None, 0
        for s in self.sets.values():
            words = set(s.name.lower().replace("/", " ").split()) | set(s.id.replace("_", " ").split())
            score = sum(1 for w in words if len(w) > 2 and w in t)
            if score > best_score:
                best, best_score = s, score
        return best if best_score >= 1 else None

    def names(self) -> list[tuple[str, str]]:
        return [(s.id, s.name) for s in self.sets.values()]
