"""Code-level safety checks on user requests and AI answers (defence in depth)."""
from __future__ import annotations

import re
from dataclasses import dataclass

from bridge.healthcare.prompts import MEDICATION_CAUTION, SHARPS_CAUTION

SHARPS = ("needle", "scalpel", "blade", "lancet", "cannula", "catheter", "ampoule", "glass", "suture needle", "trocar")
MEDICATION_WORDS = ("tablet", "capsule", "vial", "ampoule", "insulin", "mg", "ml", "dose", "medication", "medicine",
                    "drug", "injection", "antibiotic", "syrup", "pill", "blister")
FORBIDDEN_PATTERNS = (
    r"\b(how much|what dose|how many (?:mg|ml|tablets|units))\b",
    r"\bshould i (?:give|inject|administer|prescribe|take)\b",
    r"\bwhat(?:'s| is) wrong with (?:the )?patient\b",
    r"\bdiagnos(?:e|is)\b",
    r"\bis it (?:safe|ok|okay) to (?:give|inject|administer)\b",
)


@dataclass
class SafetyVerdict:
    allowed: bool
    reason: str = ""
    cautions: list[str] | None = None

    @property
    def caution_text(self) -> str:
        return " ".join(self.cautions or [])


def check_request(text: str) -> SafetyVerdict:
    """Locating and procedural guidance are allowed; dosing/diagnosis decisions are declined."""
    t = text.lower()
    for pat in FORBIDDEN_PATTERNS:
        if re.search(pat, t):
            return SafetyVerdict(False, "I can locate items and guide steps, but dosing and diagnostic decisions must "
                                        "come from the clinician and the protocol.")
    cautions = []
    if any(w in t for w in SHARPS):
        cautions.append(SHARPS_CAUTION)
    if any(re.search(rf"\b{re.escape(w)}\b", t) for w in MEDICATION_WORDS):
        cautions.append(MEDICATION_CAUTION)
    return SafetyVerdict(True, cautions=cautions)


def cautions_for_target(label: str, description: str = "") -> list[str]:
    t = f"{label} {description}".lower()
    out = []
    if any(w in t for w in SHARPS):
        out.append(SHARPS_CAUTION)
    if any(re.search(rf"\b{re.escape(w)}\b", t) for w in MEDICATION_WORDS):
        out.append(MEDICATION_CAUTION)
    return out
