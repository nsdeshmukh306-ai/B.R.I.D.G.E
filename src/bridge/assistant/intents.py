"""Local intent grammar — the fast path.

Sending "project the scalpel" to a vision model costs 600–2000 ms and a network
round-trip that a theatre may not have. Almost every sentence spoken during a
case belongs to a small, closed vocabulary: locate a thing, record a number,
move the case forward, ask the count. Those are matched here with regular
expressions in microseconds, and only genuinely open-ended requests fall
through to Gemini.

The grammar is deliberately conservative. A pattern that is not clearly one of
these intents returns `unknown`, and `unknown` means "ask the model" — never
"guess". Mis-hearing a number as a command is the failure mode that matters in
a count, so numeric intents require an explicit count context.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal, Optional

from bridge.surgical.counts import parse_spoken_count

Kind = Literal[
    "locate", "clear", "scene_describe", "whats_missing", "where_did_it_go",
    "count_begin", "count_record", "count_add", "count_accept", "count_status", "count_phase", "count_line_add",
    "case_start", "case_confirm", "case_skip", "case_next", "case_back", "case_repeat", "case_status", "case_close",
    "procedure", "mute", "unmute", "repeat", "help", "stop_talking", "unknown",
]


@dataclass
class Intent:
    kind: Kind
    text: str = ""                       # the (already reference-resolved) utterance
    target: str = ""                     # noun phrase for locate / count lines
    quantity: Optional[int] = None
    phase: str = ""                      # for count_phase / case_next
    slots: dict = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def known(self) -> bool:
        return self.kind != "unknown"


# --- vocabulary -----------------------------------------------------------------------
LOCATE_VERBS = (r"project(?:\s+onto)?", r"show(?:\s+me)?", r"point(?:\s+(?:to|at))?", r"highlight",
                r"find", r"locate", r"mark", r"light\s+up", r"where(?:'s|\s+is|\s+are)?",
                r"give\s+me", r"hand\s+me", r"pass\s+me", r"i\s+need", r"i\s+want", r"label")
_LOCATE = re.compile(r"^(?P<verb>" + "|".join(LOCATE_VERBS) + r")\s+(?:me\s+)?"
                     r"(?:the\s+|a\s+|an\s+|my\s+|some\s+)?(?P<target>.+?)"
                     r"(?:\s+please)?\s*[?.!]*$", re.I)

# Which graphic the verb asks for. A reticle answers "where is it"; an arrow
# answers "hand me that", which is a different question on a crowded tray.
_POINT_VERBS = ("point", "hand", "pass", "give")
_LABEL_VERBS = ("label",)

_CLEAR = re.compile(r"^(?:clear|clear\s+(?:the\s+)?(?:projection|screen|display)|nothing|cancel\s+that|"
                    r"turn\s+(?:it\s+)?off)\s*[.!]?$", re.I)
_SCENE = re.compile(r"^(?:what\s+(?:do\s+you\s+see|can\s+you\s+see|is\s+(?:on\s+)?(?:the\s+)?(?:tray|table|field))|"
                    r"describe\s+(?:the\s+)?(?:scene|tray|table)|read\s+(?:the\s+)?tray)\s*[?.!]*$", re.I)
_MISSING = re.compile(r"^(?:what(?:'s| is)?\s+missing|what\s+(?:is|are)\s+missing|anything\s+missing|"
                      r"show\s+(?:me\s+)?(?:what(?:'s| is)?\s+)?missing|what\s+am\s+i\s+missing)\s*[?.!]*$", re.I)
_WHERE_GONE = re.compile(r"^(?:where\s+did\s+(?:the\s+)?(?P<target>.+?)\s+go|"
                         r"last\s+seen\s+(?P<t2>.+?))\s*[?.!]*$", re.I)

_COUNT_BEGIN = re.compile(r"^(?:start|begin|do|run)\s+(?:the\s+)?(?:initial\s+|first\s+|opening\s+)?count"
                          r"(?:\s+in)?(?:\s+for\s+(?:the\s+|a\s+)?(?P<set>.+?))?\s*[.!]?$", re.I)
_COUNT_IN_ALT = re.compile(r"^(?:count\s+in|counts?\s+please|let(?:'s| us)\s+count)\s*[.!]?$", re.I)
_COUNT_PHASE = re.compile(r"^(?:(?P<phase>closing|final|first\s+closing|skin)\s+count|"
                          r"(?:start|begin|do)\s+(?:the\s+)?(?P<phase2>closing|final)\s+count|count\s+out)\s*[.!]?$", re.I)
_COUNT_ACCEPT = re.compile(r"^(?:count\s+as\s+per\s+(?:the\s+)?sheet|(?:all\s+)?(?:present\s+and\s+)?correct|"
                           r"tray\s+(?:is\s+)?complete|as\s+(?:per\s+)?(?:the\s+)?(?:sheet|list)|"
                           r"everything(?:'s| is)\s+here)\s*[.!]?$", re.I)
_COUNT_STATUS = re.compile(r"^(?:what(?:'s| is)?\s+(?:the\s+)?count|read\s+(?:me\s+)?(?:the\s+|back\s+the\s+)?count|"
                           r"count\s+status|where\s+are\s+we\s+(?:on|with)\s+(?:the\s+)?counts?|"
                           r"how\s+many\s+(?P<item>.+?))\s*[?.!]*$", re.I)
_COUNT_ADD = re.compile(r"^(?:add(?:ing)?|opening|open|extra|another)\s+(?P<rest>.+?)"
                        r"(?:\s+to\s+the\s+(?:field|count|sheet))?\s*[.!]?$", re.I)
_COUNT_LINE_ADD = re.compile(r"^(?:add|put)\s+(?P<name>.+?)\s+(?:to|on)\s+the\s+(?:count\s+)?(?:sheet|list)\s*[.!]?$", re.I)

_CASE_START = re.compile(r"^(?:start|begin|open)\s+(?:a\s+|the\s+)?(?:new\s+)?case(?:\s+(?:for|of)\s+(?P<name>.+?))?\s*[.!]?$", re.I)
_CASE_CONFIRM = re.compile(r"^(?:yes|yep|yeah|confirmed?|check(?:ed)?|done|correct|affirmative|"
                           r"that(?:'s| is)\s+(?:right|correct)|ok(?:ay)?|got\s+it|complete)\s*[.!]?$", re.I)
_CASE_SKIP = re.compile(r"^(?:skip(?:\s+(?:it|that))?|not\s+applicable|n\s*/?\s*a|no(?:pe)?|"
                        r"pass|next\s+item|doesn(?:'|)t\s+apply)\s*[.!]?$", re.I)
_CASE_NEXT = re.compile(r"^(?:next(?:\s+(?:phase|step|section))?|move\s+on|carry\s+on|continue|proceed|"
                        r"(?:start|begin|do)\s+(?:the\s+)?(?P<named>sign\s+in|time\s*out|sign\s+out|briefing|debrief))\s*[.!]?$", re.I)
_CASE_BACK = re.compile(r"^(?:back|go\s+back|previous|undo\s+that)\s*[.!]?$", re.I)
_CASE_REPEAT = re.compile(r"^(?:repeat|again|say\s+(?:that\s+)?again|what\s+was\s+that|pardon)\s*[?.!]*$", re.I)
_CASE_STATUS = re.compile(r"^(?:where\s+are\s+we|what(?:'s| is)\s+(?:the\s+)?(?:status|phase)|"
                          r"case\s+status|what\s+are\s+we\s+doing)\s*[?.!]*$", re.I)
_CASE_CLOSE = re.compile(r"^(?:close\s+(?:the\s+)?case|end\s+(?:the\s+)?case|finish\s+(?:the\s+)?case|"
                         r"case\s+(?:complete|finished)|sign\s+off)\s*[.!]?$", re.I)

_MUTE = re.compile(r"^(?:quiet|be\s+quiet|mute|silence|stop\s+alerts?|no\s+more\s+alerts?|"
                   r"stop\s+(?:talking|interrupting))\s*[.!]?$", re.I)
_UNMUTE = re.compile(r"^(?:unmute|resume\s+alerts?|you\s+can\s+talk|alerts?\s+on|speak\s+up)\s*[.!]?$", re.I)
_STOP_TALKING = re.compile(r"^(?:stop|wait|hold\s+on|shush|enough)\s*[.!]?$", re.I)
_HELP = re.compile(r"^(?:help|what\s+can\s+you\s+do|commands?|options)\s*[?.!]*$", re.I)

_COUNT_CONTEXT_PHASES = ("count_in", "count_closing", "count_final")

# Things that look like a locate target but are really chatter — refuse to project them.
_NOT_TARGETS = {"", "it", "that", "this", "them", "those", "one", "me", "us", "there", "here", "yourself"}


def parse(text: str, *, case_phase: str = "idle", counting: bool = False,
          in_checklist: bool = False) -> Intent:
    """Classify one utterance. `counting` widens the grammar to bare numbers."""
    raw = (text or "").strip()
    if not raw:
        return Intent("unknown", raw)
    t = re.sub(r"\s+", " ", raw).strip()

    # order matters: the most specific and most safety-relevant first
    if _MUTE.match(t):
        return Intent("mute", t)
    if _UNMUTE.match(t):
        return Intent("unmute", t)
    if _STOP_TALKING.match(t) and not (in_checklist or counting):
        return Intent("stop_talking", t)
    if _HELP.match(t):
        return Intent("help", t)
    if _CLEAR.match(t):
        return Intent("clear", t)
    if _CASE_CLOSE.match(t):
        return Intent("case_close", t)
    if _CASE_STATUS.match(t):
        return Intent("case_status", t)
    if _CASE_REPEAT.match(t):
        return Intent("case_repeat", t)

    m = _CASE_START.match(t)
    if m:
        return Intent("case_start", t, target=(m.group("name") or "").strip())
    m = _COUNT_BEGIN.match(t)
    if m:
        return Intent("count_begin", t, slots={"set": (m.group("set") or "").strip()})
    if _COUNT_IN_ALT.match(t):
        return Intent("count_begin", t)
    m = _COUNT_PHASE.match(t)
    if m:
        p = (m.group("phase") or m.group("phase2") or "final").lower()
        phase = "closing" if "closing" in p or "first" in p else "final"
        return Intent("count_phase", t, phase=phase)
    if _COUNT_ACCEPT.match(t) and case_phase in _COUNT_CONTEXT_PHASES:
        return Intent("count_accept", t)
    m = _COUNT_LINE_ADD.match(t)
    if m:
        return Intent("count_line_add", t, target=m.group("name").strip())
    m = _COUNT_STATUS.match(t)
    if m:
        return Intent("count_status", t, target=(m.group("item") or "").strip())
    if _MISSING.match(t):
        return Intent("whats_missing", t)
    if _SCENE.match(t):
        return Intent("scene_describe", t)
    m = _WHERE_GONE.match(t)
    if m:
        return Intent("where_did_it_go", t, target=(m.group("target") or m.group("t2") or "").strip())

    m = _COUNT_ADD.match(t)
    if m and case_phase not in ("idle",):
        parsed = parse_spoken_count(m.group("rest"))
        if parsed:
            name, qty = parsed
            return Intent("count_add", t, target=name, quantity=max(qty, 1))
        name = m.group("rest").strip()
        if name and name not in _NOT_TARGETS:
            return Intent("count_add", t, target=name, quantity=1)

    if _CASE_BACK.match(t):
        return Intent("case_back", t)
    if in_checklist and _CASE_CONFIRM.match(t):
        return Intent("case_confirm", t)
    if in_checklist and _CASE_SKIP.match(t):
        return Intent("case_skip", t)
    m = _CASE_NEXT.match(t)
    if m:
        named = (m.group("named") or "").lower().replace(" ", "")
        phase = {"signin": "sign_in", "timeout": "timeout", "signout": "sign_out",
                 "briefing": "briefing", "debrief": "debrief"}.get(named, "")
        return Intent("case_next", t, phase=phase)

    # A bare "four artery clamps" is only a count when we are actually counting.
    if counting or case_phase in _COUNT_CONTEXT_PHASES:
        parsed = parse_spoken_count(t)
        if parsed and parsed[0] not in _NOT_TARGETS:
            return Intent("count_record", t, target=parsed[0], quantity=parsed[1])

    m = _LOCATE.match(t)
    if m:
        target = _clean_target(m.group("target"))
        if target and target not in _NOT_TARGETS:
            verb = m.group("verb").split()[0].lower()
            action = ("point" if verb in _POINT_VERBS else
                      "label" if verb in _LABEL_VERBS else "highlight")
            return Intent("locate", t, target=target, slots={"action": action})

    return Intent("unknown", t, confidence=0.0)


def _clean_target(s: str) -> str:
    s = (s or "").strip().rstrip("?.!,")
    s = re.sub(r"\s+(?:on|in)\s+the\s+(?:tray|table|field|back\s+table)$", "", s, flags=re.I)
    s = re.sub(r"^(?:is|are)\s+", "", s, flags=re.I)
    # "the other screw" is still a request for a screw; the "another instance"
    # part is carried separately so the subject stays stable across follow-ups.
    s = re.sub(r"^(?:other|another|next|different)\s+", "", s, flags=re.I)
    return s.strip()


HELP_TEXT = (
    "Say: project the scalpel, or where is the needle holder. "
    "Start a case, then confirm or skip each checklist item. "
    "Start the count, then call each item and its number. "
    "Say adding two lap pads for anything opened onto the field. "
    "Closing count, final count, what's missing, what's the count, close the case. "
    "Say quiet to stop me interrupting."
)
