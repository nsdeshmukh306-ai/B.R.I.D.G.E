"""Conversation memory: what makes BRIDGE answer "and the other one" correctly.

A surgeon does not restate the noun every sentence. "Where's the Metzenbaum?" …
"the other one" … "put it back". Each of those is meaningless without the last
few turns, so BRIDGE keeps a short dialogue memory with two jobs:

* **Reference resolution** — rewrite a follow-up into a self-contained request
  before anything else sees it ("the other one" -> "the other metzenbaum
  scissors"). Deterministic and local: no AI round-trip to understand a pronoun.
* **Context for the model** — when a request *does* reach Gemini, the last few
  turns and the current case phase go in as context, so the model resolves the
  same way BRIDGE would.

Deliberately small: a ring buffer, a last-subject slot, and a set of rewrite
rules. Long conversational memory in an operating theatre is a liability, not a
feature — the case record is where things are remembered.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from bridge.perception.scene_graph import normalise_label

# Phrases that mean "the thing we were just talking about".
_ANAPHORA = (
    r"\bit\b", r"\bthat one\b", r"\bthat\b", r"\bthis one\b", r"\bthe same one\b",
    r"\bthe other one\b", r"\banother one\b", r"\bthe next one\b", r"\bthem\b", r"\bthose\b",
)
_OTHER = re.compile(r"\b(the other|another|the next|a different)\s+one\b", re.I)
# Bare "next" is a control word (next checklist item, next procedure step), never
# an anaphora — so it is deliberately absent here.
_BARE_OTHER = re.compile(r"^\s*(?:and\s+)?(?:the\s+other|another|the\s+next\s+one|other\s+one)"
                         r"(?:\s+one)?\s*[?.!]?\s*$", re.I)
_BARE_PRONOUN = re.compile(r"^\s*(?:and\s+)?(?:it|that|this|them|those)\s*[?.!]?\s*$", re.I)

# Follow-ups that are a verb plus a pronoun: "point to it", "where is it".
_VERB_PRONOUN = re.compile(
    r"^\s*(?P<verb>where(?:'s| is| are)?|show|point(?: to)?|highlight|project|find|mark|label|"
    r"put|move|clear)\s+(?:me\s+)?"
    # longest alternatives first, so "that one" is consumed whole rather than
    # leaving a stray "one" in the rewritten sentence
    r"(?:the other one|another one|the same one|that one|this one|it|that|this|them|those)\b"
    r"(?P<rest>.*)$", re.I)


@dataclass
class Turn:
    text: str
    reply: str = ""
    subject: str = ""            # normalised noun this turn was about
    intent: str = ""
    entity_ids: list[str] = field(default_factory=list)
    ts: float = field(default_factory=time.time)


class Conversation:
    """Short dialogue memory. Thread-safe (voice thread writes, UI reads)."""

    def __init__(self, max_turns: int = 12, subject_ttl_s: float = 180.0):
        self.max_turns = max_turns
        self.subject_ttl_s = subject_ttl_s
        self.turns: list[Turn] = []
        self.subject: str = ""
        self.subject_ts: float = 0.0
        self.mentioned: list[str] = []       # entity ids already shown for the current subject
        self.case_context: str = ""          # e.g. "laparotomy, closing count"
        self._lock = threading.RLock()

    # -- memory -------------------------------------------------------------------------
    def remember(self, text: str, reply: str = "", subject: str = "", intent: str = "",
                 entity_ids: list[str] | None = None) -> Turn:
        with self._lock:
            t = Turn(text=text, reply=reply, subject=normalise_label(subject) if subject else "",
                     intent=intent, entity_ids=list(entity_ids or []))
            self.turns.append(t)
            self.turns = self.turns[-self.max_turns:]
            if t.subject:
                if t.subject != self.subject:
                    self.mentioned = []
                self.subject, self.subject_ts = t.subject, t.ts
            if t.entity_ids:
                for eid in t.entity_ids:
                    if eid not in self.mentioned:
                        self.mentioned.append(eid)
            return t

    def clear(self) -> None:
        with self._lock:
            self.turns.clear()
            self.subject, self.subject_ts = "", 0.0
            self.mentioned = []

    @property
    def active_subject(self) -> str:
        with self._lock:
            if self.subject and time.time() - self.subject_ts <= self.subject_ttl_s:
                return self.subject
            return ""

    def last_reply(self) -> str:
        return self.turns[-1].reply if self.turns else ""

    # -- reference resolution --------------------------------------------------------------
    def resolve(self, text: str) -> tuple[str, bool]:
        """Rewrite a follow-up into a standalone request.

        Returns (resolved_text, wants_another). `wants_another` asks the caller to
        skip entities already shown for this subject — that is what makes "the
        other one" land on a different clamp.
        """
        raw = (text or "").strip()
        if not raw:
            return raw, False
        subject = self.active_subject
        wants_another = bool(_OTHER.search(raw) or _BARE_OTHER.match(raw))
        if not subject:
            return raw, wants_another
        if _BARE_OTHER.match(raw):
            return f"where is the other {subject}", True
        if _BARE_PRONOUN.match(raw):
            return f"where is the {subject}", False
        m = _VERB_PRONOUN.match(raw)
        if m:
            verb = m.group("verb").lower()
            rest = (m.group("rest") or "").strip()
            head = {"where": "where is", "where's": "where is", "where is": "where is",
                    "where are": "where is"}.get(verb, verb)
            article = "the other" if wants_another else "the"
            return f"{head} {article} {subject} {rest}".strip(), wants_another
        # a pronoun embedded elsewhere in a longer sentence
        out = raw
        for pat in _ANAPHORA:
            if re.search(pat, out, re.I):
                out = re.sub(pat, f"the {subject}", out, count=1, flags=re.I)
                break
        return out, wants_another

    # -- prompt context ----------------------------------------------------------------------
    def context_block(self, max_turns: int = 4) -> str:
        """Compact recent dialogue for the AI prompt. Empty when there is nothing useful."""
        with self._lock:
            recent = [t for t in self.turns if t.text][-max_turns:]
        parts = []
        if self.case_context:
            parts.append(f"Current case context: {self.case_context}.")
        if recent:
            lines = [f"  user: {t.text}" + (f"\n  bridge: {t.reply}" if t.reply else "") for t in recent]
            parts.append("Recent conversation (most recent last):\n" + "\n".join(lines))
        if self.active_subject:
            parts.append(f"If the request uses a pronoun, it most likely refers to: {self.active_subject}.")
        return "\n".join(parts)


def extract_subject(text: str) -> str:
    """Best-effort noun phrase from a spoken command, for the subject slot.

    'where is the curved artery clamp' -> 'curved artery clamp'. Purely lexical;
    the AI's `target` overrides it whenever one comes back.
    """
    t = (text or "").lower().strip().rstrip("?.!")
    t = re.sub(r"^(?:bridge[,\s]+)?", "", t)
    t = re.sub(r"^(?:where(?:'s| is| are)?|show me|show|point to|point at|point|highlight|project|"
               r"find|locate|mark|label|give me|hand me|pass me|i need|i want)\s+", "", t)
    t = re.sub(r"^(?:the|a|an|my|some)\s+", "", t)
    t = re.sub(r"\s+(?:please|now|again|for me)$", "", t)
    t = re.sub(r"^(?:other|next|another)\s+", "", t)
    return normalise_label(t)
