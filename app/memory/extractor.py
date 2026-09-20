"""Memory extraction: decide what - if anything - is worth keeping.

The rule the project is built around: DO NOT store every message. A turn earns a long-term
record only if it states something durable about the user or their world. Questions, small talk,
one-off requests and anything derived from the agent's own reasoning are rejected.

Two implementations behind one interface:
  * LLM extractor  - strict JSON schema, used when an LLM is configured;
  * heuristic extractor - pattern based, used when LLM_PROVIDER=none.
Both return the same candidate shape, and both are filtered by the same rules afterwards.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from app.llm.base import LLMClient, LLMError
from app.utils.dates import iso, utcnow
from app.utils.text import collapse_whitespace, truncate

MAX_CANDIDATES = 3
MAX_CONTENT_CHARS = 240

EXTRACTION_SYSTEM = """You maintain the long-term memory of a project-intelligence assistant.

You receive one user turn (and, for context, the assistant's answer summary). Decide whether the
USER stated something durable that will still be useful in a future session.

STORE: stated priorities, ownership and role, recurring focus areas, standing preferences about
how answers should be produced, long-lived goals, and durable facts about the user's projects.

DO NOT STORE: questions, one-off task instructions, anything the assistant inferred or retrieved,
transient status ("I'm in a meeting"), pleasantries, or restatements of data that already lives in
the source system.

Return JSON only:
{"candidates": [{"type": "preference|context|fact|goal",
                 "content": "third-person statement under 240 characters",
                 "subject": "short topic key, e.g. 'priority' or 'project:ATLAS'",
                 "importance": 0.0-1.0,
                 "confidence": 0.0-1.0,
                 "ttl_days": null or integer,
                 "reason": "why this is durable"}]}
Return {"candidates": []} when nothing qualifies. Never return more than 3 candidates."""

# Heuristic patterns. Each captures a durable statement the *user* made about themselves.
PATTERNS: list[tuple[re.Pattern, str, str, float]] = [
    (re.compile(r"\b(?:my|our)\s+(?:top\s+|highest\s+|main\s+|number one\s+)?priority\s+(?:right now\s+|currently\s+|this quarter\s+)?is\s+(?P<value>[^.;\n]{2,120})", re.I),
     "preference", "priority", 0.85),
    (re.compile(r"\b(?P<value>[A-Za-z0-9 _\-]{2,60})\s+is\s+(?:my|our)\s+(?:top|highest|main|number one)\s+priority\b[^.\n]*", re.I),
     "preference", "priority", 0.85),
    (re.compile(r"\b(?P<value>[A-Za-z0-9 _\-]{2,60}?)\s+(?:has become|became|is now|are now)\s+(?:my|our)\s+(?:top|highest|main|number one|new)\s+priority\b[^.\n]*", re.I),
     "preference", "priority", 0.85),
    (re.compile(r"\b(?P<value>project\s+[A-Za-z0-9_\-]{2,40})\s+is\s+(?:the\s+)?most important\b[^.\n]*", re.I),
     "preference", "priority", 0.8),
    (re.compile(r"\bi(?:'m| am)\s+(?:the\s+)?(?P<value>(?:engineering|product|program|project|tech|delivery)?\s*(?:manager|lead|owner|director|pm|em|architect|analyst)[^.;\n]{0,80})", re.I),
     "context", "role", 0.75),
    (re.compile(r"\bi\s+(?:own|manage|run|lead)\s+(?P<value>[^.;\n]{2,120})", re.I),
     "context", "ownership", 0.7),
    (re.compile(r"\bi\s+(?:care|worry)\s+(?:most\s+)?about\s+(?P<value>[^.;\n]{2,120})", re.I),
     "preference", "focus", 0.65),
    (re.compile(r"\b(?:always|from now on|going forward)\s+(?P<value>[^.;\n]{4,140})", re.I),
     "preference", "answer_style", 0.6),
    (re.compile(r"\b(?:remember|note)\s+that\s+(?P<value>[^.\n]{4,180})", re.I),
     "fact", "user_note", 0.7),
    (re.compile(r"\b(?:my|our)\s+goal\s+(?:this quarter\s+|for [^\s]+\s+)?is\s+(?P<value>[^.;\n]{4,160})", re.I),
     "goal", "goal", 0.75),
    (re.compile(r"\bi\s+(?:prefer|want|need)\s+(?P<value>(?:answers|responses|summaries)[^.;\n]{2,140})", re.I),
     "preference", "answer_style", 0.6),
]

REJECT_IF = (
    re.compile(r"^\s*(what|who|when|where|why|how|which|can you|could you|show|list|give me|tell me|summari[sz]e)\b", re.I),
)

# Openers that mark a turn as a statement about the user rather than a request for data.
STATEMENT_OPENERS = re.compile(
    r"^\s*(actually,?\s+)?(remember|note that|for the record|fyi|just so you know|i |i'm |i am |my |our |we )",
    re.I,
)


def looks_like_user_statement(text: str) -> bool:
    """True when the turn asserts something durable instead of asking for tracker data."""
    text = (text or "").strip()
    if not text or "?" in text:
        return False
    if any(pattern.match(text) for pattern in REJECT_IF):
        return False
    if any(pattern.search(text) for pattern, *_ in PATTERNS):
        return True
    return bool(STATEMENT_OPENERS.match(text)) and len(text.split()) <= 40

TRANSIENT = re.compile(r"\b(today only|right this second|for now only|just for this (question|message)|ignore that)\b", re.I)


@dataclass
class MemoryCandidate:
    type: str
    content: str
    subject: str = ""
    importance: float = 0.5
    confidence: float = 0.6
    ttl_days: Optional[int] = None
    reason: str = ""
    origin: str = "heuristic"

    def expires_at(self) -> Optional[str]:
        if not self.ttl_days:
            return None
        from datetime import timedelta
        return iso(utcnow() + timedelta(days=int(self.ttl_days)))


@dataclass
class ExtractionResult:
    candidates: list[MemoryCandidate] = field(default_factory=list)
    rejected: list[dict[str, str]] = field(default_factory=list)
    origin: str = "heuristic"

    def as_dict(self) -> dict[str, Any]:
        return {
            "origin": self.origin,
            "candidates": [c.__dict__ for c in self.candidates],
            "rejected": self.rejected,
        }


class MemoryExtractor:
    def __init__(self, llm: Optional[LLMClient] = None, min_importance: float = 0.35):
        self.llm = llm
        self.min_importance = min_importance

    async def extract(self, user_message: str, answer_summary: str = "") -> ExtractionResult:
        user_message = collapse_whitespace(user_message)
        if not user_message:
            return ExtractionResult(rejected=[{"content": "", "reason": "empty turn"}])

        if self.llm is not None:
            try:
                return self._filter(await self._extract_llm(user_message, answer_summary))
            except LLMError:
                pass  # fall through to heuristics rather than losing the turn entirely
        return self._filter(self._extract_heuristic(user_message))

    # --- implementations ---------------------------------------------------
    async def _extract_llm(self, user_message: str, answer_summary: str) -> ExtractionResult:
        payload = f"USER TURN:\n{user_message}"
        if answer_summary:
            payload += f"\n\nASSISTANT ANSWER SUMMARY (context only, never store this):\n{truncate(answer_summary, 600)}"
        data = await self.llm.complete_json(EXTRACTION_SYSTEM, payload, max_tokens=500)
        candidates = []
        for raw in (data.get("candidates") or [])[:MAX_CANDIDATES]:
            if not isinstance(raw, dict) or not raw.get("content"):
                continue
            candidates.append(MemoryCandidate(
                type=str(raw.get("type", "context")),
                content=truncate(collapse_whitespace(str(raw["content"])), MAX_CONTENT_CHARS, ""),
                subject=str(raw.get("subject", ""))[:60],
                importance=float(raw.get("importance", 0.5) or 0.5),
                confidence=float(raw.get("confidence", 0.6) or 0.6),
                ttl_days=raw.get("ttl_days") if isinstance(raw.get("ttl_days"), int) else None,
                reason=str(raw.get("reason", ""))[:200],
                origin="llm",
            ))
        return ExtractionResult(candidates=candidates, origin="llm")

    def _extract_heuristic(self, user_message: str) -> ExtractionResult:
        result = ExtractionResult(origin="heuristic")
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", user_message):
            sentence = sentence.strip()
            if len(sentence) < 8:
                continue
            if any(p.match(sentence) for p in REJECT_IF):
                result.rejected.append({"content": truncate(sentence, 120), "reason": "question, not a statement"})
                continue
            if TRANSIENT.search(sentence):
                result.rejected.append({"content": truncate(sentence, 120), "reason": "explicitly transient"})
                continue
            for pattern, mem_type, subject, importance in PATTERNS:
                match = pattern.search(sentence)
                if not match:
                    continue
                value = collapse_whitespace(match.groupdict().get("value", "")).strip(" ,.;")
                if not value:
                    continue
                content = self._render(mem_type, subject, value, sentence)
                result.candidates.append(MemoryCandidate(
                    type=mem_type, content=content, subject=subject, importance=importance,
                    confidence=0.7, reason=f"matched '{subject}' statement pattern",
                    origin="heuristic",
                ))
                break
            else:
                result.rejected.append({"content": truncate(sentence, 120),
                                        "reason": "no durable user statement detected"})
        result.candidates = result.candidates[:MAX_CANDIDATES]
        return result

    @staticmethod
    def _render(mem_type: str, subject: str, value: str, sentence: str) -> str:
        value = truncate(value, MAX_CONTENT_CHARS - 60, "")
        if subject == "priority":
            return f"User's stated top priority is {value}."
        if subject == "role":
            return f"User's role: {value}."
        if subject == "ownership":
            return f"User owns or leads {value}."
        if subject == "focus":
            return f"User cares most about {value}."
        if subject == "goal":
            return f"User's stated goal: {value}."
        if subject == "answer_style":
            return f"User preference for answers: {value}."
        return truncate(collapse_whitespace(sentence), MAX_CONTENT_CHARS, "")

    # --- filtering ---------------------------------------------------------
    def _filter(self, result: ExtractionResult) -> ExtractionResult:
        kept: list[MemoryCandidate] = []
        for candidate in result.candidates:
            if not candidate.content or len(candidate.content) < 8:
                result.rejected.append({"content": candidate.content, "reason": "too short"})
                continue
            if candidate.importance < self.min_importance:
                result.rejected.append({"content": candidate.content,
                                        "reason": f"importance {candidate.importance:.2f} below threshold"})
                continue
            if re.search(r"\b(thought|reasoning|chain of thought|step \d)\b", candidate.content, re.I):
                result.rejected.append({"content": candidate.content, "reason": "looks like reasoning, not a fact"})
                continue
            kept.append(candidate)
        result.candidates = kept[:MAX_CANDIDATES]
        return result
