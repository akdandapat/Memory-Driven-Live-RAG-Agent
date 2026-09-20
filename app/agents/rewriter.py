"""Query rewriting: turn a conversational turn into a standalone question.

The transcript names this as a distinct stage, separate from planning, and it earns its place:
follow-up turns are elliptical. "And what about Apollo?" or "why did that slip?" carry no
project, no time window and no subject on their own, so a planner that sees only the raw turn
plans the wrong retrieval.

Hard rule: the rewriter may only *resolve references* using the session context and remembered
entities. It may never add a constraint the user did not express. If it cannot resolve
something confidently it leaves the question alone and says so - a wrong rewrite is worse than
no rewrite, because everything downstream inherits it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.llm.base import LLMClient, LLMError
from app.memory.extractor import looks_like_user_statement
from app.memory.long_term import MemoryRecord
from app.utils.dates import resolve_time_window
from app.utils.text import truncate

ISSUE_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,29}-\d{1,9})\b")

# Turns that cannot stand alone: they point at something said earlier.
FOLLOW_UP_PREFIX = re.compile(
    r"^\s*(and\b|but\b|also\b|what about|how about|and what|and how|what else|any others|"
    r"anything else|same for|do the same|again for|now for)",
    re.I,
)
DANGLING_REFERENCES = re.compile(
    r"^\s*(and|but|so|also|then|ok|okay|right)\b|"
    r"\b(it|that|this|these|those|there|they|them|its|their|the same|the project|that one)\b",
    re.I,
)
SELF_CONTAINED_HINT = re.compile(r"\b(project|epic|sprint|board)\b", re.I)

# Turns that are about the assistant or the user, not about anything said earlier. Rewriting
# these is actively harmful: it injects a project into a question that has no subject at all.
# Deliberately narrow: greetings and questions *about the assistant*. It must not swallow
# ordinary tracker questions that happen to start with "what is" - "what is blocked there?"
# is a follow-up that genuinely needs resolving.
META_TURN = re.compile(
    r"^\s*(hi|hello|hey|thanks|thank you|what can you do|what can you help|who are you|"
    r"what are you|how do you work|are you an? )\b",
    re.I,
)

REWRITE_SYSTEM = """You rewrite one conversational turn into a standalone question.

You are given the recent session transcript, the entities discussed in it, and the user's turn.

Rules:
- Resolve pronouns, ellipsis and implicit subjects using the transcript ONLY.
- Never add a filter, date range, project or constraint the user did not express or imply.
- Never answer the question, and never add facts.
- If the turn is already standalone, or you cannot resolve a reference confidently, return it
  unchanged and set "changed" to false.
- Keep the user's intent and scope exactly. Do not broaden or narrow it.

Return JSON only:
{"rewritten": "...", "changed": true|false, "resolved": ["it -> Project Atlas"], "reason": "..."}"""


@dataclass
class RewriteResult:
    original: str
    rewritten: str
    changed: bool = False
    resolved: list[str] = field(default_factory=list)
    reason: str = ""
    method: str = "none"          # none | heuristic | llm
    carried_entities: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "rewritten": self.rewritten,
            "changed": self.changed,
            "resolved": self.resolved,
            "reason": self.reason,
            "method": self.method,
            "carried_entities": self.carried_entities,
        }


class QueryRewriter:
    def __init__(self, llm: Optional[LLMClient] = None):
        self.llm = llm

    # --- entrypoint --------------------------------------------------------
    async def rewrite(
        self,
        question: str,
        *,
        session_context: str = "",
        known_projects: Sequence[dict[str, Any]] = (),
        memories: Sequence[MemoryRecord] = (),
    ) -> RewriteResult:
        question = question.strip()
        entities = self._context_entities(question, session_context, known_projects, memories)

        if not self._needs_rewrite(question, known_projects):
            return RewriteResult(original=question, rewritten=question, changed=False,
                                 reason="already standalone", method="none",
                                 carried_entities=entities)
        if not session_context.strip() and not entities.get("project_keys"):
            return RewriteResult(original=question, rewritten=question, changed=False,
                                 reason="nothing earlier in the session to resolve against",
                                 method="none", carried_entities=entities)

        if self.llm is not None:
            try:
                result = await self._rewrite_llm(question, session_context, entities)
                if result is not None:
                    return result
            except (LLMError, ValueError, TypeError, KeyError):
                pass
        return self._rewrite_heuristic(question, entities)

    # --- decision ----------------------------------------------------------
    @staticmethod
    def _needs_rewrite(question: str, known_projects: Sequence[dict[str, Any]]) -> bool:
        low = f" {question.lower()} "
        # Never rewrite a meta turn or a statement about the user - there is no reference to
        # resolve, so any "resolution" would be an invention.
        if META_TURN.match(question) or looks_like_user_statement(question):
            return False
        if FOLLOW_UP_PREFIX.match(question):
            return True
        # A turn that already names a project or an issue key stands on its own.
        if ISSUE_KEY_RE.search(question.upper()):
            return False
        names = [str(p.get("name", "")).lower() for p in known_projects]
        keys = [str(p.get("key", "")).lower() for p in known_projects]
        if any(name and name in low for name in names) or \
                any(re.search(rf"\b{re.escape(key)}\b", low) for key in keys if key):
            return False
        if DANGLING_REFERENCES.search(question) and not SELF_CONTAINED_HINT.search(question):
            return True
        return False

    # --- context extraction ------------------------------------------------
    @staticmethod
    def _context_entities(
        question: str,
        session_context: str,
        known_projects: Sequence[dict[str, Any]],
        memories: Sequence[MemoryRecord],
    ) -> dict[str, Any]:
        """Most recent project keys, issue keys and time window mentioned earlier."""
        lines = [line for line in session_context.splitlines() if line.strip()]
        project_keys: list[str] = []
        issue_keys: list[str] = []
        window: dict[str, Any] = {}

        for line in reversed(lines):                 # most recent first
            upper = line.upper()
            for key in ISSUE_KEY_RE.findall(upper):
                if key not in issue_keys:
                    issue_keys.append(key)
            low = line.lower()
            for project in known_projects:
                key = str(project.get("key", "")).upper()
                name = str(project.get("name", "")).lower()
                short = name.replace("project ", "").strip()
                if not key or key in project_keys:
                    continue
                if re.search(rf"\b{re.escape(key.lower())}\b", low) or (name and name in low) or \
                        (short and len(short) > 3 and re.search(rf"\b{re.escape(short)}\b", low)):
                    project_keys.append(key)
            if not window.get("start"):
                candidate = resolve_time_window(line)
                if candidate.get("start"):
                    window = candidate

        if not project_keys:
            memory_text = " ".join(m.content for m in memories).lower()
            for project in known_projects:
                key = str(project.get("key", "")).upper()
                name = str(project.get("name", "")).lower()
                if key and (re.search(rf"\b{re.escape(key.lower())}\b", memory_text)
                            or (name and name in memory_text)):
                    project_keys.append(key)

        return {"project_keys": project_keys[:2], "issue_keys": issue_keys[:3], "time_window": window}

    # --- LLM path ----------------------------------------------------------
    async def _rewrite_llm(
        self, question: str, session_context: str, entities: dict[str, Any]
    ) -> Optional[RewriteResult]:
        payload = (
            f"RECENT SESSION:\n{truncate(session_context, 1600) or '(new session)'}\n\n"
            f"ENTITIES DISCUSSED EARLIER: {entities}\n\n"
            f"USER TURN:\n{question}"
        )
        data = await self.llm.complete_json(REWRITE_SYSTEM, payload, max_tokens=350)
        rewritten = str(data.get("rewritten") or "").strip()
        if not rewritten:
            return None
        changed = bool(data.get("changed", rewritten.lower() != question.lower()))
        # Guard against a rewrite that throws the question away.
        if len(rewritten) > len(question) * 4 or len(rewritten) < 8:
            return RewriteResult(original=question, rewritten=question, changed=False,
                                 reason="model rewrite rejected: implausible length",
                                 method="llm", carried_entities=entities)
        return RewriteResult(
            original=question,
            rewritten=rewritten if changed else question,
            changed=changed and rewritten.lower() != question.lower(),
            resolved=[str(r) for r in (data.get("resolved") or [])][:5],
            reason=str(data.get("reason", ""))[:200],
            method="llm",
            carried_entities=entities,
        )

    # --- deterministic path ------------------------------------------------
    def _rewrite_heuristic(self, question: str, entities: dict[str, Any]) -> RewriteResult:
        project_keys = entities.get("project_keys") or []
        issue_keys = entities.get("issue_keys") or []
        window = entities.get("time_window") or {}
        resolved: list[str] = []
        rewritten = question

        low = question.lower()
        mentions_subject = bool(ISSUE_KEY_RE.search(question.upper())) or \
            any(key.lower() in low for key in project_keys)

        if not mentions_subject and (project_keys or issue_keys):
            subject = project_keys[0] if project_keys else issue_keys[0]
            label = f"Project {subject.title()}" if subject in project_keys else subject
            match = DANGLING_REFERENCES.search(question)
            token = match.group(0).lower() if match else ""
            if token in {"it", "that", "this", "these", "those", "they", "them",
                         "the same", "the project", "that one"}:
                start, end = match.span()
                rewritten = f"{question[:start]}{label}{question[end:]}"
                resolved.append(f"{match.group(0)} -> {label}")
            elif token == "there":
                # locative: "what is blocked there" -> "... in Project Apollo"
                start, end = match.span()
                rewritten = f"{question[:start]}in {label}{question[end:]}"
                resolved.append(f"there -> in {label}")
            else:
                connector = " for " if not low.rstrip().endswith("?") else " for "
                rewritten = f"{question.rstrip('?').rstrip()}{connector}{label}?"
                resolved.append(f"implicit subject -> {label}")

        elliptical = bool(resolved) or bool(FOLLOW_UP_PREFIX.match(question))
        if elliptical and window.get("label") and not resolve_time_window(question).get("start"):
            rewritten = f"{rewritten.rstrip('?').rstrip()} in {window['label']}?"
            resolved.append(f"implicit period -> {window['label']}")

        rewritten = re.sub(r"\s{2,}", " ", rewritten).strip()
        changed = rewritten.lower() != question.lower()
        return RewriteResult(
            original=question,
            rewritten=rewritten if changed else question,
            changed=changed,
            resolved=resolved,
            reason=("resolved from the entities discussed earlier in this session"
                    if changed else "no resolvable reference found"),
            method="heuristic",
            carried_entities=entities,
        )
