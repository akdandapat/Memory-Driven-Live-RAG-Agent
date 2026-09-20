"""The planner: question -> structured, dynamic plan.

Two paths, one output contract (`Plan`):
  * LLM planner  - decomposes freely, constrained by a JSON schema and the live tool catalogue
                   discovered over MCP (so it can never plan a tool that does not exist);
  * heuristic planner - used when no LLM is configured. It is *not* a hard-coded workflow: it
                   detects intents, entities and time windows from the question and composes
                   sub-questions from those signals, so different questions produce different
                   plans. It is a degraded fallback, and the trace says which path ran.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

from app.agents.prompts import PLANNER_SYSTEM
from app.llm.base import LLMClient, LLMError
from app.memory.extractor import looks_like_user_statement
from app.models.evidence import Plan, SubQuestion
from app.utils.dates import iso, resolve_time_window, utcnow
from app.utils.text import truncate

# --- intent detection -------------------------------------------------------
INTENT_PATTERNS: dict[str, tuple[str, ...]] = {
    "change": ("what changed", "changes", "changed", "progress", "happened", "update on",
               "summarize", "summarise", "recap", "activity", "moved"),
    "risk": ("risk", "risks", "risky", "concern", "concerns", "worry", "worried", "threat",
             "danger", "problem", "problems", "exposure"),
    "cause": ("why", "caused", "cause", "reason", "root cause", "led to", "because", "explain why"),
    "blocker": ("blocked", "blocker", "blockers", "stuck", "impediment", "waiting on", "held up"),
    "deadline": ("deadline", "deadlines", "due", "overdue", "late", "slip", "slipped", "delay",
                 "delayed", "schedule", "behind", "timeline", "missed"),
    "ownership": ("who owns", "who is responsible", "who runs", "who leads", "owner", "lead of",
                  "assignee", "who is working", "responsible for"),
    "status": ("status", "state of", "how is", "where are we", "on track", "health"),
    "personal": ("should i", "what should i", "prioritize", "prioritise", "focus on", "my priority",
                 "my priorities", "for me"),
    "recall": ("what did i say", "did i tell you", "do you remember", "remember", "i told you",
               "what did i tell"),
    "discussion": ("comments", "discussion", "people saying", "saying about", "feedback",
                   "raised", "mentioned", "talking about"),
    "completed": ("completed", "done", "finished", "shipped", "delivered", "closed", "resolved"),
    "open": ("open", "outstanding", "remaining", "still open", "unresolved", "in progress",
             "not done", "pending"),
    "projects": ("what projects", "which projects", "list projects", "all projects",
                 "projects exist", "projects are there"),
}

GENERAL_KNOWLEDGE = (
    "what is", "what are", "explain", "how does", "define", "difference between", "hello", "hi ",
    "thanks", "thank you", "who are you", "what can you do",
)

ISSUE_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,29}-\d{1,9})\b")


def detect_intents(question: str) -> list[str]:
    low = f" {question.lower().strip()} "
    found = [intent for intent, needles in INTENT_PATTERNS.items()
             if any(needle in low for needle in needles)]
    return found


class Planner:
    def __init__(
        self,
        llm: Optional[LLMClient] = None,
        *,
        max_sub_questions: int = 6,
    ):
        self.llm = llm
        self.max_sub_questions = max_sub_questions

    async def plan(
        self,
        question: str,
        *,
        tool_catalogue: str,
        tool_names: Sequence[str],
        known_projects: Sequence[dict[str, Any]],
        memory_block: str = "",
        session_context: str = "",
    ) -> Plan:
        if self.llm is not None:
            try:
                plan = await self._plan_llm(
                    question, tool_catalogue, tool_names, known_projects, memory_block, session_context
                )
                if plan.sub_questions or not plan.needs_retrieval:
                    return plan
            except (LLMError, ValueError, KeyError, TypeError):
                pass  # deterministic fallback below; the trace records generated_by
        return self.plan_heuristic(question, tool_names, known_projects, memory_block)

    # --- LLM path ----------------------------------------------------------
    async def _plan_llm(
        self,
        question: str,
        tool_catalogue: str,
        tool_names: Sequence[str],
        known_projects: Sequence[dict[str, Any]],
        memory_block: str,
        session_context: str,
    ) -> Plan:
        project_lines = "\n".join(
            f"- {p.get('key')}: {p.get('name', '')}" for p in known_projects
        ) or "- (none discovered)"
        window = resolve_time_window(question)
        payload = (
            f"CURRENT TIME: {iso(utcnow())}\n\n"
            f"AVAILABLE TOOLS:\n{tool_catalogue}\n\n"
            f"PROJECTS IN THE TRACKER:\n{project_lines}\n\n"
            f"REMEMBERED USER CONTEXT (long-term memory):\n{memory_block or '(none)'}\n\n"
            f"THIS SESSION SO FAR:\n{truncate(session_context, 1500) or '(new session)'}\n\n"
            f"TIME WINDOW PARSED FROM THE QUESTION: {window}\n\n"
            f"USER QUESTION:\n{question}"
        )
        system = PLANNER_SYSTEM.replace("{max_sub_questions}", str(self.max_sub_questions))
        data = await self.llm.complete_json(system, payload, max_tokens=900)

        valid_tools = set(tool_names)
        sub_questions: list[SubQuestion] = []
        for index, raw in enumerate((data.get("sub_questions") or [])[: self.max_sub_questions], start=1):
            if not isinstance(raw, dict) or not raw.get("question"):
                continue
            hints = [t for t in (raw.get("tool_hints") or []) if t in valid_tools]
            sub_questions.append(SubQuestion(
                id=str(raw.get("id") or f"sq{index}"),
                question=str(raw["question"])[:400],
                rationale=str(raw.get("rationale", ""))[:300],
                tool_hints=hints,
                depends_on=[str(d) for d in (raw.get("depends_on") or [])],
            ))

        entities = data.get("entities") if isinstance(data.get("entities"), dict) else {}
        time_range = data.get("time_range") if isinstance(data.get("time_range"), dict) else {}
        if not time_range.get("start") and window.get("start"):
            time_range = window

        return Plan(
            goal=str(data.get("goal") or question)[:400],
            needs_retrieval=bool(data.get("needs_retrieval", True)),
            reasoning_strategy=str(data.get("reasoning_strategy", "single_hop"))[:40],
            sub_questions=sub_questions,
            required_tools=sorted({t for sq in sub_questions for t in sq.tool_hints}),
            entities={
                "project_keys": [str(k).upper() for k in (entities.get("project_keys") or [])],
                "issue_keys": [str(k).upper() for k in (entities.get("issue_keys") or [])],
                "people": [str(p) for p in (entities.get("people") or [])],
            },
            time_range={"start": time_range.get("start"), "end": time_range.get("end"),
                        "label": time_range.get("label", "")},
            generated_by="llm",
            notes=str(data.get("notes", ""))[:400],
        )

    # --- heuristic path ----------------------------------------------------
    def plan_heuristic(
        self,
        question: str,
        tool_names: Sequence[str],
        known_projects: Sequence[dict[str, Any]],
        memory_block: str = "",
    ) -> Plan:
        intents = detect_intents(question)
        explicit_projects = self._match_projects(question, known_projects)
        # Remembered context may supply the missing subject ("what should I focus on?"), but only
        # when the turn is actually asking for tracker data.
        data_intents = {"change", "risk", "cause", "blocker", "deadline", "status", "open",
                        "completed", "discussion", "personal", "ownership"}
        projects = explicit_projects
        if not projects and (set(intents) & data_intents) and "recall" not in intents:
            projects = self._match_projects(question, known_projects, memory_block)
        issue_keys = ISSUE_KEY_RE.findall(question.upper())
        window = resolve_time_window(question)
        available = set(tool_names)

        needs_retrieval = bool(projects or issue_keys or intents)
        if "recall" in intents and not explicit_projects and not issue_keys:
            # "what did I say my priority was" is answered from memory, not from the tracker
            needs_retrieval = False
        if looks_like_user_statement(question):
            # The user is telling us something, not asking for tracker data.
            return Plan(
                goal=f"Acknowledge and remember: {truncate(question, 180)}",
                needs_retrieval=False, reasoning_strategy="statement_capture", sub_questions=[],
                required_tools=[], entities={"project_keys": projects, "issue_keys": issue_keys},
                time_range=window, generated_by="heuristic",
                notes="Durable user statement; routed to memory rather than retrieval.",
            )
        low = question.lower().strip()
        if not explicit_projects and not issue_keys and not intents and \
                any(low.startswith(g) for g in GENERAL_KNOWLEDGE):
            needs_retrieval = False

        if not needs_retrieval:
            strategy = "memory_only" if "recall" in intents or "personal" in intents else "no_retrieval"
            return Plan(
                goal=f"Answer conversationally: {truncate(question, 200)}",
                needs_retrieval=False, reasoning_strategy=strategy, sub_questions=[],
                required_tools=[], entities={"project_keys": projects, "issue_keys": issue_keys},
                time_range=window, generated_by="heuristic",
                notes="No tracker retrieval required for this question.",
            )

        sub_questions: list[SubQuestion] = []

        def add(question_text: str, rationale: str, hints: list[str], depends_on: Optional[list[str]] = None) -> None:
            hints = [h for h in hints if h in available]
            if not hints or len(sub_questions) >= self.max_sub_questions:
                return
            sub_questions.append(SubQuestion(
                id=f"sq{len(sub_questions) + 1}", question=question_text, rationale=rationale,
                tool_hints=hints, depends_on=depends_on or [],
            ))

        project_label = ", ".join(projects) if projects else "the tracker"

        if "projects" in intents and not projects:
            add("Which projects exist in the tracker?", "The question asks about the project list.",
                ["search_projects"])

        if issue_keys:
            add(f"What is the current state of {', '.join(issue_keys)}?",
                "The question names specific issues.", ["get_issue"])
            if {"cause", "discussion", "blocker"} & set(intents):
                add(f"What has been discussed on {', '.join(issue_keys)}?",
                    "Causal detail usually lives in comments.", ["get_issue_comments"], ["sq1"])
            if {"cause", "deadline", "change"} & set(intents):
                add(f"What fields changed on {', '.join(issue_keys)} and when?",
                    "Field history shows what moved.", ["get_issue_changelog"], ["sq1"])

        if projects and ("ownership" in intents or "status" in intents or "change" in intents
                         or "risk" in intents or "personal" in intents):
            add(f"What are the goals, lead and scope of {project_label}?",
                "Judging change or risk requires knowing the stated goals.", ["get_project"])

        if "change" in intents and projects:
            label = window.get("label") or "the requested period"
            add(f"What activity happened in {project_label} during {label}?",
                "Creations, resolutions, transitions and due-date moves define 'what changed'.",
                ["get_project_activity"])

        if "completed" in intents and projects:
            add(f"Which issues in {project_label} were completed in the period?",
                "Completion is a resolution-date filter, not a text search.", ["search_issues"])

        if "open" in intents and projects:
            add(f"Which issues in {project_label} are still open?",
                "Outstanding work is an unresolved filter.", ["search_issues"])

        if ("deadline" in intents or "risk" in intents) and projects:
            add(f"Which issues in {project_label} are overdue or had their due date moved?",
                "Deadline pressure is the strongest objective risk signal.",
                ["get_overdue_issues", "search_issues"])

        if "personal" in intents and projects:
            add(f"Which issues in {project_label} need attention right now (overdue or blocked)?",
                "A 'what should I focus on' answer must be grounded in current pressure points.",
                ["get_overdue_issues", "get_blocked_issues"])

        if ("blocker" in intents or "risk" in intents or "cause" in intents) and projects:
            add(f"Which issues in {project_label} are blocked, and by what?",
                "Blocked work explains both risk and delay.", ["get_blocked_issues"])

        if ({"deadline", "cause"} & set(intents)) and projects and not issue_keys:
            # First hop: find which issues the question is actually about, by meaning rather
            # than by filter ("the EU cutover date").
            add(f"Which issues in {project_label} relate to: {truncate(question, 140)}?",
                "The question names a subject, not an issue key, so locate the issues first.",
                ["semantic_search", "search_issues"])

        if ({"deadline", "cause", "change"} & set(intents)) and projects and not issue_keys:
            # Second hop: current fields never show their own history, so once retrieval has
            # surfaced the relevant issues, go back for what moved on them and when.
            add(f"For the issues surfaced above, which due dates or statuses changed and when?",
                "Deadline movement is only visible in field history, which needs the issue keys "
                "discovered by the previous step.",
                ["get_issue_changelog"],
                [sq.id for sq in sub_questions[-3:]] or None)

        if ("cause" in intents or "risk" in intents or "discussion" in intents) and projects:
            add(f"What have people reported about problems in {project_label}?",
                "Causes are usually stated in discussion, not in fields.",
                ["search_comments", "semantic_search"],
                [sub_questions[-1].id] if sub_questions and "cause" in intents else [])

        if not sub_questions:
            # Fall back to a broad retrieval pass rather than answering ungrounded.
            add(f"What information in {project_label} is relevant to: {truncate(question, 160)}?",
                "No specific intent detected; retrieve broadly and let synthesis filter.",
                ["semantic_search", "search_issues"])

        strategy = "single_hop"
        if "cause" in intents:
            strategy = "causal"
        elif "risk" in intents:
            strategy = "risk_analysis"
        elif "change" in intents or window.get("start"):
            strategy = "temporal"
        if len(sub_questions) > 2 and any(sq.depends_on for sq in sub_questions):
            strategy = "multi_hop" if strategy == "single_hop" else strategy

        return Plan(
            goal=f"Answer with cited evidence: {truncate(question, 220)}",
            needs_retrieval=True,
            reasoning_strategy=strategy,
            sub_questions=sub_questions,
            required_tools=sorted({t for sq in sub_questions for t in sq.tool_hints}),
            entities={"project_keys": projects, "issue_keys": issue_keys, "people": []},
            time_range=window,
            generated_by="heuristic",
            notes=f"Intents detected: {', '.join(intents) or 'none'}",
        )

    @staticmethod
    def _match_projects(
        question: str, known_projects: Sequence[dict[str, Any]], memory_block: str = ""
    ) -> list[str]:
        """Resolve project references in the question (and optionally in remembered context)."""
        low = question.lower()
        matched: list[str] = []
        for project in known_projects:
            key = str(project.get("key", "")).upper()
            name = str(project.get("name", "")).lower()
            if not key:
                continue
            if re.search(rf"\b{re.escape(key.lower())}\b", low) or (name and name in low):
                matched.append(key)
                continue
            # "Atlas" should match "Project Atlas"
            short = name.replace("project ", "").strip()
            if short and len(short) > 3 and re.search(rf"\b{re.escape(short)}\b", low):
                matched.append(key)
        if not matched and memory_block:
            # The user's remembered priority can supply the missing subject of the question.
            memory_low = memory_block.lower()
            for project in known_projects:
                key = str(project.get("key", "")).upper()
                name = str(project.get("name", "")).lower()
                if key and (re.search(rf"\b{re.escape(key.lower())}\b", memory_low)
                            or (name and name in memory_low)):
                    matched.append(key)
        return list(dict.fromkeys(matched))
