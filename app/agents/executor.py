"""The executor: plan -> tool calls -> evidence.

Responsibilities:
  * pick tools and arguments for each sub-question (LLM selection when available, deterministic
    rules otherwise);
  * carry results forward, so a sub-question that depends on an earlier one can use the issue
    keys that earlier retrieval discovered - this is where multi-hop actually happens;
  * enforce the execution budget: no duplicate calls, a hard cap on total calls and rounds, a
    cap on retained evidence, and per-item truncation to stop context explosion;
  * judge sufficiency and run one relaxed retry round when a sub-question came back empty;
  * convert every tool item into scored, citable Evidence.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import timedelta
from typing import Any, Optional

from app.agents.evaluator import EvidenceEvaluator, Sufficiency
from app.agents.prompts import TOOL_SELECTION_SYSTEM
from app.agents.state import AgentState
from app.llm.base import LLMClient, LLMError
from app.mcp_client.client import MCPToolClient
from app.models.evidence import Evidence, Plan, SubQuestion, ToolCallRecord
from app.observability.tracing import RunTracer
from app.security.guards import ExecutionBudget
from app.security.sanitize import detect_injection
from app.utils.dates import iso, parse_dt, utcnow
from app.utils.text import keyword_overlap, truncate

ISSUE_KEY_RE = re.compile(r"\b([A-Z][A-Z0-9_]{1,29}-\d{1,9})\b")

TOOL_PRIORITY = {
    "get_issue": 1.0, "get_issue_changelog": 1.0, "get_issue_comments": 0.95,
    "get_blocked_issues": 0.95, "get_overdue_issues": 0.95, "get_project_activity": 0.9,
    "search_issues": 0.9, "search_comments": 0.85, "get_project": 0.85,
    "search_projects": 0.7, "semantic_search": 0.75,
}

# A history hop fans out over the issues the previous hop surfaced. Three was too few: the
# epic whose deadline actually moved is frequently not the top-ranked hit.
HISTORY_HOP_KEYS = 5

COMPLETION_WORDS = ("complete", "completed", "done", "finished", "shipped", "delivered", "resolved", "closed")
OPEN_WORDS = ("open", "outstanding", "remaining", "unresolved", "in progress", "still", "pending")
CREATED_WORDS = ("created", "opened", "raised", "new ", "added")
DEADLINE_WORDS = ("due", "overdue", "deadline", "late", "slip", "delay", "behind")
HIGH_PRIORITY_WORDS = ("high priority", "highest priority", "critical", "urgent", "major")


class Executor:
    def __init__(
        self,
        mcp: MCPToolClient,
        tracer: RunTracer,
        *,
        llm: Optional[LLMClient] = None,
        max_tool_calls: int = 14,
        max_rounds: int = 3,
        max_evidence: int = 40,
        evidence_chars: int = 1200,
        max_items_per_call: int = 12,
    ):
        self.mcp = mcp
        self.tracer = tracer
        self.llm = llm
        self.evaluator = EvidenceEvaluator(llm, mcp.tool_names())
        # Verdicts from round 1, used to steer the retry round.
        self.verdicts: dict[str, Sufficiency] = {}
        self.max_items_per_call = max_items_per_call
        self.evidence_chars = evidence_chars
        self.budget = ExecutionBudget(
            max_tool_calls=max_tool_calls, max_rounds=max_rounds, max_evidence=max_evidence
        )

    # --- public -------------------------------------------------------------
    async def run(self, state: AgentState, plan: Plan) -> AgentState:
        self.budget.next_round()
        for sub_question in plan.sub_questions:
            await self._execute_sub_question(state, plan, sub_question, round_index=1)

        unresolved = [sq for sq in plan.sub_questions if sq.status != "answered"]
        if unresolved and self.budget.next_round() and self.budget.can_call():
            async with self.tracer.step(
                "evidence_gap_retry", "chain",
                {"unresolved": [sq.id for sq in unresolved]},
            ) as step:
                retried = []
                for sub_question in unresolved:
                    before = len(state.evidence)
                    verdict = self.verdicts.get(sub_question.id)
                    await self._execute_sub_question(
                        state, plan, sub_question, round_index=2, relaxed=True
                    )
                    retried.append({
                        "sub_question_id": sub_question.id,
                        "gap": verdict.missing if verdict else "",
                        "followed_suggestion": bool(verdict and verdict.suggested_tool),
                        "new_evidence": len(state.evidence) - before,
                        "status": sub_question.status,
                    })
                step.set_output(retries=retried)

        state.budget_snapshot = self.budget.snapshot()
        state.sufficiency = {sq_id: v.as_dict() for sq_id, v in self.verdicts.items()}
        state.insufficient_evidence = not state.evidence
        if any(sq.status != "answered" for sq in plan.sub_questions):
            gaps = [sq.question for sq in plan.sub_questions if sq.status != "answered"]
            state.warnings.append(
                "No evidence was found for: " + "; ".join(truncate(g, 120) for g in gaps)
            )
        return state

    # --- per sub-question ---------------------------------------------------
    async def _execute_sub_question(
        self, state: AgentState, plan: Plan, sub_question: SubQuestion,
        *, round_index: int, relaxed: bool = False,
    ) -> None:
        calls = await self._select_calls(state, plan, sub_question, relaxed=relaxed)
        async with self.tracer.step(
            f"sub_question:{sub_question.id}", "chain",
            {"question": sub_question.question, "tool_hints": sub_question.tool_hints,
             "round": round_index, "relaxed": relaxed, "planned_calls": calls},
        ) as step:
            collected = 0
            for call in calls:
                collected += await self._invoke(state, sub_question, call, round_index)

            verdict = await self.evaluator.assess(sub_question, state.evidence)
            self.verdicts[sub_question.id] = verdict
            sub_question.status = "answered" if verdict.sufficient else "insufficient"
            step.set_output(
                evidence_collected=collected,
                status=sub_question.status,
                sufficiency=verdict.as_dict(),
                budget=self.budget.snapshot(),
            )

    async def _invoke(
        self, state: AgentState, sub_question: SubQuestion, call: dict[str, Any], round_index: int
    ) -> int:
        tool = call.get("tool", "")
        arguments = call.get("arguments") or {}
        signature = f"{tool}:{json.dumps(arguments, sort_keys=True, default=str)}"
        decision = self.budget.register_call(signature)

        record = ToolCallRecord(
            call_id=f"tc_{uuid.uuid4().hex[:12]}", tool_name=tool, arguments=arguments,
            sub_question_id=sub_question.id, round_index=round_index,
        )
        if decision != "ok":
            record.status = decision
            record.error = ("Identical call already made in this run" if decision == "duplicate"
                            else "Execution budget exhausted")
            state.tool_records.append(record)
            await self.tracer.record_tool_call(record)
            return 0

        started = utcnow()
        result = await self.mcp.call(tool, arguments)
        record.duration_ms = int((utcnow() - started).total_seconds() * 1000)

        if not result.get("ok", False):
            error = result.get("error", {}) or {}
            record.status = "error"
            record.error = f"{error.get('kind', 'error')}: {error.get('message', 'unknown')}"
            state.tool_records.append(record)
            await self.tracer.record_tool_call(record)
            state.warnings.append(f"{tool} failed ({error.get('kind', 'error')})")
            return 0

        items = result.get("items", []) or []
        record.result_count = len(items)
        state.tool_records.append(record)
        await self.tracer.record_tool_call(record)
        return self._absorb(state, sub_question, tool, items)

    # --- evidence -----------------------------------------------------------
    def _absorb(self, state: AgentState, sub_question: SubQuestion, tool: str, items: list[dict]) -> int:
        if len(state.evidence) >= self.budget.max_evidence:
            return 0
        seen = {e.fingerprint() for e in state.evidence}
        # Changelog entries and comments have globally unique ids, so the same entry retrieved
        # by two different calls is one piece of evidence, not two. Issues are exempt: different
        # tools annotate the same issue differently (BLOCKED/OVERDUE prefixes) and both matter.
        entry_ids = {
            (e.source_type, e.source_id) for e in state.evidence
            if e.source_type in {"changelog", "comment"}
        }
        scored: list[tuple[float, Evidence]] = []

        for item in items:
            content = str(item.get("content", "") or "")
            if not content.strip():
                continue
            injection = bool(detect_injection(content))
            data = item.get("data", {}) or {}
            evidence = Evidence(
                source="jira" if tool != "semantic_search" else "jira_index",
                source_type=str(item.get("source_type", "")),
                source_id=str(item.get("source_id", "")),
                title=str(item.get("title", "")),
                url=str(item.get("url", "")),
                timestamp=parse_dt(item.get("timestamp")),
                content=truncate(content, self.evidence_chars),
                tool_used=tool,
                sub_question_id=sub_question.id,
                project_key=str(data.get("project_key") or (str(item.get("source_id", "")).split("-")[0])),
                injection_suspected=injection,
                metadata={k: v for k, v in data.items() if k not in {"links"}},
            )
            evidence.relevance = self._score(evidence, sub_question, tool)
            entry_key = (evidence.source_type, evidence.source_id)
            if evidence.source_type in {"changelog", "comment"}:
                if entry_key in entry_ids:
                    continue
                entry_ids.add(entry_key)
            if evidence.fingerprint() in seen:
                continue
            seen.add(evidence.fingerprint())
            scored.append((evidence.relevance, evidence))

        scored.sort(key=lambda pair: pair[0], reverse=True)
        added = 0
        for _, evidence in scored[: self.max_items_per_call]:
            if len(state.evidence) >= self.budget.max_evidence:
                break
            state.add_evidence(evidence)
            sub_question.evidence_ids.append(evidence.evidence_id)
            added += 1
        return added

    def _score(self, evidence: Evidence, sub_question: SubQuestion, tool: str) -> float:
        lexical = keyword_overlap(sub_question.question, f"{evidence.title} {evidence.content}")
        priority = TOOL_PRIORITY.get(tool, 0.7)
        recency = 0.0
        if evidence.timestamp:
            age_days = max(0.0, (utcnow() - evidence.timestamp).total_seconds() / 86400.0)
            recency = 0.5 ** (age_days / 120.0)
        structural = 0.0
        upper = evidence.content[:200].upper()
        if upper.startswith("BLOCKED") or upper.startswith("OVERDUE"):
            structural = 0.15
        if evidence.metadata.get("kind") == "duedate_change":
            structural = max(structural, 0.15)
        score = 0.45 * priority + 0.30 * lexical + 0.15 * recency + structural
        if evidence.injection_suspected:
            score *= 0.15   # keep it as evidence, but never let it lead the ranking
        return round(min(score, 1.0), 4)

    # --- tool selection -----------------------------------------------------
    async def _select_calls(
        self, state: AgentState, plan: Plan, sub_question: SubQuestion, *, relaxed: bool
    ) -> list[dict[str, Any]]:
        suggested: list[dict[str, Any]] = []
        if relaxed:
            verdict = self.verdicts.get(sub_question.id)
            if verdict and verdict.suggested_tool:
                spec = self.mcp.tools.get(verdict.suggested_tool)
                allowed = set((spec.input_schema or {}).get("properties", {}) or {}) if spec else set()
                suggested = [{
                    "tool": verdict.suggested_tool,
                    "arguments": {k: v for k, v in verdict.suggested_arguments.items() if k in allowed},
                    "why": f"evaluator asked for: {verdict.missing[:120]}",
                }]

        if self.llm is not None:
            try:
                calls = await self._select_calls_llm(state, plan, sub_question, relaxed)
                if calls:
                    return suggested + calls
            except (LLMError, ValueError, TypeError, KeyError):
                pass
        return suggested + self._select_calls_heuristic(state, plan, sub_question, relaxed)

    async def _select_calls_llm(
        self, state: AgentState, plan: Plan, sub_question: SubQuestion, relaxed: bool
    ) -> list[dict[str, Any]]:
        hint_names = sub_question.tool_hints or self.mcp.tool_names()
        catalogue = self.mcp.describe_tools(list(dict.fromkeys(hint_names + self.mcp.tool_names())))
        discovered = self._discovered_issue_keys(state, sub_question)
        already = [f"{r.tool_name}({json.dumps(r.arguments, sort_keys=True, default=str)})"
                   for r in state.tool_records][-10:]
        payload = (
            f"CURRENT TIME: {iso(utcnow())}\n"
            f"PLAN GOAL: {plan.goal}\n"
            f"TIME RANGE: {plan.time_range}\n"
            f"PROJECT KEYS: {plan.entities.get('project_keys', [])}\n"
            f"ISSUE KEYS FROM EARLIER RETRIEVAL: {discovered}\n"
            f"SUB-QUESTION: {sub_question.question}\n"
            f"PLANNER TOOL HINTS: {sub_question.tool_hints}\n"
            f"CALLS ALREADY MADE: {already or '(none)'}\n"
            + ("NOTE: the previous attempt returned nothing. Widen filters or try a different tool.\n"
               if relaxed else "")
            + f"\nTOOL CATALOGUE:\n{catalogue}"
        )
        data = await self.llm.complete_json(TOOL_SELECTION_SYSTEM, payload, max_tokens=600)
        calls: list[dict[str, Any]] = []
        for raw in (data.get("calls") or [])[:3]:
            if not isinstance(raw, dict):
                continue
            tool = str(raw.get("tool", ""))
            spec = self.mcp.tools.get(tool)
            if spec is None:
                continue
            allowed = set((spec.input_schema or {}).get("properties", {}) or {})
            arguments = {k: v for k, v in (raw.get("arguments") or {}).items() if k in allowed}
            calls.append({"tool": tool, "arguments": arguments, "why": str(raw.get("why", ""))[:200]})
        return calls

    def _select_calls_heuristic(
        self, state: AgentState, plan: Plan, sub_question: SubQuestion, relaxed: bool
    ) -> list[dict[str, Any]]:
        projects = [p for p in plan.entities.get("project_keys", []) if p]
        project_key = projects[0] if projects else ""
        start = plan.time_range.get("start") or ""
        end = plan.time_range.get("end") or ""
        if relaxed:
            start_dt, end_dt = parse_dt(start), parse_dt(end)
            start = iso(start_dt - timedelta(days=45)) if start_dt else ""
            end = iso(end_dt + timedelta(days=45)) if end_dt else ""

        text = sub_question.question.lower()
        issue_keys = plan.entities.get("issue_keys", []) or self._discovered_issue_keys(state, sub_question)
        calls: list[dict[str, Any]] = []

        for tool in (sub_question.tool_hints or ["semantic_search"]):
            if tool == "search_projects":
                calls.append({"tool": tool, "arguments": {"limit": 25}})
            elif tool == "get_project" and project_key:
                calls.append({"tool": tool, "arguments": {"project_key": project_key}})
            elif tool == "get_project_activity" and project_key:
                calls.append({"tool": tool, "arguments": {
                    "project_key": project_key, "start": start, "end": end, "limit": 60}})
            elif tool == "get_blocked_issues":
                calls.append({"tool": tool, "arguments": {"project_key": project_key, "limit": 15}})
            elif tool == "get_overdue_issues":
                calls.append({"tool": tool, "arguments": {
                    "project_key": project_key, "as_of": end or iso(utcnow()), "limit": 15}})
            elif tool == "search_comments" and project_key:
                calls.append({"tool": tool, "arguments": {
                    "project_key": project_key, "start": start, "end": end, "limit": 15}})
            elif tool == "get_issue":
                for key in issue_keys[:3]:
                    calls.append({"tool": tool, "arguments": {"issue_key": key}})
            elif tool == "get_issue_comments":
                for key in issue_keys[:HISTORY_HOP_KEYS]:
                    calls.append({"tool": tool, "arguments": {"issue_key": key, "limit": 20}})
            elif tool == "get_issue_changelog":
                fields = ["duedate", "status"] if any(w in text for w in DEADLINE_WORDS) else []
                for key in issue_keys[:HISTORY_HOP_KEYS]:
                    arguments: dict[str, Any] = {"issue_key": key, "limit": 50}
                    if fields and not relaxed:
                        arguments["fields"] = fields
                    calls.append({"tool": tool, "arguments": arguments})
            elif tool == "search_issues":
                calls.append({"tool": tool, "arguments": self._issue_filters(
                    text, project_key, start, end, relaxed)})
            elif tool == "semantic_search":
                arguments = {"query": sub_question.question, "top_k": 8}
                if project_key and not relaxed:
                    arguments["project_key"] = project_key
                calls.append({"tool": tool, "arguments": arguments})

        deduped: list[dict[str, Any]] = []
        seen: set[str] = set()
        for call in calls:
            key = f"{call['tool']}:{json.dumps(call['arguments'], sort_keys=True, default=str)}"
            if key in seen:
                continue
            seen.add(key)
            deduped.append(call)
        return deduped[:6]

    @staticmethod
    def _issue_filters(
        text: str, project_key: str, start: str, end: str, relaxed: bool
    ) -> dict[str, Any]:
        arguments: dict[str, Any] = {"limit": 25, "order_by": "updated"}
        if project_key:
            arguments["project_key"] = project_key
        if any(word in text for word in COMPLETION_WORDS):
            arguments["status_category"] = "Done"
            if start:
                arguments["resolved_after"] = start
            if end:
                arguments["resolved_before"] = end
        elif any(word in text for word in DEADLINE_WORDS):
            arguments["unresolved_only"] = True
            if end:
                arguments["due_before"] = end
            arguments["order_by"] = "duedate"
        elif any(word in text for word in OPEN_WORDS):
            arguments["unresolved_only"] = True
            if start:
                arguments["updated_after"] = start
        elif any(word in text for word in CREATED_WORDS):
            if start:
                arguments["created_after"] = start
            if end:
                arguments["created_before"] = end
        else:
            if start:
                arguments["updated_after"] = start
            if end:
                arguments["updated_before"] = end
        if any(word in text for word in HIGH_PRIORITY_WORDS):
            arguments["priorities"] = ["Highest", "High"]
        if relaxed:
            for key in ("updated_before", "created_before", "resolved_before"):
                arguments.pop(key, None)
            arguments["limit"] = 30
        return arguments

    @staticmethod
    def _discovered_issue_keys(state: AgentState, sub_question: SubQuestion) -> list[str]:
        """Issue keys surfaced by earlier retrieval - the hop between sub-questions."""
        keys: list[str] = []
        for evidence in sorted(state.evidence, key=lambda e: e.relevance, reverse=True):
            if sub_question.depends_on and evidence.sub_question_id not in sub_question.depends_on:
                continue
            candidate = evidence.metadata.get("issue_key") or evidence.metadata.get("key")
            if isinstance(candidate, str) and ISSUE_KEY_RE.fullmatch(candidate):
                keys.append(candidate.upper())
            else:
                found = ISSUE_KEY_RE.findall(f"{evidence.source_id} {evidence.title}".upper())
                keys.extend(found)
        if not keys and sub_question.depends_on:
            for evidence in sorted(state.evidence, key=lambda e: e.relevance, reverse=True):
                keys.extend(ISSUE_KEY_RE.findall(f"{evidence.source_id} {evidence.title}".upper()))
        return list(dict.fromkeys(keys))[:8]
