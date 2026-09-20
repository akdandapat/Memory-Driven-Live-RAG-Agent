"""Metric computation for the evaluation harness.

Every number is measured from an actual run. Nothing here is a target or an estimate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


def _recall(expected: list[str], observed: set[str], prefix_match: bool = False) -> Optional[float]:
    if not expected:
        return None
    if prefix_match:
        hits = sum(1 for item in expected if any(o.startswith(item) for o in observed))
    else:
        hits = sum(1 for item in expected if item in observed)
    return hits / len(expected)


@dataclass
class CaseResult:
    case_id: str
    category: str
    question: str
    passed: bool = True
    failures: list[str] = field(default_factory=list)
    tool_selection_recall: Optional[float] = None
    retrieval_recall: Optional[float] = None
    plan_quality: Optional[float] = None
    evidence_relevance: Optional[float] = None
    evidence_precision: Optional[float] = None
    rewrite_correct: Optional[bool] = None
    content_recall: Optional[float] = None
    citation_valid: Optional[bool] = None
    citation_count: int = 0
    invented_citations: int = 0
    uncited_lines: int = 0
    abstained: Optional[bool] = None
    memory_used: Optional[bool] = None
    memory_action_correct: Optional[bool] = None
    hallucination: bool = False
    tool_calls: int = 0
    evidence: int = 0
    latency_ms: int = 0

    def fail(self, reason: str) -> None:
        self.passed = False
        self.failures.append(reason)

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def evaluate_case(case: dict[str, Any], result: dict[str, Any]) -> CaseResult:
    outcome = CaseResult(
        case_id=case["id"], category=case["category"], question=case["question"],
        tool_calls=len([c for c in result["tool_calls"] if c["status"] == "ok"]),
        evidence=len(result["evidence"]),
        latency_ms=result.get("latency_ms", 0),
    )
    answer = result.get("answer", "")
    lowered = answer.lower()
    tools_called = {c["tool_name"] for c in result["tool_calls"] if c["status"] == "ok"}
    evidence_sources = {e["source_id"] for e in result["evidence"]}
    cited_sources = {c["source_id"] for c in result["citations"]}

    # --- tool selection ---------------------------------------------------
    outcome.tool_selection_recall = _recall(case.get("expected_tools", []), tools_called)
    if outcome.tool_selection_recall is not None and outcome.tool_selection_recall < 1.0:
        missing = set(case["expected_tools"]) - tools_called
        outcome.fail(f"tools not called: {', '.join(sorted(missing))}")

    if case.get("min_tool_calls") and outcome.tool_calls < case["min_tool_calls"]:
        outcome.fail(f"only {outcome.tool_calls} tool calls, expected >= {case['min_tool_calls']}")

    # --- plan quality -----------------------------------------------------
    # Did the plan actually cover the aspects the question requires? Measured against the union
    # of sub-question text and the tools the plan asked for, so a plan that names the right work
    # scores well even if it words it differently.
    aspects = case.get("expected_plan_aspects", [])
    plan = result.get("plan", {}) or {}
    sub_questions = plan.get("sub_questions", []) or []
    if aspects:
        plan_text = " ".join(
            [plan.get("goal", ""), plan.get("reasoning_strategy", "")]
            + [sq.get("question", "") for sq in sub_questions]
            + [sq.get("rationale", "") for sq in sub_questions]
            + (plan.get("required_tools", []) or [])
        ).lower()
        covered = sum(1 for aspect in aspects if aspect.lower() in plan_text)
        outcome.plan_quality = covered / len(aspects)
        if covered < len(aspects):
            missing = [a for a in aspects if a.lower() not in plan_text]
            outcome.fail(f"plan does not cover: {', '.join(missing)}")
    elif case.get("expect_no_retrieval"):
        # A correct plan here is one that decided not to retrieve at all.
        outcome.plan_quality = 0.0 if plan.get("needs_retrieval") else 1.0

    # --- query rewrite ----------------------------------------------------
    if "expect_rewrite" in case:
        rewrite = result.get("rewrite", {}) or {}
        expected = bool(case["expect_rewrite"])
        outcome.rewrite_correct = bool(rewrite.get("changed")) == expected
        if not outcome.rewrite_correct:
            outcome.fail(
                f"expected rewrite={expected}, got {rewrite.get('changed')} "
                f"({rewrite.get('rewritten', '')[:80]})"
            )
        for term in case.get("expected_rewrite_terms", []):
            if term.lower() not in str(rewrite.get("rewritten", "")).lower():
                outcome.rewrite_correct = False
                outcome.fail(f"rewritten question missing {term!r}: {rewrite.get('rewritten', '')}")

    # --- retrieval --------------------------------------------------------
    outcome.retrieval_recall = _recall(case.get("expected_sources", []), evidence_sources,
                                       prefix_match=True)
    if outcome.retrieval_recall is not None and outcome.retrieval_recall < 1.0:
        missing = [s for s in case["expected_sources"]
                   if not any(o.startswith(s) for o in evidence_sources)]
        outcome.fail(f"evidence missing: {', '.join(missing)}")

    # --- answer content ---------------------------------------------------
    must_include = case.get("must_include", [])
    if must_include:
        hits = sum(1 for term in must_include if term.lower() in lowered)
        outcome.content_recall = hits / len(must_include)
        if hits < len(must_include):
            missing = [t for t in must_include if t.lower() not in lowered]
            outcome.fail(f"answer missing: {', '.join(missing)}")

    for term in case.get("must_not_include", []):
        if term.lower() in lowered:
            outcome.hallucination = True
            outcome.fail(f"answer contains forbidden text: {term!r}")

    # --- evidence relevance -----------------------------------------------
    if result["evidence"]:
        cited_ids = {c["marker"] for c in result["citations"]}
        cited = [e for e in result["evidence"] if e["evidence_id"] in cited_ids]
        outcome.evidence_precision = round(len(cited) / len(result["evidence"]), 3)
        if cited:
            outcome.evidence_relevance = round(
                sum(e["relevance"] for e in cited) / len(cited), 3
            )

    # --- citations --------------------------------------------------------
    stats = result.get("citation_stats", {}) or {}
    outcome.citation_count = len(result["citations"])
    outcome.invented_citations = int(stats.get("invented_markers_removed", 0))
    outcome.uncited_lines = int(stats.get("uncited_claims", 0))
    if result["evidence"]:
        outcome.citation_valid = (
            outcome.invented_citations == 0
            and cited_sources <= evidence_sources
            and outcome.citation_count > 0
        )
        if not outcome.citation_valid:
            outcome.fail("citations are missing or do not resolve to collected evidence")

    # --- abstention -------------------------------------------------------
    if case.get("expect_abstain"):
        outcome.abstained = bool(
            result.get("insufficient_evidence")
            or any(phrase in lowered for phrase in
                   ("does not cover", "not going to guess", "could not find", "no evidence",
                    "not going to answer"))
        )
        if not outcome.abstained:
            outcome.fail("expected the agent to say the evidence is insufficient")

    if case.get("expect_no_retrieval") and result["plan"].get("needs_retrieval"):
        outcome.fail("expected no tracker retrieval for this turn")

    # --- memory -----------------------------------------------------------
    if case.get("expect_memory_used"):
        outcome.memory_used = bool(result.get("memories_used"))
        if not outcome.memory_used:
            outcome.fail("expected long-term memory to be used")

    expected_action = case.get("expect_memory_action")
    if expected_action:
        actions = [a["action"] for a in result.get("memory_update", {}).get("actions", [])]
        outcome.memory_action_correct = expected_action in actions
        if not outcome.memory_action_correct:
            outcome.fail(f"expected memory action {expected_action}, got {actions or 'none'}")

    return outcome


def aggregate(results: list[CaseResult], memory_precision: Optional[float] = None,
              memory_recall: Optional[float] = None) -> dict[str, Any]:
    def mean(values: list[float]) -> Optional[float]:
        values = [v for v in values if v is not None]
        return round(sum(values) / len(values), 3) if values else None

    latencies = sorted(r.latency_ms for r in results)
    summary: dict[str, Any] = {
        "cases": len(results),
        "passed": sum(1 for r in results if r.passed),
        "pass_rate": round(sum(1 for r in results if r.passed) / len(results), 3) if results else 0.0,
        "tool_selection_recall": mean([r.tool_selection_recall for r in results]),
        "retrieval_recall": mean([r.retrieval_recall for r in results]),
        "plan_quality": mean([r.plan_quality for r in results]),
        "evidence_relevance": mean([r.evidence_relevance for r in results]),
        "evidence_precision": mean([r.evidence_precision for r in results]),
        "rewrite_accuracy": mean([1.0 if r.rewrite_correct else 0.0
                                  for r in results if r.rewrite_correct is not None]),
        "answer_content_recall": mean([r.content_recall for r in results]),
        "citation_validity": mean([1.0 if r.citation_valid else 0.0
                                   for r in results if r.citation_valid is not None]),
        "invented_citations_total": sum(r.invented_citations for r in results),
        "abstention_accuracy": mean([1.0 if r.abstained else 0.0
                                     for r in results if r.abstained is not None]),
        "forbidden_content_rate": round(
            sum(1 for r in results if r.hallucination) / len(results), 3) if results else 0.0,
        "mean_tool_calls": round(sum(r.tool_calls for r in results) / len(results), 2) if results else 0.0,
        "mean_evidence": round(sum(r.evidence for r in results) / len(results), 2) if results else 0.0,
        "latency_ms_p50": latencies[len(latencies) // 2] if latencies else 0,
        "latency_ms_p95": latencies[max(0, int(len(latencies) * 0.95) - 1)] if latencies else 0,
    }
    if memory_precision is not None:
        summary["memory_precision"] = memory_precision
    if memory_recall is not None:
        summary["memory_recall"] = memory_recall

    by_category: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = by_category.setdefault(result.category, {"cases": 0, "passed": 0})
        bucket["cases"] += 1
        bucket["passed"] += int(result.passed)
    for bucket in by_category.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["cases"], 3)
    summary["by_category"] = by_category
    return summary
