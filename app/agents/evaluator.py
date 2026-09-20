"""Evidence sufficiency evaluation.

After a sub-question's tools have run, something has to decide whether what came back actually
answers it. Without this step the agent silently proceeds on thin evidence and the synthesizer
writes a confident-sounding answer over nothing.

The evaluator returns a verdict plus, when it can, a concrete suggestion for the retry round -
which tool to try and with what arguments. That suggestion is what makes the second round
useful rather than a repeat of the first.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from app.agents.prompts import SUFFICIENCY_SYSTEM
from app.llm.base import LLMClient, LLMError
from app.models.evidence import Evidence, SubQuestion
from app.security.sanitize import wrap_untrusted
from app.utils.text import GENERIC_QUESTION_WORDS, tokenize, truncate

# A sub-question is not considered answered on a single weak match.
MIN_TOP_RELEVANCE = 0.25
MIN_TERM_COVERAGE = 0.34


@dataclass
class Sufficiency:
    sufficient: bool
    missing: str = ""
    suggested_tool: str = ""
    suggested_arguments: dict[str, Any] = field(default_factory=dict)
    method: str = "heuristic"
    top_relevance: float = 0.0
    term_coverage: float = 0.0
    evidence_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "sufficient": self.sufficient, "missing": self.missing,
            "suggested_tool": self.suggested_tool, "suggested_arguments": self.suggested_arguments,
            "method": self.method, "top_relevance": round(self.top_relevance, 3),
            "term_coverage": round(self.term_coverage, 3), "evidence_count": self.evidence_count,
        }


class EvidenceEvaluator:
    def __init__(self, llm: Optional[LLMClient] = None, available_tools: Sequence[str] = ()):
        self.llm = llm
        self.available_tools = set(available_tools)

    async def assess(self, sub_question: SubQuestion, evidence: Sequence[Evidence]) -> Sufficiency:
        relevant = [e for e in evidence if e.sub_question_id == sub_question.id]
        verdict = self._assess_heuristic(sub_question, relevant)

        # The cheap check is authoritative when it finds nothing at all: there is nothing for a
        # model to judge, and calling one would only add latency.
        if self.llm is None or not relevant:
            return verdict
        try:
            return await self._assess_llm(sub_question, relevant, verdict)
        except (LLMError, ValueError, TypeError, KeyError):
            return verdict

    # --- deterministic -----------------------------------------------------
    def _assess_heuristic(self, sub_question: SubQuestion, relevant: Sequence[Evidence]) -> Sufficiency:
        if not relevant:
            return Sufficiency(
                sufficient=False, missing="no evidence was returned for this sub-question",
                method="heuristic", evidence_count=0,
            )

        top = max(e.relevance for e in relevant)
        terms = {
            t for t in tokenize(sub_question.question)
            if len(t) > 3 and t not in GENERIC_QUESTION_WORDS and not t.isdigit()
        }
        corpus = " ".join(f"{e.title} {e.content}" for e in relevant).lower()
        coverage = (sum(1 for t in terms if t in corpus) / len(terms)) if terms else 1.0

        # Evidence that only survives because an injection attempt was downranked is not evidence.
        usable = [e for e in relevant if not e.injection_suspected]
        if not usable:
            return Sufficiency(
                sufficient=False,
                missing="the only matching content was flagged as an injection attempt",
                method="heuristic", top_relevance=top, term_coverage=coverage,
                evidence_count=len(relevant),
            )

        sufficient = top >= MIN_TOP_RELEVANCE and coverage >= MIN_TERM_COVERAGE
        missing = ""
        if not sufficient:
            uncovered = sorted(t for t in terms if t not in corpus)[:4]
            missing = (f"weak match (top relevance {top:.2f}, term coverage {coverage:.0%})"
                       + (f"; nothing found about: {', '.join(uncovered)}" if uncovered else ""))
        return Sufficiency(
            sufficient=sufficient, missing=missing, method="heuristic",
            top_relevance=top, term_coverage=coverage, evidence_count=len(relevant),
        )

    # --- model-judged ------------------------------------------------------
    async def _assess_llm(
        self, sub_question: SubQuestion, relevant: Sequence[Evidence], fallback: Sufficiency
    ) -> Sufficiency:
        block = "\n\n".join(
            f"[{e.evidence_id}] {e.source_type} {e.source_id} (via {e.tool_used})\n"
            f"{wrap_untrusted(truncate(e.content, 500), e.source_id)}"
            for e in sorted(relevant, key=lambda e: e.relevance, reverse=True)[:10]
        )
        payload = (
            f"SUB-QUESTION: {sub_question.question}\n"
            f"AVAILABLE TOOLS: {', '.join(sorted(self.available_tools))}\n\n"
            f"COLLECTED EVIDENCE:\n{block}"
        )
        data = await self.llm.complete_json(SUFFICIENCY_SYSTEM, payload, max_tokens=400)
        suggested_tool = str(data.get("suggested_tool", "") or "")
        if suggested_tool not in self.available_tools:
            suggested_tool = ""
        arguments = data.get("suggested_arguments")
        return Sufficiency(
            sufficient=bool(data.get("sufficient", fallback.sufficient)),
            missing=str(data.get("missing", ""))[:300],
            suggested_tool=suggested_tool,
            suggested_arguments=arguments if isinstance(arguments, dict) else {},
            method="llm",
            top_relevance=fallback.top_relevance,
            term_coverage=fallback.term_coverage,
            evidence_count=len(relevant),
        )
