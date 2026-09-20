"""Agent orchestration: the full trajectory, traced end to end.

    question
      -> short-term context + long-term memory recall
      -> plan (dynamic decomposition)
      -> [no retrieval branch] conversational / memory-only answer
      -> tool selection + execution over MCP (multiple calls, multi-hop)
      -> evidence collection, scoring, gap retry
      -> synthesis with citations, validated against real evidence
      -> memory write-back (create / update / supersede / reject)
"""
from __future__ import annotations

from typing import Any, Optional

from app.agents.executor import Executor
from app.agents.planner import Planner
from app.agents.rewriter import QueryRewriter
from app.agents.state import AgentState
from app.agents.synthesizer import ConversationalResponder, Synthesizer
from app.config import Settings
from app.llm.base import LLMClient
from app.mcp_client.client import MCPToolClient
from app.memory.manager import MemoryManager
from app.memory.retriever import MemoryRetriever
from app.memory.short_term import ShortTermMemory
from app.models.evidence import Plan
from app.observability.tracing import RunTracer
from app.utils.text import truncate


class AgentOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings,
        mcp: MCPToolClient,
        planner: Planner,
        synthesizer: Synthesizer,
        memory: MemoryManager,
        short_term: ShortTermMemory,
        tracer_factory,
        llm: Optional[LLMClient] = None,
    ):
        self.settings = settings
        self.mcp = mcp
        self.planner = planner
        self.synthesizer = synthesizer
        self.memory = memory
        self.short_term = short_term
        self.tracer_factory = tracer_factory
        self.llm = llm
        self.responder = ConversationalResponder(llm)
        self.rewriter = QueryRewriter(llm)

    async def run(self, *, question: str, user_id: str, session_id: str) -> dict[str, Any]:
        tracer: RunTracer = self.tracer_factory()
        run_id = await tracer.start_run(user_id=user_id, question=question, session_id=session_id)
        state = AgentState(question=question, user_id=user_id, session_id=session_id, run_id=run_id)
        effective_question = question

        await self.short_term.ensure_session(session_id, user_id, title=truncate(question, 80))
        await self.short_term.add_message(session_id, user_id, "user", question, run_id)

        try:
            # 1. memory ------------------------------------------------------
            async with tracer.step("memory_retrieval", "memory", {"question": question}) as step:
                state.short_term_context = await self.short_term.context_block(session_id)
                state.memories = await self.memory.recall(user_id, question)
                step.set_output(
                    memories_used=len(state.memories),
                    memories=MemoryRetriever.as_trace(state.memories),
                    short_term_turns=state.short_term_context.count("\n") + 1 if state.short_term_context else 0,
                )

            # 2. query rewrite -----------------------------------------------
            # A follow-up turn ("and what about Apollo?") carries no subject of its own. The
            # planner must see a standalone question or it plans the wrong retrieval.
            known_projects = await self._known_projects()
            async with tracer.step("query_rewrite", "llm" if self.llm else "chain",
                                   {"question": question}) as step:
                rewrite = await self.rewriter.rewrite(
                    question,
                    session_context=state.short_term_context,
                    known_projects=known_projects,
                    memories=state.memories,
                )
                state.rewrite = rewrite.as_dict()
                effective_question = rewrite.rewritten
                step.set_output(**rewrite.as_dict())

            # 3. plan --------------------------------------------------------
            async with tracer.step("planning", "llm" if self.llm else "chain",
                                   {"question": effective_question}) as step:
                memory_block = MemoryRetriever.render(state.memories)
                plan: Plan = await self.planner.plan(
                    effective_question,
                    tool_catalogue=self.mcp.describe_tools(),
                    tool_names=self.mcp.tool_names(),
                    known_projects=known_projects,
                    memory_block=memory_block,
                    session_context=state.short_term_context,
                )
                state.plan = plan
                step.set_output(plan=plan.model_dump(mode="json"))

            # 4a. no-retrieval branch ---------------------------------------
            if not plan.needs_retrieval:
                async with tracer.step("conversational_response", "llm" if self.llm else "chain", {}) as step:
                    state.answer = await self.responder.respond(state, plan)
                    step.set_output(answer_chars=len(state.answer), strategy=plan.reasoning_strategy)
            else:
                # 4b. execute -----------------------------------------------
                executor = Executor(
                    self.mcp, tracer, llm=self.llm,
                    max_tool_calls=self.settings.agent_max_tool_calls,
                    max_rounds=self.settings.agent_max_rounds,
                    max_evidence=self.settings.agent_max_evidence,
                    evidence_chars=self.settings.agent_evidence_chars,
                )
                async with tracer.step("execution", "chain",
                                       {"sub_questions": len(plan.sub_questions)}) as step:
                    await executor.run(state, plan)
                    step.set_output(
                        tool_calls=len(state.tool_records),
                        evidence=len(state.evidence),
                        budget=state.budget_snapshot,
                        sufficiency=state.sufficiency,
                        warnings=state.warnings,
                    )

                # 5. synthesis ----------------------------------------------
                async with tracer.step("synthesis", "llm" if self.llm else "chain",
                                       {"evidence": len(state.evidence)}) as step:
                    # Synthesis answers the resolved question; the original is kept for display.
                    state.question = effective_question
                    await self.synthesizer.synthesize(state, plan)
                    state.question = question
                    step.set_output(
                        citation_stats=state.citation_stats,
                        answer_chars=len(state.answer),
                        insufficient_evidence=state.insufficient_evidence,
                    )

                cited = {c.marker for c in state.citations}
                await tracer.record_evidence(state.evidence, cited)

            # 6. memory write-back -------------------------------------------
            async with tracer.step("memory_update", "memory", {"question": question}) as step:
                state.memory_update = await self.memory.observe(
                    user_id=user_id, session_id=session_id, run_id=run_id,
                    user_message=question, answer_summary=truncate(state.answer, 800),
                )
                step.set_output(**state.memory_update)

            await self.short_term.add_message(session_id, user_id, "assistant", state.answer, run_id)
            await self.short_term.maybe_summarise(session_id)

            latency_ms = await tracer.finish_run(
                answer=state.answer, plan=state.plan.model_dump(mode="json") if state.plan else {},
                status="completed", tool_calls=len(state.tool_records),
                evidence_count=len(state.evidence), memories_used=len(state.memories),
            )
            return self._payload(state, latency_ms)

        except Exception as exc:  # noqa: BLE001 - a failed run must still be traced and answered
            message = f"{type(exc).__name__}: {exc}"
            await tracer.finish_run(
                answer="", plan=state.plan.model_dump(mode="json") if state.plan else {},
                status="failed", error=message, tool_calls=len(state.tool_records),
                evidence_count=len(state.evidence), memories_used=len(state.memories),
            )
            raise

    async def _known_projects(self) -> list[dict[str, Any]]:
        """Ground the planner in the projects that actually exist (one MCP call)."""
        result = await self.mcp.call("search_projects", {"limit": 25})
        if not result.get("ok"):
            return []
        return [
            {"key": item.get("data", {}).get("key", item.get("source_id")),
             "name": item.get("data", {}).get("name", "")}
            for item in result.get("items", [])
        ]

    @staticmethod
    def _payload(state: AgentState, latency_ms: int) -> dict[str, Any]:
        return {
            "run_id": state.run_id,
            "session_id": state.session_id,
            "question": state.question,
            "rewrite": state.rewrite,
            "answer": state.answer,
            "citations": [c.model_dump(mode="json") for c in state.citations],
            "plan": state.plan.model_dump(mode="json") if state.plan else {},
            "memories_used": MemoryRetriever.as_trace(state.memories),
            "memory_update": state.memory_update,
            "tool_calls": [r.model_dump(mode="json") for r in state.tool_records],
            "evidence": [
                {
                    "evidence_id": e.evidence_id, "source": e.source, "source_type": e.source_type,
                    "source_id": e.source_id, "title": e.title, "url": e.url,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "content": e.content, "relevance": e.relevance, "tool_used": e.tool_used,
                    "sub_question_id": e.sub_question_id,
                    "injection_suspected": e.injection_suspected,
                    "cited": e.evidence_id in {c.marker for c in state.citations},
                }
                for e in state.evidence
            ],
            "citation_stats": state.citation_stats,
            "sufficiency": state.sufficiency,
            "warnings": state.warnings,
            "budget": state.budget_snapshot,
            "insufficient_evidence": state.insufficient_evidence,
            "latency_ms": latency_ms,
        }
