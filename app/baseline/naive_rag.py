"""The 'basic RAG' comparison baseline.

Deliberately naive and honest about it: one embedding of the raw question, top-k chunks from
the vector index, one generation. No decomposition, no tool calls, no structured filters, no
memory, no evidence validation. It exists so the demo can show *why* the agentic path is
different rather than just asserting it.
"""
from __future__ import annotations

import time
from typing import Any, Optional

from app.agents.prompts import BASELINE_SYSTEM
from app.database.db import Database
from app.llm.base import LLMClient, LLMError
from app.rag.retriever import SemanticRetriever
from app.security.sanitize import wrap_untrusted
from app.utils.dates import iso, utcnow
from app.utils.text import truncate


class NaiveRAG:
    def __init__(self, retriever: SemanticRetriever, db: Database, llm: Optional[LLMClient] = None,
                 top_k: int = 6):
        self.retriever = retriever
        self.db = db
        self.llm = llm
        self.top_k = top_k

    async def answer(self, question: str, user_id: str = "", session_id: Optional[str] = None) -> dict[str, Any]:
        started = time.perf_counter()
        run_id = f"base_{int(started * 1000) % 10**12:012d}"
        await self.db.execute(
            "INSERT INTO agent_runs (run_id, session_id, user_id, question, mode, status, started_at) "
            "VALUES (?, ?, ?, ?, 'baseline', 'running', ?)",
            (run_id, session_id, user_id or "anonymous", question, iso(utcnow())),
        )
        chunks = await self.retriever.retrieve(question, top_k=self.top_k, tool_used="baseline_vector_search")
        context = "\n\n".join(
            f"[{index + 1}] {wrap_untrusted(truncate(c.content, 700), c.source_id)}"
            for index, c in enumerate(chunks)
        )

        if self.llm is not None and chunks:
            try:
                response = await self.llm.complete(
                    BASELINE_SYSTEM, f"SNIPPETS:\n{context}\n\nQUESTION: {question}", max_tokens=600
                )
                answer = response.text.strip()
            except LLMError as exc:
                answer = f"Baseline generation failed: {exc}"
        elif chunks:
            answer = ("Top matching snippets (no LLM configured, so this baseline returns raw "
                      "retrieval):\n\n" + "\n\n".join(
                          f"- {truncate(c.content, 260)}" for c in chunks))
        else:
            answer = "No matching text was found in the vector index."

        latency_ms = int((time.perf_counter() - started) * 1000)
        await self.db.execute(
            "UPDATE agent_runs SET status = 'completed', answer = ?, finished_at = ?, latency_ms = ?, "
            "tool_calls = 0, evidence_count = ? WHERE run_id = ?",
            (answer, iso(utcnow()), latency_ms, len(chunks), run_id),
        )
        return {
            "run_id": run_id,
            "mode": "baseline",
            "answer": answer,
            "retrieved_chunks": [
                {"source_id": c.source_id, "url": c.url, "score": c.relevance,
                 "snippet": truncate(c.content, 240)}
                for c in chunks
            ],
            "tool_calls": 0,
            "memories_used": 0,
            "citations": [],
            "latency_ms": latency_ms,
        }
