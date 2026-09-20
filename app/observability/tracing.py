"""Run tracing.

Every agent run writes a full operational trace: memory retrieval, the generated plan, each
sub-question, every tool call with its arguments and result size, the evidence that survived
filtering, the synthesis step and the memory write-back. What is *not* recorded is hidden
chain-of-thought - the trace holds operations, artefacts and short decision summaries.

Secrets are redacted on the way in, so neither SQLite nor LangSmith ever sees a token.
"""
from __future__ import annotations

import json
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Optional

from app.database.db import Database
from app.models.evidence import Evidence, ToolCallRecord
from app.observability.langsmith_exporter import LangSmithExporter
from app.security.sanitize import redact_secrets
from app.utils.dates import iso, parse_dt, utcnow


def _dumps(value: Any) -> str:
    try:
        return json.dumps(redact_secrets(value), default=str)[:200_000]
    except (TypeError, ValueError):
        return json.dumps({"unserialisable": str(type(value))})


@dataclass
class StepHandle:
    step_id: str
    name: str
    output: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set_output(self, **kwargs: Any) -> None:
        self.output.update(kwargs)


class RunTracer:
    def __init__(self, db: Database, exporter: Optional[LangSmithExporter] = None):
        self.db = db
        self.exporter = exporter
        self.run_id = ""
        self._ordinal = 0
        self._started = 0.0

    async def start_run(
        self, *, user_id: str, question: str, session_id: Optional[str] = None,
        mode: str = "agentic",
    ) -> str:
        self.run_id = f"run_{uuid.uuid4().hex[:16]}"
        self._started = time.perf_counter()
        await self.db.execute(
            "INSERT INTO agent_runs (run_id, session_id, user_id, question, mode, status, started_at) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?)",
            (self.run_id, session_id, user_id, question, mode, iso(utcnow())),
        )
        if self.exporter:
            self.exporter.start_run(
                run_id=self.run_id, name=f"agentic-rag:{mode}", run_type="chain",
                inputs={"question": question, "user_id": user_id, "session_id": session_id},
            )
        return self.run_id

    async def finish_run(
        self, *, answer: str, plan: Optional[dict[str, Any]] = None, status: str = "completed",
        error: str = "", tool_calls: int = 0, evidence_count: int = 0, memories_used: int = 0,
    ) -> int:
        latency_ms = int((time.perf_counter() - self._started) * 1000)
        await self.db.execute(
            "UPDATE agent_runs SET status = ?, answer = ?, plan_json = ?, finished_at = ?, "
            "latency_ms = ?, tool_calls = ?, evidence_count = ?, memories_used = ?, error = ? "
            "WHERE run_id = ?",
            (status, answer, _dumps(plan or {}), iso(utcnow()), latency_ms, tool_calls,
             evidence_count, memories_used, error, self.run_id),
        )
        if self.exporter:
            self.exporter.end_run(
                run_id=self.run_id,
                outputs={"answer": answer, "tool_calls": tool_calls, "evidence": evidence_count,
                         "latency_ms": latency_ms},
                error=error,
            )
        return latency_ms

    @asynccontextmanager
    async def step(self, name: str, step_type: str, inputs: Any = None) -> AsyncIterator[StepHandle]:
        self._ordinal += 1
        ordinal = self._ordinal
        step_id = f"step_{uuid.uuid4().hex[:12]}"
        handle = StepHandle(step_id=step_id, name=name)
        started = time.perf_counter()
        started_at = iso(utcnow())
        try:
            yield handle
        except Exception as exc:  # noqa: BLE001 - record then re-raise
            handle.status = "error"
            handle.set_output(error=f"{type(exc).__name__}: {exc}")
            await self._write_step(step_id, ordinal, name, step_type, handle, inputs, started, started_at)
            raise
        await self._write_step(step_id, ordinal, name, step_type, handle, inputs, started, started_at)

    async def _write_step(self, step_id: str, ordinal: int, name: str, step_type: str,
                          handle: StepHandle, inputs: Any, started: float, started_at: str) -> None:
        if self.exporter:
            self.exporter.record_child(
                child_id=step_id, name=name,
                run_type={"llm": "llm", "tool": "tool", "retriever": "retriever"}.get(step_type, "chain"),
                inputs=inputs, outputs=handle.output,
                start_time=parse_dt(started_at), end_time=utcnow(),
                error=str(handle.output.get("error", "")) if handle.status == "error" else "",
            )
        await self.db.execute(
            "INSERT INTO run_steps (step_id, run_id, ordinal, name, step_type, status, input_json, "
            "output_json, started_at, ended_at, duration_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (step_id, self.run_id, ordinal, name, step_type, handle.status, _dumps(inputs),
             _dumps(handle.output), started_at, iso(utcnow()), int((time.perf_counter() - started) * 1000)),
        )

    async def record_tool_call(self, record: ToolCallRecord) -> None:
        if self.exporter:
            self.exporter.record_child(
                child_id=record.call_id, name=f"tool:{record.tool_name}", run_type="tool",
                inputs=record.arguments,
                outputs={"status": record.status, "result_count": record.result_count,
                         "duration_ms": record.duration_ms},
                start_time=record.started_at, error=record.error,
            )
        await self.db.execute(
            "INSERT INTO tool_calls (call_id, run_id, sub_question_id, round_index, tool_name, "
            "arguments_json, status, result_count, duration_ms, error, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (record.call_id, self.run_id, record.sub_question_id, record.round_index,
             record.tool_name, _dumps(record.arguments), record.status, record.result_count,
             record.duration_ms, record.error, iso(record.started_at)),
        )

    async def record_evidence(self, evidence: list[Evidence], cited_ids: Optional[set[str]] = None) -> None:
        if not evidence:
            return
        cited_ids = cited_ids or set()
        await self.db.executemany(
            "INSERT OR REPLACE INTO evidence (evidence_id, run_id, source, source_type, source_id, "
            "title, url, timestamp, content, relevance, tool_used, sub_question_id, project_key, "
            "injection_suspected, cited) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (e.evidence_id, self.run_id, e.source, e.source_type, e.source_id, e.title, e.url,
                 iso(e.timestamp), e.content, e.relevance, e.tool_used, e.sub_question_id,
                 e.project_key, int(e.injection_suspected), int(e.evidence_id in cited_ids))
                for e in evidence
            ],
        )

    async def mark_cited(self, cited_ids: set[str]) -> None:
        if not cited_ids:
            return
        await self.db.executemany(
            "UPDATE evidence SET cited = 1 WHERE run_id = ? AND evidence_id = ?",
            [(self.run_id, evidence_id) for evidence_id in cited_ids],
        )


class TraceReader:
    """Read side used by GET /api/trace/{run_id} and the UI trace panel."""

    def __init__(self, db: Database):
        self.db = db

    async def get_trace(self, run_id: str, user_id: Optional[str] = None) -> Optional[dict[str, Any]]:
        run = await self.db.query_one("SELECT * FROM agent_runs WHERE run_id = ?", (run_id,))
        if run is None:
            return None
        if user_id and run["user_id"] != user_id:
            return None
        steps = await self.db.query(
            "SELECT ordinal, name, step_type, status, input_json, output_json, started_at, "
            "duration_ms FROM run_steps WHERE run_id = ? ORDER BY ordinal ASC", (run_id,)
        )
        tools = await self.db.query(
            "SELECT call_id, tool_name, arguments_json, status, result_count, duration_ms, error, "
            "sub_question_id, round_index FROM tool_calls WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,)
        )
        evidence = await self.db.query(
            "SELECT evidence_id, source, source_type, source_id, title, url, timestamp, content, "
            "relevance, tool_used, sub_question_id, injection_suspected, cited FROM evidence "
            "WHERE run_id = ? ORDER BY relevance DESC", (run_id,)
        )
        memory_events = await self.db.query(
            "SELECT action, memory_id, reason, content, created_at FROM memory_events "
            "WHERE run_id = ? ORDER BY created_at ASC", (run_id,)
        )

        def _loads(value: str) -> Any:
            try:
                return json.loads(value or "{}")
            except json.JSONDecodeError:
                return {}

        return {
            "run": {
                "run_id": run["run_id"], "session_id": run["session_id"], "user_id": run["user_id"],
                "question": run["question"], "mode": run["mode"], "status": run["status"],
                "answer": run["answer"], "started_at": run["started_at"],
                "finished_at": run["finished_at"], "latency_ms": run["latency_ms"],
                "tool_calls": run["tool_calls"], "evidence_count": run["evidence_count"],
                "memories_used": run["memories_used"], "error": run["error"],
                "plan": _loads(run["plan_json"]),
            },
            "steps": [
                {**step, "input": _loads(step.pop("input_json")), "output": _loads(step.pop("output_json"))}
                for step in steps
            ],
            "tool_calls": [
                {**tool, "arguments": _loads(tool.pop("arguments_json"))} for tool in tools
            ],
            "evidence": evidence,
            "memory_events": memory_events,
        }

    async def recent_runs(self, user_id: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.db.query(
            "SELECT run_id, question, mode, status, started_at, latency_ms, tool_calls, "
            "evidence_count FROM agent_runs WHERE user_id = ? ORDER BY started_at DESC LIMIT ?",
            (user_id, limit),
        )
