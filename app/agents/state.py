"""Mutable state carried through one agent run."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from app.memory.long_term import MemoryRecord
from app.models.evidence import Citation, Evidence, Plan, ToolCallRecord


@dataclass
class AgentState:
    question: str
    user_id: str
    session_id: str
    run_id: str = ""
    mode: str = "agentic"

    short_term_context: str = ""
    memories: list[MemoryRecord] = field(default_factory=list)
    plan: Optional[Plan] = None
    evidence: list[Evidence] = field(default_factory=list)
    tool_records: list[ToolCallRecord] = field(default_factory=list)
    answer: str = ""
    citations: list[Citation] = field(default_factory=list)
    citation_stats: dict[str, Any] = field(default_factory=dict)
    memory_update: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    budget_snapshot: dict[str, int] = field(default_factory=dict)
    sufficiency: dict[str, Any] = field(default_factory=dict)
    rewrite: dict[str, Any] = field(default_factory=dict)
    insufficient_evidence: bool = False

    def next_evidence_id(self) -> str:
        return f"E{len(self.evidence) + 1}"

    def add_evidence(self, item: Evidence) -> Evidence:
        item.ensure_id(len(self.evidence) + 1)
        self.evidence.append(item)
        return item

    def evidence_by_id(self) -> dict[str, Evidence]:
        return {e.evidence_id: e for e in self.evidence}
