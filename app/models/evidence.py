"""Evidence and citation records - the backbone of grounded, traceable answers."""
from __future__ import annotations

import hashlib
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    evidence_id: str = ""
    source: str = "jira"                 # jira | jira_index | memory
    source_type: str = ""                # issue | comment | changelog | project | chunk
    source_id: str = ""                  # e.g. ATLAS-12, ATLAS-12/comment/9001
    title: str = ""
    url: str = ""
    timestamp: Optional[datetime] = None
    content: str = ""
    relevance: float = 0.0
    tool_used: str = ""
    sub_question_id: str = ""
    project_key: str = ""
    injection_suspected: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def fingerprint(self) -> str:
        base = f"{self.source}|{self.source_type}|{self.source_id}|{self.content[:200]}"
        return hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]

    def ensure_id(self, index: int) -> "Evidence":
        if not self.evidence_id:
            self.evidence_id = f"E{index}"
        return self


class Citation(BaseModel):
    marker: str                # "E3"
    source: str
    source_type: str
    source_id: str
    title: str = ""
    url: str = ""
    timestamp: Optional[datetime] = None
    tool_used: str = ""
    snippet: str = ""


class SubQuestion(BaseModel):
    id: str
    question: str
    rationale: str = ""
    tool_hints: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    status: str = "pending"      # pending | answered | insufficient
    evidence_ids: list[str] = Field(default_factory=list)


class Plan(BaseModel):
    goal: str
    needs_retrieval: bool = True
    reasoning_strategy: str = "single_hop"
    sub_questions: list[SubQuestion] = Field(default_factory=list)
    required_tools: list[str] = Field(default_factory=list)
    entities: dict[str, Any] = Field(default_factory=dict)
    time_range: dict[str, Optional[str]] = Field(default_factory=dict)
    generated_by: str = "heuristic"      # heuristic | llm
    notes: str = ""


class ToolCallRecord(BaseModel):
    call_id: str
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: str = "ok"           # ok | error | skipped_duplicate | budget_exceeded
    duration_ms: int = 0
    result_count: int = 0
    error: str = ""
    sub_question_id: str = ""
    round_index: int = 0
    started_at: datetime = Field(default_factory=datetime.utcnow)
