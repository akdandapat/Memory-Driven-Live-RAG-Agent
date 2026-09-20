"""Request/response models for the HTTP API."""
from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: Optional[str] = Field(default=None, max_length=64)
    user_id: Optional[str] = Field(default=None, max_length=64)
    compare_baseline: bool = False


class ChatResponse(BaseModel):
    run_id: str
    session_id: str
    question: str = ""
    rewrite: dict[str, Any] = Field(default_factory=dict)
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    plan: dict[str, Any] = Field(default_factory=dict)
    memories_used: list[dict[str, Any]] = Field(default_factory=list)
    memory_update: dict[str, Any] = Field(default_factory=dict)
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    citation_stats: dict[str, Any] = Field(default_factory=dict)
    sufficiency: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)
    insufficient_evidence: bool = False
    latency_ms: int = 0
    baseline: Optional[dict[str, Any]] = None


class SyncRequest(BaseModel):
    project_keys: Optional[list[str]] = None
    full: bool = False
    include_comments: bool = True
    max_issues_per_project: int = Field(default=300, ge=1, le=2000)


class MemoryCreateRequest(BaseModel):
    content: str = Field(min_length=3, max_length=500)
    type: str = "context"
    subject: str = ""
    importance: float = Field(default=0.6, ge=0.0, le=1.0)
    user_id: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    app_mode: str
    source: dict[str, Any]
    mcp: dict[str, Any]
    llm: dict[str, Any]
    embeddings: dict[str, Any]
    index: dict[str, Any]
    observability: dict[str, Any]
