"""HTTP API.

POST /api/chat          run the agent (optionally alongside the naive-RAG baseline)
POST /api/sync          pull changes from the live source into the local index
POST /api/ingest        alias for a full re-index
GET  /api/projects      projects visible through the connector
GET  /api/trace/{id}    the full operational trace for one run
GET  /api/runs          recent runs for a user
GET  /api/memory        long-term memory records
POST /api/memory        add a memory explicitly
DELETE /api/memory/{id} forget one record
GET  /api/health        component status
"""
from __future__ import annotations

import logging
import uuid
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Query, Request

from app.api.schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    MemoryCreateRequest,
    SyncRequest,
)
from app.security.guards import InvalidArgument, validate_user_id

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")


def _services(request: Request):
    services = getattr(request.app.state, "services", None)
    if services is None:
        raise HTTPException(status_code=503, detail="Services are not ready")
    return services


def _user_id(services, requested: Optional[str]) -> str:
    try:
        return validate_user_id(requested or services.settings.default_user_id)
    except InvalidArgument as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/chat", response_model=ChatResponse)
async def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    services = _services(request)
    user_id = _user_id(services, payload.user_id)
    session_id = payload.session_id or f"sess_{uuid.uuid4().hex[:12]}"

    try:
        result = await services.orchestrator.run(
            question=payload.message, user_id=user_id, session_id=session_id
        )
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - surfaced as 500 with a safe message
        logger.exception("Agent run failed")
        raise HTTPException(status_code=500, detail=f"Agent run failed: {type(exc).__name__}") from exc

    if payload.compare_baseline:
        try:
            result["baseline"] = await services.baseline.answer(payload.message, user_id, session_id)
        except Exception as exc:  # noqa: BLE001 - the baseline is a demo aid, never fatal
            result["baseline"] = {"error": f"{type(exc).__name__}: {exc}"}
    return ChatResponse(**result)


@router.post("/sync")
async def sync(payload: SyncRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    stats = await services.ingestion.sync(
        payload.project_keys, full=payload.full,
        max_issues_per_project=payload.max_issues_per_project,
        include_comments=payload.include_comments,
    )
    return {"ok": not stats.errors, "stats": stats.as_dict(),
            "index": await services.ingestion.index_stats()}


@router.post("/ingest")
async def ingest(request: Request, project_keys: Optional[list[str]] = None) -> dict[str, Any]:
    services = _services(request)
    stats = await services.ingestion.sync(project_keys, full=True)
    return {"ok": not stats.errors, "stats": stats.as_dict(),
            "index": await services.ingestion.index_stats()}


@router.get("/projects")
async def projects(request: Request) -> dict[str, Any]:
    services = _services(request)
    result = await services.mcp.call("search_projects", {"limit": 50})
    if not result.get("ok"):
        raise HTTPException(status_code=502, detail=result.get("error", {}).get("message", "source error"))
    return {"count": result.get("count", 0),
            "projects": [item.get("data", {}) | {"url": item.get("url", "")}
                         for item in result.get("items", [])]}


@router.get("/trace/{run_id}")
async def trace(run_id: str, request: Request, user_id: Optional[str] = None) -> dict[str, Any]:
    services = _services(request)
    resolved = _user_id(services, user_id)
    result = await services.trace_reader.get_trace(run_id, resolved)
    if result is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return result


@router.get("/runs")
async def runs(request: Request, user_id: Optional[str] = None,
               limit: int = Query(default=20, ge=1, le=100)) -> dict[str, Any]:
    services = _services(request)
    resolved = _user_id(services, user_id)
    return {"runs": await services.trace_reader.recent_runs(resolved, limit)}


@router.get("/memory")
async def list_memory(request: Request, user_id: Optional[str] = None,
                      include_inactive: bool = False) -> dict[str, Any]:
    services = _services(request)
    resolved = _user_id(services, user_id)
    records = await services.memory.list_memories(resolved, include_inactive)
    return {"user_id": resolved, "count": len(records), "memories": records}


@router.post("/memory")
async def create_memory(payload: MemoryCreateRequest, request: Request) -> dict[str, Any]:
    services = _services(request)
    resolved = _user_id(services, payload.user_id)
    vectors = await services.embeddings.embed_documents([payload.content])
    record = await services.memory.store.create(
        user_id=resolved, type=payload.type, content=payload.content, subject=payload.subject,
        importance=payload.importance, confidence=0.9, embedding=vectors[0],
    )
    await services.memory.store.log_event(
        run_id=None, user_id=resolved, memory_id=record.memory_id, action="created",
        reason="created explicitly through the API", content=payload.content,
    )
    return {"ok": True, "memory": record.as_dict()}


@router.delete("/memory/{memory_id}")
async def delete_memory(memory_id: str, request: Request,
                        user_id: Optional[str] = None) -> dict[str, Any]:
    services = _services(request)
    resolved = _user_id(services, user_id)
    deleted = await services.memory.delete(resolved, memory_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Memory not found for this user")
    return {"ok": True, "deleted": memory_id}


@router.get("/tools")
async def tools(request: Request) -> dict[str, Any]:
    """The live MCP tool catalogue, as discovered over the protocol."""
    services = _services(request)
    return {
        "transport": services.settings.mcp_transport,
        "tools": [
            {"name": spec.name, "signature": spec.signature(), "description": spec.description,
             "input_schema": spec.input_schema}
            for spec in services.mcp.tools.values()
        ],
    }


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    services = _services(request)
    return HealthResponse(**await services.status())
