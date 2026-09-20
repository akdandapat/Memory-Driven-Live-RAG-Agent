"""Long-term memory store: structured, embedded, supersedable records."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from app.database.db import Database
from app.utils.dates import iso, parse_dt, utcnow

MEMORY_TYPES = ("preference", "context", "fact", "goal")


@dataclass
class MemoryRecord:
    memory_id: str
    user_id: str
    type: str
    content: str
    subject: str = ""
    importance: float = 0.5
    confidence: float = 0.5
    status: str = "active"
    superseded_by: Optional[str] = None
    source_session: Optional[str] = None
    source_run: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""
    expires_at: Optional[str] = None
    last_used_at: Optional[str] = None
    use_count: int = 0
    score: float = 0.0
    score_parts: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data = {
            "memory_id": self.memory_id, "user_id": self.user_id, "type": self.type,
            "content": self.content, "subject": self.subject, "importance": self.importance,
            "confidence": self.confidence, "status": self.status,
            "superseded_by": self.superseded_by, "source_session": self.source_session,
            "source_run": self.source_run, "created_at": self.created_at,
            "updated_at": self.updated_at, "expires_at": self.expires_at,
            "last_used_at": self.last_used_at, "use_count": self.use_count,
        }
        if self.score:
            data["score"] = round(self.score, 4)
            data["score_parts"] = {k: round(v, 4) for k, v in self.score_parts.items()}
        return data


def row_to_record(row: dict[str, Any]) -> MemoryRecord:
    return MemoryRecord(
        memory_id=row["memory_id"], user_id=row["user_id"], type=row["type"],
        content=row["content"], subject=row.get("subject", "") or "",
        importance=float(row.get("importance", 0.5)), confidence=float(row.get("confidence", 0.5)),
        status=row.get("status", "active"), superseded_by=row.get("superseded_by"),
        source_session=row.get("source_session"), source_run=row.get("source_run"),
        created_at=row.get("created_at", ""), updated_at=row.get("updated_at", ""),
        expires_at=row.get("expires_at"), last_used_at=row.get("last_used_at"),
        use_count=int(row.get("use_count", 0) or 0),
    )


class LongTermMemoryStore:
    def __init__(self, db: Database, embedding_model: str):
        self.db = db
        self.embedding_model = embedding_model

    async def create(
        self,
        *,
        user_id: str,
        type: str,
        content: str,
        subject: str = "",
        importance: float = 0.5,
        confidence: float = 0.6,
        session_id: Optional[str] = None,
        run_id: Optional[str] = None,
        expires_at: Optional[str] = None,
        embedding: Optional[np.ndarray] = None,
    ) -> MemoryRecord:
        if type not in MEMORY_TYPES:
            type = "context"
        now = iso(utcnow()) or ""
        memory_id = f"mem_{uuid.uuid4().hex[:12]}"
        await self.db.execute(
            """
            INSERT INTO memories (memory_id, user_id, type, content, subject, importance, confidence,
                                  status, source_session, source_run, created_at, updated_at,
                                  expires_at, use_count, embedding, embedding_model)
            VALUES (?, ?, ?, ?, ?, ?, ?, 'active', ?, ?, ?, ?, ?, 0, ?, ?)
            """,
            (memory_id, user_id, type, content, subject, float(importance), float(confidence),
             session_id, run_id, now, now, expires_at,
             embedding.astype(np.float32).tobytes() if embedding is not None else None,
             self.embedding_model),
        )
        return MemoryRecord(memory_id=memory_id, user_id=user_id, type=type, content=content,
                            subject=subject, importance=importance, confidence=confidence,
                            source_session=session_id, source_run=run_id, created_at=now,
                            updated_at=now, expires_at=expires_at)

    async def update_content(
        self, memory_id: str, content: str, *, importance: Optional[float] = None,
        confidence: Optional[float] = None, embedding: Optional[np.ndarray] = None,
        run_id: Optional[str] = None,
    ) -> None:
        await self.db.execute(
            "UPDATE memories SET content = ?, updated_at = ?, "
            "importance = COALESCE(?, importance), confidence = COALESCE(?, confidence), "
            "embedding = COALESCE(?, embedding), source_run = COALESCE(?, source_run) "
            "WHERE memory_id = ?",
            (content, iso(utcnow()), importance, confidence,
             embedding.astype(np.float32).tobytes() if embedding is not None else None,
             run_id, memory_id),
        )

    async def supersede(self, old_id: str, new_id: str) -> None:
        await self.db.execute(
            "UPDATE memories SET status = 'superseded', superseded_by = ?, updated_at = ? "
            "WHERE memory_id = ?",
            (new_id, iso(utcnow()), old_id),
        )

    async def touch(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        now = iso(utcnow())
        await self.db.executemany(
            "UPDATE memories SET last_used_at = ?, use_count = use_count + 1 WHERE memory_id = ?",
            [(now, memory_id) for memory_id in memory_ids],
        )

    async def delete(self, user_id: str, memory_id: str) -> bool:
        rows = await self.db.execute(
            "DELETE FROM memories WHERE memory_id = ? AND user_id = ?", (memory_id, user_id)
        )
        return bool(rows)

    async def expire_due(self) -> int:
        return await self.db.execute(
            "UPDATE memories SET status = 'expired' WHERE status = 'active' "
            "AND expires_at IS NOT NULL AND expires_at < ?",
            (iso(utcnow()),),
        )

    async def active(self, user_id: str, limit: int = 500) -> list[tuple[MemoryRecord, Optional[np.ndarray]]]:
        rows = await self.db.query(
            "SELECT * FROM memories WHERE user_id = ? AND status = 'active' "
            "ORDER BY updated_at DESC LIMIT ?",
            (user_id, limit),
        )
        out = []
        for row in rows:
            vector = np.frombuffer(row["embedding"], dtype=np.float32) if row["embedding"] else None
            out.append((row_to_record(row), vector))
        return out

    async def list_all(self, user_id: str, include_inactive: bool = False, limit: int = 200) -> list[MemoryRecord]:
        sql = "SELECT * FROM memories WHERE user_id = ?"
        params: list[Any] = [user_id]
        if not include_inactive:
            sql += " AND status = 'active'"
        sql += " ORDER BY updated_at DESC LIMIT ?"
        params.append(limit)
        return [row_to_record(row) for row in await self.db.query(sql, params)]

    async def count(self, user_id: str) -> int:
        row = await self.db.query_one(
            "SELECT COUNT(*) AS n FROM memories WHERE user_id = ? AND status = 'active'", (user_id,)
        )
        return int(row["n"]) if row else 0

    async def prune_to_limit(self, user_id: str, maximum: int) -> int:
        """Keep the store bounded: drop the least important, least used, oldest records."""
        total = await self.count(user_id)
        if total <= maximum:
            return 0
        excess = total - maximum
        rows = await self.db.query(
            "SELECT memory_id FROM memories WHERE user_id = ? AND status = 'active' "
            "ORDER BY importance ASC, use_count ASC, updated_at ASC LIMIT ?",
            (user_id, excess),
        )
        for row in rows:
            await self.db.execute(
                "UPDATE memories SET status = 'expired', updated_at = ? WHERE memory_id = ?",
                (iso(utcnow()), row["memory_id"]),
            )
        return len(rows)

    async def log_event(self, *, run_id: Optional[str], user_id: str, memory_id: Optional[str],
                        action: str, reason: str = "", content: str = "") -> None:
        await self.db.execute(
            "INSERT INTO memory_events (event_id, run_id, user_id, memory_id, action, reason, content, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (f"mev_{uuid.uuid4().hex[:12]}", run_id, user_id, memory_id, action, reason,
             content[:500], iso(utcnow())),
        )

    async def events_for_run(self, run_id: str) -> list[dict[str, Any]]:
        return await self.db.query(
            "SELECT action, memory_id, reason, content, created_at FROM memory_events "
            "WHERE run_id = ? ORDER BY created_at ASC",
            (run_id,),
        )

    @staticmethod
    def age_days(record: MemoryRecord) -> float:
        created = parse_dt(record.updated_at or record.created_at)
        if created is None:
            return 999.0
        return max(0.0, (utcnow() - created).total_seconds() / 86400.0)

    @staticmethod
    def dumps_meta(data: dict[str, Any]) -> str:
        return json.dumps(data, default=str)
