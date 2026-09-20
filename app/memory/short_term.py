"""Short-term memory: the current session.

Holds the last N turns verbatim plus a rolling summary of everything older. This is
*conversation state*, scoped to one session, and it is not the same thing as long-term
memory: nothing here is promoted across sessions unless the extractor decides it is worth
keeping.
"""
from __future__ import annotations

import uuid
from typing import Any, Optional

from app.database.db import Database
from app.llm.base import LLMClient, LLMError
from app.utils.dates import iso, utcnow
from app.utils.text import truncate

SUMMARY_SYSTEM = (
    "You compress a conversation between a user and a project-intelligence assistant into a short "
    "factual summary. Keep durable facts, decisions, named projects and stated priorities. "
    "Drop pleasantries, drop the assistant's reasoning, drop anything already superseded. "
    "Write at most 120 words of plain prose."
)


class ShortTermMemory:
    def __init__(self, db: Database, llm: Optional[LLMClient] = None, window: int = 8,
                 summarise_after: int = 12):
        self.db = db
        self.llm = llm
        self.window = window
        self.summarise_after = summarise_after

    async def ensure_session(self, session_id: str, user_id: str, title: str = "") -> str:
        now = iso(utcnow())
        existing = await self.db.query_one(
            "SELECT session_id, user_id FROM sessions WHERE session_id = ?", (session_id,)
        )
        if existing:
            if existing["user_id"] != user_id:
                # Session ids are namespaced per user; refuse to cross the boundary.
                raise PermissionError("Session belongs to a different user")
            await self.db.execute(
                "UPDATE sessions SET last_active_at = ? WHERE session_id = ?", (now, session_id)
            )
            return session_id
        await self.db.execute(
            "INSERT INTO sessions (session_id, user_id, title, created_at, last_active_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, truncate(title, 120), now, now),
        )
        return session_id

    async def add_message(self, session_id: str, user_id: str, role: str, content: str,
                          run_id: Optional[str] = None) -> str:
        message_id = f"msg_{uuid.uuid4().hex[:12]}"
        await self.db.execute(
            "INSERT INTO messages (message_id, session_id, user_id, role, content, run_id, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (message_id, session_id, user_id, role, content, run_id, iso(utcnow())),
        )
        return message_id

    async def recent_messages(self, session_id: str, limit: Optional[int] = None) -> list[dict[str, Any]]:
        rows = await self.db.query(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? "
            "ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (session_id, limit or self.window),
        )
        return list(reversed(rows))

    async def get_summary(self, session_id: str) -> str:
        row = await self.db.query_one(
            "SELECT summary FROM session_summaries WHERE session_id = ?", (session_id,)
        )
        return row["summary"] if row else ""

    async def context_block(self, session_id: str) -> str:
        """Rendered short-term context for the planner/synthesizer prompts."""
        summary = await self.get_summary(session_id)
        messages = await self.recent_messages(session_id)
        lines = []
        if summary:
            lines.append(f"[earlier in this session] {summary}")
        for message in messages:
            lines.append(f"{message['role']}: {truncate(message['content'], 400)}")
        return "\n".join(lines)

    async def maybe_summarise(self, session_id: str) -> bool:
        """Roll older turns into a summary once the session grows past the window."""
        row = await self.db.query_one(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?", (session_id,)
        )
        total = int(row["n"]) if row else 0
        if total < self.summarise_after:
            return False

        older = await self.db.query(
            "SELECT role, content, created_at FROM messages WHERE session_id = ? "
            "ORDER BY created_at ASC, rowid ASC LIMIT ?",
            (session_id, max(0, total - self.window)),
        )
        if not older:
            return False
        transcript = "\n".join(f"{m['role']}: {truncate(m['content'], 500)}" for m in older)

        summary = ""
        if self.llm is not None:
            try:
                response = await self.llm.complete(SUMMARY_SYSTEM, transcript, max_tokens=300)
                summary = response.text.strip()
            except LLMError:
                summary = ""
        if not summary:
            user_turns = [m["content"] for m in older if m["role"] == "user"]
            summary = "Earlier user questions: " + truncate(" | ".join(user_turns), 700)

        await self.db.execute(
            "INSERT INTO session_summaries (session_id, summary, covered_upto, updated_at) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(session_id) DO UPDATE SET summary=excluded.summary, "
            "covered_upto=excluded.covered_upto, updated_at=excluded.updated_at",
            (session_id, summary, older[-1]["created_at"], iso(utcnow())),
        )
        return True
