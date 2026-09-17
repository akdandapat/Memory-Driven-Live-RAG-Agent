"""Thin async wrapper over SQLite.

SQLite keeps the project runnable with zero infrastructure. Every call goes through
asyncio.to_thread so the FastAPI event loop is never blocked. The SQL is plain and
portable enough to move to Postgres (+pgvector) later; see README 'Scaling'.
"""
from __future__ import annotations

import asyncio
import sqlite3
import threading
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class Database:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    # --- lifecycle ---------------------------------------------------------
    def init_schema_sync(self) -> None:
        with self._lock:
            self._conn.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
            self._conn.commit()

    async def init_schema(self) -> None:
        await asyncio.to_thread(self.init_schema_sync)

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # --- sync primitives ---------------------------------------------------
    def execute_sync(self, sql: str, params: Sequence[Any] = ()) -> int:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur.rowcount

    def executemany_sync(self, sql: str, seq: Iterable[Sequence[Any]]) -> int:
        with self._lock:
            cur = self._conn.executemany(sql, [tuple(p) for p in seq])
            self._conn.commit()
            return cur.rowcount

    def query_sync(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            return [dict(row) for row in cur.fetchall()]

    def query_one_sync(self, sql: str, params: Sequence[Any] = ()) -> Optional[dict[str, Any]]:
        rows = self.query_sync(sql, params)
        return rows[0] if rows else None

    # --- async wrappers ----------------------------------------------------
    async def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        return await asyncio.to_thread(self.execute_sync, sql, params)

    async def executemany(self, sql: str, seq: Iterable[Sequence[Any]]) -> int:
        return await asyncio.to_thread(self.executemany_sync, sql, list(seq))

    async def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self.query_sync, sql, params)

    async def query_one(self, sql: str, params: Sequence[Any] = ()) -> Optional[dict[str, Any]]:
        return await asyncio.to_thread(self.query_one_sync, sql, params)
