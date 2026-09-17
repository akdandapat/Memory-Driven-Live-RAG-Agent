"""Vector index over the local cache.

Storage is SQLite (`chunk_embeddings`) plus an in-process numpy matrix for cosine search.
That is honest about scale: it is exact search over thousands of chunks, not a distributed
ANN engine. The interface (`upsert` / `search` / `drop_document`) is deliberately the same
shape as pgvector or Qdrant so the swap is a single class - see README 'Scaling'.
"""
from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import numpy as np

from app.database.db import Database


@dataclass
class VectorHit:
    chunk_id: str
    doc_id: str
    score: float
    text: str
    metadata: dict[str, Any]


class VectorIndex:
    def __init__(self, db: Database, model_name: str, dim: int):
        self.db = db
        self.model_name = model_name
        self.dim = dim
        self._matrix: Optional[np.ndarray] = None
        self._rows: list[dict[str, Any]] = []
        self._dirty = True
        self._lock = asyncio.Lock()

    def invalidate(self) -> None:
        self._dirty = True

    async def upsert(self, chunk_ids: Sequence[str], vectors: np.ndarray) -> None:
        if len(chunk_ids) == 0:
            return
        payload = [
            (chunk_id, self.model_name, int(vectors.shape[1]), vectors[i].astype(np.float32).tobytes())
            for i, chunk_id in enumerate(chunk_ids)
        ]
        await self.db.executemany(
            "INSERT INTO chunk_embeddings (chunk_id, model, dim, vector) VALUES (?, ?, ?, ?) "
            "ON CONFLICT(chunk_id) DO UPDATE SET model=excluded.model, dim=excluded.dim, vector=excluded.vector",
            payload,
        )
        self.invalidate()

    async def _load(self) -> None:
        rows = await self.db.query(
            """
            SELECT e.chunk_id, e.vector, e.dim, c.doc_id, c.text, c.project_key, c.issue_key,
                   c.source_type, c.source_updated, c.metadata_json
            FROM chunk_embeddings e
            JOIN chunks c ON c.chunk_id = e.chunk_id
            WHERE e.model = ?
            """,
            (self.model_name,),
        )
        if not rows:
            self._matrix = np.zeros((0, self.dim), dtype=np.float32)
            self._rows = []
            self._dirty = False
            return
        vectors = [np.frombuffer(r["vector"], dtype=np.float32) for r in rows]
        width = max(v.shape[0] for v in vectors)
        padded = np.zeros((len(vectors), width), dtype=np.float32)
        for i, vector in enumerate(vectors):
            padded[i, : vector.shape[0]] = vector
        self._matrix = padded
        self._rows = rows
        self._dirty = False

    async def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 8,
        *,
        project_keys: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        issue_keys: Optional[Sequence[str]] = None,
        updated_after: Optional[str] = None,
        updated_before: Optional[str] = None,
        min_score: float = 0.0,
    ) -> list[VectorHit]:
        async with self._lock:
            if self._dirty or self._matrix is None:
                await self._load()

        if self._matrix is None or self._matrix.shape[0] == 0:
            return []

        query = query_vector.astype(np.float32)
        if query.shape[0] < self._matrix.shape[1]:
            query = np.pad(query, (0, self._matrix.shape[1] - query.shape[0]))
        else:
            query = query[: self._matrix.shape[1]]
        norm = float(np.linalg.norm(query)) or 1.0
        scores = self._matrix @ (query / norm)

        project_set = {p.upper() for p in project_keys} if project_keys else None
        type_set = set(source_types) if source_types else None
        issue_set = {k.upper() for k in issue_keys} if issue_keys else None

        candidates: list[tuple[float, dict[str, Any]]] = []
        for index, row in enumerate(self._rows):
            if project_set and (row["project_key"] or "").upper() not in project_set:
                continue
            if type_set and row["source_type"] not in type_set:
                continue
            if issue_set and (row["issue_key"] or "").upper() not in issue_set:
                continue
            updated = row["source_updated"] or ""
            if updated_after and updated and updated < updated_after:
                continue
            if updated_before and updated and updated > updated_before:
                continue
            score = float(scores[index])
            if score < min_score:
                continue
            candidates.append((score, row))

        candidates.sort(key=lambda pair: pair[0], reverse=True)
        hits: list[VectorHit] = []
        for score, row in candidates[:top_k]:
            try:
                metadata = json.loads(row["metadata_json"] or "{}")
            except json.JSONDecodeError:
                metadata = {}
            metadata.update({
                "project_key": row["project_key"],
                "issue_key": row["issue_key"],
                "source_type": row["source_type"],
                "source_updated": row["source_updated"],
            })
            hits.append(VectorHit(
                chunk_id=row["chunk_id"], doc_id=row["doc_id"], score=score,
                text=row["text"], metadata=metadata,
            ))
        return hits

    async def stats(self) -> dict[str, Any]:
        row = await self.db.query_one("SELECT COUNT(*) AS n FROM chunk_embeddings WHERE model = ?", (self.model_name,))
        docs = await self.db.query_one("SELECT COUNT(*) AS n FROM documents")
        return {
            "model": self.model_name,
            "dim": self.dim,
            "embedded_chunks": int(row["n"]) if row else 0,
            "documents": int(docs["n"]) if docs else 0,
        }
