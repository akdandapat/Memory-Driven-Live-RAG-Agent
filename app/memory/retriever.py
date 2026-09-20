"""Memory retrieval: pick the few records that actually help this question.

Never inject the whole memory database into a prompt. Score = semantic similarity to the
question, weighted by importance and recency, then threshold and cap at MEMORY_TOP_K.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np

from app.memory.long_term import LongTermMemoryStore, MemoryRecord
from app.rag.embeddings import EmbeddingProvider
from app.utils.text import keyword_overlap


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.size == 0 or b.size == 0:
        return 0.0
    width = min(a.shape[0], b.shape[0])
    a, b = a[:width], b[:width]
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


class MemoryRetriever:
    def __init__(
        self,
        store: LongTermMemoryStore,
        embeddings: EmbeddingProvider,
        *,
        top_k: int = 5,
        min_relevance: float = 0.30,
        recency_half_life_days: float = 45.0,
    ):
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.half_life = recency_half_life_days

    async def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        top_k: Optional[int] = None,
        types: Optional[Sequence[str]] = None,
    ) -> list[MemoryRecord]:
        await self.store.expire_due()
        rows = await self.store.active(user_id)
        if not rows:
            return []
        if types:
            wanted = set(types)
            rows = [(record, vector) for record, vector in rows if record.type in wanted]
        if not rows:
            return []

        query_vector = await self.embeddings.embed_query(query)
        missing = [(record, vector) for record, vector in rows if vector is None]
        if missing:
            # Backfill embeddings for records written before the provider was configured.
            vectors = await self.embeddings.embed_documents([r.content for r, _ in missing])
            for index, (record, _) in enumerate(missing):
                await self.store.update_content(record.memory_id, record.content, embedding=vectors[index])
            rows = await self.store.active(user_id)
            if types:
                rows = [(r, v) for r, v in rows if r.type in set(types)]

        scored: list[MemoryRecord] = []
        for record, vector in rows:
            similarity = _cosine(query_vector, vector) if vector is not None else 0.0
            lexical = keyword_overlap(query, f"{record.subject} {record.content}")
            semantic = max(similarity, lexical)
            recency = 0.5 ** (self.store.age_days(record) / self.half_life)
            score = 0.60 * semantic + 0.25 * record.importance + 0.15 * recency
            record.score = score
            record.score_parts = {
                "semantic": semantic, "similarity": similarity, "lexical": lexical,
                "importance": record.importance, "recency": recency,
            }
            scored.append(record)

        scored.sort(key=lambda r: r.score, reverse=True)
        selected = [r for r in scored if r.score >= self.min_relevance][: (top_k or self.top_k)]
        await self.store.touch([r.memory_id for r in selected])
        return selected

    @staticmethod
    def render(records: Sequence[MemoryRecord]) -> str:
        """Compact block injected into planner/synthesizer prompts."""
        if not records:
            return ""
        lines = []
        for record in records:
            lines.append(f"- [{record.type}] {record.content} (recorded {record.created_at[:10]})")
        return "\n".join(lines)

    @staticmethod
    def as_trace(records: Sequence[MemoryRecord]) -> list[dict[str, Any]]:
        return [
            {
                "memory_id": r.memory_id, "type": r.type, "content": r.content,
                "score": round(r.score, 3), "importance": r.importance,
                "created_at": r.created_at, "why": r.score_parts,
            }
            for r in records
        ]
