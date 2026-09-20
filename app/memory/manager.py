"""Memory manager: the write path, including conflict resolution.

Given extraction candidates it decides, per candidate, one of:
  * skip     - a near-identical record already exists (refresh its timestamp instead);
  * update   - same subject, same meaning, small wording change;
  * supersede- same subject, *different* claim -> write the new record and mark the old one
               superseded (never leave two contradictory 'top priority' facts active);
  * create   - genuinely new information.
Every decision is logged to memory_events so the UI trace can show what memory did and why.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from app.memory.extractor import ExtractionResult, MemoryCandidate, MemoryExtractor
from app.memory.long_term import LongTermMemoryStore, MemoryRecord
from app.memory.retriever import MemoryRetriever, _cosine
from app.rag.embeddings import EmbeddingProvider
from app.utils.text import keyword_overlap

SINGLETON_SUBJECTS = {"priority", "role", "goal", "answer_style"}


class MemoryManager:
    def __init__(
        self,
        store: LongTermMemoryStore,
        extractor: MemoryExtractor,
        retriever: MemoryRetriever,
        embeddings: EmbeddingProvider,
        *,
        conflict_similarity: float = 0.80,
        duplicate_similarity: float = 0.94,
        max_per_user: int = 300,
    ):
        self.store = store
        self.extractor = extractor
        self.retriever = retriever
        self.embeddings = embeddings
        self.conflict_similarity = conflict_similarity
        self.duplicate_similarity = duplicate_similarity
        self.max_per_user = max_per_user

    async def recall(self, user_id: str, query: str, top_k: Optional[int] = None) -> list[MemoryRecord]:
        return await self.retriever.retrieve(user_id, query, top_k=top_k)

    async def observe(
        self,
        *,
        user_id: str,
        session_id: Optional[str],
        run_id: Optional[str],
        user_message: str,
        answer_summary: str = "",
    ) -> dict[str, Any]:
        """Extract, resolve conflicts, persist. Returns a trace-friendly report."""
        extraction: ExtractionResult = await self.extractor.extract(user_message, answer_summary)
        actions: list[dict[str, Any]] = []

        for rejected in extraction.rejected:
            await self.store.log_event(
                run_id=run_id, user_id=user_id, memory_id=None, action="rejected",
                reason=rejected.get("reason", ""), content=rejected.get("content", ""),
            )

        if extraction.candidates:
            vectors = await self.embeddings.embed_documents([c.content for c in extraction.candidates])
        else:
            vectors = np.zeros((0, self.embeddings.dim), dtype=np.float32)

        for index, candidate in enumerate(extraction.candidates):
            action = await self._apply(
                user_id=user_id, session_id=session_id, run_id=run_id,
                candidate=candidate, vector=vectors[index],
            )
            actions.append(action)

        pruned = await self.store.prune_to_limit(user_id, self.max_per_user)
        if pruned:
            await self.store.log_event(
                run_id=run_id, user_id=user_id, memory_id=None, action="pruned",
                reason=f"store exceeded {self.max_per_user} active records", content=str(pruned),
            )

        return {
            "extractor": extraction.origin,
            "candidates_considered": len(extraction.candidates),
            "rejected": extraction.rejected,
            "actions": actions,
            "pruned": pruned,
        }

    async def _apply(
        self, *, user_id: str, session_id: Optional[str], run_id: Optional[str],
        candidate: MemoryCandidate, vector: np.ndarray,
    ) -> dict[str, Any]:
        existing = await self.store.active(user_id)
        best: Optional[tuple[float, MemoryRecord]] = None
        for record, record_vector in existing:
            if record.type != candidate.type:
                continue
            similarity = _cosine(vector, record_vector) if record_vector is not None else 0.0
            lexical = keyword_overlap(candidate.content, record.content)
            score = max(similarity, lexical)
            same_subject = bool(candidate.subject) and candidate.subject == record.subject
            if same_subject and candidate.subject in SINGLETON_SUBJECTS:
                score = max(score, self.conflict_similarity)  # one active record per singleton subject
            if best is None or score > best[0]:
                best = (score, record)

        if best and best[0] >= self.duplicate_similarity:
            record = best[1]
            await self.store.update_content(
                record.memory_id, record.content,
                importance=max(record.importance, candidate.importance),
                confidence=min(1.0, record.confidence + 0.05), run_id=run_id,
            )
            await self.store.log_event(
                run_id=run_id, user_id=user_id, memory_id=record.memory_id,
                action="skipped_duplicate",
                reason=f"similarity {best[0]:.2f} >= {self.duplicate_similarity}",
                content=candidate.content,
            )
            return {"action": "skipped_duplicate", "memory_id": record.memory_id,
                    "similarity": round(best[0], 3), "content": candidate.content}

        if best and best[0] >= self.conflict_similarity:
            old = best[1]
            if old.content.strip().lower() == candidate.content.strip().lower():
                await self.store.update_content(old.memory_id, candidate.content, run_id=run_id)
                await self.store.log_event(
                    run_id=run_id, user_id=user_id, memory_id=old.memory_id, action="updated",
                    reason="same claim, refreshed", content=candidate.content,
                )
                return {"action": "updated", "memory_id": old.memory_id,
                        "similarity": round(best[0], 3), "content": candidate.content}

            created = await self.store.create(
                user_id=user_id, type=candidate.type, content=candidate.content,
                subject=candidate.subject or old.subject, importance=candidate.importance,
                confidence=candidate.confidence, session_id=session_id, run_id=run_id,
                expires_at=candidate.expires_at(), embedding=vector,
            )
            await self.store.supersede(old.memory_id, created.memory_id)
            await self.store.log_event(
                run_id=run_id, user_id=user_id, memory_id=created.memory_id, action="superseded",
                reason=f"conflicts with {old.memory_id} ('{old.content[:80]}') at similarity {best[0]:.2f}",
                content=candidate.content,
            )
            return {"action": "superseded", "memory_id": created.memory_id,
                    "superseded_memory_id": old.memory_id, "previous_content": old.content,
                    "similarity": round(best[0], 3), "content": candidate.content}

        created = await self.store.create(
            user_id=user_id, type=candidate.type, content=candidate.content,
            subject=candidate.subject, importance=candidate.importance,
            confidence=candidate.confidence, session_id=session_id, run_id=run_id,
            expires_at=candidate.expires_at(), embedding=vector,
        )
        await self.store.log_event(
            run_id=run_id, user_id=user_id, memory_id=created.memory_id, action="created",
            reason=candidate.reason or "new durable statement", content=candidate.content,
        )
        return {"action": "created", "memory_id": created.memory_id, "type": candidate.type,
                "importance": candidate.importance, "content": candidate.content}

    async def list_memories(self, user_id: str, include_inactive: bool = False) -> list[dict[str, Any]]:
        return [r.as_dict() for r in await self.store.list_all(user_id, include_inactive)]

    async def delete(self, user_id: str, memory_id: str) -> bool:
        return await self.store.delete(user_id, memory_id)
