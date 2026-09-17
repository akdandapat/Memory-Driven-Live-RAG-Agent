"""Semantic retrieval over the local index, returned as Evidence.

This is the *unstructured* half of retrieval. Structured questions ("which issues are
overdue", "what changed on 12 June") are answered by MCP tools hitting the source directly -
forcing those through a vector store would be slower and less accurate. The planner decides
which half to use; the executor can use both for one sub-question.
"""
from __future__ import annotations

from typing import Optional, Sequence

from app.models.evidence import Evidence
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_index import VectorIndex
from app.security.sanitize import detect_injection
from app.utils.dates import parse_dt
from app.utils.text import keyword_overlap, truncate


class SemanticRetriever:
    def __init__(self, embeddings: EmbeddingProvider, index: VectorIndex, snippet_chars: int = 900):
        self.embeddings = embeddings
        self.index = index
        self.snippet_chars = snippet_chars

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 6,
        project_keys: Optional[Sequence[str]] = None,
        source_types: Optional[Sequence[str]] = None,
        issue_keys: Optional[Sequence[str]] = None,
        updated_after: Optional[str] = None,
        updated_before: Optional[str] = None,
        min_score: float = 0.05,
        tool_used: str = "semantic_search",
        sub_question_id: str = "",
    ) -> list[Evidence]:
        vector = await self.embeddings.embed_query(query)
        hits = await self.index.search(
            vector,
            top_k=top_k,
            project_keys=project_keys,
            source_types=source_types,
            issue_keys=issue_keys,
            updated_after=updated_after,
            updated_before=updated_before,
            min_score=min_score,
        )
        evidence: list[Evidence] = []
        for hit in hits:
            metadata = hit.metadata or {}
            # Blend vector similarity with lexical overlap: with the hashing provider this
            # stabilises ranking, and with a real encoder it acts as a light keyword prior.
            lexical = keyword_overlap(query, hit.text)
            score = round(0.75 * hit.score + 0.25 * lexical, 4)
            injection = bool(detect_injection(hit.text))
            evidence.append(Evidence(
                source="jira_index",
                source_type=metadata.get("source_type", "chunk"),
                source_id=str(metadata.get("source_id", hit.doc_id)),
                title=str(metadata.get("title", "")),
                url=str(metadata.get("url", "")),
                timestamp=parse_dt(metadata.get("source_updated")),
                content=truncate(hit.text, self.snippet_chars),
                relevance=score if not injection else score * 0.2,
                tool_used=tool_used,
                sub_question_id=sub_question_id,
                project_key=str(metadata.get("project_key", "")),
                injection_suspected=injection,
                metadata={"chunk_id": hit.chunk_id, "vector_score": round(hit.score, 4),
                          "lexical_score": round(lexical, 4), "issue_key": metadata.get("issue_key", "")},
            ))
        evidence.sort(key=lambda e: e.relevance, reverse=True)
        return evidence
