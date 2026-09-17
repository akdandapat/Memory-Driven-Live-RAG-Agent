"""Live source -> local index.

Pipeline: fetch (connector) -> normalise (domain models) -> document representation ->
content hash -> chunk -> embed -> upsert -> advance watermark.

Only *changed* documents are re-chunked and re-embedded: the watermark (max source_updated
per project) limits what we fetch, and the content hash stops us re-embedding unchanged text.
Jira stays the source of truth; everything written here is a disposable cache.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional, Sequence

from app.connectors.base import ConnectorError, IssueQuery, SourceConnector
from app.database.db import Database
from app.models.domain import Comment, Issue, Project
from app.rag.embeddings import EmbeddingProvider
from app.rag.vector_index import VectorIndex
from app.utils.dates import iso, parse_dt, utcnow
from app.utils.text import chunk_text, collapse_whitespace, join_nonempty

logger = logging.getLogger(__name__)


@dataclass
class SyncStats:
    mode: str = "incremental"
    projects_seen: int = 0
    issues_fetched: int = 0
    comments_fetched: int = 0
    documents_upserted: int = 0
    documents_unchanged: int = 0
    chunks_indexed: int = 0
    errors: list[str] = field(default_factory=list)
    started_at: str = ""
    finished_at: str = ""
    watermarks: dict[str, Optional[str]] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "projects_seen": self.projects_seen,
            "issues_fetched": self.issues_fetched,
            "comments_fetched": self.comments_fetched,
            "documents_upserted": self.documents_upserted,
            "documents_unchanged": self.documents_unchanged,
            "chunks_indexed": self.chunks_indexed,
            "errors": self.errors,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "watermarks": self.watermarks,
        }


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def issue_document_text(issue: Issue) -> str:
    """Flatten an issue into the text we actually want to retrieve semantically."""
    header = f"{issue.key}: {issue.summary}"
    facts = (
        f"Project {issue.project_key}. Type {issue.issue_type or 'unknown'}. "
        f"Status {issue.status or 'unknown'} ({issue.status_category or 'unknown'}). "
        f"Priority {issue.priority or 'unset'}. "
        f"Assignee {issue.assignee or 'unassigned'}. "
        f"Due {issue.duedate.date().isoformat() if issue.duedate else 'none'}. "
        f"Labels {', '.join(issue.labels) if issue.labels else 'none'}."
    )
    links = "; ".join(f"{link.type} {link.issue_key}" for link in issue.links)
    return join_nonempty([header, facts, issue.description, f"Links: {links}" if links else ""], "\n")


def project_document_text(project: Project) -> str:
    goals = project.raw_extra.get("goals") or []
    goal_text = "\n".join(f"- {g}" for g in goals)
    return join_nonempty([
        f"{project.key}: {project.name}",
        f"Lead: {project.lead}" if project.lead else "",
        project.description,
        f"Stated goals:\n{goal_text}" if goal_text else "",
    ], "\n")


class IngestionService:
    def __init__(
        self,
        db: Database,
        connector: SourceConnector,
        embeddings: EmbeddingProvider,
        index: VectorIndex,
        *,
        chunk_chars: int = 900,
        chunk_overlap: int = 150,
    ):
        self.db = db
        self.connector = connector
        self.embeddings = embeddings
        self.index = index
        self.chunk_chars = chunk_chars
        self.chunk_overlap = chunk_overlap

    # --- watermarks --------------------------------------------------------
    async def get_watermark(self, resource: str) -> Optional[str]:
        row = await self.db.query_one(
            "SELECT watermark FROM sync_state WHERE source = ? AND resource = ?",
            (self.connector.source_name, resource),
        )
        return row["watermark"] if row else None

    async def set_watermark(self, resource: str, watermark: Optional[str], stats: dict[str, Any]) -> None:
        await self.db.execute(
            "INSERT INTO sync_state (source, resource, watermark, last_sync_at, stats_json) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(source, resource) DO UPDATE SET watermark=excluded.watermark, "
            "last_sync_at=excluded.last_sync_at, stats_json=excluded.stats_json",
            (self.connector.source_name, resource, watermark, iso(utcnow()), json.dumps(stats)),
        )

    # --- document upsert ---------------------------------------------------
    async def _upsert_document(
        self,
        *,
        source_type: str,
        source_id: str,
        title: str,
        url: str,
        text: str,
        project_key: str,
        issue_key: str,
        source_created: Optional[datetime],
        source_updated: Optional[datetime],
        metadata: dict[str, Any],
    ) -> tuple[bool, int]:
        """Returns (changed, chunks_indexed)."""
        text = collapse_whitespace(text)
        if not text:
            return False, 0
        doc_id = f"{self.connector.source_name}:{source_type}:{source_id}"
        content_hash = _hash(text)
        existing = await self.db.query_one(
            "SELECT content_hash FROM documents WHERE doc_id = ?", (doc_id,)
        )
        if existing and existing["content_hash"] == content_hash:
            return False, 0

        await self.db.execute(
            """
            INSERT INTO documents (doc_id, source, source_type, source_id, project_key, issue_key,
                                   title, url, text, content_hash, source_created, source_updated,
                                   ingested_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(doc_id) DO UPDATE SET
                title=excluded.title, url=excluded.url, text=excluded.text,
                content_hash=excluded.content_hash, source_created=excluded.source_created,
                source_updated=excluded.source_updated, ingested_at=excluded.ingested_at,
                metadata_json=excluded.metadata_json, project_key=excluded.project_key,
                issue_key=excluded.issue_key
            """,
            (doc_id, self.connector.source_name, source_type, source_id, project_key, issue_key,
             title, url, text, content_hash, iso(source_created), iso(source_updated),
             iso(utcnow()), json.dumps(metadata)),
        )
        # re-chunk from scratch: chunk boundaries move when text changes
        await self.db.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
        pieces = chunk_text(text, self.chunk_chars, self.chunk_overlap)
        if not pieces:
            return True, 0
        chunk_ids = [f"{doc_id}#{i}" for i in range(len(pieces))]
        await self.db.executemany(
            "INSERT INTO chunks (chunk_id, doc_id, ordinal, text, project_key, issue_key, "
            "source_type, source_updated, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (chunk_ids[i], doc_id, i, pieces[i], project_key, issue_key, source_type,
                 iso(source_updated), json.dumps({**metadata, "title": title, "url": url,
                                                  "source_id": source_id}))
                for i in range(len(pieces))
            ],
        )
        vectors = await self.embeddings.embed_documents(pieces)
        await self.index.upsert(chunk_ids, vectors)
        return True, len(pieces)

    # --- public API --------------------------------------------------------
    async def sync(
        self,
        project_keys: Optional[Sequence[str]] = None,
        *,
        full: bool = False,
        max_issues_per_project: int = 300,
        include_comments: bool = True,
    ) -> SyncStats:
        stats = SyncStats(mode="full" if full else "incremental", started_at=iso(utcnow()) or "")
        try:
            projects = await self.connector.search_projects(limit=50)
        except ConnectorError as exc:
            stats.errors.append(f"search_projects failed: {exc}")
            stats.finished_at = iso(utcnow()) or ""
            return stats

        if project_keys:
            wanted = {k.upper() for k in project_keys}
            projects = [p for p in projects if p.key.upper() in wanted]
        stats.projects_seen = len(projects)

        for project in projects:
            try:
                await self._sync_project(project, stats, full, max_issues_per_project, include_comments)
            except ConnectorError as exc:
                stats.errors.append(f"{project.key}: {exc}")
                logger.warning("Sync failed for %s: %s", project.key, exc)

        stats.finished_at = iso(utcnow()) or ""
        return stats

    async def _sync_project(
        self,
        project: Project,
        stats: SyncStats,
        full: bool,
        max_issues: int,
        include_comments: bool,
    ) -> None:
        changed, chunks = await self._upsert_document(
            source_type="project", source_id=project.key, title=f"{project.key} - {project.name}",
            url=project.url, text=project_document_text(project), project_key=project.key,
            issue_key="", source_created=None, source_updated=utcnow(),
            metadata={"lead": project.lead, "name": project.name},
        )
        stats.documents_upserted += int(changed)
        stats.documents_unchanged += int(not changed)
        stats.chunks_indexed += chunks

        watermark = None if full else await self.get_watermark(project.key)
        query = IssueQuery(project_key=project.key, order_by="updated", order_dir="ASC")
        if watermark:
            query.updated_after = parse_dt(watermark)

        issues = await self.connector.iter_issues(query, page_size=50, max_items=max_issues)
        stats.issues_fetched += len(issues)
        newest = watermark

        for issue in issues:
            changed, chunks = await self._upsert_document(
                source_type="issue", source_id=issue.key, title=f"{issue.key} {issue.summary}",
                url=issue.url, text=issue_document_text(issue), project_key=issue.project_key,
                issue_key=issue.key, source_created=issue.created, source_updated=issue.updated,
                metadata={
                    "status": issue.status, "priority": issue.priority, "assignee": issue.assignee,
                    "issue_type": issue.issue_type, "labels": issue.labels,
                    "duedate": iso(issue.duedate), "resolved": issue.is_resolved,
                },
            )
            stats.documents_upserted += int(changed)
            stats.documents_unchanged += int(not changed)
            stats.chunks_indexed += chunks

            if include_comments:
                try:
                    comments = await self.connector.get_issue_comments(issue.key, limit=50)
                except ConnectorError as exc:
                    stats.errors.append(f"{issue.key} comments: {exc}")
                    comments = []
                stats.comments_fetched += len(comments)
                for comment in comments:
                    await self._upsert_comment(issue, comment, stats)

            issue_updated = iso(issue.updated)
            if issue_updated and (newest is None or issue_updated > newest):
                newest = issue_updated

        await self.set_watermark(project.key, newest, {
            "issues_fetched": len(issues), "last_mode": stats.mode,
        })
        stats.watermarks[project.key] = newest

    async def _upsert_comment(self, issue: Issue, comment: Comment, stats: SyncStats) -> None:
        source_id = f"{issue.key}/comment/{comment.id}"
        text = f"Comment by {comment.author or 'unknown'} on {issue.key} ({issue.summary}):\n{comment.body}"
        changed, chunks = await self._upsert_document(
            source_type="comment", source_id=source_id,
            title=f"Comment on {issue.key} by {comment.author or 'unknown'}",
            url=comment.url, text=text, project_key=issue.project_key, issue_key=issue.key,
            source_created=comment.created, source_updated=comment.updated or comment.created,
            metadata={"author": comment.author, "issue_summary": issue.summary},
        )
        stats.documents_upserted += int(changed)
        stats.documents_unchanged += int(not changed)
        stats.chunks_indexed += chunks

    async def index_stats(self) -> dict[str, Any]:
        base = await self.index.stats()
        rows = await self.db.query(
            "SELECT source_type, COUNT(*) AS n FROM documents GROUP BY source_type"
        )
        base["documents_by_type"] = {r["source_type"]: int(r["n"]) for r in rows}
        sync_rows = await self.db.query(
            "SELECT resource, watermark, last_sync_at FROM sync_state WHERE source = ?",
            (self.connector.source_name,),
        )
        base["sync_state"] = sync_rows
        return base
