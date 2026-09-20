"""Composition root: build every component once and wire them together."""
from __future__ import annotations

import logging
from typing import Any, Optional

from app.agents.orchestrator import AgentOrchestrator
from app.agents.planner import Planner
from app.agents.synthesizer import Synthesizer
from app.baseline.naive_rag import NaiveRAG
from app.config import Settings, get_settings
from app.connectors.base import SourceConnector
from app.connectors.factory import build_connector
from app.database.db import Database
from app.llm.base import LLMClient
from app.llm.factory import build_llm
from app.mcp_client.client import MCPToolClient
from app.mcp_server.server import build_server
from app.mcp_server.tools import ToolContext
from app.memory.extractor import MemoryExtractor
from app.memory.long_term import LongTermMemoryStore
from app.memory.manager import MemoryManager
from app.memory.retriever import MemoryRetriever
from app.memory.short_term import ShortTermMemory
from app.observability.langsmith_exporter import LangSmithExporter
from app.observability.tracing import RunTracer, TraceReader
from app.rag.embeddings import EmbeddingProvider, build_embeddings
from app.rag.ingestion import IngestionService
from app.rag.retriever import SemanticRetriever
from app.rag.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class Services:
    """Holds the live object graph for the process."""

    def __init__(self, settings: Optional[Settings] = None):
        self.settings: Settings = settings or get_settings()
        self.db: Database
        self.connector: SourceConnector
        self.embeddings: EmbeddingProvider
        self.index: VectorIndex
        self.retriever: SemanticRetriever
        self.ingestion: IngestionService
        self.llm: Optional[LLMClient]
        self.mcp: MCPToolClient
        self.memory: MemoryManager
        self.short_term: ShortTermMemory
        self.orchestrator: AgentOrchestrator
        self.baseline: NaiveRAG
        self.trace_reader: TraceReader
        self.exporter: LangSmithExporter
        self._mcp_server: Any = None

    async def startup(self) -> None:
        settings = self.settings
        self.db = Database(settings.db_file)
        await self.db.init_schema()

        self.connector = build_connector(settings)
        self.embeddings = build_embeddings(settings)
        self.index = VectorIndex(self.db, self.embeddings.name, self.embeddings.dim)
        self.retriever = SemanticRetriever(self.embeddings, self.index)
        self.ingestion = IngestionService(self.db, self.connector, self.embeddings, self.index)
        self.llm = build_llm(settings)

        tool_context = ToolContext(
            connector=self.connector,
            retriever=self.retriever,
            allowed_projects=settings.allowed_projects or None,
        )
        self._mcp_server = build_server(tool_context) if settings.mcp_transport == "inproc" else None
        self.mcp = MCPToolClient(settings, server_object=self._mcp_server)
        await self.mcp.connect()

        store = LongTermMemoryStore(self.db, self.embeddings.name)
        extractor = MemoryExtractor(self.llm, min_importance=settings.memory_min_importance)
        memory_retriever = MemoryRetriever(
            store, self.embeddings, top_k=settings.memory_top_k,
            min_relevance=settings.memory_min_relevance,
        )
        self.memory = MemoryManager(
            store, extractor, memory_retriever, self.embeddings,
            conflict_similarity=settings.memory_conflict_similarity,
            duplicate_similarity=settings.memory_duplicate_similarity,
            max_per_user=settings.memory_max_per_user,
        )
        self.short_term = ShortTermMemory(self.db, self.llm)

        self.exporter = LangSmithExporter(settings)
        self.trace_reader = TraceReader(self.db)

        self.orchestrator = AgentOrchestrator(
            settings=settings,
            mcp=self.mcp,
            planner=Planner(self.llm, max_sub_questions=settings.agent_max_sub_questions),
            synthesizer=Synthesizer(self.llm),
            memory=self.memory,
            short_term=self.short_term,
            tracer_factory=lambda: RunTracer(self.db, self.exporter),
            llm=self.llm,
        )
        self.baseline = NaiveRAG(self.retriever, self.db, self.llm)

        logger.info(
            "Services ready: mode=%s mcp=%s llm=%s embeddings=%s tools=%d",
            settings.app_mode, settings.mcp_transport,
            settings.llm_provider if self.llm else "none (heuristic)",
            self.embeddings.name, len(self.mcp.tools),
        )

    async def shutdown(self) -> None:
        for closer in (
            getattr(self, "mcp", None), getattr(self, "connector", None),
            getattr(self, "llm", None), getattr(self, "embeddings", None),
        ):
            if closer is None:
                continue
            try:
                await closer.close()
            except Exception as exc:  # noqa: BLE001 - shutdown is best-effort
                logger.warning("Error during shutdown of %s: %s", type(closer).__name__, exc)
        if getattr(self, "db", None) is not None:
            self.db.close()

    async def status(self) -> dict[str, Any]:
        health = await self.connector.health()
        index_stats = await self.ingestion.index_stats()
        return {
            "status": "ok" if health.ok else "degraded",
            "app_mode": self.settings.app_mode,
            "source": health.model_dump(mode="json"),
            "mcp": {
                "transport": self.settings.mcp_transport,
                "connected": bool(self.mcp.tools),
                "tools": self.mcp.tool_names(),
            },
            "llm": {
                "provider": self.settings.llm_provider,
                "active": self.llm is not None,
                "model": self.settings.llm_model if self.llm else "heuristic-fallback",
            },
            "embeddings": {"provider": self.settings.embedding_provider, "model": self.embeddings.name,
                           "dim": self.embeddings.dim},
            "index": index_stats,
            "observability": {"local_tracing": True, "langsmith": self.exporter.enabled},
        }
