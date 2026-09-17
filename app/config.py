"""Central configuration. Every secret is read from the environment - never hard-coded."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.getenv("ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- application -------------------------------------------------------
    app_mode: Literal["demo", "live"] = "demo"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    default_user_id: str = "demo-user"
    log_level: str = "INFO"

    # --- jira --------------------------------------------------------------
    jira_base_url: str = ""
    jira_email: str = ""
    jira_api_token: str = ""
    jira_allowed_projects: str = ""
    jira_timeout_seconds: float = 30.0
    jira_max_retries: int = 3
    jira_page_size: int = 50
    mock_data_dir: str = "data/mock_jira"

    # --- llm ---------------------------------------------------------------
    llm_provider: Literal["none", "openai", "anthropic"] = "none"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_base_url: str = "https://api.openai.com/v1"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 1600
    llm_timeout_seconds: float = 60.0

    # --- embeddings --------------------------------------------------------
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 512
    embedding_api_key: str = ""
    embedding_base_url: str = "https://api.openai.com/v1"

    # --- storage -----------------------------------------------------------
    db_path: str = "data/app.db"

    # --- mcp ---------------------------------------------------------------
    mcp_transport: Literal["inproc", "stdio", "http"] = "inproc"
    mcp_server_url: str = "http://localhost:8765/mcp"
    mcp_tool_timeout_seconds: float = 45.0

    # --- agent budget ------------------------------------------------------
    agent_max_tool_calls: int = 18
    agent_max_rounds: int = 3
    agent_max_sub_questions: int = 6
    agent_max_evidence: int = 60
    agent_evidence_chars: int = 1200

    # --- memory ------------------------------------------------------------
    memory_top_k: int = 5
    memory_min_importance: float = 0.35
    memory_min_relevance: float = 0.30
    memory_conflict_similarity: float = 0.80
    memory_duplicate_similarity: float = 0.94
    memory_max_per_user: int = 300

    # --- observability -----------------------------------------------------
    langsmith_tracing: bool = False
    langsmith_api_key: str = ""
    langsmith_project: str = "agentic-rag-live-data"
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @field_validator("jira_base_url")
    @classmethod
    def _strip_slash(cls, v: str) -> str:
        return v.rstrip("/")

    # --- derived helpers ---------------------------------------------------
    @property
    def allowed_projects(self) -> set[str]:
        raw = [p.strip().upper() for p in self.jira_allowed_projects.split(",")]
        return {p for p in raw if p}

    @property
    def llm_enabled(self) -> bool:
        return self.llm_provider != "none" and bool(self.llm_api_key)

    @property
    def db_file(self) -> Path:
        p = Path(self.db_path)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def mock_dir(self) -> Path:
        p = Path(self.mock_data_dir)
        return p if p.is_absolute() else PROJECT_ROOT / p

    def live_credentials_present(self) -> bool:
        return bool(self.jira_base_url and self.jira_email and self.jira_api_token)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reset_settings_cache() -> None:
    """Used by tests that patch environment variables."""
    get_settings.cache_clear()
