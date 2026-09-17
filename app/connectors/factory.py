"""Connector selection. One switch decides DEMO vs LIVE; nothing above this line changes."""
from __future__ import annotations

from app.config import Settings
from app.connectors.base import SourceConnector
from app.connectors.jira import JiraConnector
from app.connectors.mock_jira import MockJiraConnector


def build_connector(settings: Settings) -> SourceConnector:
    if settings.app_mode == "live":
        if not settings.live_credentials_present():
            raise RuntimeError(
                "APP_MODE=live but Jira credentials are missing. Set JIRA_BASE_URL, "
                "JIRA_EMAIL and JIRA_API_TOKEN, or switch APP_MODE=demo."
            )
        return JiraConnector(
            base_url=settings.jira_base_url,
            email=settings.jira_email,
            api_token=settings.jira_api_token,
            timeout=settings.jira_timeout_seconds,
            max_retries=settings.jira_max_retries,
            page_size=settings.jira_page_size,
        )
    return MockJiraConnector(settings.mock_dir)
