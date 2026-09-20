"""Build the configured LLM client, or None when running in heuristic mode."""
from __future__ import annotations

import logging
from typing import Optional

from app.config import Settings
from app.llm.anthropic_client import AnthropicClient
from app.llm.base import LLMClient, LLMError
from app.llm.openai_client import OpenAIChatClient

logger = logging.getLogger(__name__)


def build_llm(settings: Settings) -> Optional[LLMClient]:
    if settings.llm_provider == "none":
        return None
    if not settings.llm_api_key:
        logger.warning(
            "LLM_PROVIDER=%s but LLM_API_KEY is empty - falling back to heuristic planner/synthesizer",
            settings.llm_provider,
        )
        return None
    try:
        if settings.llm_provider == "openai":
            return OpenAIChatClient(
                api_key=settings.llm_api_key,
                model=settings.llm_model,
                base_url=settings.llm_base_url,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_seconds,
            )
        return AnthropicClient(
            api_key=settings.llm_api_key,
            model=settings.llm_model,
            base_url=settings.llm_base_url,
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
            timeout=settings.llm_timeout_seconds,
        )
    except LLMError as exc:
        logger.warning("LLM disabled: %s", exc)
        return None
