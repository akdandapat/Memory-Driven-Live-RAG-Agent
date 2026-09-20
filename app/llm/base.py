"""LLM abstraction.

Two providers are supported through plain HTTP (no vendor SDK pinning): any OpenAI-compatible
endpoint (OpenAI, Azure, OpenRouter, Ollama, vLLM) and the Anthropic Messages API. When
LLM_PROVIDER=none the whole system still runs using deterministic heuristics - see
`app/agents/planner.py` and `app/agents/synthesizer.py`.
"""
from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from typing import Any, Optional


class LLMError(Exception):
    """Any failure while talking to the model provider."""


@dataclass
class LLMResponse:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: int = 0


class LLMClient(abc.ABC):
    model: str = ""

    @abc.abstractmethod
    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_object: bool = False,
    ) -> LLMResponse: ...

    async def close(self) -> None:
        return None

    async def complete_json(
        self, system: str, user: str, *, max_tokens: Optional[int] = None
    ) -> dict[str, Any]:
        """Ask for JSON and parse defensively (models sometimes wrap output in fences)."""
        response = await self.complete(system, user, temperature=0.0, max_tokens=max_tokens, json_object=True)
        return parse_json_object(response.text)


def parse_json_object(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        raise LLMError("Empty model response")
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fenced:
        text = fenced.group(1).strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMError(f"Model did not return JSON: {text[:200]}")
        try:
            parsed = json.loads(text[start: end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise LLMError("Model returned JSON that is not an object")
    return parsed
