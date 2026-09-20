"""Anthropic Messages API client."""
from __future__ import annotations

import time
from typing import Optional

import httpx

from app.llm.base import LLMClient, LLMError, LLMResponse

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.anthropic.com",
        temperature: float = 0.0,
        max_tokens: int = 1600,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not api_key:
            raise LLMError("LLM_API_KEY is required for the anthropic provider")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        base = base_url.rstrip("/")
        if base.endswith("/v1"):
            base = base[: -len("/v1")]
        self._client = client or httpx.AsyncClient(
            base_url=base,
            timeout=timeout,
            headers={
                "x-api-key": api_key,
                "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json",
            },
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def complete(
        self,
        system: str,
        user: str,
        *,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        json_object: bool = False,
    ) -> LLMResponse:
        content = user
        if json_object:
            content = f"{user}\n\nRespond with a single valid JSON object and nothing else."
        payload = {
            "model": self.model,
            "system": system,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        started = time.perf_counter()
        try:
            response = await self._client.post("/v1/messages", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM transport error: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"LLM HTTP {response.status_code}: {response.text[:300]}")

        data = response.json()
        blocks = data.get("content", []) or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("input_tokens", 0)),
            completion_tokens=int(usage.get("output_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
