"""OpenAI-compatible chat completions client (also works with Azure, OpenRouter, Ollama, vLLM)."""
from __future__ import annotations

import time
from typing import Optional

import httpx

from app.llm.base import LLMClient, LLMError, LLMResponse


class OpenAIChatClient(LLMClient):
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://api.openai.com/v1",
        temperature: float = 0.0,
        max_tokens: int = 1600,
        timeout: float = 60.0,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not api_key:
            raise LLMError("LLM_API_KEY is required for the openai provider")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
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
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": self.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.max_tokens,
        }
        if json_object:
            payload["response_format"] = {"type": "json_object"}

        started = time.perf_counter()
        try:
            response = await self._client.post("/chat/completions", json=payload)
        except httpx.HTTPError as exc:
            raise LLMError(f"LLM transport error: {exc}") from exc
        if response.status_code >= 400:
            raise LLMError(f"LLM HTTP {response.status_code}: {response.text[:300]}")

        data = response.json()
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError) as exc:
            raise LLMError(f"Unexpected LLM response shape: {str(data)[:300]}") from exc
        usage = data.get("usage", {}) or {}
        return LLMResponse(
            text=text,
            model=data.get("model", self.model),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            latency_ms=int((time.perf_counter() - started) * 1000),
        )
