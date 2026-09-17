"""Embedding providers.

`hashing` is a deterministic, dependency-free lexical embedding: hashed unigrams + bigrams
with sublinear term weighting, L2 normalised. It needs no API key and makes the demo fully
offline, but it is a *lexical* signal - it will not match paraphrases the way a trained
encoder does. Set EMBEDDING_PROVIDER=openai for real semantic quality. This trade-off is
stated in the README rather than hidden.
"""
from __future__ import annotations

import abc
import hashlib
import math
from typing import Optional, Sequence

import httpx
import numpy as np

from app.utils.text import tokenize


class EmbeddingError(Exception):
    pass


class EmbeddingProvider(abc.ABC):
    name: str = "base"
    dim: int = 512

    @abc.abstractmethod
    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    async def embed_query(self, text: str) -> np.ndarray:
        vectors = await self.embed_documents([text])
        return vectors[0]

    async def close(self) -> None:
        return None


def _l2(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


class HashingEmbeddings(EmbeddingProvider):
    name = "hashing-v1"

    def __init__(self, dim: int = 512):
        self.dim = max(64, dim)

    def _bucket(self, token: str) -> int:
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self.dim

    def _vector(self, text: str) -> np.ndarray:
        tokens = tokenize(text)
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        grams = list(tokens) + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        counts: dict[int, float] = {}
        for gram in grams:
            counts[self._bucket(gram)] = counts.get(self._bucket(gram), 0.0) + 1.0
        vector = np.zeros(self.dim, dtype=np.float32)
        for bucket, count in counts.items():
            vector[bucket] = 1.0 + math.log(count)     # sublinear tf
        return vector

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        matrix = np.vstack([self._vector(t) for t in texts]).astype(np.float32)
        return _l2(matrix)


class OpenAIEmbeddings(EmbeddingProvider):
    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        base_url: str = "https://api.openai.com/v1",
        dim: int = 1536,
        timeout: float = 60.0,
        batch_size: int = 64,
        client: Optional[httpx.AsyncClient] = None,
    ):
        if not api_key:
            raise EmbeddingError("EMBEDDING_API_KEY is required for the openai embedding provider")
        self.name = model
        self.model = model
        self.dim = dim
        self.batch_size = batch_size
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t[:8000] or " " for t in texts[start: start + self.batch_size]]
            payload = {"model": self.model, "input": batch}
            if "text-embedding-3" in self.model:
                payload["dimensions"] = self.dim
            try:
                response = await self._client.post("/embeddings", json=payload)
            except httpx.HTTPError as exc:
                raise EmbeddingError(f"Embedding transport error: {exc}") from exc
            if response.status_code >= 400:
                raise EmbeddingError(f"Embedding HTTP {response.status_code}: {response.text[:300]}")
            data = response.json()
            vectors.extend(item["embedding"] for item in sorted(data["data"], key=lambda d: d["index"]))
        matrix = np.array(vectors, dtype=np.float32)
        return _l2(matrix)


def build_embeddings(settings) -> EmbeddingProvider:
    if settings.embedding_provider == "openai" and settings.embedding_api_key:
        return OpenAIEmbeddings(
            api_key=settings.embedding_api_key,
            model=settings.embedding_model,
            base_url=settings.embedding_base_url,
            dim=settings.embedding_dim,
        )
    return HashingEmbeddings(dim=settings.embedding_dim)
