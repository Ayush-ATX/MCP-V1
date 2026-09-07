"""clients/embedding_client.py — Embedding client for nemotron-3-embed-1b with SQLite caching and cosine similarity."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from typing import Sequence

from openai import OpenAI

import config
from storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)


def _text_hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode("utf-8")).hexdigest()


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two numeric vectors."""
    if not vec_a or not vec_b or len(vec_a) != len(vec_b):
        return 0.0
    dot = 0.0
    norm_a = 0.0
    norm_b = 0.0
    for a, b in zip(vec_a, vec_b):
        dot += a * b
        norm_a += a * a
        norm_b += b * b
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    sim = dot / (math.sqrt(norm_a) * math.sqrt(norm_b))
    # Clamp to [-1.0, 1.0] for precision noise
    return max(-1.0, min(1.0, float(sim)))


class EmbeddingClient:
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.base_url = (base_url or config.EMBEDDING_BASE_URL).rstrip("/")
        self.api_key = api_key or config.EMBEDDING_API_KEY
        self.model = model or config.EMBEDDING_MODEL

    async def get_embedding(self, text: str) -> list[float] | None:
        """Get embedding for a single text, utilizing cache."""
        res = await self.get_embeddings([text])
        return res[0] if res else None

    async def get_embeddings(self, texts: list[str]) -> list[list[float] | None]:
        """Get embeddings for a list of texts, checking SQLite cache first and batching misses."""
        if not texts:
            return []

        results: list[list[float] | None] = [None] * len(texts)
        uncached_indices: list[int] = []
        uncached_texts: list[str] = []

        # 1. Lookup in SQLite cache
        conn = await get_db()
        try:
            for idx, raw_text in enumerate(texts):
                t = raw_text.strip()
                if not t:
                    results[idx] = []
                    continue
                h = _text_hash(t)
                row = await conn.execute_fetchall(
                    "SELECT embedding_blob FROM embedding_cache WHERE text_hash = ? AND model = ?",
                    (h, self.model),
                )
                if row and row[0]["embedding_blob"]:
                    try:
                        vec = json.loads(row[0]["embedding_blob"])
                        results[idx] = vec
                    except Exception:
                        uncached_indices.append(idx)
                        uncached_texts.append(t)
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(t)
        finally:
            await conn.close()

        # 2. Fetch cache misses from remote API
        if uncached_texts:
            fetched_vectors = await self._fetch_remote_batch(uncached_texts)
            now_iso = datetime.now(timezone.utc).isoformat()
            conn2 = await get_db()
            try:
                for idx, t, vec in zip(uncached_indices, uncached_texts, fetched_vectors):
                    results[idx] = vec
                    if vec:
                        h = _text_hash(t)
                        preview = t[:100]
                        try:
                            await execute_with_retry(
                                conn2,
                                """
                                INSERT OR REPLACE INTO embedding_cache (text_hash, model, embedding_blob, text_preview, created_at)
                                VALUES (?, ?, ?, ?, ?)
                                """,
                                (h, self.model, json.dumps(vec), preview, now_iso),
                            )
                        except Exception as cache_err:
                            logger.warning("Failed to store vector in cache: %s", cache_err)
            finally:
                await conn2.close()

        return results

    async def _fetch_remote_batch(self, texts: list[str]) -> list[list[float] | None]:
        """Call remote embedding API in thread pool."""
        if not self.api_key:
            logger.warning("EmbeddingClient: EMBEDDING_API_KEY is not set.")
            return [None] * len(texts)

        def _sync_call() -> list[list[float] | None]:
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=config.MCP_HTTP_TIMEOUT_S,
            )
            # Batch in chunks of 32
            chunk_size = 32
            out: list[list[float] | None] = []
            for i in range(0, len(texts), chunk_size):
                chunk = texts[i : i + chunk_size]
                try:
                    # Model name normalization: try with nvidia/ prefix if needed
                    model_to_use = self.model
                    resp = client.embeddings.create(input=chunk, model=model_to_use)
                    for item in resp.data:
                        out.append(item.embedding)
                except Exception as exc:
                    logger.error("Embedding API remote call failed: %s", exc)
                    # If model didn't have nvidia/ prefix or had it, attempt alternate once
                    alt_model = f"nvidia/{self.model}" if not self.model.startswith("nvidia/") else self.model.replace("nvidia/", "")
                    try:
                        logger.info("Retrying embedding with alternate model identifier: %s", alt_model)
                        resp2 = client.embeddings.create(input=chunk, model=alt_model)
                        for item in resp2.data:
                            out.append(item.embedding)
                    except Exception as alt_exc:
                        logger.error("Alternate embedding attempt failed: %s", alt_exc)
                        for _ in chunk:
                            out.append(None)
            return out

        return await asyncio.to_thread(_sync_call)


# Global helper instance
_default_embedding_client = EmbeddingClient()


async def compute_similarity(text_a: str, text_b: str, client: EmbeddingClient | None = None) -> float:
    """Compute cosine similarity between two texts using embedding_v2."""
    c = client or _default_embedding_client
    vecs = await c.get_embeddings([text_a, text_b])
    if not vecs or vecs[0] is None or vecs[1] is None:
        return 0.0
    return cosine_similarity(vecs[0], vecs[1])
