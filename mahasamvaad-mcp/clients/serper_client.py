"""clients/serper_client.py — Direct Serper Google Search API client for web precedence evaluation."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

_SERPER_URL = "https://google.serper.dev/search"


@dataclass
class SerperSearchResult:
    ok: bool
    query: str
    organic: list[dict[str, Any]] = field(default_factory=list)
    knowledge_graph: dict[str, Any] | None = None
    http_status: int | None = None
    error: str | None = None


async def search_web(
    query: str,
    *,
    api_key: str | None = None,
    num_results: int = 10,
    timeout_s: float | None = None,
) -> SerperSearchResult:
    """Execute direct Google search query via Serper API."""
    key = api_key or config.SERPER_API_KEY
    if not key:
        return SerperSearchResult(
            ok=False,
            query=query,
            error="SERPER_API_KEY is not configured",
        )

    headers = {
        "X-API-KEY": key,
        "Content-Type": "application/json",
    }
    payload = {
        "q": query,
        "num": num_results,
    }
    timeout = httpx.Timeout(float(timeout_s or 15.0))

    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(_SERPER_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                return SerperSearchResult(
                    ok=False,
                    query=query,
                    http_status=resp.status_code,
                    error=f"Serper returned HTTP {resp.status_code}: {resp.text[:300]}",
                )

            data = resp.json()
            return SerperSearchResult(
                ok=True,
                query=query,
                organic=data.get("organic", []),
                knowledge_graph=data.get("knowledgeGraph"),
                http_status=200,
            )

    except Exception as exc:
        logger.warning("Serper search error for '%s': %s", query, exc)
        return SerperSearchResult(
            ok=False,
            query=query,
            error=f"Serper network error: {exc}",
        )
