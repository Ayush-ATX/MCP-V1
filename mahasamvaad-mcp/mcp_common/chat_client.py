"""mcp_common/chat_client.py — §4.2 shared HTTP wrapper for POST /api/v1/chat.

Design:
- Generates user_message_id / message_id (uuid4) client-side when not supplied.
- 60s connect+read timeout (corpus + web-fusion calls are slow).
- One retry on ConnectionError or 5xx with 2s backoff; no retry on 4xx.
- Never raises across the MCP boundary — returns ChatResult(ok=False, ...) on all failures.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

from mcp_common.config import MAHASAMVAAD_BASE_URL, MCP_DEFAULT_USER_ID, MCP_HTTP_TIMEOUT_S

logger = logging.getLogger(__name__)

_CHAT_ENDPOINT = "/api/v1/chat"
_RETRY_BACKOFF_S = 2.0


@dataclass
class ChatResult:
    ok: bool
    # Success fields
    message_id: str | None = None
    user_message_id: str | None = None
    response: str | None = None
    intent: str | None = None
    language: str | None = None
    route_hint: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    web_sources: list[dict[str, Any]] = field(default_factory=list)
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_seconds: float | None = None
    langfuse_trace_id: str | None = None
    model: str | None = None
    last_turn_filepaths: list[str] = field(default_factory=list)
    raw_response: dict[str, Any] | None = None
    # Failure fields
    error: str | None = None
    http_status: int | None = None


async def call_chat(
    message: str,
    *,
    web_search: bool = True,
    model_name: str | None = None,
    history: list[dict] | None = None,
    last_turn_filepaths: list[str] | None = None,
    chatroom_id: str | None = None,
    user_id: str | None = None,
) -> ChatResult:
    """Call POST /api/v1/chat and return a structured ChatResult.

    Never raises — all errors are captured and returned as ChatResult(ok=False, ...).
    """
    # Generate client-side IDs so every row is joinable even on failures
    generated_user_msg_id = str(uuid.uuid4())
    generated_msg_id = str(uuid.uuid4())
    effective_user_id = user_id or MCP_DEFAULT_USER_ID
    effective_chatroom_id = chatroom_id or f"chatroom-{str(uuid.uuid4())[:8]}"

    payload: dict[str, Any] = {
        "user_id": effective_user_id,
        "chatroom_id": effective_chatroom_id,
        "user_message_id": generated_user_msg_id,
        "message_id": generated_msg_id,
        "message": message,
        "history": history or [],
        "last_turn_filepaths": last_turn_filepaths or [],
    }
    if model_name:
        payload["model_name"] = model_name

    url = f"{MAHASAMVAAD_BASE_URL.rstrip('/')}{_CHAT_ENDPOINT}"
    params = {"web_search": str(web_search).lower()}
    timeout = httpx.Timeout(float(MCP_HTTP_TIMEOUT_S))

    async def _do_request() -> httpx.Response:
        async with httpx.AsyncClient(timeout=timeout) as client:
            return await client.post(url, json=payload, params=params)

    last_error: str | None = None
    last_status: int | None = None
    raw_body: dict | None = None

    for attempt in range(2):  # 1 retry on 5xx / connection error
        try:
            resp = await _do_request()
            last_status = resp.status_code

            if resp.status_code == 200:
                try:
                    raw_body = resp.json()
                except (json.JSONDecodeError, ValueError) as exc:
                    last_error = f"Malformed JSON in response: {exc}"
                    logger.error("call_chat: malformed JSON — %s", exc)
                    return ChatResult(
                        ok=False,
                        user_message_id=generated_user_msg_id,
                        message_id=generated_msg_id,
                        error=last_error,
                        http_status=200,
                    )
                return _parse_success(raw_body, generated_user_msg_id, generated_msg_id)

            # 4xx — do not retry
            if 400 <= resp.status_code < 500:
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                logger.warning("call_chat: 4xx response (%d), not retrying", resp.status_code)
                return ChatResult(
                    ok=False,
                    user_message_id=generated_user_msg_id,
                    message_id=generated_msg_id,
                    error=last_error,
                    http_status=resp.status_code,
                )

            # 5xx — retry once
            last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
            logger.warning("call_chat: 5xx response (%d) attempt %d", resp.status_code, attempt + 1)

        except (httpx.ConnectError, httpx.ReadTimeout, httpx.ConnectTimeout) as exc:
            last_error = f"Connection/timeout error: {exc}"
            last_status = -1
            logger.warning("call_chat: connection error attempt %d — %s", attempt + 1, exc)

        if attempt == 0:
            logger.info("call_chat: retrying after %ss backoff", _RETRY_BACKOFF_S)
            await asyncio.sleep(_RETRY_BACKOFF_S)

    return ChatResult(
        ok=False,
        user_message_id=generated_user_msg_id,
        message_id=generated_msg_id,
        error=last_error,
        http_status=last_status,
    )


def _parse_success(body: dict, user_message_id: str, fallback_msg_id: str) -> ChatResult:
    """Parse a successful 200 response body into a ChatResult."""
    metadata: dict = body.get("metadata", {}) or {}
    tokens: dict = metadata.get("tokens", {}) or {}
    sources: list = metadata.get("sources", []) or []
    web_sources: list = metadata.get("web_sources", []) or []

    server_msg_id = metadata.get("message_id") or fallback_msg_id

    return ChatResult(
        ok=True,
        user_message_id=user_message_id,
        message_id=server_msg_id,
        response=body.get("response", ""),
        last_turn_filepaths=body.get("last_turn_filepaths", []) or [],
        intent=metadata.get("intent"),
        language=metadata.get("language"),
        route_hint=metadata.get("route_hint"),
        sources=sources,
        web_sources=web_sources,
        prompt_tokens=tokens.get("prompt_tokens"),
        completion_tokens=tokens.get("completion_tokens"),
        total_tokens=tokens.get("total_tokens"),
        latency_seconds=metadata.get("latency_seconds"),
        langfuse_trace_id=metadata.get("langfuse_trace_id"),
        model=metadata.get("model"),
        raw_response=body,
        http_status=200,
    )
