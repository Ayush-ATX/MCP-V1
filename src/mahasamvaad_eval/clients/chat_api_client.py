"""clients/chat_api_client.py — Robust HTTP client wrapper for POST /api/v1/chat."""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx

import config

logger = logging.getLogger(__name__)

_RETRY_BACKOFF_S = 2.0


@dataclass
class ChatRequest:
    user_id: str
    chatroom_id: str
    user_message_id: str
    message_id: str
    message: str
    model_name: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    last_turn_filepaths: list[str] = field(default_factory=list)

    def to_payload(self) -> dict[str, Any]:
        p: dict[str, Any] = {
            "user_id": self.user_id,
            "chatroom_id": self.chatroom_id,
            "user_message_id": self.user_message_id,
            "message_id": self.message_id,
            "message": self.message,
            "history": self.history,
            "last_turn_filepaths": self.last_turn_filepaths,
        }
        if self.model_name:
            p["model_name"] = self.model_name
        return p


@dataclass
class ChatResult:
    ok: bool
    # Success fields
    message_id: str | None = None
    user_message_id: str | None = None
    chatroom_id: str | None = None
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
    base_url: str | None = None,
    chat_path: str | None = None,
    timeout_s: float | None = None,
) -> ChatResult:
    """Call POST /api/v1/chat and return a structured ChatResult.

    Never raises — all errors are captured and returned as ChatResult(ok=False, ...).
    """
    generated_user_msg_id = str(uuid.uuid4())
    generated_msg_id = str(uuid.uuid4())
    effective_user_id = user_id or config.MCP_DEFAULT_USER_ID
    effective_chatroom_id = chatroom_id or f"chatroom-{str(uuid.uuid4())[:8]}"

    req = ChatRequest(
        user_id=effective_user_id,
        chatroom_id=effective_chatroom_id,
        user_message_id=generated_user_msg_id,
        message_id=generated_msg_id,
        message=message,
        model_name=model_name,
        history=history or [],
        last_turn_filepaths=last_turn_filepaths or [],
    )

    base = (base_url or config.CHAT_API_BASE_URL or config.MAHASAMVAAD_BASE_URL).rstrip("/")
    path = chat_path or config.CHAT_API_CHAT_PATH
    url = f"{base}{path}"
    params = {"web_search": str(web_search).lower()}
    timeout = httpx.Timeout(float(timeout_s or config.MCP_HTTP_TIMEOUT_S))

    last_error: str | None = None
    last_status: int | None = None

    for attempt in range(2):
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(url, json=req.to_payload(), params=params)
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
                            chatroom_id=effective_chatroom_id,
                            error=last_error,
                            http_status=200,
                        )
                    return _parse_success(raw_body, generated_user_msg_id, generated_msg_id, effective_chatroom_id)

                # 4xx — client error, do not retry
                if 400 <= resp.status_code < 500:
                    last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                    logger.warning("call_chat: 4xx response (%d), not retrying", resp.status_code)
                    return ChatResult(
                        ok=False,
                        user_message_id=generated_user_msg_id,
                        message_id=generated_msg_id,
                        chatroom_id=effective_chatroom_id,
                        error=last_error,
                        http_status=resp.status_code,
                    )

                # 5xx — server error, retry once
                last_error = f"HTTP {resp.status_code}: {resp.text[:500]}"
                logger.warning("call_chat: 5xx response (%d) attempt %d", resp.status_code, attempt + 1)

        except httpx.RequestError as exc:
            last_error = f"Connection/timeout error: {exc}"
            last_status = -1
            logger.warning("call_chat: request error attempt %d — %s", attempt + 1, exc)

        if attempt == 0:
            logger.info("call_chat: retrying after %ss backoff", _RETRY_BACKOFF_S)
            await asyncio.sleep(_RETRY_BACKOFF_S)

    return ChatResult(
        ok=False,
        user_message_id=generated_user_msg_id,
        message_id=generated_msg_id,
        chatroom_id=effective_chatroom_id,
        error=last_error,
        http_status=last_status,
    )


def _parse_success(
    body: Any,
    user_message_id: str,
    fallback_msg_id: str,
    chatroom_id: str,
) -> ChatResult:
    """Parse a successful 200 response body into a ChatResult."""
    if not isinstance(body, dict):
        return _malformed_response(user_message_id, fallback_msg_id, chatroom_id, "top-level JSON must be an object")

    metadata = body.get("metadata", {}) or {}
    if not isinstance(metadata, dict):
        return _malformed_response(user_message_id, fallback_msg_id, chatroom_id, "metadata must be an object")

    tokens = metadata.get("tokens", {}) or {}
    if not isinstance(tokens, dict):
        return _malformed_response(user_message_id, fallback_msg_id, chatroom_id, "metadata.tokens must be an object")

    sources = metadata.get("sources", []) or []
    if not isinstance(sources, list) or not all(isinstance(source, dict) for source in sources):
        return _malformed_response(user_message_id, fallback_msg_id, chatroom_id, "metadata.sources must be an array of objects")

    web_sources = metadata.get("web_sources", []) or []
    if not isinstance(web_sources, list) or not all(isinstance(source, dict) for source in web_sources):
        return _malformed_response(user_message_id, fallback_msg_id, chatroom_id, "metadata.web_sources must be an array of objects")

    last_turn_filepaths = body.get("last_turn_filepaths", []) or []
    if not isinstance(last_turn_filepaths, list) or not all(isinstance(filepath, str) for filepath in last_turn_filepaths):
        return _malformed_response(user_message_id, fallback_msg_id, chatroom_id, "last_turn_filepaths must be an array of strings")

    server_msg_id = metadata.get("message_id") or fallback_msg_id

    return ChatResult(
        ok=True,
        user_message_id=user_message_id,
        message_id=server_msg_id,
        chatroom_id=chatroom_id,
        response=body.get("response", ""),
        last_turn_filepaths=last_turn_filepaths,
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


def _malformed_response(user_message_id: str, message_id: str, chatroom_id: str, detail: str) -> ChatResult:
    error = f"Malformed response: {detail}"
    logger.error("call_chat: %s", error)
    return ChatResult(
        ok=False,
        user_message_id=user_message_id,
        message_id=message_id,
        chatroom_id=chatroom_id,
        error=error,
        http_status=200,
    )
