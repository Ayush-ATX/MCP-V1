"""Unit tests for mcp_common/chat_client.py — mocked httpx transport.

Strategy: patch `mcp_common.chat_client.httpx` with a fake module whose
AsyncClient is a context-manager that uses our mock transport directly,
so there's no recursion and no dependency on internal httpx internals.
"""
from __future__ import annotations

import json
import sys
import os
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import httpx

from mcp_common.chat_client import call_chat


# ── Response fixture ──────────────────────────────────────────────────────────
GOOD_BODY = {
    "response": "The Maharashtra Regional and Town Planning Act governs town planning.",
    "last_turn_filepaths": [],
    "metadata": {
        "intent": "policy_exploration",
        "language": "en",
        "route_hint": "gr_corpus",
        "sources": [
            {
                "citation_id": "c1",
                "filename": "mrtp.pdf",
                "filepath": "/docs/mrtp.pdf",
                "gr_number": "GR-2019-UD-001",
                "department": "Urban Development",
                "anchors": [{"page_number": 1, "start_text": "The Act governs", "end_text": "town planning."}],
            }
        ],
        "web_sources": [],
        "latency_seconds": 1.23,
        "model": "gemini-pro",
        "tokens": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        "message_id": "server-msg-id-123",
        "langfuse_trace_id": "trace-abc",
    },
}


# ── Helper: create a fake httpx.AsyncClient context manager ──────────────────
def _fake_client(responses: list[httpx.Response | Exception]):
    """Returns a context-manager class that yields a fake client.

    responses: list of Response objects or Exceptions to return/raise in order.
               The last element is reused if the list is exhausted.
    """
    call_idx = [0]
    captured_requests: list = []

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            captured_requests.append(kwargs)
            i = min(call_idx[0], len(responses) - 1)
            call_idx[0] += 1
            item = responses[i]
            if isinstance(item, Exception):
                raise item
            return item

    class FakeAsyncClient:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return FakeClient()

        def __exit__(self, *args):
            pass

        def __call__(self, **kwargs):
            return FakeClient()

    client = FakeClient()
    client._call_idx = call_idx
    client._captured = captured_requests

    return client, call_idx, captured_requests


# Simpler approach: patch the _do_request function directly via call_chat internals
# by patching httpx.AsyncClient at the module level with a class that doesn't recurse.

def _make_mock_client_class(response_factory):
    """Create a class that can be used as httpx.AsyncClient in tests."""

    class MockAsyncClient:
        def __init__(self, **kwargs):
            self._response_factory = response_factory

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def post(self, url, **kwargs):
            return await self._response_factory(url, **kwargs)

    return MockAsyncClient


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_successful_call():
    """A 200 response is parsed into ChatResult(ok=True) with all fields."""
    resp = httpx.Response(200, json=GOOD_BODY)

    async def handler(url, **kw):
        return resp

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient):
        result = await call_chat("What is the MRTP Act?")

    assert result.ok is True
    assert result.message_id == "server-msg-id-123"
    assert result.intent == "policy_exploration"
    assert result.total_tokens == 150
    assert result.sources[0]["gr_number"] == "GR-2019-UD-001"


@pytest.mark.asyncio
async def test_timeout_returns_ok_false():
    """ConnectTimeout returns ChatResult(ok=False) and does NOT raise."""
    call_count = [0]

    async def handler(url, **kw):
        call_count[0] += 1
        raise httpx.ConnectTimeout("timed out", request=httpx.Request("POST", url))

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient), \
         patch("mcp_common.chat_client._RETRY_BACKOFF_S", 0.0):
        result = await call_chat("test message")

    assert result.ok is False
    assert result.http_status == -1
    assert call_count[0] == 2  # original + 1 retry


@pytest.mark.asyncio
async def test_5xx_retries_once():
    """5xx response retries once then returns ok=False."""
    call_count = [0]

    async def handler(url, **kw):
        call_count[0] += 1
        return httpx.Response(503, text="Service Unavailable")

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient), \
         patch("mcp_common.chat_client._RETRY_BACKOFF_S", 0.0):
        result = await call_chat("test")

    assert result.ok is False
    assert result.http_status == 503
    assert call_count[0] == 2  # one retry on 5xx


@pytest.mark.asyncio
async def test_4xx_no_retry():
    """4xx response does NOT retry — returns immediately."""
    call_count = [0]

    async def handler(url, **kw):
        call_count[0] += 1
        return httpx.Response(422, text="Unprocessable Entity")

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient):
        result = await call_chat("bad request")

    assert result.ok is False
    assert result.http_status == 422
    assert call_count[0] == 1  # no retry on 4xx


@pytest.mark.asyncio
async def test_malformed_json_returns_ok_false():
    """200 response with malformed JSON body returns ok=False without raising."""
    async def handler(url, **kw):
        return httpx.Response(200, content=b"not valid json{{", headers={"Content-Type": "application/json"})

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient):
        result = await call_chat("test")

    assert result.ok is False
    assert result.http_status == 200
    assert result.error is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        [],
        {"response": "hello", "metadata": "invalid"},
        {"response": "hello", "metadata": {"tokens": "invalid"}},
        {"response": "hello", "metadata": {"sources": {"citation_id": "c1"}}},
        {"response": "hello", "metadata": {"web_sources": {"ref": "w1"}}},
        {"response": "hello", "last_turn_filepaths": "not-a-list"},
    ],
)
async def test_invalid_response_structure_returns_ok_false(body):
    """A 200 response with an unexpected JSON shape must not cross the MCP boundary."""
    async def handler(url, **kw):
        return httpx.Response(200, json=body)

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient):
        result = await call_chat("test")

    assert result.ok is False
    assert result.http_status == 200
    assert result.error is not None


@pytest.mark.asyncio
async def test_read_error_returns_ok_false_after_retry():
    """ReadError is a normal transport failure and must follow the network retry path."""
    call_count = [0]

    async def handler(url, **kw):
        call_count[0] += 1
        raise httpx.ReadError("connection reset", request=httpx.Request("POST", url))

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient), \
         patch("mcp_common.chat_client._RETRY_BACKOFF_S", 0.0):
        result = await call_chat("test")

    assert result.ok is False
    assert result.http_status == -1
    assert call_count[0] == 2  # original + 1 retry


@pytest.mark.asyncio
async def test_uuid_generated_per_call():
    """Each call generates distinct user_message_id values."""
    ids_seen: list[str] = []

    async def handler(url, **kw):
        body = kw.get("json", {})
        ids_seen.append(body.get("user_message_id", ""))
        return httpx.Response(200, json=GOOD_BODY)

    MockClient = _make_mock_client_class(handler)
    with patch("mcp_common.chat_client.httpx.AsyncClient", MockClient):
        await call_chat("q1")
        await call_chat("q2")

    assert len(ids_seen) == 2
    assert ids_seen[0] != ids_seen[1], "Each call must generate a unique user_message_id"
    assert all(len(uid) > 0 for uid in ids_seen)
