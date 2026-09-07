"""Regression tests for query_log identifiers written by query_and_log."""
from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import httpx
import pytest

from mahasamvaad_eval.storage.db import get_db
from mahasamvaad_eval.tools.observability import handle_query_and_log as query_and_log


class _MockAsyncClient:
    def __init__(self, handler, **kwargs):
        self._handler = handler

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        pass

    async def post(self, url, **kwargs):
        return await self._handler(url, **kwargs)


async def _query_and_read_logged_chatroom(chatroom_id: str | None) -> tuple[str, str | None]:
    captured_payloads: list[dict] = []

    async def handler(url, **kwargs):
        captured_payloads.append(kwargs["json"])
        return httpx.Response(200, json={"response": "ok", "metadata": {}})

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        with patch(
            "mahasamvaad_eval.tools.observability.get_db",
            new=lambda: get_db(db_path),
        ), patch(
            "mahasamvaad_eval.clients.chat_api_client.httpx.AsyncClient",
            new=lambda **kwargs: _MockAsyncClient(handler, **kwargs),
        ):
            result = await query_and_log("test query", chatroom_id=chatroom_id)

        conn = await get_db(db_path)
        try:
            rows = await conn.execute_fetchall("SELECT chatroom_id FROM query_log WHERE id = ?", (result["logged_row_id"],))
        finally:
            await conn.close()

    return captured_payloads[0]["chatroom_id"], rows[0]["chatroom_id"]


@pytest.mark.asyncio
async def test_query_and_log_persists_explicit_chatroom_id():
    """An explicit chatroom ID must be sent upstream and written unchanged to query_log."""
    upstream_id, logged_id = await _query_and_read_logged_chatroom("test-room-123")

    assert upstream_id == "test-room-123"
    assert logged_id == "test-room-123"


@pytest.mark.asyncio
async def test_query_and_log_persists_generated_chatroom_id():
    """An omitted chatroom ID must be generated once, sent upstream, and logged unchanged."""
    upstream_id, logged_id = await _query_and_read_logged_chatroom(None)

    assert upstream_id.startswith("chatroom-")
    assert logged_id is not None
    assert logged_id == upstream_id
