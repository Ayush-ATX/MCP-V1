"""tests/test_clients.py — Unit tests for shared client wrappers."""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.chat_api_client import ChatRequest, call_chat
from clients.embedding_client import EmbeddingClient, cosine_similarity
from clients.llm_judge_client import LLMJudgeClient, _clean_json_text
from clients.serper_client import search_web
from clients.storage_client import fetch_document


# ── 1. Storage Client Tests ──────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_storage_client_success_and_error():
    mock_resp_200 = httpx.Response(200, content=b"GR Document Content for Urban Development")
    mock_resp_404 = httpx.Response(404, text="Not Found")

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp_200)):
        res = await fetch_document("/raw/gr/urban/doc123.txt")
        assert res.ok is True
        assert res.extracted_text == "GR Document Content for Urban Development"

    with patch("httpx.AsyncClient.get", AsyncMock(return_value=mock_resp_404)):
        res_fail = await fetch_document("/raw/gr/urban/missing.pdf")
        assert res_fail.ok is False
        assert "404" in (res_fail.error or "")


# ── 2. LLM Judge Client Tests ────────────────────────────────────────────────
def test_clean_json_text_helper():
    fenced = '```json\n{"score": 9.0, "rationale": "good"}\n```'
    assert _clean_json_text(fenced) == '{"score": 9.0, "rationale": "good"}'

    raw_surrounded = 'Analysis:\n{"score": 7.5}\nHope this helps!'
    assert _clean_json_text(raw_surrounded) == '{"score": 7.5}'


@pytest.mark.asyncio
async def test_llm_judge_evaluates_and_parses_json():
    client = LLMJudgeClient(api_key="mock-key")
    mock_llm_output = '```json\n{"score": 8.5, "rationale": "Accurate subsidy figures"}\n```'

    with patch.object(client, "_call_model", AsyncMock(return_value=mock_llm_output)):
        eval_res = await client.evaluate_json("Test prompt")
        assert eval_res.ok is True
        assert eval_res.score == 8.5
        assert "Accurate subsidy" in (eval_res.rationale or "")


# ── 3. Serper Client Tests ───────────────────────────────────────────────────
@pytest.mark.asyncio
async def test_serper_client():
    mock_serper_body = {
        "organic": [
            {"title": "Maha Solar Pump Scheme", "link": "https://example.com/solar", "snippet": "Official scheme info"}
        ]
    }
    mock_resp = httpx.Response(200, json=mock_serper_body)

    with patch("httpx.AsyncClient.post", AsyncMock(return_value=mock_resp)):
        res = await search_web("Maharashtra solar pump", api_key="mock-serper-key")
        assert res.ok is True
        assert len(res.organic) == 1
        assert res.organic[0]["title"] == "Maha Solar Pump Scheme"
