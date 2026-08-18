"""Unit tests for Server 3 reformulation logic — fully mocked (no network calls).

Tests:
  - generate_paraphrases: JSON parsing, markdown code-fence stripping, non-JSON LLM output
  - test_reformulation_stability: stability score computation, DB persistence, semaphore
"""
from __future__ import annotations

import asyncio
import json
import sys
import os
import tempfile
from unittest.mock import AsyncMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from server_reformulation.main import (
    generate_paraphrases,
    test_reformulation_stability as stability_tool,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MOCK_VARIANTS_JSON = json.dumps([
    {"text": "What is the purpose of the MRTP Act 1966?", "kind": "formal"},
    {"text": "What does the MRTP Act do?", "kind": "informal"},
    {"text": "The MRTP Act 1966 — what is its primary purpose?", "kind": "reorder"},
])

MOCK_VARIANTS_WITH_FENCE = f"```json\n{MOCK_VARIANTS_JSON}\n```"


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_paraphrases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_generate_paraphrases_valid_json():
    """LLM returns valid JSON array → ok=True with clean variants."""
    with patch("server_reformulation.main._call_nvidia_nim", new=AsyncMock(return_value=MOCK_VARIANTS_JSON)):
        result = await generate_paraphrases("What is the MRTP Act?", n=3)

    assert result["ok"] is True
    assert len(result["variants"]) == 3
    assert result["variants"][0]["kind"] == "formal"
    assert result["original_query"] == "What is the MRTP Act?"


@pytest.mark.asyncio
async def test_generate_paraphrases_strips_code_fence():
    """LLM wraps output in ```json ... ``` code fence → still parsed correctly."""
    with patch("server_reformulation.main._call_nvidia_nim", new=AsyncMock(return_value=MOCK_VARIANTS_WITH_FENCE)):
        result = await generate_paraphrases("What is the MRTP Act?", n=3)

    assert result["ok"] is True
    assert len(result["variants"]) == 3


@pytest.mark.asyncio
async def test_generate_paraphrases_invalid_json():
    """LLM returns garbage text → ok=False with descriptive error."""
    with patch("server_reformulation.main._call_nvidia_nim", new=AsyncMock(return_value="Sorry, I cannot help.")):
        result = await generate_paraphrases("What is the MRTP Act?")

    assert result["ok"] is False
    assert "invalid JSON" in result["error"] or "JSON" in result["error"]


@pytest.mark.asyncio
async def test_generate_paraphrases_non_array_json():
    """LLM returns a JSON object (not array) → ok=False."""
    with patch("server_reformulation.main._call_nvidia_nim", new=AsyncMock(return_value='{"text": "...", "kind": "formal"}')):
        result = await generate_paraphrases("test query")

    assert result["ok"] is False


@pytest.mark.asyncio
async def test_generate_paraphrases_n_clamped():
    """n is clamped to [1, 8]; LLM is still called with the clamped value."""
    call_args: list = []

    async def mock_nim(prompt: str) -> str:
        call_args.append(prompt)
        return MOCK_VARIANTS_JSON

    with patch("server_reformulation.main._call_nvidia_nim", new=mock_nim):
        result = await generate_paraphrases("q", n=100)

    assert result["ok"] is True
    # n=100 should have been clamped to 8 in the prompt
    assert "8" in call_args[0]


@pytest.mark.asyncio
async def test_generate_paraphrases_missing_api_key():
    """Raises ValueError when NVIDIA_API_KEY is empty → ok=False."""
    with patch("server_reformulation.main.config") as mock_cfg:
        mock_cfg.NVIDIA_API_KEY = ""
        mock_cfg.NVIDIA_BASE_URL = "https://example.com"
        mock_cfg.NVIDIA_MODEL = "test-model"
        mock_cfg.NVIDIA_LLM_TIMEOUT_S = 5.0

        result = await generate_paraphrases("test")

    assert result["ok"] is False
    assert "NVIDIA_API_KEY" in result["error"]


# ─────────────────────────────────────────────────────────────────────────────
# Tests: test_reformulation_stability (mocked chat + DB)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_log_chat(gr_numbers: list[str | None]):
    """Build a _log_chat_call mock that returns the given gr_numbers in sequence."""
    call_idx = [0]

    async def mock_log(message: str, web_search: bool) -> dict:
        i = min(call_idx[0], len(gr_numbers) - 1)
        call_idx[0] += 1
        gr = gr_numbers[i]
        return {
            "ok": gr is not None,
            "message_id": f"msg-{i}",
            "top_gr_number": gr,
            "top_filepath": f"/docs/{gr}.pdf" if gr else None,
            "error": None if gr else "no sources",
        }

    return mock_log


@pytest.mark.asyncio
async def test_stability_all_agree():
    """When all variants retrieve the same GR, stability_score=1.0."""
    variants = [
        {"text": "Formal variant", "kind": "formal"},
        {"text": "Informal variant", "kind": "informal"},
        {"text": "Reorder variant", "kind": "reorder"},
    ]
    # original + 3 variants → 4 calls, all returning "GR-001"
    gr_seq = ["GR-001", "GR-001", "GR-001", "GR-001"]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        from mcp_common import store as real_store
        with patch("server_reformulation.main._log_chat_call", new=_make_mock_log_chat(gr_seq)), \
             patch("server_reformulation.main.get_db", new=lambda: real_store.get_db(db_path)), \
             patch("server_reformulation.main.execute_with_retry", new=real_store.execute_with_retry):
            # Init schema
            conn = await real_store.get_db(db_path)
            await conn.close()

            result = await stability_tool(
                "What is the MRTP Act?",
                variants=variants,
                web_search=False,
            )

    assert result["ok"] is True
    assert result["stability_score"] == 1.0
    assert result["majority_gr_number"] == "GR-001"
    assert all(not r["odd_one_out"] for r in result["results"])


@pytest.mark.asyncio
async def test_stability_one_odd_out():
    """When one variant retrieves a different GR, it is flagged as odd_one_out."""
    variants = [
        {"text": "Formal variant", "kind": "formal"},
        {"text": "Informal variant", "kind": "informal"},
        {"text": "Reorder variant", "kind": "reorder"},
    ]
    # original=GR-001, formal=GR-001, informal=GR-001, reorder=GR-999
    gr_seq = ["GR-001", "GR-001", "GR-001", "GR-999"]

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        from mcp_common import store as real_store
        with patch("server_reformulation.main._log_chat_call", new=_make_mock_log_chat(gr_seq)), \
             patch("server_reformulation.main.get_db", new=lambda: real_store.get_db(db_path)), \
             patch("server_reformulation.main.execute_with_retry", new=real_store.execute_with_retry):
            conn = await real_store.get_db(db_path)
            await conn.close()

            result = await stability_tool(
                "test query",
                variants=variants,
                web_search=False,
            )

    assert result["ok"] is True
    assert abs(result["stability_score"] - 0.75) < 1e-9
    odd_kinds = [r["kind"] for r in result["results"] if r["odd_one_out"]]
    assert "reorder" in odd_kinds


@pytest.mark.asyncio
async def test_stability_all_null_gr():
    """When no sources are returned, stability_score=0.0 and majority_gr=None."""
    variants = [{"text": "Formal", "kind": "formal"}]
    gr_seq = [None, None]  # original + formal both null

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        from mcp_common import store as real_store
        with patch("server_reformulation.main._log_chat_call", new=_make_mock_log_chat(gr_seq)), \
             patch("server_reformulation.main.get_db", new=lambda: real_store.get_db(db_path)), \
             patch("server_reformulation.main.execute_with_retry", new=real_store.execute_with_retry):
            conn = await real_store.get_db(db_path)
            await conn.close()

            result = await stability_tool("test", variants=variants, web_search=False)

    assert result["ok"] is True
    assert result["stability_score"] == 0.0
    assert result["majority_gr_number"] is None
