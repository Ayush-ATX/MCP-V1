"""tests/test_clients.py — Unit tests for shared client wrappers."""
from __future__ import annotations

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mahasamvaad_eval.clients.chat_api_client import ChatRequest, call_chat
from mahasamvaad_eval.clients.embedding_client import EmbeddingClient, cosine_similarity
from mahasamvaad_eval.clients.llm_judge_client import LLMJudgeClient, _clean_json_text
from mahasamvaad_eval.clients.serper_client import search_web
from mahasamvaad_eval.clients.storage_client import fetch_document


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


# ── 4. FalkorDB Read-Only Security & Client Tests ────────────────────────────
def test_falkordb_read_only_firewall():
    from mahasamvaad_eval.clients.falkordb_client import SecurityViolationError, validate_read_only_cypher

    # Read queries must pass
    valid_queries = [
        "MATCH (n:GR) RETURN n LIMIT 5",
        "OPTIONAL MATCH (a)-[r:SUPERSEDES]->(b) RETURN a, r, b",
        "WITH $param AS p MATCH (x:GR {id: p}) RETURN x",
        "UNWIND [1, 2, 3] AS x RETURN x",
    ]
    for q in valid_queries:
        validate_read_only_cypher(q)

    # Mutating queries must raise SecurityViolationError
    mutating_queries = [
        "CREATE (n:GR {gr_number: '123'})",
        "MATCH (n:GR) DELETE n",
        "MATCH (n:GR) DETACH DELETE n",
        "MATCH (n:GR) SET n.is_active = false",
        "MERGE (n:GR {gr_number: '123'})",
        "MATCH (n:GR) REMOVE n.prop",
        "CALL dbms.clear()",
        "CALL db.index.fulltext.createNodeIndex('idx', ['GR'], ['name'])",
        "DROP INDEX ON :GR(gr_number)",
        "LOAD CSV FROM 'file:///malicious.csv' AS line",
    ]
    for q in mutating_queries:
        with pytest.raises(SecurityViolationError):
            validate_read_only_cypher(q)


def test_falkordb_client_get_gr_lineage():
    from mahasamvaad_eval.clients.falkordb_client import FalkorDBClient

    client = FalkorDBClient()
    mock_outgoing = [["20230525", "Forest", True, "AMENDS", "20210101", "Forest", False, "Earlier GR"]]
    mock_incoming = [["20240901", "Forest", True, "Later GR", "SUPERSEDES", "20230525", "Forest", False]]

    with patch.object(client, "ro_query", side_effect=[mock_outgoing, mock_incoming]):
        lineage = client.get_gr_lineage("20230525")
        assert len(lineage) == 2
        assert lineage[0]["relation_type"] == "amends"
        assert lineage[0]["direction"] == "outgoing"
        assert lineage[1]["relation_type"] == "supersedes"
        assert lineage[1]["direction"] == "incoming"
