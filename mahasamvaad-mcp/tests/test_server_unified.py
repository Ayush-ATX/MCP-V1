"""tests/test_server_unified.py — Test tool registrations and basic contracts of unified server."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import mcp


@pytest.mark.asyncio
async def test_all_eight_tool_namespaces_registered():
    """Verify that all 8 required tool namespaces are registered on the unified MCP server."""
    tool_names = set(mcp._tool_manager._tools.keys())

    # Namespaced tool identifiers
    expected_namespaced_tools = [
        "observability.query_and_log",
        "observability.get_dashboard_stats",
        "observability.list_recent_queries",
        "grounding.score_citation_coverage",
        "grounding.grounding_trend",
        "reformulation.generate_paraphrases",
        "reformulation.test_paraphrase_robustness",
        "aieval.detect_hallucinated_entities",
        "aieval.check_lineage_correctness",
        "aieval.evaluate_bilingual_parity",
        "aieval.run_refusal_redteam_suite",
        "aieval.check_corpus_web_precedence",
    ]

    for tool_name in expected_namespaced_tools:
        assert tool_name in tool_names, f"Expected tool '{tool_name}' not registered on server"


@pytest.mark.asyncio
async def test_backward_compatible_tool_aliases_registered():
    """Verify that backward-compatible tool aliases are registered on the unified MCP server."""
    tool_names = set(mcp._tool_manager._tools.keys())

    expected_aliases = [
        "query_and_log",
        "get_dashboard_stats",
        "list_recent_queries",
        "score_grounding",
        "grounding_trend",
        "generate_paraphrases",
        "test_reformulation_stability",
        "detect_hallucinated_entities",
        "check_lineage_correctness",
        "evaluate_bilingual_parity",
        "run_refusal_redteam_suite",
        "check_corpus_web_precedence",
    ]

    for alias in expected_aliases:
        assert alias in tool_names, f"Expected alias '{alias}' not registered on server"


@pytest.mark.asyncio
async def test_resources_registered():
    """Verify that query_log resources and templates are registered."""
    resource_uris = set(mcp._resource_manager._resources.keys())
    template_uris = set(mcp._resource_manager._templates.keys())
    assert "query_log://recent" in resource_uris
    assert "query_log://recent{?limit,intent}" in template_uris

