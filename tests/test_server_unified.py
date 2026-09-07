"""tests/test_server_unified.py — Test tool registrations and basic contracts of unified server."""
from __future__ import annotations

import os
import sys
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mahasamvaad_eval.server import mcp


@pytest.mark.asyncio
async def test_all_twelve_canonical_tools_registered():
    """Verify that exactly 12 canonical tools across all 8 domains are registered on the unified MCP server."""
    tool_names = set(mcp._tool_manager._tools.keys())

    expected_canonical_tools = [
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

    assert len(tool_names) == 12, f"Expected exactly 12 tools, but found {len(tool_names)}: {sorted(tool_names)}"
    for tool_name in expected_canonical_tools:
        assert tool_name in tool_names, f"Expected canonical tool '{tool_name}' not registered on server"


@pytest.mark.asyncio
async def test_duplicate_and_unnamespaced_aliases_not_registered():
    """Verify that removed duplicate aliases are NOT registered on the unified MCP server."""
    tool_names = set(mcp._tool_manager._tools.keys())

    removed_aliases = [
        "query_and_log",
        "get_dashboard_stats",
        "list_recent_queries",
        "grounding.score_grounding",
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

    for alias in removed_aliases:
        assert alias not in tool_names, f"Removed alias '{alias}' should not be registered on server"


@pytest.mark.asyncio
async def test_resources_registered():
    """Verify that query_log resources and templates are registered."""
    resource_uris = set(mcp._resource_manager._resources.keys())
    template_uris = set(mcp._resource_manager._templates.keys())
    assert "query_log://recent" in resource_uris
    assert "query_log://recent{?limit,intent}" in template_uris

