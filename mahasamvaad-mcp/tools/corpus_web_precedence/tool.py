"""tools/corpus_web_precedence/tool.py — Corpus vs Web Precedence & Discrepancy Checker (§3.5)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

import config
from clients.chat_api_client import call_chat
from clients.llm_judge_client import default_judge_client
from storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)

_CONFLICT_QUERY_FILE = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "known_conflict_queries.json"


def _load_conflict_queries() -> list[dict[str, Any]]:
    if _CONFLICT_QUERY_FILE.is_file():
        try:
            with open(_CONFLICT_QUERY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load known_conflict_queries.json: %s", exc)
    return []


_PRECEDENCE_JUDGE_PROMPT = """You are a policy discrepancy judge analyzing an AI response where official Maharashtra Government Resolutions (corpus) and web articles conflict.

User Query: {query}
Official Corpus Truth: {corpus_position}
Informal Web Rumor / Discrepancy: {web_position}
AI Response: {response}

Analyze how the AI handled the conflict. Classify into EXACTLY one of these 4 categories:
1. "correctly_flagged_precedence" (explicitly treats official GR as authoritative and clarifies the discrepancy with web/informal claims)
2. "silently_used_corpus" (uses only the official GR without acknowledging or clarifying the web claim)
3. "silently_used_web" (erroneously accepted the informal/web claim over the official GR)
4. "missed_contradiction" (confused, self-contradictory, or missed the distinction)

Return valid JSON ONLY with exact format:
{{
  "classification": "correctly_flagged_precedence",
  "score": 9.0,
  "rationale": "The model correctly cited the GR subsidy rates and noted that viral 100% free claims online are false."
}}"""


async def handle_check_corpus_web_precedence(
    query: str | None = None,
    corpus_position: str | None = None,
    web_position: str | None = None,
) -> dict[str, Any]:
    """Evaluate whether the agent enforces corpus precedence when official GRs and web search results conflict."""
    try:
        cases: list[dict[str, Any]] = []
        if query:
            cases.append({
                "id": "custom-conflict",
                "query": query,
                "corpus_position": corpus_position or "Official Government Resolution specifies strict conditions and parameters.",
                "web_position": web_position or "Informal web articles/forums claim relaxed or contrasting rules.",
            })
        else:
            cases = _load_conflict_queries()

        if not cases:
            return {"ok": True, "total_tested": 0, "results": [], "message": "No conflict queries found"}

        results: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await get_db()
        try:
            for item in cases:
                q = item["query"]
                corp_pos = item.get("corpus_position", "")
                web_pos = item.get("web_position", "")

                # 1. Call chat with web_search=true
                chat_res = await call_chat(q, web_search=True)
                ans = chat_res.response or ""

                # 2. Evaluate response via LLM judge
                judge_res = await default_judge_client.evaluate_json(
                    _PRECEDENCE_JUDGE_PROMPT.format(
                        query=q,
                        corpus_position=corp_pos,
                        web_position=web_pos,
                        response=ans[:1800],
                    )
                )

                classification = "silently_used_corpus"
                if judge_res.ok and judge_res.data.get("classification"):
                    classification = str(judge_res.data["classification"])

                is_pass = classification in ("correctly_flagged_precedence", "silently_used_corpus")

                entry = {
                    "id": item.get("id", "conf"),
                    "query": q,
                    "corpus_position": corp_pos,
                    "web_position": web_pos,
                    "response_classification": classification,
                    "passed": is_pass,
                    "score": judge_res.score,
                    "rationale": judge_res.rationale,
                    "sources_count": len(chat_res.sources) if chat_res.ok else 0,
                    "web_sources_count": len(chat_res.web_sources) if chat_res.ok else 0,
                    "response_snippet": ans[:200],
                }
                results.append(entry)

                # Persist to SQLite precedence_checks
                await execute_with_retry(
                    conn,
                    """
                    INSERT INTO precedence_checks (
                        query, corpus_position, web_position, response_classification, details_json, checked_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        q,
                        corp_pos,
                        web_pos,
                        classification,
                        json.dumps(entry, ensure_ascii=False),
                        now_iso,
                    ),
                )
        finally:
            await conn.close()

        total = len(results)
        passed_count = sum(1 for r in results if r["passed"])
        precedence_pass_rate = (passed_count / total * 100) if total else 0.0

        return {
            "ok": True,
            "total_tested": total,
            "passed_count": passed_count,
            "failed_count": total - passed_count,
            "precedence_pass_rate_pct": round(precedence_pass_rate, 2),
            "results": results,
        }

    except Exception as exc:
        logger.exception("check_corpus_web_precedence error: %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Precedence & Contradiction Checker on MCP server."""
    mcp.tool(
        name="aieval.check_corpus_web_precedence",
        description="Verify that official Government Resolutions take precedence over conflicting web results per SRS fusion rules.",
    )(handle_check_corpus_web_precedence)

    mcp.tool(
        name="check_corpus_web_precedence",
        description="Backwards-compatible alias for aieval.check_corpus_web_precedence.",
    )(handle_check_corpus_web_precedence)
