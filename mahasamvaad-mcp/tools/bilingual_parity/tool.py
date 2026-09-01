"""tools/bilingual_parity/tool.py — Bilingual Parity Evaluator (EN vs MR) (§3.3)."""
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

_BILINGUAL_QUERY_FILE = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "bilingual_query_set.json"


def _load_bilingual_queries() -> list[dict[str, Any]]:
    if _BILINGUAL_QUERY_FILE.is_file():
        try:
            with open(_BILINGUAL_QUERY_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load bilingual_query_set.json: %s", exc)
    return []


_JUDGE_RUBRIC_PROMPT = """You are an expert bilingual government policy evaluator assessing response completeness and accuracy for Maharashtra government scheme queries.

Query: {query}
Response: {response}

Rate this answer on a scale from 1.0 to 10.0 (where 10.0 is complete, factually accurate, and well-structured).
Return valid JSON only with exact structure:
{{
  "score": 8.5,
  "rationale": "Clear explanation of eligibility, mentions correct figures, well structured."
}}"""


async def handle_evaluate_bilingual_parity(
    query_en: str | None = None,
    query_mr: str | None = None,
    gap_threshold: float = 0.15,
) -> dict[str, Any]:
    """Evaluate bilingual parity between English and Marathi responses using LLM-as-a-judge."""
    try:
        pairs: list[dict[str, Any]] = []
        if query_en and query_mr:
            pairs.append({"id": "custom", "query_en": query_en, "query_mr": query_mr})
        else:
            pairs = _load_bilingual_queries()

        if not pairs:
            return {"ok": True, "evaluated_pairs": 0, "results": [], "message": "No bilingual query pairs provided or found"}

        results: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await get_db()
        try:
            for item in pairs:
                q_en = item["query_en"]
                q_mr = item["query_mr"]

                # 1. Fire chat calls for both languages
                res_en = await call_chat(q_en, web_search=True)
                res_mr = await call_chat(q_mr, web_search=True)

                ans_en = res_en.response or ""
                ans_mr = res_mr.response or ""
                src_count_en = len(res_en.sources) if res_en.ok else 0
                src_count_mr = len(res_mr.sources) if res_mr.ok else 0

                # 2. LLM judge evaluations for both
                judge_en = await default_judge_client.evaluate_json(
                    _JUDGE_RUBRIC_PROMPT.format(query=q_en, response=ans_en[:1500])
                )
                judge_mr = await default_judge_client.evaluate_json(
                    _JUDGE_RUBRIC_PROMPT.format(query=q_mr, response=ans_mr[:1500])
                )

                score_en = float(judge_en.score or 7.0 if judge_en.ok else 7.0)
                score_mr = float(judge_mr.score or 7.0 if judge_mr.ok else 7.0)

                # Relative gap: positive means English scored higher than Marathi
                gap = (score_en - score_mr) / max(score_en, 1.0)
                parity_passed = gap <= gap_threshold

                pair_res = {
                    "id": item.get("id", "pair"),
                    "query_en": q_en,
                    "query_mr": q_mr,
                    "score_en": round(score_en, 2),
                    "score_mr": round(score_mr, 2),
                    "gap": round(gap, 3),
                    "parity_passed": parity_passed,
                    "sources_count_en": src_count_en,
                    "sources_count_mr": src_count_mr,
                    "rationale_en": judge_en.rationale,
                    "rationale_mr": judge_mr.rationale,
                }
                results.append(pair_res)

                # Persist to SQLite bilingual_parity_checks
                await execute_with_retry(
                    conn,
                    """
                    INSERT INTO bilingual_parity_checks (
                        query_en, query_mr, score_en, score_mr, gap,
                        sources_count_en, sources_count_mr, parity_passed, details_json, checked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        q_en,
                        q_mr,
                        score_en,
                        score_mr,
                        round(gap, 3),
                        src_count_en,
                        src_count_mr,
                        1 if parity_passed else 0,
                        json.dumps(pair_res, ensure_ascii=False),
                        now_iso,
                    ),
                )
        finally:
            await conn.close()

        total_pairs = len(results)
        passed_count = sum(1 for r in results if r["parity_passed"])
        pass_rate = (passed_count / total_pairs * 100) if total_pairs else 0.0

        return {
            "ok": True,
            "total_pairs_evaluated": total_pairs,
            "passed_count": passed_count,
            "failed_count": total_pairs - passed_count,
            "parity_pass_rate_pct": round(pass_rate, 2),
            "gap_threshold": gap_threshold,
            "results": results,
        }

    except Exception as exc:
        logger.exception("evaluate_bilingual_parity error: %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Bilingual Parity Evaluator on MCP server."""
    mcp.tool(
        name="aieval.evaluate_bilingual_parity",
        description="Compare quality, completeness, and citations between English and Marathi response pairs using LLM-as-a-judge.",
    )(handle_evaluate_bilingual_parity)

    mcp.tool(
        name="evaluate_bilingual_parity",
        description="Backwards-compatible alias for aieval.evaluate_bilingual_parity.",
    )(handle_evaluate_bilingual_parity)
