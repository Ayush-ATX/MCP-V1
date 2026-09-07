"""tools/bilingual_parity/tool.py — Bilingual Parity Evaluator (EN vs MR) (§3.3)."""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from mahasamvaad_eval import config
from mahasamvaad_eval.clients.chat_api_client import ChatResult, call_chat
from mahasamvaad_eval.clients.llm_judge_client import default_judge_client
from mahasamvaad_eval.storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)


def _find_benchmark_file(filename: str) -> Path:
    candidates = [
        Path.cwd() / "benchmarks" / filename,
        Path(__file__).resolve().parent.parent.parent.parent / "benchmarks" / filename,
        Path(__file__).resolve().parent.parent.parent / "calibration_data" / filename,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return candidates[0]


_BILINGUAL_QUERY_FILE = _find_benchmark_file("bilingual_query_set.json")


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

Evaluation Criteria:
1. Factual completeness and correctness regarding Maharashtra government schemes/resolutions.
2. Clarity, structure, and precision of information.
3. Appropriate citations and reference numbers.

Rate this answer on a scale from 1.0 to 10.0 (where 10.0 is complete, factually accurate, and well-structured; 1.0 is completely inaccurate or irrelevant).
Return valid JSON only with exact structure:
{{
  "score": 8.5,
  "rationale": "Clear explanation of eligibility, mentions correct figures, well structured."
}}"""


def _get_source_id(src: dict[str, Any]) -> str:
    """Return a canonical identifier for a source."""
    if src.get("gr_number"):
        return str(src["gr_number"]).strip().lower()
    if src.get("filename"):
        return str(src["filename"]).strip().lower()
    if src.get("filepath"):
        return str(src["filepath"]).strip().lower()
    if src.get("source_pdf_url"):
        return str(src["source_pdf_url"]).strip().lower()
    if src.get("citation_id") is not None:
        return f"citation_{src['citation_id']}"
    return json.dumps(src, sort_keys=True)


def _clean_source_for_output(src: dict[str, Any]) -> dict[str, Any]:
    """Return a concise source dictionary without internal anchor arrays or grounding tokens."""
    clean: dict[str, Any] = {}
    for key in ("citation_id", "gr_number", "filename", "filepath", "date", "department", "page_number", "source_pdf_url"):
        val = src.get(key)
        if val is not None:
            clean[key] = val
    if not clean and isinstance(src, dict):
        clean = {k: v for k, v in src.items() if k not in ("anchors", "anchor_id", "start_text", "end_text", "blocks_url", "raw_file_id")}
    return clean


def _compare_bilingual_sources(
    sources_en: list[dict[str, Any]],
    sources_mr: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare retrieved sources between English and Marathi agent responses and return concise sources."""
    map_en = {_get_source_id(s): s for s in sources_en if isinstance(s, dict)}
    map_mr = {_get_source_id(s): s for s in sources_mr if isinstance(s, dict)}

    keys_en = set(map_en.keys())
    keys_mr = set(map_mr.keys())

    common_keys = sorted(keys_en & keys_mr)
    en_only_keys = sorted(keys_en - keys_mr)
    mr_only_keys = sorted(keys_mr - keys_en)

    return {
        "sources_count_en": len(sources_en),
        "sources_count_mr": len(sources_mr),
        "common_sources": [_clean_source_for_output(map_en[k]) for k in common_keys],
        "english_only_sources": [_clean_source_for_output(map_en[k]) for k in en_only_keys],
        "marathi_only_sources": [_clean_source_for_output(map_mr[k]) for k in mr_only_keys],
    }


async def handle_evaluate_bilingual_parity(
    query_en: str | None = None,
    query_mr: str | None = None,
    pairs: list[dict[str, Any]] | None = None,
    gap_threshold: float = 0.15,
) -> dict[str, Any]:
    """Evaluate bilingual parity between English and Marathi responses using LLM-as-a-judge."""
    try:
        query_pairs: list[dict[str, Any]] = []
        if pairs:
            query_pairs = pairs
        elif query_en and query_mr:
            query_pairs = [{"id": "custom", "query_en": query_en, "query_mr": query_mr}]
        else:
            query_pairs = _load_bilingual_queries()

        if not query_pairs:
            return {
                "ok": True,
                "total_pairs_evaluated": 0,
                "passed_count": 0,
                "failed_count": 0,
                "parity_pass_rate_pct": 0.0,
                "gap_threshold": gap_threshold,
                "results": [],
                "message": "No bilingual query pairs provided or found",
            }

        results: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await get_db()
        try:
            for item in query_pairs:
                q_en = item.get("query_en", "")
                q_mr = item.get("query_mr", "")
                pair_id = item.get("id", "pair")

                # 1. Fire chat calls for both languages concurrently with graceful timeout handling
                try:
                    chat_task_en = call_chat(q_en, web_search=True)
                    chat_task_mr = call_chat(q_mr, web_search=True)
                    res_en_raw, res_mr_raw = await asyncio.gather(
                        chat_task_en, chat_task_mr, return_exceptions=True
                    )
                except Exception as exc:
                    logger.warning("Bilingual pair %s chat dispatch error: %s", pair_id, exc)
                    res_en_raw = exc
                    res_mr_raw = exc

                # Parse English chat result
                if isinstance(res_en_raw, ChatResult) or hasattr(res_en_raw, "ok"):
                    res_en = res_en_raw
                elif isinstance(res_en_raw, Exception):
                    res_en = ChatResult(ok=False, error=str(res_en_raw))
                else:
                    res_en = ChatResult(ok=False, error="Unknown English response error")

                # Parse Marathi chat result
                if isinstance(res_mr_raw, ChatResult) or hasattr(res_mr_raw, "ok"):
                    res_mr = res_mr_raw
                elif isinstance(res_mr_raw, Exception):
                    res_mr = ChatResult(ok=False, error=str(res_mr_raw))
                else:
                    res_mr = ChatResult(ok=False, error="Unknown Marathi response error")

                ans_en = res_en.response or ""
                ans_mr = res_mr.response or ""
                sources_en = res_en.sources if res_en.ok else []
                sources_mr = res_mr.sources if res_mr.ok else []

                # Source-level comparison (with concise source objects)
                source_comp = _compare_bilingual_sources(sources_en, sources_mr)

                # 2. LLM judge evaluations for both responses concurrently
                judge_task_en = default_judge_client.evaluate_json(
                    _JUDGE_RUBRIC_PROMPT.format(query=q_en, response=ans_en[:1500])
                )
                judge_task_mr = default_judge_client.evaluate_json(
                    _JUDGE_RUBRIC_PROMPT.format(query=q_mr, response=ans_mr[:1500])
                )
                judge_en, judge_mr = await asyncio.gather(judge_task_en, judge_task_mr)

                # Score extraction
                score_en = judge_en.score if (judge_en.ok and judge_en.score is not None) else None
                score_mr = judge_mr.score if (judge_mr.ok and judge_mr.score is not None) else None

                # Relative gap: normalize by maximum score
                if score_en is not None and score_mr is not None:
                    max_score = max(score_en, score_mr, 1.0)
                    gap = round(abs(score_en - score_mr) / max_score, 3)
                    parity_passed = gap <= gap_threshold
                else:
                    gap = None
                    parity_passed = False

                error_msg = None
                if not res_en.ok:
                    error_msg = f"English agent query failed: {res_en.error}"
                elif not res_mr.ok:
                    error_msg = f"Marathi agent query failed: {res_mr.error}"
                elif not judge_en.ok:
                    error_msg = f"English judge failed: {judge_en.error}"
                elif not judge_mr.ok:
                    error_msg = f"Marathi judge failed: {judge_mr.error}"

                pair_res = {
                    "id": pair_id,
                    "query_en": q_en,
                    "query_mr": q_mr,
                    "score_en": score_en,
                    "score_mr": score_mr,
                    "gap": gap,
                    "parity_passed": parity_passed,
                    "sources_count_en": source_comp["sources_count_en"],
                    "sources_count_mr": source_comp["sources_count_mr"],
                    "common_sources": source_comp["common_sources"],
                    "english_only_sources": source_comp["english_only_sources"],
                    "marathi_only_sources": source_comp["marathi_only_sources"],
                    "rationale_en": judge_en.rationale or (f"Judge error: {judge_en.error}" if not judge_en.ok else None),
                    "rationale_mr": judge_mr.rationale or (f"Judge error: {judge_mr.error}" if not judge_mr.ok else None),
                    "error": error_msg,
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
                        gap,
                        source_comp["sources_count_en"],
                        source_comp["sources_count_mr"],
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
