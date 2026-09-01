"""tools/lineage_correctness/tool.py — Lineage & Supersession Correctness Checker (§3.2)."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

import config
from clients.chat_api_client import call_chat
from clients.storage_client import fetch_document
from storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)

_GROUND_TRUTH_FILE = Path(__file__).resolve().parent.parent.parent / "calibration_data" / "lineage_ground_truth.json"


def _load_ground_truth() -> list[dict[str, Any]]:
    if _GROUND_TRUTH_FILE.is_file():
        try:
            with open(_GROUND_TRUTH_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load lineage_ground_truth.json: %s", exc)
    return []


def _extract_relations_from_text(text: str) -> list[dict[str, str]]:
    """Heuristic extraction of supersession/amendment relations from text."""
    relations: list[dict[str, str]] = []
    # Pattern: X supersedes Y / X amends Y / X revokes Y
    pat = re.compile(r"(\d{8,18}|[A-Za-z0-9\-\/]+)\s+(supersedes|amends|revokes|superseded\s+by|amended\s+by|revoked\s+by|अधिक्रमित\s+करतो|दुरुस्ती\s+करतो)\s+(\d{8,18}|[A-Za-z0-9\-\/]+)", re.IGNORECASE)
    for m in pat.finditer(text):
        source = m.group(1).strip()
        rel = m.group(2).strip().lower()
        target = m.group(3).strip()
        relations.append({"source": source, "relation": rel, "target": target})
    return relations


async def handle_check_lineage_correctness(
    gr_number: str | None = None,
    query: str | None = None,
    verify_document_storage: bool = False,
) -> dict[str, Any]:
    """Verify GR lineage / supersession / amendment direction against hand-verified ground truth."""
    try:
        ground_truth = _load_ground_truth()

        target_gr = gr_number
        if not target_gr and query:
            # Extract possible GR number from query
            m = re.search(r"\b(20\d{14,16})\b", query)
            if m:
                target_gr = m.group(1)

        # 1. Fire chat call if query or GR is specified
        prompt = query or f"What is the status and supersession lineage of Maharashtra Government Resolution {target_gr}?"
        chat_res = await call_chat(prompt, web_search=False)

        raw_meta = chat_res.raw_response.get("metadata", {}) if (chat_res.ok and chat_res.raw_response) else {}
        lineage_meta = raw_meta.get("lineages") or raw_meta.get("lineage_data") or []
        response_text = chat_res.response or ""

        # Extract claimed relations
        claimed_relations: list[dict[str, str]] = []
        if isinstance(lineage_meta, list):
            for item in lineage_meta:
                if isinstance(item, dict):
                    claimed_relations.append({
                        "source": str(item.get("source_gr") or item.get("gr_number") or ""),
                        "relation": str(item.get("relation_type") or item.get("relation") or ""),
                        "target": str(item.get("target_gr") or item.get("supersedes") or item.get("amends") or ""),
                    })

        text_relations = _extract_relations_from_text(response_text)
        for tr in text_relations:
            if tr not in claimed_relations:
                claimed_relations.append(tr)

        # 2. Compare against ground truth
        matched_gt_cases = []
        if target_gr:
            matched_gt_cases = [gt for gt in ground_truth if gt.get("gr_number") == target_gr]
            if not matched_gt_cases:
                matched_gt_cases = [gt for gt in ground_truth if gt.get("target_gr") == target_gr]
        else:
            matched_gt_cases = ground_truth


        eval_results: list[dict[str, Any]] = []
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await get_db()
        try:
            for gt in matched_gt_cases:
                gt_gr = gt.get("gr_number", "")
                gt_target = gt.get("target_gr", "")
                gt_rel = gt.get("relation_type", "")
                expected_dir = gt.get("expected_direction", f"{gt_gr} {gt_rel} {gt_target}")

                # Check if claimed correctly or reversed
                classification = "missing_relation"
                actual_dir = "none"
                is_correct = False

                for cr in claimed_relations:
                    c_src = cr.get("source", "")
                    c_tgt = cr.get("target", "")
                    c_rel = cr.get("relation", "")

                    if (c_src == gt_gr and c_tgt == gt_target) or (gt_gr in response_text and gt_target in response_text and gt_rel in c_rel):
                        classification = "exact_match"
                        actual_dir = f"{c_src} {c_rel} {c_tgt}"
                        is_correct = True
                        break
                    elif c_src == gt_target and c_tgt == gt_gr:
                        classification = "reversed_direction"  # Dangerous error
                        actual_dir = f"{c_src} {c_rel} {c_tgt}"
                        is_correct = False
                        break

                eval_results.append({
                    "gr_number": gt_gr,
                    "relation_type": gt_rel,
                    "expected_direction": expected_dir,
                    "actual_direction": actual_dir,
                    "classification": classification,
                    "correct": is_correct,
                })

                # Persist to SQLite lineage_checks
                await execute_with_retry(
                    conn,
                    """
                    INSERT INTO lineage_checks (
                        gr_number, relation_type, expected_direction, actual_direction, correct, details_json, checked_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        gt_gr,
                        gt_rel,
                        expected_dir,
                        actual_dir,
                        1 if is_correct else 0,
                        json.dumps({"classification": classification, "chat_response_snippet": response_text[:200]}, ensure_ascii=False),
                        now_iso,
                    ),
                )
        finally:
            await conn.close()

        # Optional document fetch verification via storage endpoint
        doc_verification = None
        if verify_document_storage and chat_res.sources:
            first_fp = chat_res.sources[0].get("filepath")
            if first_fp:
                doc_res = await fetch_document(first_fp)
                doc_verification = {
                    "filepath": first_fp,
                    "storage_ok": doc_res.ok,
                    "extracted_text_preview": (doc_res.extracted_text or "")[:200] if doc_res.ok else None,
                }

        correct_count = sum(1 for r in eval_results if r["correct"])
        total_checks = len(eval_results)
        accuracy = (correct_count / total_checks * 100) if total_checks else 0.0

        return {
            "ok": True,
            "target_gr": target_gr,
            "total_checks": total_checks,
            "correct_count": correct_count,
            "lineage_accuracy_pct": round(accuracy, 2),
            "eval_results": eval_results,
            "claimed_relations": claimed_relations,
            "document_verification": doc_verification,
        }

    except Exception as exc:
        logger.exception("check_lineage_correctness error: %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Lineage Correctness Checker on MCP server."""
    mcp.tool(
        name="aieval.check_lineage_correctness",
        description="Verify correctness and direction of GR supersessions/amendments against hand-verified ground truth.",
    )(handle_check_lineage_correctness)

    mcp.tool(
        name="check_lineage_correctness",
        description="Backwards-compatible alias for aieval.check_lineage_correctness.",
    )(handle_check_lineage_correctness)
