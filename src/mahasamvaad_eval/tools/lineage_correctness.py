"""tools/lineage_correctness/tool.py — Lineage & Supersession Correctness Checker (§3.2)."""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mcp.server.mcpserver import MCPServer

from mahasamvaad_eval import config
from mahasamvaad_eval.clients.chat_api_client import ChatResult, call_chat
from mahasamvaad_eval.clients.falkordb_client import FalkorDBClient
from mahasamvaad_eval.clients.storage_client import fetch_document
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


_GROUND_TRUTH_FILE = _find_benchmark_file("lineage_ground_truth.json")
DEV_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")


def _load_ground_truth() -> list[dict[str, Any]]:
    if _GROUND_TRUTH_FILE.is_file():
        try:
            with open(_GROUND_TRUTH_FILE, encoding="utf-8") as f:
                return json.load(f)
        except Exception as exc:
            logger.warning("Could not load lineage_ground_truth.json: %s", exc)
    return []


def _normalize_rel_name(verb: str) -> str:
    """Map natural language and Marathi relation keywords to canonical relation types."""
    v = verb.lower()
    if any(kw in v for kw in ("supersede", "अधिक्रमित", "superseded")):
        return "supersedes"
    if any(kw in v for kw in ("amend", "सुधारणा", "amended")):
        return "amends"
    if any(kw in v for kw in ("revoke", "रद्द", "revoked", "cancel")):
        return "revokes"
    if any(kw in v for kw in ("extend", "वाढवणे", "extended")):
        return "extends"
    if any(kw in v for kw in ("suspend", "निलंबित", "suspended")):
        return "suspends"
    if any(kw in v for kw in ("reinstate", "पुनर्संचयित", "reinstated")):
        return "reinstates"
    return v


def _extract_relations_from_text(text: str) -> list[dict[str, str]]:
    """Extract supersession, amendment, and revocation relations from prose text."""
    if not text:
        return []

    t = text.translate(DEV_TO_ASCII)
    relations: list[dict[str, str]] = []

    # 1. Active: X (supersedes / amends / revokes / extends) [optional intervening words] Y
    pat_active = re.compile(
        r"\b(\d{8,18}|[A-Za-z0-9\-\/\.]+)\s+(?:has\s+been\s+issued\s+to\s+|was\s+issued\s+to\s+|hereby\s+)?"
        r"(supersedes?|amends?|revokes?|extends?|अधिक्रमित\s+(?:करतो|करून|केले)|सुधारणा\s+(?:करतो|करून|केली)|रद्द\s+(?:करतो|करून|केले))\s+"
        r"(?:the\s+earlier\s+resolution\s+|the\s+previous\s+(?:GR|resolution)\s+|earlier\s+|previous\s+|GR\s+|Government\s+Resolution\s+|शासन\s+निर्णय\s+क्र\.?\s*)?"
        r"(\d{8,18}|[A-Za-z0-9\-\/\.]+)\b",
        re.IGNORECASE,
    )
    for m in pat_active.finditer(t):
        src, verb, tgt = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        rel = _normalize_rel_name(verb)
        if src != tgt and len(src) < 40 and len(tgt) < 40:
            relations.append({"source": src, "relation": rel, "target": tgt})

    # 2. Passive: Y was superseded / amended / revoked by X
    pat_passive = re.compile(
        r"\b(\d{8,18}|[A-Za-z0-9\-\/\.]+)\s+(?:was|is|has\s+been)\s+(superseded|amended|revoked|extended)\s+by\s+"
        r"(?:the\s+later\s+resolution\s+|GR\s+|Government\s+Resolution\s+|शासन\s+निर्णय\s+क्र\.?\s*)?"
        r"(\d{8,18}|[A-Za-z0-9\-\/\.]+)\b",
        re.IGNORECASE,
    )
    for m in pat_passive.finditer(t):
        tgt, verb, src = m.group(1).strip(), m.group(2).strip(), m.group(3).strip()
        rel = _normalize_rel_name(verb)
        if src != tgt and len(src) < 40 and len(tgt) < 40:
            relations.append({"source": src, "relation": rel, "target": tgt})

    # 3. Prepositional: In supersession of Y, X was issued
    pat_prep = re.compile(
        r"\b(?:in\s+supersession\s+of|in\s+amendment\s+(?:of|to)|revoking|च्या\s+अधिक्रमणास्तव)\s+"
        r"(?:GR\s+|Government\s+Resolution\s+|शासन\s+निर्णय\s+क्र\.?\s*)?(\d{8,18}|[A-Za-z0-9\-\/\.]+)[,\s]+"
        r"(?:the\s+government\s+issued\s+|issued\s+)?(?:GR\s+|Government\s+Resolution\s+)?(\d{8,18}|[A-Za-z0-9\-\/\.]+)\b",
        re.IGNORECASE,
    )
    for m in pat_prep.finditer(t):
        tgt, src = m.group(1).strip(), m.group(2).strip()
        rel = "supersedes" if "supersession" in m.group(0).lower() or "अधिक्रमण" in m.group(0) else ("amends" if "amendment" in m.group(0).lower() else "revokes")
        if src != tgt and len(src) < 40 and len(tgt) < 40:
            relations.append({"source": src, "relation": rel, "target": tgt})

    # Deduplicate while preserving order
    unique_rels: list[dict[str, str]] = []
    seen = set()
    for r in relations:
        key = (r["source"], r["relation"], r["target"])
        if key not in seen:
            seen.add(key)
            unique_rels.append(r)

    return unique_rels


async def handle_check_lineage_correctness(
    gr_number: str | None = None,
    query: str | None = None,
    verify_document_storage: bool = False,
    use_falkordb: bool = True,
) -> dict[str, Any]:
    """Verify GR lineage / supersession / amendment direction against FalkorDB Knowledge Graph or ground truth."""
    try:
        ground_truth = _load_ground_truth()
        falkordb_client = FalkorDBClient() if use_falkordb else None
        provenance = "static_calibration_seed"

        target_gr = gr_number
        if not target_gr and query:
            # Extract possible GR number from query
            m = re.search(r"\b(20\d{14,16}|[A-Za-z0-9\-\/\.]{4,25})\b", query)
            if m:
                target_gr = m.group(1)

        # 1. Attempt to query FalkorDB Knowledge Graph for live ground-truth if available
        falkor_cases: list[dict[str, Any]] = []
        if falkordb_client and target_gr:
            try:
                live_lineage = falkordb_client.get_gr_lineage(target_gr)
                if live_lineage:
                    provenance = "falkordb_live_kg"
                    for edge in live_lineage:
                        falkor_cases.append({
                            "id": f"kg_{edge['source_gr']}_{edge['relation_type']}_{edge['target_gr']}",
                            "gr_number": edge["source_gr"],
                            "department": edge.get("source_dept", ""),
                            "relation_type": edge["relation_type"],
                            "target_gr": edge["target_gr"],
                            "expected_direction": edge["description"],
                            "source_is_active": edge.get("source_is_active", True),
                            "target_is_active": edge.get("target_is_active", False),
                        })
            except Exception as exc:
                logger.warning("FalkorDB query attempt in lineage check failed: %s", exc)

        # Select which cases to evaluate
        if falkor_cases:
            matched_gt_cases = falkor_cases
        elif target_gr:
            matched_gt_cases = [gt for gt in ground_truth if gt.get("gr_number") == target_gr]
            if not matched_gt_cases:
                matched_gt_cases = [gt for gt in ground_truth if gt.get("target_gr") == target_gr]
            if not matched_gt_cases:
                # Custom GR provided not in ground truth
                matched_gt_cases = [{
                    "id": "custom",
                    "gr_number": target_gr,
                    "relation_type": "supersedes",
                    "target_gr": "unknown",
                    "expected_direction": f"{target_gr} supersedes unknown",
                }]
        else:
            matched_gt_cases = ground_truth

        eval_results: list[dict[str, Any]] = []
        all_claimed_relations: list[dict[str, str]] = []
        doc_verification = None
        now_iso = datetime.now(timezone.utc).isoformat()
        conn = await get_db()

        try:
            for gt in matched_gt_cases:
                gt_gr = gt.get("gr_number", "")
                gt_target = gt.get("target_gr", "")
                gt_rel = gt.get("relation_type", "")
                expected_dir = gt.get("expected_direction", f"{gt_gr} {gt_rel} {gt_target}")

                # Formulate probe prompt for this specific GR
                prompt = query if query and target_gr == gt_gr else f"What is the supersession, amendment, or revocation lineage of Maharashtra Government Resolution {gt_gr}?"
                chat_res = await call_chat(prompt, web_search=False)

                claimed_relations: list[dict[str, str]] = []
                if chat_res.ok and chat_res.raw_response:
                    raw_meta = chat_res.raw_response.get("metadata", {})
                    lineage_meta = raw_meta.get("lineages") or raw_meta.get("lineage_data") or []
                    if isinstance(lineage_meta, list):
                        for item in lineage_meta:
                            if isinstance(item, dict):
                                src_val = str(item.get("source_gr") or item.get("gr_number") or "")
                                rel_val = _normalize_rel_name(str(item.get("relation_type") or item.get("relation") or ""))
                                tgt_val = str(item.get("target_gr") or item.get("supersedes") or item.get("amends") or item.get("revokes") or "")
                                if src_val and tgt_val:
                                    claimed_relations.append({
                                        "source": src_val,
                                        "relation": rel_val,
                                        "target": tgt_val,
                                    })

                # Extract relations from text response
                response_text = chat_res.response or ""
                text_relations = _extract_relations_from_text(response_text)
                for tr in text_relations:
                    if tr not in claimed_relations:
                        claimed_relations.append(tr)

                for cr in claimed_relations:
                    if cr not in all_claimed_relations:
                        all_claimed_relations.append(cr)

                # Evaluate direction and correctness against ground truth
                classification = "missing_relation"
                actual_dir = "none"
                is_correct = False

                for cr in claimed_relations:
                    c_src = cr.get("source", "")
                    c_tgt = cr.get("target", "")
                    c_rel = cr.get("relation", "")

                    if c_src == gt_gr and c_tgt == gt_target:
                        if c_rel == gt_rel:
                            classification = "exact_match"
                            actual_dir = f"{c_src} {c_rel} {c_tgt}"
                            is_correct = True
                            break
                        else:
                            classification = "wrong_relation_type"
                            actual_dir = f"{c_src} {c_rel} {c_tgt}"
                            is_correct = False
                            break
                    elif c_src == gt_target and c_tgt == gt_gr:
                        classification = "reversed_direction"  # Dangerous lineage reversal
                        actual_dir = f"{c_src} {c_rel} {c_tgt}"
                        is_correct = False
                        break

                # Fallback check if response text directly asserts the relation
                if not is_correct and classification == "missing_relation":
                    if gt_gr in response_text and gt_target in response_text:
                        idx_src = response_text.find(gt_gr)
                        idx_tgt = response_text.find(gt_target)
                        if any(kw in response_text.lower() for kw in ("supersede", "अधिक्रमित")) and gt_rel == "supersedes":
                            if idx_src < idx_tgt and "superseded by" not in response_text.lower():
                                classification = "exact_match"
                                actual_dir = f"{gt_gr} supersedes {gt_target}"
                                is_correct = True
                            elif idx_tgt < idx_src and any(kw in response_text.lower() for kw in ("supersedes", "अधिक्रमित")):
                                classification = "reversed_direction"
                                actual_dir = f"{gt_target} supersedes {gt_gr}"
                                is_correct = False
                        elif any(kw in response_text.lower() for kw in ("amend", "सुधारणा")) and gt_rel == "amends":
                            if idx_src < idx_tgt and "amended by" not in response_text.lower():
                                classification = "exact_match"
                                actual_dir = f"{gt_gr} amends {gt_target}"
                                is_correct = True
                            elif idx_tgt < idx_src and any(kw in response_text.lower() for kw in ("amends", "सुधारणा")):
                                classification = "reversed_direction"
                                actual_dir = f"{gt_target} amends {gt_gr}"
                                is_correct = False
                        elif any(kw in response_text.lower() for kw in ("revoke", "रद्द")) and gt_rel == "revokes":
                            if idx_src < idx_tgt and "revoked by" not in response_text.lower():
                                classification = "exact_match"
                                actual_dir = f"{gt_gr} revokes {gt_target}"
                                is_correct = True
                            elif idx_tgt < idx_src and any(kw in response_text.lower() for kw in ("revokes", "रद्द")):
                                classification = "reversed_direction"
                                actual_dir = f"{gt_target} revokes {gt_gr}"
                                is_correct = False

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
                        json.dumps({
                            "classification": classification,
                            "provenance": provenance,
                            "chat_response_snippet": response_text[:200],
                        }, ensure_ascii=False),
                        now_iso,
                    ),
                )

                # Optional document fetch verification
                if verify_document_storage and chat_res.sources and doc_verification is None:
                    first_fp = chat_res.sources[0].get("filepath") or chat_res.sources[0].get("filename")
                    if first_fp:
                        doc_res = await fetch_document(first_fp)
                        doc_verification = {
                            "filepath": first_fp,
                            "storage_ok": doc_res.ok,
                            "extracted_text_preview": (doc_res.extracted_text or "")[:200] if doc_res.ok else None,
                            "error": doc_res.error if not doc_res.ok else None,
                        }
        finally:
            await conn.close()

        correct_count = sum(1 for r in eval_results if r["correct"])
        total_checks = len(eval_results)
        accuracy = (correct_count / total_checks * 100) if total_checks else 0.0

        return {
            "ok": True,
            "provenance": provenance,
            "target_gr": target_gr,
            "total_checks": total_checks,
            "correct_count": correct_count,
            "lineage_accuracy_pct": round(accuracy, 2),
            "eval_results": eval_results,
            "claimed_relations": all_claimed_relations,
            "document_verification": doc_verification,
        }

    except Exception as exc:
        logger.exception("check_lineage_correctness error: %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Lineage Correctness Checker on MCP server."""
    mcp.tool(
        name="aieval.check_lineage_correctness",
        description="Verify correctness and direction of GR supersessions/amendments against live FalkorDB Knowledge Graph or ground truth.",
    )(handle_check_lineage_correctness)
