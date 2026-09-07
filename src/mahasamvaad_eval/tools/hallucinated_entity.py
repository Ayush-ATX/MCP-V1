"""tools/hallucinated_entity/tool.py — Hallucinated Entity Detector (§3.1)."""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any

from mcp.server.mcpserver import MCPServer

from mahasamvaad_eval import config
from mahasamvaad_eval.extractors.gr_regex import (
    ExtractedEntity,
    extract_entities,
    normalize_gr,
    parse_iso_date,
)
from mahasamvaad_eval.extractors.sentence_splitter import split_sentences
from mahasamvaad_eval.storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)


def _extract_citation_ids(sentence: str) -> list[int]:
    """Extract citation IDs from brackets in sentence, e.g. [14], [14][15], [14, 15]."""
    raw_matches = re.findall(r"\[([0-9\s,]+)\]", sentence)
    citation_ids: list[int] = []
    for m in raw_matches:
        parts = m.split(",")
        for p in parts:
            p_str = p.strip()
            if p_str.isdigit():
                citation_ids.append(int(p_str))
    # Deduplicate while preserving order
    return list(dict.fromkeys(citation_ids))


def _resolve_target_sources(
    cited_ids: list[int],
    all_sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map cited citation IDs to matching source dictionaries."""
    if not cited_ids or not all_sources:
        return all_sources

    target: list[dict[str, Any]] = []
    for src in all_sources:
        cid = src.get("citation_id")
        if cid is not None:
            try:
                if int(cid) in cited_ids or str(cid) in [str(c) for c in cited_ids]:
                    target.append(src)
            except (ValueError, TypeError):
                pass

    # Fallback to 1-based index if citation_id was not explicitly set on sources
    if not target:
        for cid in cited_ids:
            if 1 <= cid <= len(all_sources):
                target.append(all_sources[cid - 1])

    return target if target else all_sources


def _verify_entity_in_sources(
    ent: ExtractedEntity,
    target_sources: list[dict[str, Any]],
) -> tuple[bool, str | None, str | None, str | None]:
    """Verify an entity against target sources.

    Returns (found_in_source, expected_value, matched_source_value, mismatch_reason).
    """
    if not target_sources:
        return False, None, None, "no_sources_provided"

    ent_norm = ent.normalized.lower()
    ent_raw_lower = ent.text.lower()

    # 1. GR Number Validation
    if ent.entity_type == "gr_number":
        for src in target_sources:
            # Check source metadata: gr_number, filename, filepath, gr_code
            for field in ("gr_number", "filename", "filepath", "gr_code", "raw_file_id"):
                raw_val = src.get(field)
                if raw_val:
                    norm_src_gr = normalize_gr(str(raw_val))
                    if ent_norm in norm_src_gr or norm_src_gr in ent_norm:
                        return True, None, str(raw_val), None

            # Check anchors
            for anchor in src.get("anchors") or []:
                for a_text in (anchor.get("start_text"), anchor.get("end_text")):
                    if a_text:
                        norm_a = normalize_gr(str(a_text))
                        if ent_norm in norm_a:
                            return True, None, str(a_text), None

        expected_gr = target_sources[0].get("gr_number") or target_sources[0].get("filename")
        return False, str(expected_gr) if expected_gr else None, None, "gr_number_not_found_in_cited_source"

    # 2. Date Validation
    if ent.entity_type == "date":
        claimed_iso = parse_iso_date(ent.text) or ent_norm
        source_dates: list[str] = []

        for src in target_sources:
            raw_d = src.get("date")
            if raw_d:
                iso_d = parse_iso_date(str(raw_d)) or str(raw_d).strip().lower()
                source_dates.append(iso_d)
                if claimed_iso == iso_d or ent_raw_lower in str(raw_d).lower():
                    return True, None, str(raw_d), None

            # Check anchors for date mentions
            for anchor in src.get("anchors") or []:
                for a_text in (anchor.get("start_text"), anchor.get("end_text")):
                    if a_text:
                        a_iso = parse_iso_date(str(a_text))
                        if a_iso and a_iso == claimed_iso:
                            return True, None, str(a_text), None
                        if ent_raw_lower in str(a_text).lower() or ent_norm in str(a_text).lower():
                            return True, None, str(a_text), None

        expected_date = source_dates[0] if source_dates else None
        reason = "conflicting_date_with_cited_source" if expected_date else "date_not_found_in_cited_source"
        return False, expected_date, None, reason

    # 3. Section Reference Validation
    if ent.entity_type == "section":
        # Extract numerical & clause identifier from section text (e.g. '52', '52(1)', '15')
        sec_num_match = re.search(r"(\d+[\(\)\dA-Za-z\u0966-\u096F]*)", ent_norm)
        sec_num = sec_num_match.group(1) if sec_num_match else ent_norm

        for src in target_sources:
            for anchor in src.get("anchors") or []:
                for a_text in (anchor.get("start_text"), anchor.get("end_text")):
                    if a_text:
                        a_lower = str(a_text).lower()
                        if ent_raw_lower in a_lower or ent_norm in a_lower:
                            return True, None, str(a_text), None
                        if sec_num in a_lower and any(kw in a_lower for kw in ("section", "sec", "कलम", "नियम", "rule")):
                            return True, None, str(a_text), None

        return False, None, None, "section_not_found_in_cited_source"

    # 4. Financial Figures / Quantities
    if ent.entity_type == "financial_or_quantity":
        for src in target_sources:
            for anchor in src.get("anchors") or []:
                for a_text in (anchor.get("start_text"), anchor.get("end_text")):
                    if a_text:
                        a_lower = str(a_text).lower()
                        if ent_raw_lower in a_lower or ent_norm in a_lower:
                            return True, None, str(a_text), None

        return False, None, None, "figure_not_found_in_cited_source"

    # 5. Department Validation
    if ent.entity_type == "department":
        for src in target_sources:
            src_dept = src.get("department")
            if src_dept and (ent_raw_lower in str(src_dept).lower() or ent_norm in str(src_dept).lower()):
                return True, None, str(src_dept), None
            for anchor in src.get("anchors") or []:
                for a_text in (anchor.get("start_text"), anchor.get("end_text")):
                    if a_text:
                        a_lower = str(a_text).lower()
                        if ent_raw_lower in a_lower or ent_norm in a_lower:
                            return True, None, str(a_text), None

        return False, None, None, "department_not_found_in_cited_source"

    return False, None, None, "unsupported_entity_type"


async def handle_detect_hallucinated_entities(
    message_id: str | None = None,
    response_text: str | None = None,
    sources: list[dict[str, Any]] | None = None,
    chatroom_id: str | None = None,
) -> dict[str, Any]:
    """Extract factual entities from response and verify whether they appear in cited source anchors/metadata."""
    try:
        # If response_text or sources not supplied directly, fetch from query_log
        if message_id and (response_text is None or sources is None):
            clean_msg_id = message_id.strip()
            conn = await get_db()
            try:
                rows = await conn.execute_fetchall(
                    "SELECT chatroom_id, response_text, sources_json FROM query_log WHERE message_id = ?",
                    (clean_msg_id,),
                )
            finally:
                await conn.close()

            if not rows:
                return {
                    "ok": False,
                    "error": f"message_id not found: {message_id!r}",
                    "http_status": None,
                }
            r = rows[0]
            chatroom_id = chatroom_id or r["chatroom_id"]
            response_text = response_text if response_text is not None else (r["response_text"] or "")
            if sources is None:
                try:
                    sources = json.loads(r["sources_json"] or "[]")
                except Exception:
                    sources = []

        response_text = response_text or ""
        sources = sources or []

        sentences = split_sentences(response_text)
        checked_entities: list[dict[str, Any]] = []
        hallucinated_count = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        conn_write = await get_db()
        try:
            for sent in sentences:
                cited_ids = _extract_citation_ids(sent)
                target_sources = _resolve_target_sources(cited_ids, sources)
                claimed_citation = cited_ids[0] if len(cited_ids) == 1 else (cited_ids if cited_ids else None)

                entities = extract_entities(sent)
                for ent in entities:
                    found, exp_val, matched_val, reason = _verify_entity_in_sources(ent, target_sources)

                    if not found:
                        hallucinated_count += 1

                    item = {
                        "entity_text": ent.text,
                        "entity_type": ent.entity_type,
                        "normalized": ent.normalized,
                        "sentence": sent,
                        "citation_id_claimed": claimed_citation,
                        "found_in_source": found,
                        "expected_value": exp_val,
                        "matched_source_value": matched_val,
                        "mismatch_reason": reason,
                    }
                    checked_entities.append(item)

                    # Persist to SQLite table hallucinated_entity_checks
                    await execute_with_retry(
                        conn_write,
                        """
                        INSERT INTO hallucinated_entity_checks (
                            chatroom_id, message_id, entity_text, entity_type,
                            found_in_source, citation_id_claimed, sentence, details_json, checked_at
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            chatroom_id,
                            message_id,
                            ent.text,
                            ent.entity_type,
                            1 if found else 0,
                            str(claimed_citation) if claimed_citation is not None else None,
                            sent,
                            json.dumps(item, ensure_ascii=False),
                            now_iso,
                        ),
                    )
        finally:
            await conn_write.close()

        total_entities = len(checked_entities)
        hallucination_rate = (hallucinated_count / total_entities * 100) if total_entities else 0.0

        return {
            "ok": True,
            "message_id": message_id,
            "chatroom_id": chatroom_id,
            "total_entities": total_entities,
            "hallucinated_count": hallucinated_count,
            "hallucination_rate_pct": round(hallucination_rate, 2),
            "entities": checked_entities,
        }

    except Exception as exc:
        logger.exception("detect_hallucinated_entities error: %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Hallucinated Entity Detector on MCP server."""
    mcp.tool(
        name="aieval.detect_hallucinated_entities",
        description="Verify factual entities (GR numbers, dates, sections, figures) against cited source anchors.",
    )(handle_detect_hallucinated_entities)
