"""tools/hallucinated_entity/tool.py — Hallucinated Entity Detector (§3.1)."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

from mcp.server.mcpserver import MCPServer

import config
from calibration_data.gr_number_date_regex_patterns import extract_entities
from storage.db import execute_with_retry, get_db
from tools.grounding.tool import split_sentences

logger = logging.getLogger(__name__)


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
            conn = await get_db()
            try:
                rows = await conn.execute_fetchall(
                    "SELECT chatroom_id, response_text, sources_json FROM query_log WHERE message_id = ?",
                    (message_id,),
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
            response_text = response_text or r["response_text"] or ""
            if sources is None:
                try:
                    sources = json.loads(r["sources_json"] or "[]")
                except Exception:
                    sources = []

        response_text = response_text or ""
        sources = sources or []

        # Build source text corpus for verification
        source_texts: list[str] = []
        source_metadata_vals: list[str] = []

        for src in sources:
            for field_name in ("gr_number", "date", "department", "filename", "filepath"):
                val = src.get(field_name)
                if val:
                    source_metadata_vals.append(str(val).lower())
            for anchor in src.get("anchors") or []:
                start = anchor.get("start_text", "")
                end = anchor.get("end_text", "")
                if start:
                    source_texts.append(start.lower())
                if end:
                    source_texts.append(end.lower())

        sentences = split_sentences(response_text)
        checked_entities: list[dict[str, Any]] = []
        hallucinated_count = 0
        now_iso = datetime.now(timezone.utc).isoformat()

        conn_write = await get_db()
        try:
            for sent in sentences:
                entities = extract_entities(sent)
                for ent in entities:
                    ent_norm = ent.normalized.lower()
                    ent_raw_lower = ent.text.lower()

                    # Verification: Check if entity is present in anchor texts or source metadata
                    found_in_source = False
                    for st in source_texts:
                        if ent_raw_lower in st or ent_norm in st:
                            found_in_source = True
                            break

                    if not found_in_source:
                        for sm in source_metadata_vals:
                            if ent_raw_lower in sm or ent_norm in sm:
                                found_in_source = True
                                break

                    if not found_in_source:
                        hallucinated_count += 1

                    claimed_citation_id = sources[0].get("citation_id") if sources else None

                    item = {
                        "entity_text": ent.text,
                        "entity_type": ent.entity_type,
                        "normalized": ent.normalized,
                        "sentence": sent,
                        "citation_id_claimed": claimed_citation_id,
                        "found_in_source": found_in_source,
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
                            1 if found_in_source else 0,
                            claimed_citation_id,
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

    mcp.tool(
        name="detect_hallucinated_entities",
        description="Backwards-compatible alias for aieval.detect_hallucinated_entities.",
    )(handle_detect_hallucinated_entities)
