"""tools/grounding/tool.py — Grounding & Citation Coverage scorer (embedding_v2 & difflib_v1 fallback)."""
from __future__ import annotations

import difflib
import json
import logging
import re
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from mahasamvaad_eval import config
from mahasamvaad_eval.clients.embedding_client import EmbeddingClient, cosine_similarity
from mahasamvaad_eval.extractors.sentence_splitter import split_sentences
from mahasamvaad_eval.storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)

_SENTENCE_END = re.compile(r"(?<!\d)(?<!\w\.\w)(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences handling both Devanagari (।) and Latin (.) punctuation."""
    if not text or not text.strip():
        return []
    text = text.replace("।", "। ")
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


class GroundingStrategy(Protocol):
    async def score_sentence(self, sentence: str, anchor_texts: list[str]) -> float:
        ...


class DifflibV1Strategy:
    """difflib SequenceMatcher string overlap (v1)."""

    def score(self, sentence: str, anchor_texts: list[str]) -> float:
        if not anchor_texts:
            return 0.0
        best = 0.0
        for anchor in anchor_texts:
            ratio = difflib.SequenceMatcher(None, sentence, anchor).ratio()
            if ratio > best:
                best = ratio
        return best

    async def score_sentence(self, sentence: str, anchor_texts: list[str]) -> float:
        return self.score(sentence, anchor_texts)


class EmbeddingV2Strategy:
    """Cosine similarity using multilingual embedding model nemotron-3-embed-1b (v2)."""

    def __init__(self, client: EmbeddingClient | None = None) -> None:
        self.client = client or EmbeddingClient()

    async def score_sentence(self, sentence: str, anchor_texts: list[str]) -> float:
        if not anchor_texts or not sentence:
            return 0.0

        all_texts = [sentence] + anchor_texts
        embeddings = await self.client.get_embeddings(all_texts)
        sent_emb = embeddings[0] if embeddings else None
        if sent_emb is None:
            # Fallback to difflib if embedding unavailable
            logger.warning("Embedding not available for sentence, falling back to difflib")
            fallback = DifflibV1Strategy()
            return await fallback.score_sentence(sentence, anchor_texts)

        best_score = 0.0
        for anchor_emb in embeddings[1:]:
            if anchor_emb is None:
                continue
            sim = cosine_similarity(sent_emb, anchor_emb)
            if sim > best_score:
                best_score = sim

        return best_score


def get_grounding_strategy(method: str) -> tuple[GroundingStrategy, str]:
    clean_method = (method or config.GROUNDING_METHOD or "embedding_v2").strip().lower()
    if clean_method == "embedding_v2":
        return EmbeddingV2Strategy(), "embedding_v2"
    elif clean_method == "difflib_v1":
        return DifflibV1Strategy(), "difflib_v1"
    else:
        logger.warning("Unknown grounding method %s, defaulting to embedding_v2", method)
        return EmbeddingV2Strategy(), "embedding_v2"


async def handle_score_grounding(
    message_id: str,
    method: str | None = None,
) -> dict[str, Any]:
    """Score how well the response for a message_id is grounded in its cited anchors."""
    try:
        strategy, effective_method = get_grounding_strategy(method or config.GROUNDING_METHOD)

        conn = await get_db()
        try:
            rows = await conn.execute_fetchall(
                "SELECT response_text, sources_json FROM query_log WHERE message_id = ?",
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

        row = rows[0]
        response_text: str = row["response_text"] or ""
        sources_raw: str = row["sources_json"] or "[]"

        try:
            sources: list[dict] = json.loads(sources_raw)
        except json.JSONDecodeError:
            sources = []

        if not sources:
            scored_at = datetime.now(timezone.utc).isoformat()
            conn2 = await get_db()
            try:
                await execute_with_retry(
                    conn2,
                    "INSERT INTO grounding_scores (message_id, scored_at, coverage_pct, total_sentences, grounded_count, ungrounded_json, method) VALUES (?, ?, NULL, 0, 0, '[]', ?)",
                    (message_id, scored_at, effective_method),
                )
            finally:
                await conn2.close()
            return {
                "ok": True,
                "message_id": message_id,
                "coverage_pct": None,
                "reason": "no_sources_to_ground_against",
                "total_sentences": 0,
                "grounded_count": 0,
                "ungrounded_sentences": [],
                "method": effective_method,
            }

        # Extract anchor texts
        anchor_texts: list[str] = []
        for src in sources:
            for anchor in (src.get("anchors") or []):
                start = anchor.get("start_text", "")
                end = anchor.get("end_text", "")
                combined = f"{start} {end}".strip()
                if combined:
                    anchor_texts.append(combined)

        sentences = split_sentences(response_text)
        if not sentences:
            return {
                "ok": True,
                "message_id": message_id,
                "coverage_pct": None,
                "reason": "no_sentences_in_response",
                "total_sentences": 0,
                "grounded_count": 0,
                "ungrounded_sentences": [],
                "method": effective_method,
            }

        threshold = config.GROUNDING_THRESHOLD
        grounded_count = 0
        ungrounded: list[dict] = []

        for sent in sentences:
            best_score = await strategy.score_sentence(sent, anchor_texts)
            if best_score >= threshold:
                grounded_count += 1
            else:
                ungrounded.append({"sentence": sent, "best_match_score": round(best_score, 4)})

        total_sentences = len(sentences)
        coverage_pct = (grounded_count / total_sentences * 100) if total_sentences else 0.0

        scored_at = datetime.now(timezone.utc).isoformat()
        conn3 = await get_db()
        try:
            await execute_with_retry(
                conn3,
                "INSERT INTO grounding_scores (message_id, scored_at, coverage_pct, total_sentences, grounded_count, ungrounded_json, method) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    message_id,
                    scored_at,
                    round(coverage_pct, 2),
                    total_sentences,
                    grounded_count,
                    json.dumps(ungrounded, ensure_ascii=False),
                    effective_method,
                ),
            )
        finally:
            await conn3.close()

        logger.info(
            "score_grounding: message_id=%s coverage=%.1f%% (%d/%d) method=%s",
            message_id,
            coverage_pct,
            grounded_count,
            total_sentences,
            effective_method,
        )

        return {
            "ok": True,
            "message_id": message_id,
            "coverage_pct": round(coverage_pct, 2),
            "total_sentences": total_sentences,
            "grounded_count": grounded_count,
            "ungrounded_sentences": ungrounded,
            "method": effective_method,
        }

    except Exception as exc:
        logger.exception("score_grounding: unexpected error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


async def handle_grounding_trend(
    slice_by: str = "department",
    since: str | None = None,
    min_samples: int = 5,
) -> dict[str, Any]:
    """Aggregate grounding scores over time sliced by department, intent, or route_hint."""
    try:
        if slice_by not in ("department", "intent", "route_hint"):
            return {"ok": False, "error": "slice_by must be department | intent | route_hint", "http_status": None}

        if since is None:
            since = (datetime.now(timezone.utc) - timedelta(days=30)).isoformat()

        conn = await get_db()
        try:
            rows = await conn.execute_fetchall(
                f"""
                SELECT gs.message_id, gs.coverage_pct, gs.scored_at,
                       ql.{slice_by} AS slice_key
                FROM grounding_scores gs
                JOIN query_log ql ON ql.message_id = gs.message_id
                WHERE gs.scored_at >= ? AND gs.coverage_pct IS NOT NULL
                """,
                (since,),
            )
        finally:
            await conn.close()

        if not rows:
            return {"ok": True, "since": since, "slice_by": slice_by, "slices": []}

        from collections import defaultdict
        groups: dict[str, list] = defaultdict(list)
        for r in rows:
            key = r["slice_key"] or "(unknown)"
            groups[key].append(r)

        seven_days_ago = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        slices = []
        for key, group_rows in groups.items():
            if len(group_rows) < min_samples:
                continue

            coverages = [float(r["coverage_pct"]) for r in group_rows]
            mean_cov = statistics.mean(coverages)
            median_cov = statistics.median(coverages)

            recent = [float(r["coverage_pct"]) for r in group_rows if r["scored_at"] >= seven_days_ago]
            prior = [float(r["coverage_pct"]) for r in group_rows if r["scored_at"] < seven_days_ago]
            if recent and prior:
                trend = "improving" if statistics.mean(recent) > statistics.mean(prior) else "declining"
            elif recent:
                trend = "stable"
            else:
                trend = "no_recent_data"

            worst = min(group_rows, key=lambda r: float(r["coverage_pct"]))

            slices.append({
                "slice_key": key,
                "sample_count": len(group_rows),
                "mean_coverage_pct": round(mean_cov, 2),
                "median_coverage_pct": round(median_cov, 2),
                "trend_7d": trend,
                "worst_message_id": worst["message_id"],
                "worst_coverage_pct": round(float(worst["coverage_pct"]), 2),
            })

        slices.sort(key=lambda s: s["mean_coverage_pct"])

        return {
            "ok": True,
            "since": since,
            "slice_by": slice_by,
            "min_samples": min_samples,
            "slices": slices,
        }

    except Exception as exc:
        logger.exception("grounding_trend: unexpected error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Grounding tools on MCP server."""
    # Namespaced tool: grounding.score_citation_coverage
    mcp.tool(
        name="grounding.score_citation_coverage",
        description="Score citation coverage of a logged message_id using embedding_v2 (or difflib_v1 fallback).",
    )(handle_score_grounding)

    # Namespaced tool: grounding.grounding_trend
    mcp.tool(
        name="grounding.grounding_trend",
        description="Aggregate citation coverage trends over time sliced by department, intent, or route_hint.",
    )(handle_grounding_trend)
