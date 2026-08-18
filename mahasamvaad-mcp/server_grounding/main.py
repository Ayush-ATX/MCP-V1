"""Server 2 — mahasamvaad-grounding (§6)

Tools:
  - score_grounding  (§6.1) — difflib_v1 algorithm, pluggable strategy seam for embedding_v2
  - grounding_trend  (§6.2) — per-slice coverage trend

Reads from query_log (written by Server 1) and writes to grounding_scores.
Imports call_chat/store from mcp_common (Option A — no MCP-to-MCP composition).
"""
from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import logging
import re
import statistics
import sys
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_common import config
from mcp_common.store import execute_with_retry, get_db

logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [grounding] %(message)s",
)
logger = logging.getLogger(__name__)

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="mahasamvaad-grounding",
    version="1.0.0",
    description="Grounding / Citation Coverage scorer for MahaSamvaad answers",
)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter — handles Devanagari danda (।) and Latin full stops
# ─────────────────────────────────────────────────────────────────────────────
_SENTENCE_END = re.compile(r"(?<!\d)(?<!\w\.\w)(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences handling both Devanagari (।) and Latin (.) punctuation."""
    if not text or not text.strip():
        return []
    text = text.replace("।", "। ")
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


# ─────────────────────────────────────────────────────────────────────────────
# Grounding strategy — pluggable seam for embedding_v2
# ─────────────────────────────────────────────────────────────────────────────
class GroundingStrategy(Protocol):
    """Extension seam: implement this protocol to add embedding_v2 or other strategies."""

    def score(self, sentence: str, anchor_texts: list[str]) -> float:
        ...


class DifflibV1Strategy:
    """v1 — difflib SequenceMatcher string overlap (§6.1 step 3)."""

    def score(self, sentence: str, anchor_texts: list[str]) -> float:
        if not anchor_texts:
            return 0.0
        best = 0.0
        for anchor in anchor_texts:
            ratio = difflib.SequenceMatcher(None, sentence, anchor).ratio()
            if ratio > best:
                best = ratio
        return best


# STRETCH HOOK — embedding_v2: replace DifflibV1Strategy with cosine-similarity
# using a multilingual sentence-transformer. Same output shape; only .score() changes.
# class EmbeddingV2Strategy:
#     def score(self, sentence: str, anchor_texts: list[str]) -> float:
#         raise NotImplementedError("embedding_v2 is a v2 upgrade path — not implemented in v1")


def _get_strategy(method: str) -> GroundingStrategy:
    if method == "difflib_v1":
        return DifflibV1Strategy()
    raise ValueError(f"Unknown grounding method: {method!r}. Valid: difflib_v1")


# ─────────────────────────────────────────────────────────────────────────────
# Tool: score_grounding (§6.1)
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="score_grounding",
    description=(
        "Score how well the response text for a logged message_id is grounded in its cited sources. "
        "Uses difflib_v1 by default. Returns coverage_pct and ungrounded sentences."
    ),
)
async def score_grounding(
    message_id: str,
    method: str | None = None,
) -> dict[str, Any]:
    """§6.1 — difflib_v1 grounding algorithm, steps 1–6."""
    try:
        effective_method = method or config.GROUNDING_METHOD

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

        # Step 6: Short-circuit if sources[] is empty
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

        # Step 1 cont: Parse anchors
        anchor_texts: list[str] = []
        for src in sources:
            for anchor in (src.get("anchors") or []):
                start = anchor.get("start_text", "")
                end = anchor.get("end_text", "")
                combined = f"{start} {end}".strip()
                if combined:
                    anchor_texts.append(combined)

        # Step 2: Split response_text into sentences
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

        # Steps 3+4: Score each sentence
        strategy = _get_strategy(effective_method)
        threshold = config.GROUNDING_THRESHOLD
        grounded_count = 0
        ungrounded: list[dict] = []

        for sent in sentences:
            best_score = strategy.score(sent, anchor_texts)
            if best_score >= threshold:
                grounded_count += 1
            else:
                ungrounded.append({"sentence": sent, "best_match_score": round(best_score, 4)})

        # Step 5: coverage_pct
        total_sentences = len(sentences)
        coverage_pct = grounded_count / total_sentences * 100

        scored_at = datetime.now(timezone.utc).isoformat()
        conn3 = await get_db()
        try:
            await execute_with_retry(
                conn3,
                "INSERT INTO grounding_scores (message_id, scored_at, coverage_pct, total_sentences, grounded_count, ungrounded_json, method) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (message_id, scored_at, round(coverage_pct, 2), total_sentences, grounded_count,
                 json.dumps(ungrounded, ensure_ascii=False), effective_method),
            )
        finally:
            await conn3.close()

        logger.info(
            "score_grounding: message_id=%s coverage=%.1f%% (%d/%d) method=%s",
            message_id, coverage_pct, grounded_count, total_sentences, effective_method,
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


# ─────────────────────────────────────────────────────────────────────────────
# Tool: grounding_trend (§6.2)
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="grounding_trend",
    description=(
        "Aggregate score_grounding results over time, sliced by department / intent / route_hint. "
        "Returns per-slice mean/median coverage, sample count, 7-day trend direction, and worst message_id."
    ),
)
async def grounding_trend(
    slice_by: str = "department",
    since: str | None = None,
    min_samples: int = 5,
) -> dict[str, Any]:
    """§6.2 — Trend aggregation over grounding_scores joined to query_log."""
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


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MahaSamvaad Grounding MCP Server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default=config.MCP_TRANSPORT)
    parser.add_argument("--host", default=config.MCP_HTTP_HOST)
    parser.add_argument("--port", type=int, default=config.MCP_HTTP_PORT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.transport == "streamable-http":
        logger.info("Starting grounding server on %s:%d (streamable-http)", args.host, args.port)
        mcp.run_streamable_http_async(host=args.host, port=args.port)
    else:
        logger.info("Starting grounding server on stdio")
        mcp.run_stdio_async()


if __name__ == "__main__":
    main()
