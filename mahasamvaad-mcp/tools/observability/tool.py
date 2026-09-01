"""tools/observability/tool.py — Observability & QA logging tools and resources."""
from __future__ import annotations

import json
import logging
import statistics
from datetime import datetime, timedelta, timezone
from typing import Any

from mcp.server.mcpserver import MCPServer

import config
from clients.chat_api_client import call_chat
from storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)


async def handle_query_and_log(
    message: str,
    web_search: bool = True,
    model_name: str | None = None,
    history: list | None = None,
    last_turn_filepaths: list | None = None,
    chatroom_id: str | None = None,
    user_id: str | None = None,
) -> dict[str, Any]:
    """Execute /chat and log to query_log table unconditionally."""
    try:
        called_at = datetime.now(timezone.utc).isoformat()
        result = await call_chat(
            message,
            web_search=web_search,
            model_name=model_name,
            history=history,
            last_turn_filepaths=last_turn_filepaths,
            chatroom_id=chatroom_id,
            user_id=user_id,
        )

        department: str | None = None
        if result.sources:
            department = result.sources[0].get("department")

        conn = await get_db()
        try:
            cursor = await execute_with_retry(
                conn,
                """
                INSERT INTO query_log (
                    called_at, user_id, chatroom_id, user_message_id, message_id,
                    message, model_name, web_search, response_text, intent, language,
                    route_hint, department, sources_json, web_sources_json,
                    prompt_tokens, completion_tokens, total_tokens,
                    latency_seconds, langfuse_trace_id, http_status,
                    error_text, raw_response_json
                ) VALUES (
                    ?, ?, ?, ?, ?,
                    ?, ?, ?, ?, ?, ?,
                    ?, ?, ?, ?,
                    ?, ?, ?,
                    ?, ?, ?,
                    ?, ?
                )
                """,
                (
                    called_at,
                    user_id or config.MCP_DEFAULT_USER_ID,
                    result.chatroom_id,
                    result.user_message_id,
                    result.message_id,
                    message,
                    model_name,
                    1 if web_search else 0,
                    result.response if result.ok else None,
                    result.intent,
                    result.language,
                    result.route_hint,
                    department,
                    json.dumps(result.sources) if result.sources else None,
                    json.dumps(result.web_sources) if result.web_sources else None,
                    result.prompt_tokens,
                    result.completion_tokens,
                    result.total_tokens,
                    result.latency_seconds,
                    result.langfuse_trace_id,
                    result.http_status,
                    result.error if not result.ok else None,
                    json.dumps(result.raw_response) if result.raw_response else None,
                ),
            )
            logged_row_id = cursor.lastrowid
        finally:
            await conn.close()

        logger.info("query_and_log: logged row %d ok=%s message_id=%s", logged_row_id, result.ok, result.message_id)

        if not result.ok:
            return {
                "ok": False,
                "error": result.error,
                "http_status": result.http_status,
                "message_id": result.message_id,
                "logged_row_id": logged_row_id,
            }

        return {
            "ok": True,
            "message_id": result.message_id,
            "response": result.response,
            "intent": result.intent,
            "language": result.language,
            "route_hint": result.route_hint,
            "sources": result.sources,
            "web_sources": result.web_sources,
            "tokens": {
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            },
            "latency_seconds": result.latency_seconds,
            "langfuse_trace_id": result.langfuse_trace_id,
            "logged_row_id": logged_row_id,
        }

    except Exception as exc:
        logger.exception("query_and_log: unexpected error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


async def handle_get_dashboard_stats(
    since: str | None = None,
    group_by: str = "intent",
    outlier_z: float = 2.0,
) -> dict[str, Any]:
    """Dashboard analytics over query_log."""
    try:
        if group_by not in ("intent", "department", "day"):
            return {"ok": False, "error": "group_by must be intent | department | day", "http_status": None}

        if since is None:
            since = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()

        conn = await get_db()
        try:
            rows = await conn.execute_fetchall(
                "SELECT latency_seconds FROM query_log WHERE called_at >= ? AND latency_seconds IS NOT NULL",
                (since,),
            )
            latencies = [float(r["latency_seconds"]) for r in rows]
            latency_stats: dict[str, Any] = {}
            if latencies:
                sorted_lat = sorted(latencies)
                n = len(sorted_lat)

                def _pct(data: list[float], p: float) -> float:
                    idx = max(0, int(p / 100 * n) - 1)
                    return data[min(idx, n - 1)]

                latency_stats = {
                    "p50": _pct(sorted_lat, 50),
                    "p90": _pct(sorted_lat, 90),
                    "p99": _pct(sorted_lat, 99),
                    "count": n,
                }

            token_rows = await conn.execute_fetchall(
                """
                SELECT substr(called_at, 1, 10) AS day,
                       SUM(total_tokens) AS total,
                       SUM(prompt_tokens) AS prompt,
                       SUM(completion_tokens) AS completion
                FROM query_log WHERE called_at >= ?
                GROUP BY day ORDER BY day
                """,
                (since,),
            )
            daily_tokens = [dict(r) for r in token_rows]

            group_col = "substr(called_at, 1, 10)" if group_by == "day" else group_by
            volume_rows = await conn.execute_fetchall(
                f"""
                SELECT {group_col} AS group_key, COUNT(*) AS count
                FROM query_log WHERE called_at >= ?
                GROUP BY {group_col} ORDER BY count DESC
                """,
                (since,),
            )
            volume_by_group = [dict(r) for r in volume_rows]

            total_row = await conn.execute_fetchall("SELECT COUNT(*) AS n FROM query_log WHERE called_at >= ?", (since,))
            total_n = total_row[0]["n"] if total_row else 0

            error_row = await conn.execute_fetchall(
                "SELECT COUNT(*) AS n FROM query_log WHERE called_at >= ? AND (http_status != 200 OR error_text IS NOT NULL)",
                (since,),
            )
            error_n = error_row[0]["n"] if error_row else 0
            error_rate = (error_n / total_n * 100) if total_n else 0.0

            all_rows = await conn.execute_fetchall(
                "SELECT id, message_id, message, latency_seconds, length(response_text) AS rlen FROM query_log WHERE called_at >= ?",
                (since,),
            )
        finally:
            await conn.close()

        outliers: list[dict] = []
        valid_rows = [r for r in all_rows if r["latency_seconds"] is not None]
        if len(valid_rows) >= 2:
            lats = [float(r["latency_seconds"]) for r in valid_rows]
            rlens = [int(r["rlen"] or 0) for r in valid_rows]
            mean_lat = statistics.mean(lats)
            std_lat = statistics.pstdev(lats)
            mean_rlen = statistics.mean(rlens)
            std_rlen = statistics.pstdev(rlens)

            for r in valid_rows:
                reasons: list[str] = []
                lat = float(r["latency_seconds"])
                rlen = int(r["rlen"] or 0)
                if std_lat and abs(lat - mean_lat) > outlier_z * std_lat:
                    reasons.append("high_latency")
                if std_rlen and rlen < mean_rlen - outlier_z * std_rlen:
                    reasons.append("short_response")
                if std_rlen and rlen > mean_rlen + outlier_z * std_rlen:
                    reasons.append("long_response")
                if reasons:
                    outliers.append({
                        "message_id": r["message_id"],
                        "message": r["message"],
                        "latency_seconds": lat,
                        "response_length": rlen,
                        "reasons": reasons,
                    })

        return {
            "ok": True,
            "since": since,
            "group_by": group_by,
            "total_queries": total_n,
            "error_rate_pct": round(error_rate, 2),
            "latency_percentiles": latency_stats,
            "daily_tokens": daily_tokens,
            "volume_by_group": volume_by_group,
            "outliers": outliers,
        }

    except Exception as exc:
        logger.exception("get_dashboard_stats: unexpected error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


async def _read_recent_query_log(limit: int, intent: str | None) -> str:
    conn = await get_db()
    try:
        if intent:
            rows = await conn.execute_fetchall(
                "SELECT message_id, message, intent, department, latency_seconds, http_status, called_at FROM query_log WHERE intent = ? ORDER BY id DESC LIMIT ?",
                (intent, limit),
            )
        else:
            rows = await conn.execute_fetchall(
                "SELECT message_id, message, intent, department, latency_seconds, http_status, called_at FROM query_log ORDER BY id DESC LIMIT ?",
                (limit,),
            )
    finally:
        await conn.close()
    return json.dumps([dict(r) for r in rows], ensure_ascii=False, indent=2)


async def handle_list_recent_queries(limit: int = 50, intent: str | None = None) -> dict[str, Any]:
    try:
        limit = min(max(1, limit), 500)
        data = await _read_recent_query_log(limit=limit, intent=intent)
        return {"ok": True, "rows": json.loads(data)}
    except Exception as exc:
        logger.exception("list_recent_queries error: %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Observability tools and resources on MCP server."""
    # Tool: query_and_log
    mcp.tool(
        name="observability.query_and_log",
        description="Send a question to MahaSamvaad /chat endpoint and log to query_log. Always writes a row.",
    )(handle_query_and_log)

    mcp.tool(
        name="query_and_log",
        description="Alias for observability.query_and_log.",
    )(handle_query_and_log)

    # Tool: get_dashboard_stats
    mcp.tool(
        name="observability.get_dashboard_stats",
        description="Return aggregated analytics over query_log: latency, tokens, volume, error rate, outliers.",
    )(handle_get_dashboard_stats)

    mcp.tool(
        name="get_dashboard_stats",
        description="Alias for observability.get_dashboard_stats.",
    )(handle_get_dashboard_stats)

    # Tool: list_recent_queries
    mcp.tool(
        name="observability.list_recent_queries",
        description="Read recent rows from query_log. limit: max rows (capped at 500).",
    )(handle_list_recent_queries)

    mcp.tool(
        name="list_recent_queries",
        description="Alias for observability.list_recent_queries.",
    )(handle_list_recent_queries)

    # Resources
    @mcp.resource(
        "query_log://recent",
        name="recent_query_log",
        description="Last 50 rows from query_log, newest first.",
        mime_type="application/json",
    )
    async def recent_query_log() -> str:
        return await _read_recent_query_log(limit=50, intent=None)

    @mcp.resource(
        "query_log://recent{?limit,intent}",
        name="recent_query_log_filtered",
        description="Parameterised view of query_log with limit and intent filter.",
        mime_type="application/json",
    )
    async def recent_query_log_filtered(limit: int = 50, intent: str | None = None) -> str:
        limit = min(max(1, limit), 500)
        return await _read_recent_query_log(limit=limit, intent=intent)
