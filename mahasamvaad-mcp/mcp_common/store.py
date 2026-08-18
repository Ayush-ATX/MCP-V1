"""mcp_common/store.py — SQLite connection factory (WAL mode, busy_timeout) and schema migration.

All 5 tables + indexes are created exactly as specified in §4.1.
Migration is idempotent (CREATE TABLE IF NOT EXISTS).

Usage pattern (always explicit close):
    conn = await get_db()
    try:
        cursor = await execute_with_retry(conn, ...)
    finally:
        await conn.close()
"""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import aiosqlite

from mcp_common.config import STORE_DB_PATH

logger = logging.getLogger(__name__)

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

CREATE TABLE IF NOT EXISTS query_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    called_at TEXT NOT NULL,
    user_id TEXT,
    chatroom_id TEXT,
    user_message_id TEXT,
    message_id TEXT,
    message TEXT NOT NULL,
    model_name TEXT,
    web_search INTEGER,
    response_text TEXT,
    intent TEXT,
    language TEXT,
    route_hint TEXT,
    department TEXT,
    sources_json TEXT,
    web_sources_json TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    total_tokens INTEGER,
    latency_seconds REAL,
    langfuse_trace_id TEXT,
    http_status INTEGER,
    error_text TEXT,
    raw_response_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_query_log_time   ON query_log(called_at);
CREATE INDEX IF NOT EXISTS idx_query_log_intent ON query_log(intent);
CREATE INDEX IF NOT EXISTS idx_query_log_dept   ON query_log(department);
CREATE INDEX IF NOT EXISTS idx_query_log_msgid  ON query_log(message_id);

CREATE TABLE IF NOT EXISTS grounding_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES query_log(message_id),
    scored_at TEXT NOT NULL,
    coverage_pct REAL,
    total_sentences INTEGER,
    grounded_count INTEGER,
    ungrounded_json TEXT,
    method TEXT DEFAULT 'difflib_v1'
);

CREATE INDEX IF NOT EXISTS idx_ground_msgid ON grounding_scores(message_id);

CREATE TABLE IF NOT EXISTS reformulation_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_at TEXT NOT NULL,
    original_query TEXT NOT NULL,
    stability_score REAL,
    majority_gr TEXT,
    odd_one_out_ids TEXT
);

CREATE TABLE IF NOT EXISTS reformulation_variants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES reformulation_runs(id),
    variant_text TEXT NOT NULL,
    variant_kind TEXT,
    message_id TEXT,
    top_gr_number TEXT,
    top_filepath TEXT
);
"""


async def get_db(db_path: str | None = None) -> aiosqlite.Connection:
    """Open a WAL-mode SQLite connection, apply PRAGMA settings, and run schema migration.

    Returns an open aiosqlite.Connection. Caller is responsible for closing it via
    `await conn.close()`. Do NOT use `async with await get_db()` — that pattern
    double-initialises the underlying thread and raises RuntimeError.

    Pattern:
        conn = await get_db()
        try:
            ...
        finally:
            await conn.close()
    """
    path = db_path or STORE_DB_PATH
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    # aiosqlite.connect() returns a Connection; await it to open the thread.
    conn = await aiosqlite.connect(path)
    conn.row_factory = aiosqlite.Row
    # Enable WAL and busy_timeout for concurrent writers; executescript auto-commits.
    await conn.executescript(_SCHEMA_SQL)
    return conn


async def execute_with_retry(
    conn: aiosqlite.Connection,
    sql: str,
    params: tuple = (),
    max_retries: int = 5,
    backoff_s: float = 0.2,
) -> aiosqlite.Cursor:
    """Execute a write statement with busy-timeout retry loop.

    SQLite WAL mode serialises writers; if a lock is held, retry with backoff.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor
        except Exception as exc:  # noqa: BLE001
            if "locked" in str(exc).lower() and attempt < max_retries - 1:
                logger.warning("DB locked (attempt %d/%d) — retrying", attempt + 1, max_retries)
                await asyncio.sleep(backoff_s * (attempt + 1))
                last_exc = exc
            else:
                raise
    raise last_exc  # type: ignore[misc]
