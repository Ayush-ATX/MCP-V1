"""storage/db.py — Unified SQLite connection manager, WAL mode, busy retry loop."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

import aiosqlite

import config
from storage.migrations.schema import apply_schema, migrate_legacy_db

logger = logging.getLogger(__name__)

_db_initialized = False


async def get_db(db_path: str | None = None) -> aiosqlite.Connection:
    """Open a WAL-mode SQLite connection, apply schema migrations, and return open connection.

    Always close explicitly with `await conn.close()` in finally block.
    """
    global _db_initialized
    path_str = db_path or config.SQLITE_DB_PATH or config.STORE_DB_PATH
    db_file = Path(path_str).resolve()
    db_file.parent.mkdir(parents=True, exist_ok=True)

    conn = await aiosqlite.connect(str(db_file))
    conn.row_factory = aiosqlite.Row

    # Always ensure WAL mode and busy timeout
    await conn.execute("PRAGMA journal_mode=WAL;")
    await conn.execute("PRAGMA busy_timeout=5000;")
    await apply_schema(conn)

    if not _db_initialized:
        # Check legacy database if different
        legacy_cand = Path(config.STORE_DB_PATH).resolve()
        if legacy_cand != db_file and legacy_cand.is_file():
            await migrate_legacy_db(conn, legacy_cand)
        _db_initialized = True

    return conn


async def execute_with_retry(
    conn: aiosqlite.Connection,
    sql: str,
    params: tuple | list = (),
    max_retries: int = 5,
    backoff_s: float = 0.2,
) -> aiosqlite.Cursor:
    """Execute a write statement with busy-timeout retry loop."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor
        except Exception as exc:
            if "locked" in str(exc).lower() and attempt < max_retries - 1:
                logger.warning("DB locked (attempt %d/%d) — retrying in %.2fs", attempt + 1, max_retries, backoff_s * (attempt + 1))
                await asyncio.sleep(backoff_s * (attempt + 1))
                last_exc = exc
            else:
                raise
    if last_exc:
        raise last_exc
    raise RuntimeError("execute_with_retry failed without exception")
