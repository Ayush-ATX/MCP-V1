"""Phase 5 — Concurrent-write retry test.

Fires overlapping writes from two coroutines to the same store.db and confirms
no 'database is locked' errors surface to the caller.
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from mahasamvaad_eval.storage.db import execute_with_retry, get_db


@pytest.mark.asyncio
async def test_concurrent_writes_no_lock_error():
    """Multiple concurrent writers must not surface 'database is locked' errors."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_concurrent.db")

        # Initialise schema
        init_conn = await get_db(db_path)
        await init_conn.close()

        errors: list[Exception] = []

        async def _writer(worker_id: int, n: int = 20) -> None:
            conn = await get_db(db_path)
            try:
                for i in range(n):
                    await execute_with_retry(
                        conn,
                        "INSERT INTO query_log (called_at, message, http_status) VALUES (datetime('now'), ?, 200)",
                        (f"worker-{worker_id}-msg-{i}",),
                    )
            except Exception as exc:
                errors.append(exc)
            finally:
                await conn.close()

        # Fire 3 writers concurrently
        await asyncio.gather(
            _writer(1),
            _writer(2),
            _writer(3),
        )

        assert not errors, f"Lock errors surfaced: {[str(e) for e in errors]}"

        # Verify all rows written
        verify_conn = await get_db(db_path)
        try:
            rows = await verify_conn.execute_fetchall("SELECT COUNT(*) AS n FROM query_log")
            total = rows[0]["n"]
        finally:
            await verify_conn.close()

        assert total == 60, f"Expected 60 rows (3x20), got {total}"
