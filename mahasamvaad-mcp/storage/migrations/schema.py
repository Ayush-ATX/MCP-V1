"""storage/migrations/schema.py — Database schema definition and migration logic."""
from __future__ import annotations

import logging
from pathlib import Path
import aiosqlite

logger = logging.getLogger(__name__)

SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;

-- 1. Query Log (observability & chat history)
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

-- 2. Grounding Scores (citation coverage v1 & v2)
CREATE TABLE IF NOT EXISTS grounding_scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL REFERENCES query_log(message_id),
    scored_at TEXT NOT NULL,
    coverage_pct REAL,
    total_sentences INTEGER,
    grounded_count INTEGER,
    ungrounded_json TEXT,
    method TEXT DEFAULT 'embedding_v2'
);

CREATE INDEX IF NOT EXISTS idx_ground_msgid ON grounding_scores(message_id);
CREATE INDEX IF NOT EXISTS idx_ground_method ON grounding_scores(method);

-- 3. Reformulation Runs & Variants
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

CREATE INDEX IF NOT EXISTS idx_reform_run_id ON reformulation_variants(run_id);

-- 4. Embedding Cache (text hash + model -> vector blob)
CREATE TABLE IF NOT EXISTS embedding_cache (
    text_hash TEXT NOT NULL,
    model TEXT NOT NULL,
    embedding_blob TEXT NOT NULL,
    text_preview TEXT,
    created_at TEXT NOT NULL,
    PRIMARY KEY (text_hash, model)
);

-- 5. AI-Eval: Hallucinated Entity Checks
CREATE TABLE IF NOT EXISTS hallucinated_entity_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chatroom_id TEXT,
    message_id TEXT,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    found_in_source INTEGER NOT NULL,
    citation_id_claimed TEXT,
    sentence TEXT,
    details_json TEXT,
    checked_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_halluc_msgid ON hallucinated_entity_checks(message_id);

-- 6. AI-Eval: Lineage Correctness Checks
CREATE TABLE IF NOT EXISTS lineage_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    gr_number TEXT NOT NULL,
    relation_type TEXT,
    expected_direction TEXT,
    actual_direction TEXT,
    correct INTEGER NOT NULL,
    details_json TEXT,
    checked_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_lineage_gr ON lineage_checks(gr_number);

-- 7. AI-Eval: Bilingual Parity Checks
CREATE TABLE IF NOT EXISTS bilingual_parity_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query_en TEXT NOT NULL,
    query_mr TEXT NOT NULL,
    score_en REAL,
    score_mr REAL,
    gap REAL,
    sources_count_en INTEGER,
    sources_count_mr INTEGER,
    parity_passed INTEGER NOT NULL,
    details_json TEXT,
    checked_at TEXT NOT NULL
);

-- 8. AI-Eval: Refusal / Red-Team Checks
CREATE TABLE IF NOT EXISTS refusal_redteam_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    category TEXT NOT NULL,
    expected_refusal INTEGER NOT NULL,
    actually_refused INTEGER NOT NULL,
    response_text TEXT,
    details_json TEXT,
    checked_at TEXT NOT NULL
);

-- 9. AI-Eval: Corpus vs Web Precedence Checks
CREATE TABLE IF NOT EXISTS precedence_checks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    corpus_position TEXT,
    web_position TEXT,
    response_classification TEXT NOT NULL,
    details_json TEXT,
    checked_at TEXT NOT NULL
);
"""


async def apply_schema(conn: aiosqlite.Connection) -> None:
    """Apply the full schema to the given SQLite connection."""
    await conn.executescript(SCHEMA_SQL)
    await conn.commit()


async def migrate_legacy_db(target_conn: aiosqlite.Connection, legacy_db_path: str | Path) -> None:
    """Migrate historical rows from legacy store.db if it exists."""
    legacy_file = Path(legacy_db_path)
    if not legacy_file.is_file():
        return

    try:
        legacy_conn = await aiosqlite.connect(str(legacy_file))
        legacy_conn.row_factory = aiosqlite.Row
        try:
            # Check if query_log table exists in legacy
            tables_cursor = await legacy_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='query_log'")
            if await tables_cursor.fetchone():
                rows = await legacy_conn.execute_fetchall("SELECT * FROM query_log")
                for r in rows:
                    cols = [k for k in r.keys() if k != "id"]
                    placeholders = ", ".join("?" for _ in cols)
                    col_names = ", ".join(cols)
                    values = [r[c] for c in cols]
                    await target_conn.execute(
                        f"INSERT OR IGNORE INTO query_log ({col_names}) VALUES ({placeholders})",
                        values,
                    )

            # Check grounding_scores
            g_cursor = await legacy_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='grounding_scores'")
            if await g_cursor.fetchone():
                rows = await legacy_conn.execute_fetchall("SELECT * FROM grounding_scores")
                for r in rows:
                    cols = [k for k in r.keys() if k != "id"]
                    placeholders = ", ".join("?" for _ in cols)
                    col_names = ", ".join(cols)
                    values = [r[c] for c in cols]
                    await target_conn.execute(
                        f"INSERT OR IGNORE INTO grounding_scores ({col_names}) VALUES ({placeholders})",
                        values,
                    )

            await target_conn.commit()
            logger.info("Migrated historical records from legacy DB: %s", legacy_file)
        finally:
            await legacy_conn.close()
    except Exception as exc:
        logger.warning("Legacy DB migration skipped or failed gracefully: %s", exc)
