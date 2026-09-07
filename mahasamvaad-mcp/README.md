# MahaSamvaad Eval MCP Suite (v2.0.0)

Unified Model Context Protocol (MCP) Server for Evaluation, Observability, Grounding Scoring, Query Reformulation, and AI-Safety Evaluation of the **MahaSamvaad Government Resolution (GR) Conversational Agent**.

---

## 1. Overview & Architecture

MahaSamvaad Eval MCP consolidates all observability and evaluation layers into a **single production MCP server process** (`server.py`) backed by a shared SQLite database with WAL mode and resilient retry semantics (`storage/mahasamvaad_eval.db`).

```
mahasamvaad-mcp/
├── server.py                        # Unified MCP server entrypoint (registers all 8 domains)
├── config.py                        # Pydantic BaseSettings loading .env with fail-fast validation
├── clients/
│   ├── chat_api_client.py           # HTTP wrapper for POST /api/v1/chat (retries, timeouts, schemas)
│   ├── storage_client.py            # HTTP wrapper for GET /api/v1/storage PDF/text retrieval
│   ├── llm_judge_client.py          # NVIDIA LLM-as-a-Judge with JSON parsing and fast failover
│   ├── embedding_client.py          # nemotron-3-embed-1b embeddings with SQLite vector caching
│   └── serper_client.py             # Google Serper search API client
├── tools/
│   ├── observability/               # query_and_log, get_dashboard_stats, list_recent_queries
│   ├── grounding/                   # score_citation_coverage (embedding_v2), grounding_trend
│   ├── reformulation/               # generate_paraphrases, test_paraphrase_robustness
│   ├── hallucinated_entity/         # aieval.detect_hallucinated_entities
│   ├── lineage_correctness/         # aieval.check_lineage_correctness
│   ├── bilingual_parity/            # aieval.evaluate_bilingual_parity
│   ├── refusal_redteam/             # aieval.run_refusal_redteam_suite
│   └── corpus_web_precedence/       # aieval.check_corpus_web_precedence
├── storage/
│   ├── db.py                        # aiosqlite connection manager (WAL mode, busy retry loop)
│   └── migrations/                  # Automated SQLite schema migrations & legacy data import
├── calibration_data/
│   ├── gr_number_date_regex_patterns.py  # Regex library for GR numbers, dates, sections, figures
│   ├── lineage_ground_truth.json         # Hand-verified GR supersession/amendment chains
│   ├── bilingual_query_set.json          # Paired EN/MR queries across departments
│   ├── adversarial_queries.json          # Out-of-state, expired, loaded, off-topic queries
│   └── known_conflict_queries.json       # Curated corpus-vs-web discrepancy seed queries
├── tests/
└── .env.example
```

---

## 2. Security & Key Rotation Follow-Up

> [!CAUTION]
> **API Key Rotation Notice**:
> The NVIDIA and Serper API keys provided during initial setup and testing were shared in plaintext during development and should be considered exposed. Account owners must rotate both `LLM_JUDGE_API_KEY`, `EMBEDDING_API_KEY`, and `SERPER_API_KEY` in their respective NVIDIA and Serper consoles, and update `.env` accordingly.

---

## 3. Configuration & Environment Variables

Copy `.env.example` to `.env` and configure active credentials:

```bash
cp .env.example .env
```

| Variable | Description | Default |
| :--- | :--- | :--- |
| `CHAT_API_BASE_URL` | Base URL of the staging MahaSamvaad chat API | `http://20.40.56.211:8000` |
| `CHAT_API_CHAT_PATH` | Path to chat endpoint | `/api/v1/chat` |
| `STORAGE_API_BASE_URL` | Base URL for fetching source PDFs | `http://20.40.56.211/api/v1/storage` |
| `LLM_JUDGE_BASE_URL` | Base URL for LLM-as-a-Judge chat completions | `https://integrate.api.nvidia.com/v1` |
| `LLM_JUDGE_API_KEY` | API key for NVIDIA-hosted judge models | `nvapi-...` |
| `LLM_JUDGE_MODEL` | Primary judge model | `moonshotai/kimi-k3` |
| `REFORMULATION_MODEL` | Paraphrase and fallback judge model | `nvidia/nemotron-3.5-lightning-30b-a3b` |
| `EMBEDDING_API_KEY` | API key for embedding endpoint | `nvapi-...` |
| `EMBEDDING_MODEL` | Embedding model for grounding v2 | `nvidia/nemotron-3-embed-1b` |
| `GROUNDING_METHOD` | Grounding scoring algorithm (`embedding_v2` or `difflib_v1`) | `embedding_v2` |
| `GROUNDING_THRESHOLD`| Cosine similarity threshold to consider sentence grounded | `0.55` |
| `SERPER_API_KEY` | Google Serper search API key | `...` |
| `SQLITE_DB_PATH` | SQLite database file location | `./storage/mahasamvaad_eval.db` |
| `MCP_TRANSPORT` | MCP transport mode (`stdio` or `streamable-http`) | `stdio` |

---

## 4. Running the Unified MCP Server

### Stdio Transport (Default for Claude / Cursor / Agent IDEs)

```bash
python server.py --transport stdio
```

### Streamable-HTTP Transport (For remote staging / microservices)

```bash
python server.py --transport streamable-http --host 127.0.0.1 --port 8001
```

---

## 5. Tool Catalog

### Observability & Logging
- **`observability.query_and_log`**: Queries `/chat`, records response, metadata, tokens, latency, and returns structured result.
- **`observability.get_dashboard_stats`**: Aggregates latency percentiles (p50, p90, p99), daily token trends, volume by intent/dept, error rate, and outliers.
- **`observability.list_recent_queries`**: Reads recent rows from `query_log`.
- **Resource `query_log://recent`** / `query_log://recent{?limit,intent}`: Real-time query log view.

### Grounding / Citation Coverage (Upgraded `embedding_v2`)
- **`grounding.score_citation_coverage`**: Computes sentence embeddings for generated answers and candidate anchors using `nvidia/nemotron-3-embed-1b`, evaluates max cosine similarity per sentence, caches embeddings in SQLite `embedding_cache`, and computes `coverage_pct`.
- **`grounding.grounding_trend`**: Aggregates coverage percentiles over time sliced by department or intent.

### Query Reformulation & Stability
- **`reformulation.generate_paraphrases`**: Generates formal, informal, reordered, and Marathi question variants.
- **`reformulation.test_paraphrase_robustness`**: Executes query and variants against `/chat` (concurrency=3) and compares top retrieved `gr_number` / `filepath`.

### AI-Evaluation Tools
- **`aieval.detect_hallucinated_entities`**: Uses multilingual regex patterns to extract GR numbers, dates, sections, amounts, and verifies whether each entity is grounded in cited sources.
- **`aieval.check_lineage_correctness`**: Probes GR supersession / amendment direction against hand-verified `lineage_ground_truth.json`.
- **`aieval.evaluate_bilingual_parity`**: Evaluates completeness, quality score, and citations between paired English and Marathi questions via LLM-as-a-judge.
- **`aieval.run_refusal_redteam_suite`**: Probes out-of-state, expired, loaded, and off-topic queries to ensure localized apology / refusal compliance.
- **`aieval.check_corpus_web_precedence`**: Probes queries with conflicting corpus vs web information to verify that official GR rules take precedence per SRS guidelines.

---

## 6. Extending Calibration Data

To add new evaluation cases:

1. **Known Conflicts** (`calibration_data/known_conflict_queries.json`): Add entries with `query`, `corpus_position`, `web_position`, and `expected_behavior`.
2. **Lineage Ground Truth** (`calibration_data/lineage_ground_truth.json`): Add verified `{gr_number, department, relation_type, target_gr, expected_direction}` entries.
3. **Bilingual Queries** (`calibration_data/bilingual_query_set.json`): Add paired `{query_en, query_mr, department, key_entities}` entries.
4. **Adversarial Queries** (`calibration_data/adversarial_queries.json`): Add `{category, query, expected_refusal, reason}` entries.
5. **Regex Patterns** (`calibration_data/gr_number_date_regex_patterns.py`): Add new departmental regex patterns or date formats.

---

## 7. Running the Test Suite

```bash
# Run all unit and regression tests
pytest -v

# Run the live end-to-end smoke test against staging
python tests/run_e2e_unified_eval.py
```
