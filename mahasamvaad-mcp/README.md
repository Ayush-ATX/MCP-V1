# MahaSamvaad MCP Server Suite

Three Model Context Protocol (MCP) servers built around the MahaSamvaad staging chat endpoint (`POST /api/v1/chat`). Together they provide an agent-accessible layer for:

| Server | Name | Purpose |
|--------|------|---------|
| 1 | `mahasamvaad-observability` | Log every chat call, dashboard stats, recent query resource |
| 2 | `mahasamvaad-grounding` | Score answer grounding via difflib_v1 sentence-anchor overlap |
| 3 | `mahasamvaad-reformulation` | Generate paraphrases + test retrieval stability under phrasing variation |

All three share a single SQLite database (`data/store.db`) in WAL mode and a common HTTP chat client.

---

## Directory Layout

```
mahasamvaad-mcp/
├── mcp_common/
│   ├── config.py          # §4.3 env var loading
│   ├── store.py           # SQLite WAL factory + 5-table schema migration
│   └── chat_client.py     # Shared POST /api/v1/chat wrapper
├── server_observability/
│   └── main.py            # Server 1 — query_and_log, get_dashboard_stats, query_log://recent
├── server_grounding/
│   └── main.py            # Server 2 — score_grounding, grounding_trend
├── server_reformulation/
│   └── main.py            # Server 3 — generate_paraphrases, test_reformulation_stability
├── tests/
│   ├── test_chat_client.py        # Unit: timeout/5xx/4xx/malformed JSON/UUID
│   ├── test_sentence_splitter.py  # Unit: Devanagari danda + Latin full stop
│   ├── test_grounding.py          # Unit: difflib_v1, hand-labelled samples
│   ├── test_stability_score.py    # Unit: stability score arithmetic
│   ├── test_reformulation_unit.py # Unit: generate_paraphrases + stability (mocked)
│   ├── test_concurrent_writes.py  # Phase 5: concurrent write retry
│   └── run_e2e_demo.py            # 15-question end-to-end demo
├── data/
│   └── store.db           # Populated at runtime; excluded from VCS
├── requirements.txt
├── pytest.ini
└── .env.example
```

---

## Prerequisites

- **Python 3.11** (the venv is `venv311/` in the project root, one level up)
- **Python venv** created with: `py -3.11 -m venv venv311`
- Dependencies installed: `venv311\Scripts\pip install -r requirements.txt`

---

## Environment Variables (§4.3)

Copy `.env.example` to `.env` and adjust:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MAHASAMVAAD_BASE_URL` | `http://20.40.56.211:8000` | Staging endpoint |
| `STORE_DB_PATH` | `./data/store.db` | Shared SQLite file — **all three servers must point here** |
| `MCP_DEFAULT_USER_ID` | `mcp-eval-bot` | Default `user_id` for chat calls |
| `MCP_HTTP_TIMEOUT_S` | `60` | Per-call timeout in seconds |
| `GROUNDING_METHOD` | `difflib_v1` | `difflib_v1` \| `embedding_v2` (stretch) |
| `GROUNDING_THRESHOLD` | `0.55` | Minimum difflib ratio to count a sentence as grounded |
| `LOG_LEVEL` | `INFO` | Python logging level (all servers log to **stderr** only) |
| `REFORMULATION_API_KEY` | *(required for Server 3)* | API key for the reformulation model |
| `REFORMULATION_MODEL` | `nvidia/nemotron-3.5-lightning-30b-a3b` | Reformulation model identifier |
| `REFORMULATION_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible model API base URL |
| `MCP_TRANSPORT` | `stdio` | `stdio` \| `streamable-http` |
| `MCP_HTTP_HOST` | `127.0.0.1` | Bind address for streamable-http (localhost only — §11) |
| `MCP_HTTP_PORT` | `8001` | Port for streamable-http |

---

## Running Each Server

### stdio mode (development — default)

```powershell
# Server 1 — Observability
$env:STORE_DB_PATH="./data/store.db"
..\venv311\Scripts\python server_observability\main.py

# Server 2 — Grounding
..\venv311\Scripts\python server_grounding\main.py

# Server 3 — Reformulation
$env:REFORMULATION_API_KEY="..."
..\venv311\Scripts\python server_reformulation\main.py
```

### streamable-http mode (staging)

```powershell
# Server 1 on port 8001
..\venv311\Scripts\python server_observability\main.py --transport streamable-http --port 8001

# Server 2 on port 8002
..\venv311\Scripts\python server_grounding\main.py    --transport streamable-http --port 8002

# Server 3 on port 8003
..\venv311\Scripts\python server_reformulation\main.py --transport streamable-http --port 8003
```

> **Security (§11):** All servers bind to `127.0.0.1` by default. Never expose them on a public interface. `data/store.db` may contain sensitive query content — exclude from any public repo.

---

## Tools & Resources

### Server 1 — mahasamvaad-observability

| Name | Type | Description |
|------|------|-------------|
| `query_and_log` | Tool | Send a question to `/chat`, always writes to `query_log` |
| `get_dashboard_stats` | Tool | Latency p50/p90/p99, token cost, volume by group, error rate, outliers |
| `list_recent_queries` | Tool | Recent rows from `query_log` (limit capped at 500, optional intent filter) |
| `query_log://recent` | Resource | Last 50 rows, read-only, no live API call |

### Server 2 — mahasamvaad-grounding

| Name | Type | Description |
|------|------|-------------|
| `score_grounding` | Tool | difflib_v1 sentence-anchor overlap for a logged `message_id` |
| `grounding_trend` | Tool | Per-slice mean/median coverage trend, 7-day rolling direction |

### Server 3 — mahasamvaad-reformulation

| Name | Type | Description |
|------|------|-------------|
| `generate_paraphrases` | Tool | Gemini-generated variants tagged `formal\|informal\|reorder\|marathi` |
| `test_reformulation_stability` | Tool | Fires original + variants at `/chat`, compares `gr_number`, stability score |

---

## Running Tests

```powershell
# All unit tests (33 tests, no network required)
..\venv311\Scripts\pytest tests\ -v

# Phase 5 concurrent-write test
..\venv311\Scripts\pytest tests\test_concurrent_writes.py -v

# 2. E2E demo (staging + REFORMULATION_API_KEY required)
$env:REFORMULATION_API_KEY="..."
$env:STORE_DB_PATH="./data/store.db"
..\venv311\Scripts\python tests\run_e2e_demo.py
```

---

## Database Schema (§4.1)

Five tables, all created idempotently (`CREATE TABLE IF NOT EXISTS`):

- `query_log` — one row per `/chat` call (written by all 3 servers via shared `mcp_common`)
- `grounding_scores` — one row per `score_grounding` call (Server 2)
- `reformulation_runs` — one row per `test_reformulation_stability` call (Server 3)
- `reformulation_variants` — one row per fired variant (Server 3)

---

## Design Notes

- **Option A cross-server wiring (§8):** Servers 2 & 3 import `call_chat` and store functions from `mcp_common` directly — no MCP-to-MCP composition for v1.
- **Pluggable grounding strategy:** `DifflibV1Strategy` implements `GroundingStrategy` protocol; `embedding_v2` is a clearly marked stretch hook.
- **Option B (stretch hook):** MCP-to-MCP composition via `ClientSession` — not implemented in v1.
- **stdout is clean:** All logging goes to stderr only; stdout is reserved for the MCP stdio protocol channel.
- **Busy-timeout retry:** All DB writes use a 5-retry loop with backoff; SQLite `PRAGMA busy_timeout=5000ms` is set on every connection.
