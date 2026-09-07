# MahaSamvaad MCP Evaluation Suite (v2.0.0)

A standardized Model Context Protocol (MCP) server workspace for evaluating, observing, and testing conversational AI systems handling Maharashtra Government Resolutions (GRs).

---

## 📁 Repository Structure

```
.
├── .env / .env.example                  # Central environment configuration & secrets
├── .gitignore                           # Standardized repository ignore rules
├── requirements.txt                     # Python dependencies (FastMCP, Pydantic, FalkorDB, etc.)
├── pytest.ini                           # Pytest configuration with pythonpath = src
├── conftest.py                          # Global test fixtures & module aliases
├── README.md                            # Suite architecture & API specifications
│
├── docs/                                # Technical documentation & SRS specifications
│   ├── mcp document.pdf                 # Original MCP design and evaluation specifications
│   └── pdf_content.txt                  # Extracted specification text
│
├── benchmarks/                          # Evaluation datasets & ground-truth baselines
│   ├── adversarial_queries.json         # Out-of-state, expired, and loaded queries
│   ├── bilingual_query_set.json         # Paired EN/MR queries across departments
│   ├── known_conflict_queries.json      # Curated corpus-vs-web discrepancy seed queries
│   └── lineage_ground_truth.json        # Verified GR supersession/amendment chains
│
├── scripts/                             # Runnable evaluation CLI runners
│   ├── run_e2e_unified_eval.py          # Full E2E evaluation suite runner
│   ├── run_e2e_demo.py                  # Demo evaluation runner
│   └── run_reformulation_model.py       # Standalone paraphrase model runner
│
├── reports/                             # Generated evaluation reports (git-ignored)
│   ├── demo_output.json
│   └── demo_output.txt
│
├── src/                                 # Production MCP Service Package
│   └── mahasamvaad_eval/
│       ├── __init__.py
│       ├── server.py                    # Unified FastMCP server entrypoint (registers 8 domains)
│       ├── config.py                    # Pydantic Settings with fail-fast validation
│       │
│       ├── clients/                     # External API client connectors
│       │   ├── __init__.py
│       │   ├── chat_api_client.py       # Staging MahaSamvaad chat API HTTP client
│       │   ├── falkordb_client.py       # Read-only Knowledge Graph client with 4-layer firewall
│       │   ├── llm_judge_client.py      # NVIDIA LLM-as-a-Judge client
│       │   ├── embedding_client.py      # Nemotron embedding client with vector caching
│       │   ├── storage_client.py        # Document & PDF retrieval client
│       │   └── serper_client.py         # Google Serper web search client
│       │
│       ├── tools/                       # 8 MCP Evaluation Domains
│       │   ├── __init__.py
│       │   ├── observability.py         # Query logging, dashboard stats, analytics
│       │   ├── grounding.py             # Grounding scoring (embedding_v2 & difflib_v1)
│       │   ├── reformulation.py         # Query paraphrasing & stability scoring
│       │   ├── bilingual_parity.py      # Cross-lingual EN/MR semantic parity evaluation
│       │   ├── hallucinated_entity.py   # GR entity extraction & fact verification
│       │   ├── lineage_correctness.py   # FalkorDB graph lineage traversal & direction check
│       │   ├── corpus_web_precedence.py # Official corpus vs public web priority check
│       │   └── refusal_redteam.py       # Adversarial refusal red-team testing
│       │
│       ├── extractors/                  # Specialized parsers & regex libraries
│       │   ├── __init__.py
│       │   ├── gr_regex.py              # Departmental GR numbers, dates & section regexes
│       │   └── sentence_splitter.py     # Devanagari & Latin sentence boundary splitter
│       │
│       └── storage/                     # SQLite persistence & schema migrations
│           ├── __init__.py
│           ├── db.py                    # Async connection pool & retry logic
│           └── migrations/              # Automated schema migrations
│
└── tests/                               # Comprehensive Automated Test Suite
    ├── test_ai_eval_tools.py
    ├── test_chat_client.py
    ├── test_clients.py
    ├── test_concurrent_writes.py
    ├── test_grounding.py
    ├── test_grounding_v2_regression.py
    ├── test_observability_logging.py
    ├── test_reformulation_unit.py
    ├── test_sentence_splitter.py
    ├── test_server_unified.py
    ├── test_stability_score.py
    └── test_streamable_http_startup.py
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cp .env.example .env
# Configure your active API keys in .env
```

### 2. Run All Automated Tests
```bash
pytest
```

### 3. Start Unified FastMCP Server
```bash
# stdio transport (default)
python src/mahasamvaad_eval/server.py

# streamable-http transport
python src/mahasamvaad_eval/server.py --transport streamable-http --port 8001
```

### 4. Run E2E Benchmark Suite
```bash
python scripts/run_e2e_unified_eval.py
```
