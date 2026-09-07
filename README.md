# MahaSamvaad MCP Evaluation Suite

A standardized Model Context Protocol (MCP) server workspace for evaluating, observing, and testing conversational AI systems handling Maharashtra Government Resolutions (GRs).

---

## 📁 Repository Structure

```
.
├── docs/                               # Project documentation & requirements specifications
│   ├── mcp document.pdf                # Original MCP design and evaluation specifications
│   └── pdf_content.txt                 # Extracted specification text
│
├── mahasamvaad-mcp/                    # Core MCP evaluation service package
│   ├── server.py                       # Unified FastMCP server (registers all 8 domains)
│   ├── config.py                       # Pydantic Settings configuration loader
│   ├── clients/                        # External HTTP clients (Chat API, Judge LLM, Embeddings, Serper, Storage)
│   ├── tools/                          # 8 MCP domain tool implementations & handlers
│   │   ├── observability/              # Logging, query statistics, dashboard metrics
│   │   ├── grounding/                  # Citation coverage scoring via embedding_v2
│   │   ├── reformulation/              # Paraphrase generation & stability testing
│   │   ├── bilingual_parity/           # Cross-lingual EN/MR consistency evaluation
│   │   ├── hallucinated_entity/        # GR number, date, and section hallucination detection
│   │   ├── lineage_correctness/        # Superseded & amended GR relationship validation
│   │   ├── corpus_web_precedence/      # Official corpus vs public web precedence verification
│   │   └── refusal_redteam/            # Adversarial and out-of-scope refusal red-teaming
│   ├── storage/                        # SQLite connection manager & automated migrations
│   ├── calibration_data/               # Ground-truth test datasets & entity regex library
│   ├── tests/                          # Complete automated test suite (unit, integration, E2E)
│   ├── .env.example                    # Environment variable template
│   ├── pytest.ini                      # Pytest runner configuration
│   ├── requirements.txt                # Python dependencies
│   └── README.md                       # Detailed suite documentation & API reference
```

---

## 🚀 Quick Start

### 1. Environment Setup
```bash
cd mahasamvaad-mcp
cp .env.example .env
# Configure your NVIDIA and Serper API keys in .env
```

### 2. Run All Tests
```bash
pytest mahasamvaad-mcp/tests/
```

### 3. Start Unified MCP Server
```bash
python mahasamvaad-mcp/server.py
```
