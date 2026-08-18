"""mcp_common/config.py — §4.3 environment variable loading with sane defaults."""
from __future__ import annotations

import os


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return int(val)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


# Staging endpoint — never hardcode an IP outside this config module
MAHASAMVAAD_BASE_URL: str = _env("MAHASAMVAAD_BASE_URL", "http://20.40.56.211:8000")

# Shared SQLite file — all three servers must point at the same path
STORE_DB_PATH: str = _env("STORE_DB_PATH", "./data/store.db")

# Default user_id when the caller omits one
MCP_DEFAULT_USER_ID: str = _env("MCP_DEFAULT_USER_ID", "mcp-eval-bot")

# Per-call HTTP timeout in seconds
MCP_HTTP_TIMEOUT_S: int = _env_int("MCP_HTTP_TIMEOUT_S", 60)

# Grounding method — difflib_v1 | embedding_v2
GROUNDING_METHOD: str = _env("GROUNDING_METHOD", "difflib_v1")

# Minimum match ratio to count a sentence as grounded
GROUNDING_THRESHOLD: float = _env_float("GROUNDING_THRESHOLD", 0.55)

# Python logging level for all three servers (logs to stderr only)
LOG_LEVEL: str = _env("LOG_LEVEL", "INFO")

# NVIDIA NIM API key for paraphrase generation (Server 3)
NVIDIA_API_KEY: str = _env("NVIDIA_API_KEY", "")

# NVIDIA NIM model name
NVIDIA_MODEL: str = _env("NVIDIA_MODEL", "google/gemma-4-31b-it")

# NVIDIA NIM base URL
NVIDIA_BASE_URL: str = _env("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# NVIDIA NIM per-call timeout — large models can take 2-3 minutes
NVIDIA_LLM_TIMEOUT_S: float = _env_float("NVIDIA_LLM_TIMEOUT_S", 300.0)

# MCP transport selection: stdio | streamable-http
MCP_TRANSPORT: str = _env("MCP_TRANSPORT", "stdio")

# Host/port for streamable-http transport
MCP_HTTP_HOST: str = _env("MCP_HTTP_HOST", "127.0.0.1")
MCP_HTTP_PORT: int = _env_int("MCP_HTTP_PORT", 8001)
