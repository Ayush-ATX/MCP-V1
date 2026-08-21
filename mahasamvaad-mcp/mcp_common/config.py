"""mcp_common/config.py — §4.3 environment variable loading with sane defaults."""
from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(dotenv_path: str | Path | None = None, override: bool = False) -> None:
    """Load environment variables from a .env file into os.environ.

    Searches in:
    1. Explicit path if provided
    2. os.environ["DOTENV_PATH"] if set
    3. Project root relative to this file (../.env)
    4. Current working directory (.env)

    Existing environment variables are preserved unless override=True.
    """
    candidates: list[Path] = []
    if dotenv_path:
        candidates.append(Path(dotenv_path))
    elif "DOTENV_PATH" in os.environ:
        candidates.append(Path(os.environ["DOTENV_PATH"]))
    else:
        candidates.append(Path(__file__).resolve().parent.parent / ".env")
        candidates.append(Path.cwd() / ".env")

    target_file: Path | None = None
    for cand in candidates:
        if cand.is_file():
            target_file = cand
            break

    if not target_file:
        return

    try:
        with open(target_file, encoding="utf-8") as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("export "):
                    line = line[7:].strip()
                if "=" not in line:
                    continue
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip()
                if not key:
                    continue
                # Strip matching surrounding quotes
                if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                    val = val[1:-1]
                if override or key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


# Automatically load .env at module import time
load_dotenv()


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

# Server 3 reformulation model (OpenAI-compatible streaming client)
REFORMULATION_API_KEY: str = _env("REFORMULATION_API_KEY", "")

REFORMULATION_MODEL: str = _env("REFORMULATION_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")

REFORMULATION_BASE_URL: str = _env("REFORMULATION_BASE_URL", "https://integrate.api.nvidia.com/v1")


# MCP transport selection: stdio | streamable-http
MCP_TRANSPORT: str = _env("MCP_TRANSPORT", "stdio")

# Host/port for streamable-http transport
MCP_HTTP_HOST: str = _env("MCP_HTTP_HOST", "127.0.0.1")
MCP_HTTP_PORT: int = _env_int("MCP_HTTP_PORT", 8001)

