"""mcp_common/config.py — §4.3 environment variable loading with sane defaults and bridge to config.py."""
from __future__ import annotations

import os
from pathlib import Path
import sys

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)


def load_dotenv(dotenv_path: str | Path | None = None, override: bool = False) -> None:
    """Load environment variables from a .env file into os.environ."""
    candidates: list[Path] = []
    if dotenv_path:
        candidates.append(Path(dotenv_path))
    elif "DOTENV_PATH" in os.environ:
        candidates.append(Path(os.environ["DOTENV_PATH"]))
    else:
        candidates.append(Path(__file__).resolve().parent.parent / ".env")
        candidates.append(Path.cwd() / ".env")
        candidates.append(Path.cwd() / "mahasamvaad-mcp" / ".env")

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
                if len(val) >= 2 and ((val[0] == '"' and val[-1] == '"') or (val[0] == "'" and val[-1] == "'")):
                    val = val[1:-1]
                if override or key not in os.environ:
                    os.environ[key] = val
    except OSError:
        pass


load_dotenv()

from config import (
    CHAT_API_BASE_URL,
    CHAT_API_CHAT_PATH,
    CHAT_API_CHAT_STREAM_PATH,
    CHAT_API_HEALTH_PATH,
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    GROUNDING_METHOD,
    GROUNDING_THRESHOLD,
    INFERENCE_BASE_URL,
    LLM_JUDGE_API_KEY,
    LLM_JUDGE_BASE_URL,
    LLM_JUDGE_MODEL,
    LOG_LEVEL,
    MAHASAMVAAD_BASE_URL,
    MCP_DEFAULT_USER_ID,
    MCP_HTTP_HOST,
    MCP_HTTP_PORT,
    MCP_HTTP_TIMEOUT_S,
    MCP_TRANSPORT,
    REFORMULATION_API_KEY,
    REFORMULATION_BASE_URL,
    REFORMULATION_MODEL,
    SERPER_API_KEY,
    SQLITE_DB_PATH,
    STORE_DB_PATH,
    STORAGE_API_BASE_URL,
    Settings,
    settings,
)
