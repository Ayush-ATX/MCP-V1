"""config.py — Central typed configuration and environment loader for MahaSamvaad Eval MCP."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_env_file() -> str | None:
    candidates = [
        os.environ.get("DOTENV_PATH"),
        str(Path(__file__).resolve().parent / ".env"),
        str(Path.cwd() / ".env"),
        str(Path.cwd() / "mahasamvaad-mcp" / ".env"),
    ]
    for c in candidates:
        if c and Path(c).is_file():
            return c
    return None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_find_env_file(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Staging BG chat endpoint ---
    CHAT_API_BASE_URL: str = Field(default="http://20.40.56.211:8000")
    CHAT_API_CHAT_PATH: str = Field(default="/api/v1/chat")
    CHAT_API_CHAT_STREAM_PATH: str = Field(default="/api/v1/chat/stream")
    CHAT_API_HEALTH_PATH: str = Field(default="/health")
    MAHASAMVAAD_BASE_URL: str = Field(default="http://20.40.56.211:8000")

    # --- Storage endpoint ---
    STORAGE_API_BASE_URL: str = Field(default="http://20.40.56.211/api/v1/storage")

    # --- Secondary inference endpoint ---
    INFERENCE_BASE_URL: str = Field(default="http://20.246.76.245:3001")

    # --- LLM-as-judge (NVIDIA-hosted) ---
    LLM_JUDGE_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1")
    LLM_JUDGE_API_KEY: str = Field(default="")
    LLM_JUDGE_MODEL: str = Field(default="moonshotai/kimi-k3")

    # --- Reformulation model ---
    REFORMULATION_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1")
    REFORMULATION_API_KEY: str = Field(default="")
    REFORMULATION_MODEL: str = Field(default="nvidia/nemotron-3.5-lightning-30b-a3b")

    # --- Embedding model for grounding v2 ---
    EMBEDDING_BASE_URL: str = Field(default="https://integrate.api.nvidia.com/v1")
    EMBEDDING_API_KEY: str = Field(default="")
    EMBEDDING_MODEL: str = Field(default="nvidia/nemotron-3-embed-1b")

    # --- Grounding method ---
    GROUNDING_METHOD: str = Field(default="embedding_v2")
    GROUNDING_THRESHOLD: float = Field(default=0.55)

    # --- Web search ---
    SERPER_API_KEY: str = Field(default="")

    # --- SQLite Database ---
    SQLITE_DB_PATH: str = Field(default="./storage/mahasamvaad_eval.db")
    STORE_DB_PATH: str = Field(default="./storage/mahasamvaad_eval.db")

    # --- MCP transport ---
    MCP_TRANSPORT: str = Field(default="stdio")
    MCP_HTTP_HOST: str = Field(default="127.0.0.1")
    MCP_HTTP_PORT: int = Field(default=8001)
    MCP_DEFAULT_USER_ID: str = Field(default="mcp-eval-bot")
    MCP_HTTP_TIMEOUT_S: int = Field(default=60)
    LOG_LEVEL: str = Field(default="INFO")

    @field_validator("GROUNDING_METHOD")
    @classmethod
    def validate_grounding_method(cls, v: str) -> str:
        v_clean = v.strip().lower()
        if v_clean not in ("difflib_v1", "embedding_v2"):
            raise ValueError(f"GROUNDING_METHOD must be 'difflib_v1' or 'embedding_v2', got '{v}'")
        return v_clean

    def validate_required_keys(self, check_api_keys: bool = False) -> list[str]:
        """Verify that required URLs and (optionally) API keys are present."""
        missing: list[str] = []
        if not self.CHAT_API_BASE_URL:
            missing.append("CHAT_API_BASE_URL")
        if not self.STORAGE_API_BASE_URL:
            missing.append("STORAGE_API_BASE_URL")
        if not self.SQLITE_DB_PATH:
            missing.append("SQLITE_DB_PATH")
        if check_api_keys:
            if not self.LLM_JUDGE_API_KEY:
                missing.append("LLM_JUDGE_API_KEY")
            if not self.EMBEDDING_API_KEY:
                missing.append("EMBEDDING_API_KEY")
            if not self.SERPER_API_KEY:
                missing.append("SERPER_API_KEY")
        return missing


# Instantiate singleton
settings = Settings()

# Module-level exports for backwards compatibility with all imports
CHAT_API_BASE_URL: str = settings.CHAT_API_BASE_URL
CHAT_API_CHAT_PATH: str = settings.CHAT_API_CHAT_PATH
CHAT_API_CHAT_STREAM_PATH: str = settings.CHAT_API_CHAT_STREAM_PATH
CHAT_API_HEALTH_PATH: str = settings.CHAT_API_HEALTH_PATH
MAHASAMVAAD_BASE_URL: str = settings.MAHASAMVAAD_BASE_URL
STORAGE_API_BASE_URL: str = settings.STORAGE_API_BASE_URL
INFERENCE_BASE_URL: str = settings.INFERENCE_BASE_URL
LLM_JUDGE_BASE_URL: str = settings.LLM_JUDGE_BASE_URL
LLM_JUDGE_API_KEY: str = settings.LLM_JUDGE_API_KEY
LLM_JUDGE_MODEL: str = settings.LLM_JUDGE_MODEL
REFORMULATION_BASE_URL: str = settings.REFORMULATION_BASE_URL
REFORMULATION_API_KEY: str = settings.REFORMULATION_API_KEY
REFORMULATION_MODEL: str = settings.REFORMULATION_MODEL
EMBEDDING_BASE_URL: str = settings.EMBEDDING_BASE_URL
EMBEDDING_API_KEY: str = settings.EMBEDDING_API_KEY
EMBEDDING_MODEL: str = settings.EMBEDDING_MODEL
GROUNDING_METHOD: str = settings.GROUNDING_METHOD
GROUNDING_THRESHOLD: float = settings.GROUNDING_THRESHOLD
SERPER_API_KEY: str = settings.SERPER_API_KEY
SQLITE_DB_PATH: str = settings.SQLITE_DB_PATH
STORE_DB_PATH: str = settings.STORE_DB_PATH
MCP_TRANSPORT: str = settings.MCP_TRANSPORT
MCP_HTTP_HOST: str = settings.MCP_HTTP_HOST
MCP_HTTP_PORT: int = settings.MCP_HTTP_PORT
MCP_DEFAULT_USER_ID: str = settings.MCP_DEFAULT_USER_ID
MCP_HTTP_TIMEOUT_S: int = settings.MCP_HTTP_TIMEOUT_S
LOG_LEVEL: str = settings.LOG_LEVEL
