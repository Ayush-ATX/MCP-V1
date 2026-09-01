"""mcp_common/store.py — Backwards-compatible bridge forwarding to storage.db."""
from __future__ import annotations

import sys
from pathlib import Path

_root = str(Path(__file__).resolve().parent.parent)
if _root not in sys.path:
    sys.path.insert(0, _root)

from storage.db import get_db, execute_with_retry

__all__ = ["get_db", "execute_with_retry"]
