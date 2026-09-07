"""conftest.py — pytest configuration and module alias registration for MahaSamvaad Eval MCP."""
from __future__ import annotations

import os
import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
src_dir = root_dir / "src"
pkg_dir = src_dir / "mahasamvaad_eval"

for p in [str(pkg_dir), str(src_dir), str(root_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import mahasamvaad_eval.config as config_mod
import mahasamvaad_eval.server as server_mod
import mahasamvaad_eval.clients as clients_pkg
import mahasamvaad_eval.clients.chat_api_client as chat_api_mod
import mahasamvaad_eval.clients.embedding_client as embedding_mod
import mahasamvaad_eval.clients.falkordb_client as falkordb_mod
import mahasamvaad_eval.clients.llm_judge_client as llm_judge_mod
import mahasamvaad_eval.clients.serper_client as serper_mod
import mahasamvaad_eval.clients.storage_client as storage_mod
import mahasamvaad_eval.extractors as extractors_pkg
import mahasamvaad_eval.extractors.gr_regex as gr_regex_mod
import mahasamvaad_eval.extractors.sentence_splitter as sentence_splitter_mod
import mahasamvaad_eval.storage as storage_pkg
import mahasamvaad_eval.storage.db as db_mod
import mahasamvaad_eval.tools as tools_pkg

sys.modules["clients"] = clients_pkg
sys.modules["clients.chat_api_client"] = chat_api_mod
sys.modules["clients.embedding_client"] = embedding_mod
sys.modules["clients.falkordb_client"] = falkordb_mod
sys.modules["clients.llm_judge_client"] = llm_judge_mod
sys.modules["clients.serper_client"] = serper_mod
sys.modules["clients.storage_client"] = storage_mod
sys.modules["config"] = config_mod
sys.modules["server"] = server_mod
sys.modules["tools"] = tools_pkg
sys.modules["storage"] = storage_pkg
sys.modules["storage.db"] = db_mod
sys.modules["calibration_data.gr_number_date_regex_patterns"] = gr_regex_mod

collect_ignore = [
    "tests/test_stability_e2e.py",
    "scripts/run_e2e_unified_eval.py",
    "scripts/run_e2e_demo.py",
    "scripts/run_reformulation_model.py",
]
