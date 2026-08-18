"""Server 3 — mahasamvaad-reformulation (§7)

Tools:
  - generate_paraphrases         (§7.1) — single LLM call, temperature ~0.7
  - test_reformulation_stability (§7.2) — Option A: imports call_chat directly,
                                          semaphore=3, compares gr_number/filepath

STRETCH HOOK (not implemented): Option B — MCP-to-MCP composition via ClientSession.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any

import httpx

import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_common import config
from mcp_common.chat_client import call_chat
from mcp_common.store import execute_with_retry, get_db

logging.basicConfig(
    stream=sys.stderr,
    level=getattr(logging, config.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s [reformulation] %(message)s",
)
logger = logging.getLogger(__name__)

from mcp.server.mcpserver import MCPServer

mcp = MCPServer(
    name="mahasamvaad-reformulation",
    version="1.0.0",
    description="Query Reformulation Tester for MahaSamvaad — paraphrase + retrieval stability",
)


# ─────────────────────────────────────────────────────────────────────────────
# LLM paraphrase generation via Google Gemini
# ─────────────────────────────────────────────────────────────────────────────
def _build_paraphrase_prompt(query: str, n: int, include_marathi: bool) -> str:
    kinds_desc = "formal (official/legal language), informal (conversational), reorder (same words, different order/structure)"
    if include_marathi:
        kinds_desc += ", marathi (Marathi translation)"

    return f"""You are a multilingual paraphrase generator for Indian government scheme/act queries.

Produce exactly {n} paraphrases of the following question, preserving its exact meaning.
Tag each with its kind: formal | informal | reorder{' | marathi' if include_marathi else ''}.
Return ONLY valid JSON — an array of objects with "text" and "kind" fields.

Kinds to produce: {kinds_desc}

Original question: {query}

Output format (example):
[
  {{"text": "...", "kind": "formal"}},
  {{"text": "...", "kind": "informal"}},
  {{"text": "...", "kind": "reorder"}}
]"""


async def _call_nvidia_nim(prompt: str) -> str:
    """Call NVIDIA NIM API to generate paraphrases. Returns raw response text.

    Uses httpx (already a project dependency) to POST to the OpenAI-compatible
    /chat/completions endpoint at integrate.api.nvidia.com.
    """
    api_key = config.NVIDIA_API_KEY
    if not api_key:
        raise ValueError("NVIDIA_API_KEY is not set.")

    url = f"{config.NVIDIA_BASE_URL}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
    }
    payload = {
        "model": config.NVIDIA_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 4096,
        "temperature": 0.7,
        "top_p": 0.95,
        "stream": False,
    }

    # Timeout from config — large NIM models (gemma-4-31b-it) can take 2-3 min
    async with httpx.AsyncClient(timeout=httpx.Timeout(config.NVIDIA_LLM_TIMEOUT_S)) as client:
        resp = await client.post(url, headers=headers, json=payload)
        resp.raise_for_status()
        data = resp.json()

    # OpenAI-compatible response: choices[0].message.content
    return data["choices"][0]["message"]["content"]


# ─────────────────────────────────────────────────────────────────────────────
# Tool: generate_paraphrases (§7.1)
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="generate_paraphrases",
    description=(
        "Generate N paraphrases of a government-scheme question using NVIDIA NIM LLM. "
        "Variants tagged formal | informal | reorder | marathi. "
        "Does NOT fire them at /chat — use test_reformulation_stability for that."
    ),
)
async def generate_paraphrases(
    query: str,
    n: int = 4,
    include_marathi: bool = False,
) -> dict[str, Any]:
    """§7.1 — Single LLM call, temperature ~0.7, variants tagged by kind."""
    try:
        n = max(1, min(n, 8))
        prompt = _build_paraphrase_prompt(query, n, include_marathi)
        raw_text = await _call_nvidia_nim(prompt)

        cleaned = raw_text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            variants = json.loads(cleaned)
            if not isinstance(variants, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("generate_paraphrases: LLM returned non-JSON — %s\nRaw: %s", exc, raw_text[:500])
            return {"ok": False, "error": f"LLM returned invalid JSON: {exc}", "http_status": None}

        clean_variants = []
        for v in variants:
            if isinstance(v, dict) and "text" in v and "kind" in v:
                clean_variants.append({"text": str(v["text"]), "kind": str(v["kind"])})

        logger.info("generate_paraphrases: %d variants for query=%.60s", len(clean_variants), query)

        return {
            "ok": True,
            "original_query": query,
            "variants": clean_variants,
        }

    except Exception as exc:
        logger.exception("generate_paraphrases: unexpected error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


# ─────────────────────────────────────────────────────────────────────────────
# Shared query-and-log helper (Option A — import directly, not MCP-to-MCP)
# ─────────────────────────────────────────────────────────────────────────────
async def _log_chat_call(message: str, web_search: bool) -> dict[str, Any]:
    """Fire /chat and log to query_log — same logic as Server 1's query_and_log tool."""
    called_at = datetime.now(timezone.utc).isoformat()
    result = await call_chat(message, web_search=web_search)

    department: str | None = None
    if result.sources:
        department = result.sources[0].get("department")

    conn = await get_db()
    try:
        await execute_with_retry(
            conn,
            """
            INSERT INTO query_log (
                called_at, user_id, chatroom_id, user_message_id, message_id,
                message, model_name, web_search, response_text, intent, language,
                route_hint, department, sources_json, web_sources_json,
                prompt_tokens, completion_tokens, total_tokens,
                latency_seconds, langfuse_trace_id, http_status,
                error_text, raw_response_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                called_at, config.MCP_DEFAULT_USER_ID, None,
                result.user_message_id, result.message_id,
                message, None, 1 if web_search else 0,
                result.response if result.ok else None,
                result.intent, result.language, result.route_hint, department,
                json.dumps(result.sources) if result.sources else None,
                json.dumps(result.web_sources) if result.web_sources else None,
                result.prompt_tokens, result.completion_tokens, result.total_tokens,
                result.latency_seconds, result.langfuse_trace_id,
                result.http_status,
                result.error if not result.ok else None,
                json.dumps(result.raw_response) if result.raw_response else None,
            ),
        )
    finally:
        await conn.close()

    top_gr: str | None = None
    top_fp: str | None = None
    if result.ok and result.sources:
        top_src = result.sources[0]
        top_gr = top_src.get("gr_number")
        top_fp = top_src.get("filepath")

    return {
        "ok": result.ok,
        "message_id": result.message_id,
        "top_gr_number": top_gr,
        "top_filepath": top_fp,
        "error": result.error,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tool: test_reformulation_stability (§7.2)
# ─────────────────────────────────────────────────────────────────────────────
@mcp.tool(
    name="test_reformulation_stability",
    description=(
        "Fire the original query + generated variants at /chat, compare top gr_number/filepath "
        "across variants (not answer text), compute stability_score, and persist to reformulation_runs."
    ),
)
async def test_reformulation_stability(
    query: str,
    variants: list | None = None,
    web_search: bool = True,
) -> dict[str, Any]:
    """§7.2 — Bounded concurrency (semaphore=3), Option A shared import, gr_number comparison."""
    try:
        if not variants:
            gen_result = await generate_paraphrases(query, n=4, include_marathi=False)
            if not gen_result.get("ok"):
                return {"ok": False, "error": f"Paraphrase generation failed: {gen_result.get('error')}", "http_status": None}
            variants = gen_result["variants"]

        all_calls: list[dict] = [{"text": query, "kind": "original"}] + list(variants)

        semaphore = asyncio.Semaphore(3)

        async def _fire(item: dict) -> dict:
            async with semaphore:
                log_result = await _log_chat_call(item["text"], web_search=web_search)
                return {
                    "kind": item["kind"],
                    "text": item["text"],
                    "message_id": log_result.get("message_id"),
                    "top_gr_number": log_result.get("top_gr_number"),
                    "top_filepath": log_result.get("top_filepath"),
                    "ok": log_result.get("ok"),
                    "error": log_result.get("error"),
                }

        logger.info("test_reformulation_stability: firing %d calls for query=%.60s", len(all_calls), query)
        results: list[dict] = await asyncio.gather(*[_fire(c) for c in all_calls])

        gr_numbers = [r["top_gr_number"] for r in results if r.get("top_gr_number")]
        majority_gr: str | None = None
        stability_score: float = 0.0

        if gr_numbers:
            counter = Counter(gr_numbers)
            majority_gr = counter.most_common(1)[0][0]
            matching = sum(1 for r in results if r.get("top_gr_number") == majority_gr)
            stability_score = matching / len(results)

        for r in results:
            r["odd_one_out"] = bool(
                majority_gr and r.get("top_gr_number") and r["top_gr_number"] != majority_gr
            )

        odd_one_out_kinds = [r["kind"] for r in results if r.get("odd_one_out")]
        run_at = datetime.now(timezone.utc).isoformat()

        conn = await get_db()
        try:
            run_cursor = await execute_with_retry(
                conn,
                "INSERT INTO reformulation_runs (run_at, original_query, stability_score, majority_gr, odd_one_out_ids) VALUES (?, ?, ?, ?, ?)",
                (run_at, query, round(stability_score, 4), majority_gr, json.dumps(odd_one_out_kinds)),
            )
            run_id = run_cursor.lastrowid

            for r in results:
                await execute_with_retry(
                    conn,
                    "INSERT INTO reformulation_variants (run_id, variant_text, variant_kind, message_id, top_gr_number, top_filepath) VALUES (?, ?, ?, ?, ?, ?)",
                    (run_id, r["text"], r["kind"], r.get("message_id"), r.get("top_gr_number"), r.get("top_filepath")),
                )
        finally:
            await conn.close()

        logger.info(
            "test_reformulation_stability: run_id=%d stability=%.2f majority_gr=%s",
            run_id, stability_score, majority_gr,
        )

        return {
            "ok": True,
            "original_query": query,
            "stability_score": round(stability_score, 4),
            "majority_gr_number": majority_gr,
            "run_id": run_id,
            "results": results,
        }

    except Exception as exc:
        logger.exception("test_reformulation_stability: unexpected error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MahaSamvaad Reformulation MCP Server")
    parser.add_argument("--transport", choices=["stdio", "streamable-http"], default=config.MCP_TRANSPORT)
    parser.add_argument("--host", default=config.MCP_HTTP_HOST)
    parser.add_argument("--port", type=int, default=config.MCP_HTTP_PORT)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.transport == "streamable-http":
        logger.info("Starting reformulation server on %s:%d (streamable-http)", args.host, args.port)
        mcp.run_streamable_http_async(host=args.host, port=args.port)
    else:
        logger.info("Starting reformulation server on stdio")
        mcp.run_stdio_async()


if __name__ == "__main__":
    main()
