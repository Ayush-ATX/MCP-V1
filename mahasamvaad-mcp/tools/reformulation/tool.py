"""tools/reformulation/tool.py — Query reformulation & retrieval stability tester."""
from __future__ import annotations

import asyncio
import json
import logging
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from mcp.server.mcpserver import MCPServer
from openai import OpenAI

import config
from clients.chat_api_client import call_chat
from storage.db import execute_with_retry, get_db

logger = logging.getLogger(__name__)


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


def _stream_reformulation_response(prompt: str) -> str:
    api_key = config.REFORMULATION_API_KEY or config.LLM_JUDGE_API_KEY
    if not api_key:
        raise ValueError("REFORMULATION_API_KEY is not set.")

    client = OpenAI(
        base_url=config.REFORMULATION_BASE_URL,
        api_key=api_key,
        timeout=config.MCP_HTTP_TIMEOUT_S,
    )
    extra_body = None
    if "nemotron" in config.REFORMULATION_MODEL:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    kwargs: dict[str, Any] = {
        "model": config.REFORMULATION_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "top_p": 0.95,
        "max_tokens": 2048,
        "stream": True,
    }
    if extra_body:
        kwargs["extra_body"] = extra_body

    stream = client.chat.completions.create(**kwargs)
    content_parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta is not None:
            content = getattr(delta, "content", None)
            if content is not None:
                content_parts.append(content)

    return "".join(content_parts)


async def _call_reformulation_model(prompt: str) -> str:
    return await asyncio.to_thread(_stream_reformulation_response, prompt)


async def handle_generate_paraphrases(
    query: str,
    n: int = 4,
    include_marathi: bool = False,
) -> dict[str, Any]:
    """Generate N paraphrases of a government scheme question."""
    try:
        n = max(1, min(n, 8))
        prompt = _build_paraphrase_prompt(query, n, include_marathi)
        raw_text = await _call_reformulation_model(prompt)

        cleaned = raw_text.strip()
        if "```" in cleaned:
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        try:
            variants = json.loads(cleaned)
            if not isinstance(variants, list):
                raise ValueError("Expected a JSON array")
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("generate_paraphrases: non-JSON output — %s", exc)
            return {"ok": False, "error": f"LLM returned invalid JSON: {exc}", "http_status": None}

        clean_variants = []
        for v in variants:
            if isinstance(v, dict) and "text" in v and "kind" in v:
                clean_variants.append({"text": str(v["text"]), "kind": str(v["kind"])})

        return {
            "ok": True,
            "original_query": query,
            "variants": clean_variants,
        }

    except Exception as exc:
        logger.exception("generate_paraphrases: error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


async def _log_chat_call(message: str, web_search: bool) -> dict[str, Any]:
    """Fire /chat and log to query_log."""
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


async def handle_test_paraphrase_robustness(
    query: str,
    variants: list | None = None,
    web_search: bool = True,
) -> dict[str, Any]:
    """Test retrieval stability across query paraphrases."""
    try:
        if not variants:
            gen_result = await handle_generate_paraphrases(query, n=4, include_marathi=False)
            if not gen_result.get("ok"):
                return {"ok": False, "error": f"Paraphrase generation failed: {gen_result.get('error')}", "http_status": None}
            variants = gen_result["variants"]

        normalized_variants: list[dict[str, Any]] = []
        for idx, v in enumerate(variants):
            if isinstance(v, str):
                normalized_variants.append({"text": v, "kind": f"variant_{idx + 1}"})
            elif isinstance(v, dict):
                normalized_variants.append({
                    "text": str(v.get("text", "")),
                    "kind": str(v.get("kind", f"variant_{idx + 1}")),
                })

        all_calls: list[dict] = [{"text": query, "kind": "original"}] + normalized_variants
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

        return {
            "ok": True,
            "original_query": query,
            "stability_score": round(stability_score, 4),
            "majority_gr_number": majority_gr,
            "run_id": run_id,
            "results": results,
        }

    except Exception as exc:
        logger.exception("test_paraphrase_robustness: error — %s", exc)
        return {"ok": False, "error": str(exc), "http_status": None}


def register(mcp: MCPServer) -> None:
    """Register Reformulation tools on MCP server."""
    # Namespaced tool: reformulation.generate_paraphrases
    mcp.tool(
        name="reformulation.generate_paraphrases",
        description="Generate N paraphrases of a question tagged formal | informal | reorder | marathi.",
    )(handle_generate_paraphrases)

    mcp.tool(
        name="generate_paraphrases",
        description="Backwards-compatible alias for generate_paraphrases.",
    )(handle_generate_paraphrases)

    # Namespaced tool: reformulation.test_paraphrase_robustness
    mcp.tool(
        name="reformulation.test_paraphrase_robustness",
        description="Fire query + variants at /chat, compare top GR numbers, compute stability score.",
    )(handle_test_paraphrase_robustness)

    mcp.tool(
        name="test_reformulation_stability",
        description="Backwards-compatible alias for test_paraphrase_robustness.",
    )(handle_test_paraphrase_robustness)
