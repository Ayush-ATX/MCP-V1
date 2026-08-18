"""
End-to-end demo script: runs all 15 curated questions through all three servers.

Usage:
    python tests/run_e2e_demo.py

Output:
    demo_output.json  - structured results for the intern report
    demo_output.txt   - human-readable summary

Environment variables required:
    NVIDIA_API_KEY  - for Server 3 paraphrase generation (NVIDIA NIM)
    STORE_DB_PATH   - defaults to ./data/store.db
"""
from __future__ import annotations
import sys, io
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import asyncio
import json
import os
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mcp_common import config
from mcp_common.chat_client import call_chat
from mcp_common.store import execute_with_retry, get_db

from server_grounding.main import DifflibV1Strategy, split_sentences
from server_reformulation.main import test_reformulation_stability

QUESTIONS = [
    "What is the primary purpose of the Maharashtra Regional and Town Planning Act, 1966?",
    "Who can file an RTI application",
    "What rules govern development in agricultural zones near functional railway stations?",
    "What is the premium rate for changing from an Agricultural or No-Development zone to a 'Commercial' zone?",
    "To whom should an application for Zone Change under Section 20(4) of the MRTP Act be submitted?",
    "\u092e\u0948\u0902\u0928\u0947 \u090f\u0915 \u0911\u0928\u0932\u093e\u0907\u0928 \u0906\u0935\u0947\u0926\u0928 \u0926\u093e\u092f\u0930 \u0915\u093f\u092f\u093e \u0939\u0948; \u092e\u0941\u091d\u0947 \u0938\u0942\u091a\u0928\u093e \u0905\u0927\u093f\u0915\u093e\u0930\u0940 \u0938\u0947 \u0915\u093f\u0924\u0928\u0947 \u0926\u093f\u0928\u094b\u0902 \u092e\u0947\u0902 \u091c\u093e\u0928\u0915\u093e\u0930\u0940 \u092e\u093f\u0932 \u091c\u093e\u0928\u0940 \u091a\u093e\u0939\u093f\u090f?",
    "\u0906\u0930\u091f\u0940\u0906\u0908 \u0905\u0927\u093f\u0928\u093f\u092f\u092e \u0915\u0947 \u0924\u0939\u0924 \u0924\u0943\u0924\u0940\u092f \u092a\u0915\u094d\u0937 \u0915\u094d\u092f\u093e \u0939\u0948?",
    "RTS 2015 \u0905\u0927\u093f\u0928\u093f\u092f\u092e \u0915\u092c \u092a\u093e\u0930\u093f\u0924 \u0915\u093f\u092f\u093e \u0917\u092f\u093e \u0925\u093e?",
    "\u092f\u0926\u093f \u092e\u0941\u091d\u0947 \u092a\u094d\u0930\u0925\u092e \u0905\u092a\u0940\u0932\u0940\u092f \u092a\u094d\u0930\u093e\u0927\u093f\u0915\u093e\u0930\u0940 \u0926\u094d\u0935\u093e\u0930\u093e \u0926\u093f\u090f \u0917\u090f \u0928\u093f\u0930\u094d\u0927\u093e\u0930\u093f\u0924 \u0938\u092e\u092f \u092e\u0947\u0902 \u0938\u093e\u0930\u094d\u0935\u091c\u0928\u093f\u0915 \u0938\u0947\u0935\u093e \u092a\u094d\u0930\u093e\u092a\u094d\u0924 \u0928\u0939\u0940\u0902 \u0939\u094b\u0924\u0940 \u0939\u0948, \u0924\u094b \u0915\u094d\u092f\u093e \u0939\u094b\u0917\u093e?",
    "\u092e\u0939\u093e\u0930\u093e\u0937\u094d\u091f\u094d\u0930 \u092e\u0947\u0902 \u0915\u093f\u0938\u0940 \u0915\u094d\u0937\u0947\u0924\u094d\u0930 \u0915\u0947 \u0932\u093f\u090f \u0938\u0902\u0930\u091a\u0928\u093e \u092f\u094b\u091c\u0928\u093e (Structure Plan) \u0924\u0948\u092f\u093e\u0930 \u0915\u0930\u0928\u0947 \u0915\u0947 \u0932\u093f\u090f \u0915\u094c\u0928 \u091c\u093f\u092e\u094d\u092e\u0947\u0926\u093e\u0930 \u0939\u0948?",
    "\u090f\u092e\u0906\u0930\u091f\u0940\u092a\u0940 \u0915\u093e\u092f\u0926\u094d\u092f\u093e\u0924 \u092a\u0930\u093f\u092d\u093e\u0937\u093f\u0924 \u0915\u0947\u0932\u094d\u092f\u093e\u0928\u0941\u0938\u093e\u0930, \u0935\u093f\u0915\u093e\u0938 \u0906\u0930\u093e\u0916\u0921\u094d\u092f\u093e\u0924\u0940\u0932 '\u092e\u0939\u0924\u094d\u0924\u094d\u0935\u092a\u0942\u0930\u094d\u0923 \u0938\u094d\u0935\u0930\u0942\u092a\u093e\u0924\u0940\u0932 \u0938\u0941\u0927\u093e\u0930\u0923\u093e' \u092e\u094d\u0939\u0923\u091c\u0947 \u0915\u093e\u092f?",
    "\u0935\u093f\u0915\u093e\u0938 \u0906\u0930\u093e\u0916\u0921\u094d\u092f\u093e\u0924 \u0930\u093e\u0916\u0940\u0935 \u0920\u0947\u0935\u0932\u0947\u0932\u093e \u0916\u0947\u0933\u093e\u091a\u093e \u092e\u0948\u0926\u093e\u0928 \u0915\u094b\u0923\u0924\u094d\u092f\u093e \u092a\u0930\u093f\u0938\u094d\u0925\u093f\u0924\u0940\u0924 \u0907\u0924\u0930 \u0915\u093e\u092e\u093e\u0902\u0938\u093e\u0920\u0940 \u0924\u093e\u0924\u094d\u092a\u0941\u0930\u0924\u0947 \u0935\u093e\u092a\u0930\u0932\u0947 \u091c\u093e\u090a \u0936\u0915\u0924\u0947?",
    "\u0938\u094d\u0935\u091a\u094d\u091b \u092d\u093e\u0930\u0924 \u092e\u093f\u0936\u0928-\u0936\u0939\u0930\u0940 \u092e\u094d\u0939\u0923\u091c\u0947 \u0915\u093e\u092f \u0906\u0923\u093f \u0924\u0947 \u0936\u0939\u0930\u093e\u0902\u092e\u0927\u0940\u0932 \u0915\u091a\u0930\u093e \u0906\u0923\u093f \u091f\u093e\u0915\u093e\u090a \u092a\u0926\u093e\u0930\u094d\u0925 \u0915\u0938\u093e \u0938\u093e\u092b \u0915\u0930\u0924\u0947?",
    "\u0905\u092e\u0943\u0924 \u0968.\u0966 \u092e\u093e\u091d\u094d\u092f\u093e \u0918\u0930\u093e\u0932\u093e \u0938\u094d\u0935\u091a\u094d\u091b \u092a\u093e\u0923\u0940 \u092e\u093f\u0933\u0935\u0942\u0928 \u0926\u0947\u0923\u094d\u092f\u093e\u0924 \u0915\u0936\u0940 \u092e\u0926\u0924 \u0915\u0930\u0947\u0932 \u0906\u0923\u093f \u0915\u094b\u0923\u0924\u094d\u092f\u093e \u0915\u094d\u0937\u0947\u0924\u094d\u0930\u093e\u0902\u0928\u093e \u092a\u094d\u0930\u093e\u0927\u093e\u0928\u094d\u092f \u0926\u093f\u0932\u0947 \u091c\u093e\u0924 \u0906\u0939\u0947?",
    "\u091c\u0930 \u0915\u094b\u0923\u0940 \u092c\u0947\u0915\u093e\u092f\u0926\u0947\u0936\u0940\u0930\u092a\u0923\u0947 \u091d\u093e\u0921 \u0924\u094b\u0921\u0932\u0947 \u0915\u093f\u0902\u0935\u093e \u092f\u093e \u0915\u093e\u092f\u0926\u094d\u092f\u093e\u091a\u094d\u092f\u093e \u0928\u093f\u092f\u092e\u093e\u0902\u091a\u0947 \u0909\u0932\u094d\u0932\u0902\u0918\u0928 \u0915\u0947\u0932\u0947 \u0924\u0930 \u0915\u093e\u092f \u0939\u094b\u0908\u0932?",
]


async def _log_and_score(message: str, web_search: bool = True) -> dict:
    """Fire /chat, log to query_log, score grounding. Returns combined result."""
    called_at = datetime.now(timezone.utc).isoformat()
    result = await call_chat(message, web_search=web_search)

    department = None
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

    # Score grounding
    grounding_result: dict = {"coverage_pct": None, "reason": "skipped"}
    if result.ok and result.sources:
        strategy = DifflibV1Strategy()
        anchor_texts = []
        for src in result.sources:
            for anchor in (src.get("anchors") or []):
                combined = f"{anchor.get('start_text', '')} {anchor.get('end_text', '')}".strip()
                if combined:
                    anchor_texts.append(combined)

        sentences = split_sentences(result.response or "")
        if sentences and anchor_texts:
            grounded = sum(
                1 for s in sentences
                if strategy.score(s, anchor_texts) >= config.GROUNDING_THRESHOLD
            )
            coverage_pct = grounded / len(sentences) * 100
            grounding_result = {
                "coverage_pct": round(coverage_pct, 2),
                "total_sentences": len(sentences),
                "grounded_count": grounded,
            }

            scored_at = datetime.now(timezone.utc).isoformat()
            conn2 = await get_db()
            try:
                await execute_with_retry(
                    conn2,
                    "INSERT INTO grounding_scores (message_id, scored_at, coverage_pct, total_sentences, grounded_count, ungrounded_json, method) VALUES (?, ?, ?, ?, ?, '[]', 'difflib_v1')",
                    (result.message_id, scored_at, round(coverage_pct, 2), len(sentences), grounded),
                )
            finally:
                await conn2.close()
        elif not anchor_texts:
            grounding_result = {"coverage_pct": None, "reason": "no_sources_to_ground_against"}

    return {
        "question": message,
        "ok": result.ok,
        "message_id": result.message_id,
        "intent": result.intent,
        "language": result.language,
        "department": department,
        "latency_seconds": result.latency_seconds,
        "total_tokens": result.total_tokens,
        "top_gr": result.sources[0].get("gr_number") if result.sources else None,
        "source_count": len(result.sources) if result.sources else 0,
        "grounding": grounding_result,
        "error": result.error if not result.ok else None,
    }


async def main():
    print("=" * 70)
    print("MahaSamvaad MCP Suite -- End-to-End Demo (15 questions)")
    print("=" * 70)
    print(f"Staging: {config.MAHASAMVAAD_BASE_URL}")
    print(f"Store:   {config.STORE_DB_PATH}")
    print()

    import httpx
    try:
        resp = httpx.get(f"{config.MAHASAMVAAD_BASE_URL}/health", timeout=10)
        print(f"[OK] Health check: {resp.status_code} {resp.text}")
    except Exception as e:
        print(f"[FAIL] Health check FAILED: {e}")
        return

    # Ensure DB is migrated
    init_conn = await get_db()
    await init_conn.close()
    print("[OK] Database schema migrated")
    print()

    results = []
    stability_results = []

    for i, q in enumerate(QUESTIONS, 1):
        q_display = q[:70] if len(q) <= 70 else q[:67] + "..."
        print(f"[{i:02d}/{len(QUESTIONS)}] {q_display}")
        try:
            r = await _log_and_score(q)
            results.append(r)
            status = "[OK]" if r["ok"] else "[FAIL]"
            grounding = r["grounding"].get("coverage_pct")
            grounding_str = f"{grounding:.1f}%" if grounding is not None else "N/A"
            lat = r.get("latency_seconds") or 0
            print(
                f"  {status} intent={r['intent']} dept={r['department']} "
                f"lat={lat:.2f}s grounding={grounding_str} top_gr={r['top_gr']}"
            )
        except Exception as exc:
            print(f"  [FAIL] ERROR: {exc}")
            results.append({"question": q, "ok": False, "error": str(exc)})

    print()
    print("-" * 70)
    print("Server 3: Reformulation stability on selected questions")
    print("-" * 70)

    stability_questions = [
        "What is the primary purpose of the Maharashtra Regional and Town Planning Act, 1966?",
        "To whom should an application for Zone Change under Section 20(4) of the MRTP Act be submitted?",
    ]

    for q in stability_questions:
        print(f"\nStability test: {q[:70]}")
        try:
            stab_result = await test_reformulation_stability(q, web_search=True)
            stability_results.append(stab_result)
            print(f"  stability_score={stab_result.get('stability_score')} majority_gr={stab_result.get('majority_gr_number')}")
            for r in stab_result.get("results", []):
                oo = " <- ODD ONE OUT" if r.get("odd_one_out") else ""
                print(f"  [{r['kind']:10}] gr={r.get('top_gr_number')}{oo}")
        except Exception as exc:
            print(f"  [FAIL] ERROR: {exc}")
            stability_results.append({"question": q, "ok": False, "error": str(exc)})

    total = len(results)
    ok_count = sum(1 for r in results if r.get("ok"))
    avg_latency = sum(r.get("latency_seconds") or 0 for r in results if r.get("latency_seconds")) / max(ok_count, 1)
    avg_grounding = [r["grounding"]["coverage_pct"] for r in results if r.get("grounding", {}).get("coverage_pct") is not None]

    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Questions processed: {total}")
    print(f"Successful:          {ok_count}/{total}")
    print(f"Average latency:     {avg_latency:.2f}s")
    if avg_grounding:
        print(f"Average grounding:   {sum(avg_grounding)/len(avg_grounding):.1f}%")
    print(f"Stability tests:     {len(stability_results)}")

    output = {
        "demo_run_at": datetime.now(timezone.utc).isoformat(),
        "staging_url": config.MAHASAMVAAD_BASE_URL,
        "total_questions": total,
        "successful": ok_count,
        "avg_latency_seconds": round(avg_latency, 2),
        "avg_grounding_pct": round(sum(avg_grounding) / len(avg_grounding), 2) if avg_grounding else None,
        "question_results": results,
        "stability_results": stability_results,
    }

    with open("demo_output.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\n[OK] Results saved to demo_output.json")

    with open("demo_output.txt", "w", encoding="utf-8") as f:
        f.write("MahaSamvaad MCP Suite -- E2E Demo Results\n")
        f.write(f"Run at: {output['demo_run_at']}\n\n")
        for i, r in enumerate(results, 1):
            f.write(f"[{i:02d}] {r['question'][:80]}\n")
            f.write(f"     ok={r.get('ok')} intent={r.get('intent')} dept={r.get('department')} ")
            f.write(f"lat={r.get('latency_seconds', 'N/A')}s grounding={r.get('grounding', {}).get('coverage_pct')}%\n")
            f.write(f"     top_gr={r.get('top_gr')} message_id={r.get('message_id')}\n\n")
        f.write("\nStability Tests\n")
        f.write("=" * 70 + "\n")
        for sr in stability_results:
            if sr.get("ok"):
                f.write(f"Query: {sr.get('original_query', '')[:80]}\n")
                f.write(f"  stability_score={sr.get('stability_score')} majority_gr={sr.get('majority_gr_number')}\n")
                for r in sr.get("results", []):
                    oo = " <- ODD ONE OUT" if r.get("odd_one_out") else ""
                    f.write(f"  [{r['kind']:10}] gr={r.get('top_gr_number')}{oo}\n")
                f.write("\n")

    print("[OK] Human-readable summary saved to demo_output.txt")


if __name__ == "__main__":
    asyncio.run(main())
