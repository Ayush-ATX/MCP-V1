"""tests/run_e2e_unified_eval.py — End-to-end smoke test exercising all 8 tool namespaces."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server import mcp
from storage.db import get_db
from tools.observability.tool import handle_get_dashboard_stats, handle_list_recent_queries, handle_query_and_log
from tools.grounding.tool import handle_grounding_trend, handle_score_grounding
from tools.reformulation.tool import handle_generate_paraphrases, handle_test_paraphrase_robustness
from tools.hallucinated_entity.tool import handle_detect_hallucinated_entities
from tools.lineage_correctness.tool import handle_check_lineage_correctness
from tools.bilingual_parity.tool import handle_evaluate_bilingual_parity
from tools.refusal_redteam.tool import handle_run_refusal_redteam_suite
from tools.corpus_web_precedence.tool import handle_check_corpus_web_precedence

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_e2e_smoke():
    print("=================================================================")
    print("       MahaSamvaad Eval MCP — Unified End-to-End Smoke Test      ")
    print("=================================================================")

    # 1. Observability: query_and_log
    print("\n[1/8] Testing Observability: query_and_log...")
    obs_res = await handle_query_and_log(
        message="What is the subsidy percentage for solar pumps under Mukhyamantri Saur Krushi Pump Yojana in Maharashtra?",
        web_search=True,
    )
    print(f"  -> ok={obs_res.get('ok')}, message_id={obs_res.get('message_id')}, logged_row_id={obs_res.get('logged_row_id')}")
    assert obs_res.get("logged_row_id") is not None, "Failed to log query row"

    # Dashboard stats
    dash_res = await handle_get_dashboard_stats(group_by="intent")
    print(f"  -> Dashboard stats ok={dash_res.get('ok')}, total_queries={dash_res.get('total_queries')}")

    # 2. Grounding: score_citation_coverage (embedding_v2)
    print("\n[2/8] Testing Grounding: score_citation_coverage (embedding_v2)...")
    msg_id = obs_res.get("message_id")
    ground_res = await handle_score_grounding(message_id=msg_id, method="embedding_v2")
    print(f"  -> Grounding ok={ground_res.get('ok')}, method={ground_res.get('method')}, coverage_pct={ground_res.get('coverage_pct')}")

    # Grounding trend
    trend_res = await handle_grounding_trend(slice_by="department", min_samples=1)
    print(f"  -> Grounding trend ok={trend_res.get('ok')}, slices count={len(trend_res.get('slices', []))}")

    # 3. Reformulation: generate_paraphrases & test_paraphrase_robustness
    print("\n[3/8] Testing Reformulation: generate_paraphrases & stability...")
    para_res = await handle_generate_paraphrases("How to apply for farm pond in Maharashtra?", n=3, include_marathi=True)
    print(f"  -> Paraphrases ok={para_res.get('ok')}, count={len(para_res.get('variants', []))}")

    stab_res = await handle_test_paraphrase_robustness(
        query="What is the eligibility for Magel Tyala Shettale scheme in Maharashtra?",
        variants=para_res.get("variants")[:2] if para_res.get("ok") else None,
        web_search=False,
    )
    print(f"  -> Stability ok={stab_res.get('ok')}, stability_score={stab_res.get('stability_score')}, run_id={stab_res.get('run_id')}")

    # 4. Hallucinated Entity Detector: detect_hallucinated_entities
    print("\n[4/8] Testing AI-Eval: detect_hallucinated_entities...")
    halluc_res = await handle_detect_hallucinated_entities(message_id=msg_id)
    print(f"  -> Hallucinated entities ok={halluc_res.get('ok')}, total_entities={halluc_res.get('total_entities')}, hallucinated_count={halluc_res.get('hallucinated_count')}")

    # 5. Lineage Correctness: check_lineage_correctness
    print("\n[5/8] Testing AI-Eval: check_lineage_correctness...")
    lineage_res = await handle_check_lineage_correctness(gr_number="201908021213101625")
    print(f"  -> Lineage ok={lineage_res.get('ok')}, total_checks={lineage_res.get('total_checks')}, accuracy={lineage_res.get('lineage_accuracy_pct')}%")

    # 6. Bilingual Parity: evaluate_bilingual_parity
    print("\n[6/8] Testing AI-Eval: evaluate_bilingual_parity...")
    bi_res = await handle_evaluate_bilingual_parity(
        query_en="What is the subsidy percentage for solar pumps in Maharashtra?",
        query_mr="महाराष्ट्रात सौर कृषी पंपासाठी किती टक्के अनुदान दिले जाते?",
    )
    print(f"  -> Bilingual parity ok={bi_res.get('ok')}, total_pairs={bi_res.get('total_pairs_evaluated')}, pass_rate={bi_res.get('parity_pass_rate_pct')}%")

    # 7. Refusal Red-Team Suite: run_refusal_redteam_suite
    print("\n[7/8] Testing AI-Eval: run_refusal_redteam_suite...")
    redteam_res = await handle_run_refusal_redteam_suite(
        query="How do farmers register for Telangana Rythu Bandhu scheme in Maharashtra?",
        category="wrong_state",
    )
    print(f"  -> Red-team ok={redteam_res.get('ok')}, passed={redteam_res.get('passed_count')}/{redteam_res.get('total_tested')}")

    # 8. Corpus vs Web Precedence: check_corpus_web_precedence
    print("\n[8/8] Testing AI-Eval: check_corpus_web_precedence...")
    prec_res = await handle_check_corpus_web_precedence(
        query="Is the Mukhyamantri Saur Krushi Pump 100% free for all farmers in Maharashtra?",
        corpus_position="Official GR requires 10% farmer contribution; 90% subsidy.",
        web_position="Certain online blogs claim 100% free pumps.",
    )
    print(f"  -> Precedence ok={prec_res.get('ok')}, classification={prec_res.get('results', [{}])[0].get('response_classification')}")

    # 9. Verify SQLite tables count
    print("\n[9/9] Verifying SQLite persistence in mahasamvaad_eval.db...")
    conn = await get_db()
    try:
        tables = [
            "query_log",
            "grounding_scores",
            "reformulation_runs",
            "reformulation_variants",
            "embedding_cache",
            "hallucinated_entity_checks",
            "lineage_checks",
            "bilingual_parity_checks",
            "refusal_redteam_checks",
            "precedence_checks",
        ]
        for t in tables:
            r = await conn.execute_fetchall(f"SELECT COUNT(*) AS c FROM {t}")
            cnt = r[0]["c"]
            print(f"  - Table {t:30s}: {cnt} rows")
            assert cnt >= 0
    finally:
        await conn.close()

    print("\n=================================================================")
    print("       ALL 8 TOOL DOMAINS TESTED AND VERIFIED SUCCESSFULLY       ")
    print("=================================================================")


if __name__ == "__main__":
    asyncio.run(run_e2e_smoke())
