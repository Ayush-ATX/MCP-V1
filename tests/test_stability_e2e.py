"""Focused stability test: generate paraphrases with the reformulation model + fire at /chat.

Run:
    python tests/test_stability_e2e.py

Requires:
    REFORMULATION_API_KEY  env var
    STORE_DB_PATH   env var (defaults ./data/store.db)
    Staging must be reachable at MAHASAMVAAD_BASE_URL
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mahasamvaad_eval.tools.reformulation import test_reformulation_stability


async def main():
    query = "What is the primary purpose of the Maharashtra Regional and Town Planning Act, 1966?"
    print(f"Testing stability for: {query[:70]}")
    print("Generating paraphrases with the reformulation model...")

    result = await test_reformulation_stability(query, web_search=True)

    if not result.get("ok"):
        print(f"[FAIL] {result.get('error')}")
        return

    print(f"\nstability_score = {result['stability_score']}")
    print(f"majority_gr     = {result['majority_gr_number']}")
    print(f"run_id          = {result['run_id']}")
    print()
    for r in result.get("results", []):
        oo = " <- ODD ONE OUT" if r.get("odd_one_out") else ""
        print(f"  [{r['kind']:10}] gr={r.get('top_gr_number')}{oo}")

    print("\n[OK] Stability test complete.")


if __name__ == "__main__":
    asyncio.run(main())
