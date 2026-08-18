"""Unit tests for grounding score logic — hand-labelled sample pairs."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from server_grounding.main import DifflibV1Strategy, split_sentences

THRESHOLD = 0.55

# ── Sample data ───────────────────────────────────────────────────────────────
ANCHOR_TEXTS = [
    "The Maharashtra Regional and Town Planning Act 1966 governs the preparation of development plans.",
    "A development plan shall be prepared for every planning area.",
    "The State Government may constitute a Planning Authority for the purpose.",
]

# Fully grounded response — every sentence appears verbatim or near-verbatim in anchors
GROUNDED_RESPONSE = (
    "The Maharashtra Regional and Town Planning Act 1966 governs the preparation of development plans. "
    "A development plan shall be prepared for every planning area. "
    "The State Government may constitute a Planning Authority for the purpose."
)

# Partially fabricated — two extra invented sentences
PARTIAL_RESPONSE = (
    "The Maharashtra Regional and Town Planning Act 1966 governs the preparation of development plans. "
    "The Act also requires all buildings to be painted white. "
    "Every citizen must register their property with the planning authority every year."
)


def _score_response(response: str, anchors: list[str]) -> dict:
    strategy = DifflibV1Strategy()
    sentences = split_sentences(response)
    grounded = 0
    ungrounded = []
    for s in sentences:
        best = strategy.score(s, anchors)
        if best >= THRESHOLD:
            grounded += 1
        else:
            ungrounded.append({"sentence": s, "best_match_score": best})
    coverage = grounded / len(sentences) * 100 if sentences else 0
    return {"coverage_pct": coverage, "grounded": grounded, "total": len(sentences), "ungrounded": ungrounded}


def test_fully_grounded_response():
    result = _score_response(GROUNDED_RESPONSE, ANCHOR_TEXTS)
    assert result["coverage_pct"] == 100.0, f"Expected 100%, got {result['coverage_pct']}"
    assert result["grounded"] == result["total"]


def test_partially_fabricated_response():
    result = _score_response(PARTIAL_RESPONSE, ANCHOR_TEXTS)
    # At least one sentence should be grounded, at least one ungrounded
    assert 0 < result["coverage_pct"] < 100.0, f"Expected partial coverage, got {result['coverage_pct']}"
    assert len(result["ungrounded"]) >= 1


def test_empty_anchors_returns_zero():
    strategy = DifflibV1Strategy()
    score = strategy.score("This is a sentence.", [])
    assert score == 0.0


def test_threshold_boundary():
    """A near-exact match should exceed threshold; a very different string should not."""
    strategy = DifflibV1Strategy()
    # Near-exact — should exceed 0.55
    high = strategy.score(
        "Maharashtra Regional and Town Planning Act governs development plans",
        ["The Maharashtra Regional and Town Planning Act 1966 governs the preparation of development plans."],
    )
    assert high >= THRESHOLD, f"Expected >= {THRESHOLD}, got {high}"

    # Completely different — should be well below 0.55
    low = strategy.score(
        "The cat sat on the mat with a hat",
        ["The Maharashtra Regional and Town Planning Act 1966 governs the preparation of development plans."],
    )
    assert low < THRESHOLD, f"Expected < {THRESHOLD}, got {low}"


def test_no_sources_coverage_is_null():
    """When sources is empty, the algorithm should short-circuit (tested at integration level)."""
    # At unit level, verify that scoring against empty anchors gives 0 per sentence
    strategy = DifflibV1Strategy()
    sentences = split_sentences("This is a sentence. And another.")
    for s in sentences:
        score = strategy.score(s, [])
        assert score == 0.0
