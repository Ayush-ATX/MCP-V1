"""Unit tests for stability-score arithmetic in test_reformulation_stability."""
from __future__ import annotations

from collections import Counter


def _compute_stability(results: list[dict]) -> tuple[float, str | None, list[str]]:
    """Replicate the stability score computation from Server 3."""
    gr_numbers = [r["top_gr_number"] for r in results if r.get("top_gr_number")]
    majority_gr = None
    stability_score = 0.0
    odd_one_out = []

    if gr_numbers:
        counter = Counter(gr_numbers)
        majority_gr = counter.most_common(1)[0][0]
        matching = sum(1 for r in results if r.get("top_gr_number") == majority_gr)
        stability_score = matching / len(results)
        odd_one_out = [r["kind"] for r in results if r.get("top_gr_number") and r["top_gr_number"] != majority_gr]

    return stability_score, majority_gr, odd_one_out


def test_all_agree():
    results = [
        {"kind": "original", "top_gr_number": "GR-001"},
        {"kind": "formal", "top_gr_number": "GR-001"},
        {"kind": "informal", "top_gr_number": "GR-001"},
        {"kind": "reorder", "top_gr_number": "GR-001"},
    ]
    score, majority, odd = _compute_stability(results)
    assert score == 1.0
    assert majority == "GR-001"
    assert odd == []


def test_one_disagrees():
    results = [
        {"kind": "original", "top_gr_number": "GR-001"},
        {"kind": "formal", "top_gr_number": "GR-001"},
        {"kind": "informal", "top_gr_number": "GR-001"},
        {"kind": "reorder", "top_gr_number": "GR-999"},  # odd one out
    ]
    score, majority, odd = _compute_stability(results)
    assert abs(score - 0.75) < 1e-9, f"Expected 0.75, got {score}"
    assert majority == "GR-001"
    assert "reorder" in odd


def test_all_disagree():
    results = [
        {"kind": "original", "top_gr_number": "GR-001"},
        {"kind": "formal", "top_gr_number": "GR-002"},
        {"kind": "informal", "top_gr_number": "GR-003"},
        {"kind": "reorder", "top_gr_number": "GR-004"},
    ]
    score, majority, odd = _compute_stability(results)
    # All different — majority is whichever appears first in most_common, score = 1/4
    assert abs(score - 0.25) < 1e-9, f"Expected 0.25, got {score}"


def test_null_gr_not_counted_as_match():
    """A null top_gr_number should not count as matching the majority."""
    results = [
        {"kind": "original", "top_gr_number": "GR-001"},
        {"kind": "formal", "top_gr_number": "GR-001"},
        {"kind": "informal", "top_gr_number": None},  # failed call
        {"kind": "reorder", "top_gr_number": None},
    ]
    score, majority, odd = _compute_stability(results)
    # 2 match GR-001 out of 4 total
    assert abs(score - 0.5) < 1e-9, f"Expected 0.5, got {score}"
    assert majority == "GR-001"


def test_all_null_gr():
    """If no GR numbers at all, stability_score=0 and majority_gr=None."""
    results = [
        {"kind": "original", "top_gr_number": None},
        {"kind": "formal", "top_gr_number": None},
    ]
    score, majority, odd = _compute_stability(results)
    assert score == 0.0
    assert majority is None
    assert odd == []
