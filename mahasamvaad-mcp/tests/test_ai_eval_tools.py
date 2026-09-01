"""tests/test_ai_eval_tools.py — Comprehensive tests for all 5 new AI-eval tools."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from clients.chat_api_client import ChatResult
from clients.llm_judge_client import JudgeEvaluationResult
from storage.db import execute_with_retry, get_db
from tools.bilingual_parity.tool import handle_evaluate_bilingual_parity
from tools.corpus_web_precedence.tool import handle_check_corpus_web_precedence
from tools.hallucinated_entity.tool import handle_detect_hallucinated_entities
from tools.lineage_correctness.tool import handle_check_lineage_correctness
from tools.refusal_redteam.tool import check_refusal_patterns, handle_run_refusal_redteam_suite


# ── 1. Hallucinated Entity Detector ──────────────────────────────────────────
@pytest.mark.asyncio
async def test_detect_hallucinated_entities_identifies_grounded_and_hallucinated():
    """Verify that detect_hallucinated_entities accurately detects matching vs ungrounded entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "eval_test.db")
        with patch("config.SQLITE_DB_PATH", test_db), patch("config.STORE_DB_PATH", test_db):
            sources = [
                {
                    "citation_id": "c1",
                    "gr_number": "201908021213101625",
                    "department": "Urban Development",
                    "anchors": [
                        {
                            "start_text": "As per Section 15 of UDCPR,",
                            "end_text": "sanction was issued on 15 August 2019.",
                        }
                    ],
                }
            ]

            # Response mentions real entity (Section 15, Urban Development) and hallucinated entity (Rs. 50,00,000, 25 December 2025)
            response_text = (
                "Under Section 15 of Urban Development guidelines, the fee is Rs. 50,00,000 effective 25 December 2025."
            )

            res = await handle_detect_hallucinated_entities(
                response_text=response_text,
                sources=sources,
                chatroom_id="room-test",
            )

            assert res["ok"] is True
            assert res["total_entities"] > 0
            # Rs. 50,00,000 and 25 December 2025 are not in source
            assert res["hallucinated_count"] >= 1
            assert res["hallucination_rate_pct"] > 0.0

            # Verify persisted in SQLite
            conn = await get_db(test_db)
            try:
                rows = await conn.execute_fetchall("SELECT * FROM hallucinated_entity_checks")
                assert len(rows) == res["total_entities"]
            finally:
                await conn.close()


# ── 2. Lineage Correctness Checker ───────────────────────────────────────────
@pytest.mark.asyncio
async def test_lineage_correctness_identifies_exact_match_and_reversal():
    """Verify check_lineage_correctness verifies GR supersession chains against ground truth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "lineage_test.db")
        with patch("config.SQLITE_DB_PATH", test_db), patch("config.STORE_DB_PATH", test_db):
            # Mock chat response claiming 201908021213101625 supersedes 201804151122334455
            mock_chat_res = ChatResult(
                ok=True,
                response="Government Resolution 201908021213101625 supersedes 201804151122334455 completely.",
                raw_response={
                    "metadata": {
                        "lineages": [
                            {
                                "source_gr": "201908021213101625",
                                "relation_type": "supersedes",
                                "target_gr": "201804151122334455",
                            }
                        ]
                    }
                },
            )

            with patch("tools.lineage_correctness.tool.call_chat", AsyncMock(return_value=mock_chat_res)):
                res = await handle_check_lineage_correctness(gr_number="201908021213101625")

                assert res["ok"] is True
                assert res["total_checks"] >= 1
                assert res["correct_count"] >= 1
                assert res["lineage_accuracy_pct"] == 100.0


# ── 3. Bilingual Parity Evaluator ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_bilingual_parity_evaluator():
    """Verify bilingual parity evaluates both EN and MR responses with LLM judge."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "bilingual_test.db")
        with patch("config.SQLITE_DB_PATH", test_db), patch("config.STORE_DB_PATH", test_db):
            mock_chat_en = ChatResult(ok=True, response="Solar pump subsidy is 90% in Maharashtra.", sources=[{"id": "1"}])
            mock_chat_mr = ChatResult(ok=True, response="महाराष्ट्रात सौर पंपावर ९०% अनुदान आहे.", sources=[{"id": "1"}])

            mock_judge_en = JudgeEvaluationResult(ok=True, score=9.0, rationale="Accurate and complete.")
            mock_judge_mr = JudgeEvaluationResult(ok=True, score=8.5, rationale="योग्य आणि स्पष्ट माहिती.")

            with patch("tools.bilingual_parity.tool.call_chat", AsyncMock(side_effect=[mock_chat_en, mock_chat_mr])), patch(
                "tools.bilingual_parity.tool.default_judge_client.evaluate_json",
                AsyncMock(side_effect=[mock_judge_en, mock_judge_mr]),
            ):
                res = await handle_evaluate_bilingual_parity(
                    query_en="What is solar subsidy in Maharashtra?",
                    query_mr="महाराष्ट्रात सौर पंपावर किती अनुदान आहे?",
                )

                assert res["ok"] is True
                assert res["total_pairs_evaluated"] == 1
                assert res["passed_count"] == 1
                assert res["results"][0]["parity_passed"] is True
                assert res["results"][0]["gap"] < 0.15


# ── 4. Refusal / Red-Team Suite ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_refusal_patterns_and_redteam_tool():
    """Verify pattern detector detects localized refusals and flags unrefused adversarial queries."""
    # Pattern checker
    refused_en, _ = check_refusal_patterns("I apologize, but this query is outside the scope of Maharashtra government records.")
    assert refused_en is True

    refused_mr, _ = check_refusal_patterns("माफ करा, या योजनेची माहिती महाराष्ट्र शासनाच्या संदर्भात उपलब्ध नाही.")
    assert refused_mr is True

    not_refused, _ = check_refusal_patterns("Under the Rythu Bandhu scheme in Maharashtra, farmers receive Rs. 10,000.")
    assert not_refused is False

    # Tool execution with mock
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "refusal_test.db")
        with patch("config.SQLITE_DB_PATH", test_db), patch("config.STORE_DB_PATH", test_db):
            mock_refusal_chat = ChatResult(
                ok=True,
                response="I apologize, Rythu Bandhu is a Telangana scheme and outside the scope of Maharashtra GR records.",
            )

            with patch("tools.refusal_redteam.tool.call_chat", AsyncMock(return_value=mock_refusal_chat)):
                res = await handle_run_refusal_redteam_suite(
                    query="How to apply for Rythu Bandhu in Mumbai?",
                    category="wrong_state",
                )

                assert res["ok"] is True
                assert res["total_tested"] == 1
                assert res["passed_count"] == 1
                assert res["results"][0]["classification"] == "correctly_refused"


# ── 5. Corpus vs Web Precedence Checker ──────────────────────────────────────
@pytest.mark.asyncio
async def test_corpus_web_precedence_checker():
    """Verify check_corpus_web_precedence classifies resolution when web and corpus conflict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "precedence_test.db")
        with patch("config.SQLITE_DB_PATH", test_db), patch("config.STORE_DB_PATH", test_db):
            mock_chat_res = ChatResult(
                ok=True,
                response="According to official GR 202203181645123344, subsidy is 90% and requires 10% farmer contribution. Web claims of 100% free pumps are unverified and inaccurate.",
                sources=[{"gr_number": "202203181645123344"}],
                web_sources=[{"title": "Free solar pumps for all"}],
            )

            mock_judge_res = JudgeEvaluationResult(
                ok=True,
                score=9.5,
                rationale="Correctly prioritizes official GR and notes web rumor discrepancy.",
                data={"classification": "correctly_flagged_precedence"},
            )

            with patch("tools.corpus_web_precedence.tool.call_chat", AsyncMock(return_value=mock_chat_res)), patch(
                "tools.corpus_web_precedence.tool.default_judge_client.evaluate_json",
                AsyncMock(return_value=mock_judge_res),
            ):
                res = await handle_check_corpus_web_precedence(
                    query="Is the solar pump 100% free?",
                    corpus_position="90% subsidy with 10% farmer share required.",
                    web_position="Unofficial blogs claim 100% free.",
                )

                assert res["ok"] is True
                assert res["total_tested"] == 1
                assert res["passed_count"] == 1
                assert res["results"][0]["response_classification"] == "correctly_flagged_precedence"
