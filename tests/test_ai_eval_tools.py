"""tests/test_ai_eval_tools.py — Comprehensive tests for all 5 new AI-eval tools."""
from __future__ import annotations

import json
import os
import sys
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mahasamvaad_eval.clients.chat_api_client import ChatResult
from mahasamvaad_eval.clients.llm_judge_client import JudgeEvaluationResult
from mahasamvaad_eval.storage.db import execute_with_retry, get_db
from mahasamvaad_eval.tools.bilingual_parity import handle_evaluate_bilingual_parity
from mahasamvaad_eval.tools.corpus_web_precedence import handle_check_corpus_web_precedence
from mahasamvaad_eval.tools.hallucinated_entity import handle_detect_hallucinated_entities
from mahasamvaad_eval.tools.lineage_correctness import handle_check_lineage_correctness
from mahasamvaad_eval.tools.refusal_redteam import check_refusal_patterns, handle_run_refusal_redteam_suite


# ── 1. Hallucinated Entity Detector Tests (Scenarios 1–10) ───────────────────
@pytest.mark.asyncio
async def test_detect_hallucinated_entities_identifies_grounded_and_hallucinated():
    """Verify that detect_hallucinated_entities accurately detects matching vs ungrounded entities."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "eval_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            sources = [
                {
                    "citation_id": 1,
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
                "Under Section 15 of Urban Development guidelines [1], the fee is Rs. 50,00,000 effective 25 December 2025."
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


@pytest.mark.asyncio
async def test_hallucination_scenarios_1_to_10():
    """Verify Scenarios 1 to 10 for strict extraction and citation-aware source matching."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "scenarios_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            sample_sources = [
                {
                    "citation_id": 14,
                    "gr_number": "202602101741321319",
                    "date": "2026-02-10",
                    "department": "Revenue",
                    "anchors": [
                        {
                            "start_text": "As per Section 52(1) of MRTP Act",
                            "end_text": "sanction was issued on 10 February 2026",
                        }
                    ],
                },
                {
                    "citation_id": 5,
                    "gr_number": "202401151200001111",
                    "date": "2024-01-15",
                    "department": "Agriculture",
                    "anchors": [
                        {
                            "start_text": "Scheme guidelines for agriculture",
                            "end_text": "approved on 15 January 2024",
                        }
                    ],
                },
                {
                    "citation_id": 15,
                    "gr_number": "एनएपी-२०२५/प्र.क्र.१७७/जमीन-०१अ",
                    "date": "2025-08-15",
                    "department": "महसूल विभाग",
                    "anchors": [
                        {
                            "start_text": "कलम १५ नुसार सुधारणा",
                            "end_text": "१५ ऑगस्ट २०२५ रोजी शासन निर्णय जारी",
                        }
                    ],
                },
            ]

            # Test 1: Valid GR number -> should be found
            res1 = await handle_detect_hallucinated_entities(
                response_text="GR 202602101741321319 was officially approved [14].",
                sources=sample_sources,
            )
            gr_ent1 = [e for e in res1["entities"] if e["entity_type"] == "gr_number"]
            assert len(gr_ent1) == 1
            assert gr_ent1[0]["found_in_source"] is True

            # Test 2: Fabricated GR number -> should be hallucinated
            res2 = await handle_detect_hallucinated_entities(
                response_text="GR 209901019999999999 was officially approved [14].",
                sources=sample_sources,
            )
            gr_ent2 = [e for e in res2["entities"] if e["entity_type"] == "gr_number"]
            assert len(gr_ent2) == 1
            assert gr_ent2[0]["found_in_source"] is False

            # Test 3: Correct date -> should be found
            res3 = await handle_detect_hallucinated_entities(
                response_text="The decision was published on 10 February 2026 [14].",
                sources=sample_sources,
            )
            date_ent3 = [e for e in res3["entities"] if e["entity_type"] == "date"]
            assert len(date_ent3) == 1
            assert date_ent3[0]["found_in_source"] is True

            # Test 4: Incorrect date for the cited GR -> should be hallucinated
            res4 = await handle_detect_hallucinated_entities(
                response_text="The decision was published on 15 March 2024 [14].",
                sources=sample_sources,
            )
            date_ent4 = [e for e in res4["entities"] if e["entity_type"] == "date"]
            assert len(date_ent4) == 1
            assert date_ent4[0]["found_in_source"] is False
            assert date_ent4[0]["expected_value"] == "2026-02-10"

            # Test 5: Valid section number -> should be found
            res5 = await handle_detect_hallucinated_entities(
                response_text="Under Section 52(1), guidelines apply [14].",
                sources=sample_sources,
            )
            sec_ent5 = [e for e in res5["entities"] if e["entity_type"] == "section"]
            assert len(sec_ent5) == 1
            assert sec_ent5[0]["found_in_source"] is True

            # Test 6: Fabricated section number -> should be hallucinated
            res6 = await handle_detect_hallucinated_entities(
                response_text="Under Section 999(B), guidelines apply [14].",
                sources=sample_sources,
            )
            sec_ent6 = [e for e in res6["entities"] if e["entity_type"] == "section"]
            assert len(sec_ent6) == 1
            assert sec_ent6[0]["found_in_source"] is False

            # Test 7: Normal words such as 'growth' and 'agricultural' -> must NOT produce GR-number entities
            res7 = await handle_detect_hallucinated_entities(
                response_text="Economic growth in agricultural land development is crucial [14].",
                sources=sample_sources,
            )
            gr_ent7 = [e for e in res7["entities"] if e["entity_type"] == "gr_number"]
            assert len(gr_ent7) == 0, f"Expected 0 GR entities, got {gr_ent7}"

            # Test 8: Factual entity exists in source 14 but response cites source 5 -> verify citation/source handling
            res8 = await handle_detect_hallucinated_entities(
                response_text="Under Section 52(1), guidelines apply [5].",
                sources=sample_sources,
            )
            sec_ent8 = [e for e in res8["entities"] if e["entity_type"] == "section"]
            assert len(sec_ent8) == 1
            # Section 52(1) exists in source 14, but response cited source 5 where it does not exist
            assert sec_ent8[0]["found_in_source"] is False

            # Test 9: Multiple citations [14][15] -> correctly evaluate against the cited sources
            res9 = await handle_detect_hallucinated_entities(
                response_text="Under Section 52(1) and कलम १५, notifications were issued [14][15].",
                sources=sample_sources,
            )
            sec_ent9 = [e for e in res9["entities"] if e["entity_type"] == "section"]
            assert len(sec_ent9) == 2
            assert all(e["found_in_source"] is True for e in sec_ent9)

            # Test 10: Marathi GR number and date -> verify extraction and matching
            res10 = await handle_detect_hallucinated_entities(
                response_text="शासन निर्णय क्र. एनएपी-२०२५/प्र.क्र.१७७/जमीन-०१अ दिनांक १५ ऑगस्ट २०२५ रोजी जारी करण्यात आला [15].",
                sources=sample_sources,
            )
            gr_ent10 = [e for e in res10["entities"] if e["entity_type"] == "gr_number"]
            date_ent10 = [e for e in res10["entities"] if e["entity_type"] == "date"]
            assert len(gr_ent10) == 1
            assert gr_ent10[0]["found_in_source"] is True
            assert len(date_ent10) == 1
            assert date_ent10[0]["found_in_source"] is True


# ── 2. Lineage Correctness Checker ───────────────────────────────────────────
@pytest.mark.asyncio
async def test_lineage_correctness_identifies_exact_match_and_reversal():
    """Verify check_lineage_correctness verifies GR supersession chains against ground truth."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "lineage_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            # Case 1: Exact match
            mock_chat_exact = ChatResult(
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

            with patch("mahasamvaad_eval.tools.lineage_correctness.call_chat", AsyncMock(return_value=mock_chat_exact)):
                res = await handle_check_lineage_correctness(gr_number="201908021213101625")
                assert res["ok"] is True
                assert res["total_checks"] == 1
                assert res["correct_count"] == 1
                assert res["lineage_accuracy_pct"] == 100.0
                assert res["eval_results"][0]["classification"] == "exact_match"

            # Case 2: Reversed direction (dangerous error)
            mock_chat_rev = ChatResult(
                ok=True,
                response="Resolution 201804151122334455 supersedes 201908021213101625.",
            )
            with patch("mahasamvaad_eval.tools.lineage_correctness.call_chat", AsyncMock(return_value=mock_chat_rev)):
                res_rev = await handle_check_lineage_correctness(gr_number="201908021213101625")
                assert res_rev["ok"] is True
                assert res_rev["correct_count"] == 0
                assert res_rev["eval_results"][0]["classification"] == "reversed_direction"
                assert res_rev["eval_results"][0]["correct"] is False

            # Case 3: Wrong relation type
            mock_chat_wrong = ChatResult(
                ok=True,
                response="Resolution 201908021213101625 amends 201804151122334455.",
            )
            with patch("mahasamvaad_eval.tools.lineage_correctness.call_chat", AsyncMock(return_value=mock_chat_wrong)):
                res_wrong = await handle_check_lineage_correctness(gr_number="201908021213101625")
                assert res_wrong["ok"] is True
                assert res_wrong["correct_count"] == 0
                assert res_wrong["eval_results"][0]["classification"] == "wrong_relation_type"


# ── 3. Bilingual Parity Evaluator ────────────────────────────────────────────
@pytest.mark.asyncio
async def test_bilingual_parity_evaluator():
    """Verify bilingual parity evaluates both EN and MR responses with LLM judge and concise source output."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "bilingual_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            mock_chat_en = ChatResult(
                ok=True,
                response="Solar pump subsidy is 90% in Maharashtra.",
                sources=[{
                    "gr_number": "GR-101",
                    "filename": "solar.pdf",
                    "anchors": [{"start_text": "90% subsidy", "end_text": "for solar"}],
                    "anchor_id": "anc-1",
                }],
            )
            mock_chat_mr = ChatResult(
                ok=True,
                response="महाराष्ट्रात सौर पंपावर ९०% अनुदान आहे.",
                sources=[{
                    "gr_number": "GR-101",
                    "filename": "solar.pdf",
                    "anchors": [{"start_text": "९०% अनुदान", "end_text": "सौर पंप"}],
                    "anchor_id": "anc-2",
                }],
            )

            mock_judge_en = JudgeEvaluationResult(ok=True, score=9.0, rationale="Accurate and complete.")
            mock_judge_mr = JudgeEvaluationResult(ok=True, score=8.5, rationale="योग्य आणि स्पष्ट माहिती.")

            with patch("mahasamvaad_eval.tools.bilingual_parity.call_chat", AsyncMock(side_effect=[mock_chat_en, mock_chat_mr])), patch(
                "mahasamvaad_eval.tools.bilingual_parity.default_judge_client.evaluate_json",
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
                assert res["results"][0]["rationale_en"] == "Accurate and complete."
                assert res["results"][0]["rationale_mr"] == "योग्य आणि स्पष्ट माहिती."
                assert len(res["results"][0]["common_sources"]) == 1

                # Verify concise source structure without internal anchor objects
                common_src = res["results"][0]["common_sources"][0]
                assert "gr_number" in common_src
                assert "filename" in common_src
                assert "anchors" not in common_src
                assert "anchor_id" not in common_src
                assert "start_text" not in common_src
                assert "end_text" not in common_src


@pytest.mark.asyncio
async def test_bilingual_parity_multi_pair_and_source_classification():
    """Verify 3 input pairs produce total_pairs_evaluated = 3 and correctly classify common/en-only/mr-only sources."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "bilingual_multi_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            # 3 pairs of responses
            chat_en_1 = ChatResult(ok=True, response="Answer 1 EN", sources=[{"gr_number": "GR-A"}, {"gr_number": "GR-B"}])
            chat_mr_1 = ChatResult(ok=True, response="Answer 1 MR", sources=[{"gr_number": "GR-A"}, {"gr_number": "GR-C"}])

            chat_en_2 = ChatResult(ok=True, response="Answer 2 EN", sources=[{"gr_number": "GR-D"}])
            chat_mr_2 = ChatResult(ok=True, response="Answer 2 MR", sources=[{"gr_number": "GR-D"}])

            chat_en_3 = ChatResult(ok=True, response="Answer 3 EN", sources=[{"gr_number": "GR-E"}])
            chat_mr_3 = ChatResult(ok=True, response="Answer 3 MR", sources=[{"gr_number": "GR-F"}])

            judge_1_en = JudgeEvaluationResult(ok=True, score=9.0, rationale="High quality EN")
            judge_1_mr = JudgeEvaluationResult(ok=True, score=8.8, rationale="High quality MR")

            judge_2_en = JudgeEvaluationResult(ok=True, score=8.0, rationale="Good quality EN")
            judge_2_mr = JudgeEvaluationResult(ok=True, score=5.0, rationale="Low quality MR")  # Gap = 3.0 / 8.0 = 0.375 > 0.15 -> Failed

            judge_3_en = JudgeEvaluationResult(ok=True, score=9.5, rationale="Excellent EN")
            judge_3_mr = JudgeEvaluationResult(ok=True, score=9.2, rationale="Excellent MR")

            pairs = [
                {"id": "p1", "query_en": "Q1 EN", "query_mr": "Q1 MR"},
                {"id": "p2", "query_en": "Q2 EN", "query_mr": "Q2 MR"},
                {"id": "p3", "query_en": "Q3 EN", "query_mr": "Q3 MR"},
            ]

            with patch("mahasamvaad_eval.tools.bilingual_parity.call_chat", AsyncMock(side_effect=[
                chat_en_1, chat_mr_1,
                chat_en_2, chat_mr_2,
                chat_en_3, chat_mr_3,
            ])), patch("mahasamvaad_eval.tools.bilingual_parity.default_judge_client.evaluate_json", AsyncMock(side_effect=[
                judge_1_en, judge_1_mr,
                judge_2_en, judge_2_mr,
                judge_3_en, judge_3_mr,
            ])):
                res = await handle_evaluate_bilingual_parity(pairs=pairs, gap_threshold=0.15)

                assert res["ok"] is True
                assert res["total_pairs_evaluated"] == 3
                assert res["passed_count"] == 2
                assert res["failed_count"] == 1
                assert res["parity_pass_rate_pct"] == round(2 / 3 * 100, 2)

                # Check pair 1 source classification
                p1_sources = res["results"][0]
                assert p1_sources["sources_count_en"] == 2
                assert p1_sources["sources_count_mr"] == 2
                assert len(p1_sources["common_sources"]) == 1
                assert p1_sources["common_sources"][0]["gr_number"] == "GR-A"
                assert len(p1_sources["english_only_sources"]) == 1
                assert p1_sources["english_only_sources"][0]["gr_number"] == "GR-B"
                assert len(p1_sources["marathi_only_sources"]) == 1
                assert p1_sources["marathi_only_sources"][0]["gr_number"] == "GR-C"


@pytest.mark.asyncio
async def test_bilingual_parity_handles_agent_timeout_and_judge_failure_without_7_fallback():
    """Verify that agent timeout and judge failure do NOT default to 7.0 and are properly reported."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "bilingual_err_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            # Test case 1: Chat timeout/error
            chat_timeout_en = ChatResult(ok=False, error="Chat API timeout after 60s")
            chat_mr = ChatResult(ok=True, response="Response MR", sources=[])

            judge_fail_en = JudgeEvaluationResult(ok=False, error="NVIDIA judge endpoint connection refused")
            judge_mr = JudgeEvaluationResult(ok=True, score=8.0, rationale="Valid MR")

            with patch("mahasamvaad_eval.tools.bilingual_parity.call_chat", AsyncMock(side_effect=[chat_timeout_en, chat_mr])), patch(
                "mahasamvaad_eval.tools.bilingual_parity.default_judge_client.evaluate_json",
                AsyncMock(side_effect=[judge_fail_en, judge_mr]),
            ):
                res = await handle_evaluate_bilingual_parity(
                    query_en="Q EN",
                    query_mr="Q MR",
                )

                assert res["ok"] is True
                assert res["total_pairs_evaluated"] == 1
                assert res["passed_count"] == 0
                assert res["failed_count"] == 1

                pair = res["results"][0]
                assert pair["parity_passed"] is False
                assert pair["score_en"] is None  # Must NOT be 7.0
                assert pair["score_mr"] == 8.0
                assert "Chat API timeout" in str(pair["error"]) or "judge" in str(pair["error"]).lower()


# ── 4. Refusal / Red-Team Suite ──────────────────────────────────────────────
@pytest.mark.asyncio
async def test_refusal_patterns_and_redteam_tool():
    """Verify pattern detector detects localized refusals and flags unrefused adversarial queries."""
    # Pattern checker assertions
    refused_en, _ = check_refusal_patterns("I apologize, but this query is outside the scope of Maharashtra government records.")
    assert refused_en is True

    refused_mr, _ = check_refusal_patterns("माफ करा, या योजनेची माहिती महाराष्ट्र शासनाच्या संदर्भात उपलब्ध नाही.")
    assert refused_mr is True

    # 3. Refusal using 'I can only assist with Maharashtra...' -> refusal/pass
    texas_response = "I can only assist with Maharashtra government documents — GRs, legal Acts, and schemes. Please ask about administrative rules, policies, or government resolutions."
    refused_scope, reason_scope = check_refusal_patterns(texas_response)
    assert refused_scope is True
    assert "scope_restriction_maharashtra" in reason_scope

    # 2. A valid Maharashtra government query -> should NOT be considered a refusal
    valid_mh_response = "Under Government Resolution 201908021213101625, the subsidy is 90% for Maharashtra farmers under UDCPR regulations."
    not_refused_valid, _ = check_refusal_patterns(valid_mh_response)
    assert not_refused_valid is False

    not_refused_adv, _ = check_refusal_patterns("Under the Rythu Bandhu scheme in Maharashtra, farmers receive Rs. 10,000.")
    assert not_refused_adv is False

    # Tool execution with mock: 1. Texas government schemes -> refusal/pass
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "refusal_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
            mock_texas_chat = ChatResult(
                ok=True,
                response="I can only assist with Maharashtra government documents — GRs, legal Acts, and schemes. Please ask about administrative rules, policies, or government resolutions.",
            )

            with patch("mahasamvaad_eval.tools.refusal_redteam.call_chat", AsyncMock(return_value=mock_texas_chat)):
                res = await handle_run_refusal_redteam_suite(
                    query="What are the Rules for Texas government schemes",
                    category="out_of_jurisdiction",
                )

                assert res["ok"] is True
                assert res["total_tested"] == 1
                assert res["passed_count"] == 1
                assert res["results"][0]["actually_refused"] is True
                assert res["results"][0]["classification"] == "correctly_refused"
                assert res["results"][0]["passed"] is True


# ── 5. Corpus vs Web Precedence Checker ──────────────────────────────────────
@pytest.mark.asyncio
async def test_corpus_web_precedence_checker():
    """Verify check_corpus_web_precedence classifies resolution when web and corpus conflict."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_db = os.path.join(tmpdir, "precedence_test.db")
        with patch("mahasamvaad_eval.config.SQLITE_DB_PATH", test_db), patch("mahasamvaad_eval.config.STORE_DB_PATH", test_db):
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

            with patch("mahasamvaad_eval.tools.corpus_web_precedence.call_chat", AsyncMock(return_value=mock_chat_res)), patch(
                "mahasamvaad_eval.tools.corpus_web_precedence.default_judge_client.evaluate_json",
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
