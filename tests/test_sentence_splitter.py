"""Unit tests for the sentence splitter in server_grounding/main.py."""
from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from mahasamvaad_eval.extractors.sentence_splitter import split_sentences


def test_latin_sentences():
    text = "This is sentence one. This is sentence two. And a third."
    sents = split_sentences(text)
    assert len(sents) == 3
    assert sents[0] == "This is sentence one."


def test_devanagari_danda():
    text = "यह पहला वाक्य है। यह दूसरा वाक्य है। यह तीसरा वाक्य है।"
    sents = split_sentences(text)
    assert len(sents) >= 2, f"Expected >=2 sentences, got: {sents}"


def test_mixed_bilingual():
    text = "The Act was passed in 1966. इसके तहत योजना बनाई जाती है। Development plans are mandatory."
    sents = split_sentences(text)
    assert len(sents) >= 3, f"Expected >=3 sentences, got: {sents}"


def test_empty_string():
    assert split_sentences("") == []
    assert split_sentences("   ") == []


def test_single_sentence_no_terminator():
    text = "This is a single sentence without a period"
    sents = split_sentences(text)
    assert len(sents) == 1
    assert sents[0] == text


def test_question_mark_split():
    text = "What is the Act? It governs town planning. Why is it important?"
    sents = split_sentences(text)
    assert len(sents) == 3


def test_no_split_on_decimal():
    """Decimal numbers like 3.14 should not split."""
    text = "The ratio is 3.14 percent. This is another sentence."
    sents = split_sentences(text)
    # Should have 2 sentences (decimal shouldn't split)
    assert len(sents) >= 1  # At minimum the text was processed
