"""src/mahasamvaad_eval/extractors/sentence_splitter.py — Sentence splitter for Devanagari and Latin scripts."""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<!\d)(?<!\w\.\w)(?<=[.!?।])\s+")


def split_sentences(text: str) -> list[str]:
    """Split text into sentences handling both Devanagari (।) and Latin (.) punctuation."""
    if not text or not text.strip():
        return []
    text = text.replace("।", "। ")
    parts = _SENTENCE_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]
