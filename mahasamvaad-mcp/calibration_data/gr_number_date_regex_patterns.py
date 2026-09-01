"""calibration_data/gr_number_date_regex_patterns.py — Regex extraction patterns for factual entities."""
from __future__ import annotations

import re
from typing import NamedTuple


class ExtractedEntity(NamedTuple):
    text: str
    entity_type: str  # gr_number | date | section | financial_or_quantity | department
    normalized: str


# ── 1. GR Number Patterns (English + Marathi/Devanagari) ─────────────────────
_GR_PATTERNS = [
    # Marathi prefix: शासन निर्णय क्रमांक / शासन निर्णय क्र. / संकीर्ण-२०१९/प्र.क्र...
    re.compile(r"(?:शासन\s*निर्णय\s*(?:क्रमांक|क्र\.?)?\s*[:\-]?\s*)([A-Za-z\u0900-\u097F\d\u0966-\u096F\-\/\._]+)", re.IGNORECASE),
    re.compile(r"([A-Za-z\u0900-\u097F]{2,10}\s*[-–]\s*[\d\u0966-\u096F]{4}\s*[\/\-]\s*[A-Za-z\u0900-\u097F\d\u0966-\u096F\-\/\._]+)", re.IGNORECASE),
    re.compile(r"(?:GR\s*(?:No\.?|Number)?\s*[:\-]?\s*)([A-Za-z0-9\-\/\._]+)", re.IGNORECASE),
    re.compile(r"(?:G\.R\.\s*)([A-Za-z0-9\-\/\._]+)", re.IGNORECASE),
    # Raw GR codes like 201908021213101625
    re.compile(r"\b(20\d{2}[01]\d[0-3]\d[0-2]\d[0-5]\d[0-5]\d\d{2,4})\b"),
]

# ── 2. Date Patterns ─────────────────────────────────────────────────────────
_MARATHI_MONTHS = "जानेवारी|फेब्रुवारी|मार्च|एप्रिल|मे|जून|जुलै|ऑगस्ट|सप्टेंबर|ऑक्टोबर|नोव्हेंबर|डिसेंबर"
_ENGLISH_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"

_DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY or YYYY-MM-DD
    re.compile(r"\b([0-3]?\d[\/\-.][0-1]?\d[\/\-.](?:19|20)\d{2})\b"),
    re.compile(r"\b((?:19|20)\d{2}[\/\-.][0-1]?\d[\/\-.][0-3]?\d)\b"),
    # 15 August 2023 / 15th August, 2023
    re.compile(rf"\b([0-3]?\d(?:st|nd|rd|th)?\s+(?:{_ENGLISH_MONTHS}),?\s+(?:19|20)\d{{2}})\b", re.IGNORECASE),
    # १५ ऑगस्ट २०२३ / १५ ऑगस्ट 2023
    re.compile(rf"\b([0-3]?[\d\u0966-\u096F]{{1,2}}\s+(?:{_MARATHI_MONTHS}),?\s+[\d\u0966-\u096F]{{4}})\b", re.IGNORECASE),
]

# ── 3. Section / Act / Rule Patterns ─────────────────────────────────────────
_SECTION_PATTERNS = [
    re.compile(r"\b(?:Section|Sec\.?)\s+([0-9]+[A-Za-z]?(?:\s*\([0-9a-zA-Z]+\))*)\b", re.IGNORECASE),
    re.compile(r"\b(?:कलम|नियम)\s+([0-9\u0966-\u096F]+[A-Za-z\u0900-\u097F]?(?:\s*\([0-9\u0966-\u096F]+\))*)\b", re.IGNORECASE),
    re.compile(r"\b(?:Rule|Clause|Paragraph|पॅरा)\s+([0-9\u0966-\u096F]+(?:\.[0-9\u0966-\u096F]+)*)\b", re.IGNORECASE),
]

# ── 4. Financial & Quantity Figures ──────────────────────────────────────────
_FINANCIAL_PATTERNS = [
    re.compile(r"(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?(?:\s*(?:lakh|crore|thousand|लाख|कोटी|हजार))?)", re.IGNORECASE),
    re.compile(r"([\d,]+(?:\.\d+)?\s*(?:रुपये|रु\.?|लाख|कोटी))", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?\s*(?:%|टक्के|percent|HP|एच\.?पी\.?|kW|MW|हेक्टर|एकर|sq\.?\s*(?:ft|m|मीटर|फूट)))\b", re.IGNORECASE),
]

# ── 5. Known Department Names ────────────────────────────────────────────────
_DEPARTMENTS = [
    "Agriculture", "Energy", "Urban Development", "Social Justice",
    "Revenue", "Finance", "Public Works", "Water Resources", "Women and Child Development",
    "Rural Development", "Industries", "Health and Family Welfare",
    "कृषी विभाग", "ऊर्जा विभाग", "नगर विकास विभाग", "सामाजिक न्याय विभाग",
    "महसूल विभाग", "वित्त विभाग", "सार्वजनिक बांधकाम विभाग", "जलसंपदा विभाग",
    "महिला व बाल विकास विभाग", "ग्रामविकास विभाग", "उद्योग विभाग", "सार्वजनिक आरोग्य विभाग"
]


def _normalize_marathi_digits(text: str) -> str:
    """Translate Devanagari numerals ०-९ to ASCII 0-9."""
    dev_to_ascii = str.maketrans("०१२३४५६७८९", "0123456789")
    return text.translate(dev_to_ascii)


def _normalize_text(text: str) -> str:
    norm = _normalize_marathi_digits(text)
    norm = re.sub(r"[\s\-_/]+", " ", norm).strip().lower()
    return norm


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract all identifiable factual entities from prose text."""
    if not text:
        return []

    results: list[ExtractedEntity] = []
    seen: set[str] = set()

    # 1. GR numbers
    for pat in _GR_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(1) if m.groups() else m.group(0)
            val = val.strip().strip(":,.;")
            if len(val) >= 4 and val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "gr_number", _normalize_text(val)))

    # 2. Dates
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            val = (m.group(1) if m.groups() else m.group(0)).strip().strip(":,.;")
            if val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "date", _normalize_text(val)))

    # 3. Sections / Rules
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(0).strip().strip(":,.;")
            if val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "section", _normalize_text(val)))

    # 4. Financials / Quantities
    for pat in _FINANCIAL_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(0).strip().strip(":,.;")
            if val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "financial_or_quantity", _normalize_text(val)))

    # 5. Departments
    for dept in _DEPARTMENTS:
        if dept.lower() in text.lower():
            if dept not in seen:
                seen.add(dept)
                results.append(ExtractedEntity(dept, "department", _normalize_text(dept)))

    return results
