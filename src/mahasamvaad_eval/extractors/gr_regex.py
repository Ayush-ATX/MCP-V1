"""calibration_data/gr_number_date_regex_patterns.py — Regex extraction patterns for factual entities."""
from __future__ import annotations

import re
from typing import NamedTuple


class ExtractedEntity(NamedTuple):
    text: str
    entity_type: str  # gr_number | date | section | financial_or_quantity | department
    normalized: str


# ── Translation Table & Month Dictionaries ────────────────────────────────────
DEV_TO_ASCII = str.maketrans("०१२३४५६७८९", "0123456789")

_MARATHI_MONTHS_MAP = {
    "जानेवारी": 1, "फेब्रुवारी": 2, "मार्च": 3, "एप्रिल": 4, "मे": 5, "जून": 6,
    "जुलै": 7, "ऑगस्ट": 8, "सप्टेंबर": 9, "ऑक्टोबर": 10, "नोव्हेंबर": 11, "डिसेंबर": 12,
}

_ENGLISH_MONTHS_MAP = {
    "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
    "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
}

_MARATHI_MONTHS = "|".join(_MARATHI_MONTHS_MAP.keys())
_ENGLISH_MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"


def _normalize_marathi_digits(text: str) -> str:
    """Translate Devanagari numerals ०-९ to ASCII 0-9."""
    return text.translate(DEV_TO_ASCII)


def normalize_gr(gr_str: str) -> str:
    """Normalize a GR number by translating Devanagari numerals, lowercasing, and stripping punctuation."""
    norm = _normalize_marathi_digits(gr_str).lower()
    norm = re.sub(r"[\s\-–/\._]+", "", norm)
    return norm


def parse_iso_date(raw_text: str) -> str | None:
    """Parse a textual date representation into standard ISO YYYY-MM-DD format."""
    text = _normalize_marathi_digits(raw_text).strip().lower()

    # 1. YYYY-MM-DD or YYYY/MM/DD
    m1 = re.match(r"^(19|20\d{2})[-/\.](0?[1-9]|1[0-2])[-/\.](0?[1-9]|[12]\d|3[01])$", text)
    if m1:
        return f"{int(m1.group(1)):04d}-{int(m1.group(2)):02d}-{int(m1.group(3)):02d}"

    # 2. DD-MM-YYYY or DD/MM/YYYY
    m2 = re.match(r"^(0?[1-9]|[12]\d|3[01])[-/\.](0?[1-9]|1[0-2])[-/\.](19\d{2}|20\d{2})$", text)
    if m2:
        return f"{int(m2.group(3)):04d}-{int(m2.group(2)):02d}-{int(m2.group(1)):02d}"

    # 3. DD Month YYYY (English or Marathi) e.g., '10 February 2026', '10th February 2026', '१० फेब्रुवारी २०२६'
    m3 = re.match(r"^(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?\s+([^\s\d,]+),?\s+(19\d{2}|20\d{2})$", text)
    if m3:
        day = int(m3.group(1))
        month_str = m3.group(2).lower()
        year = int(m3.group(3))
        month = _ENGLISH_MONTHS_MAP.get(month_str) or _MARATHI_MONTHS_MAP.get(month_str)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"

    # 4. Month DD, YYYY e.g., 'February 10, 2026'
    m4 = re.match(r"^([a-z]+)\s+(0?[1-9]|[12]\d|3[01])(?:st|nd|rd|th)?,?\s+(19\d{2}|20\d{2})$", text)
    if m4:
        month_str = m4.group(1).lower()
        day = int(m4.group(2))
        year = int(m4.group(3))
        month = _ENGLISH_MONTHS_MAP.get(month_str)
        if month:
            return f"{year:04d}-{month:02d}-{day:02d}"

    return None


def _normalize_text(text: str) -> str:
    norm = _normalize_marathi_digits(text)
    norm = re.sub(r"[\s\-_/]+", " ", norm).strip().lower()
    return norm


def _is_valid_gr_token(val: str) -> bool:
    """Validate that an extracted candidate GR number is genuine and not an ordinary word fragment."""
    v = val.strip().strip(":,.;()[]{}*\"'")
    if len(v) < 4:
        return False
    # Must contain at least one digit or standard departmental slash/hyphen structure
    has_digit = bool(re.search(r"[\d\u0966-\u096F]", v))
    has_gr_structure = ("/" in v or "-" in v or "–" in v) and len(v) >= 6
    return has_digit or has_gr_structure


# ── 1. GR Number Patterns ─────────────────────────────────────────────────────
_GR_PATTERNS = [
    # 18-digit raw GR codes (e.g. 201908021213101625, 202602101741321319)
    re.compile(r"\b(20\d{2}[01]\d[0-3]\d[0-2]\d[0-5]\d[0-5]\d\d{2,4})\b"),
    # Word-bounded English prefix: GR No. / GR: / Government Resolution No.
    re.compile(r"\b(?:GR|G\.R\.|Government Resolution)\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F\-\/\._]+)", re.IGNORECASE),
    # Word-bounded Marathi prefix: शासन निर्णय क्रमांक / शासन निर्णय क्र.
    re.compile(r"\b(?:शासन\s*निर्णय)\s*(?:क्रमांक|क्र\.?)?\s*[:\-]?\s*([A-Za-z0-9\u0900-\u097F\-\/\._]+)", re.IGNORECASE),
    # Word-bounded departmental alphanumeric formats e.g. एनएपी-२०२५/प्र.क्र.१७७/जमीन-०१अ, MIS-2018/CR-45/UD-12
    re.compile(r"\b([A-Za-z\u0900-\u097F]{2,10}\s*[-–]\s*[\d\u0966-\u096F]{2,4}\s*[\/\-]\s*[A-Za-z0-9\u0900-\u097F\d\u0966-\u096F\-\/\._]+)\b", re.IGNORECASE),
]

# ── 2. Date Patterns ─────────────────────────────────────────────────────────
_DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    re.compile(r"\b([0-3]?\d[\/\-.][0-1]?\d[\/\-.](?:19|20)\d{2})\b"),
    # YYYY-MM-DD or YYYY/MM/DD
    re.compile(r"\b((?:19|20)\d{2}[\/\-.][0-1]?\d[\/\-.][0-3]?\d)\b"),
    # 15 August 2023 / 15th August, 2023 / 10 Feb 2026
    re.compile(rf"\b([0-3]?\d(?:st|nd|rd|th)?\s+(?:{_ENGLISH_MONTHS}),?\s+(?:19|20)\d{{2}})\b", re.IGNORECASE),
    # February 10, 2026
    re.compile(rf"\b((?:{_ENGLISH_MONTHS})\s+[0-3]?\d(?:st|nd|rd|th)?,?\s+(?:19|20)\d{{2}})\b", re.IGNORECASE),
    # १५ ऑगस्ट २०२३ / १० फेब्रुवारी २०२६
    re.compile(rf"\b([0-3]?[\d\u0966-\u096F]{{1,2}}\s+(?:{_MARATHI_MONTHS}),?\s+[\d\u0966-\u096F]{{4}})\b", re.IGNORECASE),
]

# ── 3. Section / Act / Rule Patterns ─────────────────────────────────────────
_SECTION_PATTERNS = [
    # Section 52, Section 52(1), Sections 52 to 57, Sec. 12
    re.compile(r"\b(?:Sections?|Secs?\.?)\s+([0-9]+[A-Za-z]?(?:\s*\([0-9a-zA-Z\u0966-\u096F]+\))*(?:\s*(?:to|and|-|–)\s*[0-9]+[A-Za-z]?(?:\s*\([0-9a-zA-Z\u0966-\u096F]+\))*)*)", re.IGNORECASE),
    # कलम १५, कलम १५(१), कलमे १५ ते २०, नियम ४(२)
    re.compile(r"\b(?:कलम|कलमे|नियम)\s+([0-9\u0966-\u096F]+[A-Za-z\u0900-\u097F]?(?:\s*\([0-9\u0966-\u096F]+\))*(?:\s*(?:ते|आणि|व|-|–)\s*[0-9\u0966-\u096F]+[A-Za-z\u0900-\u097F]?(?:\s*\([0-9\u0966-\u096F]+\))*)*)", re.IGNORECASE),
]

# ── 4. Financial & Quantity Figures ──────────────────────────────────────────
_FINANCIAL_PATTERNS = [
    re.compile(r"\b(?:Rs\.?|INR|₹)\s*([\d,]+(?:\.\d+)?(?:\s*(?:lakh|crore|thousand|लाख|कोटी|हजार))?)\b", re.IGNORECASE),
    re.compile(r"\b([\d,]+(?:\.\d+)?\s*(?:रुपये|रु\.?|लाख|कोटी))\b", re.IGNORECASE),
    re.compile(r"\b(\d+(?:\.\d+)?\s*(?:%|टक्के|percent))\b", re.IGNORECASE),
]

# ── 5. Known Department Names ────────────────────────────────────────────────
_DEPARTMENTS = [
    "Agriculture", "Energy", "Urban Development", "Social Justice",
    "Revenue", "Finance", "Public Works", "Water Resources", "Women and Child Development",
    "Rural Development", "Industries", "Health and Family Welfare",
    "कृषी विभाग", "ऊर्जा विभाग", "नगर विकास विभाग", "सामाजिक न्याय विभाग",
    "महसूल विभाग", "वित्त विभाग", "सार्वजनिक बांधकाम विभाग", "जलसंपदा विभाग",
    "महिला व बाल विकास विभाग", "ग्रामविकास विभाग", "उद्योग विभाग", "सार्वजनिक आरोग्य विभाग",
]


def extract_entities(text: str) -> list[ExtractedEntity]:
    """Extract all identifiable factual entities from prose text with strict validation."""
    if not text:
        return []

    results: list[ExtractedEntity] = []
    seen: set[str] = set()

    # 1. GR numbers
    for pat in _GR_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(1) if m.groups() else m.group(0)
            val = val.strip().strip(":,.;()[]{}*\"'")
            if _is_valid_gr_token(val) and val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "gr_number", normalize_gr(val)))

    # 2. Dates
    for pat in _DATE_PATTERNS:
        for m in pat.finditer(text):
            val = (m.group(1) if m.groups() else m.group(0)).strip().strip(":,.;()[]{}*\"'")
            if val not in seen:
                seen.add(val)
                iso = parse_iso_date(val)
                norm_date = iso if iso else _normalize_text(val)
                results.append(ExtractedEntity(val, "date", norm_date))

    # 3. Sections / Rules
    for pat in _SECTION_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(0).strip().strip(":,.;()[]{}*\"'")
            if val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "section", _normalize_text(val)))

    # 4. Financials / Quantities
    for pat in _FINANCIAL_PATTERNS:
        for m in pat.finditer(text):
            val = m.group(0).strip().strip(":,.;()[]{}*\"'")
            if val not in seen:
                seen.add(val)
                results.append(ExtractedEntity(val, "financial_or_quantity", _normalize_text(val)))

    # 5. Departments
    for dept in _DEPARTMENTS:
        # Require word boundary for English department names to avoid sub-word false matches
        if re.search(r"\b" + re.escape(dept) + r"\b", text, re.IGNORECASE):
            if dept not in seen:
                seen.add(dept)
                results.append(ExtractedEntity(dept, "department", _normalize_text(dept)))

    return results
