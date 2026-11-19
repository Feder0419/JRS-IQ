"""Form-code vocabulary and title-OCR rules shared by every stage."""
import re

# canonical display name for each form code, keyed by its alphanumerics only
# (so "Form_H3_01", "H-3" and "h3" all map to "H-3")
FORM_DISPLAY = {"H3": "H-3", "P3": "P-3", "154A": "154-A", "U1A": "U-1A", "P2": "P-2"}

# OCR of these faded title strips rarely reproduces the short code itself
# ("P-3" comes out as "P-8", "PLS", ...), but the longer title words survive
# much better. Checked in order; first hit wins. The P-3 patterns need the
# plural / "AND" title wording because the singular "superheater, waterwall,
# economizer" small print under item 3 appears on H-3 pages too.
OCR_RULES = [
    ("P-3", re.compile(r"SUPER\w{0,3}ATERS|WATER\W?WALLS\W+AND|AND\W+ECONOM")),
    ("154-A", re.compile(r"ALL\W+TYPES\W+OF\W+BOILERS|\b154\s*-?\s*A\b")),
    ("P-3", re.compile(r"\bFORM\W+P\s*-?\s*3\b")),
    ("H-3", re.compile(r"\bFORM\W+H\s*-?\s*3\b|MODEL\W+N[O0]")),
]

# generic wording of a front page's title block; only about half of the
# badly faded fronts produce any of it, so it is supporting evidence only
FRONT_KEYWORDS = re.compile(r"MANUFACT|DATA\W+RE[PF]ORT|REQUIRED\W+BY|PROVISIONS|CODE\W+RULES")


def canon_code(code):
    s = re.sub(r"^FORM[_\s-]*", "", code.strip().upper())
    s = re.sub(r"_\d+$", "", s)
    return FORM_DISPLAY.get(re.sub(r"[^A-Z0-9]", "", s), s)


def ocr_form_code(text):
    text = text.upper()
    for code, pat in OCR_RULES:
        if pat.search(text):
            return code
    return None


def has_front_keywords(text):
    return bool(FRONT_KEYWORDS.search(text.upper()))


def safe_label(label):
    if label == "_unclassified":
        return label
    return "form_" + label.lower().replace(" ", "_").replace("/", "-")
