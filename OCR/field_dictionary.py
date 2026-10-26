"""
Keyword dictionary describing which fields matter for filing/lookup purposes
(not full-page transcription) and how to recognize them in noisy OCR output.

Each field has:
  aliases  - label text that appears on the form, pointing at the value
             (matched fuzzily, case-insensitive, against OCR'd row text)
  pattern  - regex the *value* should roughly match, used both to pull the
             value out of the row and as a fallback full-document scan for
             fields whose value is distinctive even without its label
             (e.g. material spec numbers) surviving OCR better than labels do
  reject_words  - (optional) words that match the pattern but are really form
             labels/headings, never a value (e.g. "FORM", "NAME")
  require_digit - (optional) value must contain at least one digit
  look_below - (optional, default True) if False, only read the value from the
             label's own row, never the cell below it
  fallback - (optional, default True) if False, never guess the value from a
             document-wide pattern scan when its label isn't found
  required - if True and never found, the document should be escalated
             even if no single field had a low confidence score
"""

# Form label words that the loose ID patterns below would otherwise accept as
# a serial / board / boiler number. Compared case-insensitively.
LABEL_WORDS = {
    "form", "name", "nat", "natl", "national", "board", "no", "number", "serial",
    "boiler", "vessel", "same", "date", "type", "by", "for", "of", "and", "the",
    "mfr", "mfrs", "manufacturer", "manufactured", "state", "back", "report",
}

FIELDS = {
    "form_number": {
        "aliases": ["form"],
        "pattern": r"\bP-?\d{1,3}\b|\bFORM\s*[A-Z]?-?\d{1,3}\b",
        "required": False,
    },
    "report_date": {
        "aliases": ["date"],
        "pattern": (
            r"(Jan(uary)?|Feb(ruary)?|Mar(ch)?|Apr(il)?|May|Jun(e)?|Jul(y)?|"
            r"Aug(ust)?|Sep(t)?(ember)?|Oct(ober)?|Nov(ember)?|Dec(ember)?)"
            r"\.?\s+\d{1,2},?\s*(19|20)?\d{2}"
            r"|\d{1,2}/\d{1,2}/(19|20)?\d{2}"
        ),
        "required": False,
    },
    "year_built": {
        "aliases": ["year built"],
        "pattern": r"(18|19|20)\d{2}",
        "required": False,
    },
    "boiler_no": {
        "aliases": ["boiler no", "boiler number", "vessel no"],
        "pattern": r"[A-Za-z0-9\-]{1,15}",
        "reject_words": LABEL_WORDS,
        "require_digit": True,
        "fallback": False,
        "required": False,
    },
    "serial_no": {
        "aliases": ["serial no", "mfrs serial", "mfr's serial", "manufacturer's serial"],
        "pattern": r"[A-Za-z0-9\-]{1,20}",
        "reject_words": LABEL_WORDS,
        "require_digit": True,
        "fallback": False,
        "required": True,
    },
    "national_board_no": {
        "aliases": ["national board", "natl board", "nat. board", "state and state no"],
        "pattern": r"[A-Za-z0-9\-]{1,15}",
        "reject_words": LABEL_WORDS,
        "require_digit": True,
        "fallback": False,
        "required": True,
    },
    "manufactured_by": {
        "aliases": ["manufactured by"],
        "pattern": r".{3,60}",
        "look_below": False,
        "required": True,
    },
    "manufactured_for": {
        "aliases": ["manufactured for"],
        "pattern": r".{3,60}", #. includes all chars except newlines
        "look_below": False,
        "required": False,
    },
    "material_spec": {
        # Distinctive enough (SA###) that it's worth a full-document regex
        #fallback even when the "Material spec. no." label is unreadable.
        "aliases": ["material spec", "spec no", "material spec. no"],
        "pattern": r"S[AB]\s?-?\d{2,4}[A-Z]?", #form S(A/B) with optional space/hyphen, 2-4 digits and trailing char
        "required": False,
    },
}

# Confidence below this -> escalate that field to a human even if it was found.
FIELD_CONFIDENCE_THRESHOLD = 0.75

# Fuzzy-match tolerance for alias matching against noisy OCR text
# (0 = exact substring only, higher = more forgiving of misreads).
#Allows for strings like "manufactored" to be read as "manufactured"
ALIAS_FUZZ_RATIO = 0.75
