"""
Extracts only the handful of fields needed to file/look up a scanned report
(dates, serial/boiler numbers, material specs, manufacturer names, ...)
instead of transcribing the whole page. OCR is used for what it is good at:
reading the printed keywords ("Year built", "Serial No.", ...) to find where each
field is. A small cropped image of every field it locates is saved, so a vision
model (or a person) can read the value, handwriting included, from that crop
instead of the whole page. OCR's own reading is kept alongside, and doubtful
values are flagged for review.

Usage: python field_extraction.py [input] [--denoise] [--invert]
  input defaults to pdfs_to_process/. It may be a folder of PNGs/PDFs, a
  single PDF, or a single PNG.
  PDFs are rendered to PNGs (output/pdf_pages/) first; --denoise/--invert apply
  to those PDF-derived pages only. Existing PNGs are used as-is.
Outputs: output/field_index.csv, output/field_index.json, output/field_crops/*.png
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from field_crops import locate_field_crops
from form_templates import match_page
from field_dictionary import ALIAS_FUZZ_RATIO, FIELD_CONFIDENCE_THRESHOLD, FIELDS, KNOWN_WORDS
from llm_verify import verify_field
from pdf_preprocess import pdf_to_pngs

import os
from google import genai
from google.genai import types

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "field_index.csv"
OUTPUT_JSON = OUTPUT_DIR / "field_index.json"
FIELD_CROPS_DIR = OUTPUT_DIR / "field_crops"
PDF_PAGES_DIR = OUTPUT_DIR / "pdf_pages"
OCR_BOXES_DIR = OUTPUT_DIR / "ocr_boxes"
MAX_REJECTED_BOXES = 4   # a refused value cell is a few boxes at most
MAX_REJECTED_TEXT = 40   # ...and short; longer prose that doesn't fit a pattern is not a value cell
PDF_INPUT_DIR = Path("pdfs_to_process")  # default input: drop PDFs here and run with no arguments

#Turns a pdf into a list of images (one per page) which are then used in run_ocr 
def collect_images(input_path: Path, denoise: bool, invert: bool) -> list[tuple[str, list[Path]]]:
    """
    Resolve the input into (document name, page images) pairs, rendering any
    PDFs to PNG first. A PDF is one document with one image per page; a
    standalone PNG is a one-page document.
    """
    if input_path.is_dir():
        files = sorted(input_path.iterdir())
    else:
        files = [input_path]

    documents = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".png":
            documents.append((f.name, [f]))
        elif suffix == ".pdf":
            print(f"Rendering {f.name} at 300 DPI...")
            documents.append((f.name, pdf_to_pngs(f, PDF_PAGES_DIR, denoise=denoise, invert=invert)))
    return documents

#Stores text read in from ocr, confidence score, and a bounding box of the text
@dataclass
class OcrBox:
    text: str
    score: float
    box: list  # [x1, y1, x2, y2]: top-left and bottom-right pixel corners

#Actually run the ocr on the images
def run_ocr(ocr: PaddleOCR, image_path: Path) -> list[OcrBox]:
    boxes = []
    for result in ocr.predict(str(image_path)):
        data = result.json["res"]
        for text, score, box in zip(data["rec_texts"], data["rec_scores"], data["rec_boxes"]): #Records 3 lists, one of texts, their scores, and their bounding boxes
            # float()/int() convert NumPy types to plain Python ones so the
            # results can be written to JSON later.
            boxes.append(OcrBox(text=text, score=float(score), box=[int(v) for v in box]))
    return boxes

def run_gemini(file_path: Path) -> dict:
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    mime = (
        "application/pdf"
        if file_path.suffix.lower() == ".pdf"
        else "image/png"
    )

    prompt = f"""
    Read this scanned manufacturing report.

    Extract these fields:
    {list(FIELDS.keys())}

    Return a JSON object with exactly those field names.
    Each field must contain:
      "value": extracted text or null
      "confidence": number between 0 and 1

    Confidence is your estimated confidence in the reading.

    Preserve the text as written in the document.
    Do not guess unreadable information.
    Use null when a field cannot be read.
    Return only JSON.
    """

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=[
            types.Part.from_bytes(
                data=file_path.read_bytes(),
                mime_type=mime
            ),
            prompt
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json"
        )
    )

    return json.loads(response.text)

def group_into_rows(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    """Group boxes into text rows by vertical-center overlap, tolerant of skew."""
    rows: list[list[OcrBox]] = []
    # Go through boxes top to bottom (by vertical center).
    for b in sorted(boxes, key=lambda b: (b.box[1] + b.box[3]) / 2):
        b_center = (b.box[1] + b.box[3]) / 2
        b_height = b.box[3] - b.box[1]
        for row in rows:
            row_center = sum((r.box[1] + r.box[3]) / 2 for r in row) / len(row)
            row_height = sum(r.box[3] - r.box[1] for r in row) / len(row)
            # Same row if the centers are within 60% of a text height of each
            # other; the tolerance absorbs scans that are slightly tilted.
            if abs(b_center - row_center) <= max(b_height, row_height) * 0.6:
                row.append(b)
                break
        else:
            # No existing row was close enough, so this box starts a new row.
            rows.append([b])
    # Read each row left to right, and order the rows top to bottom.
    for row in rows:
        row.sort(key=lambda b: b.box[0])
    rows.sort(key=lambda row: min(b.box[1] for b in row))
    return rows


def fuzzy_contains(haystack: str, needle: str, min_ratio: float) -> bool:
    """
    Word-level fuzzy match: every word in `needle` must have a close match
    among haystack's words. Deliberately word-level rather than a sliding
    character window over the whole phrase - on short/garbled OCR text a
    character-soup SequenceMatcher ratio hits false positives constantly
    (e.g. "al diamete" vs "boiler no " share enough letters in the wrong
    order to clear 0.75), so this is the difference between "manufactured
    by" matching only near that actual label vs. matching random table junk.
    """
    haystack_words = re.findall(r"[a-z0-9']+", haystack.lower())
    needle_words = needle.lower().split()
    for nw in needle_words:
        if not any(
            rw == nw or (len(nw) >= 3 and SequenceMatcher(None, rw, nw).ratio() >= min_ratio)
            for rw in haystack_words
        ):
            return False
    return True


def clean_text(text: str) -> str:
    """Drop non-ASCII characters (OCR noise like stray CJK glyphs) so values are safe to write anywhere."""
    return text.encode("ascii", "ignore").decode("ascii").strip()


def find_value(spec: dict, pattern: re.Pattern, text: str, rejections: list | None = None) -> str | None:
    """
    First pattern match in `text` that is an acceptable value for this field:
    not a known label word, containing a digit if the field requires one, and
    (for names) mostly letters. If `rejections` is given, the reason each
    candidate was turned down is appended to it.
    """
    reject = spec.get("reject_words", ())
    cleaned = clean_text(text)
    matched_any = False
    for m in pattern.finditer(cleaned):
        matched_any = True
        value = m.group(0)
        reason = None
        if value.lower() in reject:
            reason = "label word, not a value"
        elif spec.get("require_digit") and not any(c.isdigit() for c in value):
            reason = "contains no digit"
        elif spec.get("min_alpha_ratio"):
            # mostly digits/symbols: noise, not a name
            chars = [c for c in value if not c.isspace()]
            if not chars or sum(c.isalpha() for c in chars) / len(chars) < spec["min_alpha_ratio"]:
                reason = "mostly digits/symbols"
        if reason is None and spec.get("require_capital") and not re.search(r"\b[A-Z][A-Za-z]{2,}", value):
            reason = "no capitalized word"  # lowercase text is body copy misread as a name
        if reason:
            if rejections is not None:
                rejections.append((value, reason))
            continue
        return value
    if rejections is not None and not matched_any:
        rejections.append((cleaned, "does not fit the field's pattern"))
    return None


def autocorrect_words(text: str) -> tuple[str, list[str]]:
    """
    Replace words that are a near miss of a KNOWN_WORDS entry ("Aqud-Chem" ->
    "Aqua-Chem"). Conservative: 5+ letters and a very close match only, so real
    words are left alone. Returns the new text and the changes made.
    """
    known = {w.lower(): w for w in KNOWN_WORDS}
    changes = []

    def fix(m: re.Match) -> str:
        word = m.group(0)
        if len(word) < 5 or word.lower() in known:
            return word
        best = max(known, key=lambda k: SequenceMatcher(None, word.lower(), k).ratio())
        if SequenceMatcher(None, word.lower(), best).ratio() >= 0.85:
            changes.append(f"{word} -> {known[best]}")
            return known[best]
        return word

    return re.sub(r"[A-Za-z][A-Za-z-]*", fix, text), changes


def extract_fields(rows: list[list[OcrBox]]) -> tuple[dict, dict]:
    """
    For each field: scan rows for a label alias, take the OCR boxes after it
    in that row as the value. If no labeled row matches (label itself didn't
    survive OCR), fall back to a full-document regex scan for fields whose
    value has a distinctive enough shape to find unlabeled (e.g. SA53B).

    Returns (found, rejected). `rejected` maps a field to the label-adjacent
    boxes whose text failed validation (e.g. an "H" where a board number should
    be), with the reason. Where to crop for each field is decided separately, in
    field_crops.py.
    """
    found = {}
    rejected = {}

    for field_name, spec in FIELDS.items():
        pattern = re.compile(spec["pattern"], re.IGNORECASE)

        skip_words = spec.get("skip_row_words", ())

        for row_index, row in enumerate(rows):
            row_text = " ".join(b.text for b in row)
            if any(fuzzy_contains(row_text, w, ALIAS_FUZZ_RATIO) for w in skip_words):
                continue  # e.g. a certificate-expiry line, not the report date
            matched_alias = next(
                (a for a in spec["aliases"] if fuzzy_contains(row_text, a, ALIAS_FUZZ_RATIO)),
                None,
            )
            if not matched_alias:
                continue

            # Find where the label ends: the last box in the row that looks like
            # the label's final word (e.g. "by" in "manufactured by"). Everything
            # to the right of it is treated as the value.
            alias_last_word = matched_alias.split()[-1].lower()
            split_idx = 0
            for i, b in enumerate(row):
                if alias_last_word in b.text.lower() or \
                   SequenceMatcher(None, b.text.lower(), alias_last_word).ratio() > 0.6:
                    split_idx = i + 1

            # Candidate places for the value: to the right of the label in the
            # same row, then (these forms are tables) in the cell directly below.
            candidates = []
            if row[split_idx:]:
                candidates.append(("same row", row[split_idx:]))
            # (Free-text fields skip this: the cell below is usually a printed caption.)
            below = boxes_below_label(row, split_idx, matched_alias, rows) if spec.get("look_below", True) else []
            if below:
                candidates.append(("cell below", below))

            value_boxes, value_text = None, None
            for where, group in candidates:
                text = " ".join(b.text for b in group).strip(" .:-")
                reasons = []
                value_text = find_value(spec, pattern, text, reasons)
                if value_text is not None:
                    value_boxes = group
                    break
                # Not accepted as the value, but remember where it was and why.
                # Only candidates that look like a value cell are kept: the
                # label words also fuzzy-match rows of body text ("DATA REPORT"),
                # and a long run of prose isn't something to re-read. Every box on
                # the page is still saved in ocr_boxes/.
                reason_text = "; ".join(dict.fromkeys(r for _, r in reasons))
                if len(group) > MAX_REJECTED_BOXES or (
                    len(text) > MAX_REJECTED_TEXT and reason_text == "does not fit the field's pattern"
                ):
                    continue
                rejected.setdefault(field_name, []).append({
                    "text": text, "where": where, "reason": reason_text, "row": row_index,
                    "score": round(min(b.score for b in group), 3),
                    # one entry per OCR box (same order), so a single cell can be re-read
                    "boxes": [b.box for b in group],
                    "box_texts": [b.text for b in group],
                    "box_scores": [round(b.score, 3) for b in group],
                })
            if value_text is None:
                # Nothing fit the expected shape (date, spec code, ...) - don't
                # keep raw noise as the value (the boxes are kept in `rejected`).
                continue

            # Take the first (topmost) match, not the highest-confidence one:
            # these forms put report metadata near the top, and a later row
            # that coincidentally matches the same short alias (e.g. "date"
            # inside a certificate-expiry line) is more likely to be a false
            # positive than the genuine field, however it happened to OCR.
            conf = min(b.score for b in value_boxes)
            found[field_name] = {
                "value": value_text, "confidence": conf,
                "boxes": [b.box for b in value_boxes], "source": "label",
            }
            if spec.get("vocabulary"):
                found[field_name]["value"], fixes = autocorrect_words(value_text)
                if fixes:
                    found[field_name]["autocorrected"] = fixes
            break

        # No usable label row: scan every row for text shaped like the value.
        # These matches are marked "pattern_fallback" and always get reviewed.
        # Fields with "fallback": False (serial/board/boiler numbers) are never
        # guessed this way - any digit string would match - so they stay unfound.
        if field_name not in found and spec.get("fallback", True):
            for row in rows:
                row_text = " ".join(b.text for b in row)
                if any(fuzzy_contains(row_text, w, ALIAS_FUZZ_RATIO) for w in skip_words):
                    continue
                value = find_value(spec, pattern, row_text)
                if value:
                    contributing = [b for b in row if b.text and b.text in value]
                    found[field_name] = {
                        "value": value,
                        "confidence": min((b.score for b in contributing), default=0.5),
                        "boxes": [b.box for b in contributing] or [row[0].box],
                        "source": "pattern_fallback",
                    }
                    break

    apply_sanity_checks(found)
    return found, rejected


def value_ok(spec: dict, text: str) -> bool:
    """Does `text` have the shape of this field's value (used to judge whether a crop zone holds one)?"""
    return find_value(spec, re.compile(spec["pattern"], re.IGNORECASE), text) is not None


def save_field_crops(image_path: Path, rows: list[list[OcrBox]], found: dict) -> tuple[dict, dict]:
    """
    Save a small image of every located field to field_crops/<page>__<field>.png
    and return ({field: {crop_file, crop_box, crop_kind}}, form_template_info).
    These crops, not the whole page, are what a vision model should be sent.

    If the page is lined up with a known form type (form_templates/), that form's
    field rectangles are used and only those fields get a crop (a field the form
    does not have on this page gets none). Otherwise the crop is placed from the
    keywords OCR found. A field with no crop is left for the model to find on the
    full page.
    """
    FIELD_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    with Image.open(image_path) as page:
        template = match_page(page)
        if template["template"]:
            located = template["fields"]
        else:
            located = locate_field_crops(rows, found, page.width, page.height, fuzzy_contains, value_ok)
        for field_name, crop in located.items():
            path = FIELD_CROPS_DIR / f"{image_path.stem}__{field_name}.png"
            page.crop(tuple(crop["box"])).save(path)
            saved[field_name] = {
                "crop_file": f"{FIELD_CROPS_DIR.name}/{path.name}", "crop_box": crop["box"], "crop_kind": crop["kind"],
            }
    return saved, {"name": template["template"], "inliers": template["inliers"]}


def find_label_boxes(row: list[OcrBox], split_idx: int, alias: str, whole_row_fallback: bool = False) -> list[OcrBox]:
    """
    The boxes in `row` that make up the label (the matched keyword), not its
    neighbours. If none stand out, returns [] - or the whole row when
    `whole_row_fallback` is set (used for cropping, never for finding values).
    """
    alias_words = alias.lower().split()
    boxes = [
        b for b in row[:split_idx]
        if any(w in b.text.lower() or SequenceMatcher(None, b.text.lower(), w).ratio() > 0.6 for w in alias_words)
    ] or row[max(split_idx - 1, 0):split_idx]
    return boxes or (row if whole_row_fallback else [])


def boxes_below_label(row: list[OcrBox], split_idx: int, alias: str, rows: list[list[OcrBox]]) -> list[OcrBox]:
    """
    Boxes in the table cell directly under a label: the nearest row below it
    that overlaps the label's horizontal span (padded a little, since values
    are often centered or indented relative to their label).
    """
    label_boxes = find_label_boxes(row, split_idx, alias)
    if not label_boxes:
        return []

    x1 = min(b.box[0] for b in label_boxes)
    x2 = max(b.box[2] for b in label_boxes)
    y2 = max(b.box[3] for b in label_boxes)
    height = max(b.box[3] - b.box[1] for b in label_boxes)
    pad = (x2 - x1) * 0.25

    candidates = []
    for other in rows:
        if other is row:
            continue
        for b in other:
            center_y = (b.box[1] + b.box[3]) / 2
            # below the label, but not so far down it's a different section
            if b.text.lstrip().startswith("("):
                continue  # printed caption like "(Name of part...)", not a filled-in value
            if y2 - height * 0.3 < center_y < y2 + height * 4 and b.box[2] > x1 - pad and b.box[0] < x2 + pad:
                candidates.append(b)
    if not candidates:
        return []

    candidates.sort(key=lambda b: (b.box[1] + b.box[3]) / 2)
    first_y = (candidates[0].box[1] + candidates[0].box[3]) / 2
    nearest = [b for b in candidates if abs((b.box[1] + b.box[3]) / 2 - first_y) <= height * 0.6]
    return sorted(nearest, key=lambda b: b.box[0])


# Common ASME boiler/pressure-vessel material specs (SA-###), used to catch
# misreads like "SA 31". Not exhaustive: an unlisted spec is flagged for a
# human to confirm, not rejected.
KNOWN_MATERIAL_SPECS = {
    36, 53, 105, 106, 178, 179, 181, 182, 192, 193, 194, 210, 213, 214, 216, 234,
    240, 266, 278, 285, 299, 302, 307, 312, 333, 336, 351, 515, 516, 517, 533, 537,
    570, 612, 675,
}
EARLIEST_REPORT_YEAR = 1900


def apply_sanity_checks(found: dict) -> None:
    """
    Cross-check extracted values for plausibility. A failing value is kept (so
    the reviewer can see it) but gets a "flag" that forces human review.
    """
    year_built = int(found["year_built"]["value"]) if "year_built" in found else None

    entry = found.get("report_date")
    if entry:
        year_token = re.findall(r"\d+", entry["value"])[-1]
        year = int(year_token)
        if year_token in ("19", "20"):
            # Just the century: the last two digits of the year were lost.
            entry.setdefault("flag", f"report_date year looks cut off ('{year_token}')")
        if year < 100:
            year += 1900  # two-digit year on these forms means 19xx
        if not EARLIEST_REPORT_YEAR <= year <= datetime.now().year:
            entry.setdefault("flag", f"report_date year {year} is not plausible")
        elif year_built and year < year_built:
            entry.setdefault("flag", f"report_date year {year} is before year built {year_built}")

    entry = found.get("material_spec")
    if entry:
        value = entry["value"].replace("_", "-")
        m = re.match(r"S([AB])\s?-?(\d+)", value, re.IGNORECASE)
        if m:
            # normalize "SA 516" -> "SA-516" and "SA_285-C" -> "SA-285-C", keeping any grade
            entry["value"] = f"S{m.group(1).upper()}-{m.group(2)}" + value[m.end():]
            if int(m.group(2)) not in KNOWN_MATERIAL_SPECS:
                entry.setdefault("flag", f"{entry['value']} is not a known material spec")

    # On these forms the boiler number and the manufacturer's serial number are
    # the same value, so a mismatch means at least one was misread. The national
    # board number is a different number: if it matches either, it was copied.
    boiler, serial, board = (found.get(n) for n in ("boiler_no", "serial_no", "national_board_no"))
    if boiler and serial and boiler["value"] != serial["value"]:
        for entry in (boiler, serial):
            entry.setdefault("flag", "boiler_no and serial_no differ (normally identical on these forms)")
    if board:
        for other_name, other in (("serial_no", serial), ("boiler_no", boiler)):
            if other and board["value"] == other["value"]:
                board.setdefault("flag", f"same value as {other_name} (should be a different number)")
    if serial and year_built and serial["value"] == str(year_built):
        serial.setdefault("flag", "same as year built")


def save_ocr_boxes(image_path: Path, boxes: list[OcrBox]) -> str:
    """
    Write every box OCR found on the page (text, score, [x1, y1, x2, y2]),
    whether or not it ended up in a field. Returns the file's path relative to
    the output folder. This is the raw material for re-reading a region with a
    different recognizer without re-running OCR.
    """
    OCR_BOXES_DIR.mkdir(parents=True, exist_ok=True)
    path = OCR_BOXES_DIR / f"{image_path.stem}.json"
    path.write_text(json.dumps(
        {"image": image_path.name, "boxes": [{"text": b.text, "score": round(b.score, 4), "box": b.box} for b in boxes]},
        indent=1,
    ), encoding="utf-8")
    return f"{OCR_BOXES_DIR.name}/{path.name}"


def process_document(ocr: PaddleOCR, image_path: Path) -> dict:
    """OCR one page image, extract its fields, and decide whether a human must review it."""
    ocr_boxes = run_ocr(ocr, image_path)
    boxes_file = save_ocr_boxes(image_path, ocr_boxes)
    rows = group_into_rows(ocr_boxes)
    extracted, rejected = extract_fields(rows)
    crops, form_template = save_field_crops(image_path, rows, extracted)
    no_crop = {"crop_file": None, "crop_box": None, "crop_kind": None}

    # "escalate" means a person should look at this page; the reasons say why.
    result = {
        "document": image_path.name, "ocr_boxes_file": boxes_file, "form_template": form_template,
        "fields": {}, "escalate": False, "escalation_reasons": [],
    }

    for field_name, spec in FIELDS.items():
        entry = extracted.get(field_name)
        if entry is None:
            # Field wasn't found at all; only a problem if the field is required.
            # Any refused candidate boxes are kept so something else can re-read them.
            result["fields"][field_name] = {
                "value": None, "confidence": None, "llm_checked": False,
                "boxes": [], "rejected": rejected.get(field_name, []),
                **crops.get(field_name, no_crop),
            }
            if spec["required"]:
                result["escalate"] = True
                result["escalation_reasons"].append(f"{field_name}: not found")
            continue

        confidence = entry["confidence"]
        llm_checked = False

        # A fallback match found its value's *shape* (a date, a spec code)
        # somewhere in the document without ever confirming it sits next to
        # the field's actual label - so it can OCR with high confidence and
        # still be the wrong field's value entirely (e.g. picking up a
        # certificate-expiry date instead of the report date). Confidence
        # alone can't catch that, so fallback matches always get reviewed.
        needs_review = (
            confidence < FIELD_CONFIDENCE_THRESHOLD
            or entry["source"] == "pattern_fallback"
            or bool(entry.get("flag"))  # failed a plausibility check
        )

        if needs_review:
            # Ask the LLM verifier for a second opinion (a no-op without an API
            # key), looking at the same field crop that was saved above.
            crop = crops.get(field_name)
            verify_boxes = [crop["crop_box"]] if crop else entry["boxes"]
            verdict = verify_field(image_path, verify_boxes, field_name, entry["value"])
            llm_checked = verdict.used_llm

            # Outcomes, in order: LLM agrees -> accept; LLM disagrees -> use its
            # text but still escalate; no LLM -> escalate for a human to decide.
            if verdict.used_llm and verdict.agrees and entry["source"] == "label" and not entry.get("flag"):
                pass  # LLM confirmed a labeled OCR reading despite low confidence; don't escalate
            elif verdict.used_llm and verdict.llm_text:
                entry["value"] = verdict.llm_text
                result["escalate"] = True
                result["escalation_reasons"].append(
                    f"{field_name}: LLM correction '{verdict.llm_text}', needs human confirmation"
                )
            elif entry.get("flag"):
                result["escalate"] = True
                result["escalation_reasons"].append(f"{field_name}: {entry['flag']}")
            elif entry["source"] == "pattern_fallback":
                result["escalate"] = True
                result["escalation_reasons"].append(
                    f"{field_name}: label not found, value guessed from pattern match elsewhere in document"
                )
            else:
                result["escalate"] = True
                result["escalation_reasons"].append(f"{field_name}: low confidence ({confidence:.2f})")

        result["fields"][field_name] = {
            "value": entry["value"], "confidence": round(confidence, 3),
            "source": entry["source"], "llm_checked": llm_checked,
            "boxes": entry["boxes"], "rejected": rejected.get(field_name, []),
            **crops.get(field_name, no_crop),
            **({"autocorrected": entry["autocorrected"]} if entry.get("autocorrected") else {}),
        }

    return result


def merge_pages(doc_name: str, page_results: list[dict]) -> dict:
    """
    Combine the per-page results of one PDF into a single document entry.

    For each field, take the value from the first page that found it by label;
    if no page found it by label, take the earliest pattern-fallback guess.
    Preferring the earliest page matters because back pages often yield
    high-confidence junk (e.g. the "FORM U-1A (BACK)" title). Escalation is
    then recomputed from the merged fields, so a field missing on the back page
    doesn't flag a document whose front page had it.
    """
    merged = {
        "document": doc_name,
        "pages": [p["document"] for p in page_results],
        "ocr_boxes_files": [p["ocr_boxes_file"] for p in page_results],
        "form_templates": [p["form_template"] for p in page_results],
        "fields": {}, "escalate": False, "escalation_reasons": [],
    }

    for field_name in FIELDS:
        # (page result, field entry) pairs that found a value, in page order
        found = [(p, p["fields"][field_name]) for p in page_results if p["fields"][field_name]["value"] is not None]
        labeled = [pair for pair in found if pair[1].get("source") == "label"]
        # With no value anywhere, still point at a page that has a crop for the field.
        cropped = [(p, p["fields"][field_name]) for p in page_results if p["fields"][field_name].get("crop_file")]
        source_page, entry = (labeled or found or cropped or [(page_results[0], page_results[0]["fields"][field_name])])[0]

        merged["fields"][field_name] = dict(entry)
        if found:
            merged["fields"][field_name]["page"] = source_page["document"]
        else:
            # No page produced a value: keep the refused candidates from every
            # page (each tagged with its page), not just the first one's.
            merged["fields"][field_name]["rejected"] = [
                {**r, "page": p["document"]}
                for p in page_results for r in p["fields"][field_name].get("rejected", [])
            ]

        # Keep only the reasons about this field from the page that supplied it.
        reasons = [r for r in source_page["escalation_reasons"] if r.startswith(f"{field_name}:")]
        if reasons:
            merged["escalate"] = True
            merged["escalation_reasons"].extend(reasons)

    return merged


def main():
    global OUTPUT_DIR, OUTPUT_CSV, OUTPUT_JSON, FIELD_CROPS_DIR, PDF_PAGES_DIR, OCR_BOXES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=PDF_INPUT_DIR)
    parser.add_argument("--denoise", action="store_true", help="light denoise on PDF-derived pages")
    parser.add_argument("--invert", action="store_true", help="invert PDF-derived pages")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="where to write results (use a scratch folder to avoid overwriting reviewed output)")
    args = parser.parse_args()

    OUTPUT_DIR = args.output_dir
    OUTPUT_CSV, OUTPUT_JSON = OUTPUT_DIR / "field_index.csv", OUTPUT_DIR / "field_index.json"
    FIELD_CROPS_DIR, PDF_PAGES_DIR = OUTPUT_DIR / "field_crops", OUTPUT_DIR / "pdf_pages"
    OCR_BOXES_DIR = OUTPUT_DIR / "ocr_boxes"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    # Find original PDF/PNG files WITHOUT rendering them yet.
    if args.input.is_dir():
        sources = sorted(
            p for p in args.input.iterdir()
            if p.is_file() and p.suffix.lower() in (".pdf", ".png")
        )
    elif args.input.is_file() and args.input.suffix.lower() in (".pdf", ".png"):
        sources = [args.input]
    else:
        sources = []

    if not sources:
        print(f"No PDFs or PNGs found in {args.input}.")
        return

    FIELD_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Do not initialize PaddleOCR unless Gemini misses a field.
    ocr = None
    all_results = []

    def has_value(value):
        return value is not None and bool(str(value).strip())

    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    for source in sources:
        doc_name = source.name
        print(f"\nRunning Gemini on {doc_name}...")

        # STEP 1: Send original PDF/PNG directly to Gemini.
        gemini_results = run_gemini(source)

        # STEP 2: Check which fields Gemini missed.
        missing_fields = [
            name for name in FIELDS
            if not has_value(
                (gemini_results.get(name) or {}).get("value")
            )
        ]

        paddle_ran = bool(missing_fields)

        if missing_fields:
            print(f"Gemini missing: {missing_fields}")
            print("Running PaddleOCR fallback...")

            # Initialize PaddleOCR only when needed.
            if ocr is None:
                from paddleocr import PaddleOCR

                #If have enough compute try: PP-OCRv6_medium_det, PP-OCRv6_medium_rec
                ocr = PaddleOCR(
                    lang="en",
                    device="cpu",
                    text_detection_model_name="PP-OCRv6_small_det",
                    text_recognition_model_name="PP-OCRv6_small_rec",
                    enable_mkldnn=False,
                    cpu_threads=1,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )

            # Render only this document and run existing OCR.
            rendered = collect_images(
                source, args.denoise, args.invert
            )
            _, image_paths = rendered[0]

            page_results = []
            for image_path in image_paths:
                print(f"Processing {image_path.name}...")
                page_results.append(
                    process_document(ocr, image_path)
                )

            merged = merge_pages(doc_name, page_results)

        else:
            print("All fields found. Skipping PaddleOCR.")

            # Build the same output structure without OCR.
            merged = {
                "document": doc_name,
                "pages": [],
                "ocr_boxes_files": [],
                "fields": {
                    name: {
                        "value": None,
                        "confidence": None,
                        "llm_checked": False,
                        "boxes": [],
                        "rejected": [],
                        "crop_file": None,
                        "crop_box": None,
                        "crop_kind": None,
                    }
                    for name in FIELDS
                },
                "escalate": False,
                "escalation_reasons": [],
            }

        # Reset PaddleOCR warnings.
        # Review flags will be rebuilt from final selected values.
        merged["escalate"] = False
        merged["escalation_reasons"] = []

        # STEP 3: Merge Gemini results with optional PaddleOCR.
        for name in FIELDS:
            field = merged["fields"][name]

            p_value = field["value"]
            p_conf = field["confidence"]

            g = gemini_results.get(name) or {}
            g_value = g.get("value")
            g_conf = g.get("confidence")

            agree = (
                has_value(g_value)
                and has_value(p_value)
                and normalize(g_value) == normalize(p_value)
            )

            if agree:
                selected = g_value
                source_model = "both"
                status = "agreed"

            elif has_value(g_value):
                selected = g_value
                source_model = "gemini"
                status = (
                    "review_required"
                    if has_value(p_value)
                    else "gemini_only"
                )

            elif has_value(p_value):
                selected = p_value
                source_model = "paddleocr"
                status = "review_required"

            else:
                selected = None
                source_model = "none"
                status = "not_found"

            # Preserve both readings.
            field["paddleocr"] = {
                "value": p_value,
                "confidence": p_conf,
                "ran": paddle_ran,
            }

            field["gemini"] = {
                "value": g_value,
                "confidence": g_conf,
            }

            field["selection"] = {
                "value": selected,
                "source": source_model,
                "status": status,
            }

            field["value"] = selected
            field["confidence"] = (
                g_conf
                if source_model in ("gemini", "both")
                else p_conf
            )

            if status == "review_required":
                merged["escalate"] = True
                merged["escalation_reasons"].append(
                    f"{name}: model disagreement or PaddleOCR fallback"
                )

            elif status == "not_found" and FIELDS[name]["required"]:
                merged["escalate"] = True
                merged["escalation_reasons"].append(
                    f"{name}: required field not found"
                )

        all_results.append(merged)


    # Each page saved crops for every keyword it found, but a PDF's merged result
    # uses only one page's crop per field. Drop the rest so field_crops/ holds just
    # the images the output points at (the ones to send to a vision model).
    used = {f["crop_file"] for r in all_results for f in r["fields"].values() if f.get("crop_file")}
    for crop in FIELD_CROPS_DIR.glob("*.png"):
        if f"{FIELD_CROPS_DIR.name}/{crop.name}" not in used:
            crop.unlink()

    # JSON keeps the full detail (confidence, source, LLM status) per field;
    # the CSV below is a flat one-row-per-document summary of values only.
    OUTPUT_JSON.write_text(json.dumps(all_results, indent=2))

    field_names = list(FIELDS.keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "escalate", "escalation_reasons"] + field_names)
        for r in all_results:
            writer.writerow(
                [r["document"], r["escalate"], "; ".join(r["escalation_reasons"])]
                + [r["fields"][fn]["value"] or "" for fn in field_names]
            )

    n_escalate = sum(1 for r in all_results if r["escalate"])
    print(f"\nDone. {len(all_results)} documents processed, {n_escalate} flagged for human review.")
    print(f"Index: {OUTPUT_CSV}")
    print(f"Field crops: {FIELD_CROPS_DIR}")


if __name__ == "__main__":
    main()
