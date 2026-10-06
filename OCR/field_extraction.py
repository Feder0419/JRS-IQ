"""
Extracts only the handful of fields needed to file/look up a scanned report
(dates, serial/boiler numbers, material specs, manufacturer names, ...)
instead of transcribing the whole page. Low-confidence fields get a second
opinion from a vision LLM (llm_verify.py); anything still shaky after that
gets flagged into a review queue for a human instead of guessed at.

Usage: python field_extraction.py [input] [--denoise] [--invert]
  input defaults to pdfs_to_process/. It may be a folder of PNGs/PDFs, a
  single PDF, or a single PNG.
  PDFs are rendered to PNGs (output/pdf_pages/) first; --denoise/--invert apply
  to those PDF-derived pages only. Existing PNGs are used as-is.
Outputs: output/field_index.csv, output/field_index.json, output/review_queue/*.png
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from paddleocr import PaddleOCR

from field_dictionary import ALIAS_FUZZ_RATIO, FIELD_CONFIDENCE_THRESHOLD, FIELDS
from llm_verify import verify_field
from pdf_preprocess import pdf_to_pngs

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "field_index.csv"
OUTPUT_JSON = OUTPUT_DIR / "field_index.json"
REVIEW_QUEUE_DIR = OUTPUT_DIR / "review_queue"
PDF_PAGES_DIR = OUTPUT_DIR / "pdf_pages"
PDF_INPUT_DIR = Path("pdfs_to_process")  # default input: drop PDFs here and run with no arguments


def collect_images(input_path: Path, denoise: bool, invert: bool) -> list[Path]:
    """Resolve the input into a list of PNGs, rendering any PDFs to PNG first."""
    if input_path.is_dir():
        files = sorted(input_path.iterdir())
    else:
        files = [input_path]

    images = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".png":
            images.append(f)
        elif suffix == ".pdf":
            print(f"Rendering {f.name} at 300 DPI...")
            images.extend(pdf_to_pngs(f, PDF_PAGES_DIR, denoise=denoise, invert=invert))
    return images


@dataclass
class OcrBox:
    text: str
    score: float
    box: list  # [x1, y1, x2, y2]


def run_ocr(ocr: PaddleOCR, image_path: Path) -> list[OcrBox]:
    boxes = []
    for result in ocr.predict(str(image_path)):
        data = result.json["res"]
        for text, score, box in zip(data["rec_texts"], data["rec_scores"], data["rec_boxes"]):
            boxes.append(OcrBox(text=text, score=float(score), box=list(box)))
    return boxes


def group_into_rows(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    """Group boxes into text rows by vertical-center overlap, tolerant of skew."""
    rows: list[list[OcrBox]] = []
    for b in sorted(boxes, key=lambda b: (b.box[1] + b.box[3]) / 2):
        b_center = (b.box[1] + b.box[3]) / 2
        b_height = b.box[3] - b.box[1]
        for row in rows:
            row_center = sum((r.box[1] + r.box[3]) / 2 for r in row) / len(row)
            row_height = sum(r.box[3] - r.box[1] for r in row) / len(row)
            if abs(b_center - row_center) <= max(b_height, row_height) * 0.6:
                row.append(b)
                break
        else:
            rows.append([b])
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


def extract_fields(rows: list[list[OcrBox]]) -> dict:
    """
    For each field: scan rows for a label alias, take the OCR boxes after it
    in that row as the value. If no labeled row matches (label itself didn't
    survive OCR), fall back to a full-document regex scan for fields whose
    value has a distinctive enough shape to find unlabeled (e.g. SA53B).
    """
    found = {}

    for field_name, spec in FIELDS.items():
        pattern = re.compile(spec["pattern"], re.IGNORECASE)

        for row in rows:
            row_text = " ".join(b.text for b in row)
            matched_alias = next(
                (a for a in spec["aliases"] if fuzzy_contains(row_text, a, ALIAS_FUZZ_RATIO)),
                None,
            )
            if not matched_alias:
                continue

            alias_last_word = matched_alias.split()[-1].lower()
            split_idx = 0
            for i, b in enumerate(row):
                if alias_last_word in b.text.lower() or \
                   SequenceMatcher(None, b.text.lower(), alias_last_word).ratio() > 0.6:
                    split_idx = i + 1
            value_boxes = row[split_idx:]
            if not value_boxes:
                continue

            value_text = " ".join(b.text for b in value_boxes).strip(" .:-")
            match = pattern.search(value_text)
            if not match:
                # Value didn't fit the expected shape (date, spec code, ...)
                # - discard rather than keep raw noise as if it were real.
                continue
            value_text = match.group(0)

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
            break

        if field_name not in found:
            for row in rows:
                row_text = " ".join(b.text for b in row)
                match = pattern.search(row_text)
                if match:
                    contributing = [b for b in row if b.text and b.text in match.group(0)]
                    found[field_name] = {
                        "value": match.group(0),
                        "confidence": min((b.score for b in contributing), default=0.5),
                        "boxes": [b.box for b in contributing] or [row[0].box],
                        "source": "pattern_fallback",
                    }
                    break

    return found


def process_document(ocr: PaddleOCR, image_path: Path) -> dict:
    rows = group_into_rows(run_ocr(ocr, image_path))
    extracted = extract_fields(rows)

    result = {"document": image_path.name, "fields": {}, "escalate": False, "escalation_reasons": []}

    for field_name, spec in FIELDS.items():
        entry = extracted.get(field_name)
        if entry is None:
            result["fields"][field_name] = {"value": None, "confidence": None, "llm_checked": False}
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
        needs_review = confidence < FIELD_CONFIDENCE_THRESHOLD or entry["source"] == "pattern_fallback"

        if needs_review:
            crop_path = REVIEW_QUEUE_DIR / f"{image_path.stem}__{field_name}.png"
            verdict = verify_field(image_path, entry["boxes"], field_name, entry["value"], crop_save_path=crop_path)
            llm_checked = verdict.used_llm

            if verdict.used_llm and verdict.agrees and entry["source"] == "label":
                pass  # LLM confirmed a labeled OCR reading despite low confidence; don't escalate
            elif verdict.used_llm and verdict.llm_text:
                entry["value"] = verdict.llm_text
                result["escalate"] = True
                result["escalation_reasons"].append(
                    f"{field_name}: LLM correction '{verdict.llm_text}', needs human confirmation"
                )
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
        }

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=PDF_INPUT_DIR)
    parser.add_argument("--denoise", action="store_true", help="light denoise on PDF-derived pages")
    parser.add_argument("--invert", action="store_true", help="invert PDF-derived pages")
    args = parser.parse_args()

    PDF_INPUT_DIR.mkdir(exist_ok=True)
    images = collect_images(args.input, args.denoise, args.invert)
    if not images:
        print(f"No PDFs or PNGs found in {args.input}. Drop files there and re-run.")
        return

    ocr = PaddleOCR(
        lang="en",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    REVIEW_QUEUE_DIR.mkdir(parents=True, exist_ok=True)

    all_results = []
    for image_path in images:
        print(f"Processing {image_path.name}...")
        all_results.append(process_document(ocr, image_path))

    OUTPUT_JSON.write_text(json.dumps(all_results, indent=2))

    field_names = list(FIELDS.keys())
    with open(OUTPUT_CSV, "w", newline="") as f:
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
    print(f"Review crops: {REVIEW_QUEUE_DIR}")


if __name__ == "__main__":
    main()
