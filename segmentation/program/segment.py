"""Segment scanned ASME form PDFs into one-record-per-file PDFs, sorted by
form type, with continuation pages rotated back to right-side-up.

This version processes an explicit set of page ranges (SAMPLE_JOBS) so we
can produce example output for review before running the full corpus.
"""
import csv
import re
from pathlib import Path

import fitz
import numpy as np
import pytesseract
from PIL import Image

import lib

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "pdf"
OUT_DIR = ROOT / "segmented_form"

# OCR on these old, faded scans is too noisy for fuzzy text matching
# (short codes like "H-3" score well against random garbage under any
# fuzzy ratio). Only trust OCR when the exact code pattern is present;
# otherwise defer to the structural nearest-neighbour classifier.
FORM_PATTERNS = {
    "154-A": re.compile(r"\b154\s*-?\s*A\b"),
    "H-3": re.compile(r"\bH\s*-?\s*3\b"),
    "P-3": re.compile(r"\bP\s*-?\s*3\b"),
    "U-1A": re.compile(r"\bU\s*-?\s*1\s*A\b"),
    "P-2": re.compile(r"\bP\s*-?\s*2\b"),
}

# rotation to apply to a "continuation" page in each source file (front
# pages are always kept as scanned, rotation 0)
CONTINUATION_ROTATION = {
    "25.3.3632Dl0001-1.pdf": 180,
    "25.3.3632Dl0001-2.pdf": 180,
}
DEFAULT_CONTINUATION_ROTATION = 0

# manually verified (file, page) -> (role, label) used to seed the
# nearest-neighbour role/label classifier
EXEMPLARS = [
    ("25.3.3632Dl0001-1.pdf", 0, "front", "154-A"),
    ("25.3.3632Dl0001-1.pdf", 2, "front", "154-A"),
    ("25.3.3632Dl0001-1.pdf", 4, "front", "154-A"),
    ("25.3.3632Dl0001-1.pdf", 50, "front", "154-A"),
    ("25.3.3632Dl0001-1.pdf", 201, "front", "P-3"),
    ("25.3.3632Dl0001-1.pdf", 996, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 998, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 1001, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 1002, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 1004, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 343, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 348, "front", "P-3"),
    ("25.3.3632Dl0001-1.pdf", 350, "front", "P-3"),
    ("25.3.3632Dl0001-1.pdf", 685, "front", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 3083, "front", "H-3"),
    ("CHEMI-TROL REPORTS 1969.pdf", 0, "front", "U-1A"),
    ("CHEMI-TROL REPORTS 1969.pdf", 5, "front", "U-1A"),
    ("CHEMI-TROL REPORTS 1969.pdf", 10, "front", "U-1A"),
    ("CHEMI-TROL REPORTS 1969.pdf", 15, "front", "U-1A"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 0, "front", "P-2"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 6, "front", "P-2"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 10, "front", "P-2"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 20, "front", "P-2"),
    ("COASTAL IRON WKS REPORTS 1981.pdf", 0, "front", "U-1A"),
    ("COASTAL IRON WKS REPORTS 1981.pdf", 4, "front", "U-1A"),
    ("25.3.3632Dl0001-1.pdf", 1, "continuation", "154-A"),
    ("25.3.3632Dl0001-1.pdf", 3, "continuation", "154-A"),
    ("25.3.3632Dl0001-1.pdf", 995, "continuation", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 997, "continuation", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 999, "continuation", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 1000, "continuation", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 341, "continuation", "H-3"),
    ("25.3.3632Dl0001-1.pdf", 344, "continuation", "H-3"),
    ("COASTAL IRON WKS REPORTS 1981.pdf", 1, "continuation", "U-1A"),
    ("COASTAL IRON WKS REPORTS 1981.pdf", 5, "continuation", "U-1A"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 5, "continuation", "P-2"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 11, "continuation", "P-2"),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 21, "continuation", "P-2"),
]

# demo ranges for the sample run: (file, first_page, last_page_inclusive)
SAMPLE_JOBS = [
    ("25.3.3632Dl0001-1.pdf", 0, 9),
    ("25.3.3632Dl0001-1.pdf", 340, 355),
    ("25.3.3632Dl0001-1.pdf", 993, 1010),
    ("CHEMI-TROL REPORTS 1969.pdf", 0, 5),
    ("CLEAVER BROOKS REPORTS 1971.pdf", 0, 5),
    ("COASTAL IRON WKS REPORTS 1981.pdf", 0, 11),
]


def build_exemplar_bank():
    bank = []
    docs = {}
    for fname, pidx, role, label in EXEMPLARS:
        if fname not in docs:
            docs[fname] = fitz.open(PDF_DIR / fname)
        gray = lib.render_gray(docs[fname], pidx, zoom=1.3)
        bw = lib.binarize(gray)
        fp = lib.fingerprint(bw)
        bank.append(dict(fp=fp, role=role, label=label, file=fname, page=pidx))
    for d in docs.values():
        d.close()
    return bank


def classify_role(fp, bank, k=5):
    """Role (front/continuation) via distance-weighted vote over the k
    nearest exemplars. Plain (unweighted) majority vote lets 3 mediocre
    same-role matches outvote a single near-exact match of the other role,
    which happened in practice - weighting by inverse distance fixes that.
    """
    top = _distances(fp, bank)[:k]
    weight = {}
    for dist, ex in top:
        weight[ex["role"]] = weight.get(ex["role"], 0.0) + 1.0 / (dist + 1e-3)
    role = max(weight, key=weight.get)
    confidence = 1 - top[0][0]
    return role, confidence


def classify_label(fp, bank):
    """Label via single nearest front-exemplar. A majority vote over k
    neighbours would be biased toward whichever label happens to have
    the most exemplars, so use plain 1-NN among front exemplars instead.
    """
    front_bank = [e for e in bank if e["role"] == "front"]
    dists = _distances(fp, front_bank)
    return dists[0][1]["label"], 1 - dists[0][0]


def _distances(fp, bank):
    dists = []
    for ex in bank:
        cos = 1 - np.dot(fp, ex["fp"]) / (np.linalg.norm(fp) * np.linalg.norm(ex["fp"]) + 1e-9)
        dists.append((cos, ex))
    dists.sort(key=lambda x: x[0])
    return dists


def ocr_title(doc, page_index):
    page = doc[page_index]
    r = page.rect
    clip = fitz.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * 0.16)
    pix = page.get_pixmap(matrix=fitz.Matrix(4, 4), clip=clip)
    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    try:
        return pytesseract.image_to_string(img, config="--psm 6")
    except Exception:
        return ""


def match_form_code(text):
    """Exact-pattern match only - fuzzy ratio matching against short codes
    like "H-3" false-positives constantly on noisy OCR garbage, so we only
    trust a hit that actually contains the code's distinctive pattern.
    """
    if not text.strip():
        return None
    text = text.upper()
    hits = [label for label, pat in FORM_PATTERNS.items() if pat.search(text)]
    return hits[0] if len(hits) == 1 else None


def safe_label(label):
    return "form_" + label.lower().replace(" ", "_")


def process_job(fname, first, last, bank, manifest_rows):
    doc = fitz.open(PDF_DIR / fname)
    cont_rot = CONTINUATION_ROTATION.get(fname, DEFAULT_CONTINUATION_ROTATION)

    records = []  # list of dicts: label, source_conf, pages=[(idx, rotation)]
    current = None

    for pidx in range(first, min(last + 1, len(doc))):
        gray = lib.render_gray(doc, pidx, zoom=1.3)
        bw = lib.binarize(gray)
        fp = lib.fingerprint(bw)
        role, conf = classify_role(fp, bank)

        if role == "front":
            nn_label, nn_conf = classify_label(fp, bank)
            text = ocr_title(doc, pidx)
            ocr_label = match_form_code(text)
            label = ocr_label or nn_label
            label_source = "ocr_exact" if ocr_label else "nn"
            conf = nn_conf
            if current:
                records.append(current)
            current = dict(label=label, label_source=label_source, nn_conf=conf,
                            pages=[(pidx, 0)])
        else:  # continuation
            if current is None:
                current = dict(label="_unclassified", label_source="orphan_continuation",
                                nn_conf=conf, pages=[])
            current["pages"].append((pidx, cont_rot))

    if current:
        records.append(current)
    doc.close()

    # assemble output PDFs
    doc = fitz.open(PDF_DIR / fname)
    stem = Path(fname).stem
    for rec in records:
        folder_name = safe_label(rec["label"])
        out_folder = OUT_DIR / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)
        first_page = rec["pages"][0][0]
        out_name = f"{stem}_p{first_page + 1:05d}.pdf"
        out_path = out_folder / out_name

        new_doc = fitz.open()
        for src_idx, rot in rec["pages"]:
            new_doc.insert_pdf(doc, from_page=src_idx, to_page=src_idx)
            new_doc[-1].set_rotation(rot)
        new_doc.save(out_path)
        new_doc.close()

        page_range = f"{first_page + 1}-{rec['pages'][-1][0] + 1}"
        manifest_rows.append(dict(
            source_file=fname, source_pages=page_range, n_pages=len(rec["pages"]),
            form_type=rec["label"], label_source=rec["label_source"],
            nn_confidence=f'{rec["nn_conf"]:.2f}', output_path=str(out_path.relative_to(ROOT)),
        ))
    doc.close()


def main():
    OUT_DIR.mkdir(exist_ok=True)
    bank = build_exemplar_bank()
    manifest_rows = []
    for fname, first, last in SAMPLE_JOBS:
        print(f"processing {fname} pages {first}-{last}")
        process_job(fname, first, last, bank, manifest_rows)

    manifest_path = OUT_DIR / "manifest_sample.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {len(manifest_rows)} records to {manifest_path}")


if __name__ == "__main__":
    main()
