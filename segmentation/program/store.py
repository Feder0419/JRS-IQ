"""Paths and I/O for the persistent page store (segmentation/program/store/).

Every page ever ingested keeps its fingerprints and OCR here, so later steps
(and pages imported later) never need to re-render old PDFs. Pages are
0-based in code and 1-based in every CSV a human reads.
"""
import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STORE = HERE / "store"
PDF_DIR = HERE.parents[1] / "pdf"

PAGES_CSV = STORE / "pages.csv"            # per-page OCR summary
OCR_JSONL = STORE / "ocr.jsonl"            # per-page, per-orientation OCR text
FP_NPZ = STORE / "fingerprints.npz"        # per-page fingerprints, both orientations
LABELS_CSV = STORE / "page_labels.csv"     # page-level label library (truth)
LABEL_FIXES_CSV = STORE / "label_fixes.csv"
ROLES_CSV = STORE / "roles.csv"            # role + rotation decided for every page
ROLE_REVIEW_CSV = STORE / "role_review.csv"
FRONT_DIR = STORE / "front_clusters"
FRONT_CSV = STORE / "front_clusters.csv"
FRONT_PAGES_CSV = STORE / "front_pages.csv"
FRONT_MODEL = STORE / "front_model.pkl"

FP_KINDS = ("full0", "full180", "hdr0", "hdr180")


def key(file, page):
    return f"{file}|{page}"


def read_csv(path):
    path = Path(path)
    if not path.exists():
        return []
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def load_fingerprints():
    """-> (keys list, {kind: float32 array aligned with keys})"""
    if not FP_NPZ.exists():
        return [], {}
    with np.load(FP_NPZ, allow_pickle=False) as z:
        keys = list(z["keys"])
        arrays = {k: z[k].astype(np.float32) for k in FP_KINDS}
    return keys, arrays


def save_fingerprints(keys, arrays):
    STORE.mkdir(parents=True, exist_ok=True)
    tmp = FP_NPZ.with_name("fingerprints.tmp.npz")
    np.savez_compressed(tmp, keys=np.array(keys), **{k: arrays[k].astype(np.float16) for k in FP_KINDS})
    tmp.replace(FP_NPZ)


def load_pages():
    """(file, 0-based page) -> pages.csv row with numeric fields parsed"""
    out = {}
    for r in read_csv(PAGES_CSV):
        r["page0"] = int(r["page"]) - 1
        for k in ("ink", "ocr_score0", "ocr_score180", "ocr_margin"):
            r[k] = float(r[k])
        r["ocr_rot"] = int(r["ocr_rot"])
        out[(r["file"], r["page0"])] = r
    return out


def load_ocr_text():
    """(file, 0-based page, rotation) -> {"title": str, "text": str}"""
    out = {}
    if OCR_JSONL.exists():
        with open(OCR_JSONL, encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                out[(d["file"], d["page"] - 1, d["rot"])] = d
    return out


def load_labels():
    """(file, 0-based page) -> page_labels.csv row"""
    out = {}
    for r in read_csv(LABELS_CSV):
        r["rotation"] = int(r["rotation"]) if r["rotation"] else None
        out[(r["file"], int(r["page"]) - 1)] = r
    return out


def load_roles():
    out = {}
    for r in read_csv(ROLES_CSV):
        r["rotation"] = int(r["rotation"])
        out[(r["file"], int(r["page"]) - 1)] = r
    return out


def pdf_page_counts(pdf_dir=PDF_DIR):
    import fitz
    counts = {}
    for p in sorted(Path(pdf_dir).glob("*.pdf")):
        with fitz.open(p) as doc:
            counts[p.name] = len(doc)
    return counts
