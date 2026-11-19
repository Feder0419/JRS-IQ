"""OCR the title strip of every front page, for stage2_segment.py.

Uses the same page classification as stage 2 (cluster labels + page
overrides) to find front pages and the rotation they will be written out
at, renders the top TITLE_FRAC of each one upright, runs Tesseract on it
and caches the raw text in discovery_full/front_ocr.csv. Stage 2 turns that
text into a form code (see OCR_RULES there), so the matching rules can be
tuned without re-running OCR.

Incremental: pages already in the cache at the same rotation are skipped,
so after relabelling clusters just rerun this and only new fronts are OCR'd.

Usage:
    python ocr_titles.py [--workers 16]
"""
import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import fitz
import pytesseract
from PIL import Image

import stage2_segment as s2

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TITLE_FRAC = 0.22  # H-3 titles sit under a REP/MODEL line, ~10% down the page
ZOOM = 3

FIELDS = ["source_file", "page", "rotation", "text"]


def ocr_pages(job):
    pdf_path, items = job
    pytesseract.pytesseract.tesseract_cmd = TESSERACT
    doc = fitz.open(pdf_path)
    out = []
    for pidx, rot in items:
        pix = doc[pidx].get_pixmap(matrix=fitz.Matrix(ZOOM, ZOOM).prerotate(rot))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        img = img.crop((0, 0, img.width, int(img.height * TITLE_FRAC)))
        try:
            text = pytesseract.image_to_string(img, config="--psm 6")
        except Exception as e:
            text = f"<ocr error: {e}>"
        out.append(dict(source_file=Path(pdf_path).name, page=pidx + 1, rotation=rot,
                        text=" ".join(text.split())))
    doc.close()
    return out


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(here / "discovery_full" / "labels_template.csv"))
    ap.add_argument("--clusters", default=str(here / "discovery_full" / "page_clusters.pkl"))
    ap.add_argument("--overrides", default=str(here / "discovery_full" / "page_overrides.csv"))
    ap.add_argument("--out", default=str(here / "discovery_full" / "front_ocr.csv"))
    ap.add_argument("--pdf-dir", default=str(here.parents[1] / "pdf"))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--chunk", type=int, default=50, help="pages per worker task")
    args = ap.parse_args()

    labels = s2.load_labels(args.labels)
    overrides = s2.load_overrides(args.overrides)
    by_page = s2.load_page_clusters(args.clusters)
    cached = s2.load_ocr(args.out)

    jobs, n_todo = [], 0
    for pdf_path in sorted(Path(args.pdf_dir).glob("*.pdf")):
        with fitz.open(pdf_path) as doc:
            n = len(doc)
        pages = s2.classify_pages(pdf_path.name, n, by_page, labels, overrides)
        todo = [(i, p["rotation"]) for i, p in enumerate(pages)
                if p["role"] == "front" and cached.get((pdf_path.name, i), (None,))[0] != p["rotation"]]
        n_todo += len(todo)
        for k in range(0, len(todo), args.chunk):
            jobs.append((str(pdf_path), todo[k:k + args.chunk]))
    print(f"{len(cached)} pages cached, {n_todo} front pages to OCR with {args.workers} workers")

    rows = {(f, p): dict(source_file=f, page=p + 1, rotation=rot, text=text)
            for (f, p), (rot, text) in cached.items()}
    done = 0
    with ProcessPoolExecutor(args.workers) as pool:
        for result in pool.map(ocr_pages, jobs):
            for r in result:
                rows[(r["source_file"], r["page"] - 1)] = r
            done += len(result)
            print(f"  {done}/{n_todo}", flush=True)

    with open(args.out, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows[k] for k in sorted(rows))
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
