"""Stage 1: sample pages from all PDFs, build layout fingerprints, cluster
them into template groups, and dump a visual report for manual labeling.
"""
import pickle
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import AgglomerativeClustering

import lib

ROOT = Path(__file__).resolve().parents[2]
PDF_DIR = ROOT / "pdf"
OUT_DIR = Path(__file__).resolve().parent / "discovery"
OUT_DIR.mkdir(exist_ok=True)

BIG_FILES = ["25.3.3632Dl0001-1.pdf", "25.3.3632Dl0001-2.pdf"]
SMALL_FILES = [
    "CHEMI-TROL REPORTS 1969.pdf",
    "CLEAVER BROOKS REPORTS 1971.pdf",
    "COASTAL IRON WKS REPORTS 1981.pdf",
]

BLOCK_SIZE = 14
NUM_BLOCKS = 15


def block_starts(n_pages: int, block_size: int, num_blocks: int):
    if n_pages <= block_size:
        return [0]
    span = n_pages - block_size
    return sorted(set(int(i * span / (num_blocks - 1)) for i in range(num_blocks)))


def collect_pages():
    """Yields (file_name, page_index) for every page we'll sample."""
    for fname in BIG_FILES:
        doc = fitz.open(PDF_DIR / fname)
        n = len(doc)
        for start in block_starts(n, BLOCK_SIZE, NUM_BLOCKS):
            for p in range(start, min(start + BLOCK_SIZE, n)):
                yield fname, p
        doc.close()
    for fname in SMALL_FILES:
        doc = fitz.open(PDF_DIR / fname)
        n = len(doc)
        for p in range(n):
            yield fname, p
        doc.close()


def main():
    targets = list(collect_pages())
    print(f"sampling {len(targets)} pages")

    records = []  # one per (file, page)
    thumbs = {}  # (file, page) -> small PIL image (as scanned, 0 rotation)
    open_docs = {}

    for i, (fname, pidx) in enumerate(targets):
        if fname not in open_docs:
            open_docs[fname] = fitz.open(PDF_DIR / fname)
        doc = open_docs[fname]
        gray = lib.render_gray(doc, pidx, zoom=1.3)
        bw = lib.binarize(gray)

        sideways = lib.looks_sideways(bw)
        fp0 = lib.fingerprint(bw)
        bw180 = lib.rotate_bw(bw, 2)
        fp180 = lib.fingerprint(bw180)

        records.append(dict(file=fname, page=pidx, rot=0, fp=fp0, sideways=sideways))
        records.append(dict(file=fname, page=pidx, rot=180, fp=fp180, sideways=sideways))

        thumb = Image.fromarray(255 - gray).resize((150, 190))
        thumb = Image.fromarray(255 - np.array(thumb))
        thumbs[(fname, pidx)] = thumb

        if (i + 1) % 50 == 0:
            print(f"  processed {i+1}/{len(targets)}")

    for d in open_docs.values():
        d.close()

    with open(OUT_DIR / "records.pkl", "wb") as f:
        pickle.dump({"records": records, "thumbs": thumbs}, f)

    print(f"done rendering. sideways-flagged pages: {sum(r['sideways'] for r in records)//2}")
    cluster(records, thumbs)


def cluster(records, thumbs, distance_threshold=0.35):
    X = np.stack([r["fp"] for r in records])
    clu = AgglomerativeClustering(
        n_clusters=None, distance_threshold=distance_threshold,
        linkage="average", metric="cosine",
    )
    labels = clu.fit_predict(X)
    n_clusters = labels.max() + 1
    print(f"found {n_clusters} clusters (threshold={distance_threshold})")

    for r, lab in zip(records, labels):
        r["cluster"] = int(lab)

    sizes = np.bincount(labels)
    order = np.argsort(-sizes)

    report_path = OUT_DIR / "cluster_report.txt"
    with open(report_path, "w", encoding="utf-8") as rep:
        for lab in order:
            members = [r for r in records if r["cluster"] == lab]
            files = sorted(set(m["file"] for m in members))
            rots = sorted(set(m["rot"] for m in members))
            rep.write(f"cluster {lab}: {len(members)} members, files={files}, rots={rots}\n")
            make_montage(members, thumbs, OUT_DIR / f"cluster_{lab:02d}.png")
    print(f"wrote {report_path} and per-cluster montages in {OUT_DIR}")


def make_montage(members, thumbs, out_path, max_items=8):
    sample = members[:max_items]
    w, h = 150, 190
    cols = min(4, len(sample))
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, (h + 16) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, m in enumerate(sample):
        thumb = thumbs[(m["file"], m["page"])]
        if m["rot"] == 180:
            thumb = thumb.rotate(180)
        x = (i % cols) * w
        y = (i // cols) * (h + 16)
        sheet.paste(thumb, (x, y + 16))
        label = f'{m["file"][:10]} p{m["page"]} r{m["rot"]}'
        draw.rectangle([x, y, x + w, y + 16], fill="yellow")
        draw.text((x + 2, y + 2), label, fill="black")
    sheet.save(out_path)


if __name__ == "__main__":
    main()
