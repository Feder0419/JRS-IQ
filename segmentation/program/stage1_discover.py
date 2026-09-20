"""Stage 1 (offline, run once per corpus of PDFs).

Scans every page of every PDF in pdf/, groups them by structural layout
(box/rule positions - not OCR text, since these old scans are too degraded
for reliable text matching), and writes out, for a human to review with no
Claude/network involvement:

  discovery_full/
    clusters/cluster_000.png ...      - representative example pages per cluster
    labels_template.csv               - one row per cluster, ready to fill in
    page_clusters.pkl                 - every page's cluster assignment (for stage 2)

Fill in labels_template.csv (role / form_code / rotation per cluster, see
the "role" column doc below), save it as labels.csv in the same folder, then
run stage2_segment.py - that step needs no PDF re-reading of the fingerprint
step, it just joins page_clusters.pkl against your labels.

Usage:
    python stage1_discover.py [--k 80] [--pdf-dir ../../pdf] [--out discovery_full]

labels.csv columns to fill in by hand, after opening each cluster's montage:
  role        "front" (starts a new record - has its own identity/company/
              serial fields), "continuation" (belongs to whatever front page
              precedes it - a schedule/detail table with no header of its
              own), or "ignore" (blank/junk/unreadable - not part of any
              record). Leave blank -> treated as "ignore" (and a warning is
              printed by stage2 so nothing is silently dropped).
  form_code   Only for role=front. Whatever short code identifies this
              template, e.g. "H-3", "P-3", "154-A", "U-1A", "P-2" - read it
              off the montage image, or invent a label like "unknown-1" if
              you can't make it out and just want a placeholder folder name.
  rotation    Degrees (0/90/180/270) needed to make the montage's examples
              upright, IF the montage consistently shows one orientation.
              Leave blank if unsure or if the cluster's examples show mixed
              orientations (this is expected for "continuation" clusters -
              dense tables look similar whichever way up, so the fingerprint
              can't tell them apart; stage2 falls back to a per-source-file
              default rotation for continuation pages instead).
  notes       Free text, optional.
"""
import argparse
import csv
import pickle
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw
from sklearn.cluster import MiniBatchKMeans

import lib

DEFAULT_K = 80
MONTAGE_CLOSEST = 6
MONTAGE_FARTHEST = 3
THUMB_SIZE = (150, 190)


class DocCache:
    """Keeps every source PDF open for the duration of the run instead of
    re-opening per page/thumbnail - matters for the multi-GB source files.
    """

    def __init__(self, pdf_dir: Path):
        self.pdf_dir = pdf_dir
        self._docs = {}

    def get(self, fname: str) -> fitz.Document:
        if fname not in self._docs:
            self._docs[fname] = fitz.open(self.pdf_dir / fname)
        return self._docs[fname]

    def close_all(self):
        for d in self._docs.values():
            d.close()
        self._docs.clear()


def build_fingerprints(pdf_dir: Path, docs: DocCache, progress_every=200):
    pages = list(lib.iter_pdf_pages(pdf_dir))
    print(f"found {len(pages)} pages across "
          f"{len(set(f for f, _ in pages))} files")

    records = []
    for i, (fname, pidx) in enumerate(pages, 1):
        doc = docs.get(fname)
        gray = lib.render_gray(doc, pidx, zoom=1.3)
        bw = lib.binarize(gray)
        fp0 = lib.fingerprint(bw)
        bw180 = lib.rotate_bw(bw, 2)
        fp180 = lib.fingerprint(bw180)
        records.append(dict(file=fname, page=pidx, rot=0, fp=fp0))
        records.append(dict(file=fname, page=pidx, rot=180, fp=fp180))
        if i % progress_every == 0:
            print(f"  fingerprinted {i}/{len(pages)}")
    print(f"done fingerprinting {len(pages)} pages "
          f"({len(records)} vectors incl. both rotations)")
    return records


def cluster_records(records, k):
    X = np.stack([r["fp"] for r in records]).astype(np.float32)
    print(f"clustering {len(records)} vectors into {k} clusters (MiniBatchKMeans)...")
    km = MiniBatchKMeans(n_clusters=k, random_state=0, n_init=10, batch_size=1024)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_[labels]
    dists = np.linalg.norm(X - centers, axis=1)
    for r, lab, d in zip(records, labels, dists):
        r["cluster"] = int(lab)
        r["dist"] = float(d)


def render_thumb(docs: DocCache, fname, pidx, rot):
    doc = docs.get(fname)
    gray = lib.render_gray(doc, pidx, zoom=1.0)
    img = Image.fromarray(gray)
    if rot:
        img = img.rotate(rot, expand=True)
    return img.resize(THUMB_SIZE)


def make_montage(docs: DocCache, members, out_path):
    members_sorted = sorted(members, key=lambda r: r["dist"])
    sample = members_sorted[:MONTAGE_CLOSEST]
    if len(members_sorted) > MONTAGE_CLOSEST:
        tail = members_sorted[MONTAGE_CLOSEST:]
        sample = sample + tail[-MONTAGE_FARTHEST:]

    w, h = THUMB_SIZE
    cols = min(4, len(sample))
    rows = (len(sample) + cols - 1) // cols
    sheet = Image.new("RGB", (w * cols, (h + 16) * rows), "white")
    draw = ImageDraw.Draw(sheet)
    for i, m in enumerate(sample):
        thumb = render_thumb(docs, m["file"], m["page"], m["rot"])
        x = (i % cols) * w
        y = (i // cols) * (h + 16)
        sheet.paste(thumb, (x, y + 16))
        label = f'{m["file"][:14]} p{m["page"] + 1} rot{m["rot"]} d{m["dist"]:.2f}'
        draw.rectangle([x, y, x + w, y + 16], fill="yellow")
        draw.text((x + 2, y + 2), label, fill="black")
    sheet.save(out_path)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="number of clusters")
    ap.add_argument("--pdf-dir", default=str(here.parents[1] / "pdf"))
    ap.add_argument("--out", default=str(here / "discovery_full"))
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    docs = DocCache(pdf_dir)
    try:
        records = build_fingerprints(pdf_dir, docs)
        cluster_records(records, args.k)

        # persist per-page assignment for stage2 - drop the heavy fp vectors,
        # stage2 never needs to re-render or re-fingerprint anything.
        slim = [{k: v for k, v in r.items() if k != "fp"} for r in records]
        with open(out_dir / "page_clusters.pkl", "wb") as f:
            pickle.dump(slim, f)

        by_cluster = {}
        for r in records:
            by_cluster.setdefault(r["cluster"], []).append(r)

        rows = []
        for cid in sorted(by_cluster):
            members = by_cluster[cid]
            montage_name = f"cluster_{cid:03d}.png"
            make_montage(docs, members, clusters_dir / montage_name)
            files_present = sorted(set(m["file"] for m in members))
            rows.append(dict(
                cluster_id=cid,
                n_members=len(members),
                files=";".join(files_present)[:150],
                montage_file=f"clusters/{montage_name}",
                role="", form_code="", rotation="", notes="",
            ))
            print(f"  cluster {cid}: {len(members)} members -> {montage_name}")
    finally:
        docs.close_all()

    template_path = out_dir / "labels_template.csv"
    with open(template_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} clusters to {template_path}")
    print(f"montages in {clusters_dir}")
    print("Next: open each clusters/cluster_XXX.png, fill in role/form_code/rotation "
          "for every row (see column docs in this script's module docstring), "
          f"save as {out_dir / 'labels.csv'}, then run stage2_segment.py")


if __name__ == "__main__":
    main()
