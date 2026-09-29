"""Stage 1 (offline, run once per corpus of PDFs).

Scans every page of every PDF in pdf/, groups them by structural layout
(box/rule positions - not OCR text, since these old scans are too degraded
for reliable text matching), and writes out, for a human to review with no
Claude/network involvement:

  discovery_full/
    clusters/cluster_000/thumbnail.png  - low-res grid of sample pages, for
                                           quickly scanning all 80 clusters
    clusters/cluster_000/full/*.png     - same samples, full resolution, for
                                           actually reading box labels/text
    ... one such folder per cluster ...
    labels_template.csv               - one row per cluster, ready to fill in
    page_clusters.pkl                 - every page's cluster assignment (for stage 2)

Fill in labels_template.csv directly (role / form_code / rotation per
cluster, see the "role" column doc below) - there is no separate labels.csv
copy, this file IS the one stage2_segment.py reads. Once filled in, run
stage2_segment.py - that step needs no PDF re-reading of the fingerprint
step, it just joins page_clusters.pkl against your labels.

Because labels_template.csv is now the live, hand-edited file, rerunning
this script (e.g. with --from-cache to pick up a new --full-zoom) will NOT
overwrite it once you've started filling it in: if any row already has a
non-blank "role", the fresh output is written to labels_template.NEW.csv
instead, with a warning, so your tagging work is never silently clobbered.

Usage:
    python stage1_discover.py [--k 80] [--pdf-dir ../../pdf] [--out discovery_full]
                               [--full-zoom 2.5]

    Add --from-cache to skip re-fingerprinting/re-clustering and just
    regenerate clusters/ (thumbnails + full images) from an existing
    page_clusters.pkl in --out - much faster than a full rerun, e.g. after
    changing --full-zoom or wanting fresh images without reprocessing every
    PDF page again:
        python stage1_discover.py --from-cache --full-zoom 3.0

labels_template.csv columns to fill in by hand, after opening each cluster's
folder (thumbnail.png for a quick look, full/ for pages you can actually read):
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
SAMPLE_CLOSEST = 6
SAMPLE_FARTHEST = 3
THUMB_SIZE = (150, 190)
FULL_ZOOM = 2.5  # render zoom for the per-page full-resolution images


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


def select_samples(members):
    """Pick a representative subset of a cluster's members: the
    SAMPLE_CLOSEST examples nearest the cluster center (typical pages) plus
    the SAMPLE_FARTHEST examples furthest away (edge cases worth a second
    look), sorted closest-first. Used for both the thumbnail grid and the
    full-resolution images, in the same order, so the two views line up.
    """
    members_sorted = sorted(members, key=lambda r: r["dist"])
    sample = members_sorted[:SAMPLE_CLOSEST]
    if len(members_sorted) > SAMPLE_CLOSEST:
        tail = members_sorted[SAMPLE_CLOSEST:]
        sample = sample + tail[-SAMPLE_FARTHEST:]
    return sample


def make_montage(docs: DocCache, sample, out_path):
    """Small grid of all sampled pages for quickly scanning many clusters at
    once - too low-res to read box labels/handwriting off, see the sibling
    full/ folder for that.
    """
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
        label = f'{i + 1:02d} {m["file"][:14]} p{m["page"] + 1} rot{m["rot"]} d{m["dist"]:.2f}'
        draw.rectangle([x, y, x + w, y + 16], fill="yellow")
        draw.text((x + 2, y + 2), label, fill="black")
    sheet.save(out_path)


def save_full_images(docs: DocCache, sample, out_dir: Path, zoom: float):
    """Full-resolution render of each sampled page as its own file, so it's
    actually legible. Filenames repeat the same index/source/page/rotation/
    distance info as the thumbnail grid's labels (same closest-first order),
    so a tagger can cross-reference either view.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, m in enumerate(sample, 1):
        doc = docs.get(m["file"])
        gray = lib.render_gray(doc, m["page"], zoom=zoom)
        img = Image.fromarray(gray)
        if m["rot"]:
            img = img.rotate(m["rot"], expand=True)
        stem = Path(m["file"]).stem[:40]
        fname = f'{i:02d}_{stem}_p{m["page"] + 1:05d}_rot{m["rot"]}_d{m["dist"]:.2f}.png'
        img.save(out_dir / fname)


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="number of clusters")
    ap.add_argument("--pdf-dir", default=str(here.parents[1] / "pdf"))
    ap.add_argument("--out", default=str(here / "discovery_full"))
    ap.add_argument("--full-zoom", type=float, default=FULL_ZOOM,
                     help="render zoom for the full-resolution per-page images")
    ap.add_argument("--from-cache", action="store_true",
                     help="skip re-fingerprinting/re-clustering and reuse the "
                          "page_clusters.pkl already in --out - fast path for "
                          "regenerating clusters/ images only (e.g. after "
                          "changing --full-zoom)")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)
    clusters_dir = out_dir / "clusters"
    clusters_dir.mkdir(parents=True, exist_ok=True)

    docs = DocCache(pdf_dir)
    try:
        if args.from_cache:
            cache_path = out_dir / "page_clusters.pkl"
            print(f"--from-cache: loading {cache_path}, skipping fingerprint/cluster steps")
            with open(cache_path, "rb") as f:
                records = pickle.load(f)
        else:
            records = build_fingerprints(pdf_dir, docs)
            cluster_records(records, args.k)

            # persist per-page assignment for stage2 - drop the heavy fp
            # vectors, stage2 never needs to re-render or re-fingerprint
            # anything, and --from-cache reruns of this script don't either.
            slim = [{k: v for k, v in r.items() if k != "fp"} for r in records]
            with open(out_dir / "page_clusters.pkl", "wb") as f:
                pickle.dump(slim, f)

        by_cluster = {}
        for r in records:
            by_cluster.setdefault(r["cluster"], []).append(r)

        rows = []
        for cid in sorted(by_cluster):
            members = by_cluster[cid]
            sample = select_samples(members)
            cluster_dir = clusters_dir / f"cluster_{cid:03d}"
            cluster_dir.mkdir(parents=True, exist_ok=True)
            make_montage(docs, sample, cluster_dir / "thumbnail.png")
            save_full_images(docs, sample, cluster_dir / "full", zoom=args.full_zoom)
            files_present = sorted(set(m["file"] for m in members))
            rows.append(dict(
                cluster_id=cid,
                n_members=len(members),
                files=";".join(files_present)[:150],
                montage_file=f"clusters/cluster_{cid:03d}/thumbnail.png",
                role="", form_code="", rotation="", notes="",
            ))
            print(f"  cluster {cid}: {len(members)} members -> "
                  f"cluster_{cid:03d}/ (thumbnail.png + full/{len(sample)} pages)")
    finally:
        docs.close_all()

    template_path = out_dir / "labels_template.csv"
    already_tagged = False
    if template_path.exists():
        with open(template_path, encoding="utf-8-sig") as f:
            already_tagged = any((row.get("role") or "").strip() for row in csv.DictReader(f))
    if already_tagged:
        template_path = out_dir / "labels_template.NEW.csv"
        print(f"\nWARNING: existing labels_template.csv already has tagged rows - "
              f"not overwriting it. Fresh cluster rows written to {template_path} "
              "instead; merge in any new/changed clusters by hand if needed.")

    with open(template_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nwrote {len(rows)} clusters to {template_path}")
    print(f"cluster folders in {clusters_dir}")
    print("Next: open each clusters/cluster_XXX/ folder (thumbnail.png for a quick "
          "look, full/ for legible full-resolution pages) and fill in "
          "role/form_code/rotation directly in labels_template.csv for every row "
          "(see column docs in this script's module docstring), then run stage2_segment.py")


if __name__ == "__main__":
    main()
