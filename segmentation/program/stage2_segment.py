"""Stage 2 (offline, no Claude/network needed).

Consumes discovery_full/page_clusters.pkl (every page's cluster assignment,
from stage1_discover.py) and discovery_full/labels_template.csv (filled in
by hand - role/form_code/rotation per cluster) and produces one PDF per
record, sorted into segmented_form/<form_code>/, with continuation pages
rotated back to right-side-up. Does not re-open PDFs for classification -
only to copy pages into the output files - so it's fast and needs no image
processing.

NOTE: tagging happens directly in labels_template.csv (no separate
labels.csv copy) - see stage1_discover.py's module docstring for the
guard that protects a filled-in labels_template.csv from being clobbered
by a later stage1 rerun.

Usage:
    python stage2_segment.py [--labels discovery_full/labels_template.csv]
                              [--clusters discovery_full/page_clusters.pkl]
                              [--pdf-dir ../../pdf] [--out ../../segmented_form]

Record-grouping rule: walking each source file's pages in order, a page
classified "front" starts a new record; consecutive "continuation" pages
attach to the record currently open; a page that's "ignore", or a
"continuation" with no record open yet (e.g. the file started mid-record),
is written out on its own into segmented_form/_unclassified/ rather than
silently dropped.
"""
import argparse
import csv
import pickle
from collections import defaultdict
from pathlib import Path

import fitz

# Continuation-page fingerprints are close to rotation-symmetric (a dense
# detail table looks similar either way up), so the per-cluster 'rotation'
# field in labels.csv is often left blank for those clusters. When that
# happens we fall back to this verified per-source-file convention instead
# of guessing from the page content. Established by manually reading many
# sample pages across both large files: every continuation page in them is
# scanned upside down relative to its front page; the small files need no
# correction at all.
DEFAULT_CONTINUATION_ROTATION = {
    "25.3.3632Dl0001-1.pdf": 180,
    "25.3.3632Dl0001-2.pdf": 180,
}


def load_labels(path):
    labels = {}
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            cid = int(row["cluster_id"])
            role = (row.get("role") or "").strip().lower()
            rotation = (row.get("rotation") or "").strip()
            labels[cid] = dict(
                role=role or "ignore",
                form_code=(row.get("form_code") or "").strip(),
                rotation=int(rotation) if rotation else None,
            )
    return labels


def load_page_clusters(path):
    with open(path, "rb") as f:
        records = pickle.load(f)
    by_page = defaultdict(dict)  # (file, page) -> {rot: record}
    for r in records:
        by_page[(r["file"], r["page"])][r["rot"]] = r
    return by_page


def resolve_page(entries, labels):
    """Decide (role, form_code, rotation_to_apply_to_raw_page) for one
    source page from its rot0/rot180 cluster memberships. Front-page
    fingerprints are strongly orientation-dependent (the wrong rotation of
    a front page matches a different, worse cluster), so whichever
    rotation lands closest to a front-labelled cluster wins; continuation
    pages are handled separately below since that signal doesn't exist
    for them.
    """
    candidates = []
    for rot, rec in entries.items():
        label = labels.get(rec["cluster"])
        if label is None:
            continue
        candidates.append((rot, rec, label))

    front = [c for c in candidates if c[2]["role"] == "front"]
    if front:
        rot, rec, label = min(front, key=lambda c: c[1]["dist"])
        extra = label["rotation"] if label["rotation"] is not None else 0
        return "front", (label["form_code"] or "unknown"), (rot + extra) % 360

    continuation = [c for c in candidates if c[2]["role"] == "continuation"]
    if continuation:
        rot, rec, label = min(continuation, key=lambda c: c[1]["dist"])
        # None (not 0) signals "no cluster-level rotation override" so the
        # caller can tell that apart from a genuine 0-degree override and
        # fall back to the per-source-file default instead.
        applied = (rot + label["rotation"]) % 360 if label["rotation"] is not None else None
        return "continuation", None, applied

    return "ignore", None, 0


def safe_label(label):
    return "form_" + label.lower().replace(" ", "_").replace("/", "-")


def process_file(fname, doc, by_page, labels, out_dir, manifest_rows):
    cont_rotation_default = DEFAULT_CONTINUATION_ROTATION.get(fname, 0)

    records = []
    current = None
    for pidx in range(len(doc)):
        entries = by_page.get((fname, pidx))
        if not entries:
            role, form_code, rotation = "ignore", None, 0
        else:
            role, form_code, rotation = resolve_page(entries, labels)
            if role == "continuation" and rotation is None:
                rotation = cont_rotation_default

        if role == "front":
            if current:
                records.append(current)
            current = dict(label=form_code, pages=[(pidx, rotation)])
        elif role == "continuation" and current:
            current["pages"].append((pidx, rotation))
        else:  # 'ignore', or an orphan continuation with nothing open yet
            if current:
                records.append(current)
            current = None
            records.append(dict(label="_unclassified", pages=[(pidx, rotation)]))

    if current:
        records.append(current)

    stem = Path(fname).stem
    for rec in records:
        folder = out_dir / safe_label(rec["label"])
        folder.mkdir(parents=True, exist_ok=True)
        first_page = rec["pages"][0][0]
        out_path = folder / f"{stem}_p{first_page + 1:05d}.pdf"

        new_doc = fitz.open()
        for src_idx, rot in rec["pages"]:
            new_doc.insert_pdf(doc, from_page=src_idx, to_page=src_idx)
            new_doc[-1].set_rotation(rot)
        new_doc.save(out_path)
        new_doc.close()

        manifest_rows.append(dict(
            source_file=fname,
            source_pages=f"{first_page + 1}-{rec['pages'][-1][0] + 1}",
            n_pages=len(rec["pages"]),
            form_type=rec["label"],
            output_path=str(out_path),
        ))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(here / "discovery_full" / "labels_template.csv"))
    ap.add_argument("--clusters", default=str(here / "discovery_full" / "page_clusters.pkl"))
    ap.add_argument("--pdf-dir", default=str(here.parents[1] / "pdf"))
    ap.add_argument("--out", default=str(here.parents[1] / "segmented_form"))
    args = ap.parse_args()

    labels = load_labels(args.labels)
    missing = [cid for cid, lab in labels.items()
               if lab["role"] == "ignore" and not lab["form_code"]]
    if missing:
        print(f"WARNING: {len(missing)} clusters have no role set "
              f"(treated as 'ignore', pages routed to _unclassified): {missing}")

    by_page = load_page_clusters(args.clusters)
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    manifest_rows = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"processing {pdf_path.name}")
        doc = fitz.open(pdf_path)
        process_file(pdf_path.name, doc, by_page, labels, out_dir, manifest_rows)
        doc.close()

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {len(manifest_rows)} records to {manifest_path}")


if __name__ == "__main__":
    main()
