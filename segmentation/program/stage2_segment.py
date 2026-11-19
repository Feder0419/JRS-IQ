"""Stage 2 (offline, no Claude/network needed).

Consumes discovery_full/page_clusters.pkl (every page's cluster assignment,
from stage1_discover.py) and discovery_full/labels_template.csv (filled in
by hand - role/form_code/rotation per cluster) and produces one PDF per
record, sorted into segmented_form/form_<code>/, with continuation pages
rotated back to right-side-up. Does not re-open PDFs for classification -
only to copy pages into the output files - so it's fast and needs no image
processing.

Two optional inputs refine the cluster labels:

  discovery_full/page_overrides.csv - hand-written per-file / per-page-range
      rules that take priority over the cluster labels. Columns:
        source_file       PDF file name
        first_page        1-based, inclusive; blank = start of file
        last_page         1-based, inclusive; blank = end of file
        role              front / continuation / ignore; blank = keep cluster role
        form_code         e.g. U-1A; blank = keep OCR / cluster code
        rotation          final rotation for every page in range; blank = computed
        pages_per_record  N: every N-th page from first_page is a front, the
                          others continuations (overrides the role column)
        notes             free text
      Later rows win over earlier rows where ranges overlap.

  discovery_full/front_ocr.csv - title-strip OCR of every front page, written
      by ocr_titles.py. Clusters group pages by box/rule layout only, so forms
      with the same layout but a different title (H-3 vs P-3) end up in the
      same cluster; the OCR text tells them apart. A front page's form code is
      taken from (highest priority first): override, OCR keywords, cluster.

NOTE: tagging happens directly in labels_template.csv (no separate
labels.csv copy) - see stage1_discover.py's module docstring for the
guard that protects a filled-in labels_template.csv from being clobbered
by a later stage1 rerun.

Usage:
    python stage2_segment.py [--labels discovery_full/labels_template.csv]
                              [--clusters discovery_full/page_clusters.pkl]
                              [--overrides discovery_full/page_overrides.csv]
                              [--ocr discovery_full/front_ocr.csv]
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
from collections import Counter, defaultdict
from pathlib import Path

import fitz

from forms import canon_code, ocr_form_code, safe_label

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

ROLES = {"front", "continuation", "ignore"}


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


def load_overrides(path):
    path = Path(path)
    if not path.exists():
        return []
    rules = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            get = lambda k: (row.get(k) or "").strip()
            role = get("role").lower()
            if role and role not in ROLES:
                raise ValueError(f"{path}: bad role {role!r} in row {row}")
            rules.append(dict(
                file=get("source_file"),
                first=int(get("first_page")) - 1 if get("first_page") else 0,
                last=int(get("last_page")) - 1 if get("last_page") else None,
                role=role or None,
                form_code=canon_code(get("form_code")) if get("form_code") else None,
                rotation=int(get("rotation")) if get("rotation") else None,
                pages_per_record=int(get("pages_per_record")) if get("pages_per_record") else None,
            ))
    return rules


def load_ocr(path):
    """(file, 0-based page) -> (rotation the OCR was done at, text)"""
    path = Path(path)
    if not path.exists():
        return {}
    with open(path, encoding="utf-8-sig") as f:
        return {(r["source_file"], int(r["page"]) - 1): (int(r["rotation"]), r["text"])
                for r in csv.DictReader(f)}


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


def classify_pages(fname, n_pages, by_page, labels, overrides):
    """Per page: dict(role, cluster_code, override_code, rotation)."""
    cont_rotation_default = DEFAULT_CONTINUATION_ROTATION.get(fname, 0)
    pages = []
    for pidx in range(n_pages):
        entries = by_page.get((fname, pidx))
        role, code, rotation = resolve_page(entries, labels) if entries else ("ignore", None, 0)
        pages.append(dict(role=role, cluster_code=code and canon_code(code),
                          override_code=None, rotation=rotation))

    for rule in overrides:
        if rule["file"] != fname:
            continue
        last = n_pages - 1 if rule["last"] is None else min(rule["last"], n_pages - 1)
        for pidx in range(rule["first"], last + 1):
            p = pages[pidx]
            if rule["pages_per_record"]:
                p["role"] = "front" if (pidx - rule["first"]) % rule["pages_per_record"] == 0 else "continuation"
            elif rule["role"]:
                p["role"] = rule["role"]
            if rule["form_code"]:
                p["override_code"] = rule["form_code"]
            if rule["rotation"] is not None:
                p["rotation"] = rule["rotation"]

    for p in pages:
        if p["rotation"] is None:
            p["rotation"] = cont_rotation_default if p["role"] == "continuation" else 0
    return pages


def front_label(fname, pidx, page, ocr):
    """(form code, which source it came from, ocr code) for a front page."""
    ocr_code = None
    hit = ocr.get((fname, pidx))
    if hit and hit[0] == page["rotation"]:
        ocr_code = ocr_form_code(hit[1])
    if page["override_code"]:
        return page["override_code"], "override", ocr_code
    if ocr_code:
        return ocr_code, "ocr", ocr_code
    return (page["cluster_code"] or "unknown"), ("cluster" if hit else "cluster_no_ocr"), ocr_code


def process_file(fname, doc, by_page, labels, overrides, ocr, out_dir, manifest_rows):
    pages = classify_pages(fname, len(doc), by_page, labels, overrides)

    records = []
    current = None
    for pidx, page in enumerate(pages):
        role, rotation = page["role"], page["rotation"]
        if role == "front":
            if current:
                records.append(current)
            label, source, ocr_code = front_label(fname, pidx, page, ocr)
            current = dict(label=label, label_source=source, cluster_code=page["cluster_code"],
                           ocr_code=ocr_code, pages=[(pidx, rotation)])
        elif role == "continuation" and current:
            current["pages"].append((pidx, rotation))
        else:  # 'ignore', or an orphan continuation with nothing open yet
            if current:
                records.append(current)
            current = None
            records.append(dict(label="_unclassified",
                                label_source="orphan_continuation" if role == "continuation" else "ignore",
                                cluster_code=None, ocr_code=None, pages=[(pidx, rotation)]))

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
            label_source=rec["label_source"],
            cluster_code=rec["cluster_code"] or "",
            ocr_code=rec["ocr_code"] or "",
            output_path=str(out_path),
        ))


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(here / "discovery_full" / "labels_template.csv"))
    ap.add_argument("--clusters", default=str(here / "discovery_full" / "page_clusters.pkl"))
    ap.add_argument("--overrides", default=str(here / "discovery_full" / "page_overrides.csv"))
    ap.add_argument("--ocr", default=str(here / "discovery_full" / "front_ocr.csv"))
    ap.add_argument("--pdf-dir", default=str(here.parents[1] / "pdf"))
    ap.add_argument("--out", default=str(here.parents[1] / "segmented_form"))
    args = ap.parse_args()

    labels = load_labels(args.labels)
    missing = [cid for cid, lab in labels.items()
               if lab["role"] == "ignore" and not lab["form_code"]]
    if missing:
        print(f"WARNING: {len(missing)} clusters have no role set "
              f"(treated as 'ignore', pages routed to _unclassified): {missing}")

    overrides = load_overrides(args.overrides)
    ocr = load_ocr(args.ocr)
    print(f"{len(overrides)} override rules, OCR text for {len(ocr)} pages")
    if not ocr:
        print("WARNING: no OCR cache - form codes come from cluster labels only "
              "(run ocr_titles.py first to tell H-3 / P-3 etc. apart)")

    by_page = load_page_clusters(args.clusters)
    pdf_dir = Path(args.pdf_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)

    manifest_rows = []
    for pdf_path in sorted(pdf_dir.glob("*.pdf")):
        print(f"processing {pdf_path.name}")
        doc = fitz.open(pdf_path)
        process_file(pdf_path.name, doc, by_page, labels, overrides, ocr, out_dir, manifest_rows)
        doc.close()

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"wrote {len(manifest_rows)} records to {manifest_path}")
    print("records by form:", dict(Counter(r["form_type"] for r in manifest_rows).most_common()))
    print("label source:", dict(Counter(r["label_source"] for r in manifest_rows).most_common()))
    no_ocr = sum(r["label_source"] == "cluster_no_ocr" for r in manifest_rows)
    if no_ocr and ocr:
        print(f"WARNING: {no_ocr} front pages have no OCR text at their current rotation "
              f"- rerun ocr_titles.py to fill them in")


if __name__ == "__main__":
    main()
