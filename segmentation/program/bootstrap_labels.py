"""Step 2: turn the old per-cluster tags into a per-page label library.

Reads the hand-tagged discovery_full/labels_template.csv (+ page_clusters.pkl
and page_overrides.csv) and writes store/page_labels.csv - one row per page
with role / rotation / form code and where that came from. Labels live on
pages from here on, so re-clustering never invalidates them.

Three automatic corrections are applied on top of the tags, each listed in
store/label_fixes.csv for review:

  fix_sandwich  pages of an "ignore" cluster whose members mostly sit between
                two front pages (front -> X -> front) are the second page of a
                two-page record: relabelled continuation
  fix_ocr       a non-front page whose title OCR reads a form title
                (e.g. "SUPERHEATERS, WATERWALLS, AND ECONOMIZERS") is a front
                page, turned whichever way the title read
  fix_rotation  a front page whose whole-page OCR reads clearly better the
                other way up gets that rotation (front orientation by OCR was
                right on 237/237 checked pages)

Rows whose source is "human" are never overwritten, so corrections made by
hand in page_labels.csv survive a rerun.

Usage:
    python bootstrap_labels.py
"""
import argparse
from collections import Counter, defaultdict

import stage2_segment as s2
import store
from forms import canon_code

FIELDS = ["file", "page", "role", "rotation", "form_code", "source",
          "tag_role", "tag_rotation", "tag_cluster", "tag_code", "note"]
FIX_FIELDS = ["file", "page", "source", "old_role", "new_role", "old_rotation", "new_rotation",
              "tag_cluster", "evidence"]

SANDWICH_SHARE = 0.6
SANDWICH_MIN = 10
ROTATION_MARGIN = 0.3


def best_cluster(entries):
    return min(entries.values(), key=lambda r: r["dist"])["cluster"] if entries else None


def override_pages(fname, n, overrides):
    covered = set()
    for rule in overrides:
        if rule["file"] == fname:
            last = n - 1 if rule["last"] is None else min(rule["last"], n - 1)
            covered.update(range(rule["first"], last + 1))
    return covered


def find_sandwich_clusters(per_file):
    hits, totals = Counter(), Counter()
    for fname, (tag, clusters, covered) in per_file.items():
        n = len(tag)
        for i, p in enumerate(tag):
            if p["role"] != "ignore" or i in covered or clusters[i] is None:
                continue
            totals[clusters[i]] += 1
            if 0 < i < n - 1 and tag[i - 1]["role"] == "front" and tag[i + 1]["role"] == "front":
                hits[clusters[i]] += 1
    chosen = {c for c in totals if totals[c] >= SANDWICH_MIN and hits[c] / totals[c] >= SANDWICH_SHARE}
    return chosen, hits, totals


def main():
    here = store.HERE
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels", default=str(here / "discovery_full" / "labels_template.csv"))
    ap.add_argument("--clusters", default=str(here / "discovery_full" / "page_clusters.pkl"))
    ap.add_argument("--overrides", default=str(here / "discovery_full" / "page_overrides.csv"))
    args = ap.parse_args()

    tags = s2.load_labels(args.labels)
    by_page = s2.load_page_clusters(args.clusters)
    overrides = s2.load_overrides(args.overrides)
    pages = store.load_pages()
    human = {fp: r for fp, r in store.load_labels().items() if r["source"] == "human"}
    counts = store.pdf_page_counts()

    per_file = {}
    for fname, n in counts.items():
        tag = s2.classify_pages(fname, n, by_page, tags, [])
        clusters = [best_cluster(by_page.get((fname, i))) for i in range(n)]
        per_file[fname] = (tag, clusters, override_pages(fname, n, overrides))
    sandwich, hits, totals = find_sandwich_clusters(per_file)
    print("ignore clusters whose pages mostly sit between two fronts (-> continuation):")
    for c in sorted(sandwich, key=lambda c: -totals[c]):
        print(f"  cluster {c}: {hits[c]}/{totals[c]} sandwiched")

    rows, fixes, missing_ocr = [], [], 0
    for fname, n in counts.items():
        tag, clusters, covered = per_file[fname]
        final = s2.classify_pages(fname, n, by_page, tags, overrides)
        cont_rot = s2.DEFAULT_CONTINUATION_ROTATION.get(fname, 0)
        for i in range(n):
            t = tag[i]
            row = dict(file=fname, page=i + 1, tag_role=t["role"], tag_rotation=t["rotation"],
                       tag_cluster="" if clusters[i] is None else clusters[i],
                       tag_code=t["cluster_code"] or "", form_code="", note="")
            if (fname, i) in human:
                rows.append(human[(fname, i)])
                continue
            if i in covered:
                f = final[i]
                row.update(role=f["role"], rotation=f["rotation"], form_code=f["override_code"] or "",
                           source="override")
                rows.append(row)
                continue
            row.update(role=t["role"], rotation=t["rotation"], source="tag")
            pg = pages.get((fname, i))
            if pg is None:
                missing_ocr += 1

            def fix(source, role, rotation, evidence, form_code=""):
                fixes.append(dict(file=fname, page=i + 1, source=source, old_role=row["role"],
                                  new_role=role, old_rotation=row["rotation"], new_rotation=rotation,
                                  tag_cluster=row["tag_cluster"], evidence=evidence))
                row.update(role=role, rotation=rotation, source=source)
                if form_code:
                    row["form_code"] = form_code

            if row["role"] == "ignore" and clusters[i] in sandwich:
                fix("fix_sandwich", "continuation", cont_rot,
                    f"cluster {clusters[i]}: {hits[clusters[i]]}/{totals[clusters[i]]} pages sit between two fronts")
            if pg and row["role"] != "front":
                found = {rot: pg[f"title_code{rot}"] for rot in (0, 180) if pg[f"title_code{rot}"]}
                if found:
                    rot = pg["ocr_rot"] if pg["ocr_rot"] in found else next(iter(found))
                    fix("fix_ocr", "front", rot, f"title reads {found[rot]} at {rot} deg", found[rot])
            elif pg and row["role"] == "front" and pg["ocr_margin"] >= ROTATION_MARGIN \
                    and pg["ocr_rot"] != row["rotation"]:
                fix("fix_rotation", "front", pg["ocr_rot"],
                    f"page OCR score {pg['ocr_score0']:.0f} at 0 vs {pg['ocr_score180']:.0f} at 180")
            if row["role"] == "front" and pg and not row["form_code"]:
                code = pg[f"title_code{row['rotation']}"]
                if code:
                    row["form_code"] = canon_code(code)
                    row["note"] = "form_code from title OCR"
            rows.append(row)

    store.write_csv(store.LABELS_CSV, rows, FIELDS)
    store.write_csv(store.LABEL_FIXES_CSV, fixes, FIX_FIELDS)
    if missing_ocr:
        print(f"WARNING: {missing_ocr} pages not in the store yet (run ingest.py) - OCR fixes skipped for them")
    print(f"\nwrote {len(rows)} page labels to {store.LABELS_CSV}")
    print("  role:", dict(Counter(r["role"] for r in rows)))
    print("  source:", dict(Counter(r["source"] for r in rows)))
    print(f"wrote {len(fixes)} automatic fixes to {store.LABEL_FIXES_CSV} for review")
    by_fix = defaultdict(Counter)
    for f in fixes:
        by_fix[f["source"]][f"{f['old_role']}->{f['new_role']}"] += 1
    for k, v in by_fix.items():
        print(f"  {k}: {dict(v)}")


if __name__ == "__main__":
    main()
