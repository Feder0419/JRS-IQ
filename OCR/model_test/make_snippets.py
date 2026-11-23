"""
Step 1: cut the snippets.

Takes a folder of PDFs / images (or a text file listing paths, one per line) and, for every
page, lines it up with the known form types (form_templates/) and crops one snippet per
field. No OCR and no model are involved. Writes, inside --out:

    manifest.json        what was cut, per document (read by the next steps)
    pages/               the full pages, as they will be sent in the full-page arm
    snippets/            one PNG per field that got a snippet
    ground_truth.csv     a blank answer sheet to fill in by hand (not overwritten if present)

Pages must be upright (the segmentation step does that). A PDF is one document with all its
pages in order; an image file is a one-page document.

    python make_snippets.py --docs my_pdfs --out runs/run1
    python make_snippets.py --docs docs.txt --out runs/run1 --min-inliers 100
"""
import argparse
import csv
import json
import re
from pathlib import Path

import pymupdf
from PIL import Image

from common import FIELD_NAMES
import form_templates as ft

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


def gather(src: Path) -> list[Path]:
    if src.is_dir():
        return sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES | {".pdf"})
    lines = [l.strip() for l in src.read_text(encoding="utf-8").splitlines()]
    return [Path(l) for l in lines if l and not l.startswith("#")]


def load_pages(path: Path, dpi: int) -> list[Image.Image]:
    if path.suffix.lower() == ".pdf":
        pages = []
        for page in pymupdf.open(path):
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            pages.append(Image.frombytes("L", (pix.width, pix.height), pix.samples))
        return pages
    return [Image.open(path).convert("L")]


def accept_match(result: dict, min_inliers: int, min_margin: float) -> tuple[bool, str | None]:
    """Is the best template a confident enough match to trust its field rectangles?"""
    scores = sorted(result["scores"].values(), reverse=True)
    best = scores[0] if scores else 0
    second = scores[1] if len(scores) > 1 else 0
    if result["template"] is None or best < min_inliers:
        return False, "no template matched"
    if second and best < min_margin * second:
        return False, f"ambiguous: best {best} vs runner-up {second}"
    return True, None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--docs", required=True, type=Path, help="folder of PDFs/images, or a text file of paths")
    ap.add_argument("--out", required=True, type=Path, help="run folder to create")
    ap.add_argument("--dpi", type=int, default=300, help="render PDFs at this DPI; snippets are cropped from this (default 300)")
    ap.add_argument("--full-max-side", type=int, default=2048,
                    help="longest side, in px, of the full-page image sent in the full-page arm (0 = send as rendered). "
                         "Set this to what you normally send to the model (default 2048)")
    ap.add_argument("--min-inliers", type=int, default=ft.MIN_INLIERS, help="template match strength required (default %(default)s)")
    ap.add_argument("--min-margin", type=float, default=1.0,
                    help="best match must beat the runner-up by this factor, to reject look-alike forms (default 1.0 = off; try 2)")
    args = ap.parse_args()

    ft.MIN_INLIERS = args.min_inliers
    files = gather(args.docs)
    if not files:
        raise SystemExit(f"no documents found in {args.docs}")
    for sub in ("pages", "snippets"):
        (args.out / sub).mkdir(parents=True, exist_ok=True)

    docs, used_ids = [], set()
    for path in files:
        doc_id = re.sub(r"[^A-Za-z0-9._-]+", "_", path.stem)
        while doc_id in used_ids:
            doc_id += "_x"
        used_ids.add(doc_id)
        pages_out, snippet_for = [], {}
        for number, img in enumerate(load_pages(path, args.dpi), start=1):
            result = ft.match_page(img)
            ok, reason = accept_match(result, args.min_inliers, args.min_margin)
            fields = result["fields"] if ok else {}

            full = img
            if args.full_max_side and max(img.size) > args.full_max_side:
                scale = args.full_max_side / max(img.size)
                full = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
            page_file = f"pages/{doc_id}_p{number}.png"
            full.save(args.out / page_file)

            cut = {}
            for field, e in fields.items():
                snippet_file = f"snippets/{doc_id}_p{number}__{field}.png"
                img.crop(tuple(e["box"])).save(args.out / snippet_file)
                cut[field] = {"snippet": snippet_file, "box": e["box"], "kind": e["kind"]}
                snippet_for.setdefault(field, {"page": number, "file": snippet_file})  # first page that has it wins
            top = sorted(result["scores"].items(), key=lambda kv: -kv[1])[:2]
            pages_out.append({"index": number, "full_page": page_file, "size": list(img.size),
                              "template": result["template"] if ok else None, "inliers": result["inliers"],
                              "top_scores": dict(top), "rejected_because": reason, "fields": cut})
        docs.append({"id": doc_id, "source": str(path), "pages": pages_out, "snippets": snippet_for,
                     "no_snippet_fields": [f for f in FIELD_NAMES if f not in snippet_for]})
        tpl = ", ".join(p["template"] or "none" for p in pages_out)
        print(f"{doc_id}: {len(pages_out)} page(s) -> templates [{tpl}], {len(snippet_for)}/{len(FIELD_NAMES)} fields have a snippet")

    manifest = {"settings": {"dpi": args.dpi, "full_max_side": args.full_max_side, "min_inliers": args.min_inliers,
                             "min_margin": args.min_margin}, "docs": docs}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    truth = args.out / "ground_truth.csv"
    if not truth.exists():
        with truth.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["document", "field", "value", "notes"])
            for d in docs:
                for field in FIELD_NAMES:
                    w.writerow([d["id"], field, "", ""])
    n_with = sum(1 for d in docs for f in FIELD_NAMES if f in d["snippets"])
    print(f"\n{len(docs)} documents; {n_with}/{len(docs) * len(FIELD_NAMES)} fields have a snippet "
          f"({100 * n_with / (len(docs) * len(FIELD_NAMES)):.0f}%). Wrote {args.out / 'manifest.json'}")
    print(f"Fill in {truth} (value as written on the document; N/A if the field is not on it), then run run_comparison.py.")


if __name__ == "__main__":
    main()
