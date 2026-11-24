"""
Checks the field crops against hand-marked targets (crop_targets.json).

For every marked field it rebuilds the crop from a saved ocr_boxes/ folder (no
OCR is re-run), then reports:
  covered  - share of the target rectangle inside the crop (100% = nothing cut off)
  page     - crop area as a share of the page (smaller = cheaper for the AI model)
  top-most - for fields the form prints once near the top: whether the crop is
             the top-most plausible place rather than a copy further down

Usage: python crop_eval.py <output folder with ocr_boxes/> [--verbose]
"""
import argparse
import json
from pathlib import Path

from PIL import Image

import field_extraction as fe

TARGETS = Path(__file__).with_name("crop_targets.json")


def overlap(a, b) -> float:
    """Share of rectangle b that lies inside rectangle a."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(w, 0) * max(h, 0) / ((b[2] - b[0]) * (b[3] - b[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    targets = json.loads(TARGETS.read_text())["pages"]

    total = full = missing = 0
    cover_sum = page_sum = 0.0
    for page, fields in targets.items():
        data = json.loads((args.output_dir / "ocr_boxes" / f"{page}.json").read_text())
        boxes = [fe.OcrBox(b["text"], b["score"], b["box"]) for b in data["boxes"]]
        with Image.open(args.output_dir / "pdf_pages" / data["image"]) as im:
            w, h = im.size
        rows = fe.group_into_rows(boxes)
        found, _ = fe.extract_fields(rows)
        located = fe.locate_field_crops(rows, found, w, h, fe.fuzzy_contains, fe.value_ok)
        for field, target in fields.items():
            total += 1
            region = located.get(field)
            crop = region["box"] if region else None
            if crop is None:
                missing += 1
                print(f"  NO CROP   {page[-24:]:24} {field}")
                continue
            cov = overlap(crop, target)
            share = (crop[2] - crop[0]) * (crop[3] - crop[1]) / (w * h)
            cover_sum += cov
            page_sum += share
            full += cov >= 0.98
            if args.verbose or cov < 0.98:
                print(f"  {'ok ' if cov >= 0.98 else 'CUT'} {cov:5.0%} of target, {share:5.1%} of page  "
                      f"{region['kind']:13} {page[-24:]:24} {field}")
    got = total - missing
    print(f"\n{full}/{total} crops contain their whole target ({missing} with no crop); "
          f"mean coverage {cover_sum / total:.0%}, mean crop size {page_sum / max(got, 1):.1%} of the page")


if __name__ == "__main__":
    main()
