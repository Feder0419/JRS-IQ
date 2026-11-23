"""
Decides which part of the page to crop for each field.

The crop is what a vision model reads instead of the whole page, so it has to
(a) contain the whole value, label and caption, with nothing cut off, and
(b) be the right place: the field as printed near the top of the form, not a
    look-alike keyword in a paragraph or a repeat further down the page.

How a field's place is chosen:
 1. Find every box whose text looks like the field's keyword (`aliases`, plus
    `caption_aliases`). Only label-like boxes count: the keyword must start the
    box or the box must be short, so a sentence that merely contains "National
    Board" or "furnished for" is ignored.
 2. Give each one a zone: for a label ("Year Built 1971") the zone reaches right
    of it; for a caption ("(Mfrs. Serial No.)") the value is printed above it,
    so the zone reaches up. Both are generous, to cover handwriting that sits
    off the line.
 3. Keep the zones that actually contain something shaped like the value, and
    take the top-most one (these forms put the report's own data above the
    certificates further down the page).
 4. If no zone holds a value-shaped box, still crop around the top-most keyword,
    since the value may be handwriting OCR could not read.
No keyword at all means no crop: the caller sends the whole page instead.
"""
import re
from difflib import SequenceMatcher

from field_dictionary import ALIAS_FUZZ_RATIO, FIELDS

CROP_PAD = 25            # px added on every side of the zone (pages are 300 DPI)
LABEL_START_CHARS = 3    # the keyword must begin within this many characters of its box ("1. Manufactured")...
LABEL_EXTRA_CHARS = 25   # ...unless the whole box is no longer than the keyword plus this
ZONE_UP_LABEL, ZONE_DOWN_LABEL = 1.0, 2.2      # label zone, in line heights above / below the label
ZONE_UP_CAPTION, ZONE_DOWN_CAPTION = 2.4, 0.4  # caption zone: the value sits above its caption
ZONE_SIDE_CAPTION = 0.75 # caption zone reaches this share of the caption's width left and right
LEFT_PAD = 0.5           # label zone starts this many line heights left of the label
FULL_LINE_SLACK = 3      # extra line heights past the outermost text of a full-line value
SAME_ROW_H = 6           # a label zone and a caption zone this close (in line heights) are one field
SAME_LINE_PX = 70        # two hits of the same kind this close vertically are on one line


def _words(text: str) -> list[tuple[int, str]]:
    return [(m.start(), m.group(0)) for m in re.finditer(r"[a-z0-9']+", text.lower())]


def _same_word(token: str, word: str) -> bool:
    return token == word or (len(word) >= 3 and SequenceMatcher(None, token, word).ratio() >= ALIAS_FUZZ_RATIO)


def _label_like(text: str, alias: str) -> bool:
    """
    True if `text` is a form label/caption rather than body text that happens to hold
    the keyword. A short box is taken as a label. In a longer one (a label fused with
    its value) the keyword must open the box and its words must follow each other, so
    "Manufacturers' Partial Data Reports ... signed by" does not pass for "manufactured by".
    """
    if len(text) <= len(alias) + LABEL_EXTRA_CHARS:
        return True
    words = alias.lower().split()
    tokens = _words(text)
    for i, (pos, token) in enumerate(tokens):
        if _same_word(token, words[0]):
            if pos > LABEL_START_CHARS:
                return False
            rest = [t for _, t in tokens[i + 1:i + 1 + len(words)]]  # the next few words
            return all(any(_same_word(t, w) for t in rest) for w in words[1:])
    return False


def _starts_with_word(text: str, word: str) -> bool:
    return any(_same_word(token, word) for _, token in _words(text))


def _union(boxes) -> list[float]:
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def _centre_in(box, zone) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]


def find_anchors(rows, aliases, skip_words, fuzzy_contains) -> list[dict]:
    """Every label-like place the keywords appear: {"boxes": [OcrBox, ...], "alias": str}."""
    anchors = []
    for row in rows:
        row_text = " ".join(b.text for b in row)
        if any(fuzzy_contains(row_text, w, ALIAS_FUZZ_RATIO) for w in skip_words):
            continue  # e.g. the certificate-expiry line
        for alias in aliases:
            first = alias.lower().split()[0]
            for i, start in enumerate(row):
                if not _starts_with_word(start.text, first):
                    continue  # the label must begin in this box
                for k in (1, 2, 3):  # the keyword may be split over a few neighbouring boxes
                    window = row[i:i + k]
                    text = " ".join(b.text for b in window)
                    if fuzzy_contains(text, alias, ALIAS_FUZZ_RATIO) and _label_like(text, alias):
                        anchors.append({"boxes": window, "alias": alias})
                        break
    return anchors


def zone_for(anchor: dict, layout: str, spec: dict, all_boxes, page_w: int) -> list[float]:
    """The generous region in which this anchor's value should lie (page pixels)."""
    rect = _union([b.box for b in anchor["boxes"]])
    h = min(max(rect[3] - rect[1], 35), 70)
    w = rect[2] - rect[0]
    if layout == "above":
        zone = [rect[0] - ZONE_SIDE_CAPTION * w, rect[1] - ZONE_UP_CAPTION * h,
                rect[2] + ZONE_SIDE_CAPTION * w, rect[3] + ZONE_DOWN_CAPTION * h]
        band = [0, zone[1], page_w, rect[1]]
    else:
        zone = [rect[0] - spec.get("crop_left", LEFT_PAD) * h, rect[1] - ZONE_UP_LABEL * h,
                rect[2] + spec.get("crop_right", 12) * h, rect[3] + spec.get("crop_down", ZONE_DOWN_LABEL) * h]
        band = [0, rect[1] - 0.5 * h, page_w, rect[3] + 0.5 * h]
    if spec.get("full_line"):
        # the value runs along the whole line: reach the outermost text in that band
        in_band = [b.box for b in all_boxes if _centre_in(b.box, band)] + [rect]
        zone[0] = min(zone[0], min(b[0] for b in in_band))
        zone[2] = max(zone[2], max(b[2] for b in in_band) + FULL_LINE_SLACK * h)  # room for the underline's tail
    return zone


def _holds_value(zone, anchor, spec: dict, all_boxes, value_ok) -> bool:
    """Does the zone contain a box shaped like this field's value (not just its own label)?"""
    own = {id(b) for b in anchor["boxes"]}
    label_len = len(anchor["alias"]) + 8
    hint = re.compile(spec["crop_hint"], re.IGNORECASE) if spec.get("crop_hint") else None
    for b in all_boxes:
        if not _centre_in(b.box, zone):
            continue
        if id(b) in own and len(b.text) <= label_len:
            continue  # the bare label itself
        if (hint and hint.search(b.text)) or value_ok(spec, b.text):
            return True
    return False


def locate_field_crops(rows, found: dict, page_w: int, page_h: int, fuzzy_contains, value_ok) -> dict:
    """
    {field: {"box": [x1, y1, x2, y2], "kind": ...}} for every field whose keyword was
    located. `value_ok(spec, text)` says whether text has the shape of the field's
    value. kind: "label" / "caption" / "label+caption" (a value-shaped box was in the
    zone), "keyword_only" (keyword found, no value-shaped box), "value_only"
    (no keyword, only a pattern match elsewhere on the page).
    """
    all_boxes = [b for row in rows for b in row]
    result = {}
    for field, spec in FIELDS.items():
        skip = spec.get("skip_row_words", ())
        main = spec.get("crop_layout", "right")
        candidates = []  # (has_value, top y, layout, zone)
        groups = [(spec["aliases"], main), (spec.get("caption_aliases", []), "above"),
                  (spec.get("crop_label_aliases", []), "right")]
        for aliases, layout in groups:
            for anchor in find_anchors(rows, aliases, skip, fuzzy_contains):
                zone = zone_for(anchor, layout, spec, all_boxes, page_w)
                candidates.append({
                    "has_value": _holds_value(zone, anchor, spec, all_boxes, value_ok),
                    "y": min(b.box[1] for b in anchor["boxes"]), "layout": layout, "zone": zone,
                })

        zones, kind = [], None
        if candidates:
            with_value = [c for c in candidates if c["has_value"]]
            pool = with_value or candidates
            best = min(pool, key=lambda c: c["y"])  # top-most
            zones, layouts = [best["zone"]], {best["layout"]}
            # Keyword hits on the same line belong together: the same field often has a
            # label and a caption ("Manufactured by" / "(Name and address of
            # manufacturer)"), and a garbled caption can leave a neighbour as the best hit
            # ("(State and State No.)" beside "(Natl. Board No.)").
            for c in pool:
                if c is best:
                    continue
                reach = SAME_ROW_H * 45 if c["layout"] != best["layout"] else SAME_LINE_PX
                if abs(c["y"] - best["y"]) <= reach:
                    zones.append(c["zone"])
                    layouts.add(c["layout"])
            kind = ("label+caption" if len(layouts) == 2 else "caption" if "above" in layouts else "label") \
                if with_value else "keyword_only"
        elif found.get(field, {}).get("source") == "pattern_fallback":
            # No keyword survived OCR, but the value's shape was found: crop around it,
            # reaching left to where its label would be.
            rect = _union(found[field]["boxes"])
            h = min(max(rect[3] - rect[1], 35), 70)
            zones, kind = [[rect[0] - 8 * h, rect[1] - 1.2 * h, rect[2] + 1.0 * h, rect[3] + 1.2 * h]], "value_only"

        if zones:
            x1, y1, x2, y2 = _union(zones)
            result[field] = {
                "box": [int(max(x1 - CROP_PAD, 0)), int(max(y1 - CROP_PAD, 0)),
                        int(min(x2 + CROP_PAD, page_w)), int(min(y2 + CROP_PAD, page_h))],
                "kind": kind,
            }
    return result
