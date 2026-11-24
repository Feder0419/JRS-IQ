"""
Step 3: score the three ways of reading against your hand-checked answers.

  full      the model read every field from the full page(s)
  snippets  the model read each field from its snippet only (fields with no snippet are unanswered)
  hybrid    what you would actually run: use the snippet answer when it is usable, otherwise
            fall back to the full-page answer. A snippet answer is usable when it has a value,
            names the label it saw (label_seen), and is not marked low confidence. The hybrid
            costs the snippet request plus, only for documents where some field fell back,
            the full-page request.

Writes report.md, scores_by_field.csv and disagreements.csv into the run folder.

    python score.py --run runs/run1 --truth runs/run1/ground_truth.csv
    python score.py --run runs/run1 --truth runs/run1/ground_truth.csv --price-in 0.30 --price-out 2.50
"""
import argparse
import csv
import re
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from common import FIELD_NAMES, blank, load_manifest

MONTHS = {m: i for i, m in enumerate(["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}
TEXT_FIELDS = {"manufactured_by", "manufactured_for"}


def alnum(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def parse_date(s: str):
    """(month, day, year) from 'September 17, 1971', '9/30/69', '9/30 , 19 69', 'MAR 28 1969'; else None."""
    s = str(s).lower().replace(",", " ")
    month = None
    m = re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?", s)
    if m:
        month = MONTHS[m.group(1)]
        nums = re.findall(r"\d+", s[m.end():])
    else:
        nums = re.findall(r"\d+", s)
        if len(nums) < 2:
            return None
        month, nums = int(nums[0]), nums[1:]
    if not nums or not 1 <= month <= 12:
        return None
    day, rest = int(nums[0]), "".join(nums[1:])
    if not 1 <= day <= 31:
        return None
    year = int(rest) if len(rest) == 4 else (1900 + int(rest) if len(rest) == 2 else None)
    return (month, day, year)


def same(field: str, got, truth: str) -> bool:
    """Is the model's value the right answer? Forgiving about case, spacing and punctuation."""
    if blank(got):
        return False
    if field == "report_date":
        a, b = parse_date(got), parse_date(truth)
        if a and b:
            return a == b
    if alnum(got) == alnum(truth):
        return True
    if field in TEXT_FIELDS:  # names and addresses: allow a stray character or two
        return SequenceMatcher(None, alnum(got), alnum(truth)).ratio() >= 0.92
    return False


def load_truth(path: Path) -> dict:
    """{(document, field): value}. Blank = not filled in yet (not scored); N/A = not on the document."""
    truth = {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            v = (row.get("value") or "").strip()
            if v:
                truth[(row["document"], row["field"])] = v
    return truth


def correct(field, got, truth_value) -> bool:
    if truth_value.strip().lower() in ("n/a", "na"):
        return blank(got)          # the right answer is "not there"
    return same(field, got, truth_value)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--truth", required=True, type=Path)
    ap.add_argument("--price-in", type=float, default=None, help="price per 1M input tokens, to estimate cost")
    ap.add_argument("--price-out", type=float, default=None, help="price per 1M output tokens")
    ap.add_argument("--allow-low", action="store_true", help="in the hybrid, accept snippet answers marked low confidence")
    args = ap.parse_args()

    run = args.run
    manifest = load_manifest(run)
    template_of = {d["id"]: ", ".join(sorted({p["template"] or "none" for p in d["pages"]})) for d in manifest["docs"]}
    truth = load_truth(args.truth)

    answers = defaultdict(dict)   # (arm, doc, field) -> row
    with (run / "results.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            answers[(r["arm"], r["document"], r["field"])] = r
    usage = defaultdict(lambda: {"in": 0, "out": 0, "calls": 0, "images": 0, "pixels": 0, "known": True})
    with (run / "usage.csv").open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            u = usage[(r["document"], r["arm"])]
            u["calls"] += int(r["calls"] or 0); u["images"] += int(r["images"] or 0); u["pixels"] += int(float(r["pixels"] or 0))
            if r["input_tokens"] in ("", None):
                u["known"] = False
            else:
                u["in"] += int(float(r["input_tokens"])); u["out"] += int(float(r["output_tokens"] or 0))

    def usable(r) -> bool:
        return (r is not None and not blank(r["value"]) and not blank(r["label_seen"])
                and (args.allow_low or (r["confidence"] or "").lower() != "low"))

    scored = [(d, f) for (d, f) in truth if any(x["id"] == d for x in manifest["docs"])]
    stats = {arm: Counter() for arm in ("full", "snippets", "hybrid")}
    by_field = defaultdict(lambda: defaultdict(Counter))
    by_tpl = defaultdict(lambda: defaultdict(Counter))
    paired = Counter()
    rows_out = []
    fallback_docs = set()
    snippet_only = Counter()   # the snippet arm on fields that had a snippet

    for doc, field in scored:
        t = truth[(doc, field)]
        full = answers.get(("full", doc, field))
        snip = answers.get(("snippets", doc, field))
        had = snip is not None
        use_snip = had and usable(snip)
        hybrid_value = snip["value"] if use_snip else (full["value"] if full else None)
        if had and not use_snip:
            fallback_docs.add(doc)
        if not had:
            fallback_docs.add(doc)
        ok = {"full": correct(field, full["value"] if full else None, t),
              "snippets": had and correct(field, snip["value"], t),
              "hybrid": correct(field, hybrid_value, t)}
        for arm, v in ok.items():
            stats[arm]["n"] += 1; stats[arm]["right"] += v
            by_field[field][arm]["n"] += 1; by_field[field][arm]["right"] += v
            by_tpl[template_of[doc]][arm]["n"] += 1; by_tpl[template_of[doc]][arm]["right"] += v
        if had:
            snippet_only["with_snippet"] += 1
            snippet_only["answered"] += not blank(snip["value"])
            snippet_only["labelled"] += usable(snip)
            snippet_only["snip_right"] += ok["snippets"]
            snippet_only["full_right_same_fields"] += ok["full"]
            if usable(snip) and not ok["snippets"]:
                snippet_only["confident_wrong"] += 1
            paired["snip_right_full_wrong"] += ok["snippets"] and not ok["full"]
            paired["full_right_snip_wrong"] += ok["full"] and not ok["snippets"]
        if ok["hybrid"] != ok["full"]:
            paired["hybrid_right_full_wrong" if ok["hybrid"] else "full_right_hybrid_wrong"] += 1
        if not (ok["full"] == ok["snippets"] == ok["hybrid"]):
            rows_out.append([doc, field, template_of[doc], t, full["value"] if full else "", snip["value"] if had else "(no snippet)",
                             snip["label_seen"] if had else "", hybrid_value, ok["full"], ok["snippets"], ok["hybrid"]])

    def tokens(arm_docs):
        tin = tout = 0; known = True
        for doc, arm in arm_docs:
            u = usage.get((doc, arm))
            if u is None:
                continue
            known &= u["known"]; tin += u["in"]; tout += u["out"]
        return tin, tout, known

    docs_scored = sorted({d for d, _ in scored})
    full_in, full_out, k1 = tokens([(d, "full") for d in docs_scored])
    snip_in, snip_out, k2 = tokens([(d, "snippets") for d in docs_scored])
    fb_in, fb_out, _ = tokens([(d, "full") for d in docs_scored if d in fallback_docs])
    hyb_in, hyb_out = snip_in + fb_in, snip_out + fb_out
    tokens_known = k1 and k2 and (full_in > 0)

    def pct(a, b):
        return f"{100 * a / b:.1f}%" if b else "n/a"

    lines = ["# Snippets vs full page: results", ""]
    lines += [f"Documents scored: {len(docs_scored)}; fields scored: {stats['full']['n']} "
              f"(only fields you filled in; `N/A` means the field is not on the document).", ""]
    lines += ["## Accuracy", "", "| Arm | Right | Accuracy |", "|---|---|---|"]
    for arm in ("full", "snippets", "hybrid"):
        lines.append(f"| {arm} | {stats[arm]['right']}/{stats[arm]['n']} | {pct(stats[arm]['right'], stats[arm]['n'])} |")
    lines += ["", "The snippets arm only answers fields that had a snippet, so unanswered fields count as wrong there; the hybrid is the one to compare with full.", ""]
    n = max(snippet_only["with_snippet"], 1)
    lines += ["## The snippet arm, where a snippet existed", "",
              f"- fields with a snippet: {snippet_only['with_snippet']}  (the rest fall back to the full page)",
              f"- model gave a value: {pct(snippet_only['answered'], snippet_only['with_snippet'])}",
              f"- value AND a label it saw (usable): {pct(snippet_only['labelled'], snippet_only['with_snippet'])}",
              f"- snippet right: {pct(snippet_only['snip_right'], snippet_only['with_snippet'])}  vs full page on the same fields: {pct(snippet_only['full_right_same_fields'], snippet_only['with_snippet'])}",
              f"- **confident wrong** (usable answer, wrong value: the failure that slips past the fallback): {snippet_only['confident_wrong']}"
              f"  ({pct(snippet_only['confident_wrong'], snippet_only['labelled'])} of usable answers)", ""]
    lines += ["## Paired comparison (where the arms disagree)", "",
              f"- snippet right, full page wrong: {paired['snip_right_full_wrong']}",
              f"- full page right, snippet wrong: {paired['full_right_snip_wrong']}",
              f"- hybrid right, full page wrong: {paired['hybrid_right_full_wrong']}",
              f"- full page right, hybrid wrong: {paired['full_right_hybrid_wrong']}", "",
              "With a few hundred fields, a difference of one or two points is within noise; look at these counts and `disagreements.csv`.", ""]
    lines += ["## Fallback", "",
              f"- documents where at least one field fell back to the full page: {len(fallback_docs & set(docs_scored))}/{len(docs_scored)}", ""]
    lines += ["## Cost (tokens)", ""]
    if tokens_known:
        lines += ["| Arm | Input tokens | Output tokens | vs full page (input) |", "|---|---|---|---|",
                  f"| full | {full_in} | {full_out} | 100% |",
                  f"| snippets only | {snip_in} | {snip_out} | {pct(snip_in, full_in)} |",
                  f"| hybrid (snippets + full page where needed) | {hyb_in} | {hyb_out} | {pct(hyb_in, full_in)} |", ""]
        if args.price_in is not None and args.price_out is not None:
            def money(i, o):
                return i / 1e6 * args.price_in + o / 1e6 * args.price_out
            lines += [f"Estimated cost at the prices given: full ${money(full_in, full_out):.4f}, snippets ${money(snip_in, snip_out):.4f}, "
                      f"hybrid ${money(hyb_in, hyb_out):.4f}  (hybrid = {pct(money(hyb_in, hyb_out), money(full_in, full_out))} of full)", ""]
    else:
        lines += ["The adapter did not report token counts, so cost is shown as image counts and pixels instead:", ""]
        for arm in ("full", "snippets"):
            imgs = sum(usage[(d, arm)]["images"] for d in docs_scored if (d, arm) in usage)
            px = sum(usage[(d, arm)]["pixels"] for d in docs_scored if (d, arm) in usage)
            calls = sum(usage[(d, arm)]["calls"] for d in docs_scored if (d, arm) in usage)
            lines.append(f"- {arm}: {calls} requests, {imgs} images, {px / 1e6:.1f} megapixels")
        lines.append("")
    lines += ["## By field", "", "| Field | n | full | snippets | hybrid |", "|---|---|---|---|---|"]
    csv_rows = []
    for f in FIELD_NAMES:
        if f in by_field:
            c = by_field[f]
            lines.append(f"| {f} | {c['full']['n']} | {pct(c['full']['right'], c['full']['n'])} | {pct(c['snippets']['right'], c['snippets']['n'])} | {pct(c['hybrid']['right'], c['hybrid']['n'])} |")
            csv_rows.append([f, c["full"]["n"], c["full"]["right"], c["snippets"]["right"], c["hybrid"]["right"]])
    lines += ["", "## By form template", "", "| Template | n | full | snippets | hybrid |", "|---|---|---|---|---|"]
    for t, c in sorted(by_tpl.items()):
        lines.append(f"| {t} | {c['full']['n']} | {pct(c['full']['right'], c['full']['n'])} | {pct(c['snippets']['right'], c['snippets']['n'])} | {pct(c['hybrid']['right'], c['hybrid']['n'])} |")

    (run / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with (run / "scores_by_field.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["field", "n", "full_right", "snippets_right", "hybrid_right"]); w.writerows(csv_rows)
    with (run / "disagreements.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["document", "field", "template", "truth", "full_value", "snippet_value", "snippet_label_seen", "hybrid_value", "full_ok", "snippets_ok", "hybrid_ok"])
        w.writerows(rows_out)
    print("\n".join(lines))
    print(f"\nWrote {run / 'report.md'}, scores_by_field.csv and disagreements.csv")


if __name__ == "__main__":
    main()
