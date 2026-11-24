"""
Step 2: ask the model, both ways, for every document.

  full      all pages of the report as images -> every field   (the baseline)
  snippets  only the snippet images             -> the fields that have one

By default the snippets arm makes ONE request per document holding all of that document's
snippets (fewer requests = less per-call overhead). --mode field sends one request per
snippet instead, to see whether that changes accuracy.

Every reply is saved under <run>/raw/, so a run can be stopped and resumed (finished calls
are skipped), and results.csv / usage.csv are rebuilt from those files at the end.

    python run_comparison.py --run runs/run1 --adapter gemini --model <model id> --limit 3
    python run_comparison.py --run runs/run1 --adapter dryrun            # plumbing check, no calls
"""
import argparse
import csv
import json
import time
from pathlib import Path

from PIL import Image

from adapters import get_adapter
from common import FIELD_NAMES, load_manifest, parse_reply
from prompts import full_page_prompt, snippets_prompt


def call(adapter, images, prompt, raw_file: Path, redo: bool, retries: int = 2):
    """One model call, cached on disk. Returns the saved record."""
    if raw_file.exists() and not redo:
        return json.loads(raw_file.read_text(encoding="utf-8"))
    last_error = None
    for attempt in range(retries + 1):
        try:
            start = time.time()
            text, usage = adapter.read(images, prompt)
            record = {"text": text, "usage": usage, "seconds": round(time.time() - start, 2),
                      "images": [str(p) for p in images], "pixels": sum(Image.open(p).width * Image.open(p).height for p in images),
                      "error": None}
            raw_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
            return record
        except Exception as e:  # noqa: BLE001 - keep the run going; the error is recorded
            last_error = repr(e)
            time.sleep(2 * (attempt + 1))
    record = {"text": "", "usage": {}, "seconds": 0, "images": [str(p) for p in images], "pixels": 0, "error": last_error}
    raw_file.write_text(json.dumps(record, indent=1), encoding="utf-8")
    return record


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run folder made by make_snippets.py")
    ap.add_argument("--adapter", required=True, help="dryrun | anthropic | gemini | custom (see adapters.py)")
    ap.add_argument("--model", default=None, help="model id for the adapter")
    ap.add_argument("--mode", choices=("doc", "field"), default="doc", help="snippets arm: one request per document (default) or per snippet")
    ap.add_argument("--limit", type=int, default=0, help="only the first N documents (use 2-3 first to check everything works)")
    ap.add_argument("--redo", action="store_true", help="ignore saved replies and ask again")
    args = ap.parse_args()

    run = args.run
    manifest = load_manifest(run)
    adapter = get_adapter(args.adapter, args.model)
    (run / "raw").mkdir(exist_ok=True)
    docs = manifest["docs"][:args.limit] if args.limit else manifest["docs"]

    for n, d in enumerate(docs, start=1):
        print(f"[{n}/{len(docs)}] {d['id']}")
        # --- full-page arm: every page, every field
        pages = [run / p["full_page"] for p in d["pages"]]
        call(adapter, pages, full_page_prompt(FIELD_NAMES, len(pages)), run / "raw" / f"{d['id']}__full.json", args.redo)

        # --- snippets arm: only the fields that got a snippet
        fields = [f for f in FIELD_NAMES if f in d["snippets"]]
        if not fields:
            continue
        if args.mode == "doc":
            images = [run / d["snippets"][f]["file"] for f in fields]
            call(adapter, images, snippets_prompt(fields), run / "raw" / f"{d['id']}__snippets.json", args.redo)
        else:
            for f in fields:
                call(adapter, [run / d["snippets"][f]["file"]], snippets_prompt([f]), run / "raw" / f"{d['id']}__snip__{f}.json", args.redo)

    # --- rebuild results.csv and usage.csv from the saved replies
    result_rows, usage_rows = [], []
    for d in docs:
        def add_usage(arm, rec, calls=1):
            u = rec.get("usage") or {}
            usage_rows.append([d["id"], arm, calls, u.get("input_tokens"), u.get("output_tokens"), len(rec.get("images", [])),
                               rec.get("pixels", 0), rec.get("seconds", 0), bool(rec.get("text")) and not rec.get("error")])

        full_file = run / "raw" / f"{d['id']}__full.json"
        if full_file.exists():
            rec = json.loads(full_file.read_text(encoding="utf-8")); add_usage("full", rec)
            reply = parse_reply(rec["text"])
            for f in FIELD_NAMES:
                e = reply.get(f, {})
                result_rows.append([d["id"], "full", f, e.get("value"), e.get("label_seen"), e.get("confidence"), f in d["snippets"]])

        fields = [f for f in FIELD_NAMES if f in d["snippets"]]
        recs = []
        if args.mode == "doc":
            sf = run / "raw" / f"{d['id']}__snippets.json"
            if sf.exists():
                rec = json.loads(sf.read_text(encoding="utf-8")); recs.append(rec); add_usage("snippets", rec)
                reply = parse_reply(rec["text"])
        else:
            reply = {}
            files = [(f, run / "raw" / f"{d['id']}__snip__{f}.json") for f in fields]
            recs = [json.loads(p.read_text(encoding="utf-8")) for _, p in files if p.exists()]
            for f, p in files:
                if p.exists():
                    reply.update({k: v for k, v in parse_reply(json.loads(p.read_text(encoding="utf-8"))["text"]).items() if k == f})
            if recs:
                usage_rows.append([d["id"], "snippets", len(recs), sum((r.get("usage") or {}).get("input_tokens") or 0 for r in recs) or None,
                                   sum((r.get("usage") or {}).get("output_tokens") or 0 for r in recs) or None,
                                   sum(len(r.get("images", [])) for r in recs), sum(r.get("pixels", 0) for r in recs),
                                   round(sum(r.get("seconds", 0) for r in recs), 2), all(r.get("text") for r in recs)])
        if recs:
            for f in fields:
                e = reply.get(f, {})
                result_rows.append([d["id"], "snippets", f, e.get("value"), e.get("label_seen"), e.get("confidence"), True])

    with (run / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["document", "arm", "field", "value", "label_seen", "confidence", "had_snippet"]); w.writerows(result_rows)
    with (run / "usage.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["document", "arm", "calls", "input_tokens", "output_tokens", "images", "pixels", "seconds", "reply_ok"]); w.writerows(usage_rows)
    print(f"\nWrote {run / 'results.csv'} and {run / 'usage.csv'}. Next: python score.py --run {run} --truth {run / 'ground_truth.csv'}")


if __name__ == "__main__":
    main()
