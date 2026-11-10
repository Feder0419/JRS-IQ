"""
Score pipeline output against the hand-checked values in ground_truth.json.

Usage: python evaluate.py [pipeline_output.json] [--verbose]

Per field it counts exact matches, matches after ignoring case/punctuation,
wrong values and missing values. "Silent errors" (wrong values the pipeline did
not flag for review) matter most: a flagged error costs a person a look, a
silent one goes into the database.
"""

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "output" / "field_index.json"
GROUND_TRUTH = HERE / "ground_truth.json"


def norm(value) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true", help="list every wrong or missing field")
    args = parser.parse_args()

    truth = json.loads(GROUND_TRUTH.read_text(encoding="utf-8"))
    results = {r["document"]: r for r in json.loads(args.output.read_text(encoding="utf-8"))}

    totals = {"exact": 0, "loose": 0, "wrong": 0, "missing": 0, "silent_wrong": 0}
    for doc, fields in truth.items():
        result = results.get(doc)
        if result is None:
            print(f"{doc}: not in output, skipped")
            continue
        flagged = {r.split(":")[0] for r in result["escalation_reasons"]}
        for name, expected in fields.items():
            got = result["fields"].get(name, {}).get("value")
            if got == expected:
                kind = "exact"
            elif got and norm(got) == norm(expected):
                kind = "loose"
            elif not got:
                kind = "missing"
            else:
                kind = "wrong"
            totals[kind] += 1
            silent = kind == "wrong" and name not in flagged
            totals["silent_wrong"] += silent
            if args.verbose and kind in ("wrong", "missing", "loose"):
                note = " (SILENT)" if silent else (" (flagged)" if name in flagged else "")
                print(f"  {kind:7} {doc[:28]:28} {name:18} got={got!r}  want={expected!r}{note}")

    n = sum(totals[k] for k in ("exact", "loose", "wrong", "missing"))
    print(f"\n{n} fields scored")
    print(f"  exact {totals['exact']}   loose (case/punctuation) {totals['loose']}   "
          f"wrong {totals['wrong']}   missing {totals['missing']}")
    print(f"  correct (exact+loose): {totals['exact'] + totals['loose']}/{n}")
    print(f"  silent errors (wrong and not flagged): {totals['silent_wrong']}")


if __name__ == "__main__":
    main()
