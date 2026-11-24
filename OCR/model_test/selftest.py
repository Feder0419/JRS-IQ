"""
Checks that the harness itself works, with no model and no API key:
  python selftest.py

1. unit tests of the answer normalisation and reply parsing;
2. cuts snippets from sample PDFs in the repo;
3. runs both arms with a fake model whose answers are known, and checks the scores.
"""
import csv
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import adapters                      # noqa: E402
from common import FIELD_NAMES, parse_reply   # noqa: E402
from score import correct, parse_date, same   # noqa: E402


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        raise SystemExit(1)


def unit_tests():
    check(parse_date("September 17, 1971") == (9, 17, 1971), "date: month name")
    check(parse_date("9/30 , 19 69") == (9, 30, 1969), "date: split year '19 69'")
    check(parse_date("MAR 28 1969") == (3, 28, 1969), "date: stamped month")
    check(parse_date("2-25-81") == (2, 25, 1981), "date: two-digit year")
    check(same("report_date", "9/30/1969", "9-30-69"), "dates in different formats match")
    check(not same("report_date", "9/30/1969", "9/31/1969"), "different dates do not match")
    check(same("serial_no", "11804-R-1", "11804 r 1"), "ids: punctuation and case ignored")
    check(not same("serial_no", "11804-R-1", "11804-R-2"), "ids: different value does not match")
    check(same("material_spec", "SA285C", "SA-285-C"), "material spec: hyphens ignored")
    check(same("manufactured_by", "Rite Engr. & Mfg. Corp., 9441 Washburn Rd., Downey, CA",
               "Rite Engr & Mfg Corp, 9441 Washburn Rd, Downey, CA."), "names: punctuation ignored")
    check(correct("national_board_no", None, "N/A"), "N/A truth: a null answer is right")
    check(not correct("national_board_no", "230", "N/A"), "N/A truth: an invented value is wrong")
    check(not correct("serial_no", None, "123"), "a null answer is wrong when there is a value")
    r = parse_reply('Sure!\n```json\n{"fields": {"year_built": {"value": "1969", "label_seen": "Year Built", "confidence": "HIGH"}}}\n```')
    check(r["year_built"]["value"] == "1969" and r["year_built"]["confidence"] == "high", "reply parsing: fenced JSON")
    check(parse_reply("I cannot read this") == {}, "reply parsing: no JSON gives {}")


class Fake(adapters.Adapter):
    """Reads 'v_<field>' for every field asked for. In the snippets arm serial_no is wrong but looks confident."""
    name = "fake"

    def read(self, images, prompt):
        import re
        fields = re.findall(r"^- (?:image \d+: )?([a-z_]+)(?::|$)", prompt, re.MULTILINE)
        fields = [f for f in dict.fromkeys(fields) if f in FIELD_NAMES]
        snippets = "SMALL CROP" in prompt
        out = {}
        for f in fields:
            v = "WRONG" if (snippets and f == "serial_no") else f"v_{f}"
            out[f] = {"value": v, "label_seen": "L", "confidence": "high"}
        import json
        return json.dumps({"fields": out}), {"input_tokens": 1000 * len(images) if not snippets else 100 * len(images), "output_tokens": 50}


def pipeline_test():
    adapters.ADAPTERS["fake"] = Fake
    sample = [HERE.parent / "pdfs_to_process" / n for n in (
        "CLEAVER_BROOKS_REPORTS_1971_page_33.pdf", "COASTAL_IRON_WKS_REPORTS_1981_page_13.pdf")]
    sample = [p for p in sample if p.exists()]
    check(len(sample) == 2, "sample PDFs found in the repo")
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        listing = tmp / "docs.txt"
        listing.write_text("\n".join(str(p) for p in sample), encoding="utf-8")
        run = tmp / "run"
        py = sys.executable
        subprocess.run([py, str(HERE / "make_snippets.py"), "--docs", str(listing), "--out", str(run)], check=True, cwd=HERE, capture_output=True)
        check((run / "manifest.json").exists() and (run / "ground_truth.csv").exists(), "make_snippets writes a manifest and an answer sheet")
        check(len(list((run / "snippets").glob("*.png"))) > 0, "snippet images were written")

        # run_comparison + score in this process so the fake adapter is used
        import run_comparison, score
        sys.argv = ["run_comparison", "--run", str(run), "--adapter", "fake"]
        run_comparison.main()
        # answer sheet: v_<field> for every field, except material_spec which we say is not on the document
        rows = list(csv.DictReader((run / "ground_truth.csv").open(encoding="utf-8")))
        with (run / "ground_truth.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["document", "field", "value", "notes"])
            for r in rows:
                w.writerow([r["document"], r["field"], "N/A" if r["field"] == "material_spec" else f"v_{r['field']}", ""])
        sys.argv = ["score", "--run", str(run), "--truth", str(run / "ground_truth.csv")]
        score.main()
        report = (run / "report.md").read_text(encoding="utf-8")
        check("## Accuracy" in report and "confident wrong" in report, "report.md has the sections")
        check("hybrid" in report and "Cost (tokens)" in report, "report.md has hybrid and cost")
        # Confidently wrong in the snippet arm: serial_no (the fake gets it wrong on purpose), and
        # material_spec (the answer sheet says N/A, but the fake invents a value) wherever a snippet exists.
        manifest = __import__("common").load_manifest(run)
        expected = sum(1 for d in manifest["docs"] for f in ("serial_no", "material_spec") if f in d["snippets"])
        line = [l for l in report.splitlines() if "confident wrong" in l][0]
        check(f"): {expected} " in line, f"confident-wrong count equals the {expected} deliberately wrong snippet answers")


if __name__ == "__main__":
    unit_tests()
    pipeline_test()
    print("\nAll checks passed.")
