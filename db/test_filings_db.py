"""
Checks the flagging and load logic without a MongoDB server (a fake collection stands in).
    python test_filings_db.py
"""
import copy
import sys
from pathlib import Path

import filings_db as db


def check(cond, msg):
    print(("ok   " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


class FakeCollection:
    """Just enough of a pymongo collection for load() and resolve()."""
    def __init__(self):
        self.docs = {}

    def find_one(self, q):
        return copy.deepcopy(self.docs.get(q["_id"]))

    def replace_one(self, q, doc, upsert=False):
        new = q["_id"] not in self.docs
        self.docs[q["_id"]] = copy.deepcopy(doc)

        class R:
            upserted_id = q["_id"] if new and upsert else None
        return R()


def fresh_entry():
    """What the pipeline writes for a freshly read PDF."""
    ok = {"value": "x", "confidence": 0.95, "source": "label"}
    fields = {n: dict(ok) for n in db.FIELD_NAMES}
    fields["serial_no"] = {"value": "4", "confidence": 0.60, "source": "label"}                  # low confidence
    fields["year_built"] = {"value": "1832", "confidence": 0.5, "source": "pattern_fallback"}    # guessed
    fields["report_date"] = {"value": None, "confidence": None, "source": None}                   # nothing read
    fields["national_board_no"] = {"value": "9015", "confidence": 0.9, "source": "label"}         # flagged by a sanity check
    return {"document": "A.pdf", "fields": fields, "escalate": True, "pages": ["A_p001.png"],
            "escalation_reasons": ["serial_no: low confidence (0.60)", "national_board_no: same value as serial_no (should be a different number)",
                                   "report_date: not found"]}


def main():
    doc = db.build_document(fresh_entry(), Path("x.json"))
    f = doc["fields"]
    check(not f["form_number"]["needs_review"], "a confident labelled value is not flagged")
    check(f["serial_no"]["needs_review"] and f["serial_no"]["review_reasons"] == ["low confidence (0.60)"],
          "low confidence is flagged once, not twice")
    check(f["year_built"]["review_reasons"] == ["low confidence (0.50)", "found by pattern only (label not found)"],
          "a pattern-only guess is flagged, with both of its reasons")
    check(f["report_date"]["review_reasons"] == ["no value read"], "a missing value is flagged ('not found' not repeated)")
    check("same value as serial_no" in f["national_board_no"]["review_reasons"][0], "the pipeline's own reason is carried over")
    check(doc["review_fields"] == ["report_date", "year_built", "serial_no", "national_board_no"] or
          set(doc["review_fields"]) == {"report_date", "year_built", "serial_no", "national_board_no"}, "document lists the flagged fields")
    check(doc["needs_review"] is True, "document is flagged when any field is")

    # a hand-reviewed file: confirmed/corrected are clear, needs human / unsure are flagged
    rev = {"document": "B.pdf", "escalate": True, "escalation_reasons": ["national_board_no: uncertain"], "fields": {
        n: {"value": "v", "ocr_value": "o", "review_status": "corrected", "uncertain": False, "source": "review", "note": None}
        for n in db.FIELD_NAMES}}
    rev["fields"]["national_board_no"].update(uncertain=True, review_status="confirmed", note="handwritten, hard to read")
    rev["fields"]["serial_no"]["review_status"] = "needs human"
    b = db.build_document(rev, Path("x.json"))
    check(not b["fields"]["boiler_no"]["needs_review"] and b["fields"]["boiler_no"]["reviewed"], "a reviewed field is not flagged")
    check(b["fields"]["national_board_no"]["review_reasons"] == ["reviewer was unsure"], "an unsure field is flagged once")
    check(b["fields"]["national_board_no"]["note"] == "handwritten, hard to read", "the reviewer's note is kept")
    check(b["fields"]["serial_no"]["review_reasons"] == ["reviewer marked: needs human"], "'needs human' is flagged")
    check(b["fields"]["boiler_no"]["ocr_value"] == "o", "the original OCR reading is kept next to the value")

    # a missing field entry in the input is stored as a flagged blank, not dropped
    gap = {"document": "C.pdf", "fields": {}, "escalation_reasons": []}
    g = db.build_document(gap, Path("x.json"))
    check(len(g["fields"]) == 9 and g["review_fields"] == db.FIELD_NAMES, "a missing field is stored and flagged")

    # load + resolve + reload
    coll = FakeCollection()
    check(db.load(coll, [db.build_document(fresh_entry(), Path("x"))]) == (1, 0), "first load inserts")
    check(db.load(coll, [db.build_document(fresh_entry(), Path("x"))]) == (0, 1), "second load updates, no duplicate")
    db.resolve(coll, "A.pdf", "serial_no", "47508")
    s = coll.docs["A.pdf"]
    check(s["fields"]["serial_no"]["value"] == "47508" and not s["fields"]["serial_no"]["needs_review"], "resolve sets the value and clears the flag")
    check("serial_no" not in s["review_fields"] and s["needs_review"], "resolve updates the document's list, other flags stay")
    check(s["fields"]["serial_no"]["ocr_value"] == "4", "resolve keeps what OCR originally read")
    db.load(coll, [db.build_document(fresh_entry(), Path("x"))])
    check(coll.docs["A.pdf"]["fields"]["serial_no"]["value"] == "47508", "re-loading raw OCR does not overwrite a resolved field")
    db.resolve(coll, "A.pdf", "year_built", None)
    check(coll.docs["A.pdf"]["fields"]["year_built"]["value"] == "1832" and coll.docs["A.pdf"]["fields"]["year_built"]["reviewed"],
          "resolve with no value confirms the stored one")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
