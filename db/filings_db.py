"""
A very simple MongoDB store for the OCR pipeline's results.

One document per scanned PDF (its _id is the PDF's file name), holding the nine fields that
were read. Every field carries a `needs_review` flag and the reasons for it, so people can see
what still has to be checked. The document itself has `needs_review` (true if any field does)
and `review_fields` (which ones).

    python filings_db.py load [index.json] [--dry-run]   # put the read data into MongoDB
    python filings_db.py review                          # list every field that needs review
    python filings_db.py resolve DOC FIELD [VALUE]       # a person checked it: set VALUE, or
                                                         # confirm the stored value if none is given

Settings (environment variables; the defaults suit a local server):
    MONGODB_URI           mongodb://localhost:27017
    MONGODB_DB            jrs_iq
    MONGODB_COLLECTION    filings
    MONGODB_TLS_CA_FILE   only if the server uses TLS: the CA certificate that signed its certificate

Re-running `load` is safe: documents are replaced by file name (never duplicated), and a field
a person has resolved here is not overwritten by fresh OCR output.

A flag means "the pipeline had doubt about this value", not "this value is wrong", and an
unflagged value is not guaranteed right: the pipeline can be confidently wrong.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

FIELD_NAMES = [
    "form_number", "report_date", "year_built", "boiler_no", "serial_no",
    "national_board_no", "manufactured_by", "manufactured_for", "material_spec",
]
LOW_CONFIDENCE = 0.75   # same threshold the OCR pipeline uses
DEFAULT_INDEX = Path(__file__).resolve().parent.parent / "OCR" / "output" / "field_index.json"


# ---------------------------------------------------------------- building documents

def blank(value) -> bool:
    return value is None or str(value).strip() == ""


def human_checked(entry: dict) -> bool:
    """A person already looked at this field and was happy with it."""
    return (entry.get("source") == "review"
            and entry.get("review_status") in ("confirmed", "corrected")
            and not entry.get("uncertain"))


def review_reasons(name: str, entry: dict, escalation_reasons: list[str]) -> list[str]:
    """Why this field needs a person to look at it (empty = no flag)."""
    if human_checked(entry):
        return []
    reasons = []
    value = entry.get("value")
    if blank(value):
        reasons.append("no value read")
    if entry.get("review_status") == "needs human":
        reasons.append("reviewer marked: needs human")
    if entry.get("uncertain"):
        reasons.append("reviewer was unsure")
    conf = entry.get("confidence")
    if conf is not None and conf < LOW_CONFIDENCE:
        reasons.append(f"low confidence ({conf:.2f})")
    if entry.get("source") == "pattern_fallback":
        reasons.append("found by pattern only (label not found)")
    # the pipeline's own reasons for this field, e.g. "serial_no: serial_no equals year built"
    for text in escalation_reasons:
        field, _, why = text.partition(":")
        why = why.strip()
        if field.strip() != name or not why:
            continue
        if why in ("uncertain", "not found") or why.startswith("label not found") or why in reasons:
            continue                       # already covered above
        reasons.append(why)
    return reasons


def build_field(name: str, entry: dict | None, escalation_reasons: list[str]) -> dict:
    entry = entry or {}
    reasons = review_reasons(name, entry, escalation_reasons)
    field = {
        "value": entry.get("value"),
        "ocr_value": entry.get("ocr_value", entry.get("value")),   # what OCR read, kept next to the value
        "confidence": entry.get("confidence"),
        "source": entry.get("source"),
        "needs_review": bool(reasons),
        "review_reasons": reasons,
        "reviewed": human_checked(entry),
        "note": entry.get("note"),
    }
    if entry.get("crop_file"):
        field["crop_file"] = entry["crop_file"]       # path of the snippet image, to help the reviewer
    return field


def set_doc_flags(doc: dict) -> dict:
    flagged = [n for n in FIELD_NAMES if doc["fields"][n]["needs_review"]]
    doc["review_fields"] = flagged
    doc["needs_review"] = bool(flagged)
    return doc


def build_document(entry: dict, source_file: Path) -> dict:
    """One entry of field_index.json -> the document stored in MongoDB."""
    reasons = entry.get("escalation_reasons", [])
    fields = {n: build_field(n, (entry.get("fields") or {}).get(n), reasons) for n in FIELD_NAMES}
    doc = {"_id": entry["document"], "fields": fields, "pages": entry.get("pages", []),
           "meta": {"loaded_at": datetime.now(timezone.utc), "source_file": str(source_file)}}
    return set_doc_flags(doc)


def keep_resolved(new: dict, existing: dict | None) -> list[str]:
    """Copy fields people resolved in the database into `new`, unless the incoming one is itself reviewed."""
    kept = []
    for name in FIELD_NAMES:
        old = (existing or {}).get("fields", {}).get(name)
        if old and old.get("reviewed") and not new["fields"][name]["reviewed"]:
            new["fields"][name] = old
            kept.append(name)
    if kept:
        set_doc_flags(new)
    return kept


# ---------------------------------------------------------------- database

def get_collection():
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
    options = {"serverSelectionTimeoutMS": 5000}
    ca = os.environ.get("MONGODB_TLS_CA_FILE")
    if ca:
        options.update(tls=True, tlsCAFile=ca)
    uri = os.environ.get("MONGODB_URI", "mongodb://localhost:27017")
    client = MongoClient(uri, **options)
    try:
        client.admin.command("ping")
    except PyMongoError:
        sys.exit(f"Could not reach MongoDB at {uri}. Is mongod running? (see README.md)")
    coll = client[os.environ.get("MONGODB_DB", "jrs_iq")][os.environ.get("MONGODB_COLLECTION", "filings")]
    coll.create_index("needs_review")
    coll.create_index("fields.serial_no.value")
    return coll


def load(coll, docs: list[dict]) -> tuple[int, int]:
    inserted = updated = 0
    for d in docs:
        kept = keep_resolved(d, coll.find_one({"_id": d["_id"]}))
        if kept:
            print(f"{d['_id']}: kept values already resolved here for {', '.join(kept)}")
        result = coll.replace_one({"_id": d["_id"]}, d, upsert=True)
        if result.upserted_id is not None:
            inserted += 1
        else:
            updated += 1
    return inserted, updated


def resolve(coll, document: str, field: str, value: str | None) -> dict:
    """A person checked a field: store their value (or confirm the current one) and clear its flag."""
    if field not in FIELD_NAMES:
        sys.exit(f"unknown field '{field}'; choose from: {', '.join(FIELD_NAMES)}")
    doc = coll.find_one({"_id": document})
    if doc is None:
        sys.exit(f"no document named '{document}'")
    f = doc["fields"][field]
    if value is not None:
        f["value"] = value
    f.update(needs_review=False, review_reasons=[], reviewed=True, source="review",
             reviewed_at=datetime.now(timezone.utc))
    set_doc_flags(doc)
    coll.replace_one({"_id": document}, doc)
    return f


def flagged_fields(docs) -> list[tuple[str, str, dict]]:
    return [(d["_id"], n, d["fields"][n]) for d in docs for n in d.get("review_fields", [])]


# ---------------------------------------------------------------- command line

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("load", help="load field_index.json into MongoDB")
    p.add_argument("index", nargs="?", type=Path, default=DEFAULT_INDEX)
    p.add_argument("--dry-run", action="store_true", help="show what would be stored; no server needed")
    sub.add_parser("review", help="list the fields that need review")
    p = sub.add_parser("resolve", help="mark a field as checked by a person")
    p.add_argument("document"); p.add_argument("field"); p.add_argument("value", nargs="?")
    args = ap.parse_args()

    if args.cmd == "load":
        entries = json.loads(args.index.read_text(encoding="utf-8"))
        docs = [build_document(e, args.index) for e in entries]
        if args.dry_run:
            for d in docs:
                print(f"{d['_id']}: {len(d['review_fields'])} field(s) need review")
                for n in d["review_fields"]:
                    f = d["fields"][n]
                    print(f"    {n}: {f['value']!r}  <- {'; '.join(f['review_reasons'])}")
            total = sum(len(d["review_fields"]) for d in docs)
            print(f"\nDry run: {len(docs)} documents, {total} flagged field(s), nothing written.")
            return
        inserted, updated = load(get_collection(), docs)
        flagged = sum(len(d["review_fields"]) for d in docs)
        print(f"\nLoaded {inserted} new and {updated} updated documents; {flagged} field(s) flagged for review.")
    elif args.cmd == "review":
        coll = get_collection()
        items = flagged_fields(coll.find({"needs_review": True}).sort("_id", 1))
        if not items:
            print("Nothing needs review.")
            return
        print(f"{len(items)} field(s) need review:\n")
        last = None
        for doc_id, name, f in items:
            if doc_id != last:
                print(doc_id); last = doc_id
            note = f"  [note: {f['note']}]" if f.get("note") else ""
            print(f"  {name:18} value={f['value']!r}  <- {'; '.join(f['review_reasons'])}{note}")
    elif args.cmd == "resolve":
        f = resolve(get_collection(), args.document, args.field, args.value)
        print(f"{args.document} / {args.field}: marked reviewed, value = {f['value']!r}")


if __name__ == "__main__":
    main()
