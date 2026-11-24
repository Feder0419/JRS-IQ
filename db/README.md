# Filings database (MongoDB, very simple)

Stores what the OCR pipeline read from each scanned PDF, and marks every field that needs a
person to check it. One document per PDF; images are not stored (only the file name and, if
present, the path of the field's snippet image).

## Setup

```
pip install -r requirements.txt
```

Run MongoDB Community Server locally and leave it running in a terminal. The simplest way is
without TLS (nothing leaves your machine):

```
mongod --dbpath C:\mongodb\data --bind_ip 127.0.0.1
```

If you start the server the way it was first set up (with `--tlsMode requireTLS`), tell the
client where the CA certificate is: `set MONGODB_TLS_CA_FILE=C:\mongodb\certs\ca.pem`.

## Use

```
python filings_db.py load --dry-run      # show what would be stored and what is flagged; no server needed
python filings_db.py load                # load OCR/output/field_index.json (or pass another file)
python filings_db.py review              # list every field that needs review
python filings_db.py resolve DOC FIELD 47508   # a person checked it: store 47508 and clear the flag
python filings_db.py resolve DOC FIELD         # ...or just confirm the stored value
python test_filings_db.py                # checks the logic without a server
```

`DOC` is the PDF's file name, for example `COASTAL_IRON_WKS_REPORTS_1981_page_13.pdf`.

## What is stored

Database `jrs_iq`, collection `filings`, `_id` = the PDF's file name:

```
_id            "COASTAL_IRON_WKS_REPORTS_1981_page_13.pdf"
needs_review   true if any field needs review
review_fields  ["national_board_no", ...]
fields:
  <one entry for each of the nine fields: form_number, report_date, year_built, boiler_no,
   serial_no, national_board_no, manufactured_by, manufactured_for, material_spec>
    value          what we currently believe (a person's value once resolved)
    ocr_value      what the pipeline originally read (kept after a person resolves it)
    confidence     0-1, absent for hand-entered values
    source         label | pattern_fallback | review
    needs_review   true if a person should check this field
    review_reasons why, e.g. ["low confidence (0.53)"], ["no value read"], ["reviewer was unsure"]
    reviewed       true once a person has checked it
    note           a reviewer's note, if any
    crop_file      path of the snippet image for this field, if the pipeline made one
pages, meta    page images read, when and from which file it was loaded
```

## When a field is flagged

- no value was read
- confidence is below 0.75
- the value was found by pattern only (its label was not found)
- the pipeline's own checks complained (for example boiler and serial numbers differ, or a
  year is not plausible)
- in a hand-reviewed file: the reviewer marked it "needs human" or "unsure"

A field a person has already confirmed or corrected is not flagged. Loading again is safe:
documents are replaced by file name (no duplicates), and a field resolved in the database with
`resolve` is not overwritten by fresh OCR output.

## Know this

- A flag means "the pipeline had doubt", not "this is wrong", and **an unflagged value is not
  guaranteed right**: the pipeline can be confidently wrong, especially on serial and board
  numbers. Spot-check unflagged values too.
- There is no login and no TLS by default: fine on one machine, not for sharing. A fuller version
  (TLS, schema validation) is parked outside the repo.
- The current `OCR/output/field_index.json` is the hand-reviewed file and has no form number, so
  `form_number` shows as "no value read" for those documents.
- The server connection has not been tested from here (only the logic, with a fake collection).
