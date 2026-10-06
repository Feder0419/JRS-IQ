# OCR Field Extraction Pipeline

Extracts a handful of filing/lookup fields (report date, serial number, boiler
number, material spec, manufacturer, ...) from scanned, degraded manufacturing
reports, instead of transcribing the whole page. Uncertain fields are flagged
for human review.

## Install

Python 3.10+ (the code uses `X | None` type hints).

```
pip install paddlepaddle paddleocr pymupdf opencv-python pillow numpy
pip install anthropic        # only needed for the optional LLM verification step
```

Optional: set `ANTHROPIC_API_KEY` to turn on LLM verification (see below).

## Quick start

1. Drop one or more PDFs into `pdfs_to_process/`.
2. Run from this folder:

```
python field_extraction.py
```

Optional cleanup flags (off by default; inversion can hurt on normal
dark-on-light scans, so test on your documents):

```
python field_extraction.py --denoise --invert
```

You can also pass a specific input: a folder, a single PDF, or a single PNG.

```
python field_extraction.py path/to/report.pdf
python field_extraction.py test_images          # existing PNGs, used as-is
```

## Pipeline

1. **PDF to PNG**: each PDF page is rendered at 300 DPI to
   `output/pdf_pages/<pdf>_p001.png`, ... Existing PNGs are never reprocessed.
   `--denoise` / `--invert` apply only to these PDF-derived pages.
2. **OCR**: PaddleOCR reads each PNG, giving text, confidence and a bounding
   box per text line.
3. **Row grouping**: boxes are grouped into rows by vertical position.
4. **Field extraction**: for each field in `field_dictionary.py`, find a row
   with a (fuzzy-matched) label, take the text after it, and check it against
   the field's regex. If no label is found, fall back to a whole-document regex
   scan.
5. **Review**: fields with confidence below 0.75, or found only by the regex
   fallback, are cropped to `output/review_queue/` and sent to the LLM verifier.
   Documents are flagged for human review if a required field is missing, the
   LLM corrected a value, or confidence stays low.

## Outputs (in `output/`)

| Path | Contents |
| --- | --- |
| `field_index.csv` | One row per page: flags, escalation reasons, field values |
| `field_index.json` | Same, plus per-field confidence, source and LLM-check status |
| `review_queue/*.png` | Cropped images of fields needing a second look |
| `pdf_pages/*.png` | Rendered PDF pages that were OCR'd |

## Files

| File | Purpose |
| --- | --- |
| `field_extraction.py` | Main pipeline and entry point |
| `field_dictionary.py` | Field definitions (aliases, value regex, required flag), confidence threshold, fuzzy-match tolerance |
| `pdf_preprocess.py` | PDF to PNG at 300 DPI, with optional denoise and invert. Also runnable on its own |
| `llm_verify.py` | Vision-LLM second opinion on low-confidence fields |
| `PaddleOcrTest.py` | Scratch script: prints raw OCR text and scores for PNGs in `good_test_images/` |

## Folders

| Folder | Purpose |
| --- | --- |
| `pdfs_to_process/` | Default input: put PDFs here |
| `test_images/` | Image-edit variants of an example report (default, denoise, inverted, ...) kept for comparison |
| `good_test_images/` | Best-performing test image(s), used by `PaddleOcrTest.py` |
| `output/` | Generated results |

## Current status

- LLM verification is **not active**: with no `ANTHROPIC_API_KEY` set, it only
  saves the review crops and returns, so every low-confidence or fallback field
  is escalated to a human.
- Before enabling it, check the model name hardcoded in `llm_verify.py`
  (`claude-sonnet-5`) is a valid model ID. The call also has no error handling
  yet, so an API failure would stop the run.
- LLM review crops come from the OCR'd image, so with `--invert` the model sees
  the inverted version.

## Adding or changing fields

Edit `FIELDS` in `field_dictionary.py`: add label `aliases`, a `pattern` for
the value, and set `required` if a missing field should flag the document.
