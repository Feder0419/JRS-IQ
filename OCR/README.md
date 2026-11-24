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

**GPU (NVIDIA only):** the pipeline asks for `device="gpu:0"`, which needs the
GPU build of PaddlePaddle instead of `paddlepaddle`. Match the CUDA version in
the index URL to your driver (`nvidia-smi` shows the maximum it supports):

```
pip uninstall paddlepaddle
pip install paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
python -c "import paddle; print(paddle.device.is_compiled_with_cuda())"   # should print True
```

Without a GPU, change `device="gpu:0"` to `"cpu"` in `field_extraction.py`.

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

Use `--output-dir some_folder` to write results somewhere else. Do this for
test runs: the default `output/` folder holds the hand-reviewed JSON, which a
normal run overwrites.

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
   the field's regex. If the label has nothing after it on the same row, look at
   the table cell directly below. Label words (FORM, NAME, ...) are rejected as
   serial/board/boiler numbers. Names must be mostly letters and contain a
   capitalized word, and are spell-corrected against a small vocabulary
   (`Aqud-Chem` to `Aqua-Chem`; the fix is recorded as `autocorrected`). Rows
   mentioning "expires" are ignored when looking for the report date. If no
   label is found, fall back to a whole-document regex scan (not used for
   serial, board or boiler numbers or for names, which are left empty).
   Non-ASCII characters are stripped from values.
5. **Sanity checks** (a failing value is kept but flagged for review): a report
   date with an implausible or cut-off year (`September 17, 19`) or a year before
   the year built; a material spec not on the known list (the grade, as in
   `SA-516-70`, is kept); a boiler number that differs from the serial number
   (they were identical on every report checked); a national board number equal
   to the serial or boiler number; a serial number equal to the year built.
6. **Merge**: a PDF's pages are combined into one entry per PDF, preferring the
   earliest page that found a field by its label.
7. **Field crops**: a small image of each field is saved to
   `output/field_crops/<page>__<field>.png`. OCR does what it is good at,
   finding the printed keywords, and a vision model reads the value, handwriting
   included, from the crop instead of the whole page. Where the crop goes
   depends on whether the page is a form type we have a template for:
   - **Known form** (`form_templates.py`, `form_templates/templates.json`): the
     page is lined up with a stored exemplar picture of the form by matching its
     printed structure (rules, titles, captions; handwriting and typed values just
     fail to match), which gives this scan's shift, scale and tilt. The field
     rectangles marked on the exemplar are then moved onto the page. This does not
     depend on OCR reading any label, and the number of matching features says
     whether the page is that form at all. Only the fields the template lists get
     a crop (`crop_kind` `template:<name>`); the page-level `form_template` in the
     output says which template matched (a PDF's merged entry lists one per page
     under `form_templates`).
   - **Unknown form, or no match**: the crop is placed from keywords
     (`field_crops.py`):
   - A keyword only counts if it looks like a label: it starts its text box or
     the box is short. A sentence that merely contains "National Board" or
     "furnished for" is ignored.
   - Labels ("Year Built 1971") get a zone to their right; captions
     ("(Mfrs. Serial No.)") get a zone above them, because on these forms the
     caption is printed under the filled-in value. Zones are generous, so
     handwriting that sits off the line is still inside.
   - Of the zones that contain something shaped like the value, the top-most is
     used: the report's own data sits above the certificates further down, and
     a repeated "Date" lower on the page is not picked over the first one.
   - Keyword hits on the same line are combined (label plus caption, or a
     neighbouring caption when one is garbled).
   - If no keyword is found there is **no crop** (`crop_file` is null) and the
     model should be given the whole page for that field. `crop_kind` says how
     trustworthy a crop is: `label`, `caption`, `label+caption` (a value-shaped
     box was inside), `keyword_only` (keyword found but no readable value, e.g.
     handwriting), `value_only` (no keyword, only a pattern match).
8. **Review**: fields with confidence below 0.75, found only by the regex
   fallback, or failing a sanity check are flagged. If `ANTHROPIC_API_KEY` is set
   they also get a second opinion from the LLM verifier, which looks at the same
   crop. Documents are flagged for human review if a required field is missing,
   the LLM corrected a value, or confidence stays low.

## Outputs (in `output/`)

| Path | Contents |
| --- | --- |
| `field_index.csv` | One row per PDF: flags, escalation reasons, field values |
| `field_index.json` | Same, plus per-field confidence, source and LLM-check status. |
| `field_crops/*.png` | One cropped image per located field (the image to send to a vision model). `field_index.json` gives each field's `crop_file`, `crop_box` (pixels on the page) and `crop_kind` |
| `pdf_pages/*.png` | Rendered PDF pages that were OCR'd |
| `ocr_boxes/<page>.json` | Every text box OCR found on that page (text, score, `[x1, y1, x2, y2]` pixels), whether or not it became a field value |

Each field in `field_index.json` also carries `boxes` (where its accepted value
was read) and `rejected` (label-adjacent boxes whose text failed validation,
with the reason and per-box text, score and coordinates). A field with no value
can still have `rejected` boxes, so a second reader can be pointed at them.

## Files

| File | Purpose |
| --- | --- |
| `field_extraction.py` | Main pipeline and entry point |
| `field_dictionary.py` | Field definitions (aliases, value regex, required flag), confidence threshold, fuzzy-match tolerance |
| `pdf_preprocess.py` | PDF to PNG at 300 DPI, with optional denoise and invert. Also runnable on its own |
| `llm_verify.py` | Vision-LLM second opinion on low-confidence fields |
| `form_templates.py` | Lines a page up with a known form type and returns its field rectangles (see step 7) |
| `form_templates/` | `templates.json` (where each field sits on each form type) and one exemplar image per template |
| `field_crops.py` | Keyword-based crop placement, used when no template matches (see step 7) |
| `crop_eval.py` | Checks that every crop fully contains its hand-marked target in `crop_targets.json` |
| `crop_targets.json` | Hand-marked rectangle (label/value/caption) each crop must contain, for the sample reports |
| `evaluate.py` | Scores a pipeline output JSON against `ground_truth.json` |
| `ground_truth.json` | Hand-checked field values for the sample reports |
| `PaddleOcrTest.py` | Deprecated scratch script: prints raw OCR text and scores for PNGs in `good_test_images/` |

## Folders

| Folder | Purpose |
| --- | --- |
| `pdfs_to_process/` | Default input: put PDFs here |
| `test_images/` | Image-edit variants of an example report (default, denoise, inverted, ...) kept for comparison |
| `good_test_images/` | Best-performing test image(s), used by `PaddleOcrTest.py` |
| `output/` | Generated results |

## Current status

- LLM verification is **not active**: with no `ANTHROPIC_API_KEY` set it does
  nothing, so every low-confidence or fallback field is escalated to a human.
  Field crops are saved either way.
- Form templates exist for H-3 (front in two layouts, back, certificates page),
  P-3, 154-A, P-2 (front, back) and U-1A (1969 front; 1981 front and back). A page
  that matches none (a new form type or layout, or a very degraded scan) falls back
  to keyword crops, and pages must already be upright. Each template was drawn from
  one exemplar page, and the 1981 U-1A front has had no independent test page.
  Add a template for each new layout (see `form_templates/templates.json`).
- Crops placed from keywords are only as good as keyword finding: a field whose label and caption
  OCR both garbled gets no crop at all (the model then reads the full page).
  `python crop_eval.py <output folder>` checks the crops of the sample reports
  against `crop_targets.json` (27/27 contain their whole target, averaging about
  4% of the page). Like the extraction rules, the crop rules were tuned on only 3
  reports, so mark targets for new ones before trusting that number.
- Before enabling it, check the model name hardcoded in `llm_verify.py`
  (`claude-sonnet-5`) is a valid model ID. The call also has no error handling
  yet, so an API failure would stop the run.
- **Re-running overwrites `field_index.json`.** If you have hand-corrected
  values in it (`review_status`, `ocr_value`), copy the file first, or use
  `--output-dir` for test runs.
- Serial numbers are still often misread (a truncated `4`, or the year). The
  rules above mostly turn these into flagged values rather than fixing them.
- `ground_truth.json` holds the reviewed values for the 3 sample reports and
  `python evaluate.py <output.json> --verbose` scores a run against it (correct
  fields, and "silent errors": wrong values that were not flagged). Most of the
  rules were written from these same 3 reports, so the score overstates how
  well they will do on new ones. Add more reviewed reports before trusting it.
- Field crops come from the OCR'd image, so with `--invert` the model sees the
  inverted version.

## Adding or changing fields

Edit `FIELDS` in `field_dictionary.py`: add label `aliases`, a `pattern` for
the value, and set `required` if a missing field should flag the document.
Optional per-field settings: `reject_words`, `require_digit`, `fallback` (guess
from a document-wide scan), `look_below` (read the cell under the label),
`min_alpha_ratio`, `require_capital`, `skip_row_words` and `vocabulary`. Crop
placement has its own settings (`crop_layout`, `caption_aliases`,
`crop_label_aliases`, `crop_right`, `crop_left`, `crop_down`, `full_line`, `crop_hint`). They are
documented at the top of `field_dictionary.py`.
