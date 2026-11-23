# Snippets vs full page: model test

Question: if a model reads small **snippets** (one crop per field, cut by the form-template
method) instead of the whole **full page**, how much accuracy do we lose or gain, and how many
tokens (money) do we save? And what happens when a snippet isn't usable and we fall back to the
full page?

This folder is self-contained: it cuts the snippets, asks a model both ways, and scores the
answers against hand-checked values. Nothing here calls a model unless you pick an adapter.
The full reasoning, what to measure and how to decide are in [TEST_PLAN.md](TEST_PLAN.md).

## What you need

- Python 3.10+ and: `pip install -r requirements.txt` (plus `anthropic` or `google-genai` for
  your model, or write your own adapter: see `adapters.py`)
- 30-40 report PDFs/images with **upright pages** (the segmentation step rotates them), ideally
  a mix of form types and scan quality (see TEST_PLAN.md)
- API access to the model you want to test
- an hour or so to fill in the answer sheet by hand

Check the harness first (no model, no key needed): `python selftest.py`

## Steps

```
cd OCR/model_test

# 1. cut snippets for your documents (a folder of PDFs/images, or a text file of paths)
python make_snippets.py --docs path/to/pdfs --out runs/run1
#    look at runs/run1/snippets/ and runs/run1/manifest.json. Set --full-max-side to the longest
#    side you normally send for a full page (default 2048) so the baseline is realistic.

# 2. fill in the answer sheet: runs/run1/ground_truth.csv
#    one row per document and field. Type the value as it is written on the document.
#    N/A = the field is not on the document. Leave blank to skip a row.

# 3. ask the model both ways (try --limit 2 first)
python run_comparison.py --run runs/run1 --adapter gemini --model <model id> --limit 2
python run_comparison.py --run runs/run1 --adapter gemini --model <model id>

# 4. score it (add prices per million tokens to get an estimated cost)
python score.py --run runs/run1 --truth runs/run1/ground_truth.csv --price-in <$> --price-out <$>
```

The result is `runs/run1/report.md`, plus `disagreements.csv` (every field where the ways of
reading differ, with your answer next to them) and `scores_by_field.csv`. Saved replies are in
`runs/run1/raw/`, so an interrupted run resumes where it stopped.

## The three arms

| Arm | What the model sees | Which fields |
|---|---|---|
| `full` | every page of the report | all of them |
| `snippets` | only the snippet crops (one request per document by default; `--mode field` = one per snippet) | the fields that got a snippet |
| `hybrid` | snippets first; a field falls back to the full-page answer when the snippet answer has no value, no `label_seen`, or low confidence | all of them |

The hybrid is what you would actually run. Its cost is the snippet request plus, only for
documents where something fell back, the full-page request. No extra model calls are needed to
score it: it is worked out from the other two arms.

## Files

| File | Purpose |
|---|---|
| `make_snippets.py` | step 1: cut the snippets (uses `../form_templates.py`, no OCR) |
| `run_comparison.py` | step 2: ask the model for each document, both ways |
| `score.py` | step 3: accuracy, fallback rate, confident-wrong cases, tokens, cost |
| `prompts.py` | the prompts for both arms (edit wording here, keep both arms equally careful) |
| `adapters.py` | model adapters: `dryrun`, `anthropic`, `gemini`, `custom` |
| `selftest.py` | checks the harness works, with no model |
| `docs_example.txt` | example list-of-documents file |

## Things to know

- The `anthropic` and `gemini` adapters were written from the providers' documented usage and
  have **not been run** by whoever wrote them. Run `--limit 2` first and read `runs/run1/raw/`.
- `make_snippets.py` rejects weak template matches (a page then gets no snippets and is read in
  full). `--min-inliers` (default 50) and `--min-margin` (default off; try 2) make it stricter.
  Re-run with different values to see the trade between coverage and wrong-place snippets.
- Pages that match no known form type get no snippets; that is the fallback case being tested,
  not an error. Forms covered so far: H-3, P-3, 154-A, P-2, U-1A (see `../form_templates/templates.json`).
- Don't copy your answer sheet from the model's output: it must be what is actually on the page.
