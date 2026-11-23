# Test plan: reading snippets vs reading the full page

## Why we are testing this

The form-template method finds each field on a known form and cuts a small image of it. If a
model can read those snippets about as accurately as it reads the whole page, we send far fewer
pixels per report. Whenever a snippet is missing or unusable, the pipeline falls back to the
full page, so a bad snippet should cost a full-page read, not a wrong answer.

What we do **not** yet know, because nobody has run a model on the snippets:

1. Does reading a snippet cost accuracy (or gain it, since a crop keeps full resolution while a
   full page may be downscaled)?
2. How many tokens / how much money does it save once fallbacks are included?
3. How often does a snippet look usable but give a **wrong** answer (the case the fallback
   cannot catch)?

## Design

Same documents, same model, same temperature (0), same answer format. The only thing that
changes is the image(s) the model sees.

| Arm | Input |
|---|---|
| A. `full` | all pages of the report (baseline; match how you send pages today) |
| B. `snippets` | one crop per field that the template method found |
| C. `hybrid` | B, with a field falling back to A's answer when B's answer is unusable |

C needs no extra model calls; it is computed from A and B.

A snippet answer is **usable** when it has a value, says which printed label or caption it saw
next to the value (`label_seen`), and is not marked low confidence. A crop that does not show
the label is treated as "not found", on the principle that we should only trust a value we can
see the label for.

## Choosing the documents

- **30-40 reports** to start. More is better: with about 300 fields scored, one or two points of
  difference is noise (roughly +/-3 to 5 points either way).
- Cover **every form type** you care about (H-3 front, P-3, 154-A, P-2, U-1A 1969, U-1A 1981),
  and both H-3 front layouts ("Identification / Boiler No." and "Unit Identification / ID Nos.").
- Include **hard cases on purpose**: faded or noisy scans, handwritten dates and board numbers,
  and a few pages of a form type that has **no template** (to exercise the fallback).
- Pick them **at random within each type**, not the cleanest ones. Don't use the exemplar pages
  the templates were drawn from (`../form_templates/*.png`).
- Pages must be upright. Give each report as one PDF with its pages in order (front, then back),
  or one image per page.

## Ground truth

For each document and field, write what is actually on the page. `N/A` if the field is not on the
document (for example the report date on the "Unit Identification" H-3 layout, or a blank
"Manufactured for"). Leave a row blank to skip it. If you can, have a second person check a
sample. Do not take the answers from the model being tested; if you pre-fill from another
system, verify every value by eye.

## Metrics (all in `report.md`)

| Metric | Why |
|---|---|
| Accuracy per arm, per field, per form template | the main comparison; the per-template table shows which forms the snippets fail on |
| Snippet answered / usable rate | how often a snippet is good enough to use |
| **Confident wrong** count (usable snippet answer, wrong value) | the failure the fallback cannot catch; review every one |
| Fallback rate (documents with at least one field falling back) | how often the saving is lost |
| Input / output tokens per arm, hybrid vs full | the cost question; add `--price-in/--price-out` for money |
| Paired disagreement counts | where A and B differ, who is right: more informative than a 1-point gap |

If the adapter cannot report token counts, `report.md` falls back to request, image and
megapixel counts as a proxy. Remember that images may be billed per token, per tile, or per
request: check what your model actually charges before trusting a pixel ratio.

## Suggested decision rule (adjust to your budget)

Adopt snippets-first if all of these hold:

1. hybrid accuracy is **within about 1 point of full** (or better), and the disagreement counts
   don't show the snippets losing on one particular field or template;
2. hybrid input tokens are **comfortably below full** (for example under about 70%), including
   the fallback cost;
3. **confident wrong is rare** (for example 1% or less of usable snippet answers), and the ones
   that exist are understood (wrong template? label ambiguous?);
4. the fallback rate is acceptable (for example under about a third of documents).

These numbers are starting points, not facts about this data.

## Variations worth running afterwards (each is one flag)

- `--full-max-side` at the sizes you really use, to see how much of the result depends on how the
  full page is downscaled.
- `--mode field` (one request per snippet) vs the default (one per document): does it change
  accuracy, and how much does the extra request overhead cost?
- A stricter `--min-inliers` or `--min-margin` in `make_snippets.py`: fewer snippets, fewer
  wrong-place ones. Does the confident-wrong count fall?
- Edit `prompts.py`, for example name the expected label for each field more explicitly.
- Add a fourth arm that pairs the full page with OCR text (your existing setup), to compare the
  "paired" result you already have against the hybrid on the same documents.

## Limits of this test

- It measures the model's reading, not the template method's locating: a snippet that fails to
  cover the field shows up here as a miss or a wrong answer, but the cause needs looking at
  (`runs/<run>/snippets/` next to `disagreements.csv`).
- It covers only the form types that have templates. How the pipeline behaves on form types that
  nobody has templated is the fallback path, and is only exercised by the unmatched pages you include.
- Small samples: treat a single run as an estimate, not a verdict.
