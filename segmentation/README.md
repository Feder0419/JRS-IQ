# segmentation

Scripts that split scanned PDF forms into one record per file, sorted by
form type, with flipped pages rotated back to right-side-up.

## Where to put the PDFs

Create a folder named `pdf` **next to** (outside of) this `segmentation`
folder, and drop the PDF files to process straight into it:

```
JRS-IQ/              <- repo root
├── pdf/              <- create this folder, put PDFs directly here (not in git)
│   ├── xxx.pdf
│   └── yyy.pdf
└── segmentation/      <- this repo's code
    └── program/
```

The scripts default to looking for `pdf/` two levels up from
`segmentation/program/`, so no code changes or extra arguments are needed.

## Setup

```
cd segmentation
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

## How to run

```
cd segmentation\program
python stage1_discover.py          # step 1: scan all PDFs, produce discovery_full/ for manual labeling
```

Once it finishes, open each cluster's folder under `discovery_full/clusters/`
(`thumbnail.png` for a quick look, `full/` for full-resolution pages you can
actually read) and fill in `discovery_full/labels_template.csv` directly
(one row per cluster: role / form code / rotation - see the comment block at
the top of stage1_discover.py for the column meanings), then run:

```
python ocr_titles.py               # step 2a: OCR the title of every front page (cached, incremental)
python stage2_segment.py           # step 2b: split, rotate, and write the results
```

Clusters group pages by layout only, so two refinements sit on top of the
cluster labels:

- `discovery_full/page_overrides.csv` - per-file / per-page-range rules
  (role, form code, rotation, pages per record) that win over the cluster
  labels. Used for the three small files, whose pages get merged into mixed
  clusters by the big files.
- `discovery_full/front_ocr.csv` - title OCR written by `ocr_titles.py`;
  stage 2 reads form codes from its title keywords (e.g. P-3's
  "SUPERHEATERS, WATERWALLS, AND ECONOMIZERS"), which separates forms that
  share a layout such as H-3 and P-3. Rerun `ocr_titles.py` after changing
  labels or overrides.

Form code priority: override > OCR > cluster label. Results land in
`../../segmented_form/form_<code>/`, along with a `manifest.csv` whose
`label_source` / `cluster_code` / `ocr_code` columns show where each
record's form code came from.

See the comment block at the top of each script for more detail on the
fields.

## New pipeline (page-level labels, fronts-only clustering) - in progress

Replaces the stage1/stage2 flow above step by step. Everything persistent
lives in `program/store/`; labels are kept per page, so re-clustering or
importing new PDFs never invalidates earlier tagging work.

```
python ingest.py            # 1. fingerprint + OCR every new page (incremental, both orientations)
python bootstrap_labels.py  # 2. old cluster tags -> store/page_labels.csv, with automatic fixes
python classify_roles.py    # 3. front / continuation / ignore + rotation for every page
python cluster_fronts.py    # 4. cluster fronts only, OCR health check, split mixed clusters
```

Files to review by hand:

- `store/label_fixes.csv` - every label the bootstrap changed automatically
  (ignore pages sitting between two fronts -> continuation; pages whose title
  OCR reads a form title -> front; front rotations corrected by OCR).
  To overrule one, edit its row in `store/page_labels.csv` and set
  `source` to `human`; human rows are never overwritten.
- `store/role_review.csv` - pages whose label the nearest-neighbour check
  disagrees with, and new pages it is not confident about (pending pool).
- `store/front_clusters.csv` + `store/front_clusters/cluster_XXX/` - one row
  per front-page cluster with an OCR health-check `status`
  (`ready` / `needs_review` / `split` / `pending`) and a `suggested_code`;
  fill in `form_code` / `confirmed`.
