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

Once it finishes, open every image under `discovery_full/clusters/`, fill in
`discovery_full/labels_template.csv` (one row per image: role / form code /
rotation needed), save it as `labels.csv` in the same folder, then run:

```
python stage2_segment.py           # step 2: split, rotate, and write the results
```

Results land in `../../segmented_form/<form_code>/`, along with a
`manifest.csv` you can use to spot-check the output.

See the comment block at the top of each script for more detail on the
fields.
