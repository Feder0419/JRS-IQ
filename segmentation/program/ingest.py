"""Step 1: fingerprint + OCR every page not yet in the store.

Incremental - drop new PDFs into pdf/ and rerun; pages already stored are
skipped. Writes store/fingerprints.npz, store/pages.csv, store/ocr.jsonl.
Checkpoints as it goes, so an interrupted run resumes where it stopped.

Usage:
    python ingest.py [--workers 16] [--pdf-dir ../../pdf]
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

import features
import store
from forms import has_front_keywords, ocr_form_code

PAGE_FIELDS = ["file", "page", "ink", "ocr_score0", "ocr_score180", "ocr_rot", "ocr_margin",
               "title_code0", "title_code180", "front_kw0", "front_kw180"]


def summarize(fname, f):
    s0, s1 = f["ocr"][0]["score"], f["ocr"][180]["score"]
    row = dict(file=fname, page=f["page"] + 1, ink=round(f["ink"], 4),
               ocr_score0=round(s0, 1), ocr_score180=round(s1, 1),
               ocr_rot=0 if s0 >= s1 else 180,
               ocr_margin=round(abs(s0 - s1) / max(s0, s1, 1.0), 3))
    for rot in (0, 180):
        title = f["ocr"][rot]["title"]
        row[f"title_code{rot}"] = ocr_form_code(title) or ""
        row[f"front_kw{rot}"] = int(has_front_keywords(title))
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf-dir", default=str(store.PDF_DIR))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    ap.add_argument("--chunk", type=int, default=20, help="pages per worker task")
    ap.add_argument("--checkpoint", type=int, default=1500, help="save every N pages")
    args = ap.parse_args()

    keys, arrays = store.load_fingerprints()
    pages = {(r["file"], int(r["page"]) - 1): r for r in store.read_csv(store.PAGES_CSV)}
    # a crash between the two saves leaves extra rows on one side - only trust both
    done = set(keys) & {store.key(f, p) for f, p in pages}
    if len(done) != len(keys):
        idx = [i for i, k in enumerate(keys) if k in done]
        keys = [keys[i] for i in idx]
        arrays = {k: v[idx] for k, v in arrays.items()}
    pages = {fp: r for fp, r in pages.items() if store.key(*fp) in done}

    counts = store.pdf_page_counts(args.pdf_dir)
    jobs = []
    for fname, n in counts.items():
        todo = [p for p in range(n) if store.key(fname, p) not in done]
        for k in range(0, len(todo), args.chunk):
            jobs.append((str(Path(args.pdf_dir) / fname), todo[k:k + args.chunk]))
    n_todo = sum(len(j[1]) for j in jobs)
    print(f"{len(done)} pages already stored, {n_todo} to ingest "
          f"across {len(counts)} files with {args.workers} workers")
    if not n_todo:
        return

    new_keys, new_fp, ocr_lines = [], {k: [] for k in store.FP_KINDS}, []

    def checkpoint():
        nonlocal keys, arrays
        if not new_keys:
            return
        stacked = {k: np.stack(new_fp[k]) for k in store.FP_KINDS}
        arrays = {k: np.concatenate([arrays[k], stacked[k]]) if arrays else stacked[k]
                  for k in store.FP_KINDS}
        keys = keys + new_keys
        with open(store.OCR_JSONL, "a", encoding="utf-8") as f:
            f.writelines(ocr_lines)
        store.save_fingerprints(keys, arrays)
        rows = sorted(pages.values(), key=lambda r: (r["file"], int(r["page"])))
        store.write_csv(store.PAGES_CSV, rows, PAGE_FIELDS)
        new_keys.clear(); ocr_lines.clear()
        for v in new_fp.values():
            v.clear()

    store.STORE.mkdir(parents=True, exist_ok=True)
    t0, n_done, since_ckpt = time.time(), 0, 0
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(features.worker, j) for j in jobs]
        for fut in as_completed(futures):
            pdf_path, results = fut.result()
            fname = Path(pdf_path).name
            for f in results:
                new_keys.append(store.key(fname, f["page"]))
                for k in store.FP_KINDS:
                    new_fp[k].append(f[k])
                pages[(fname, f["page"])] = summarize(fname, f)
                for rot in (0, 180):
                    o = f["ocr"][rot]
                    ocr_lines.append(json.dumps(dict(file=fname, page=f["page"] + 1, rot=rot,
                                                     title=o["title"], text=o["text"])) + "\n")
            n_done += len(results); since_ckpt += len(results)
            if since_ckpt >= args.checkpoint:
                checkpoint(); since_ckpt = 0
            el = time.time() - t0
            print(f"  {n_done}/{n_todo} pages  {el / 60:.1f} min elapsed, "
                  f"~{el / n_done * (n_todo - n_done) / 60:.1f} min left", flush=True)
    checkpoint()
    print(f"done: {len(keys)} pages in store")


if __name__ == "__main__":
    main()
