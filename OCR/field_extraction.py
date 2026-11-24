"""
Extracts only the handful of fields needed to file/look up a scanned report
(dates, serial/boiler numbers, material specs, manufacturer names, ...)
instead of transcribing the whole page. OCR is used for what it is good at:
reading the printed keywords ("Year built", "Serial No.", ...) to find where each
field is. A small cropped image of every field it locates is saved, so a vision
model (or a person) can read the value, handwriting included, from that crop
instead of the whole page. OCR's own reading is kept alongside, and doubtful
values are flagged for review.

Usage: python field_extraction.py [input] [--denoise] [--invert]
  input defaults to pdfs_to_process/. It may be a folder of PNGs/PDFs, a
  single PDF, or a single PNG.
  Gemini receives each original PDF/PNG through its Batch API. PaddleOCR runs
  only for missing/low-confidence fields (threshold configurable). PDF pages
  are rendered locally only when PaddleOCR is needed.
Outputs: output/field_index.csv and field_index.json, plus gemini_batches/.
Gemini requests are queued asynchronously; the script waits and resumes with
its saved batch/job state if restarted using the same --output-dir.
"""
from __future__ import annotations
import time
import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path

from PIL import Image

from field_crops import locate_field_crops
from form_templates import match_page
from field_dictionary import ALIAS_FUZZ_RATIO, FIELD_CONFIDENCE_THRESHOLD, FIELDS, KNOWN_WORDS
from llm_verify import verify_field
from pdf_preprocess import pdf_to_pngs

import os
from google import genai
from google.genai import types

OUTPUT_DIR = Path("output")
OUTPUT_CSV = OUTPUT_DIR / "field_index.csv"
OUTPUT_JSON = OUTPUT_DIR / "field_index.json"
FIELD_CROPS_DIR = OUTPUT_DIR / "field_crops"
PDF_PAGES_DIR = OUTPUT_DIR / "pdf_pages"
OCR_BOXES_DIR = OUTPUT_DIR / "ocr_boxes"
MAX_REJECTED_BOXES = 4   # a refused value cell is a few boxes at most
MAX_REJECTED_TEXT = 40   # ...and short; longer prose that doesn't fit a pattern is not a value cell
PDF_INPUT_DIR = Path("pdfs_to_process")  # default input: drop PDFs here and run with no arguments

#Turns a pdf into a list of images (one per page) which are then used in run_ocr 
def collect_images(input_path: Path, denoise: bool, invert: bool) -> list[tuple[str, list[Path]]]:
    """
    Resolve the input into (document name, page images) pairs, rendering any
    PDFs to PNG first. A PDF is one document with one image per page; a
    standalone PNG is a one-page document.
    """
    if input_path.is_dir():
        files = sorted(input_path.iterdir())
    else:
        files = [input_path]

    documents = []
    for f in files:
        suffix = f.suffix.lower()
        if suffix == ".png":
            documents.append((f.name, [f]))
        elif suffix == ".pdf":
            print(f"Rendering {f.name} at 300 DPI...")
            documents.append((f.name, pdf_to_pngs(f, PDF_PAGES_DIR, denoise=denoise, invert=invert)))
    return documents

#Stores text read in from ocr, confidence score, and a bounding box of the text
@dataclass
class OcrBox:
    text: str
    score: float
    box: list  # [x1, y1, x2, y2]: top-left and bottom-right pixel corners

#Actually run the ocr on the images
def run_ocr(ocr: PaddleOCR, image_path: Path) -> list[OcrBox]:
    boxes = []
    for result in ocr.predict(str(image_path)):
        data = result.json["res"]
        for text, score, box in zip(data["rec_texts"], data["rec_scores"], data["rec_boxes"]): #Records 3 lists, one of texts, their scores, and their bounding boxes
            # float()/int() convert NumPy types to plain Python ones so the
            # results can be written to JSON later.
            boxes.append(OcrBox(text=text, score=float(score), box=[int(v) for v in box]))
    return boxes

def run_gemini_batch(
    sources: list[Path],
    batch_size: int = 1000,
    max_active: int = 2,
    poll_seconds: int = 30,
    max_attempts: int = 2,
) -> dict:
    """Submit, monitor, collect, retry, and cache Gemini batch requests.

    Returns {document_filename: {field: {value, confidence}}}; a document whose
    Gemini requests fail gets an empty dict so existing PaddleOCR can still run.
    All persistent state lives under OUTPUT_DIR/gemini_batches/.
    """
    import hashlib
    import math

    if not sources:
        return {}
    if min(batch_size, max_active, poll_seconds, max_attempts) < 1:
        raise ValueError("batch_size, max_active, poll_seconds and max_attempts must be >= 1")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in PowerShell before running.")

    model = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
    work = OUTPUT_DIR / "gemini_batches"
    work.mkdir(parents=True, exist_ok=True)
    manifest_path = work / "jobs.json"
    results_path = work / "results.json"
    errors_path = work / "failed_documents.json"

    def atomic_json(path: Path, obj) -> None:
        temp = path.with_name(path.name + ".tmp")
        temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temp, path)

    def sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for piece in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(piece)
        return digest.hexdigest()

    prompt = f"""
Read this scanned manufacturing report and extract these fields:
{list(FIELDS.keys())}

Return a JSON object with exactly those field names. Each field must be an
object with 'value' (text or null) and 'confidence' (number 0 to 1 or null).
Preserve text exactly as written, including leading zeros and punctuation.
Never guess unreadable information. Use null for unreadable or absent values.
Return only valid JSON.
""".strip()
    names = [p.name for p in sources]
    if len(set(names)) != len(names):
        raise ValueError("Input document filenames must be unique")
    identity = {
        "model": model,
        "batch_size": batch_size,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "field_config": json.dumps(FIELDS, sort_keys=True, default=str),
        "inputs": {p.name: {"size": p.stat().st_size, "sha256": sha256_file(p)} for p in sources},
    }
    run_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("identity") != identity:
            raise RuntimeError(
                "Inputs, model, prompt or batch size differ from the saved run. "
                "Use a NEW --output-dir, or restore the original inputs/settings."
            )
    else:
        manifest = {"identity": identity, "jobs": {}}
        atomic_json(manifest_path, manifest)

    results = json.loads(results_path.read_text(encoding="utf-8")) if results_path.exists() else {}
    if not isinstance(results, dict):
        raise RuntimeError("Gemini results cache is corrupt: expected a JSON object")
    results = {key: value for key, value in results.items() if key in identity["inputs"]}
    errors = {}
    terminal = {
        "JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED", "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED", "JOB_STATE_PAUSED",
    }

    def state_name(job) -> str:
        state = getattr(job, "state", None)
        return getattr(state, "name", str(state or "UNKNOWN"))

    class FatalGeminiError(RuntimeError):
        """Bad credentials, model ID, or unsupported input; retry cannot fix it."""

    def api_call(label, call, retries=4):
        for i in range(retries):
            try:
                return call()
            except Exception as exc:
                code = getattr(exc, "code", None)
                # A bad key/model/request won't be repaired by waiting.
                if code in (400, 401, 403, 404):
                    error = FatalGeminiError(f"{label}: {exc}")
                    error.code = code
                    raise error from exc
                if i == retries - 1:
                    raise RuntimeError(f"{label} failed after {retries} attempts: {exc}") from exc
                wait = min(2 ** i * 3, 30)
                print(f"  {label} failed ({exc}); retrying in {wait}s")
                time.sleep(wait)

    def validate_response(payload):
        if not isinstance(payload, dict):
            raise ValueError("Gemini returned something other than a JSON object")
        if not any(name in payload for name in FIELDS):
            raise ValueError("Gemini returned no recognized fields")
        cleaned = {}
        for field_name in FIELDS:
            record = payload.get(field_name)
            if record is None:
                cleaned[field_name] = {"value": None, "confidence": None}
                continue
            if not isinstance(record, dict):
                raise ValueError(f"{field_name}: expected an object")
            value, conf = record.get("value"), record.get("confidence")
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{field_name}: value must be text or null")
            if conf is not None and (
                isinstance(conf, bool) or not isinstance(conf, (int, float))
                or not math.isfinite(conf) or not 0 <= conf <= 1
            ):
                raise ValueError(f"{field_name}: confidence must be 0..1 or null")
            cleaned[field_name] = {"value": value, "confidence": conf}
        return cleaned

    def download_nonempty(client, file_name: str) -> bytes:
        data = client.files.download(file=file_name)
        if not data:
            raise RuntimeError("Batch response file is empty or not ready yet")
        return data

    def parse_jsonl(data: bytes, expected: set[str]) -> None:
        if not data.strip():
            raise ValueError("Downloaded Gemini batch result is empty")
        parsed_count = 0
        for line_no, line in enumerate(data.decode("utf-8-sig").splitlines(), 1):
            row = None
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                name = row.get("key")
                if name not in expected:
                    print(f"  Unexpected result key at line {line_no}: {name!r}")
                    continue
                if row.get("error") or not row.get("response"):
                    errors[name] = f"Gemini returned an error: {row.get('error')}"
                    continue
                candidates = row["response"].get("candidates") or []
                parts = candidates[0]["content"]["parts"]
                response_text = "".join(part.get("text", "") for part in parts)
                fields = validate_response(json.loads(response_text))
                results[name] = fields
                errors.pop(name, None)
                parsed_count += 1
            except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
                name = row.get("key") if isinstance(row, dict) else None
                if name in expected:
                    errors[name] = f"Malformed response: {exc}"
                else:
                    print(f"  Invalid JSONL line {line_no}: {exc}")
        atomic_json(results_path, results)
        print(f"  Parsed {parsed_count} usable results; total {len(results)}/{len(sources)}")

    def safe_cleanup(client, job_info) -> None:
        # Don't delete documents before their batch is terminal.
        if job_info.get("cleaned"):
            return
        pending = []
        for file_name in job_info.get("uploads", []) + [job_info.get("request_file")]:
            if not file_name:
                continue
            try:
                client.files.delete(name=file_name)
            except Exception as exc:
                print(f"  Warning: could not clean up uploaded file {file_name}: {exc}")
                pending.append(file_name)
        job_info["uploads"] = pending
        job_info["request_file"] = None
        job_info["cleaned"] = not pending
        atomic_json(manifest_path, manifest)

    def recover_pending_job(client, display_name: str):
        # Narrow crash window: server accepted the create request but the client
        # stopped before it could save the new job name locally.
        try:
            for found in client.batches.list():
                if getattr(found, "display_name", None) == display_name:
                    return found.name
        except Exception as exc:
            print(f"  Could not search existing jobs: {exc}")
        return None

    chunks = [sources[i:i + batch_size] for i in range(0, len(sources), batch_size)]
    print(f"Gemini model: {model} | documents: {len(sources)} | chunks: {len(chunks)}")
    print(f"Resume state: {work}")
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    try:
        for attempt in range(max_attempts):
            if len(results) == len(sources):
                break
            print(f"\nGemini pass {attempt + 1}/{max_attempts}")
            for start in range(0, len(chunks), max_active):
                active = {}
                for idx in range(start, min(start + max_active, len(chunks))):
                    chunk = chunks[idx]
                    todo = [p for p in chunk if p.name not in results]
                    if not todo:
                        continue
                    job_key = f"try{attempt + 1}_batch{idx:05d}"
                    job_info = manifest["jobs"].get(job_key)
                    if job_info is None:
                        job_info = {
                            "name": None,
                            "display_name": f"jrs-iq-{run_id}-{job_key}",
                            "files": [], "uploads": [], "request_file": None,
                            "cleaned": False,
                        }
                        manifest["jobs"][job_key] = job_info
                        atomic_json(manifest_path, manifest)

                    if not job_info.get("name"):
                        # Reuse uploads and the JSONL if we crashed during create.
                        if not job_info.get("request_file"):
                            path = work / f"{job_key}.jsonl"
                            uploaded_names = []
                            accepted = []
                            with path.open("w", encoding="utf-8") as handle:
                                for source in todo:
                                    mime = "application/pdf" if source.suffix.lower() == ".pdf" else "image/png"
                                    if mime == "application/pdf" and source.stat().st_size > 50_000_000:
                                        errors[source.name] = "PDF exceeds Gemini's 50 MB PDF limit"
                                        continue
                                    try:
                                        file = api_call(
                                            f"Upload {source.name}",
                                            lambda p=source, m=mime: client.files.upload(
                                                file=str(p), config=types.UploadFileConfig(mime_type=m)
                                            ),
                                        )
                                        deadline = time.monotonic() + 300
                                        while state_name(file) == "PROCESSING":
                                            if time.monotonic() > deadline:
                                                raise RuntimeError("File processing exceeded five minutes")
                                            time.sleep(2)
                                            file = api_call(f"Check upload {source.name}",
                                                            lambda f=file: client.files.get(name=f.name))
                                        if state_name(file) == "FAILED" or not file.uri:
                                            raise RuntimeError("File processing failed or file URI missing")
                                        uploaded_names.append(file.name)
                                        accepted.append(source.name)
                                        request = {
                                            "key": source.name,
                                            "request": {
                                                "contents": [{"role": "user", "parts": [
                                                    {"fileData": {"mimeType": mime, "fileUri": file.uri}},
                                                    {"text": prompt},
                                                ]}],
                                                "generationConfig": {"responseMimeType": "application/json"},
                                            },
                                        }
                                        handle.write(json.dumps(request, ensure_ascii=False) + "\n")
                                    except FatalGeminiError:
                                        raise
                                    except Exception as exc:
                                        errors[source.name] = f"Upload error: {exc}"
                                        print(f"  {source.name}: {errors[source.name]}")
                            job_info["uploads"] = uploaded_names
                            job_info["files"] = accepted
                            atomic_json(manifest_path, manifest)
                            if not accepted:
                                print(f"{job_key}: no uploaded documents; retry on next pass")
                                safe_cleanup(client, job_info)
                                continue
                            try:
                                remote_jsonl = api_call(
                                    f"Upload JSONL {job_key}",
                                    lambda: client.files.upload(
                                        file=str(path), config=types.UploadFileConfig(mime_type="jsonl")
                                    ),
                                )
                                job_info["request_file"] = remote_jsonl.name
                                atomic_json(manifest_path, manifest)
                            except FatalGeminiError:
                                raise
                            except Exception as exc:
                                print(f"{job_key}: could not upload JSONL: {exc}")
                                for name in accepted:
                                    errors[name] = f"JSONL upload failed: {exc}"
                                safe_cleanup(client, job_info)
                                continue

                        # Only list existing jobs when recovering a previously
                        # prepared submission or after an ambiguous API error.
                        job_name = None
                        if job_info.get("submitted_before"):
                            job_name = recover_pending_job(client, job_info["display_name"])
                        if not job_name:
                            try:
                                # A create() retry might submit duplicate billable
                                # batches if the response was lost. Check the job
                                # list before issuing another create request.
                                for submit_attempt in range(3):
                                    job_info["submitted_before"] = True
                                    atomic_json(manifest_path, manifest)
                                    try:
                                        job = client.batches.create(
                                            model=model,
                                            src=job_info["request_file"],
                                            config={"display_name": job_info["display_name"]},
                                        )
                                        job_name = job.name
                                        break
                                    except Exception as exc:
                                        recovered = recover_pending_job(client, job_info["display_name"])
                                        if recovered:
                                            job_name = recovered
                                            break
                                        if getattr(exc, "code", None) in (400, 401, 403, 404):
                                            raise FatalGeminiError(f"Batch submission rejected: {exc}") from exc
                                        if submit_attempt == 2:
                                            raise
                                        time.sleep(3 * (submit_attempt + 1))
                            except FatalGeminiError:
                                raise
                            except Exception as exc:
                                print(f"{job_key}: submission failed: {exc}")
                                for name in job_info.get("files", []):
                                    errors[name] = f"Batch submission failed: {exc}"
                                continue
                        job_info["name"] = job_name
                        atomic_json(manifest_path, manifest)
                        print(f"Submitted {job_key}: {len(job_info['files'])} documents, {job_name}")
                    active[job_key] = job_info["name"]

                # All batch monitoring stays in the same execution.
                started = time.monotonic()
                while active:
                    for job_key, job_name in list(active.items()):
                        info = manifest["jobs"][job_key]
                        try:
                            job = api_call(f"Poll {job_key}", lambda j=job_name: client.batches.get(name=j))
                        except FatalGeminiError as exc:
                            if getattr(exc, "code", None) != 404:
                                raise
                            # An expired/deleted job can no longer be polled.
                            # Retry its documents in the next batch pass.
                            print(f"  {job_key}: job not found; retrying missing documents")
                            del active[job_key]
                            info["state"] = "JOB_STATE_EXPIRED"
                            atomic_json(manifest_path, manifest)
                            for name in info.get("files", []):
                                if name not in results:
                                    errors[name] = "Remote Gemini batch job expired or disappeared"
                            safe_cleanup(client, info)
                            continue
                        except Exception as exc:
                            # Leave job names saved for recovery after network failure.
                            raise RuntimeError(
                                f"Cannot retrieve {job_name}: {exc}. Rerun the SAME command to resume."
                            ) from exc
                        state = state_name(job)
                        if state not in terminal:
                            print(f"  {job_key}: {state}")
                            continue
                        del active[job_key]
                        info["state"] = state
                        atomic_json(manifest_path, manifest)
                        print(f"  {job_key}: {state}")
                        raw_path = work / f"{job_key}_responses.jsonl"
                        if raw_path.is_file():
                            raw = raw_path.read_bytes()
                        else:
                            output_file = getattr(getattr(job, "dest", None), "file_name", None)
                            if output_file:
                                try:
                                    raw = api_call(
                                        f"Download {job_key}",
                                        lambda f=output_file: download_nonempty(client, f),
                                    )
                                    raw_path.write_bytes(raw)
                                except Exception as exc:
                                    raw = b""
                                    print(f"  {job_key}: download failed: {exc}")
                            else:
                                raw = b""
                        if raw:
                            try:
                                parse_jsonl(raw, set(info.get("files", [])))
                            except Exception as exc:
                                print(f"  {job_key}: cannot parse downloaded JSONL: {exc}")
                        if state != "JOB_STATE_SUCCEEDED":
                            print(f"  {job_key}: job ended unsuccessfully; missing documents will be retried")
                        for name in info.get("files", []):
                            if name not in results and name not in errors:
                                errors[name] = f"No usable Gemini response ({state})"
                        safe_cleanup(client, info)
                    if active:
                        if time.monotonic() - started > 49 * 3600:
                            raise RuntimeError(
                                "Gemini jobs still running after 49 hours. "
                                "Use the SAME output folder to resume later."
                            )
                        time.sleep(poll_seconds)

            missing = [p.name for p in sources if p.name not in results]
            print(f"Pass {attempt + 1}: {len(results)}/{len(sources)} documents have Gemini results")
            if not missing:
                break

        # Any remaining failures go to the existing local PaddleOCR path rather
        # than preventing the entire document index from being written.
        missing = [p.name for p in sources if p.name not in results]
        atomic_json(errors_path, {name: errors.get(name, "No valid Gemini result after retries") for name in missing})
        if missing:
            print(f"WARNING: Gemini failed for {len(missing)} documents; using PaddleOCR and flagging review")
        return {p.name: results.get(p.name, {}) for p in sources}
    finally:
        client.close()


def group_into_rows(boxes: list[OcrBox]) -> list[list[OcrBox]]:
    """Group boxes into text rows by vertical-center overlap, tolerant of skew."""
    rows: list[list[OcrBox]] = []
    # Go through boxes top to bottom (by vertical center).
    for b in sorted(boxes, key=lambda b: (b.box[1] + b.box[3]) / 2):
        b_center = (b.box[1] + b.box[3]) / 2
        b_height = b.box[3] - b.box[1]
        for row in rows:
            row_center = sum((r.box[1] + r.box[3]) / 2 for r in row) / len(row)
            row_height = sum(r.box[3] - r.box[1] for r in row) / len(row)
            # Same row if the centers are within 60% of a text height of each
            # other; the tolerance absorbs scans that are slightly tilted.
            if abs(b_center - row_center) <= max(b_height, row_height) * 0.6:
                row.append(b)
                break
        else:
            # No existing row was close enough, so this box starts a new row.
            rows.append([b])
    # Read each row left to right, and order the rows top to bottom.
    for row in rows:
        row.sort(key=lambda b: b.box[0])
    rows.sort(key=lambda row: min(b.box[1] for b in row))
    return rows


def fuzzy_contains(haystack: str, needle: str, min_ratio: float) -> bool:
    """
    Word-level fuzzy match: every word in `needle` must have a close match
    among haystack's words. Deliberately word-level rather than a sliding
    character window over the whole phrase - on short/garbled OCR text a
    character-soup SequenceMatcher ratio hits false positives constantly
    (e.g. "al diamete" vs "boiler no " share enough letters in the wrong
    order to clear 0.75), so this is the difference between "manufactured
    by" matching only near that actual label vs. matching random table junk.
    """
    haystack_words = re.findall(r"[a-z0-9']+", haystack.lower())
    needle_words = needle.lower().split()
    for nw in needle_words:
        if not any(
            rw == nw or (len(nw) >= 3 and SequenceMatcher(None, rw, nw).ratio() >= min_ratio)
            for rw in haystack_words
        ):
            return False
    return True


def clean_text(text: str) -> str:
    """Drop non-ASCII characters (OCR noise like stray CJK glyphs) so values are safe to write anywhere."""
    return text.encode("ascii", "ignore").decode("ascii").strip()


def find_value(spec: dict, pattern: re.Pattern, text: str, rejections: list | None = None) -> str | None:
    """
    First pattern match in `text` that is an acceptable value for this field:
    not a known label word, containing a digit if the field requires one, and
    (for names) mostly letters. If `rejections` is given, the reason each
    candidate was turned down is appended to it.
    """
    reject = spec.get("reject_words", ())
    cleaned = clean_text(text)
    matched_any = False
    for m in pattern.finditer(cleaned):
        matched_any = True
        value = m.group(0)
        reason = None
        if value.lower() in reject:
            reason = "label word, not a value"
        elif spec.get("require_digit") and not any(c.isdigit() for c in value):
            reason = "contains no digit"
        elif spec.get("min_alpha_ratio"):
            # mostly digits/symbols: noise, not a name
            chars = [c for c in value if not c.isspace()]
            if not chars or sum(c.isalpha() for c in chars) / len(chars) < spec["min_alpha_ratio"]:
                reason = "mostly digits/symbols"
        if reason is None and spec.get("require_capital") and not re.search(r"\b[A-Z][A-Za-z]{2,}", value):
            reason = "no capitalized word"  # lowercase text is body copy misread as a name
        if reason:
            if rejections is not None:
                rejections.append((value, reason))
            continue
        return value
    if rejections is not None and not matched_any:
        rejections.append((cleaned, "does not fit the field's pattern"))
    return None


def autocorrect_words(text: str) -> tuple[str, list[str]]:
    """
    Replace words that are a near miss of a KNOWN_WORDS entry ("Aqud-Chem" ->
    "Aqua-Chem"). Conservative: 5+ letters and a very close match only, so real
    words are left alone. Returns the new text and the changes made.
    """
    known = {w.lower(): w for w in KNOWN_WORDS}
    changes = []

    def fix(m: re.Match) -> str:
        word = m.group(0)
        if len(word) < 5 or word.lower() in known:
            return word
        best = max(known, key=lambda k: SequenceMatcher(None, word.lower(), k).ratio())
        if SequenceMatcher(None, word.lower(), best).ratio() >= 0.85:
            changes.append(f"{word} -> {known[best]}")
            return known[best]
        return word

    return re.sub(r"[A-Za-z][A-Za-z-]*", fix, text), changes


def extract_fields(rows: list[list[OcrBox]]) -> tuple[dict, dict]:
    """
    For each field: scan rows for a label alias, take the OCR boxes after it
    in that row as the value. If no labeled row matches (label itself didn't
    survive OCR), fall back to a full-document regex scan for fields whose
    value has a distinctive enough shape to find unlabeled (e.g. SA53B).

    Returns (found, rejected). `rejected` maps a field to the label-adjacent
    boxes whose text failed validation (e.g. an "H" where a board number should
    be), with the reason. Where to crop for each field is decided separately, in
    field_crops.py.
    """
    found = {}
    rejected = {}

    for field_name, spec in FIELDS.items():
        pattern = re.compile(spec["pattern"], re.IGNORECASE)

        skip_words = spec.get("skip_row_words", ())

        for row_index, row in enumerate(rows):
            row_text = " ".join(b.text for b in row)
            if any(fuzzy_contains(row_text, w, ALIAS_FUZZ_RATIO) for w in skip_words):
                continue  # e.g. a certificate-expiry line, not the report date
            matched_alias = next(
                (a for a in spec["aliases"] if fuzzy_contains(row_text, a, ALIAS_FUZZ_RATIO)),
                None,
            )
            if not matched_alias:
                continue

            # Find where the label ends: the last box in the row that looks like
            # the label's final word (e.g. "by" in "manufactured by"). Everything
            # to the right of it is treated as the value.
            alias_last_word = matched_alias.split()[-1].lower()
            split_idx = 0
            for i, b in enumerate(row):
                if alias_last_word in b.text.lower() or \
                   SequenceMatcher(None, b.text.lower(), alias_last_word).ratio() > 0.6:
                    split_idx = i + 1

            # Candidate places for the value: to the right of the label in the
            # same row, then (these forms are tables) in the cell directly below.
            candidates = []
            if row[split_idx:]:
                candidates.append(("same row", row[split_idx:]))
            # (Free-text fields skip this: the cell below is usually a printed caption.)
            below = boxes_below_label(row, split_idx, matched_alias, rows) if spec.get("look_below", True) else []
            if below:
                candidates.append(("cell below", below))

            value_boxes, value_text = None, None
            for where, group in candidates:
                text = " ".join(b.text for b in group).strip(" .:-")
                reasons = []
                value_text = find_value(spec, pattern, text, reasons)
                if value_text is not None:
                    value_boxes = group
                    break
                # Not accepted as the value, but remember where it was and why.
                # Only candidates that look like a value cell are kept: the
                # label words also fuzzy-match rows of body text ("DATA REPORT"),
                # and a long run of prose isn't something to re-read. Every box on
                # the page is still saved in ocr_boxes/.
                reason_text = "; ".join(dict.fromkeys(r for _, r in reasons))
                if len(group) > MAX_REJECTED_BOXES or (
                    len(text) > MAX_REJECTED_TEXT and reason_text == "does not fit the field's pattern"
                ):
                    continue
                rejected.setdefault(field_name, []).append({
                    "text": text, "where": where, "reason": reason_text, "row": row_index,
                    "score": round(min(b.score for b in group), 3),
                    # one entry per OCR box (same order), so a single cell can be re-read
                    "boxes": [b.box for b in group],
                    "box_texts": [b.text for b in group],
                    "box_scores": [round(b.score, 3) for b in group],
                })
            if value_text is None:
                # Nothing fit the expected shape (date, spec code, ...) - don't
                # keep raw noise as the value (the boxes are kept in `rejected`).
                continue

            # Take the first (topmost) match, not the highest-confidence one:
            # these forms put report metadata near the top, and a later row
            # that coincidentally matches the same short alias (e.g. "date"
            # inside a certificate-expiry line) is more likely to be a false
            # positive than the genuine field, however it happened to OCR.
            conf = min(b.score for b in value_boxes)
            found[field_name] = {
                "value": value_text, "confidence": conf,
                "boxes": [b.box for b in value_boxes], "source": "label",
            }
            if spec.get("vocabulary"):
                found[field_name]["value"], fixes = autocorrect_words(value_text)
                if fixes:
                    found[field_name]["autocorrected"] = fixes
            break

        # No usable label row: scan every row for text shaped like the value.
        # These matches are marked "pattern_fallback" and always get reviewed.
        # Fields with "fallback": False (serial/board/boiler numbers) are never
        # guessed this way - any digit string would match - so they stay unfound.
        if field_name not in found and spec.get("fallback", True):
            for row in rows:
                row_text = " ".join(b.text for b in row)
                if any(fuzzy_contains(row_text, w, ALIAS_FUZZ_RATIO) for w in skip_words):
                    continue
                value = find_value(spec, pattern, row_text)
                if value:
                    contributing = [b for b in row if b.text and b.text in value]
                    found[field_name] = {
                        "value": value,
                        "confidence": min((b.score for b in contributing), default=0.5),
                        "boxes": [b.box for b in contributing] or [row[0].box],
                        "source": "pattern_fallback",
                    }
                    break

    apply_sanity_checks(found)
    return found, rejected


def value_ok(spec: dict, text: str) -> bool:
    """Does `text` have the shape of this field's value (used to judge whether a crop zone holds one)?"""
    return find_value(spec, re.compile(spec["pattern"], re.IGNORECASE), text) is not None


def save_field_crops(image_path: Path, rows: list[list[OcrBox]], found: dict) -> tuple[dict, dict]:
    """
    Save a small image of every located field to field_crops/<page>__<field>.png
    and return ({field: {crop_file, crop_box, crop_kind}}, form_template_info).
    These crops, not the whole page, are what a vision model should be sent.

    If the page is lined up with a known form type (form_templates/), that form's
    field rectangles are used and only those fields get a crop (a field the form
    does not have on this page gets none). Otherwise the crop is placed from the
    keywords OCR found. A field with no crop is left for the model to find on the
    full page.
    """
    FIELD_CROPS_DIR.mkdir(parents=True, exist_ok=True)
    saved = {}
    with Image.open(image_path) as page:
        template = match_page(page)
        if template["template"]:
            located = template["fields"]
        else:
            located = locate_field_crops(rows, found, page.width, page.height, fuzzy_contains, value_ok)
        for field_name, crop in located.items():
            path = FIELD_CROPS_DIR / f"{image_path.stem}__{field_name}.png"
            page.crop(tuple(crop["box"])).save(path)
            saved[field_name] = {
                "crop_file": f"{FIELD_CROPS_DIR.name}/{path.name}", "crop_box": crop["box"], "crop_kind": crop["kind"],
            }
    return saved, {"name": template["template"], "inliers": template["inliers"]}


def find_label_boxes(row: list[OcrBox], split_idx: int, alias: str, whole_row_fallback: bool = False) -> list[OcrBox]:
    """
    The boxes in `row` that make up the label (the matched keyword), not its
    neighbours. If none stand out, returns [] - or the whole row when
    `whole_row_fallback` is set (used for cropping, never for finding values).
    """
    alias_words = alias.lower().split()
    boxes = [
        b for b in row[:split_idx]
        if any(w in b.text.lower() or SequenceMatcher(None, b.text.lower(), w).ratio() > 0.6 for w in alias_words)
    ] or row[max(split_idx - 1, 0):split_idx]
    return boxes or (row if whole_row_fallback else [])


def boxes_below_label(row: list[OcrBox], split_idx: int, alias: str, rows: list[list[OcrBox]]) -> list[OcrBox]:
    """
    Boxes in the table cell directly under a label: the nearest row below it
    that overlaps the label's horizontal span (padded a little, since values
    are often centered or indented relative to their label).
    """
    label_boxes = find_label_boxes(row, split_idx, alias)
    if not label_boxes:
        return []

    x1 = min(b.box[0] for b in label_boxes)
    x2 = max(b.box[2] for b in label_boxes)
    y2 = max(b.box[3] for b in label_boxes)
    height = max(b.box[3] - b.box[1] for b in label_boxes)
    pad = (x2 - x1) * 0.25

    candidates = []
    for other in rows:
        if other is row:
            continue
        for b in other:
            center_y = (b.box[1] + b.box[3]) / 2
            # below the label, but not so far down it's a different section
            if b.text.lstrip().startswith("("):
                continue  # printed caption like "(Name of part...)", not a filled-in value
            if y2 - height * 0.3 < center_y < y2 + height * 4 and b.box[2] > x1 - pad and b.box[0] < x2 + pad:
                candidates.append(b)
    if not candidates:
        return []

    candidates.sort(key=lambda b: (b.box[1] + b.box[3]) / 2)
    first_y = (candidates[0].box[1] + candidates[0].box[3]) / 2
    nearest = [b for b in candidates if abs((b.box[1] + b.box[3]) / 2 - first_y) <= height * 0.6]
    return sorted(nearest, key=lambda b: b.box[0])


# Common ASME boiler/pressure-vessel material specs (SA-###), used to catch
# misreads like "SA 31". Not exhaustive: an unlisted spec is flagged for a
# human to confirm, not rejected.
KNOWN_MATERIAL_SPECS = {
    36, 53, 105, 106, 178, 179, 181, 182, 192, 193, 194, 210, 213, 214, 216, 234,
    240, 266, 278, 285, 299, 302, 307, 312, 333, 336, 351, 515, 516, 517, 533, 537,
    570, 612, 675,
}
EARLIEST_REPORT_YEAR = 1900


def apply_sanity_checks(found: dict) -> None:
    """
    Cross-check extracted values for plausibility. A failing value is kept (so
    the reviewer can see it) but gets a "flag" that forces human review.
    """
    year_built = int(found["year_built"]["value"]) if "year_built" in found else None

    entry = found.get("report_date")
    if entry:
        year_token = re.findall(r"\d+", entry["value"])[-1]
        year = int(year_token)
        if year_token in ("19", "20"):
            # Just the century: the last two digits of the year were lost.
            entry.setdefault("flag", f"report_date year looks cut off ('{year_token}')")
        if year < 100:
            year += 1900  # two-digit year on these forms means 19xx
        if not EARLIEST_REPORT_YEAR <= year <= datetime.now().year:
            entry.setdefault("flag", f"report_date year {year} is not plausible")
        elif year_built and year < year_built:
            entry.setdefault("flag", f"report_date year {year} is before year built {year_built}")

    entry = found.get("material_spec")
    if entry:
        value = entry["value"].replace("_", "-")
        m = re.match(r"S([AB])\s?-?(\d+)", value, re.IGNORECASE)
        if m:
            # normalize "SA 516" -> "SA-516" and "SA_285-C" -> "SA-285-C", keeping any grade
            entry["value"] = f"S{m.group(1).upper()}-{m.group(2)}" + value[m.end():]
            if int(m.group(2)) not in KNOWN_MATERIAL_SPECS:
                entry.setdefault("flag", f"{entry['value']} is not a known material spec")

    # On these forms the boiler number and the manufacturer's serial number are
    # the same value, so a mismatch means at least one was misread. The national
    # board number is a different number: if it matches either, it was copied.
    boiler, serial, board = (found.get(n) for n in ("boiler_no", "serial_no", "national_board_no"))
    if boiler and serial and boiler["value"] != serial["value"]:
        for entry in (boiler, serial):
            entry.setdefault("flag", "boiler_no and serial_no differ (normally identical on these forms)")
    if board:
        for other_name, other in (("serial_no", serial), ("boiler_no", boiler)):
            if other and board["value"] == other["value"]:
                board.setdefault("flag", f"same value as {other_name} (should be a different number)")
    if serial and year_built and serial["value"] == str(year_built):
        serial.setdefault("flag", "same as year built")


def save_ocr_boxes(image_path: Path, boxes: list[OcrBox]) -> str:
    """
    Write every box OCR found on the page (text, score, [x1, y1, x2, y2]),
    whether or not it ended up in a field. Returns the file's path relative to
    the output folder. This is the raw material for re-reading a region with a
    different recognizer without re-running OCR.
    """
    OCR_BOXES_DIR.mkdir(parents=True, exist_ok=True)
    path = OCR_BOXES_DIR / f"{image_path.stem}.json"
    path.write_text(json.dumps(
        {"image": image_path.name, "boxes": [{"text": b.text, "score": round(b.score, 4), "box": b.box} for b in boxes]},
        indent=1,
    ), encoding="utf-8")
    return f"{OCR_BOXES_DIR.name}/{path.name}"


def process_document(ocr: PaddleOCR, image_path: Path) -> dict:
    """OCR one page image, extract its fields, and decide whether a human must review it."""
    ocr_boxes = run_ocr(ocr, image_path)
    boxes_file = save_ocr_boxes(image_path, ocr_boxes)
    rows = group_into_rows(ocr_boxes)
    extracted, rejected = extract_fields(rows)
    crops, form_template = save_field_crops(image_path, rows, extracted)
    no_crop = {"crop_file": None, "crop_box": None, "crop_kind": None}

    # "escalate" means a person should look at this page; the reasons say why.
    result = {
        "document": image_path.name, "ocr_boxes_file": boxes_file, "form_template": form_template,
        "fields": {}, "escalate": False, "escalation_reasons": [],
    }

    for field_name, spec in FIELDS.items():
        entry = extracted.get(field_name)
        if entry is None:
            # Field wasn't found at all; only a problem if the field is required.
            # Any refused candidate boxes are kept so something else can re-read them.
            result["fields"][field_name] = {
                "value": None, "confidence": None, "llm_checked": False,
                "boxes": [], "rejected": rejected.get(field_name, []),
                **crops.get(field_name, no_crop),
            }
            if spec["required"]:
                result["escalate"] = True
                result["escalation_reasons"].append(f"{field_name}: not found")
            continue

        confidence = entry["confidence"]
        llm_checked = False

        # A fallback match found its value's *shape* (a date, a spec code)
        # somewhere in the document without ever confirming it sits next to
        # the field's actual label - so it can OCR with high confidence and
        # still be the wrong field's value entirely (e.g. picking up a
        # certificate-expiry date instead of the report date). Confidence
        # alone can't catch that, so fallback matches always get reviewed.
        needs_review = (
            confidence < FIELD_CONFIDENCE_THRESHOLD
            or entry["source"] == "pattern_fallback"
            or bool(entry.get("flag"))  # failed a plausibility check
        )

        if needs_review:
            # Ask the LLM verifier for a second opinion (a no-op without an API
            # key), looking at the same field crop that was saved above.
            crop = crops.get(field_name)
            verify_boxes = [crop["crop_box"]] if crop else entry["boxes"]
            verdict = verify_field(image_path, verify_boxes, field_name, entry["value"])
            llm_checked = verdict.used_llm

            # Outcomes, in order: LLM agrees -> accept; LLM disagrees -> use its
            # text but still escalate; no LLM -> escalate for a human to decide.
            if verdict.used_llm and verdict.agrees and entry["source"] == "label" and not entry.get("flag"):
                pass  # LLM confirmed a labeled OCR reading despite low confidence; don't escalate
            elif verdict.used_llm and verdict.llm_text:
                entry["value"] = verdict.llm_text
                result["escalate"] = True
                result["escalation_reasons"].append(
                    f"{field_name}: LLM correction '{verdict.llm_text}', needs human confirmation"
                )
            elif entry.get("flag"):
                result["escalate"] = True
                result["escalation_reasons"].append(f"{field_name}: {entry['flag']}")
            elif entry["source"] == "pattern_fallback":
                result["escalate"] = True
                result["escalation_reasons"].append(
                    f"{field_name}: label not found, value guessed from pattern match elsewhere in document"
                )
            else:
                result["escalate"] = True
                result["escalation_reasons"].append(f"{field_name}: low confidence ({confidence:.2f})")

        result["fields"][field_name] = {
            "value": entry["value"], "confidence": round(confidence, 3),
            "source": entry["source"], "llm_checked": llm_checked,
            "boxes": entry["boxes"], "rejected": rejected.get(field_name, []),
            **crops.get(field_name, no_crop),
            **({"autocorrected": entry["autocorrected"]} if entry.get("autocorrected") else {}),
        }

    return result


def merge_pages(doc_name: str, page_results: list[dict]) -> dict:
    """
    Combine the per-page results of one PDF into a single document entry.

    For each field, take the value from the first page that found it by label;
    if no page found it by label, take the earliest pattern-fallback guess.
    Preferring the earliest page matters because back pages often yield
    high-confidence junk (e.g. the "FORM U-1A (BACK)" title). Escalation is
    then recomputed from the merged fields, so a field missing on the back page
    doesn't flag a document whose front page had it.
    """
    merged = {
        "document": doc_name,
        "pages": [p["document"] for p in page_results],
        "ocr_boxes_files": [p["ocr_boxes_file"] for p in page_results],
        "form_templates": [p["form_template"] for p in page_results],
        "fields": {}, "escalate": False, "escalation_reasons": [],
    }

    for field_name in FIELDS:
        # (page result, field entry) pairs that found a value, in page order
        found = [(p, p["fields"][field_name]) for p in page_results if p["fields"][field_name]["value"] is not None]
        labeled = [pair for pair in found if pair[1].get("source") == "label"]
        # With no value anywhere, still point at a page that has a crop for the field.
        cropped = [(p, p["fields"][field_name]) for p in page_results if p["fields"][field_name].get("crop_file")]
        source_page, entry = (labeled or found or cropped or [(page_results[0], page_results[0]["fields"][field_name])])[0]

        merged["fields"][field_name] = dict(entry)
        if found:
            merged["fields"][field_name]["page"] = source_page["document"]
        else:
            # No page produced a value: keep the refused candidates from every
            # page (each tagged with its page), not just the first one's.
            merged["fields"][field_name]["rejected"] = [
                {**r, "page": p["document"]}
                for p in page_results for r in p["fields"][field_name].get("rejected", [])
            ]

        # Keep only the reasons about this field from the page that supplied it.
        reasons = [r for r in source_page["escalation_reasons"] if r.startswith(f"{field_name}:")]
        if reasons:
            merged["escalate"] = True
            merged["escalation_reasons"].extend(reasons)

    return merged


def main():
    global OUTPUT_DIR, OUTPUT_CSV, OUTPUT_JSON, FIELD_CROPS_DIR, PDF_PAGES_DIR, OCR_BOXES_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("input", nargs="?", type=Path, default=PDF_INPUT_DIR)
    parser.add_argument("--denoise", action="store_true", help="light denoise on PDF-derived pages")
    parser.add_argument("--invert", action="store_true", help="invert PDF-derived pages")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="where to write results (use a scratch folder to avoid overwriting reviewed output)")
    parser.add_argument("--batch-size", type=int, default=5,
                        help="documents per Gemini batch job; start with 5, increase after testing")
    parser.add_argument("--batch-concurrency", type=int, default=2,
                        help="maximum active Gemini batch jobs")
    parser.add_argument("--batch-poll-seconds", type=int, default=30,
                        help="seconds between batch status checks")
    parser.add_argument("--batch-attempts", type=int, default=2,
                        help="number of Gemini attempts per document")
    parser.add_argument("--gemini-confidence-threshold", type=float, default=0.0,
                        help="also run PaddleOCR if Gemini confidence is below this threshold (default: missing fields only)")
    args = parser.parse_args()

    if not 0 <= args.gemini_confidence_threshold <= 1:
        parser.error("--gemini-confidence-threshold must be between 0 and 1")

    OUTPUT_DIR = args.output_dir
    OUTPUT_CSV, OUTPUT_JSON = OUTPUT_DIR / "field_index.csv", OUTPUT_DIR / "field_index.json"
    FIELD_CROPS_DIR, PDF_PAGES_DIR = OUTPUT_DIR / "field_crops", OUTPUT_DIR / "pdf_pages"
    OCR_BOXES_DIR = OUTPUT_DIR / "ocr_boxes"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    
    # Find original PDF/PNG files WITHOUT rendering them yet.
    if args.input.is_dir():
        sources = sorted(
            p for p in args.input.iterdir()
            if p.is_file() and p.suffix.lower() in (".pdf", ".png")
        )
    elif args.input.is_file() and args.input.suffix.lower() in (".pdf", ".png"):
        sources = [args.input]
    else:
        sources = []

    if not sources:
        print(f"No PDFs or PNGs found in {args.input}.")
        return

    FIELD_CROPS_DIR.mkdir(parents=True, exist_ok=True)

    # Run Gemini Batch API for all documents.
    gemini_by_doc = run_gemini_batch(
        sources,
        batch_size=args.batch_size,
        max_active=args.batch_concurrency,
        poll_seconds=args.batch_poll_seconds,
        max_attempts=args.batch_attempts,
    )
    batch_failures = json.loads(
        (OUTPUT_DIR / "gemini_batches" / "failed_documents.json").read_text(encoding="utf-8")
    )

    # Do not initialize PaddleOCR unless Gemini misses a field.
    ocr = None
    all_results = []

    def has_value(value):
        return value is not None and bool(str(value).strip())

    def normalize(value):
        return re.sub(r"[^a-z0-9]", "", str(value or "").lower())

    # Keep a local per-document checkpoint, too: a restart shouldn't need to
    # redo expensive PaddleOCR for documents already merged successfully.
    import hashlib
    batch_folder = OUTPUT_DIR / "gemini_batches"
    resume_identity = {
        "job_manifest": json.loads((batch_folder / "jobs.json").read_text(encoding="utf-8"))["identity"],
        "gemini_results_sha256": hashlib.sha256((batch_folder / "results.json").read_bytes()).hexdigest()
                                 if (batch_folder / "results.json").exists() else None,
        "gemini_failures_sha256": hashlib.sha256((batch_folder / "failed_documents.json").read_bytes()).hexdigest(),
        "denoise": args.denoise, "invert": args.invert,
        "gemini_confidence_threshold": args.gemini_confidence_threshold,
    }
    progress_path = OUTPUT_DIR / "field_index_progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("identity") != resume_identity:
            progress = {"identity": resume_identity, "documents": {}}
    else:
        progress = {"identity": resume_identity, "documents": {}}

    def checkpoint_progress():
        temp = progress_path.with_name(progress_path.name + ".tmp")
        temp.write_text(json.dumps(progress, indent=2), encoding="utf-8")
        os.replace(temp, progress_path)

    def blank_document(doc_name):
        return {
            "document": doc_name, "pages": [], "ocr_boxes_files": [],
            "fields": {name: {
                "value": None, "confidence": None, "llm_checked": False,
                "boxes": [], "rejected": [],
                "crop_file": None, "crop_box": None, "crop_kind": None,
            } for name in FIELDS},
            "escalate": False, "escalation_reasons": [],
        }

    for source in sources:
        doc_name = source.name
        if doc_name in progress["documents"]:
            print(f"\nResuming {doc_name}: already processed locally")
            all_results.append(progress["documents"][doc_name])
            continue
        print(f"\nProcessing {doc_name}...")

        # STEP 1: Read the completed Gemini batch result.
        gemini_results = gemini_by_doc[source.name]

        # STEP 2: Check which fields Gemini missed.
        missing_fields = [
            name for name in FIELDS
            if (
                not has_value((gemini_results.get(name) or {}).get("value"))
                or (
                    args.gemini_confidence_threshold > 0
                    and ((gemini_results.get(name) or {}).get("confidence") is None
                         or (gemini_results.get(name) or {}).get("confidence")
                            < args.gemini_confidence_threshold)
                )
            )
        ]

        paddle_ran = bool(missing_fields)
        paddle_error = None

        if missing_fields:
            print(f"Gemini missing: {missing_fields}")
            print("Running PaddleOCR fallback...")

            # Initialize PaddleOCR only when needed.
            if ocr is None:
                from paddleocr import PaddleOCR

                #If have enough compute try: PP-OCRv6_medium_det, PP-OCRv6_medium_rec
                ocr = PaddleOCR(
                    lang="en",
                    device="cpu",
                    text_detection_model_name="PP-OCRv6_small_det",
                    text_recognition_model_name="PP-OCRv6_small_rec",
                    enable_mkldnn=False,
                    cpu_threads=1,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )

            # Render only this document and run existing OCR.
            try:
                rendered = collect_images(source, args.denoise, args.invert)
                _, image_paths = rendered[0]
                if not image_paths:
                    raise ValueError("Document had no renderable pages")
                page_results = []
                for image_path in image_paths:
                    print(f"Processing {image_path.name}...")
                    page_results.append(process_document(ocr, image_path))
                merged = merge_pages(doc_name, page_results)
            except Exception as exc:
                # Keep processing other documents, but explicitly flag this one.
                paddle_error = f"PaddleOCR processing failed: {type(exc).__name__}: {exc}"
                print(f"WARNING: {doc_name}: {paddle_error}")
                merged = blank_document(doc_name)

        else:
            print("All fields found. Skipping PaddleOCR.")
            merged = blank_document(doc_name)

        # Reset PaddleOCR warnings.
        # Review flags will be rebuilt from final selected values.
        merged["escalate"] = False
        merged["escalation_reasons"] = []

        # STEP 3: Merge Gemini results with optional PaddleOCR.
        if doc_name in batch_failures:
            merged["escalate"] = True
            merged["escalation_reasons"].append(
                f"Gemini batch failed: {batch_failures[doc_name]}; PaddleOCR fallback needs review"
            )
        if paddle_error:
            merged["escalate"] = True
            merged["escalation_reasons"].append(paddle_error)
        for name in FIELDS:
            field = merged["fields"][name]

            p_value = field["value"]
            p_conf = field["confidence"]

            g = gemini_results.get(name) or {}
            g_value = g.get("value")
            g_conf = g.get("confidence")

            agree = (
                has_value(g_value)
                and has_value(p_value)
                and normalize(g_value) == normalize(p_value)
            )

            if agree:
                selected = g_value
                source_model = "both"
                status = "agreed"

            elif has_value(g_value):
                selected = g_value
                source_model = "gemini"
                status = (
                    "review_required"
                    if has_value(p_value)
                    else "gemini_only"
                )

            elif has_value(p_value):
                selected = p_value
                source_model = "paddleocr"
                status = "review_required"

            else:
                selected = None
                source_model = "none"
                status = "not_found"

            # Preserve both readings.
            field["paddleocr"] = {
                "value": p_value,
                "confidence": p_conf,
                "ran": paddle_ran,
                **({"error": paddle_error} if paddle_error else {}),
            }

            field["gemini"] = {
                "value": g_value,
                "confidence": g_conf,
                **({"error": batch_failures[doc_name]} if doc_name in batch_failures else {}),
            }

            field["selection"] = {
                "value": selected,
                "source": source_model,
                "status": status,
            }

            field["value"] = selected
            field["confidence"] = (
                g_conf
                if source_model in ("gemini", "both")
                else p_conf
            )

            if status == "review_required":
                merged["escalate"] = True
                merged["escalation_reasons"].append(
                    f"{name}: model disagreement or PaddleOCR fallback"
                )

            elif status == "not_found" and FIELDS[name]["required"]:
                merged["escalate"] = True
                merged["escalation_reasons"].append(
                    f"{name}: required field not found"
                )

        all_results.append(merged)
        # Do not cache a failed local-OCR attempt; a restart can retry it
        # after dependencies or the source file have been repaired.
        if not paddle_error:
            progress["documents"][doc_name] = merged
            checkpoint_progress()


    # Each page saved crops for every keyword it found, but a PDF's merged result
    # uses only one page's crop per field. Drop the rest so field_crops/ holds just
    # the images the output points at (the ones to send to a vision model).
    used = {f["crop_file"] for r in all_results for f in r["fields"].values() if f.get("crop_file")}
    for crop in FIELD_CROPS_DIR.glob("*.png"):
        if f"{FIELD_CROPS_DIR.name}/{crop.name}" not in used:
            crop.unlink()

    # JSON keeps the full detail (confidence, source, LLM status) per field;
    # the CSV below is a flat one-row-per-document summary of values only.
    OUTPUT_JSON.write_text(json.dumps(all_results, indent=2))

    field_names = list(FIELDS.keys())
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["document", "escalate", "escalation_reasons"] + field_names)
        for r in all_results:
            writer.writerow(
                [r["document"], r["escalate"], "; ".join(r["escalation_reasons"])]
                + [r["fields"][fn]["value"] or "" for fn in field_names]
            )

    n_escalate = sum(1 for r in all_results if r["escalate"])
    print(f"\nDone. {len(all_results)} documents processed, {n_escalate} flagged for human review.")
    print(f"Index: {OUTPUT_CSV}")
    print(f"Field crops: {FIELD_CROPS_DIR}")


if __name__ == "__main__":
    main()
