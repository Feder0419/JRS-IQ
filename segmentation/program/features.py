"""Per-page feature extraction: layout fingerprints and OCR, both orientations.

Each page is rendered once and turned into:
  full0/full180  whole-page layout fingerprint (lib.fingerprint) upright /
                 turned 180 degrees
  hdr0/hdr180    higher-resolution fingerprint of just the top HDR_FRAC of the
                 page, where the form title sits - what tells same-layout
                 forms (H-3 vs P-3) apart
  OCR per orientation: a whole-page confidence score (which way up the text
                 reads - reliable for fronts, ~96% for confident continuation
                 pages), the title strip text, and the whole-page text
"""
import re

import cv2
import fitz
import numpy as np
import pytesseract
from PIL import Image

import lib

TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
RENDER_ZOOM = 3.0
PAGE_OCR_SCALE = 2 / 3     # whole-page OCR runs at zoom 2
TITLE_FRAC = 0.22
HDR_FRAC = 0.25
HDR_W, HDR_H = 600, 160
HDR_GW, HDR_GH = 60, 16

_GOOD_WORD = re.compile(r"[A-Za-z][A-Za-z'.,:()-]{2,}")


def header_fingerprint(gray):
    top = gray[: int(gray.shape[0] * HDR_FRAC)]
    resized = cv2.resize(top, (HDR_W, HDR_H), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (HDR_W // 15, 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, HDR_H // 6))
    hmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    feats = [cv2.resize(m.astype(np.float32), (HDR_GW, HDR_GH), interpolation=cv2.INTER_AREA).flatten() / 255.0
             for m in (hmask, vmask, bw)]
    return np.concatenate(feats).astype(np.float32)


def page_ocr(gray):
    """-> (confidence score, text) for the whole page as given"""
    small = cv2.resize(gray, None, fx=PAGE_OCR_SCALE, fy=PAGE_OCR_SCALE, interpolation=cv2.INTER_AREA)
    d = pytesseract.image_to_data(Image.fromarray(small), config="--psm 3",
                                  output_type=pytesseract.Output.DICT)
    score, words = 0.0, []
    for w, c in zip(d["text"], d["conf"]):
        w, c = w.strip(), float(c)
        if not w or c < 0:
            continue
        words.append(w)
        if c >= 60 and _GOOD_WORD.fullmatch(w):
            score += c
    return score, " ".join(words)


def title_ocr(gray):
    top = gray[: int(gray.shape[0] * TITLE_FRAC)]
    text = pytesseract.image_to_string(Image.fromarray(top), config="--psm 6")
    return " ".join(text.split())


def render_gray(doc, pidx, zoom=RENDER_ZOOM):
    pix = doc[pidx].get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()


def page_features(doc, pidx):
    gray0 = render_gray(doc, pidx)
    out = dict(page=pidx, ocr={})
    bw0 = lib.binarize(gray0)
    out["ink"] = float((bw0 > 0).mean())
    out["full0"] = lib.fingerprint(bw0)
    out["full180"] = lib.fingerprint(lib.rotate_bw(bw0, 2))
    for rot in (0, 180):
        gray = gray0 if rot == 0 else np.ascontiguousarray(np.rot90(gray0, 2))
        out[f"hdr{rot}"] = header_fingerprint(gray)
        try:
            score, text = page_ocr(gray)
            title = title_ocr(gray)
        except Exception as e:  # a single unreadable page must not kill a long batch
            score, text, title = 0.0, f"<ocr error: {e}>", ""
        out["ocr"][rot] = dict(score=score, text=text, title=title)
    return out


def worker(job):
    """job = (pdf path, [0-based page indices]) -> (file name, [features])"""
    pytesseract.pytesseract.tesseract_cmd = TESSERACT
    pdf_path, pages = job
    with fitz.open(pdf_path) as doc:
        return pdf_path, [page_features(doc, p) for p in pages]
