"""Shared helpers for segmenting scanned ASME form PDFs by layout."""
from pathlib import Path

import cv2
import numpy as np
import fitz

FP_W, FP_H = 300, 380  # fixed working size for all layout fingerprints
GRID_W, GRID_H = 30, 38  # downsample grid for fingerprint vectors


def iter_pdf_pages(pdf_dir: Path):
    """Yields (file_name, page_index) for every page of every *.pdf in
    pdf_dir, sorted by file name then page order.
    """
    for pdf_path in sorted(Path(pdf_dir).glob("*.pdf")):
        doc = fitz.open(pdf_path)
        n = len(doc)
        doc.close()
        for p in range(n):
            yield pdf_path.name, p


def render_gray(doc: fitz.Document, page_index: int, zoom: float = 1.5) -> np.ndarray:
    page = doc[page_index]
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return arr


def binarize(gray: np.ndarray) -> np.ndarray:
    """Returns uint8 image, 255 = ink, 0 = background, resized to FP_W x FP_H."""
    resized = cv2.resize(gray, (FP_W, FP_H), interpolation=cv2.INTER_AREA)
    _, bw = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def line_masks(bw: np.ndarray):
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, FP_W // 15), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(9, FP_H // 15)))
    hmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, hk)
    vmask = cv2.morphologyEx(bw, cv2.MORPH_OPEN, vk)
    return hmask, vmask


def _downsample(mask: np.ndarray, gw: int, gh: int) -> np.ndarray:
    h, w = mask.shape
    resized = cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA)
    return resized / 255.0


def fingerprint(bw: np.ndarray) -> np.ndarray:
    """Structural layout fingerprint: concatenation of downsampled
    horizontal-line, vertical-line and raw-ink density grids. Captures
    the document TEMPLATE (box/rule positions), not the typed content.
    """
    hmask, vmask = line_masks(bw)
    h_feat = _downsample(hmask, GRID_W, GRID_H).flatten()
    v_feat = _downsample(vmask, GRID_W, GRID_H).flatten()
    ink_feat = _downsample(bw, GRID_W // 2, GRID_H // 2).flatten()
    return np.concatenate([h_feat, v_feat, ink_feat]).astype(np.float32)


def rotate_bw(bw: np.ndarray, k: int) -> np.ndarray:
    """k = number of 90 degree counter-clockwise rotations (0..3)."""
    return np.rot90(bw, k).copy()


def n_long_hlines(bw: np.ndarray, min_len_frac: float = 0.5) -> int:
    hmask, _ = line_masks(bw)
    w = hmask.shape[1]
    n, _, stats, _ = cv2.connectedComponentsWithStats((hmask > 0).astype(np.uint8), connectivity=8)
    return sum(1 for i in range(1, n) if stats[i][2] >= min_len_frac * w)


def looks_sideways(bw: np.ndarray) -> bool:
    """Cheap sanity check for a 90/270-rotated page: compares total length
    of long horizontal vs vertical ruled lines. Real pages in this corpus
    are dominated by horizontal rules; a sideways page would flip that.
    """
    hmask, vmask = line_masks(bw)
    h_score = (hmask > 0).sum()
    v_score = (vmask > 0).sum()
    return v_score > 1.3 * h_score
