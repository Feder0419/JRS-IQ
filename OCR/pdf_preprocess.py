"""
Converts PDFs into per-page PNGs for OCR, with optional cleanup steps.

Pages are rendered at 300 DPI. Denoising and inversion are off by default and
opt-in, since they only help on some scans (inversion in particular can hurt
on normal dark-on-light pages).

Usage: python pdf_preprocess.py input.pdf [--denoise] [--invert] [--out DIR]
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pymupdf

DPI = 300
DENOISE_STRENGTH = 6  # fastNlMeansDenoising 'h'; keep low so thin strokes survive


def preprocess_page(img: np.ndarray, denoise: bool, invert: bool) -> np.ndarray:
    """img: grayscale uint8. Denoise runs before invert, while the image is still smooth."""
    if denoise:
        img = cv2.fastNlMeansDenoising(img, None, h=DENOISE_STRENGTH)
    if invert:
        img = cv2.bitwise_not(img)
    return img


def pdf_to_pngs(
    pdf_path: Path,
    out_dir: Path,
    denoise: bool = False,
    invert: bool = False,
    dpi: int = DPI,
) -> list[Path]:
    """Render every page to <out_dir>/<pdf stem>_p001.png, ... and return the paths."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    with pymupdf.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            img = preprocess_page(img, denoise, invert)
            out_path = out_dir / f"{pdf_path.stem}_p{i:03d}.png"
            cv2.imwrite(str(out_path), img)
            paths.append(out_path)
    return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("output/pdf_pages"))
    args = parser.parse_args()
    for p in pdf_to_pngs(args.pdf, args.out, args.denoise, args.invert):
        print(p)
