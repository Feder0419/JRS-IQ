"""
Field crops from a dictionary of form types (form_templates/templates.json).

For a form we know, every field sits in roughly the same place on the printed
page, so reading keywords is the roundabout way to find it (and fails whenever OCR
garbles a label). Instead each form type stores an exemplar picture plus a
rectangle per field. A new page is lined up with each exemplar by matching the
printed structure (titles, rules, captions: handwriting and typed values simply
fail to match and are ignored), which gives the shift, scale and tilt of this
scan, and the rectangles are moved to match.

The number of matching features doubles as a form detector: a page of some other
form type matches nothing well, so it is rejected (and the caller falls back to
keyword-based crops or the whole page) rather than getting confident wrong boxes.

Pages must be upright (the segmentation step rotates them).
"""
import json
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

TEMPLATE_DIR = Path(__file__).with_name("form_templates")
WORK_WIDTH = 1500        # pages and exemplars are brought to this width (the frame the field rectangles are in)
FEATURE_WIDTH = 1000     # ...and features are found on a copy this wide: coarser is steadier on blurry scans
MIN_INLIERS = 50         # fewer matching features than this: not that form
RATIO = 0.85             # Lowe ratio test for feature matches
PAD = 20                 # px (at WORK_WIDTH) of room added around every moved rectangle
SCALE_RANGE = (0.6, 1.7)  # a believable alignment: no wild stretching or shrinking
MAX_SHEAR = 0.25


def _to_work(image) -> tuple[np.ndarray, float]:
    """Grey image at WORK_WIDTH, and the factor from that frame back to the original."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    if isinstance(image, Image.Image):
        image = np.array(image.convert("L"))
    elif image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    factor = image.shape[1] / WORK_WIDTH
    work = cv2.resize(image, (WORK_WIDTH, round(image.shape[0] / factor)), interpolation=cv2.INTER_AREA)
    return work, factor


def _features(work: np.ndarray):
    """
    SIFT features of the page, as points in the WORK_WIDTH frame. The page is first
    reduced to black ink on white: on grey, noisy, uneven scans the raw pixels differ
    too much from the exemplar's for features to match (a third of real H-3 scans
    found nothing), but the printed rules and lettering survive thresholding.
    """
    scale = FEATURE_WIDTH / work.shape[1]
    small = cv2.resize(work, (FEATURE_WIDTH, round(work.shape[0] * scale)), interpolation=cv2.INTER_AREA)
    small = cv2.GaussianBlur(small, (0, 0), 1.0)
    small = cv2.adaptiveThreshold(small, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12)
    small = cv2.GaussianBlur(small, (0, 0), 1.5)
    kps, desc = cv2.SIFT_create(nfeatures=6000, contrastThreshold=0.03).detectAndCompute(small, None)
    points = np.float32([k.pt for k in kps]) / scale if kps else np.zeros((0, 2), np.float32)
    return points, desc


@lru_cache(maxsize=1)
def load_templates() -> dict:
    """{name: {description, form, fields, size, keypoints, descriptors}} from templates.json."""
    data = json.loads((TEMPLATE_DIR / "templates.json").read_text(encoding="utf-8"))["templates"]
    out = {}
    for name, t in data.items():
        work, _ = _to_work(TEMPLATE_DIR / t["exemplar"])
        points, desc = _features(work)
        out[name] = {**t, "size": work.shape[::-1], "points": points, "descriptors": desc}
    return out


def _align(page_points, page_desc, tpl: dict):
    """(affine matrix exemplar -> page, inlier count) or (None, 0)."""
    if page_desc is None or tpl["descriptors"] is None or len(page_desc) < 8:
        return None, 0
    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(tpl["descriptors"], page_desc, k=2)
    good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < RATIO * n.distance]
    if len(good) < 8:
        return None, 0
    src = tpl["points"][[m.queryIdx for m in good]]
    dst = page_points[[m.trainIdx for m in good]]
    matrix, inliers = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=6.0,
                                           maxIters=5000, confidence=0.995)
    if matrix is None:
        return None, 0
    # Reject alignments that stretch, squash or shear the page unbelievably.
    sx, sy = np.hypot(*matrix[:, 0]), np.hypot(*matrix[:, 1])
    shear = abs(float(matrix[:, 0] @ matrix[:, 1])) / (sx * sy)
    if not (SCALE_RANGE[0] <= sx <= SCALE_RANGE[1] and SCALE_RANGE[0] <= sy <= SCALE_RANGE[1]) or shear > MAX_SHEAR:
        return None, 0
    return matrix, int(inliers.sum())


def move_box(box, matrix, factor: float, page_w: int, page_h: int) -> list[int]:
    """Carry an exemplar rectangle onto the page: through the alignment, padded, in page pixels."""
    x1, y1, x2, y2 = box
    corners = np.float32([[x1, y1], [x2, y1], [x2, y2], [x1, y2]])
    moved = corners @ matrix[:, :2].T + matrix[:, 2]
    lo, hi = moved.min(axis=0) - PAD, moved.max(axis=0) + PAD
    return [int(max(lo[0] * factor, 0)), int(max(lo[1] * factor, 0)),
            int(min(hi[0] * factor, page_w)), int(min(hi[1] * factor, page_h))]


def match_page(image) -> dict:
    """
    Line a page up with every form type. Returns {"template": best name or None,
    "inliers": its feature count, "scores": {name: feature count}, "fields": {field: {"box", "kind"}}}.
    `template` is None (and `fields` empty) when no form type matches well enough.
    """
    work, factor = _to_work(image)
    page_h, page_w = round(work.shape[0] * factor), round(work.shape[1] * factor)
    points, desc = _features(work)
    scores, matrices = {}, {}
    for name, tpl in load_templates().items():
        matrices[name], scores[name] = _align(points, desc, tpl)
    best = max(scores, key=scores.get) if scores else None
    result = {"template": None, "inliers": scores.get(best, 0), "scores": scores, "fields": {}}
    if best is None or scores[best] < MIN_INLIERS:
        return result
    result["template"] = best
    for field, box in load_templates()[best]["fields"].items():
        result["fields"][field] = {"box": move_box(box, matrices[best], factor, page_w, page_h), "kind": f"template:{best}"}
    return result
