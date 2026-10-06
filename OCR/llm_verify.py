"""
Second-opinion verification for low-confidence OCR field reads.

Crops the region around a field's OCR boxes and sends the actual pixels to a
vision-capable LLM, since a model that can see the strokes has a shot at
reads PaddleOCR's recognizer got wrong (this matters especially on the
bleed-through/ghosted regions where the misread is a pixel problem, not a
language-model problem a text-only re-ranker could fix).

No API key is wired in yet (ANTHROPIC_API_KEY unset) - verify_field() detects
that and returns immediately with used_llm=False so the rest of the pipeline
(extraction, confidence scoring, escalation) runs standalone. Set the env var
and this starts making real calls with no other code changes needed.
"""

import base64
import os
from dataclasses import dataclass
from pathlib import Path

from PIL import Image


@dataclass
class LLMVerifyResult:
    used_llm: bool
    agrees: bool | None       # None when used_llm is False
    llm_text: str | None      # what the LLM read, if it disagreed
    raw_response: str | None


def crop_field_region(image_path: Path, boxes: list[list[int]], padding: int = 12) -> Image.Image:
    """boxes: list of [x1, y1, x2, y2] for every OCR box contributing to the field."""
    img = Image.open(image_path)
    x1 = max(min(b[0] for b in boxes) - padding, 0)
    y1 = max(min(b[1] for b in boxes) - padding, 0)
    x2 = min(max(b[2] for b in boxes) + padding, img.width)
    y2 = min(max(b[3] for b in boxes) + padding, img.height)
    return img.crop((x1, y1, x2, y2))


def verify_field(
    image_path: Path,
    boxes: list[list[int]],
    field_name: str,
    ocr_text: str,
    crop_save_path: Path | None = None,
) -> LLMVerifyResult:
    """
    Ask a vision LLM to independently read the cropped region and compare
    against PaddleOCR's candidate text. Returns used_llm=False (no-op) if no
    API key is configured, so callers can always treat this as "best effort".
    """
    crop = crop_field_region(image_path, boxes)
    if crop_save_path is not None:
        crop_save_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_save_path)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return LLMVerifyResult(used_llm=False, agrees=None, llm_text=None, raw_response=None)

    try:
        import anthropic
    except ImportError as e:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is set but the 'anthropic' package isn't installed. "
            "Run: pip install anthropic"
        ) from e

    import io
    buf = io.BytesIO()
    crop.save(buf, format="PNG")
    image_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    client = anthropic.Anthropic(api_key=api_key)
    prompt = (
        f"This is a cropped region from a scanned, degraded manufacturing report. "
        f"An OCR system read the '{field_name}' field in this crop as: \"{ocr_text}\".\n"
        f"Look at the image and tell me: does that reading look correct? "
        f"Reply with just 'AGREE' if it's correct, or the corrected text if it's wrong. "
        f"If the crop is too degraded to read confidently, reply 'UNREADABLE'."
    )
    response = client.messages.create(
        model="claude-sonnet-5",  # swap for claude-haiku-4-5-20251001 if cost matters more than accuracy
        max_tokens=100,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": image_b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )
    raw = response.content[0].text.strip()

    if raw.upper().startswith("AGREE"):
        return LLMVerifyResult(used_llm=True, agrees=True, llm_text=None, raw_response=raw)
    if raw.upper().startswith("UNREADABLE"):
        return LLMVerifyResult(used_llm=True, agrees=False, llm_text=None, raw_response=raw)
    return LLMVerifyResult(used_llm=True, agrees=False, llm_text=raw, raw_response=raw)
