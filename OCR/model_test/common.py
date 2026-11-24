"""Shared helpers for the snippet-vs-full-page test."""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))          # so form_templates / field_dictionary can be imported

from field_dictionary import FIELDS           # noqa: E402  (field names, in a fixed order)

FIELD_NAMES = list(FIELDS)
ARMS = ("full", "snippets", "hybrid")


def load_manifest(run: Path) -> dict:
    return json.loads((run / "manifest.json").read_text(encoding="utf-8"))


def parse_reply(text: str) -> dict:
    """
    {field: {"value", "label_seen", "confidence"}} from a model reply, or {} if it is not
    the JSON we asked for. Tolerates code fences and chatter around the JSON object.
    """
    if not text:
        return {}
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    fields = data.get("fields", data) if isinstance(data, dict) else {}
    out = {}
    for name, entry in fields.items():
        if isinstance(entry, dict):
            out[name] = {"value": entry.get("value"), "label_seen": entry.get("label_seen"),
                         "confidence": (entry.get("confidence") or "").lower() or None}
        else:  # a bare value: accept it, but there is no label evidence
            out[name] = {"value": entry, "label_seen": None, "confidence": None}
    return out


def blank(value) -> bool:
    return value is None or str(value).strip().lower() in ("", "null", "none", "n/a", "na")
