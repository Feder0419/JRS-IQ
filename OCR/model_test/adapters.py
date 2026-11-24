"""
Model adapters. The test only needs one thing from a model: given some images and a
prompt, return the reply text and (if the API reports it) token usage.

    reply, usage = adapter.read([Path("a.png"), Path("b.png")], prompt)
    usage = {"input_tokens": int | None, "output_tokens": int | None}

Use --adapter dryrun to check the plumbing without calling anything. To use a different
provider, copy `Custom` below, implement read(), and register it in ADAPTERS.

NOTE: the anthropic and gemini adapters were written from the providers' documented
usage and have NOT been run by the author (no access). Try them on --limit 2 first.
"""
import base64
import os
from pathlib import Path


class Adapter:
    name = "base"

    def __init__(self, model: str | None = None):
        self.model = model

    def read(self, images: list[Path], prompt: str) -> tuple[str, dict]:
        raise NotImplementedError


class DryRun(Adapter):
    """Calls nothing; answers 'could not read' for everything. For testing the plumbing."""
    name = "dryrun"

    def read(self, images, prompt):
        return '{"fields": {}}', {"input_tokens": None, "output_tokens": None}


class Anthropic(Adapter):
    """pip install anthropic; set ANTHROPIC_API_KEY; pass --model <model id>."""
    name = "anthropic"

    def __init__(self, model=None):
        super().__init__(model)
        import anthropic
        self.client = anthropic.Anthropic()
        if not model:
            raise SystemExit("--model is required for the anthropic adapter")

    def read(self, images, prompt):
        content = []
        for p in images:
            data = base64.standard_b64encode(Path(p).read_bytes()).decode("ascii")
            content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}})
        content.append({"type": "text", "text": prompt})
        resp = self.client.messages.create(model=self.model, max_tokens=2500, temperature=0,
                                           messages=[{"role": "user", "content": content}])
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        return text, {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}


class Gemini(Adapter):
    """pip install google-genai pillow; set GEMINI_API_KEY (or GOOGLE_API_KEY); pass --model <model id>."""
    name = "gemini"

    def __init__(self, model=None):
        super().__init__(model)
        from google import genai
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
        if not model:
            raise SystemExit("--model is required for the gemini adapter")

    def read(self, images, prompt):
        from PIL import Image
        contents = [Image.open(p) for p in images] + [prompt]
        resp = self.client.models.generate_content(model=self.model, contents=contents)
        usage = getattr(resp, "usage_metadata", None)
        return resp.text or "", {"input_tokens": getattr(usage, "prompt_token_count", None),
                                 "output_tokens": getattr(usage, "candidates_token_count", None)}


class Custom(Adapter):
    """Copy this for any other provider (or your friend's existing wrapper)."""
    name = "custom"

    def read(self, images, prompt):
        raise NotImplementedError("implement Custom.read() and register it in ADAPTERS")


ADAPTERS = {"dryrun": DryRun, "anthropic": Anthropic, "gemini": Gemini, "custom": Custom}


def get_adapter(name: str, model: str | None) -> Adapter:
    if name not in ADAPTERS:
        raise SystemExit(f"unknown adapter '{name}'; choose from {', '.join(ADAPTERS)}")
    return ADAPTERS[name](model)
