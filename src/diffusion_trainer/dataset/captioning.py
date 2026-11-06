"""VLM recaptioning with the MOSS-Image recipe (USERGUIDE "Image Captioning").

The prompt is the official one verbatim: a JSON object with long / medium /
short captions, watermark boxes and border flags, conditioned on an "original
caption reference" that the model is told to correct against the image. The
reference is built from the source's own metadata (artist, series, characters,
tags) — the user guide's tip that added world knowledge improves captions.

The model is any OpenAI-compatible chat endpoint (vLLM serving Qwen3.5
27B / 35B-A3B or Gemma 4, the recommended captioners), called with the request
shape the guide documents: thinking on, temperature 1, top_p 0.95.

Responses are validated against the schema before they are accepted; a reply
that fails validation is retried rather than stored, so every stored record is
safe to train on.
"""

import base64
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

logger = logging.getLogger("diffusion_trainer.dataset.captioning")

CAPTION_PROMPT = """You are an image recaptioner. Return exactly one JSON object with this schema:
{
  "caption": {
    "long": "a detailed image description, including all non-watermark visible text",
    "medium": "a one-paragraph image description",
    "short": "a one-sentence image description"
  },
  "watermark": [
    {"box_2d": [y_min, x_min, y_max, x_max], "label": "watermark text"}
  ],
  "borders": ["none"]
}

Rules:
- caption.long:
    - Describe the composition, key subjects, actions, setting, colors, lighting, and artistic style in detail. Include only information visible in the image.
    - Preserve visible text exactly, excluding watermarks. Do not translate, paraphrase, romanize, or normalize image text.
    - Describe text layout as much as possible: line breaks, ordering, relative position, alignment, size hierarchy, font/style, color, spacing, and how text blocks are arranged.
- caption.medium: compress long into one paragraph.
- caption.short: summarize the main subject, action, or relationship in one sentence.
- Do not describe watermarks in any caption field. Record them only in watermark.
- watermark:
    - Use normalized 0–1000 [y_min, x_min, y_max, x_max] coordinates relative to the full image.
    - Set label to the visible watermark text, or a short description if unreadable.
    - Use [] if there are no watermarks.
- borders:
    - Record evident borders or margins using one or more of "left", "right", "up", and "down".
    - Use ["none"] if there are no evident borders or margins. Do not combine "none" with other directions.
- All three caption fields must be non-empty strings.
- Treat the original caption as a reference. Correct any claims that are not supported by the image.
- Return only the JSON object, without Markdown or explanations.

Here is the original caption reference:
{{reference}}"""  # noqa: E501, RUF001 - the official prompt verbatim, en dash and long lines included

BORDER_DIRECTIONS = frozenset({"left", "right", "up", "down"})
# Synthetic tags the trainer pins in front (quality tiers, era buckets) say
# nothing about the image's content; feeding them to the captioner as "facts"
# only invites it to parrot them.
REFERENCE_SKIPPED_CATEGORIES = frozenset({"quality", "year", "meta"})
_REFERENCE_LABELS = {"artist": "Artist", "copyright": "Series", "character": "Characters", "general": "Tags"}

_THINK_END = "</think>"
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class CaptionValidationError(ValueError):
    """The model's reply does not satisfy the caption schema."""


@dataclass(frozen=True)
class Watermark:
    box_2d: list[int]
    label: str


@dataclass(frozen=True)
class CaptionRecord:
    long: str
    medium: str
    short: str
    watermark: list[Watermark] = field(default_factory=list)
    borders: list[str] = field(default_factory=lambda: ["none"])

    def to_json_dict(self) -> dict:
        return asdict(self)


def read_caption_records(path: Path) -> dict[str, dict]:
    """``captions.jsonl`` -> key -> record; the last record per key wins, so a later re-caption supersedes. Missing file = empty."""
    records: dict[str, dict] = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    records[record["key"]] = record
    return records


def build_reference(tags: Sequence[str], categories: Sequence[str] | None = None, original_caption: str = "") -> str:
    """The "original caption reference": the source caption if any, plus categorized tags.

    Booru tags read as underscores-for-spaces; they are de-underscored and
    grouped by category so the captioner can tell an artist name from a
    character name instead of guessing.
    """
    categories = categories if categories is not None else ["general"] * len(tags)
    grouped: dict[str, list[str]] = {}
    for tag, category in zip(tags, categories, strict=True):
        if category in REFERENCE_SKIPPED_CATEGORIES:
            continue
        label = _REFERENCE_LABELS.get(category, "Tags")
        grouped.setdefault(label, []).append(tag.replace("_", " "))
    lines = [original_caption.strip()] if original_caption.strip() else []
    lines += [f"{label}: {', '.join(values)}" for label, values in grouped.items()]
    return "\n".join(lines) if lines else "(none)"


def build_prompt(reference: str) -> str:
    return CAPTION_PROMPT.replace("{{reference}}", reference)


def _strip_reasoning(text: str) -> str:
    """Drop an inline ``<think>`` block when the server runs without a reasoning parser."""
    return text.rsplit(_THINK_END, 1)[-1] if _THINK_END in text else text


def parse_caption_response(text: str) -> CaptionRecord:
    """Parse and validate one reply; raise ``CaptionValidationError`` on any schema violation."""
    match = _JSON_OBJECT_RE.search(_strip_reasoning(text))
    if match is None:
        msg = "no JSON object in reply"
        raise CaptionValidationError(msg)
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        msg = f"invalid JSON: {exc}"
        raise CaptionValidationError(msg) from exc

    caption = data.get("caption")
    if not isinstance(caption, dict):
        msg = "missing caption object"
        raise CaptionValidationError(msg)
    texts = {}
    for name in ("long", "medium", "short"):
        value = caption.get(name)
        if not isinstance(value, str) or not value.strip():
            msg = f"caption.{name} must be a non-empty string"
            raise CaptionValidationError(msg)
        texts[name] = value.strip()

    watermarks = []
    for item in data.get("watermark") or []:
        box = item.get("box_2d") if isinstance(item, dict) else None
        if not (isinstance(box, list) and len(box) == 4 and all(isinstance(v, int | float) and 0 <= v <= 1000 for v in box)):
            msg = f"bad watermark box: {item!r}"
            raise CaptionValidationError(msg)
        watermarks.append(Watermark(box_2d=[int(v) for v in box], label=str(item.get("label", "")).strip()))

    borders = data.get("borders") or ["none"]
    if not isinstance(borders, list):
        msg = f"bad borders: {borders!r}"
        raise CaptionValidationError(msg)
    if borders != ["none"] and not (set(borders) <= BORDER_DIRECTIONS and len(set(borders)) == len(borders)):
        msg = f"borders must be ['none'] or distinct directions, got {borders!r}"
        raise CaptionValidationError(msg)

    return CaptionRecord(**texts, watermark=watermarks, borders=list(borders))


def image_data_url(path: Path) -> str:
    suffix = path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg"}.get(suffix, suffix)
    return f"data:image/{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


@dataclass(frozen=True)
class CaptionClient:
    """One OpenAI-compatible chat endpoint, called the way the MOSS user guide documents."""

    endpoint: str
    model: str
    max_tokens: int = 16384
    temperature: float = 1.0
    top_p: float = 0.95
    enable_thinking: bool = True
    timeout: float = 900.0

    def request_body(self, prompt: str, image_url: str) -> dict:
        return {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                },
            ],
            "n": 1,
            "stream": False,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }

    def caption(self, http: httpx.Client, image_path: Path, reference: str, *, attempts: int = 3) -> CaptionRecord:
        """Caption one image, re-sampling up to ``attempts`` times on an invalid reply."""
        body = self.request_body(build_prompt(reference), image_data_url(image_path))
        last_error: Exception | None = None
        for _ in range(attempts):
            response = http.post(f"{self.endpoint.rstrip('/')}/chat/completions", json=body, timeout=self.timeout)
            response.raise_for_status()
            choice = response.json()["choices"][0]
            if choice.get("finish_reason") == "length":
                last_error = CaptionValidationError("reply truncated at max_tokens")
                continue
            try:
                return parse_caption_response(choice["message"].get("content") or "")
            except CaptionValidationError as exc:
                last_error = exc
        msg = f"{image_path.name}: no valid caption after {attempts} attempts ({last_error})"
        raise CaptionValidationError(msg)
