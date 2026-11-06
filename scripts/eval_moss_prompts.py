"""Render a fixed prompt set across MOSS-Image models x LoRAs for side-by-side review.

Every (model, LoRA) pair renders every prompt with the same seed, size, steps
and CFG, so a column of the output sheets is a controlled comparison. The
prompt set is a TOML file (see ``configs/eval/anime_prompts_v1.toml``).

Memory: all prompts are encoded once per model, then the 9B text encoder
leaves the GPU before any denoising, so a run fits next to a training job.
LoRAs are hooked in with LyCORIS ``apply_to``/``restore`` (no weight merging),
so every LoRA sees the untouched base weights.

    uv run python scripts/eval_moss_prompts.py \
        --prompts configs/eval/anime_prompts_v1.toml \
        --model base=/path/to/MOSS-Image-Base --model sft=/path/to/MOSS-Image-SFT \
        --lora tags=out/moss_image/cmp-tags/cmp-tags/cmp-tags.safetensors \
        --output-dir out/eval/cmp

Outputs: ``singles/<model>/<lora>/<prompt>.png`` (existing files are skipped,
so reruns resume), ``sheets/<category>.jpg`` (rows = model/LoRA, columns =
prompts) and ``index.html`` (each prompt with its text and all renders).
"""

import argparse
import logging
from html import escape
from pathlib import Path
from typing import Any

import torch
from lycoris import create_lycoris_from_weights
from PIL import Image, ImageDraw, ImageFont
from rich.logging import RichHandler

from diffusion_trainer.config.prompt_set import EvalPrompt, load_prompt_set
from diffusion_trainer.finetune.moss_image import encode_moss_prompts
from diffusion_trainer.finetune.utils import load_moss_image_pipeline

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("eval_moss_prompts")

NO_LORA = "none"
SHEET_CELL_HEIGHT = 384
SHEET_LABEL_WIDTH = 220
SHEET_HEADER_HEIGHT = 28


def parse_labeled(values: list[str], flag: str) -> dict[str, str]:
    labeled = {}
    for value in values:
        label, sep, path = value.partition("=")
        if not sep or not label or not path:
            msg = f"{flag} expects label=path, got {value!r}"
            raise ValueError(msg)
        labeled[label] = path
    return labeled


def single_path(output_dir: Path, model: str, lora: str, prompt: EvalPrompt) -> Path:
    return output_dir / "singles" / model / lora / f"{prompt.name}.png"


def encode_all(pipe: Any, prompts: list[EvalPrompt], device: torch.device) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:  # noqa: ANN401
    """Text -> (embeds, mask) for every distinct prompt and negative prompt, unpadded."""
    texts = sorted({p.prompt for p in prompts} | {p.negative_prompt for p in prompts})
    pipe.text_encoder.to(device)
    encoded = {}
    with torch.autocast(device.type, dtype=torch.bfloat16):
        for text in texts:
            embeds, mask = encode_moss_prompts(pipe.tokenizer, pipe.text_encoder, [text], max_sequence_length=4096, device=device)
            encoded[text] = (embeds, mask)
    # Drop it rather than park it on the CPU: the pipeline takes its execution
    # device from the first module it finds, and the encoder comes first.
    pipe.register_modules(text_encoder=None)
    torch.cuda.empty_cache()
    logger.info("Encoded %d distinct texts", len(texts))
    return encoded


@torch.no_grad()
def render_model(
    model_label: str,
    model_path: str,
    loras: dict[str, str],
    prompts: list[EvalPrompt],
    output_dir: Path,
) -> None:
    device = torch.device("cuda")
    pending = {lora: [p for p in prompts if not single_path(output_dir, model_label, lora, p).exists()] for lora in [NO_LORA, *loras]}
    if not any(pending.values()):
        logger.info("[%s] everything already rendered", model_label)
        return

    pipe = load_moss_image_pipeline(model_path, torch.bfloat16)
    encoded = encode_all(pipe, prompts, device)
    pipe.transformer.to(device)
    pipe.vae.to(device)

    for lora_label, todo in pending.items():
        if not todo:
            continue
        network = None
        if lora_label != NO_LORA:
            network, _ = create_lycoris_from_weights(1.0, loras[lora_label], pipe.transformer)
            network.apply_to()
            network.to(device, dtype=torch.bfloat16)
        try:
            for index, prompt in enumerate(todo, start=1):
                embeds, mask = encoded[prompt.prompt]
                neg_embeds, neg_mask = encoded[prompt.negative_prompt]
                generator = torch.Generator(device=device).manual_seed(prompt.seed)
                with torch.autocast(device.type, dtype=torch.bfloat16):
                    image = pipe(
                        prompt_embeds=embeds,
                        prompt_attention_mask=mask,
                        negative_prompt_embeds=neg_embeds,
                        negative_prompt_attention_mask=neg_mask,
                        width=prompt.width,
                        height=prompt.height,
                        num_inference_steps=prompt.steps,
                        guidance_scale=prompt.guidance_scale,
                        generator=generator,
                    ).images[0]
                path = single_path(output_dir, model_label, lora_label, prompt)
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path)
                logger.info("[%s/%s] %d/%d %s", model_label, lora_label, index, len(todo), prompt.name)
        finally:
            if network is not None:
                network.restore()
                del network
                torch.cuda.empty_cache()

    del pipe, encoded
    torch.cuda.empty_cache()


def load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    for name in ("DejaVuSans.ttf", "arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_sheets(output_dir: Path, rows: list[tuple[str, str]], prompts: list[EvalPrompt]) -> None:
    """One JPEG per category: rows = (model, LoRA), columns = that category's prompts."""
    font = load_font(18)
    sheets_dir = output_dir / "sheets"
    sheets_dir.mkdir(parents=True, exist_ok=True)
    for category in dict.fromkeys(p.category for p in prompts):
        columns = [p for p in prompts if p.category == category]
        widths = [round(p.width * SHEET_CELL_HEIGHT / p.height) for p in columns]
        sheet = Image.new("RGB", (SHEET_LABEL_WIDTH + sum(widths), SHEET_HEADER_HEIGHT + SHEET_CELL_HEIGHT * len(rows)), "white")
        draw = ImageDraw.Draw(sheet)
        x = SHEET_LABEL_WIDTH
        for prompt, width in zip(columns, widths, strict=True):
            draw.text((x + 4, 4), prompt.name, fill="black", font=font)
            x += width
        for row_index, (model, lora) in enumerate(rows):
            y = SHEET_HEADER_HEIGHT + row_index * SHEET_CELL_HEIGHT
            draw.text((8, y + SHEET_CELL_HEIGHT // 2 - 10), f"{model} / {lora}", fill="black", font=font)
            x = SHEET_LABEL_WIDTH
            for prompt, width in zip(columns, widths, strict=True):
                path = single_path(output_dir, model, lora, prompt)
                if path.exists():
                    with Image.open(path) as image:
                        sheet.paste(image.convert("RGB").resize((width, SHEET_CELL_HEIGHT), Image.Resampling.LANCZOS), (x, y))
                x += width
        sheet.save(sheets_dir / f"{category}.jpg", quality=90)
        logger.info("Wrote %s", sheets_dir / f"{category}.jpg")


def build_index(output_dir: Path, rows: list[tuple[str, str]], prompts: list[EvalPrompt]) -> None:
    parts = [
        "<!doctype html><meta charset=utf-8><title>MOSS eval</title>",
        (
            "<style>body{font-family:sans-serif;margin:16px}figure{display:inline-block;margin:4px;text-align:center}"
            "img{height:320px}figcaption{font-size:13px}p{max-width:1100px}</style>"
        ),
    ]
    for category in dict.fromkeys(p.category for p in prompts):
        parts.append(f"<h2>{escape(category)}</h2>")
        for prompt in (p for p in prompts if p.category == category):
            parts.append(f"<h3>{escape(prompt.name)}</h3><p>{escape(prompt.prompt)}</p>")
            for model, lora in rows:
                rel = single_path(output_dir, model, lora, prompt).relative_to(output_dir).as_posix()
                parts.append(f'<figure><a href="{rel}"><img src="{rel}" loading=lazy></a><figcaption>{escape(model)} / {escape(lora)}</figcaption></figure>')
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a prompt set across MOSS-Image models x LoRAs")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--model", action="append", required=True, help="label=path to a MOSS-Image bundle (repeatable)")
    parser.add_argument("--lora", action="append", default=[], help="label=path to a LyCORIS .safetensors (repeatable)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--only", nargs="+", default=None, help="Render only these prompt names or categories")
    parser.add_argument("--steps", type=int, default=None, help="Override every prompt's step count")
    args = parser.parse_args()

    prompts = load_prompt_set(args.prompts, only=args.only, steps=args.steps)
    models = parse_labeled(args.model, "--model")
    loras = parse_labeled(args.lora, "--lora")
    for model_label, model_path in models.items():
        render_model(model_label, model_path, loras, prompts, args.output_dir)

    rows = [(model, lora) for model in models for lora in [NO_LORA, *loras]]
    build_sheets(args.output_dir, rows, prompts)
    build_index(args.output_dir, rows, prompts)


if __name__ == "__main__":
    main()
