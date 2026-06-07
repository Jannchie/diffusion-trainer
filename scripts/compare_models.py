"""Side-by-side comparison grids for two (or more) SD 1.5 checkpoints.

Driven by a TOML config (see ``configs/compare/``) listing the models with
their per-model inference settings (each checkpoint runs at its own
training-matched optimum: scheduler config is taken from the saved pipeline,
``guidance_rescale``/``clip_skip`` from the config) and a shared prompt set.
All models share prompt text, seed and initial latents; prompts go through the
A1111-style attention parser (``diffusion_prompt_embedder``) exactly like
training previews, so escape literal parens as ``\\(...\\)``.

Usage:
    uv run python scripts/compare_models.py --config configs/compare/aom3b2-vs-pictoria-v0.3.toml

Outputs under ``output_dir``: ``singles/<model>/<name>.png`` (raw outputs),
``comparisons/<name>.png`` (labeled side-by-side panels) and ``index.html``
(contact sheet grouped by category). Existing files are skipped, so reruns
resume where they left off.
"""

import argparse
import textwrap
import tomllib
from dataclasses import dataclass, field, replace
from html import escape
from pathlib import Path

import torch
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipeline
from diffusion_prompt_embedder import get_embeddings_sd15
from PIL import Image, ImageDraw, ImageFont
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn

from diffusion_trainer.finetune.utils import load_sd15_pipeline

console = Console()

LABEL_BAR_HEIGHT = 36
CAPTION_LINE_HEIGHT = 18
CAPTION_PADDING = 8
CAPTION_WRAP_CHARS = 130
PANEL_GAP = 4


@dataclass(frozen=True)
class ModelSpec:
    """One checkpoint with its inference-time settings."""

    name: str
    path: str
    guidance_rescale: float = 0.0
    clip_skip: int = 2
    guidance_scale: float | None = None  # falls back to the global value


@dataclass(frozen=True)
class PromptSpec:
    """One prompt shared by every model."""

    prompt: str
    category: str = "uncategorized"
    negative_prompt: str | None = None
    width: int = 768
    height: int = 768


@dataclass(frozen=True)
class CompareConfig:
    """Parsed comparison config."""

    output_dir: Path
    models: list[ModelSpec]
    prompts: list[PromptSpec]
    seeds: list[int] = field(default_factory=lambda: [47])
    steps: int = 28
    guidance_scale: float = 7.5
    negative_prompt: str = ""
    # Hires fix (A1111-style two-pass): upscale the base output and img2img it
    # at `hires_strength`. Off when hires_scale <= 1. Small faces in full-body
    # shots live below the 8x-VAE latent resolution at the base size, so
    # single-pass outputs systematically understate every model's detail
    # ceiling — enable this to evaluate models the way the ecosystem runs them.
    hires_scale: float = 0.0
    hires_strength: float = 0.45
    hires_steps: int = 20


def load_config(path: Path) -> CompareConfig:
    """Parse the TOML comparison config into typed specs."""
    with path.open("rb") as f:
        raw = tomllib.load(f)
    models = [ModelSpec(**entry) for entry in raw["models"]]
    prompts = [PromptSpec(**entry) for entry in raw["prompts"]]
    return CompareConfig(
        output_dir=Path(raw["output_dir"]),
        models=models,
        prompts=prompts,
        seeds=list(raw.get("seeds", [47])),
        steps=int(raw.get("steps", 28)),
        guidance_scale=float(raw.get("guidance_scale", 7.5)),
        negative_prompt=str(raw.get("negative_prompt", "")),
        hires_scale=float(raw.get("hires_scale", 0.0)),
        hires_strength=float(raw.get("hires_strength", 0.45)),
        hires_steps=int(raw.get("hires_steps", 20)),
    )


def slugify(text: str, max_length: int = 48) -> str:
    """Filesystem-safe slug from the leading prompt tags."""
    cleaned = "".join(ch if ch.isalnum() else "-" for ch in text.lower())
    while "--" in cleaned:
        cleaned = cleaned.replace("--", "-")
    return cleaned.strip("-")[:max_length].rstrip("-")


def sample_name(index: int, prompt: PromptSpec, seed: int) -> str:
    """Stable output basename for one (prompt, seed) cell."""
    return f"{index:03d}-{prompt.category}-{slugify(prompt.prompt)}-seed{seed}"


@torch.no_grad()
def generate_one(
    pipeline: object,
    model: ModelSpec,
    prompt: PromptSpec,
    config: CompareConfig,
    seed: int,
) -> Image.Image:
    """Run one prompt through one pipeline with training-preview-identical conditioning."""
    device = pipeline.device  # type: ignore[attr-defined]
    negative = prompt.negative_prompt if prompt.negative_prompt is not None else config.negative_prompt
    guidance_scale = model.guidance_scale if model.guidance_scale is not None else config.guidance_scale
    generator = torch.Generator(device=device).manual_seed(seed)
    with torch.autocast(device.type):
        prompt_embeds, neg_prompt_embeds = get_embeddings_sd15(
            pipeline.tokenizer,  # type: ignore[attr-defined]
            pipeline.text_encoder,  # type: ignore[attr-defined]
            prompt=prompt.prompt,
            neg_prompt=negative,
            clip_skip=model.clip_skip,
            pad_last_block=True,
        )
        result = pipeline(  # type: ignore[operator]
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_prompt_embeds,
            num_inference_steps=config.steps,
            generator=generator,
            width=prompt.width,
            height=prompt.height,
            guidance_scale=guidance_scale,
            guidance_rescale=model.guidance_rescale,
            output_type="latent",
        )
    image = _decode_fp32(pipeline, result.images)
    if config.hires_scale > 1.0:
        ctx = HiresContext(prompt_embeds=prompt_embeds, neg_prompt_embeds=neg_prompt_embeds, generator=generator)
        image = _hires_pass(pipeline, image, ctx, config, model)
    return image


def _decode_fp32(pipeline: object, latents: torch.Tensor) -> Image.Image:
    """Decode latents in fp32 outside autocast.

    fp16 VAE decode intermittently overflows to NaN (pure-black frames) on both
    anime VAEs at 640+ resolutions; the UNet latents themselves are clean. The
    VAE is cast to fp32 at load time.
    """
    vae = pipeline.vae  # type: ignore[attr-defined]
    image = vae.decode(latents.to(torch.float32) / vae.config.scaling_factor).sample
    return pipeline.image_processor.postprocess(image, output_type="pil")[0]  # type: ignore[attr-defined]


@dataclass(frozen=True)
class HiresContext:
    """Embeddings and generator threaded from the base pass into the hires pass."""

    prompt_embeds: torch.Tensor
    neg_prompt_embeds: torch.Tensor
    generator: torch.Generator


def _hires_pass(
    pipeline: object,
    image: Image.Image,
    ctx: HiresContext,
    config: CompareConfig,
    model: ModelSpec,
) -> Image.Image:
    """A1111-style hires fix: lanczos-upscale the base output, then img2img it."""
    width = int(image.width * config.hires_scale) // 8 * 8
    height = int(image.height * config.hires_scale) // 8 * 8
    upscaled = image.resize((width, height), Image.LANCZOS)
    img2img = StableDiffusionImg2ImgPipeline.from_pipe(pipeline)
    img2img.set_progress_bar_config(disable=True)
    device = pipeline.device  # type: ignore[attr-defined]
    with torch.autocast(device.type):
        result = img2img(
            prompt_embeds=ctx.prompt_embeds,
            negative_prompt_embeds=ctx.neg_prompt_embeds,
            image=upscaled,
            strength=config.hires_strength,
            num_inference_steps=config.hires_steps,
            generator=ctx.generator,
            guidance_scale=model.guidance_scale if model.guidance_scale is not None else config.guidance_scale,
            guidance_rescale=model.guidance_rescale,
            output_type="latent",
        )
    return _decode_fp32(pipeline, result.images)


def _default_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def compose_comparison(
    images: list[tuple[str, Image.Image]],
    prompt: PromptSpec,
    seed: int,
    config: CompareConfig,
) -> Image.Image:
    """Stitch per-model outputs into one labeled panel with the prompt as caption."""
    width, height = images[0][1].size  # actual size (hires fix may upscale past prompt.width/height)
    hires_note = f" | hires x{config.hires_scale}@{config.hires_strength}" if config.hires_scale > 1.0 else ""
    caption_lines = [
        *textwrap.wrap(f"prompt: {prompt.prompt}", width=CAPTION_WRAP_CHARS),
        f"seed {seed} | steps {config.steps} | cfg {config.guidance_scale} | {width}x{height}{hires_note} | category: {prompt.category}",
    ]
    caption_height = CAPTION_PADDING * 2 + CAPTION_LINE_HEIGHT * len(caption_lines)
    total_width = width * len(images) + PANEL_GAP * (len(images) - 1)
    total_height = LABEL_BAR_HEIGHT + height + caption_height

    canvas = Image.new("RGB", (total_width, total_height), (24, 24, 24))
    draw = ImageDraw.Draw(canvas)
    label_font = _default_font(20)
    caption_font = _default_font(13)

    for column, (model_name, image) in enumerate(images):
        x_offset = column * (width + PANEL_GAP)
        bbox = draw.textbbox((0, 0), model_name, font=label_font)
        text_x = x_offset + (width - (bbox[2] - bbox[0])) // 2
        draw.text((text_x, (LABEL_BAR_HEIGHT - (bbox[3] - bbox[1])) // 2), model_name, fill=(235, 235, 235), font=label_font)
        canvas.paste(image, (x_offset, LABEL_BAR_HEIGHT))

    text_y = LABEL_BAR_HEIGHT + height + CAPTION_PADDING
    for line in caption_lines:
        draw.text((CAPTION_PADDING, text_y), line, fill=(200, 200, 200), font=caption_font)
        text_y += CAPTION_LINE_HEIGHT
    return canvas


def write_index_html(config: CompareConfig, cells: list[tuple[PromptSpec, int, Path]]) -> Path:
    """Write a contact sheet grouped by category; comparison images embedded by relative path."""
    model_names = " vs ".join(model.name for model in config.models)
    parts = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        f"<title>{escape(model_names)}</title>",
        "<style>body{background:#181818;color:#ddd;font-family:sans-serif;margin:24px}",
        "h2{border-bottom:1px solid #444;padding-bottom:4px;margin-top:40px}",
        "img{max-width:100%;height:auto;display:block;margin:12px 0 32px}</style>",
        f"</head><body><h1>{escape(model_names)}</h1>",
        f"<p>{len(config.prompts)} prompts x {len(config.seeds)} seeds | steps {config.steps} | cfg {config.guidance_scale}</p>",
    ]
    current_category = None
    for prompt, _seed, image_path in cells:
        if prompt.category != current_category:
            current_category = prompt.category
            parts.append(f"<h2>{escape(current_category)}</h2>")
        parts.append(f"<img src='{escape(image_path.relative_to(config.output_dir).as_posix())}' loading='lazy'>")
    parts.append("</body></html>")
    index_path = config.output_dir / "index.html"
    index_path.write_text("\n".join(parts), encoding="utf-8")
    return index_path


def run(config: CompareConfig) -> None:
    """Generate every (model, prompt, seed) image, then the composites and index."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    comparisons_dir = config.output_dir / "comparisons"
    comparisons_dir.mkdir(parents=True, exist_ok=True)

    pipelines = {}
    for model in config.models:
        pipeline = load_sd15_pipeline(model.path, torch.float16)
        pipeline.to(device)
        pipeline.vae.to(torch.float32)  # see generate_one: fp16 decode NaNs intermittently
        pipeline.set_progress_bar_config(disable=True)
        pipelines[model.name] = pipeline
        console.log(f"loaded [bold]{model.name}[/] from {model.path} (scheduler: {type(pipeline.scheduler).__name__}, "
                    f"prediction: {pipeline.scheduler.config.get('prediction_type')}, guidance_rescale: {model.guidance_rescale})")

    cells: list[tuple[PromptSpec, int, Path]] = []
    total = len(config.prompts) * len(config.seeds)
    progress = Progress(
        SpinnerColumn(), "[progress.description]{task.description}", BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), TimeRemainingColumn(),
        console=console,
    )
    with progress:
        task = progress.add_task("comparing", total=total)
        for index, prompt in enumerate(config.prompts):
            for seed in config.seeds:
                name = sample_name(index, prompt, seed)
                comparison_path = comparisons_dir / f"{name}.png"
                cells.append((prompt, seed, comparison_path))
                if comparison_path.exists():
                    progress.advance(task)
                    continue
                outputs: list[tuple[str, Image.Image]] = []
                for model in config.models:
                    single_path = config.output_dir / "singles" / model.name / f"{name}.png"
                    if single_path.exists():
                        image = Image.open(single_path).convert("RGB")
                    else:
                        image = generate_one(pipelines[model.name], model, prompt, config, seed)
                        single_path.parent.mkdir(parents=True, exist_ok=True)
                        image.save(single_path)
                    outputs.append((model.name, image))
                compose_comparison(outputs, prompt, seed, config).save(comparison_path)
                progress.advance(task)

    index_path = write_index_html(config, cells)
    console.log(f"done: {total} comparisons -> {comparisons_dir}")
    console.log(f"contact sheet: {index_path}")


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True, help="Comparison TOML (see configs/compare/)")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override output_dir from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.output_dir is not None:
        # dataclasses.replace keeps every other field (incl. hires settings).
        config = replace(config, output_dir=args.output_dir)
    run(config)


if __name__ == "__main__":
    main()
