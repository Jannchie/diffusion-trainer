"""Prepare the gpt-image-2 synthetic set into the canonical prepared-dataset layout.

Reuses the SAME components as the production pipeline so the on-disk format is
byte-for-byte the format the trainer/streamer already consume:

  <dataset_dir>/tags/ab/cd/<sha256>.txt    -- WD-tagger tags, trigger-prefixed
  <dataset_dir>/latents/ab/cd/<sha256>.npz -- ft-mse VAE latents (SD15 768 buckets)
  <dataset_dir>/latents/latents_meta.*     -- train_resolution / original_size / crop_ltrb
  <dataset_dir>/metadata.parquet           -- training manifest (CreateParquetProcessor)

Two synthetic-data marker tags are prepended to every image's tag list so the
style is addressable / droppable downstream exactly like the booru meta tags:
  gpt_2_image, generated

Tags use the same sha256 (file content) as the latents, so the two pair up in
CreateParquetProcessor.

Example:
    uv run python scripts/prepare_gpt_image_2.py \
        --img-dir gpt-image-2 \
        --dataset-dir datasets/sd15-gpt-image-2-768 \
        --vae-path "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
"""

import argparse
import logging
from pathlib import Path

import torch
from PIL import Image
from rich.logging import RichHandler
from rich.progress import track

from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import SimpleLatentsProcessor
from diffusion_trainer.dataset.processors.tagging_processor import SimpleTagger
from diffusion_trainer.dataset.utils import (
    calculate_file_sha256,
    load_latents_meta,
    retrieve_image_paths,
    sharded_path,
    write_latents_meta,
)

logger = logging.getLogger("diffusion_trainer")

# Synthetic-data markers prepended to every tag list (user-chosen). Kept as
# ordinary tags so they can be droppable/excludable like the booru meta tags.
TRIGGER_TAGS = ["gpt_2_image", "generated"]


def merge_tags(trigger: list[str], wd_tags: list[str]) -> list[str]:
    """Prefix trigger tags, then WD tags, de-duplicated while preserving order."""
    seen: set[str] = set()
    out: list[str] = []
    for tag in [*trigger, *wd_tags]:
        if tag and tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare gpt-image-2 synthetic set into the prepared-dataset layout")
    parser.add_argument("--img-dir", type=str, required=True, help="Directory of source images")
    parser.add_argument("--dataset-dir", type=str, required=True, help="Output prepared-dataset directory")
    parser.add_argument("--vae-path", type=str, required=True, help="VAE (local file/dir or URL) used to encode latents")
    parser.add_argument("--base-resolution", type=int, default=768, help="Bucket base resolution (SD15 lineage = 768)")
    parser.add_argument("--general-threshold", type=float, default=0.35, help="WD general tag threshold")
    parser.add_argument("--character-threshold", type=float, default=0.9, help="WD character tag threshold")
    parser.add_argument("--vae-dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"], help="VAE compute dtype")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])

    dataset_dir = Path(args.dataset_dir).absolute()
    tags_dir = dataset_dir / "tags"
    latents_dir = dataset_dir / "latents"
    tags_dir.mkdir(parents=True, exist_ok=True)
    latents_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.vae_dtype]

    image_paths = sorted(retrieve_image_paths(Path(args.img_dir), recursive=True))
    logger.info("Found %d images in %s", len(image_paths), args.img_dir)

    tagger = SimpleTagger(general_threshold=args.general_threshold, character_threshold=args.character_threshold)
    latents_processor = SimpleLatentsProcessor(args.vae_path, dtype=dtype, base_resolution=args.base_resolution)

    latents_meta = load_latents_meta(latents_dir)

    try:
        for image_path in track(image_paths, description="Tagging + encoding..."):
            sha256 = calculate_file_sha256(image_path)
            image = Image.open(image_path).convert("RGB")

            # --- tags (trigger-prefixed WD tags), keyed by file sha256 ---
            tag_path = sharded_path(tags_dir, sha256, "txt")
            if not (tag_path.exists() and tag_path.read_text(encoding="utf-8").strip()):
                tags = merge_tags(TRIGGER_TAGS, tagger.tag_image(image))
                tag_path.parent.mkdir(parents=True, exist_ok=True)
                tag_path.write_text(", ".join(tags), encoding="utf-8")

            # --- latents (SD15 768 buckets), keyed by the same sha256 ---
            npz_path = sharded_path(latents_dir, sha256, "npz")
            if not (npz_path.exists() and sha256 in latents_meta):
                latents_meta[sha256] = latents_processor.process_by_pil(image, save_npz_path=npz_path)
    finally:
        if latents_meta:
            write_latents_meta(latents_dir, latents_meta)

    CreateParquetProcessor(dataset_dir)()
    logger.info("Prepared dataset written to %s", dataset_dir)


if __name__ == "__main__":
    main()
