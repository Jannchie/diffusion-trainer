"""Build trainable datasets from an exported + recaptioned image set, one per text variant.

Input is the staging directory that ``export_pictoria_images.py`` and
``caption_images.py`` fill (``manifest.parquet``, ``captions.jsonl``,
``caption_inputs/<key>.jpg``). The latents are encoded ONCE, from the caption
inputs (1536 px on the long side, plenty for 1024 buckets), into
``<output>/latents``; every variant is then a directory holding only its own
``metadata.parquet`` and a ``latents`` symlink to the shared tree, so an A/B of
the conditioning text trains on bit-identical latents:

- ``tags``          -- the category-ordered Pictoria tags, no caption;
- ``caption``       -- the VLM recaptions only (long / medium / short as a
  ``captions`` list, one picked per sample);
- ``caption_tags``  -- both: ``<caption>, <tags>``.

Example (the 2,000 best by SILVA, MOSS-Image latents):
    uv run python scripts/build_captioned_dataset.py \
        --source-dir datasets/pictoria-5star --output-dir datasets/moss-pictoria-top2k \
        --vae-path /path/to/MOSS-Image-Base --top 2000
"""

import argparse
import logging
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rich.logging import RichHandler
from rich.progress import track

from diffusion_trainer.dataset.captioning import read_caption_records
from diffusion_trainer.dataset.processors.latents_generate_processor import SimpleLatentsProcessor
from diffusion_trainer.dataset.utils import load_latents_meta, sharded_path, write_latents_meta
from diffusion_trainer.utils.dtype import str_to_dtype

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("build_captioned_dataset")

VARIANTS = ("tags", "caption", "caption_tags")
META_FLUSH_INTERVAL = 500

SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("train_resolution", pa.list_(pa.int64())),
        ("original_size", pa.list_(pa.int64())),
        ("crop_ltrb", pa.list_(pa.int64())),
        ("tags", pa.list_(pa.string())),
        ("tag_categories", pa.list_(pa.string())),
        ("captions", pa.list_(pa.string())),
    ],
)


def load_captions(path: Path) -> dict[str, list[str]]:
    """Key -> [long, medium, short]."""
    return {key: [record["caption"][length] for length in ("long", "medium", "short")] for key, record in read_caption_records(path).items()}


def encode(rows: list[dict], inputs_dir: Path, latents_dir: Path, processor: SimpleLatentsProcessor) -> dict[str, dict]:
    latents_meta = load_latents_meta(latents_dir)
    pending = [row for row in rows if not (row["key"] in latents_meta and sharded_path(latents_dir, row["key"], "npz").exists())]
    logger.info("%d images to encode (%d cached)", len(pending), len(rows) - len(pending))
    try:
        for done, row in enumerate(track(pending, description="Encoding latents..."), start=1):
            npz_path = sharded_path(latents_dir, row["key"], "npz")
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(inputs_dir / f"{row['key']}.jpg") as image:
                latents_meta[row["key"]] = processor.process_by_pil(image, save_npz_path=npz_path)
            if done % META_FLUSH_INTERVAL == 0:
                write_latents_meta(latents_dir, latents_meta)
    finally:
        write_latents_meta(latents_dir, latents_meta)
    return latents_meta


def variant_row(row: dict, meta: dict, captions: list[str], variant: str) -> dict:
    with_tags = variant in {"tags", "caption_tags"}
    return {
        "key": row["key"],
        **meta,
        "tags": row["tags"] if with_tags else [],
        "tag_categories": row["tag_categories"] if with_tags else [],
        "captions": captions if variant != "tags" else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode an exported, recaptioned image set into per-variant training datasets")
    parser.add_argument("--source-dir", type=Path, required=True, help="manifest.parquet + captions.jsonl + caption_inputs/")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vae-path", type=str, required=True)
    parser.add_argument("--vae-dtype", type=str, default="bf16")
    parser.add_argument("--base-resolution", type=int, default=1024)
    parser.add_argument("--top", type=int, default=0, help="Keep the N best by aesthetic score (manifest order); 0 = all")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    args = parser.parse_args()

    captions = load_captions(args.source_dir / "captions.jsonl")
    rows = [row for row in pq.read_table(args.source_dir / "manifest.parquet").to_pylist() if row["key"] in captions]
    rows.sort(key=lambda row: -(row["aesthetic"] if row["aesthetic"] is not None else float("-inf")))
    if args.top:
        rows = rows[: args.top]
    logger.info("%d captioned images selected", len(rows))

    latents_dir = args.output_dir / "latents"
    processor = SimpleLatentsProcessor(args.vae_path, dtype=str_to_dtype(args.vae_dtype), base_resolution=args.base_resolution)
    latents_meta = encode(rows, args.source_dir / "caption_inputs", latents_dir, processor)

    for variant in args.variants:
        variant_dir = args.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        link = variant_dir / "latents"
        if not link.exists():
            link.symlink_to(os.path.relpath(latents_dir, variant_dir), target_is_directory=True)
        table = pa.Table.from_pylist(
            [variant_row(row, latents_meta[row["key"]], captions[row["key"]], variant) for row in rows if row["key"] in latents_meta],
            schema=SCHEMA,
        )
        pq.write_table(table, variant_dir / "metadata.parquet")
        logger.info("Wrote %s (%d rows)", variant_dir / "metadata.parquet", table.num_rows)


if __name__ == "__main__":
    main()
