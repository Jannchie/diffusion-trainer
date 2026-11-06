"""Publish an exported image dataset (images + tags + VLM captions) to the HF Hub, incrementally.

Joins ``manifest.parquet`` (from ``export_pictoria_images.py``) with
``captions.jsonl`` (from ``caption_images.py``) and writes parquet shards with
the ORIGINAL image bytes embedded (an ``Image`` feature, so the Hub viewer and
``datasets.load_dataset`` decode them), then uploads them to a dataset repo.

Incremental by content: the keys already on the Hub are read from the remote
shards' ``key`` column (range requests, no image bytes downloaded), and only
new keys are packed, as ``data/<batch>-<idx>.parquet`` with a fresh batch id.
Earlier shards are never rewritten, so a later run with a grown manifest just
appends. Rows without a caption are held back unless ``--allow-uncaptioned``.

Example:
    uv run python scripts/publish_image_dataset.py \
        --dataset-dir datasets/pictoria-5star --repo-id Jannchie/pictoria-5star
"""

import argparse
import json
import logging
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, HfFileSystem
from rich.logging import RichHandler

from diffusion_trainer.dataset.captioning import read_caption_records

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("publish_image_dataset")

DATA_DIR = "data"
DEFAULT_SHARD_BYTES = 512 << 20
# Small row groups keep the Hub viewer and streaming readers from pulling
# hundreds of MB of image bytes to show a handful of rows.
ROW_GROUP_SIZE = 32

WATERMARK_TYPE = pa.struct([("box_2d", pa.list_(pa.int16())), ("label", pa.string())])
SCHEMA = pa.schema(
    [
        ("image", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
        ("key", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("rating", pa.int8()),
        ("score", pa.int8()),
        ("aesthetic", pa.float32()),
        ("published_at", pa.string()),
        ("source_url", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("tag_categories", pa.list_(pa.string())),
        ("caption_long", pa.string()),
        ("caption_medium", pa.string()),
        ("caption_short", pa.string()),
        ("watermark", pa.list_(WATERMARK_TYPE)),
        ("borders", pa.list_(pa.string())),
        ("caption_model", pa.string()),
    ],
)


def _value(dtype: str) -> dict:
    return {"dtype": dtype, "_type": "Value"}


def _list(feature: dict) -> dict:
    return {"feature": feature, "_type": "List"}


# The `datasets` feature spec, stored in the parquet schema metadata: without
# it the image column is just a struct of bytes to the Hub.
HF_FEATURES = {
    "image": {"_type": "Image"},
    "key": _value("string"),
    "width": _value("int32"),
    "height": _value("int32"),
    "rating": _value("int8"),
    "score": _value("int8"),
    "aesthetic": _value("float32"),
    "published_at": _value("string"),
    "source_url": _value("string"),
    "tags": _list(_value("string")),
    "tag_categories": _list(_value("string")),
    "caption_long": _value("string"),
    "caption_medium": _value("string"),
    "caption_short": _value("string"),
    "watermark": _list({"box_2d": _list(_value("int16")), "label": _value("string")}),
    "borders": _list(_value("string")),
    "caption_model": _value("string"),
}
SCHEMA = SCHEMA.with_metadata({"huggingface": json.dumps({"info": {"features": HF_FEATURES}})})


def remote_keys(repo_id: str) -> set[str]:
    fs = HfFileSystem()
    keys: set[str] = set()
    try:
        files = fs.glob(f"datasets/{repo_id}/{DATA_DIR}/*.parquet")
    except FileNotFoundError:
        return keys
    for file in files:
        with fs.open(file, "rb") as f:
            keys.update(pq.read_table(f, columns=["key"]).column("key").to_pylist())
    return keys


def to_row(row: dict, caption: dict | None) -> dict:
    record = (caption or {}).get("caption") or {}
    return {
        "image": {"bytes": Path(row["source_path"]).read_bytes(), "path": f"{row['key']}.{row['ext']}"},
        **{name: row[name] for name in ("key", "width", "height", "rating", "score", "aesthetic", "published_at", "source_url", "tags", "tag_categories")},
        "caption_long": record.get("long"),
        "caption_medium": record.get("medium"),
        "caption_short": record.get("short"),
        "watermark": record.get("watermark"),
        "borders": record.get("borders"),
        "caption_model": (caption or {}).get("model"),
    }


def write_shards(rows: list[dict], captions: dict[str, dict], out_dir: Path, batch: str, shard_bytes: int) -> list[Path]:
    """Pack rows into ~``shard_bytes`` parquet files, streaming images in one shard at a time."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buffer: list[dict] = []
    size = 0

    def flush() -> None:
        nonlocal buffer, size
        if not buffer:
            return
        path = out_dir / f"{batch}-{len(paths):05d}.parquet"
        pq.write_table(pa.Table.from_pylist(buffer, schema=SCHEMA), path, row_group_size=ROW_GROUP_SIZE)
        logger.info("Wrote %s (%d rows, %.0f MB)", path.name, len(buffer), size / 1e6)
        paths.append(path)
        buffer, size = [], 0

    for row in rows:
        packed = to_row(row, captions.get(row["key"]))
        buffer.append(packed)
        size += len(packed["image"]["bytes"])
        if size >= shard_bytes:
            flush()
    flush()
    return paths


def dataset_card(repo_id: str, total_rows: int) -> str:
    return f"""---
pretty_name: {repo_id.rpartition("/")[2]}
tags:
- not-for-all-audiences
task_categories:
- text-to-image
configs:
- config_name: default
  data_files:
  - split: train
    path: {DATA_DIR}/*.parquet
---

# {repo_id.rpartition("/")[2]}

Curated illustrations ({total_rows} images): manually rated 5 stars in Pictoria, canonical posts only,
ordered by SILVA aesthetic score within each upload batch. Original image bytes, untouched.

Images are third-party artworks kept for private research use; see `source_url` for each origin.
Content ratings 1-4 (general / sensitive / questionable / explicit) are all included — filter on `rating` (0 = unrated).

| column | content |
| --- | --- |
| `image` | original file bytes |
| `key` | SHA256 of the file |
| `width`, `height` | pixel size |
| `rating` | 0 unrated, 1 general, 2 sensitive, 3 questionable, 4 explicit |
| `score` | manual star score (5) |
| `aesthetic` | SILVA aesthetic score (0-1) |
| `published_at`, `source_url` | origin metadata |
| `tags`, `tag_categories` | category-ordered tags (quality, year, artist, copyright, character, general, meta) |
| `caption_long` / `caption_medium` / `caption_short` | VLM recaptions, MOSS-Image recipe |
| `watermark` | watermark boxes, `[y_min, x_min, y_max, x_max]` in 0-1000 |
| `borders` | evident borders (`none` or directions) |
| `caption_model` | the captioning VLM |

Captions follow the MOSS-Image user guide's recaptioning prompt, with the post's tags as the reference.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Publish an exported image dataset to the HF Hub, incrementally")
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Directory with manifest.parquet and captions.jsonl")
    parser.add_argument("--repo-id", type=str, required=True)
    parser.add_argument("--public", action="store_true", help="Create the repo public (default: private)")
    parser.add_argument("--shard-mb", type=int, default=DEFAULT_SHARD_BYTES >> 20)
    parser.add_argument("--allow-uncaptioned", action="store_true", help="Also publish rows that have no caption yet")
    parser.add_argument("--dry-run", action="store_true", help="Write the shards locally but do not upload")
    args = parser.parse_args()

    manifest = pq.read_table(args.dataset_dir / "manifest.parquet").to_pylist()
    captions = read_caption_records(args.dataset_dir / "captions.jsonl")

    api = HfApi()
    if not args.dry_run:
        api.create_repo(args.repo_id, repo_type="dataset", private=not args.public, exist_ok=True)
    published = set() if args.dry_run else remote_keys(args.repo_id)

    new_rows = [row for row in manifest if row["key"] not in published]
    held_back = [row for row in new_rows if row["key"] not in captions]
    if not args.allow_uncaptioned:
        new_rows = [row for row in new_rows if row["key"] in captions]
    logger.info(
        "%d in manifest, %d already on the Hub, %d to publish, %d held back without captions",
        len(manifest),
        len(published),
        len(new_rows),
        0 if args.allow_uncaptioned else len(held_back),
    )
    if not new_rows:
        return

    batch = time.strftime("%Y%m%d%H%M%S")
    staging = args.dataset_dir / "publish"
    write_shards(new_rows, captions, staging / DATA_DIR, batch, args.shard_mb << 20)
    (staging / "README.md").write_text(dataset_card(args.repo_id, len(published) + len(new_rows)), encoding="utf-8")
    if args.dry_run:
        logger.info("Dry run: shards are in %s", staging)
        return

    # Resumable and parallel; files already on the Hub (same content) are skipped.
    api.upload_large_folder(repo_id=args.repo_id, repo_type="dataset", folder_path=staging, private=not args.public)
    # The shards are a full copy of the images; once on the Hub they are dead weight.
    for path in (staging / DATA_DIR).glob("*.parquet"):
        path.unlink()
    logger.info("Published %d rows to https://huggingface.co/datasets/%s", len(new_rows), args.repo_id)


if __name__ == "__main__":
    main()
