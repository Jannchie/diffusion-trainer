"""Build a prepared dataset directly from a Pictoria SQLite snapshot (no server needed).

Selects posts by manual star score and a minimum short-edge resolution, takes
tags/captions from the database instead of running WD-tagger, encodes VAE
latents and produces the standard prepared layout (latents tree, sidecar tags,
``metadata.parquet``) consumable by training or ``scripts/export_dataset.py``.

Example:
    uv run python scripts/load_dataset_from_pictoria_db.py \
        --db /tmp/pictoria.sqlite \
        --images-root /mnt/e/pictoria/server/illustration/images \
        --output-dir datasets/pictoria-5star-768 \
        --vae-path https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors \
        --vae-dtype bf16 --base-resolution 768 --score 5 --min-short-edge 768
"""

import argparse
import logging
import re
import sqlite3
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rich.logging import RichHandler
from rich.progress import track

from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import SimpleLatentsProcessor
from diffusion_trainer.dataset.utils import calculate_file_sha256, load_latents_meta, sharded_path, write_latents_meta
from diffusion_trainer.utils.dtype import str_to_dtype

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("load_dataset_from_pictoria_db")

META_FLUSH_INTERVAL = 500
PREFETCH_WORKERS = 4
PREFETCH_DEPTH = 8

# The trainer keys everything by content SHA256. Legacy Pictoria rows store a
# 32-char hash in the sha256 column; those keys must be recomputed from the file.
SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class PictoriaPost:
    """One selected post row from the Pictoria database."""

    post_id: int
    image_path: Path
    sha256: str
    caption: str
    tags: list[str]


def query_posts(db_path: Path, images_root: Path, *, score: int, min_short_edge: int) -> list[PictoriaPost]:
    """Read matching posts (with their tags) from a Pictoria SQLite snapshot."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        where = "score = ? AND MIN(width, height) >= ?"
        rows = db.execute(
            f"SELECT id, file_path, file_name, extension, sha256, caption FROM posts WHERE {where}",  # noqa: S608
            (score, min_short_edge),
        ).fetchall()
        tags_by_post: dict[int, list[str]] = {}
        tag_rows = db.execute(
            f"SELECT post_id, tag_name FROM post_has_tag WHERE post_id IN (SELECT id FROM posts WHERE {where}) ORDER BY post_id, tag_name",  # noqa: S608
            (score, min_short_edge),
        )
        for post_id, tag_name in tag_rows:
            tags_by_post.setdefault(post_id, []).append(tag_name)
    finally:
        db.close()

    posts, seen = [], set()
    rehashed = 0
    for post_id, file_path, file_name, extension, sha256, caption in rows:
        image_path = images_root / file_path / f"{file_name}.{extension}"
        key = sha256.lower()
        if not SHA256_HEX_RE.match(key):
            try:
                key = calculate_file_sha256(image_path)
                rehashed += 1
            except OSError:
                logger.warning("Skipping post %d: legacy hash and unreadable file %s", post_id, image_path)
                continue
        if key in seen:
            continue  # duplicate content under another path; first occurrence wins
        seen.add(key)
        posts.append(
            PictoriaPost(
                post_id=post_id,
                image_path=image_path,
                sha256=key,
                caption=caption or "",
                tags=tags_by_post.get(post_id, []),
            ),
        )
    if rehashed:
        logger.info("Recomputed SHA256 for %d posts with legacy short hashes", rehashed)
    if len(posts) < len(rows):
        logger.info("Skipped %d duplicate/unreadable posts", len(rows) - len(posts))
    return posts


def _load_image(post: PictoriaPost) -> Image.Image:
    """Read and fully decode an image in a prefetch thread (slow mount + CPU decode)."""
    image = Image.open(post.image_path)
    image.load()
    return image


def encode_posts(posts: list[PictoriaPost], output_dir: Path, processor: SimpleLatentsProcessor) -> None:
    """Write sidecar tags and encode latents for every post (idempotent re-runs).

    Image reads/decodes run in a small prefetch pool so the GPU encode never
    waits on the (slow) image mount; metadata is flushed periodically and on
    interruption so a crashed run resumes where it left off.
    """
    latents_dir = output_dir / "latents"
    tags_dir = output_dir / "tags"
    latents_meta = load_latents_meta(latents_dir)

    pending: list[PictoriaPost] = []
    for post in posts:
        tag_path = sharded_path(tags_dir, post.sha256, "txt")
        tag_path.parent.mkdir(parents=True, exist_ok=True)
        tag_path.write_text(", ".join(post.tags), encoding="utf-8")
        npz_path = sharded_path(latents_dir, post.sha256, "npz")
        if not (npz_path.exists() and post.sha256 in latents_meta):
            pending.append(post)
    logger.info("%d posts to encode (%d already cached)", len(pending), len(posts) - len(pending))

    failed = 0
    try:
        with ThreadPoolExecutor(max_workers=PREFETCH_WORKERS) as pool:
            iterator = iter(pending)
            window: deque[tuple[PictoriaPost, Future[Image.Image]]] = deque()

            def top_up() -> None:
                for post in islice(iterator, PREFETCH_DEPTH - len(window)):
                    window.append((post, pool.submit(_load_image, post)))

            top_up()
            for encoded in track(range(1, len(pending) + 1), description="Encoding latents..."):
                post, future = window.popleft()
                top_up()  # keep the prefetch window full while the GPU works
                try:
                    latents_meta[post.sha256] = processor.process_by_pil(future.result(), save_npz_path=sharded_path(latents_dir, post.sha256, "npz"))
                except Exception:
                    failed += 1
                    logger.exception("Failed to encode %s (post %d)", post.image_path, post.post_id)
                if encoded % META_FLUSH_INTERVAL == 0:
                    write_latents_meta(latents_dir, latents_meta)
    finally:
        # Flush even on interruption so already-encoded items keep their metadata.
        if latents_meta:
            write_latents_meta(latents_dir, latents_meta)
    if failed:
        logger.warning("%d images failed to encode and were skipped", failed)


def merge_captions(metadata_path: Path, posts: list[PictoriaPost]) -> None:
    """Add/replace the caption column in metadata.parquet from the database."""
    captions = {post.sha256: post.caption for post in posts if post.caption}
    if not captions:
        return
    table = pq.read_table(metadata_path)
    column = pa.array([captions.get(key, "") for key in table.column("key").to_pylist()])
    if "caption" in table.column_names:
        table = table.set_column(table.column_names.index("caption"), "caption", column)
    else:
        table = table.append_column("caption", column)
    pq.write_table(table, metadata_path)
    logger.info("Merged %d captions into %s", len(captions), metadata_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prepared dataset from a Pictoria SQLite snapshot")
    parser.add_argument("--db", type=Path, required=True, help="Path to a pictoria.sqlite snapshot (read-only)")
    parser.add_argument("--images-root", type=Path, required=True, help="Pictoria images root (posts.file_path is relative to it)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Prepared dataset output directory")
    parser.add_argument("--vae-path", type=str, required=True, help="VAE model path or URL")
    parser.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32", "bf16"], default="bf16", help="VAE dtype")
    parser.add_argument("--base-resolution", type=int, default=1024, help="Bucket base resolution (512/768/1024)")
    parser.add_argument("--score", type=int, default=5, help="Exact manual star score to select")
    parser.add_argument("--min-short-edge", type=int, default=0, help="Minimum short-edge pixel size")
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N posts (0 = all, for smoke tests)")
    args = parser.parse_args()

    posts = query_posts(args.db, args.images_root, score=args.score, min_short_edge=args.min_short_edge)
    logger.info("Selected %d posts (score=%d, short edge >= %d)", len(posts), args.score, args.min_short_edge)
    if args.limit:
        posts = posts[: args.limit]
    if not posts:
        logger.warning("Nothing to do")
        return

    processor = SimpleLatentsProcessor(args.vae_path, dtype=str_to_dtype(args.vae_dtype), base_resolution=args.base_resolution)
    encode_posts(posts, args.output_dir, processor)
    CreateParquetProcessor(args.output_dir)()
    merge_captions(args.output_dir / "metadata.parquet", posts)
    logger.info("Prepared dataset at %s", args.output_dir)


if __name__ == "__main__":
    main()
