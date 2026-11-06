"""Export a Pictoria selection as an image dataset staging directory (no VAE, no GPU).

The image-level counterpart of ``load_dataset_from_pictoria_db.py``: the same
selection (star score, rating, short edge, canonical posts only) and the same
category-ordered tags, but instead of encoding latents it writes

- ``manifest.parquet`` -- one row per image, sorted by the aesthetic score
  (best first): ``key`` (content SHA256), ``source_path`` (local, never
  published), ``ext``, ``width``/``height``, ``rating``, ``score``,
  ``aesthetic`` (``--aesthetic-scorer``, SILVA by default), ``published_at``,
  ``source_url``, ``tags``/``tag_categories`` and ``original_caption``;
- ``caption_inputs/<key>.jpg`` -- downscaled RGB copies for VLM captioning
  (``scripts/caption_images.py``), small enough to ship to a GPU box.

Re-running with a grown selection adds the new rows and only renders the new
caption inputs; ``scripts/publish_image_dataset.py`` then publishes from here.

Example (all 5-star posts, live DB opened read-only):
    uv run python scripts/export_pictoria_images.py \
        --db /mnt/e/pictoria/server/illustration/images/.pictoria/pictoria.sqlite \
        --images-root /mnt/e/pictoria/server/illustration/images \
        --output-dir datasets/pictoria-5star --score 5
"""

import argparse
import logging
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from load_dataset_from_pictoria_db import QualityTagging, SelectionFilters, query_posts
from PIL import Image
from rich.logging import RichHandler
from rich.progress import track

from diffusion_trainer.dataset.era_tags import DEFAULT_ERA_SPEC, parse_era_spec
from diffusion_trainer.dataset.quality_tags import DEFAULT_QUALITY_TAGS

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("export_pictoria_images")

MANIFEST_FILENAME = "manifest.parquet"
CAPTION_INPUTS_DIR = "caption_inputs"

MANIFEST_SCHEMA = pa.schema(
    [
        ("key", pa.string()),
        ("post_id", pa.int64()),
        ("source_path", pa.string()),
        ("ext", pa.string()),
        ("width", pa.int32()),
        ("height", pa.int32()),
        ("rating", pa.int8()),
        ("score", pa.int8()),
        ("aesthetic", pa.float32()),
        ("published_at", pa.string()),
        ("source_url", pa.string()),
        ("tags", pa.list_(pa.string())),
        ("tag_categories", pa.list_(pa.string())),
        ("original_caption", pa.string()),
    ],
)


def fetch_post_details(db_path: Path, post_ids: list[int], scorer: str) -> dict[int, dict]:
    """Per-post fields ``query_posts`` does not carry: size, rating, score, dates, source, aesthetic."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        db.execute("CREATE TEMP TABLE wanted (id INTEGER PRIMARY KEY)")
        db.executemany("INSERT INTO wanted VALUES (?)", [(post_id,) for post_id in post_ids])
        rows = db.execute(
            "SELECT p.id, p.extension, p.width, p.height, p.rating, p.score, p.published_at, p.source, a.score "
            "FROM posts p JOIN wanted w ON w.id = p.id "
            "LEFT JOIN post_aesthetic_scores a ON a.post_id = p.id AND a.scorer = ?",
            (scorer,),
        ).fetchall()
    finally:
        db.close()
    fields = ("ext", "width", "height", "rating", "score", "published_at", "source_url", "aesthetic")
    return {row[0]: dict(zip(fields, row[1:], strict=True)) for row in rows}


def render_caption_input(source: Path, target: Path, max_side: int) -> None:
    """Downscale to ``max_side`` on the long edge, flatten alpha onto white, save as JPEG."""
    with Image.open(source) as image:
        image.seek(0)  # first frame of animated GIF/WebP
        rgba = image.convert("RGBA")
    rgba.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", rgba.size, (255, 255, 255))
    canvas.paste(rgba, mask=rgba.getchannel("A"))
    tmp = target.with_suffix(".tmp")
    canvas.save(tmp, "JPEG", quality=92)
    tmp.replace(target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a Pictoria selection as an image-dataset staging directory")
    parser.add_argument("--db", type=Path, required=True, help="pictoria.sqlite (opened read-only)")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score", type=int, default=None, help="Exact manual star score (overrides --min-score)")
    parser.add_argument("--min-score", type=int, default=0)
    parser.add_argument("--max-rating", type=int, default=None)
    parser.add_argument("--min-short-edge", type=int, default=0)
    parser.add_argument("--aesthetic-scorer", type=str, default="silva", help="post_aesthetic_scores.scorer to sort by and record")
    parser.add_argument("--top", type=int, default=0, help="Keep only the N best by aesthetic score (0 = all)")
    parser.add_argument("--caption-input-size", type=int, default=1536, help="Long-edge size of the captioning copies")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--prune-missing", action="store_true", help="Drop manifest rows that left the selection")
    args = parser.parse_args()

    filters = SelectionFilters(
        min_short_edge=args.min_short_edge,
        min_score=args.min_score,
        score=args.score,
        max_rating=args.max_rating,
        include_hidden_duplicates=False,
    )
    quality = QualityTagging(tags=list(DEFAULT_QUALITY_TAGS), scorer=args.aesthetic_scorer, thresholds=None)
    posts = query_posts(args.db, args.images_root, filters=filters, quality=quality, era_bounds=parse_era_spec(DEFAULT_ERA_SPEC))
    details = fetch_post_details(args.db, [post.post_id for post in posts], args.aesthetic_scorer)

    rows = [
        {
            "key": post.sha256,
            "post_id": post.post_id,
            "source_path": str(post.image_path),
            **details[post.post_id],
            "tags": post.tags,
            "tag_categories": post.tag_categories,
            "original_caption": post.caption,
        }
        for post in posts
    ]
    rows.sort(key=lambda row: -(row["aesthetic"] if row["aesthetic"] is not None else float("-inf")))
    if args.top:
        rows = rows[: args.top]
    logger.info("Selected %d posts (%s)", len(rows), filters)

    manifest_path = args.output_dir / MANIFEST_FILENAME
    if manifest_path.exists() and not args.prune_missing:
        current = {row["key"] for row in rows}
        kept = [row for row in pq.read_table(manifest_path).to_pylist() if row["key"] not in current]
        if kept:
            logger.info("Keeping %d earlier rows outside the current selection (--prune-missing drops them)", len(kept))
        rows += kept
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=MANIFEST_SCHEMA), manifest_path)
    logger.info("Wrote %s (%d rows)", manifest_path, len(rows))

    inputs_dir = args.output_dir / CAPTION_INPUTS_DIR
    inputs_dir.mkdir(exist_ok=True)
    todo = [row for row in rows if not (inputs_dir / f"{row['key']}.jpg").exists()]
    logger.info("%d caption inputs to render (%d cached)", len(todo), len(rows) - len(todo))

    def render(row: dict) -> str | None:
        try:
            render_caption_input(Path(row["source_path"]), inputs_dir / f"{row['key']}.jpg", args.caption_input_size)
        except Exception as exc:  # one unreadable file must not stop the export
            return f"{row['source_path']}: {exc}"
        return None

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        errors = [error for error in track(pool.map(render, todo), total=len(todo), description="Rendering caption inputs...") if error]
    for error in errors:
        logger.warning("Failed to render %s", error)


if __name__ == "__main__":
    main()
