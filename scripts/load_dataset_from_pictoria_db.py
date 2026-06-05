"""Build a prepared dataset directly from a Pictoria SQLite snapshot (no server needed).

Selects posts by manual star score, content rating and a minimum short-edge
resolution (near-duplicate groups contribute only their canonical post), takes
tags/captions from the database instead of running WD-tagger, encodes VAE
latents and produces the standard prepared layout (latents tree, sidecar tags,
``metadata.parquet``) consumable by training or ``scripts/export_dataset.py``.

Tags are written category-aware: each post gets one mutually exclusive quality
tag (manual star score first, SILVA aesthetic score as fallback, thresholds
auto-calibrated against the rated population) and one era tag bucketed from
``published_at`` (anime style drifts across years; tagging the era makes it a
controllable axis instead of an averaged-out confound), followed by artist,
copyright, character, general and meta tags. ``metadata.parquet`` carries a
``tag_categories`` column parallel to ``tags`` so training can pin the
quality/era/artist/copyright/character prefix and shuffle only the general
tail.

Example (full-library 768 build):
    uv run python scripts/load_dataset_from_pictoria_db.py \
        --db ~/datasets-cache/pictoria-snapshot.sqlite \
        --images-root /mnt/e/pictoria/server/illustration/images \
        --output-dir datasets/pictoria-full-768 \
        --vae-path https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors \
        --vae-dtype bf16 --base-resolution 768 --min-short-edge 768
"""

import argparse
import logging
import re
import sqlite3
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image
from rich.logging import RichHandler
from rich.progress import track

from diffusion_trainer.dataset.era_tags import DEFAULT_ERA_SPEC, ERA_CATEGORY, EraBound, era_tag_for_year, parse_era_spec
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import SimpleLatentsProcessor
from diffusion_trainer.dataset.quality_tags import DEFAULT_QUALITY_TAGS, NUM_QUALITY_TIERS, QUALITY_CATEGORY, calibrate_thresholds, quality_tier
from diffusion_trainer.dataset.utils import calculate_file_sha256, load_latents_meta, sharded_path, write_latents_meta
from diffusion_trainer.utils.dtype import str_to_dtype

logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[RichHandler()])
logger = logging.getLogger("load_dataset_from_pictoria_db")

META_FLUSH_INTERVAL = 500
PREFETCH_WORKERS = 4
PREFETCH_DEPTH = 8

# Prompt-order ranks: global conditioning axes (quality, era) first, then
# style/series/character conditioning, free-form general tags, and meta last
# (training configs typically exclude meta entirely via tag_category_order).
# Unknown categories sort as general.
CATEGORY_RANKS = {QUALITY_CATEGORY: 0, ERA_CATEGORY: 1, "artist": 2, "copyright": 3, "character": 4, "general": 5, "meta": 6}

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
    tag_categories: list[str]
    """Category per tag, parallel to ``tags`` (already in prompt order)."""


@dataclass(frozen=True)
class SelectionFilters:
    """Post selection criteria translated into a SQL WHERE clause."""

    min_short_edge: int
    min_score: int
    score: int | None
    max_rating: int | None
    include_hidden_duplicates: bool

    def to_sql(self) -> tuple[str, list[int]]:
        conditions = ["MIN(width, height) >= ?"]
        params = [self.min_short_edge]
        if self.score is not None:
            conditions.append("score = ?")
            params.append(self.score)
        elif self.min_score > 0:
            conditions.append("score >= ?")
            params.append(self.min_score)
        if self.max_rating is not None:
            conditions.append("rating <= ?")
            params.append(self.max_rating)
        if not self.include_hidden_duplicates:
            conditions.append("canonical_post_id IS NULL")
        return " AND ".join(conditions), params


# {where} is assembled from fixed literals in SelectionFilters.to_sql; all values are bound parameters.
_TAGS_QUERY = (
    "SELECT pht.post_id, pht.tag_name, COALESCE(g.name, 'general') "
    "FROM post_has_tag pht "
    "LEFT JOIN tags t ON t.name = pht.tag_name "
    "LEFT JOIN tag_groups g ON g.id = t.group_id "
    "WHERE pht.post_id IN (SELECT id FROM posts WHERE {where}) "
    "ORDER BY pht.post_id, pht.tag_name"
)


def _fetch_tags_by_post(db: sqlite3.Connection, where: str, params: list[int]) -> dict[int, list[tuple[str, str]]]:
    """(tag name, category) pairs per selected post, alphabetical within a post."""
    tags_by_post: dict[int, list[tuple[str, str]]] = {}
    rows = db.execute(_TAGS_QUERY.format(where=where), params)
    for post_id, tag_name, category in rows:
        if not tag_name.strip():
            continue  # the library contains a stray empty tag; never emit it
        tags_by_post.setdefault(post_id, []).append((tag_name, category))
    return tags_by_post


def _fetch_aesthetic_scores(db: sqlite3.Connection, scorer: str) -> dict[int, float]:
    return dict(db.execute("SELECT post_id, score FROM post_aesthetic_scores WHERE scorer = ?", (scorer,)))


def resolve_quality_thresholds(db: sqlite3.Connection, scorer: str, override: list[float] | None) -> list[float]:
    """Aesthetic-score cuts for the 5 quality tiers: explicit override or calibration.

    Calibration uses the *whole* rated library (not just the selected posts) so
    the fallback follows the curator's standards even for narrow selections.
    """
    if override is not None:
        if len(override) != NUM_QUALITY_TIERS - 1:
            msg = f"--silva-thresholds needs {NUM_QUALITY_TIERS - 1} ascending values, got {len(override)}"
            raise ValueError(msg)
        return override
    samples = db.execute(
        "SELECT p.score, a.score FROM posts p JOIN post_aesthetic_scores a ON a.post_id = p.id AND a.scorer = ? WHERE p.score BETWEEN 1 AND ?",
        (scorer, NUM_QUALITY_TIERS),
    ).fetchall()
    thresholds = calibrate_thresholds(samples)
    logger.info("Calibrated %s thresholds from %d rated posts: %s", scorer, len(samples), [round(t, 4) for t in thresholds])
    return thresholds


def _ordered_post_tags(
    tag_pairs: list[tuple[str, str]],
    pinned_pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """All tags of one post in prompt order; returns (names, categories).

    ``pinned_pairs`` are synthesized (tag, category) pairs (quality, era) that
    always lead the sequence in the order given.
    """
    pairs = pinned_pairs + sorted(tag_pairs, key=lambda pair: (CATEGORY_RANKS.get(pair[1], CATEGORY_RANKS["general"]), pair[0]))
    return [name for name, _ in pairs], [category for _, category in pairs]


def _published_year(published_at: str | None) -> int | None:
    """Publication year from an ISO timestamp string, tolerating junk."""
    if not published_at:
        return None
    try:
        return int(str(published_at)[:4])
    except ValueError:
        return None


@dataclass(frozen=True)
class QualityTagging:
    """Quality-tag emission settings (empty ``tags`` disables quality tagging)."""

    tags: list[str]
    scorer: str
    thresholds: list[float] | None
    """Explicit aesthetic-score cuts; None auto-calibrates from rated posts."""


@dataclass
class _PinnedTagger:
    """Synthesizes the pinned (quality, era) tag pairs per post and tracks their distributions."""

    quality: QualityTagging
    era_bounds: list[EraBound]
    aesthetic_by_post: dict[int, float]
    thresholds: list[float]
    tier_counts: Counter = field(default_factory=Counter)
    era_counts: Counter = field(default_factory=Counter)

    def pinned_pairs(self, post_id: int, score: int, published_at: str | None) -> list[tuple[str, str]]:
        pairs = []
        if self.quality.tags:
            tier = quality_tier(score, self.aesthetic_by_post.get(post_id), self.thresholds)
            if tier is not None:
                pairs.append((self.quality.tags[tier], QUALITY_CATEGORY))
            self.tier_counts[self.quality.tags[tier] if tier is not None else "(no quality tag)"] += 1
        if self.era_bounds:
            era_tag = era_tag_for_year(_published_year(published_at), self.era_bounds)
            if era_tag is not None:
                pairs.append((era_tag, ERA_CATEGORY))
            self.era_counts[era_tag or "(no era tag)"] += 1
        return pairs

    def log_distributions(self) -> None:
        if self.tier_counts:
            logger.info("Quality tier distribution: %s", dict(sorted(self.tier_counts.items(), key=lambda item: -item[1])))
        if self.era_counts:
            logger.info("Era tag distribution: %s", dict(sorted(self.era_counts.items(), key=lambda item: -item[1])))


def query_posts(
    db_path: Path,
    images_root: Path,
    *,
    filters: SelectionFilters,
    quality: QualityTagging,
    era_bounds: list[EraBound],
) -> list[PictoriaPost]:
    """Read matching posts (with categorized tags plus quality/era tags) from a Pictoria SQLite snapshot."""
    db = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    aesthetic_by_post: dict[int, float] = {}
    thresholds: list[float] = []
    try:
        where, params = filters.to_sql()
        rows = db.execute(
            f"SELECT id, file_path, file_name, extension, sha256, caption, score, published_at FROM posts WHERE {where}",  # noqa: S608
            params,
        ).fetchall()
        tags_by_post = _fetch_tags_by_post(db, where, params)
        if quality.tags:
            aesthetic_by_post = _fetch_aesthetic_scores(db, quality.scorer)
            thresholds = resolve_quality_thresholds(db, quality.scorer, quality.thresholds)
    finally:
        db.close()

    tagger = _PinnedTagger(quality=quality, era_bounds=era_bounds, aesthetic_by_post=aesthetic_by_post, thresholds=thresholds)
    posts, seen = [], set()
    rehashed = 0
    for post_id, file_path, file_name, extension, sha256, caption, score, published_at in rows:
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
        tags, tag_categories = _ordered_post_tags(tags_by_post.get(post_id, []), tagger.pinned_pairs(post_id, score, published_at))
        posts.append(
            PictoriaPost(
                post_id=post_id,
                image_path=image_path,
                sha256=key,
                caption=caption or "",
                tags=tags,
                tag_categories=tag_categories,
            ),
        )
    if rehashed:
        logger.info("Recomputed SHA256 for %d posts with legacy short hashes", rehashed)
    if len(posts) < len(rows):
        logger.info("Skipped %d duplicate/unreadable posts", len(rows) - len(posts))
    tagger.log_distributions()
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


def _set_column(table: pa.Table, name: str, values: list, value_type: pa.DataType) -> pa.Table:
    column = pa.array(values, type=value_type)
    if name in table.column_names:
        return table.set_column(table.column_names.index(name), name, column)
    return table.append_column(name, column)


def merge_post_columns(metadata_path: Path, posts: list[PictoriaPost], *, prune_missing: bool = False) -> None:
    """Overwrite tags/tag_categories/caption in metadata.parquet from the database.

    CreateParquetProcessor rebuilds the manifest from sidecar txt files, which
    keeps tag order but knows nothing about categories; rewriting both columns
    from the same source guarantees they stay aligned. Rows from earlier runs
    in the same output dir keep their existing values (categories stay null,
    which training treats as the legacy flat behavior) — unless
    ``prune_missing`` drops them, e.g. after score changes moved posts out of
    the selection. Pruning only trims the manifest; cached latents/sidecars
    stay on disk and revive for free if a post re-enters the selection.
    """
    by_key = {post.sha256: post for post in posts}
    table = pq.read_table(metadata_path)
    if prune_missing:
        keep = [key in by_key for key in table.column("key").to_pylist()]
        if not all(keep):
            logger.info("Pruned %d manifest rows no longer in the current selection", len(keep) - sum(keep))
            table = table.filter(pa.array(keep))
    keys = table.column("key").to_pylist()
    existing_tags = table.column("tags").to_pylist() if "tags" in table.column_names else [[] for _ in keys]
    existing_captions = table.column("caption").to_pylist() if "caption" in table.column_names else ["" for _ in keys]

    tags_values, category_values, captions = [], [], []
    for key, old_tags, old_caption in zip(keys, existing_tags, existing_captions, strict=True):
        post = by_key.get(key)
        tags_values.append(post.tags if post else old_tags)
        category_values.append(post.tag_categories if post else None)
        captions.append(post.caption if post else (old_caption or ""))

    table = _set_column(table, "tags", tags_values, pa.list_(pa.string()))
    table = _set_column(table, "tag_categories", category_values, pa.list_(pa.string()))
    table = _set_column(table, "caption", captions, pa.string())
    pq.write_table(table, metadata_path)
    logger.info("Merged tags/categories/captions for %d posts into %s", len(by_key), metadata_path)


def _parse_csv_floats(text: str) -> list[float]:
    return [float(part) for part in text.split(",") if part.strip()]


def _parse_quality_tags(text: str) -> list[str]:
    tags = [part.strip() for part in text.split(",") if part.strip()]
    if tags and len(tags) != NUM_QUALITY_TIERS:
        msg = f"--quality-tags needs {NUM_QUALITY_TIERS} comma-separated names (worst..best) or an empty string to disable, got {len(tags)}"
        raise ValueError(msg)
    return tags


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a prepared dataset from a Pictoria SQLite snapshot")
    parser.add_argument("--db", type=Path, required=True, help="Path to a pictoria.sqlite snapshot (read-only)")
    parser.add_argument("--images-root", type=Path, required=True, help="Pictoria images root (posts.file_path is relative to it)")
    parser.add_argument("--output-dir", type=Path, required=True, help="Prepared dataset output directory")
    parser.add_argument("--vae-path", type=str, required=True, help="VAE model path or URL")
    parser.add_argument("--vae-dtype", type=str, choices=["fp16", "fp32", "bf16"], default="bf16", help="VAE dtype")
    parser.add_argument("--base-resolution", type=int, default=1024, help="Bucket base resolution (512/768/1024)")
    parser.add_argument("--score", type=int, default=None, help="Exact manual star score to select (overrides --min-score)")
    parser.add_argument("--min-score", type=int, default=0, help="Minimum manual star score (0 includes unrated posts)")
    parser.add_argument("--max-rating", type=int, default=None, help="Maximum content rating to include (default: all)")
    parser.add_argument("--min-short-edge", type=int, default=0, help="Minimum short-edge pixel size")
    parser.add_argument(
        "--include-hidden-duplicates",
        action="store_true",
        help="Also include near-duplicate posts hidden behind a canonical post (default: canonical only)",
    )
    parser.add_argument(
        "--quality-tags",
        type=str,
        default=",".join(DEFAULT_QUALITY_TAGS),
        help="5 comma-separated quality tag names (worst..best); pass an empty string to disable quality tagging",
    )
    parser.add_argument("--aesthetic-scorer", type=str, default="silva", help="post_aesthetic_scores.scorer used as fallback for unrated posts")
    parser.add_argument(
        "--era-tags",
        type=str,
        default=DEFAULT_ERA_SPEC,
        help="Era buckets from published_at as 'tag:max_year,...' with an open-ended last entry (e.g. 'oldest:2014,...,newest'); empty string disables",
    )
    parser.add_argument(
        "--silva-thresholds",
        type=str,
        default=None,
        help="4 ascending aesthetic-score cuts for the 5 tiers (e.g. '0.25,0.4,0.6,0.81'); default: auto-calibrate from rated posts",
    )
    parser.add_argument(
        "--prune-missing",
        action="store_true",
        help="Drop manifest rows whose posts no longer match the selection (e.g. re-scored); cached latents stay on disk",
    )
    parser.add_argument("--limit", type=int, default=0, help="Only process the first N posts (0 = all, for smoke tests)")
    args = parser.parse_args()

    filters = SelectionFilters(
        min_short_edge=args.min_short_edge,
        min_score=args.min_score,
        score=args.score,
        max_rating=args.max_rating,
        include_hidden_duplicates=args.include_hidden_duplicates,
    )
    quality = QualityTagging(
        tags=_parse_quality_tags(args.quality_tags),
        scorer=args.aesthetic_scorer,
        thresholds=_parse_csv_floats(args.silva_thresholds) if args.silva_thresholds else None,
    )
    posts = query_posts(args.db, args.images_root, filters=filters, quality=quality, era_bounds=parse_era_spec(args.era_tags))
    logger.info("Selected %d posts (%s)", len(posts), filters)
    if args.limit:
        posts = posts[: args.limit]
    if not posts:
        logger.warning("Nothing to do")
        return

    processor = SimpleLatentsProcessor(args.vae_path, dtype=str_to_dtype(args.vae_dtype), base_resolution=args.base_resolution)
    encode_posts(posts, args.output_dir, processor)
    CreateParquetProcessor(args.output_dir)()
    merge_post_columns(args.output_dir / "metadata.parquet", posts, prune_missing=args.prune_missing)
    logger.info("Prepared dataset at %s", args.output_dir)


if __name__ == "__main__":
    main()
