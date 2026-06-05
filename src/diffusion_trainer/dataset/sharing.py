"""Export/import prepared datasets as bucket-grouped tar shards for sharing.

Distribution format (one directory, suitable for a HuggingFace dataset repo):

- ``metadata.parquet`` -- the training manifest plus ``shard`` and
  ``npz_sha256`` columns added at export time.
- ``shards/<W>x<H>-<idx>.tar`` -- latent ``.npz`` files grouped by training
  resolution, member paths keep the canonical ``ab/cd/<sha256>.npz`` layout.
- ``README.md`` -- generated dataset card.

Each shard only contains samples of a single bucket, so the format stays
consumable by sequential/streaming readers later without repacking.
"""

import hashlib
import logging
import re
import tarfile
from collections import defaultdict
from dataclasses import dataclass, field
from io import BytesIO
from os import PathLike
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_trainer.dataset.utils import row_latents_meta, sharded_path, write_latents_meta

logger = logging.getLogger("diffusion_trainer.dataset")

METADATA_FILENAME = "metadata.parquet"
SHARDS_DIR_NAME = "shards"
DEFAULT_SHARD_SIZE_BYTES = 1 << 30  # 1 GiB

# Tar members must keep the canonical sharded layout; anything else (e.g. a
# path-traversal name in a hand-crafted tar) is rejected at import time.
SHARD_MEMBER_RE = re.compile(r"^([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{64})\.npz$")


def _member_name(key: str) -> str:
    """Tar member path for a key: the canonical sharded layout, relative."""
    return sharded_path("", key, "npz").as_posix()


def read_verified_member(tar: tarfile.TarFile, member: tarfile.TarInfo, row: dict) -> bytes | None:
    """Read one tar member's bytes, verifying the manifest's ``npz_sha256`` when present."""
    fileobj = tar.extractfile(member)
    if fileobj is None:
        return None
    data = fileobj.read()
    expected = row.get("npz_sha256")
    if expected and hashlib.sha256(data).hexdigest() != expected:
        logger.warning("Checksum mismatch for %s, skipping", row["key"])
        return None
    return data


@dataclass
class ExportStats:
    exported: int = 0
    skipped: int = 0
    shards: int = 0
    total_bytes: int = 0
    bucket_summary: list[tuple[str, int, int, int]] = field(default_factory=list)
    """Per-bucket rows of (bucket name, samples, shard count, bytes)."""


@dataclass
class ImportStats:
    imported: int = 0
    skipped_existing: int = 0
    failed: int = 0


class _ShardWriter:
    """Append members into ``<prefix>-<idx>.tar``, rolling once the size budget is hit.

    Tar metadata (mtime/uid/gid/mode) is fixed so re-exporting an unchanged
    dataset produces byte-identical shards (dedup-friendly on remotes).
    """

    def __init__(self, shards_dir: Path, prefix: str, shard_size_bytes: int) -> None:
        self.shards_dir = shards_dir
        self.prefix = prefix
        self.shard_size_bytes = shard_size_bytes
        self.shard_names: list[str] = []
        self._tar: tarfile.TarFile | None = None
        self._shard_name = ""
        self._index = 0
        self._written = 0

    def _open_next(self) -> tarfile.TarFile:
        self.close()
        self._shard_name = f"{self.prefix}-{self._index:05d}.tar"
        self._index += 1
        self._written = 0
        # The handle intentionally outlives this method (rolled/closed via close()).
        self._tar = tarfile.open(self.shards_dir / self._shard_name, "w")  # noqa: SIM115
        self.shard_names.append(self._shard_name)
        return self._tar

    def add(self, member_name: str, data: bytes) -> str:
        """Add one member and return the shard file name it landed in."""
        tar = self._tar
        if tar is None or (self._written > 0 and self._written + len(data) > self.shard_size_bytes):
            tar = self._open_next()
        info = tarfile.TarInfo(member_name)
        info.size = len(data)
        info.mtime = 0
        info.mode = 0o644
        tar.addfile(info, BytesIO(data))
        self._written += len(data)
        return self._shard_name

    def close(self) -> None:
        if self._tar is not None:
            self._tar.close()
            self._tar = None


def _group_rows_by_bucket(rows: list[dict], latents_dir: Path) -> tuple[dict[tuple[int, int], list[dict]], int]:
    """Group manifest rows by training resolution, dropping rows without a readable npz."""
    buckets: dict[tuple[int, int], list[dict]] = defaultdict(list)
    skipped = 0
    for row in rows:
        train_resolution = row.get("train_resolution")
        if not train_resolution:
            logger.warning("Skipping %s: missing train_resolution", row.get("key"))
            skipped += 1
            continue
        if not sharded_path(latents_dir, row["key"], "npz").exists():
            logger.warning("Skipping %s: npz file not found", row["key"])
            skipped += 1
            continue
        buckets[(int(train_resolution[0]), int(train_resolution[1]))].append(row)
    return buckets, skipped


def export_dataset(
    dataset_dir: str | PathLike,
    output_dir: str | PathLike,
    *,
    shard_size_bytes: int = DEFAULT_SHARD_SIZE_BYTES,
    dataset_name: str | None = None,
    vae_name: str | None = None,
) -> ExportStats:
    """Pack a prepared dataset directory into the bucket-grouped shard format."""
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.exists():
        msg = f"{metadata_path} not found; run the prepare pipeline (CreateParquetProcessor) first"
        raise FileNotFoundError(msg)

    table = pq.read_table(metadata_path)
    latents_dir = dataset_dir / "latents"
    buckets, skipped = _group_rows_by_bucket(table.to_pylist(), latents_dir)
    if not buckets:
        msg = f"No exportable rows in {metadata_path}"
        raise ValueError(msg)

    shards_dir = output_dir / SHARDS_DIR_NAME
    shards_dir.mkdir(parents=True, exist_ok=True)

    stats = ExportStats(skipped=skipped)
    exported_rows: list[dict] = []
    for bucket in sorted(buckets):
        bucket_name = f"{bucket[0]}x{bucket[1]}"
        writer = _ShardWriter(shards_dir, bucket_name, shard_size_bytes)
        bucket_bytes = 0
        try:
            for row in sorted(buckets[bucket], key=lambda r: r["key"]):
                key = row["key"]
                data = sharded_path(latents_dir, key, "npz").read_bytes()
                row["shard"] = writer.add(_member_name(key), data)
                row["npz_sha256"] = hashlib.sha256(data).hexdigest()
                bucket_bytes += len(data)
                exported_rows.append(row)
        finally:
            writer.close()
        stats.bucket_summary.append((bucket_name, len(buckets[bucket]), len(writer.shard_names), bucket_bytes))
        stats.shards += len(writer.shard_names)
        stats.total_bytes += bucket_bytes
    stats.exported = len(exported_rows)

    column_names = list(table.column_names)
    column_names.extend(extra for extra in ("shard", "npz_sha256") if extra not in column_names)
    out_table = pa.table({name: [row.get(name) for row in exported_rows] for name in column_names})
    pq.write_table(out_table, output_dir / METADATA_FILENAME)

    name = dataset_name or output_dir.name
    (output_dir / "README.md").write_text(_render_dataset_card(name, stats, vae_name), encoding="utf-8")
    logger.info("Exported %d samples (%d skipped) into %d shards at %s", stats.exported, stats.skipped, stats.shards, output_dir)
    return stats


def _human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GiB"


def _render_dataset_card(name: str, stats: ExportStats, vae_name: str | None) -> str:
    bucket_rows = "\n".join(f"| {bucket} | {samples} | {shards} | {_human_size(num_bytes)} |" for bucket, samples, shards, num_bytes in stats.bucket_summary)
    vae_line = vae_name or "_fill in the VAE used to encode the latents_"
    return f"""# {name}

Pre-encoded [diffusion-trainer](https://github.com/Jannchie/diffusion-trainer) dataset: \
VAE latents packed into bucket-grouped tar shards, plus a parquet manifest.

## Layout

| file | description |
| --- | --- |
| `metadata.parquet` | one row per image: `key` (SHA256), `tags`, `train_resolution`, `original_size`, `crop_ltrb`, `shard`, `npz_sha256`, optional extras |
| `shards/<W>x<H>-<idx>.tar` | latent `.npz` files (member path `ab/cd/<sha256>.npz`), grouped by training resolution |

## Buckets

| bucket | samples | shards | size |
| --- | --- | --- | --- |
{bucket_rows}

Total: {stats.exported} samples, {_human_size(stats.total_bytes)}.

## Reproducibility

- VAE: {vae_line}
- Latents are raw VAE encoder outputs; `scaling_factor` is **not** pre-applied (the trainer applies it at training time).
- Each shard only contains samples of a single bucket, so the dataset is also consumable by sequential/streaming readers.

## Import

```bash
uv run python scripts/import_dataset.py --source <this-repo-id-or-dir> --target-dir datasets/{name}
```

Then point `dataset_path` in the training config to `datasets/{name}`.
"""


def _extract_shard(shard_path: Path, rows_by_key: dict[str, dict], latents_dir: Path, stats: ImportStats) -> None:
    with tarfile.open(shard_path, "r") as tar:
        for member in tar:
            match = SHARD_MEMBER_RE.match(member.name)
            if match is None or not member.isfile():
                logger.warning("Skipping unexpected tar member %r in %s", member.name, shard_path.name)
                stats.failed += 1
                continue
            key = match.group(3)
            row = rows_by_key.get(key)
            if row is None:
                logger.warning("Skipping %s: not present in metadata.parquet", key)
                stats.failed += 1
                continue
            dst = sharded_path(latents_dir, key, "npz")
            if dst.exists():
                stats.skipped_existing += 1
                continue
            data = read_verified_member(tar, member, row)
            if data is None:
                stats.failed += 1
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
            stats.imported += 1


def _materialize_tags(rows: list[dict], tags_dir: Path) -> None:
    """Write tags back as sidecar txt files so re-running the prepare pipeline keeps them."""
    for row in rows:
        tags = row.get("tags") or []
        if not tags:
            continue
        tag_path = sharded_path(tags_dir, row["key"], "txt")
        if tag_path.exists():
            continue
        tag_path.parent.mkdir(parents=True, exist_ok=True)
        tag_path.write_text(", ".join(tags), encoding="utf-8")


def import_dataset(source_dir: str | PathLike, target_dir: str | PathLike) -> ImportStats:
    """Unpack an exported dataset into the local prepared-dataset layout.

    The result is equivalent to a locally prepared dataset: latent npz tree,
    sidecar tag files, ``latents/latents_meta.parquet`` and the training
    ``metadata.parquet``. Already-extracted npz files are skipped, so an
    interrupted import can simply be re-run.
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    metadata_path = source_dir / METADATA_FILENAME
    if not metadata_path.exists():
        msg = f"{metadata_path} not found; not an exported dataset directory"
        raise FileNotFoundError(msg)
    table = pq.read_table(metadata_path)
    if "shard" not in table.column_names:
        msg = f"{metadata_path} has no 'shard' column; not an exported dataset"
        raise ValueError(msg)

    rows = table.to_pylist()
    rows_by_key = {row["key"]: row for row in rows}
    latents_dir = target_dir / "latents"
    stats = ImportStats()
    for shard_name in sorted({row["shard"] for row in rows}):
        shard_path = source_dir / SHARDS_DIR_NAME / shard_name
        if not shard_path.exists():
            logger.warning("Shard %s not found, skipping its samples", shard_name)
            stats.failed += sum(1 for row in rows if row["shard"] == shard_name)
            continue
        _extract_shard(shard_path, rows_by_key, latents_dir, stats)

    _materialize_tags(rows, target_dir / "tags")
    latents_meta = {row["key"]: meta for row in rows if (meta := row_latents_meta(row)) is not None}
    if latents_meta:
        write_latents_meta(latents_dir, latents_meta)
    target_dir.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, target_dir / METADATA_FILENAME)
    logger.info("Imported %d samples into %s (%d already present, %d failed)", stats.imported, target_dir, stats.skipped_existing, stats.failed)
    return stats
