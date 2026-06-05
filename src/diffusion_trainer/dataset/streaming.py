"""Stream training batches directly from exported tar shards (local or HuggingFace Hub).

The streaming dataset consumes the distribution format produced by
``diffusion_trainer.dataset.sharing.export_dataset`` without unpacking it:

- ``metadata.parquet`` is loaded fully into memory (it is small) and acts as
  the per-sample metadata lookup, exactly like the map-style dataset.
- Shards download lazily inside the (spawn-started) DataLoader workers, so
  epoch 1 overlaps training with the transfer and later epochs hit the local
  HF cache; interrupted downloads resume for free. ``from_hub`` pins the
  resolved commit sha so cached shards resolve without any network call.
  Call :meth:`StreamingDiffusionDataset.prefetch_shards` to warm the cache
  up front instead (fail-fast, fully local epoch 1).
- Shards are bucket-pure (one training resolution per shard), so batches are
  assembled from consecutive samples and stay resolution-consistent. Partial
  batches are flushed at shard boundaries, which keeps ``len()`` exact and
  independent of worker count or epoch.

Shuffling is two-level: the shard order is reshuffled every epoch (same
permutation in every worker, which then take a disjoint ``[id::num_workers]``
slice), and samples pass through a fixed-size shuffle buffer inside each
shard. Member order within a shard is already content-random (sorted by
SHA256), so this is enough randomness for SGD.
"""

import logging
import tarfile
from collections import defaultdict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from io import BytesIO
from os import PathLike
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download
from pyarrow import parquet as pq
from torch.utils.data import IterableDataset, get_worker_info

from diffusion_trainer.dataset.dataset import TagFilters
from diffusion_trainer.dataset.sharing import METADATA_FILENAME, SHARD_MEMBER_RE, SHARDS_DIR_NAME, read_verified_member
from diffusion_trainer.dataset.utils import row_latents_meta

logger = logging.getLogger("diffusion_trainer.dataset")

ShardSource = Callable[[str], Path]
"""Resolve a shard file name (e.g. ``1024x1024-00000.tar``) to a local tar path."""


@dataclass(frozen=True)
class LocalShardSource:
    """Shards living in a local export directory."""

    export_dir: Path

    def __call__(self, shard_name: str) -> Path:
        return self.export_dir / SHARDS_DIR_NAME / shard_name


@dataclass(frozen=True)
class HfShardSource:
    """Shards in a HuggingFace dataset repo, downloaded lazily and cached by the hub client.

    ``from_hub`` pins ``revision`` to a commit sha, so already-cached shards
    resolve as a pure filesystem lookup (no network) inside DataLoader workers.
    """

    repo_id: str
    revision: str | None = None

    def __call__(self, shard_name: str) -> Path:
        return Path(hf_hub_download(self.repo_id, f"{SHARDS_DIR_NAME}/{shard_name}", repo_type="dataset", revision=self.revision))


def _pop_random(buffer: list[dict], rng: np.random.Generator) -> dict:
    i = int(rng.integers(len(buffer)))
    buffer[i], buffer[-1] = buffer[-1], buffer[i]
    return buffer.pop()


class StreamingDiffusionDataset(IterableDataset):
    """Iterable dataset yielding ready-made batches (``list[dict]``) from tar shards.

    Use with ``DataLoader(dataset, batch_size=None, collate_fn=DiffusionDataset.collate_fn)``;
    automatic batching must stay disabled because batches are pre-assembled here
    to keep every batch bucket-consistent. ``len()`` is the exact number of
    batches per epoch. Call ``set_epoch`` before each epoch to reshuffle; the
    DataLoader must not use ``persistent_workers`` so the updated epoch reaches
    workers through re-pickling.
    """

    def __init__(self, rows: list[dict], shard_source: ShardSource, batch_size: int, *, seed: int = 47, shuffle_buffer_size: int = 256) -> None:
        self.shard_source = shard_source
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.shuffle = True
        self.drop_last = False
        self.epoch = 0

        self.rows_by_key: dict[str, dict] = {}
        skipped = 0
        for row in rows:
            if not row.get("shard") or row_latents_meta(row) is None:
                skipped += 1
                continue
            self.rows_by_key[row["key"]] = row
        if skipped:
            logger.warning("Ignored %d manifest rows without shard/metadata fields", skipped)
        if not self.rows_by_key:
            msg = "No usable rows in the manifest; was the dataset exported with scripts/export_dataset.py?"
            raise ValueError(msg)
        self._rebuild_groups()

    def _rebuild_groups(self) -> None:
        """Recompute per-(shard, bucket) counts and the shard list from rows_by_key."""
        group_counts: dict[tuple[str, tuple[int, int]], int] = defaultdict(int)
        for row in self.rows_by_key.values():
            train_resolution = row["train_resolution"]
            group_counts[(row["shard"], (int(train_resolution[0]), int(train_resolution[1])))] += 1
        self._group_counts = dict(group_counts)
        self.shard_names = sorted({shard for shard, _ in self._group_counts})

    def apply_tag_filters(self, tag_filters: TagFilters | None) -> "StreamingDiffusionDataset":
        """Drop manifest rows not matching the filters (chainable, returns self).

        Filtered rows leave ``rows_by_key``, so ``_read_member`` skips their
        shard members and ``len()`` stays exact for the subset — one exported
        dataset can serve many runs (e.g. a best-quality-only experiment).
        """
        if not tag_filters:
            return self
        kept = {key: row for key, row in self.rows_by_key.items() if tag_filters.matches(list(row.get("tags") or []))}
        filtered_out = len(self.rows_by_key) - len(kept)
        if not kept:
            msg = f"Tag filters {tag_filters} match no samples in the manifest"
            raise ValueError(msg)
        if filtered_out:
            logger.info("Tag filters excluded %d samples (%d kept)", filtered_out, len(kept))
            self.rows_by_key = kept
            self._rebuild_groups()
        return self

    @classmethod
    def from_export_dir(cls, export_dir: str | PathLike, batch_size: int, *, seed: int = 47, shuffle_buffer_size: int = 256) -> "StreamingDiffusionDataset":
        export_dir = Path(export_dir)
        rows = pq.read_table(export_dir / METADATA_FILENAME).to_pylist()
        return cls(rows, LocalShardSource(export_dir), batch_size, seed=seed, shuffle_buffer_size=shuffle_buffer_size)

    @classmethod
    def from_hub(
        cls,
        repo_id: str,
        batch_size: int,
        *,
        revision: str | None = None,
        seed: int = 47,
        shuffle_buffer_size: int = 256,
    ) -> "StreamingDiffusionDataset":
        metadata_path = hf_hub_download(repo_id, METADATA_FILENAME, repo_type="dataset", revision=revision)
        rows = pq.read_table(metadata_path).to_pylist()
        # Pin the resolved commit (snapshots/<sha>/metadata.parquet): every
        # sample of the run comes from one revision, and cached shards resolve
        # through the commit-hash fast path without any network round-trip.
        commit_sha = Path(metadata_path).parent.name
        return cls(rows, HfShardSource(repo_id, commit_sha), batch_size, seed=seed, shuffle_buffer_size=shuffle_buffer_size)

    def prefetch_shards(self) -> None:
        """Resolve every shard up front (e.g. warm the HF cache before training).

        Optional: workers download lazily by default, overlapping epoch 1 with
        the transfer. Prefetching trades that overlap for a fully local epoch 1
        (useful on flaky networks or to fail fast on missing shards).
        """
        logger.info("Prefetching %d shards", len(self.shard_names))
        for shard_name in self.shard_names:
            self.shard_source(shard_name)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def print_bucket_info(self) -> None:
        bucket_counts: dict[tuple[int, int], int] = defaultdict(int)
        for (_shard, bucket), count in self._group_counts.items():
            bucket_counts[bucket] += count
        for bucket in sorted(bucket_counts):
            logger.info("Bucket %s: %s samples (streaming)", bucket, bucket_counts[bucket])

    def __len__(self) -> int:
        """Exact batches per epoch: partial batches flush per (shard, bucket) group."""
        if self.drop_last:
            return sum(count // self.batch_size for count in self._group_counts.values())
        return sum(-(-count // self.batch_size) for count in self._group_counts.values())

    def __iter__(self) -> Iterator[list[dict]]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        num_workers = worker.num_workers if worker else 1
        shard_names = self._epoch_shard_order()
        rng = np.random.default_rng((self.seed, self.epoch, worker_id))
        for shard_name in shard_names[worker_id::num_workers]:
            yield from self._iter_shard_batches(shard_name, rng)

    def _epoch_shard_order(self) -> list[str]:
        """Per-epoch shard permutation; identical in every worker so slices stay disjoint."""
        if not self.shuffle:
            return self.shard_names
        rng = np.random.default_rng((self.seed, self.epoch))
        return [self.shard_names[i] for i in rng.permutation(len(self.shard_names))]

    def _iter_shard_batches(self, shard_name: str, rng: np.random.Generator) -> Iterator[list[dict]]:
        try:
            tar_path = Path(self.shard_source(shard_name))
        except Exception:
            logger.exception("Failed to fetch shard %s, skipping its batches this epoch", shard_name)
            return
        # Shards are bucket-pure, but route per bucket anyway so a hand-built
        # mixed shard still produces resolution-consistent batches.
        pending: dict[tuple[int, int], list[dict]] = defaultdict(list)
        for sample in self._iter_shard_samples(tar_path, rng):
            bucket = (int(sample["train_resolution"][0]), int(sample["train_resolution"][1]))
            batch = pending[bucket]
            batch.append(sample)
            if len(batch) == self.batch_size:
                yield batch
                pending[bucket] = []
        if not self.drop_last:
            yield from (batch for batch in pending.values() if batch)

    def _iter_shard_samples(self, tar_path: Path, rng: np.random.Generator) -> Iterator[dict]:
        buffer: list[dict] = []
        with tarfile.open(tar_path, mode="r|") as tar:  # pure sequential read, also works on pipes
            for member in tar:
                sample = self._read_member(tar, member)
                if sample is None:
                    continue
                if not self.shuffle:
                    yield sample
                    continue
                buffer.append(sample)
                if len(buffer) >= self.shuffle_buffer_size:
                    yield _pop_random(buffer, rng)
        while buffer:
            yield _pop_random(buffer, rng)

    def _read_member(self, tar: tarfile.TarFile, member: tarfile.TarInfo) -> dict | None:
        match = SHARD_MEMBER_RE.match(member.name)
        if match is None or not member.isfile():
            logger.warning("Skipping unexpected tar member %r", member.name)
            return None
        row = self.rows_by_key.get(match.group(3))
        if row is None:
            return None  # not part of this manifest (e.g. filtered rows)
        data = read_verified_member(tar, member, row)
        if data is None:
            return None
        with np.load(BytesIO(data)) as npz:
            if "latents" not in npz:
                logger.warning("npz for %s has no latents, skipping", row["key"])
                return None
            latents = npz["latents"]
        return {
            "img_latents": latents,
            "crop_ltrb": np.asarray(row["crop_ltrb"], dtype=np.int64),
            "original_size": np.asarray(row["original_size"], dtype=np.int64),
            "train_resolution": np.asarray(row["train_resolution"], dtype=np.int64),
            "caption": row.get("caption") or "",
            "tags": list(row.get("tags") or []),
            "tag_categories": list(row.get("tag_categories") or []),
        }
