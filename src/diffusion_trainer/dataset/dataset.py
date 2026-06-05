import json
import logging
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from pyarrow import parquet as pq
from torch.utils.data import Dataset, Sampler

if TYPE_CHECKING:
    import pandas as pd

from diffusion_trainer.dataset.utils import sharded_path
from diffusion_trainer.shared import get_progress

logger = logging.getLogger("diffusion_trainer.dataset")


def process_tags(tags: list[str] | str | None) -> list[str]:
    """Process tags."""
    if tags is None:
        return []
    return tags.split(",") if isinstance(tags, str) else tags


@dataclass(frozen=True)
class TagFilters:
    """Declarative manifest subsetting: keep a sample iff it has at least one
    ``include_any`` entry (when given) and none of ``exclude``.

    Lets one exported dataset serve many runs (e.g. ``("best quality",)`` for
    a top-tier-only experiment) instead of maintaining parallel exports.
    """

    include_any: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return bool(self.include_any or self.exclude)

    def matches(self, tags: Sequence[str]) -> bool:
        tag_set = set(tags)
        if self.include_any and tag_set.isdisjoint(self.include_any):
            return False
        return not (self.exclude and not tag_set.isdisjoint(self.exclude))


def process_caption(caption: str | None) -> str:
    """Process caption."""
    return caption if caption is not None else ""


@dataclass
class DiffusionTrainingItem:
    npz_path: str
    caption: str | None
    tags: list[str] | None
    # Metadata from the parquet manifest (single source of truth). None falls
    # back to legacy NPZ files that embed these fields next to the latents.
    crop_ltrb: list[int] | None = None
    original_size: list[int] | None = None
    train_resolution: list[int] | None = None
    # Per-tag category names parallel to ``tags`` (quality/artist/copyright/
    # character/general/meta). Empty for datasets prepared without categories;
    # prompt assembly then treats every tag as "general" (legacy behavior).
    tag_categories: list[str] | None = None


@dataclass
class DiffusionBatch:
    img_latents: torch.Tensor
    crop_ltrb: torch.Tensor
    original_size: torch.Tensor
    train_resolution: torch.Tensor
    caption: list[str]
    tags: list[list[str]]
    tag_categories: list[list[str]]


class DiffusionDataset(Dataset):
    def __init__(self, buckets: dict[tuple[int, int], list[DiffusionTrainingItem]]) -> None:
        self.buckets = buckets
        self.bucket_boundaries = []
        self.bucket_starts = []
        self.bucket_keys = list(buckets.keys())

        last_index = 0
        for key in self.bucket_keys:
            self.bucket_starts.append(last_index)
            length = len(buckets[key])
            last_index += length
            self.bucket_boundaries.append(last_index)

    def print_bucket_info(self) -> None:
        for key in self.bucket_keys:
            logger.info("Bucket %s: %s samples", key, len(self.buckets[key]))

    @staticmethod
    def collate_fn(batch: list[dict]) -> DiffusionBatch:
        img_latents = torch.stack([torch.from_numpy(item["img_latents"]) for item in batch])
        crop_ltrb = torch.stack([torch.from_numpy(item["crop_ltrb"]) for item in batch])
        original_size = torch.stack([torch.from_numpy(item["original_size"]) for item in batch])
        train_resolution = torch.stack([torch.from_numpy(item["train_resolution"]) for item in batch])
        caption = [item["caption"] for item in batch]
        tags = [item["tags"] for item in batch]
        tag_categories = [item.get("tag_categories") or [] for item in batch]
        return DiffusionBatch(
            img_latents=img_latents,
            crop_ltrb=crop_ltrb,
            original_size=original_size,
            train_resolution=train_resolution,
            caption=caption,
            tags=tags,
            tag_categories=tag_categories,
        )

    @staticmethod
    def from_ss(
        metadata_path: str | PathLike,
        ds_path: str | PathLike | None = None,
    ) -> "DiffusionDataset":
        buckets: dict[tuple[int, int], list[DiffusionTrainingItem]] = defaultdict(list)
        path = Path(metadata_path)
        ds_path = path.parent if ds_path is None else Path(ds_path)
        metadata = json.load(path.open())
        progress = get_progress()
        with progress:
            for key in progress.track(metadata, description="Processing metadata"):
                train_resolution = metadata[key].get("train_resolution")
                if train_resolution is None:
                    logger.warning("Skipping %s: missing train_resolution", key)
                    continue
                buckets[tuple(train_resolution)].append(
                    DiffusionTrainingItem(
                        str(ds_path / key) + ".npz",
                        process_caption(metadata[key].get("caption")),
                        process_tags(metadata[key].get("tags")),
                    ),
                )
            buckets = dict(sorted(buckets.items()))
        logger.info("Buckets created, Here are the buckets information:")
        return DiffusionDataset(buckets)

    @staticmethod
    def _row_int_list(row: "pd.Series", column: str) -> list[int] | None:
        """Read an int-list column from a parquet row, tolerating missing columns."""
        value = row.get(column)
        if value is None:
            return None
        return [int(v) for v in (value.tolist() if hasattr(value, "tolist") else value)]

    @staticmethod
    def _row_str_list(row: "pd.Series", column: str) -> list[str]:
        """Read a string-list column from a parquet row, tolerating missing columns."""
        value = row.get(column)
        if value is None:
            return []
        if isinstance(value, list):
            return [str(v) for v in value]
        if hasattr(value, "tolist"):
            return [str(v) for v in value.tolist()]
        return [str(value)]

    @staticmethod
    def from_parquet(parquet_path: str | PathLike, *, tag_filters: TagFilters | None = None) -> "DiffusionDataset":
        parquet_path = Path(parquet_path)
        logger.info('Reading dataset from "%s"', parquet_path)
        table = pq.read_table(parquet_path)
        metadata = table.to_pandas()
        buckets: dict[tuple[int, int], list[DiffusionTrainingItem]] = defaultdict(list)
        filtered_out = 0
        for _idx, row in metadata.iterrows():
            key = row["key"]
            # Use SHA256-based directory structure: ab/cd/abcd...npz
            npz_path = sharded_path(parquet_path.parent / "latents", key, "npz")
            train_resolution = DiffusionDataset._row_int_list(row, "train_resolution")
            if train_resolution is None:
                logger.warning("Skipping %s: missing train_resolution", key)
                continue
            tags = DiffusionDataset._row_str_list(row, "tags")
            if tag_filters and not tag_filters.matches(tags):
                filtered_out += 1
                continue
            buckets[(train_resolution[0], train_resolution[1])].append(
                DiffusionTrainingItem(
                    npz_path=npz_path.as_posix(),
                    caption=row.get("caption", ""),  # Use empty string if caption doesn't exist
                    tags=tags,
                    crop_ltrb=DiffusionDataset._row_int_list(row, "crop_ltrb"),
                    original_size=DiffusionDataset._row_int_list(row, "original_size"),
                    train_resolution=train_resolution,
                    tag_categories=DiffusionDataset._row_str_list(row, "tag_categories"),
                ),
            )
        if filtered_out:
            logger.info("Tag filters excluded %d samples (%d kept)", filtered_out, sum(len(v) for v in buckets.values()))
        return DiffusionDataset(buckets)

    def get_bucket_index(self, idx: int) -> int:
        # Use binary search to find the correct bucket
        return bisect_right(self.bucket_boundaries, idx)

    def get_bucket_key(self, idx: int) -> tuple[int, int]:
        return self.bucket_keys[self.get_bucket_index(idx)]

    def __len__(self) -> int:
        return sum(len(v) for v in self.buckets.values())

    @staticmethod
    def _load_item(item: DiffusionTrainingItem) -> dict:
        # collate_fn stacks all four arrays unconditionally, so a file missing any
        # of them must raise here (not return None) to trigger the bucket fallback
        # in __getitem__ instead of crashing later in collate. The context manager
        # also closes the underlying zip handle (zip-npz arrays are in-memory copies).
        # Metadata comes from the parquet manifest when present (latent-only NPZ);
        # legacy NPZ files that embed the fields are the fallback.
        with np.load(item.npz_path) as npz:
            if "latents" not in npz:
                msg = f"npz {item.npz_path} is missing ['latents']"
                raise KeyError(msg)
            meta: dict[str, np.ndarray] = {}
            missing = []
            for key in ("crop_ltrb", "original_size", "train_resolution"):
                value = getattr(item, key)
                if value is not None:
                    meta[key] = np.asarray(value, dtype=np.int64)
                elif key in npz:
                    meta[key] = npz[key]
                else:
                    missing.append(key)
            if missing:
                msg = f"npz {item.npz_path} is missing {missing} (neither in manifest nor embedded)"
                raise KeyError(msg)
            return {
                "img_latents": npz["latents"],
                **meta,
                "caption": item.caption,
                "tags": item.tags,
                "tag_categories": item.tag_categories or [],
            }

    def __getitem__(self, idx: int) -> dict:
        bucket_index = self.get_bucket_index(idx)
        bucket_key = self.bucket_keys[bucket_index]
        bucket_items = self.buckets[bucket_key]
        bucket_start_idx = self.bucket_starts[bucket_index]
        local_idx = idx - bucket_start_idx

        # A single corrupt/half-written npz should not kill a long training run.
        # Fall back to other items in the SAME bucket so latent shapes still match.
        n = len(bucket_items)
        for offset in range(n):
            item = bucket_items[(local_idx + offset) % n]
            try:
                return self._load_item(item)
            except Exception as e:
                logger.warning("Skipping unreadable latent %s: %s", item.npz_path, e)
        msg = f"All {n} items in bucket {bucket_key} are unreadable"
        raise RuntimeError(msg)


class BucketBasedBatchSampler(Sampler):
    def __init__(self, dataset: DiffusionDataset, batch_size: int, *, shuffle: bool = True, seed: int = 47) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Generator[list[int], None, None]:
        rng = np.random.default_rng(self.seed + self.epoch)
        batche_indices_list = []
        logger.debug("Prepare batch indices...")
        for i, key in enumerate(self.dataset.bucket_keys):
            key_start_idx = self.dataset.bucket_starts[i]
            bucket_items = self.dataset.buckets[key]
            indices = list(range(len(bucket_items)))
            if self.shuffle:
                rng.shuffle(indices)
            for j in range(0, len(bucket_items), self.batch_size):
                batch_indices = [key_start_idx + indices[k] for k in range(j, min(j + self.batch_size, len(indices)))]
                batche_indices_list.append(batch_indices)
        if self.shuffle:
            rng.shuffle(batche_indices_list)
        logger.debug("Batch indices prepared!")
        for batch_indices in batche_indices_list:
            yield batch_indices

    def __len__(self) -> int:
        return sum(len(v) // self.batch_size + int(len(v) % self.batch_size > 0) for v in self.dataset.buckets.values())
