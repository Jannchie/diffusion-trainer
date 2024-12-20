import json
import logging
from bisect import bisect_right
from collections import defaultdict
from collections.abc import Generator
from dataclasses import dataclass
from os import PathLike
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from diffusion_trainer.shared import get_progress

logger = logging.getLogger("diffusion_trainer.dataset")


def process_tags(tags: list[str] | str | None) -> list[str]:
    """Process tags."""
    if tags is None:
        return []
    return tags.split(",") if isinstance(tags, str) else tags


def process_caption(caption: str | None) -> str:
    """Process caption."""
    return caption if caption is not None else ""


@dataclass
class DiffusionTrainingItem:
    key: str
    npz_path: str
    caption: str
    tags: list[str]


@dataclass
class DiffusionBatch:
    img_latents: torch.Tensor
    crop_ltrb: torch.Tensor
    original_size: torch.Tensor
    train_resolution: torch.Tensor
    caption: list[str]
    tags: list[list[str]]


class DiffusionDataset(Dataset):
    def __init__(self, buckets: dict[tuple[int, int], list[DiffusionTrainingItem]]) -> None:
        self.buckets = buckets
        self.bucket_boundaries = []
        self.bucket_keys = list(buckets.keys())

        last_index = 0
        for key in self.bucket_keys:
            length = len(buckets[key])
            last_index += length
            self.bucket_boundaries.append(last_index)
            logger.info("Bucket %s: %s samples", key, length)

    @staticmethod
    def collate_fn(batch: list[dict]) -> DiffusionBatch:
        img_latents = torch.stack([torch.from_numpy(item["img_latents"]) for item in batch])
        crop_ltrb = torch.stack([torch.from_numpy(item["crop_ltrb"]) for item in batch])
        original_size = torch.stack([torch.from_numpy(item["original_size"]) for item in batch])
        train_resolution = torch.stack([torch.from_numpy(item["train_resolution"]) for item in batch])
        caption = [item["caption"] for item in batch]
        tags = [item["tags"] for item in batch]
        return DiffusionBatch(
            img_latents=img_latents,
            crop_ltrb=crop_ltrb,
            original_size=original_size,
            train_resolution=train_resolution,
            caption=caption,
            tags=tags,
        )

    @staticmethod
    def from_metadata(
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
                buckets[tuple(metadata[key].get("train_resolution"))].append(
                    DiffusionTrainingItem(
                        key,
                        str(ds_path / key) + ".npz",
                        process_caption(metadata[key].get("caption")),
                        process_tags(metadata[key].get("tags")),
                    ),
                )
                metadata[key]
            buckets = dict(sorted(buckets.items()))
        logger.info("Buckets created, Here are the buckets information:")
        return DiffusionDataset(buckets)

    def get_bucket_key(self, idx: int) -> tuple[int, int]:
        # Use binary search to find the correct bucket
        bucket_index = bisect_right(self.bucket_boundaries, idx)
        return self.bucket_keys[bucket_index]

    def __len__(self) -> int:
        return sum(len(v) for v in self.buckets.values())

    def __getitem__(self, idx: int) -> dict:
        bucket_key = self.get_bucket_key(idx)
        bucket_items = self.buckets[bucket_key]
        bucket_start_idx = self.bucket_boundaries[self.bucket_keys.index(bucket_key) - 1] if self.bucket_keys.index(bucket_key) > 0 else 0
        item = bucket_items[idx - bucket_start_idx]
        npz = np.load(item.npz_path)
        img_latents = npz.get("latents")
        crop_ltrb = npz.get("crop_ltrb")
        original_size = npz.get("original_size")
        train_resolution = npz.get("train_resolution")
        return {
            "img_latents": img_latents,
            "crop_ltrb": crop_ltrb,
            "original_size": original_size,
            "train_resolution": train_resolution,
            "caption": item.caption,
            "tags": item.tags,
        }


class BucketBasedBatchSampler(Sampler):
    def __init__(self, dataset: DiffusionDataset, batch_size: int, *, shuffle: bool = True) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self) -> Generator[list[int], None, None]:
        batche_indices_list = []
        logger.debug("Prepare batch indices...")
        rng = np.random.default_rng()
        for i, key in enumerate(self.dataset.bucket_keys):
            key_start_idx = self.dataset.bucket_boundaries[i - 1] if i > 0 else 0
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
