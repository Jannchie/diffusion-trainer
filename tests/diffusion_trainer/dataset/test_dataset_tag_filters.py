"""Declarative manifest subsetting: TagFilters across both dataset paths."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diffusion_trainer.dataset.dataset import DiffusionDataset, TagFilters
from diffusion_trainer.dataset.sharing import export_dataset
from diffusion_trainer.dataset.streaming import StreamingDiffusionDataset
from diffusion_trainer.dataset.utils import sharded_path

QUALITY_BY_INDEX = ["best quality", "best quality", "good quality", "normal quality"]


def make_dataset(tmp_path: Path) -> Path:
    """Four samples, one bucket; tags carry a per-sample quality tier."""
    dataset_dir = tmp_path / "ds"
    keys = [f"{i:02x}" * 32 for i in range(len(QUALITY_BY_INDEX))]
    for i, key in enumerate(keys):
        npz_path = sharded_path(dataset_dir / "latents", key, "npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, latents=np.full((4, 8, 8), i, dtype=np.float32))
    table = pa.table({
        "key": keys,
        "caption": keys,  # lets tests identify samples in loaded items
        "tags": [[quality, "1girl"] for quality in QUALITY_BY_INDEX],
        "train_resolution": [[1024, 1024]] * len(keys),
        "original_size": [[2048, 2048]] * len(keys),
        "crop_ltrb": [[0, 0, 1024, 1024]] * len(keys),
    })
    pq.write_table(table, dataset_dir / "metadata.parquet")
    return dataset_dir


def test_tag_filters_semantics() -> None:
    assert TagFilters().matches(["best quality", "1girl"])
    assert not TagFilters()  # empty filter is falsy -> treated as no-op
    assert TagFilters(include_any=("best quality",)).matches(["best quality", "1girl"])
    assert not TagFilters(include_any=("best quality",)).matches(["good quality", "1girl"])
    assert TagFilters(include_any=("good quality", "best quality")).matches(["good quality"])  # OR semantics
    assert not TagFilters(exclude=("oldest",)).matches(["best quality", "oldest"])
    assert not TagFilters(include_any=("best quality",), exclude=("oldest",)).matches(["best quality", "oldest"])  # exclude wins
    assert not TagFilters(include_any=("best quality",)).matches([])


def test_from_parquet_applies_filters(tmp_path: Path) -> None:
    parquet_path = make_dataset(tmp_path) / "metadata.parquet"

    full = DiffusionDataset.from_parquet(parquet_path)
    best_only = DiffusionDataset.from_parquet(parquet_path, tag_filters=TagFilters(include_any=("best quality",)))
    floor = DiffusionDataset.from_parquet(parquet_path, tag_filters=TagFilters(include_any=("best quality", "good quality")))
    no_normal = DiffusionDataset.from_parquet(parquet_path, tag_filters=TagFilters(exclude=("normal quality",)))

    assert len(full) == 4
    assert len(best_only) == 2
    assert len(floor) == 3
    assert len(no_normal) == 3
    assert all("best quality" in best_only[i]["tags"] for i in range(len(best_only)))


def test_streaming_applies_filters(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dataset(make_dataset(tmp_path), export_dir)

    dataset = StreamingDiffusionDataset.from_export_dir(export_dir, batch_size=2).apply_tag_filters(TagFilters(include_any=("best quality",)))

    assert len(dataset) == 1  # 2 matching samples -> one batch
    samples = [sample for batch in dataset for sample in batch]
    assert sorted(sample["caption"] for sample in samples) == [f"{i:02x}" * 32 for i in range(2)]
    assert all("best quality" in sample["tags"] for sample in samples)


def test_streaming_filter_matching_nothing_raises(tmp_path: Path) -> None:
    export_dir = tmp_path / "export"
    export_dataset(make_dataset(tmp_path), export_dir)
    dataset = StreamingDiffusionDataset.from_export_dir(export_dir, batch_size=2)

    with pytest.raises(ValueError, match="match no samples"):
        dataset.apply_tag_filters(TagFilters(include_any=("worst quality",)))
