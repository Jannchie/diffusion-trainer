"""Tests for StreamingDiffusionDataset (training directly from exported tar shards)."""

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import DataLoader

from diffusion_trainer.dataset.dataset import DiffusionBatch, DiffusionDataset
from diffusion_trainer.dataset.sharing import export_dataset
from diffusion_trainer.dataset.streaming import StreamingDiffusionDataset
from diffusion_trainer.dataset.utils import sharded_path

BUCKET_A = [1024, 1024]
BUCKET_B = [832, 1216]


def make_export(tmp_path: Path, *, n_a: int = 7, n_b: int = 3, shard_size_bytes: int = 1 << 30) -> Path:
    """Build a prepared dataset (caption = key for traceability) and export it."""
    dataset_dir = tmp_path / "ds"
    keys, resolutions = [], []
    for i in range(n_a + n_b):
        key = f"{i:02x}" * 32
        npz_path = sharded_path(dataset_dir / "latents", key, "npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, latents=np.full((4, 8, 8), i, dtype=np.float32))
        keys.append(key)
        resolutions.append(BUCKET_A if i < n_a else BUCKET_B)
    table = pa.table({
        "key": keys,
        "caption": keys,  # lets tests identify samples in collated batches
        "tags": [["t"]] * len(keys),
        "train_resolution": resolutions,
        "original_size": [[2048, 2048]] * len(keys),
        "crop_ltrb": [[0, 0, 1024, 1024]] * len(keys),
    })
    pq.write_table(table, dataset_dir / "metadata.parquet")
    export_dir = tmp_path / "export"
    export_dataset(dataset_dir, export_dir, shard_size_bytes=shard_size_bytes)
    return export_dir


def collect_batches(dataset: StreamingDiffusionDataset) -> list[list[dict]]:
    return list(iter(dataset))


def test_yields_every_sample_once_in_bucket_consistent_batches(tmp_path: Path) -> None:
    dataset = StreamingDiffusionDataset.from_export_dir(make_export(tmp_path), batch_size=2)

    batches = collect_batches(dataset)

    assert len(batches) == len(dataset)
    captions = [sample["caption"] for batch in batches for sample in batch]
    assert sorted(captions) == sorted(f"{i:02x}" * 32 for i in range(10))
    for batch in batches:
        assert len(batch) <= 2
        resolutions = {tuple(sample["train_resolution"].tolist()) for sample in batch}
        assert len(resolutions) == 1
    # 7 samples @1024 -> 4 batches, 3 samples @832 -> 2 batches.
    assert len(batches) == 6


def test_len_is_exact_with_many_small_shards(tmp_path: Path) -> None:
    # 1-byte shard budget -> one sample per shard -> every batch is a partial of 1.
    dataset = StreamingDiffusionDataset.from_export_dir(make_export(tmp_path, shard_size_bytes=1), batch_size=2)

    assert len(dataset) == 10
    assert [len(batch) for batch in collect_batches(dataset)] == [1] * 10

    dataset.drop_last = True
    assert len(dataset) == 0
    assert collect_batches(dataset) == []


def test_epoch_reshuffles_deterministically(tmp_path: Path) -> None:
    dataset = StreamingDiffusionDataset.from_export_dir(make_export(tmp_path, shard_size_bytes=1), batch_size=1, shuffle_buffer_size=4)

    def epoch_captions(epoch: int) -> list[str]:
        dataset.set_epoch(epoch)
        return [sample["caption"] for batch in collect_batches(dataset) for sample in batch]

    assert epoch_captions(0) == epoch_captions(0)  # same epoch -> same order
    assert epoch_captions(0) != epoch_captions(1)  # new epoch -> reshuffled
    assert sorted(epoch_captions(0)) == sorted(epoch_captions(1))  # same samples


def test_multiworker_dataloader_yields_diffusion_batches(tmp_path: Path) -> None:
    dataset = StreamingDiffusionDataset.from_export_dir(make_export(tmp_path, shard_size_bytes=3000), batch_size=2)
    loader = DataLoader(dataset, batch_size=None, collate_fn=DiffusionDataset.collate_fn, num_workers=2)

    captions = []
    for batch in loader:
        assert isinstance(batch, DiffusionBatch)
        assert batch.img_latents.shape[1:] == (4, 8, 8)
        assert batch.img_latents.shape[0] == len(batch.caption)
        captions.extend(batch.caption)
    assert sorted(captions) == sorted(f"{i:02x}" * 32 for i in range(10))


def test_checksum_mismatch_skips_sample(tmp_path: Path) -> None:
    export_dir = make_export(tmp_path)
    table = pq.read_table(export_dir / "metadata.parquet")
    checksums = table.column("npz_sha256").to_pylist()
    checksums[0] = "0" * 64
    table = table.set_column(table.column_names.index("npz_sha256"), "npz_sha256", pa.array(checksums))
    pq.write_table(table, export_dir / "metadata.parquet")

    dataset = StreamingDiffusionDataset.from_export_dir(export_dir, batch_size=2)
    captions = [sample["caption"] for batch in collect_batches(dataset) for sample in batch]

    assert len(captions) == 9  # the tampered sample is dropped, the rest survive
