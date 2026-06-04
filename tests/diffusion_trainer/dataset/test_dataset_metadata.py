"""Tests for the parquet-manifest metadata flow (latent-only NPZ files)."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from diffusion_trainer.dataset.dataset import DiffusionDataset, DiffusionTrainingItem
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.utils import load_latents_meta, sharded_path, write_latents_meta

KEY_A = "a" * 64
KEY_B = "b" * 64

META_A = {"train_resolution": [1024, 1024], "original_size": [2048, 2048], "crop_ltrb": [0, 0, 1024, 1024]}
META_B = {"train_resolution": [832, 1216], "original_size": [1000, 1500], "crop_ltrb": [0, 12, 832, 1228]}


def write_latent_only_npz(latents_dir: Path, key: str) -> Path:
    npz_path = sharded_path(latents_dir, key, "npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, latents=np.zeros((4, 8, 8), dtype=np.float32))
    return npz_path

def write_legacy_npz(latents_dir: Path, key: str, meta: dict[str, list[int]]) -> Path:
    npz_path = sharded_path(latents_dir, key, "npz")
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        latents=np.zeros((4, 8, 8), dtype=np.float32),
        train_resolution=np.asarray(meta["train_resolution"], dtype=np.int64),
        original_size=np.asarray(meta["original_size"], dtype=np.int64),
        crop_ltrb=np.asarray(meta["crop_ltrb"], dtype=np.int64),
    )
    return npz_path


def test_load_item_uses_manifest_metadata(tmp_path: Path) -> None:
    npz_path = write_latent_only_npz(tmp_path / "latents", KEY_A)
    item = DiffusionTrainingItem(npz_path=npz_path.as_posix(), caption="", tags=[], **META_A)

    loaded = DiffusionDataset._load_item(item)  # noqa: SLF001

    assert loaded["img_latents"].shape == (4, 8, 8)
    np.testing.assert_array_equal(loaded["train_resolution"], META_A["train_resolution"])
    np.testing.assert_array_equal(loaded["original_size"], META_A["original_size"])
    np.testing.assert_array_equal(loaded["crop_ltrb"], META_A["crop_ltrb"])


def test_load_item_falls_back_to_legacy_npz(tmp_path: Path) -> None:
    npz_path = write_legacy_npz(tmp_path / "latents", KEY_A, META_A)
    item = DiffusionTrainingItem(npz_path=npz_path.as_posix(), caption="", tags=[])

    loaded = DiffusionDataset._load_item(item)  # noqa: SLF001

    np.testing.assert_array_equal(loaded["crop_ltrb"], META_A["crop_ltrb"])


def test_latents_meta_roundtrip(tmp_path: Path) -> None:
    write_latents_meta(tmp_path, {KEY_A: META_A, KEY_B: META_B})

    assert load_latents_meta(tmp_path) == {KEY_A: META_A, KEY_B: META_B}


def test_create_parquet_from_latents_meta_end_to_end(tmp_path: Path) -> None:
    latents_dir = tmp_path / "latents"
    write_latent_only_npz(latents_dir, KEY_A)
    write_latent_only_npz(latents_dir, KEY_B)
    write_latents_meta(latents_dir, {KEY_A: META_A, KEY_B: META_B})

    tag_path = sharded_path(tmp_path / "tags", KEY_A, "txt")
    tag_path.parent.mkdir(parents=True, exist_ok=True)
    tag_path.write_text("1girl, solo", encoding="utf-8")

    CreateParquetProcessor(tmp_path)(max_workers=2)

    parquet_path = tmp_path / "metadata.parquet"
    table = pq.read_table(parquet_path)
    rows = {row["key"]: row for row in table.to_pylist()}
    assert rows[KEY_A]["tags"] == ["1girl", "solo"]
    assert rows[KEY_A]["crop_ltrb"] == META_A["crop_ltrb"]
    assert rows[KEY_B]["train_resolution"] == META_B["train_resolution"]

    dataset = DiffusionDataset.from_parquet(parquet_path)
    assert len(dataset) == 2
    loaded = dataset[0]
    assert loaded["img_latents"].shape == (4, 8, 8)
    assert loaded["train_resolution"].tolist() in (META_A["train_resolution"], META_B["train_resolution"])


def test_create_parquet_falls_back_to_legacy_npz(tmp_path: Path) -> None:
    latents_dir = tmp_path / "latents"
    write_legacy_npz(latents_dir, KEY_A, META_A)

    CreateParquetProcessor(tmp_path)(max_workers=2)

    table = pq.read_table(tmp_path / "metadata.parquet")
    rows = table.to_pylist()
    assert len(rows) == 1
    assert rows[0]["crop_ltrb"] == META_A["crop_ltrb"]


def test_create_parquet_skips_latent_only_npz_without_meta(tmp_path: Path) -> None:
    latents_dir = tmp_path / "latents"
    write_latent_only_npz(latents_dir, KEY_A)  # no latents_meta row, no embedded meta
    write_legacy_npz(latents_dir, KEY_B, META_B)

    CreateParquetProcessor(tmp_path)(max_workers=2)

    table = pq.read_table(tmp_path / "metadata.parquet")
    rows = table.to_pylist()
    assert [row["key"] for row in rows] == [KEY_B]
