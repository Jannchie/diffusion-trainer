"""Tests for dataset export/import (bucket-grouped tar shards)."""

import tarfile
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from diffusion_trainer.dataset.dataset import DiffusionDataset
from diffusion_trainer.dataset.sharing import export_dataset, import_dataset
from diffusion_trainer.dataset.utils import load_latents_meta, sharded_path

KEY_A = "a" * 64
KEY_B = "b" * 64
KEY_C = "c" * 64

ROWS = {
    KEY_A: {"tags": ["1girl", "solo"], "train_resolution": [1024, 1024], "original_size": [2048, 2048], "crop_ltrb": [0, 0, 1024, 1024]},
    KEY_B: {"tags": [], "train_resolution": [1024, 1024], "original_size": [1500, 1500], "crop_ltrb": [0, 0, 1024, 1024]},
    KEY_C: {"tags": ["scenery"], "train_resolution": [832, 1216], "original_size": [1000, 1500], "crop_ltrb": [0, 12, 832, 1228]},
}


def make_dataset(dataset_dir: Path) -> None:
    """Build a minimal prepared dataset: latent-only npz tree + metadata.parquet."""
    rng = np.random.default_rng(0)
    for key in ROWS:
        npz_path = sharded_path(dataset_dir / "latents", key, "npz")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz_path, latents=rng.standard_normal((4, 8, 8)).astype(np.float32))
    keys = list(ROWS)
    table = pa.table({
        "key": keys,
        "tags": [ROWS[key]["tags"] for key in keys],
        "train_resolution": [ROWS[key]["train_resolution"] for key in keys],
        "original_size": [ROWS[key]["original_size"] for key in keys],
        "crop_ltrb": [ROWS[key]["crop_ltrb"] for key in keys],
    })
    pq.write_table(table, dataset_dir / "metadata.parquet")


def test_export_creates_bucket_grouped_shards(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")
    stats = export_dataset(tmp_path / "ds", tmp_path / "out")

    assert stats.exported == 3
    assert sorted(p.name for p in (tmp_path / "out" / "shards").iterdir()) == ["1024x1024-00000.tar", "832x1216-00000.tar"]
    with tarfile.open(tmp_path / "out" / "shards" / "1024x1024-00000.tar") as tar:
        assert sorted(m.name for m in tar) == [f"{k[:2]}/{k[2:4]}/{k}.npz" for k in (KEY_A, KEY_B)]

    table = pq.read_table(tmp_path / "out" / "metadata.parquet")
    rows = {row["key"]: row for row in table.to_pylist()}
    assert rows[KEY_C]["shard"] == "832x1216-00000.tar"
    assert len(rows[KEY_A]["npz_sha256"]) == 64
    assert (tmp_path / "out" / "README.md").exists()


def test_export_rolls_shards_by_size(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")
    stats = export_dataset(tmp_path / "ds", tmp_path / "out", shard_size_bytes=1)

    # 1-byte budget: every member lands in its own shard.
    assert stats.shards == 3
    names = sorted(p.name for p in (tmp_path / "out" / "shards").iterdir())
    assert names == ["1024x1024-00000.tar", "1024x1024-00001.tar", "832x1216-00000.tar"]


def test_export_skips_rows_without_npz(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")
    sharded_path(tmp_path / "ds" / "latents", KEY_B, "npz").unlink()

    stats = export_dataset(tmp_path / "ds", tmp_path / "out")

    assert stats.exported == 2
    assert stats.skipped == 1
    keys = pq.read_table(tmp_path / "out" / "metadata.parquet").column("key").to_pylist()
    assert KEY_B not in keys


def test_import_roundtrip(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")
    export_dataset(tmp_path / "ds", tmp_path / "out")
    stats = import_dataset(tmp_path / "out", tmp_path / "imported")

    assert stats.imported == 3
    assert stats.failed == 0
    for key in ROWS:
        src = sharded_path(tmp_path / "ds" / "latents", key, "npz")
        dst = sharded_path(tmp_path / "imported" / "latents", key, "npz")
        assert dst.read_bytes() == src.read_bytes()
    # Sidecar tags and latents_meta are materialized so the prepare pipeline stays usable.
    assert sharded_path(tmp_path / "imported" / "tags", KEY_A, "txt").read_text(encoding="utf-8") == "1girl, solo"
    assert load_latents_meta(tmp_path / "imported" / "latents")[KEY_C]["crop_ltrb"] == ROWS[KEY_C]["crop_ltrb"]

    dataset = DiffusionDataset.from_parquet(tmp_path / "imported" / "metadata.parquet")
    assert len(dataset) == 3
    assert dataset[0]["img_latents"].shape == (4, 8, 8)


def test_import_is_idempotent(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")
    export_dataset(tmp_path / "ds", tmp_path / "out")
    import_dataset(tmp_path / "out", tmp_path / "imported")
    stats = import_dataset(tmp_path / "out", tmp_path / "imported")

    assert stats.imported == 0
    assert stats.skipped_existing == 3


def test_import_rejects_traversal_member(tmp_path: Path) -> None:
    source = tmp_path / "out"
    (source / "shards").mkdir(parents=True)
    data = b"malicious"
    with tarfile.open(source / "shards" / "1024x1024-00000.tar", "w") as tar:
        info = tarfile.TarInfo("../evil.npz")
        info.size = len(data)
        tar.addfile(info, BytesIO(data))
    table = pa.table({
        "key": [KEY_A],
        "tags": [["x"]],
        "train_resolution": [[1024, 1024]],
        "original_size": [[2048, 2048]],
        "crop_ltrb": [[0, 0, 1024, 1024]],
        "shard": ["1024x1024-00000.tar"],
        "npz_sha256": ["0" * 64],
    })
    pq.write_table(table, source / "metadata.parquet")

    stats = import_dataset(source, tmp_path / "imported")

    assert stats.imported == 0
    assert stats.failed == 1
    assert not (tmp_path / "evil.npz").exists()
    assert not list((tmp_path / "imported" / "latents").rglob("*.npz")) if (tmp_path / "imported" / "latents").exists() else True


def test_import_skips_checksum_mismatch(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")
    export_dataset(tmp_path / "ds", tmp_path / "out")
    # Tamper the recorded checksums so every extraction fails verification.
    table = pq.read_table(tmp_path / "out" / "metadata.parquet")
    table = table.set_column(table.column_names.index("npz_sha256"), "npz_sha256", pa.array(["0" * 64] * len(table)))
    pq.write_table(table, tmp_path / "out" / "metadata.parquet")

    stats = import_dataset(tmp_path / "out", tmp_path / "imported")

    assert stats.imported == 0
    assert stats.failed == 3


def test_import_requires_shard_column(tmp_path: Path) -> None:
    make_dataset(tmp_path / "ds")  # plain prepared dataset, not an export
    with pytest.raises(ValueError, match="shard"):
        import_dataset(tmp_path / "ds", tmp_path / "imported")
