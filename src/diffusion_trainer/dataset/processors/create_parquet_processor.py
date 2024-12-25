import threading
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from diffusion_trainer.dataset.utils import retrieve_npz_path
from diffusion_trainer.shared import get_progress


class CreateParquetProcessor:
    def __init__(self, meta_dir: str | PathLike) -> None:
        self.meta_dir = Path(meta_dir)
        self.latents_dir = (self.meta_dir / "latents").resolve()
        self.tags_dir = (self.meta_dir / "tags").resolve()
        self.captions_dir = (self.meta_dir / "captions").resolve()

    lock = threading.Lock()

    def __call__(self, max_workers: int = 8) -> None:
        items = []
        progress = get_progress()
        npz_path_list = list(retrieve_npz_path(Path(self.meta_dir)))
        task = progress.add_task("Processing metadata files", total=len(npz_path_list))

        def process_metadata_files(
            npz_path: Path,
        ) -> None:
            key = npz_path.relative_to(self.latents_dir).with_suffix("")
            npz = np.load(npz_path)
            tag_file = self.meta_dir / "tags" / f"{key}.txt"
            caption_file = self.meta_dir / "caption" / f"{key}.txt"

            caption = caption_file.read_text() if caption_file.exists() else ""
            tags = tag_file.read_text().split(",") if tag_file.exists() else []

            item = {
                "key": key.as_posix(),
                "caption": caption,
                "tags": tags,
                "train_resolution": npz.get("train_resolution").tolist(),
                "original_size": npz.get("original_size").tolist(),
                "crop_ltrb": npz.get("crop_ltrb").tolist(),
            }
            with self.lock:
                items.append(item)
            progress.update(task, advance=1)
        with progress, ThreadPoolExecutor(max_workers=max_workers) as executor:
            for npz_path in npz_path_list:
                executor.submit(process_metadata_files, npz_path)

        # convert items to dictionary
        items_dict = {}
        for item in items:
            for key, value in item.items():
                items_dict.setdefault(key, []).append(value)

        # 将 items 转换为 parquet 文件
        table = pa.table(items_dict)
        pq.write_table(table, self.meta_dir / "metadata.parquet")
