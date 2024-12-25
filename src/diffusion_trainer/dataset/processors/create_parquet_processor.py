from collections import defaultdict
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
        self.process(max_workers)

    def process(self, max_workers: int) -> None:
        items = defaultdict(list)
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

            with self.lock:
                items["key"].append(key.as_posix())
                items["caption"].append(caption)
                items["tags"].append(tags)
                items["train_resolution"].append(npz.get("train_resolution").tolist())
                items["original_size"].append(npz.get("original_size").tolist())
                items["crop_ltrb"].append(npz.get("crop_ltrb").tolist())
            progress.update(task, advance=1)

        with progress, ThreadPoolExecutor(max_workers=max_workers) as executor:
            for npz_path in npz_path_list:
                executor.submit(process_metadata_files, npz_path)

        table = pa.table(items)
        pq.write_table(table, self.meta_dir / "metadata.parquet")
