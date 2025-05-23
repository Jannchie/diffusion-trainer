import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from os import PathLike
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from rich import get_console
from rich.console import Console

from diffusion_trainer.dataset.utils import retrieve_npz_path
from diffusion_trainer.shared import get_progress

console = get_console()


class CreateParquetProcessor:
    def __init__(self, target_dir: str | PathLike, console: Console = console) -> None:
        self.target_dir = Path(target_dir)
        self.latents_dir = (self.target_dir / "latents").resolve()
        self.tags_dir = (self.target_dir / "tags").resolve()
        self.captions_dir = (self.target_dir / "captions").resolve()
        self.console = console

    lock = threading.Lock()

    def __call__(self, max_workers: int = 8) -> None:
        self.process(max_workers)

    def process(self, max_workers: int) -> None:
        items = defaultdict(list)
        progress = get_progress()
        npz_path_list = list(retrieve_npz_path(self.target_dir))
        self.console.log(self.target_dir)
        task = progress.add_task("Processing metadata files", total=len(npz_path_list))

        def process_metadata_files(
            npz_path: Path,
        ) -> None:
            key = npz_path.relative_to(self.latents_dir).with_suffix("")
            npz = np.load(npz_path)
            tag_file = self.target_dir / "tags" / f"{key}.txt"
            caption_file = self.target_dir / "captions" / f"{key}.txt"

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
        pq.write_table(table, self.target_dir / "metadata.parquet")
