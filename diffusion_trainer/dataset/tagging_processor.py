"""Use the WD Tagger to tag images and merge the tags with existing metadata."""

import argparse
import json
import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import torch
from PIL import Image
from wdtagger import Tagger

from diffusion_trainer.dataset.utils import get_meta_key_from_path, glob_images_pathlib
from diffusion_trainer.shared import get_progress, logger


@dataclass
class Args:
    """Arguments for the script."""

    meta_path: str
    ds_path: str
    num_workers: int
    skip_existing: bool


@dataclass
class WorkerArgs:
    """Arguments for the worker function."""

    task_queue: Queue
    result_queue: Queue
    i: int
    metadata: dict[str, dict[str, str]]
    ds_path_str: str
    lock: threading.Lock
    skip_existing: bool


def worker(args: WorkerArgs) -> None:
    """Worker function to process images using Tagger."""
    tagger = Tagger(
        model_repo="SmilingWolf/wd-vit-large-tagger-v3",
        slient=True,
    )

    task_queue = args.task_queue
    result_queue = args.result_queue
    metadata = args.metadata
    ds_path_str = args.ds_path_str
    lock = args.lock

    while True:
        batch_files = task_queue.get()
        if batch_files is None:
            break  # terminate signal received

        len_before = len(batch_files)
        batch_files = [f for f in batch_files if not Path(f).with_suffix(".txt").exists()]
        if args.skip_existing:
            meta_keys = [get_meta_key_from_path(image_path, ds_path=Path(ds_path_str)) for image_path in batch_files]
            batch_files = [f for f, key in zip(batch_files, meta_keys, strict=False) if metadata.get(key) is None]

        results = []
        if batch_files:
            try:
                images: list[Image.Image] = [Image.open(image_path) for image_path in batch_files]
                results = tagger.tag(images, general_threshold=0.35, character_threshold=0.9)  # type: ignore
            except Exception as e:
                logger.exception(e)
                continue

            # If result is not iterable, make it iterable
            if not isinstance(results, list):
                results = [results]

            for image_path, result in zip(batch_files, results, strict=False):
                with lock:
                    meta_key = get_meta_key_from_path(image_path, ds_path=Path(ds_path_str))
                    before_tags = metadata[meta_key].get("tags", "").split(", ")
                    more_tags = result.general_tags_string.split(", ")
                    more_tags_not_in_before = [tag for tag in more_tags if tag not in before_tags]
                    final_tags = before_tags + more_tags_not_in_before
                    final_tags_str = ", ".join(final_tags)
                    metadata[meta_key]["tags"] = final_tags_str

        result_queue.put(len_before)

    result_queue.put(None)


class TaggingProcessor:
    """Process images using the WD Tagger and merge the tags with existing metadata."""

    def __init__(self, meta_path: str, ds_path: str, num_workers: int, *, skip_existing: bool) -> None:
        """Initialize."""
        self.meta_path = Path(meta_path)
        self.ds_path = Path(ds_path)
        self.num_workers = num_workers
        self.skip_existing = skip_existing
        self.batch_size = 4

        self.image_paths = glob_images_pathlib(self.ds_path, recursive=True)
        self.metadata = json.load(self.meta_path.open(encoding="utf-8"))

        self.task_queue = Queue()
        self.result_queue = Queue()
        self.lock = threading.Lock()
        self.gpu_count = torch.cuda.device_count()

    def __call__(self) -> None:
        """Run the image tagging process."""
        self._populate_task_queue()
        workers = self._create_workers()

        for w in workers:
            w.start()

        # Send termination signals to workers
        for _ in workers:
            self.task_queue.put(None)

        self._process_results(workers)

        # Optionally save the updated metadata back to the file
        with self.meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(self.metadata, meta_file, indent=2)

    def _populate_task_queue(self) -> None:
        for i in range(0, len(self.image_paths), self.batch_size):
            batch_files = self.image_paths[i : i + self.batch_size]
            self.task_queue.put(batch_files)

    def _create_workers(self) -> list[threading.Thread]:
        return [
            threading.Thread(
                target=worker,
                args=(
                    WorkerArgs(
                        task_queue=self.task_queue,
                        result_queue=self.result_queue,
                        i=i % self.gpu_count,
                        metadata=self.metadata,
                        ds_path_str=str(self.ds_path),
                        lock=self.lock,
                        skip_existing=self.skip_existing,
                    ),
                ),
            )
            for i in range(self.num_workers)
        ]

    def _process_results(self, workers: list[threading.Thread]) -> None:
        with get_progress() as progress:
            task = progress.add_task("Tagging...", total=len(self.image_paths))

            processed_count = 0
            done_workers = 0

            while done_workers < self.num_workers:
                item = self.result_queue.get()
                if item is None:
                    done_workers += 1
                else:
                    processed_count += item
                    progress.update(task, completed=processed_count)

            for w in workers:
                w.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", type=str, required=True)
    parser.add_argument("--ds_path", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--skip_existing", action="store_true")
    args = Args(**vars(parser.parse_args()))
    processor = TaggingProcessor(**vars(args))
    processor()
