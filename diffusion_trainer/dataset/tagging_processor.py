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

from diffusion_trainer.dataset.utils import get_meta_key_from_path, retrieve_image_paths
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


def worker(args: WorkerArgs) -> None:  # noqa: C901
    """Worker function to process images using Tagger."""
    tagger = Tagger()

    task_queue = args.task_queue
    result_queue = args.result_queue
    metadata = args.metadata
    ds_path_str = args.ds_path_str
    lock = args.lock
    ds_path = Path(ds_path_str)
    while True:
        batch_files = task_queue.get()
        if batch_files is None:
            break  # terminate signal received

        len_before = len(batch_files)

        if args.skip_existing:
            meta_keys = [get_meta_key_from_path(image_path, ds_path=ds_path) for image_path in batch_files]
            batch_files = [f for f, key in zip(batch_files, meta_keys, strict=False) if metadata.get(key, {}).get("tags") is None]

        def filter_valid_files(paths: list[Path], images: list[Image.Image]) -> tuple[list[Path], list[Image.Image]]:
            result_path, result_image = [], []
            for path, image in zip(paths, images, strict=False):
                try:
                    image.load()
                    result_path.append(path)
                    result_image.append(image)
                except Exception:
                    logger.exception(f"Error loading image {path}.")
            return result_path, result_image

        if not batch_files:
            continue
        try:
            images: list[Image.Image] = [Image.open(image_path) for image_path in batch_files]
            (valid_batch_files, valid_images) = filter_valid_files(batch_files, images)
            if not valid_batch_files:
                continue
            results = tagger.tag(valid_images, general_threshold=0.35, character_threshold=0.9)  # type: ignore
        except Exception as e:
            logger.exception(e)
            continue

        for image_path, result in zip(valid_batch_files, results, strict=False):
            with lock:
                try:
                    meta_key = get_meta_key_from_path(image_path, ds_path=Path(ds_path_str))
                    before_tags = metadata[meta_key].get("tags", "").split(", ")
                    more_tags = result.general_tags_string.split(", ")
                    more_tags_not_in_before = [tag for tag in more_tags if tag not in before_tags]
                    final_tags = before_tags + more_tags_not_in_before
                    final_tags_str = ", ".join(final_tags)
                    metadata[meta_key]["tags"] = final_tags_str
                except Exception:
                    logger.exception('Error processing metadata "%s"', image_path)
        result_queue.put(len_before)

    result_queue.put(None)


class TaggingProcessor:
    """Process images using the WD Tagger and merge the tags with existing metadata."""

    def __init__(  # noqa: PLR0913
        self,
        meta_path: str,
        ds_path: str,
        num_workers: int,
        batch_size: int = 16,
        *,
        skip_existing: bool,
        ignore_hidden: bool = True,
        recursive: bool = True,
    ) -> None:
        """Initialize."""
        self.meta_path = Path(meta_path)
        self.ds_path = Path(ds_path)
        self.num_workers = num_workers
        self.skip_existing = skip_existing
        self.batch_size = batch_size

        self.image_paths = list(retrieve_image_paths(self.ds_path, ignore_hidden=ignore_hidden, recursive=recursive))
        logger.info(f"Found {len(self.image_paths)} images in {self.ds_path}")
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

        # Save the updated metadata back to the file
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
