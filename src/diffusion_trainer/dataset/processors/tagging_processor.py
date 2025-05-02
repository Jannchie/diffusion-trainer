"""Use the WD Tagger to tag images and merge the tags with existing metadata."""

import argparse
import logging
import threading
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from queue import Queue

import torch
from PIL import Image
from wdtagger import Tagger

from diffusion_trainer.dataset.utils import get_meta_key_from_path, retrieve_image_paths, retrieve_text_path
from diffusion_trainer.shared import get_progress, logger

wdtagger_logger = logging.getLogger("wdtagger")
wdtagger_logger.setLevel(logging.ERROR)


@dataclass
class Args:
    """Arguments for the script."""

    image_base_path: str
    meta_base_path: str
    num_workers: int
    skip_existing: bool


@dataclass
class WorkerArgs:
    """Arguments for the worker function."""

    image_paths: list[Path]
    image_base_path: Path
    target_path: Path
    task_queue: Queue
    result_queue: Queue
    i: int
    lock: threading.Lock


def read_tags(meta_path: Path, key: str) -> list[str]:
    """Read tags from metadata."""
    file = get_tags_file_path(meta_path, key)
    if not file.exists():
        return []
    with file.open("r") as f:
        return f.read().split(", ")


def write_tags(meta_path: Path, key: str, tags: list[str]) -> None:
    """Write tags to metadata."""
    file = get_tags_file_path(meta_path, key)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        f.write(", ".join(tags))


def get_tags_file_path(meta_path: Path, key: str) -> Path:
    return Path(meta_path / "tags" / f"{key}.txt")


def worker(args: WorkerArgs) -> None:  # noqa: C901
    """Worker function to process images using Tagger."""
    tagger = Tagger()

    task_queue = args.task_queue
    result_queue = args.result_queue
    lock = args.lock
    while True:
        batch_files = task_queue.get()
        if batch_files is None:
            break  # terminate signal received

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
                    meta_key = get_meta_key_from_path(image_path, base_path=args.image_base_path)
                    before_tags = read_tags(args.target_path, meta_key)
                    more_tags = result.general_tags_string.split(", ")
                    more_tags_not_in_before = [tag for tag in more_tags if tag not in before_tags]
                    final_tags = before_tags + more_tags_not_in_before
                    write_tags(args.target_path, meta_key, final_tags)
                except Exception:
                    logger.exception('Error processing metadata "%s"', image_path)
        result_queue.put(len(batch_files))

    result_queue.put(None)


class TaggingProcessor:
    """Process images using the WD Tagger and merge the tags with existing metadata."""

    def __init__(  # noqa: PLR0913
        self,
        img_path: str | PathLike,
        target_path: str | PathLike | None = None,
        *,
        num_workers: int = 4,
        batch_size: int = 16,
        skip_existing: bool = True,
        ignore_hidden: bool = True,
        recursive: bool = True,
    ) -> None:
        """Initialize."""
        self.img_path = Path(img_path).absolute()
        if not target_path:
            logger.info("Metadata path not set. Using %s as metadata path.", self.img_path / "metadata")
            self.target_path = self.img_path / "metadata"
        else:
            self.target_path = Path(target_path).absolute()

        self.num_workers = num_workers
        self.skip_existing = skip_existing
        self.batch_size = batch_size

        self.image_paths = list(retrieve_image_paths(self.img_path, ignore_hidden=ignore_hidden, recursive=recursive))
        logger.info(f"Found {len(self.image_paths)} images in {self.img_path}")
        self.skip_count = 0
        if skip_existing:
            tags_base_path = self.target_path / "tags"
            already_exists = retrieve_text_path(tags_base_path)
            already_exists_relative_path = {p.relative_to(tags_base_path) for p in already_exists}
            before_count = len(self.image_paths)
            self.image_paths = [p for p in self.image_paths if p.relative_to(self.img_path).with_suffix(".txt") not in already_exists_relative_path]
            after_count = len(self.image_paths)
            self.skip_count = before_count - after_count
            logger.info("Skipping %s - %s = %s existing files", f"{before_count:,}", f"{after_count:,}", f"{self.skip_count:,}")

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
                        target_path=self.target_path,
                        image_paths=self.image_paths,
                        task_queue=self.task_queue,
                        result_queue=self.result_queue,
                        i=i % self.gpu_count,
                        image_base_path=self.img_path,
                        lock=self.lock,
                    ),
                ),
            )
            for i in range(self.num_workers)
        ]

    def _process_results(self, workers: list[threading.Thread]) -> None:
        with get_progress() as progress:
            task = progress.add_task("Tagging...", total=len(self.image_paths), completed=self.skip_count)
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
