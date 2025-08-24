"""Use the WD Tagger to tag images with SHA256-based directory structure."""

import argparse
import hashlib
import logging
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import torch
from PIL import Image
from wdtagger import Tagger

from diffusion_trainer.dataset.utils import retrieve_image_paths
from diffusion_trainer.shared import get_progress, logger

wdtagger_logger = logging.getLogger("wdtagger")
wdtagger_logger.setLevel(logging.ERROR)


@dataclass
class Args:
    """Arguments for the script (backward compatibility)."""

    image_base_path: str
    meta_base_path: str
    num_workers: int
    skip_existing: bool


@dataclass
class TaggingPayload:
    """Payload for writing tags."""

    save_path: Path
    tags: list[str]


class SimpleTagger:
    """Simple tagger using WD Tagger."""

    def __init__(
        self,
        general_threshold: float = 0.35,
        character_threshold: float = 0.9,
    ) -> None:
        """Initialize the tagger."""
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.tagger = Tagger()

    def tag_image(self, image: Image.Image) -> list[str]:
        """Tag a single image and return combined tags."""
        result = self.tagger.tag(
            image,
            general_threshold=self.general_threshold,
            character_threshold=self.character_threshold,
        )

        # Extract tags
        tags = []
        if hasattr(result, "general_tags_string") and result.general_tags_string:
            general_tags = result.general_tags_string.split(", ")
            tags.extend(general_tags)

        if hasattr(result, "character_tags_string") and result.character_tags_string:
            character_tags = result.character_tags_string.split(", ")
            tags.extend(character_tags)

        # Remove duplicates while preserving order
        seen = set()
        unique_tags = []
        for tag in tags:
            if tag and tag not in seen:
                seen.add(tag)
                unique_tags.append(tag)

        return unique_tags


class TaggingProcessor:
    """Process images using the WD Tagger with SHA256-based directory structure (replaces original)."""

    def __init__(  # noqa: PLR0913
        self,
        img_path: str,
        target_path: str | None = None,
        *,
        num_workers: int = 4,
        batch_size: int = 16,  # Kept for compatibility but not used in SHA256 version
        skip_existing: bool = True,
        ignore_hidden: bool = True,
        recursive: bool = True,
        general_threshold: float = 0.35,
        character_threshold: float = 0.9,
    ) -> None:
        """Initialize (compatible with original interface)."""
        self.img_path = Path(img_path).absolute()
        if not target_path:
            logger.info("Metadata path not set. Using %s as metadata path.", self.img_path / "metadata")
            self.target_path = self.img_path / "metadata"
        else:
            self.target_path = Path(target_path).absolute()

        self.num_workers = num_workers
        self.skip_existing = skip_existing
        self.batch_size = batch_size  # Kept for compatibility

        # Create output directory
        self.target_path.mkdir(parents=True, exist_ok=True)

        # Threading components
        self.reader_threads = []
        self.process_threads = []
        self.writer_threads = []

        self.read_queue = Queue[Path | None](maxsize=num_workers * 2)
        self.process_queue = Queue[tuple[Path, Image.Image] | None](maxsize=num_workers)
        self.write_queue = Queue[TaggingPayload | None](maxsize=num_workers)

        self.progress_counter = 0
        self.progress_lock = threading.Lock()

        # Create multiple tagger instances
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.tagger_list = [
            SimpleTagger(
                general_threshold=general_threshold,
                character_threshold=character_threshold,
            )
            for _ in range(self.gpu_count)
        ]

        # Get image paths
        self.image_paths = list(retrieve_image_paths(self.img_path, ignore_hidden=ignore_hidden, recursive=recursive))
        logger.info("Found %d images in %s", len(self.image_paths), self.img_path)

        self.skip_count = 0
        if skip_existing:
            # Count existing files for progress tracking
            existing_count = 0
            for image_path in self.image_paths:
                tag_path = self.get_tag_save_path(image_path)
                if tag_path.exists():
                    try:
                        with tag_path.open("r", encoding="utf-8") as f:
                            if f.read().strip():  # Has content
                                existing_count += 1
                    except Exception:  # noqa: S110
                        pass
            self.skip_count = existing_count
            logger.info("Skipping %d existing files", self.skip_count)

        self.lock = threading.Lock()

    @staticmethod
    def calculate_sha256(image_path: Path) -> str:
        """Calculate SHA256 hash of image content."""
        hash_sha256 = hashlib.sha256()
        with image_path.open("rb") as f:
            while chunk := f.read(65536):  # 64KB chunks
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_tag_save_path(self, image_path: Path) -> Path:
        """Get tag save path with SHA256-based directory structure."""
        sha256_hash = self.calculate_sha256(image_path)
        dir1 = sha256_hash[:2]
        dir2 = sha256_hash[2:4]
        return self.target_path / dir1 / dir2 / f"{sha256_hash}.txt"

    def read_image(self) -> None:
        """Read images and check for existing files."""
        while image_path := self.read_queue.get():
            if image_path is None:
                break

            try:
                tag_save_path = self.get_tag_save_path(image_path)

                # Check if output file already exists
                if self.skip_existing and tag_save_path.exists():
                    try:
                        # Verify the existing file is valid
                        with tag_save_path.open("r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:  # File exists and has content
                                with self.progress_lock:
                                    self.progress_counter += 1
                                continue
                    except Exception:
                        logger.warning("Corrupted file %s, reprocessing...", tag_save_path)

                # Load and process image
                image = Image.open(image_path).convert("RGB")
                self.process_queue.put((image_path, image))

            except Exception as e:
                logger.error("Error reading %s: %s", image_path, e)
                with self.progress_lock:
                    self.progress_counter += 1

    def process_image(self, tagger: SimpleTagger) -> None:
        """Process images using WD Tagger."""
        while True:
            data = self.process_queue.get()
            if data is None:
                break

            image_path, image = data

            try:
                # Tag the image
                tags = tagger.tag_image(image)

                # Get save path
                tag_save_path = self.get_tag_save_path(image_path)

                # Create payload for writing
                payload = TaggingPayload(
                    save_path=tag_save_path,
                    tags=tags,
                )

                self.write_queue.put(payload)

            except Exception as e:
                logger.error("Error processing %s: %s", image_path, e)
                with self.progress_lock:
                    self.progress_counter += 1

    def write_tags(self) -> None:
        """Write tags to text files."""
        while True:
            payload = self.write_queue.get()
            if payload is None:
                break

            try:
                payload.save_path.parent.mkdir(parents=True, exist_ok=True)
                with payload.save_path.open("w", encoding="utf-8") as f:
                    f.write(", ".join(payload.tags))

                with self.progress_lock:
                    self.progress_counter += 1

            except Exception as e:
                logger.error("Error writing %s: %s", payload.save_path, e)
                with self.progress_lock:
                    self.progress_counter += 1

    def __call__(self) -> None:  # noqa: C901, PLR0912
        """Run the image tagging process."""
        total_images = len(self.image_paths)

        if total_images == 0:
            logger.info("No images found in %s", self.img_path)
            return

        logger.info("Found %d images to process", total_images)

        # Create and start threads
        self.reader_threads = [threading.Thread(target=self.read_image, daemon=True) for _ in range(self.num_workers)]
        self.process_threads = [
            threading.Thread(target=self.process_image, args=(tagger,), daemon=True)
            for tagger in self.tagger_list
        ]
        self.writer_threads = [threading.Thread(target=self.write_tags, daemon=True) for _ in range(self.num_workers)]

        # Start all threads
        for thread in self.reader_threads + self.process_threads + self.writer_threads:
            thread.start()

        # Process with progress tracking
        with get_progress() as progress:
            task = progress.add_task("Tagging...", total=total_images, completed=self.skip_count)

            # Add image paths to queue in batches
            batch_size = 1000
            for i in range(0, len(self.image_paths), batch_size):
                batch = self.image_paths[i:i + batch_size]
                for image_path in batch:
                    self.read_queue.put(image_path)

                # Small delay to avoid overwhelming the queue
                if i > 0:
                    time.sleep(0.1)

            # Monitor progress
            last_count = self.skip_count
            stall_count = 0
            while self.progress_counter < total_images:
                current_count = self.progress_counter
                progress.update(task, completed=current_count)

                # Check for stalled processing
                if current_count == last_count:
                    stall_count += 1
                    if stall_count > 100:  # 10 seconds
                        logger.warning("Processing seems stalled at %d/%d", current_count, total_images)
                        stall_count = 0
                else:
                    stall_count = 0

                last_count = current_count
                time.sleep(0.1)

            # Signal threads to stop
            for _ in range(self.num_workers):
                self.read_queue.put(None)
            for _ in range(len(self.tagger_list)):
                self.process_queue.put(None)
            for _ in range(self.num_workers):
                self.write_queue.put(None)

            # Wait for all threads to complete
            for thread in self.reader_threads + self.process_threads + self.writer_threads:
                thread.join()

            progress.update(task, completed=total_images)

        logger.info("Successfully tagged %d images", total_images)


# Backward compatibility functions
def read_tags(meta_path: Path, key: str) -> list[str]:
    """Read tags from metadata (backward compatibility - not used in SHA256 version)."""
    file = get_tags_file_path(meta_path, key)
    if not file.exists():
        return []
    with file.open("r") as f:
        return f.read().split(", ")


def write_tags(meta_path: Path, key: str, tags: list[str]) -> None:
    """Write tags to metadata (backward compatibility - not used in SHA256 version)."""
    file = get_tags_file_path(meta_path, key)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w") as f:
        f.write(", ".join(tags))


def get_tags_file_path(meta_path: Path, key: str) -> Path:
    """Get tags file path (backward compatibility)."""
    return Path(meta_path / "tags" / f"{key}.txt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tagging processor with SHA256-based directory structure",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--img_path", type=str, required=True, help="Input directory containing images")
    parser.add_argument("--target_path", type=str, required=True, help="Output directory for tag files")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of worker threads")
    parser.add_argument("--general_threshold", type=float, default=0.35, help="General tag threshold (0.0-1.0)")
    parser.add_argument("--character_threshold", type=float, default=0.9, help="Character tag threshold (0.0-1.0)")
    parser.add_argument("--no_skip_existing", action="store_true", help="Do not skip existing files")

    args = parser.parse_args()

    # Validate thresholds
    if not 0.0 <= args.general_threshold <= 1.0:
        logger.error("general_threshold must be between 0.0 and 1.0")
        sys.exit(1)

    if not 0.0 <= args.character_threshold <= 1.0:
        logger.error("character_threshold must be between 0.0 and 1.0")
        sys.exit(1)

    processor = TaggingProcessor(
        img_path=args.img_path,
        target_path=args.target_path,
        num_workers=args.num_workers,
        general_threshold=args.general_threshold,
        character_threshold=args.character_threshold,
        skip_existing=not args.no_skip_existing,
    )

    start_time = time.time()
    processor()
    end_time = time.time()

    elapsed = end_time - start_time
    logger.info("Tagging completed in %.2f seconds", elapsed)
