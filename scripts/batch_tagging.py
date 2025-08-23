"""Batch tagging script for processing images and generating tags using WD Tagger."""

import argparse
import hashlib
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Queue

import torch
from PIL import Image
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn, TimeRemainingColumn
from wdtagger import Tagger

# Import from the project
from diffusion_trainer.dataset.utils import retrieve_image_paths
from diffusion_trainer.shared import logger

# Global console for consistent rich output
console = Console()


@dataclass
class TaggingTask:
    """Task for tagging pipeline."""

    image_path: Path
    sha256_hash: str | None = None
    output_path: Path | None = None


@dataclass
class TaggingPayload:
    """Payload for writing tags."""

    save_path: Path
    tags: list[str]


class BatchTagger:
    """Batch tagger for generating image tags using WD Tagger."""

    def __init__(  # noqa: PLR0913
        self,
        output_dir: str,
        *,
        num_readers: int = 4,
        num_writers: int = 4,
        general_threshold: float = 0.35,
        character_threshold: float = 0.9,
        skip_existing: bool = True,
    ) -> None:
        """Initialize the batch tagger."""
        self.output_dir = Path(output_dir)
        self.num_readers = num_readers
        self.num_writers = num_writers
        self.general_threshold = general_threshold
        self.character_threshold = character_threshold
        self.skip_existing = skip_existing

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Threading components
        self.read_queue = Queue[TaggingTask | None](maxsize=num_readers * 2)
        self.process_queue = Queue[tuple[TaggingTask, Image.Image] | None](maxsize=num_readers)
        self.write_queue = Queue[TaggingPayload | None](maxsize=num_writers)

        self.progress_counter = 0
        self.progress_lock = threading.Lock()
        self.progress = None
        self.progress_task_id = None

        # Create multiple tagger instances for GPU processing
        self.gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        self.taggers = []
        for _ in range(self.gpu_count):
            tagger = Tagger()
            self.taggers.append(tagger)

    @staticmethod
    def calculate_sha256(image_path: Path) -> str:
        """Calculate SHA256 hash of image content."""
        hash_sha256 = hashlib.sha256()
        with image_path.open("rb") as f:
            # Use larger chunk size for better performance
            while chunk := f.read(65536):  # 64KB chunks
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def get_output_path(self, sha256_hash: str) -> Path:
        """Get output path with two-level directory structure."""
        dir1 = sha256_hash[:2]
        dir2 = sha256_hash[2:4]
        return self.output_dir / dir1 / dir2 / f"{sha256_hash}.txt"

    @staticmethod
    def load_image(image_path: Path) -> Image.Image:
        """Load image from path."""
        return Image.open(image_path).convert("RGB")

    @staticmethod
    def save_tags(payload: TaggingPayload) -> None:
        """Save the tags as a txt file."""
        payload.save_path.parent.mkdir(parents=True, exist_ok=True)
        with payload.save_path.open("w", encoding="utf-8") as f:
            f.write(", ".join(payload.tags))

    def read_image_worker(self) -> None:
        """Reader thread worker function."""
        while True:
            task = self.read_queue.get()
            if task is None:
                break

            try:
                # Calculate SHA256 and output path lazily
                if task.sha256_hash is None:
                    task.sha256_hash = self.calculate_sha256(task.image_path)
                if task.output_path is None:
                    task.output_path = self.get_output_path(task.sha256_hash)

                # Check if output file already exists
                if self.skip_existing and task.output_path.exists():
                    try:
                        # Verify the existing file is valid
                        with task.output_path.open("r", encoding="utf-8") as f:
                            content = f.read().strip()
                            if content:  # File exists and has content
                                self._update_progress()
                                continue
                    except Exception:
                        # File is corrupted, reprocess
                        logger.warning("Corrupted file %s, reprocessing...", task.output_path)

                # Load image
                image = self.load_image(task.image_path)
                self.process_queue.put((task, image))

            except Exception as e:
                logger.error("Error reading %s: %s", task.image_path, e)
                self._update_progress()

    def process_image_worker(self, tagger: Tagger) -> None:
        """Processor thread worker function."""
        while True:
            data = self.process_queue.get()
            if data is None:
                break

            task, image = data

            try:
                # Tag the image
                result = tagger.tag(
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

                # Create payload for writing
                if task.output_path is None:
                    msg = "output_path should be set by reader"
                    raise ValueError(msg)  # noqa: TRY301

                payload = TaggingPayload(
                    save_path=task.output_path,
                    tags=unique_tags,
                )

                self.write_queue.put(payload)

            except Exception as e:
                logger.error("Error processing %s: %s", task.image_path, e)
                self._update_progress()

    def write_tags_worker(self) -> None:
        """Writer thread worker function."""
        while True:
            payload = self.write_queue.get()
            if payload is None:
                break

            try:
                self.save_tags(payload)
                self._update_progress()
            except Exception as e:
                logger.error("Error writing %s: %s", payload.save_path, e)
                self._update_progress()

    def setup_threads(self) -> tuple[list[threading.Thread], list[threading.Thread], list[threading.Thread]]:
        """Setup all worker threads."""
        reader_threads = [threading.Thread(target=self.read_image_worker, daemon=True) for _ in range(self.num_readers)]

        processor_threads = [threading.Thread(target=self.process_image_worker, args=(tagger,), daemon=True) for tagger in self.taggers]

        writer_threads = [threading.Thread(target=self.write_tags_worker, daemon=True) for _ in range(self.num_writers)]

        return reader_threads, processor_threads, writer_threads

    def start_threads(self, reader_threads: list[threading.Thread], processor_threads: list[threading.Thread], writer_threads: list[threading.Thread]) -> None:
        """Start all worker threads."""
        for thread in reader_threads + processor_threads + writer_threads:
            thread.start()

    def join_threads(self, reader_threads: list[threading.Thread], processor_threads: list[threading.Thread], writer_threads: list[threading.Thread]) -> None:
        """Join all worker threads."""
        for thread in reader_threads + processor_threads + writer_threads:
            thread.join()

    def _update_progress(self) -> None:
        """Update progress counter and progress bar."""
        with self.progress_lock:
            self.progress_counter += 1
            if self.progress is not None and self.progress_task_id is not None:
                self.progress.update(self.progress_task_id, completed=self.progress_counter)

    def process_directory(self, input_dir: str) -> None:  # noqa: C901
        """Process all images in the input directory."""
        input_path = Path(input_dir)

        # Get all image paths
        console.print("Scanning for images...", style="blue")
        image_paths = list(retrieve_image_paths(input_path, recursive=True))
        total_images = len(image_paths)

        if total_images == 0:
            console.print("No images found in the input directory.", style="red")
            return

        console.print(f"Found {total_images} images to process.", style="green")

        # Setup and start threads
        reader_threads, processor_threads, writer_threads = self.setup_threads()
        self.start_threads(reader_threads, processor_threads, writer_threads)

        # Process with progress tracking
        with Progress(
            SpinnerColumn(),
            "[progress.description]{task.description}",
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            task_id = progress.add_task("Tagging images...", total=total_images)

            # Set progress references for worker threads
            self.progress = progress
            self.progress_task_id = task_id

            # Add image paths to reader queue in batches
            console.print("Starting image tagging...", style="blue")
            batch_size = 1000
            for i in range(0, len(image_paths), batch_size):
                batch = image_paths[i : i + batch_size]
                for image_path in batch:
                    # Create task with just the image path, SHA256 calculated lazily
                    task = TaggingTask(image_path=image_path)
                    self.read_queue.put(task)

                # Give some time for processing to avoid overwhelming the queue
                if i > 0:
                    time.sleep(0.1)

            # Monitor progress and check for stalls
            last_count = 0
            stall_count = 0
            while self.progress_counter < total_images:
                current_count = self.progress_counter

                # Check for stalled processing
                if current_count == last_count:
                    stall_count += 1
                    if stall_count > 100:  # 10 seconds of no progress
                        console.print(f"⚠️  Processing seems stalled at {current_count}/{total_images}", style="yellow")
                        stall_count = 0
                else:
                    stall_count = 0

                last_count = current_count
                time.sleep(0.1)

            # Signal threads to stop
            for _ in range(self.num_readers):
                self.read_queue.put(None)
            for _ in range(len(self.taggers)):
                self.process_queue.put(None)
            for _ in range(self.num_writers):
                self.write_queue.put(None)

            # Wait for all threads to complete
            self.join_threads(reader_threads, processor_threads, writer_threads)

            # Clean up progress references
            self.progress = None
            self.progress_task_id = None

        console.print(f"✅ Successfully tagged {total_images} images!", style="bold green")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Batch tagging for generating image tags using WD Tagger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing images to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for generated tag files",
    )

    # Optional arguments
    parser.add_argument(
        "--num_readers",
        type=int,
        default=4,
        help="Number of reader threads for loading images",
    )
    parser.add_argument(
        "--num_writers",
        type=int,
        default=4,
        help="Number of writer threads for saving tags",
    )
    parser.add_argument(
        "--general_threshold",
        type=float,
        default=0.35,
        help="Threshold for general tags (0.0-1.0)",
    )
    parser.add_argument(
        "--character_threshold",
        type=float,
        default=0.9,
        help="Threshold for character tags (0.0-1.0)",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_true",
        help="Do not skip existing files (reprocess all images)",
    )

    args = parser.parse_args()

    # Validate arguments
    input_path = Path(args.input_dir)
    if not input_path.exists():
        logger.error("Input directory '%s' does not exist.", args.input_dir)
        return

    if not input_path.is_dir():
        logger.error("Input path '%s' is not a directory.", args.input_dir)
        return

    # Validate thresholds
    if not 0.0 <= args.general_threshold <= 1.0:
        logger.error("general_threshold must be between 0.0 and 1.0")
        return

    if not 0.0 <= args.character_threshold <= 1.0:
        logger.error("character_threshold must be between 0.0 and 1.0")
        return

    # Print configuration using both console and logger
    console.print("\n🏷️  [bold blue]Batch Image Tagging Configuration[/bold blue]")
    console.print(f"  📁 Input Directory: [green]{args.input_dir}[/green]")
    console.print(f"  📂 Output Directory: [green]{args.output_dir}[/green]")
    console.print(f"  🎯 General Threshold: [cyan]{args.general_threshold}[/cyan]")
    console.print(f"  👤 Character Threshold: [cyan]{args.character_threshold}[/cyan]")
    console.print(f"  👥 Reader Threads: {args.num_readers}")
    console.print(f"  ✍️  Writer Threads: {args.num_writers}")
    console.print(f"  ⏭️  Skip Existing: {not args.no_skip_existing}")

    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        console.print(f"  🎮 GPU Count: [bold]{gpu_count}[/bold]")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            console.print(f"    • GPU {i}: [dim]{gpu_name}[/dim]")
    else:
        console.print("  💻 Using CPU (CUDA not available)")

    console.print()

    # Create tagger and run
    try:
        tagger = BatchTagger(
            output_dir=args.output_dir,
            num_readers=args.num_readers,
            num_writers=args.num_writers,
            general_threshold=args.general_threshold,
            character_threshold=args.character_threshold,
            skip_existing=not args.no_skip_existing,
        )

        start_time = time.time()
        tagger.process_directory(args.input_dir)
        end_time = time.time()

        elapsed = end_time - start_time
        logger.info("🎉 Tagging completed in %.2f seconds!", elapsed)

    except Exception as e:
        logger.error("Error during tagging: %s", e)
        logger.exception("Tagging failed")


if __name__ == "__main__":
    main()
