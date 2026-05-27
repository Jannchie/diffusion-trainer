"""Use the WD Tagger to tag images with SHA256-based directory structure."""

import argparse
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

import torch
from PIL import Image
from wdtagger import Tagger

from diffusion_trainer.dataset.processors.base import ThreadedPipelineProcessor
from diffusion_trainer.dataset.utils import calculate_file_sha256, retrieve_image_paths, sharded_path
from diffusion_trainer.shared import get_progress, logger

wdtagger_logger = logging.getLogger("wdtagger")
wdtagger_logger.setLevel(logging.ERROR)


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


class TaggingProcessor(ThreadedPipelineProcessor[Path, tuple[Path, Image.Image], "TaggingPayload"]):
    """Process images into SHA256-based tag files."""

    def __init__(  # noqa: PLR0913
        self,
        img_path: str,
        target_path: str | None = None,
        *,
        num_workers: int = 4,
        skip_existing: bool = True,
        ignore_hidden: bool = True,
        recursive: bool = True,
        general_threshold: float = 0.35,
        character_threshold: float = 0.9,
        tag_source: Literal["wd_tagger", "sidecar_txt"] = "wd_tagger",
    ) -> None:
        """Initialize (compatible with original interface)."""
        self.img_path = Path(img_path).absolute()
        if not target_path:
            logger.info("Metadata path not set. Using %s as metadata path.", self.img_path / "metadata")
            self.target_path = self.img_path / "metadata"
        else:
            self.target_path = Path(target_path).absolute()

        self.skip_existing = skip_existing
        self.tag_source = tag_source
        self.target_path.mkdir(parents=True, exist_ok=True)

        gpu_count = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if tag_source == "wd_tagger":
            self.tagger_list = [
                SimpleTagger(general_threshold=general_threshold, character_threshold=character_threshold)
                for _ in range(gpu_count)
            ]
        else:
            self.tagger_list = []

        self.image_paths = list(retrieve_image_paths(self.img_path, ignore_hidden=ignore_hidden, recursive=recursive))
        logger.info("Found %d images in %s", len(self.image_paths), self.img_path)
        logger.info("Tag source: %s", self.tag_source)

        self.skip_count = self._count_existing_tags() if skip_existing else 0
        if skip_existing:
            logger.info("Skipping %d existing files", self.skip_count)

        super().__init__(
            num_reader=num_workers,
            num_writer=num_workers,
            num_process_workers=len(self.tagger_list),
            description="Tagging...",
            poll_interval=0.1,
            stall_ticks=100,
            initial_completed=self.skip_count,
            enqueue_batch_size=1000,
            enqueue_batch_pause=0.1,
        )

    def _count_existing_tags(self) -> int:
        """Count images that already have non-empty tag files (for progress display)."""
        existing_count = 0
        for image_path in self.image_paths:
            tag_path = self.get_tag_save_path(image_path)
            if tag_path.exists():
                try:
                    if tag_path.read_text(encoding="utf-8").strip():
                        existing_count += 1
                except Exception:  # noqa: S110
                    pass
        return existing_count

    @staticmethod
    def calculate_sha256(image_path: Path) -> str:
        """Calculate SHA256 hash of image content."""
        return calculate_file_sha256(image_path)

    def get_tag_save_path(self, image_path: Path) -> Path:
        """Get tag save path with SHA256-based directory structure."""
        return sharded_path(self.target_path, self.calculate_sha256(image_path), "txt")

    @staticmethod
    def parse_tags_text(tags_text: str) -> list[str]:
        """Parse comma-separated tags and preserve order while removing duplicates."""
        seen: set[str] = set()
        normalized_tags = []
        for raw_tag in tags_text.split(","):
            tag = raw_tag.strip()
            if tag and tag not in seen:
                seen.add(tag)
                normalized_tags.append(tag)
        return normalized_tags

    @staticmethod
    def get_sidecar_tag_path(image_path: Path) -> Path:
        """Return the same-name sidecar txt path for an image."""
        return image_path.with_suffix(".txt")

    def load_sidecar_tags(self, image_path: Path) -> list[str]:
        """Load tags from a same-name sidecar txt file."""
        sidecar_tag_path = self.get_sidecar_tag_path(image_path)
        if not sidecar_tag_path.exists():
            msg = f"Missing sidecar tag file for image: {image_path}"
            raise FileNotFoundError(msg)
        return self.parse_tags_text(sidecar_tag_path.read_text(encoding="utf-8"))

    # ----- ThreadedPipelineProcessor stages -----

    def get_items(self) -> list[Path]:
        return self.image_paths

    def make_process_worker(self, index: int) -> object:
        return self.tagger_list[index]

    def read_item(self, item: Path) -> tuple[Path, Image.Image] | None:
        tag_save_path = self.get_tag_save_path(item)
        if self.skip_existing and tag_save_path.exists():
            try:
                if tag_save_path.read_text(encoding="utf-8").strip():
                    return None  # already tagged
            except Exception:
                logger.warning("Corrupted file %s, reprocessing...", tag_save_path)
        image = Image.open(item).convert("RGB")
        return (item, image)

    def process_item(self, worker: object, loaded: tuple[Path, Image.Image]) -> "TaggingPayload":
        image_path, image = loaded
        tagger = cast("SimpleTagger", worker)
        tags = tagger.tag_image(image)
        return TaggingPayload(save_path=self.get_tag_save_path(image_path), tags=tags)

    def write_item(self, payload: "TaggingPayload") -> None:
        payload.save_path.parent.mkdir(parents=True, exist_ok=True)
        with payload.save_path.open("w", encoding="utf-8") as f:
            f.write(", ".join(payload.tags))

    def import_sidecar_tags(self) -> None:
        """Import existing same-name txt files into the SHA256-based tags directory."""
        total_images = len(self.image_paths)
        if total_images == 0:
            logger.info("No images found in %s", self.img_path)
            return

        logger.info("Importing same-name sidecar txt tags for %d images", total_images)
        with get_progress() as progress:
            task = progress.add_task("Importing tags...", total=total_images, completed=self.skip_count)
            completed = self.skip_count
            for image_path in self.image_paths:
                tag_save_path = self.get_tag_save_path(image_path)
                if self.skip_existing and tag_save_path.exists():
                    try:
                        if tag_save_path.read_text(encoding="utf-8").strip():
                            continue
                    except Exception:
                        logger.warning("Corrupted file %s, reprocessing...", tag_save_path)

                tags = self.load_sidecar_tags(image_path)
                tag_save_path.parent.mkdir(parents=True, exist_ok=True)
                tag_save_path.write_text(", ".join(tags), encoding="utf-8")
                completed += 1
                progress.update(task, completed=completed)

    def __call__(self) -> None:
        """Run the image tagging process."""
        if self.tag_source == "sidecar_txt":
            self.import_sidecar_tags()
            return
        self.run()


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
    parser.add_argument(
        "--tag_source",
        type=str,
        choices=["wd_tagger", "sidecar_txt"],
        default="wd_tagger",
        help="Tag source: run WD Tagger or import same-name sidecar txt files",
    )
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
        tag_source=args.tag_source,
        skip_existing=not args.no_skip_existing,
    )

    start_time = time.time()
    processor()
    end_time = time.time()

    elapsed = end_time - start_time
    logger.info("Tagging completed in %.2f seconds", elapsed)
