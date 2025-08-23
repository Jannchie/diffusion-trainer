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

from diffusion_trainer.shared import get_progress

console = get_console()


class CreateParquetProcessor:
    """Create parquet metadata files from SHA256-based directory structure."""

    def __init__(self, target_dir: str | PathLike, console: Console = console) -> None:
        self.target_dir = Path(target_dir)
        self.console = console

    lock = threading.Lock()

    def __call__(self, max_workers: int = 8) -> None:
        self.process(max_workers)

    def _find_all_npz_files(self) -> list[Path]:
        """Find all NPZ files in SHA256-based directory structure."""
        npz_files = []
        latents_dir = self.target_dir / "latents"
        if not latents_dir.exists():
            return npz_files

        for dir1 in latents_dir.iterdir():
            if not dir1.is_dir() or len(dir1.name) != 2:
                continue
            for dir2 in dir1.iterdir():
                if not dir2.is_dir() or len(dir2.name) != 2:
                    continue
                for file in dir2.iterdir():
                    if file.suffix == ".npz" and len(file.stem) == 64:  # SHA256 hash length
                        npz_files.append(file)
        return npz_files

    def _find_corresponding_txt_file(self, npz_path: Path) -> Path | None:
        """Find corresponding txt file for NPZ file based on SHA256 hash."""
        sha256_hash = npz_path.stem
        dir1 = sha256_hash[:2]
        dir2 = sha256_hash[2:4]

        # Look for tag file in various possible locations
        possible_tag_paths = [
            # Tags directory at same level as latents
            self.target_dir / "tags" / dir1 / dir2 / f"{sha256_hash}.txt",
            # Same directory structure but different root
            self.target_dir.parent / "tags" / dir1 / dir2 / f"{sha256_hash}.txt",
            # Legacy structure
            self.target_dir / "tags" / f"{sha256_hash}.txt",
            # Direct tags subdirectory (if target_dir is root)
            Path(str(self.target_dir).replace("/latents", "/tags")) / dir1 / dir2 / f"{sha256_hash}.txt",
        ]

        for tag_path in possible_tag_paths:
            if tag_path.exists():
                return tag_path
        return None

    def process(self, max_workers: int) -> None:
        """Process all NPZ files and create parquet metadata."""
        items = defaultdict(list)
        progress = get_progress()
        npz_path_list = self._find_all_npz_files()

        self.console.log(f"Found {len(npz_path_list)} NPZ files in {self.target_dir}")

        if not npz_path_list:
            self.console.log("No NPZ files found. Make sure latents have been generated.")
            return

        task = progress.add_task("Processing metadata files", total=len(npz_path_list))

        def process_metadata_files(npz_path: Path) -> None:
            try:
                sha256_hash = npz_path.stem
                npz = np.load(npz_path)

                # Find corresponding tag file
                tag_file = self._find_corresponding_txt_file(npz_path)

                # Read tags
                if tag_file and tag_file.exists():
                    tags_text = tag_file.read_text(encoding="utf-8").strip()
                    tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()] if tags_text else []
                else:
                    tags = []

                with self.lock:
                    items["key"].append(sha256_hash)  # Use SHA256 hash as key
                    items["tags"].append(tags)
                    items["train_resolution"].append(npz.get("train_resolution").tolist())
                    items["original_size"].append(npz.get("original_size").tolist())
                    items["crop_ltrb"].append(npz.get("crop_ltrb").tolist())

            except Exception as e:
                self.console.log(f"Error processing {npz_path}: {e}")

            progress.update(task, advance=1)

        with progress, ThreadPoolExecutor(max_workers=max_workers) as executor:
            for npz_path in npz_path_list:
                executor.submit(process_metadata_files, npz_path)

        if items:
            table = pa.table(items)
            output_path = self.target_dir / "metadata.parquet"
            pq.write_table(table, output_path)
            self.console.log(f"Created metadata parquet file: {output_path}")
            self.console.log(f"Processed {len(items['key'])} entries")
        else:
            self.console.log("No valid entries found to create parquet file")


def retrieve_npz_path(target_dir: Path) -> list[Path]:
    """Retrieve NPZ paths from SHA256-based directory structure (backward compatibility)."""
    processor = CreateParquetProcessor(target_dir)
    return processor._find_all_npz_files()
