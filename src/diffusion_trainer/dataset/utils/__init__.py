"""Utility functions for dataset module."""

import os
from collections.abc import Generator
from pathlib import Path


def get_meta_key_from_path(path: Path, base_path: Path) -> str:
    """Get the metadata key from a path."""
    return path.relative_to(base_path.resolve()).with_suffix("").as_posix()


IMAGE_EXTENSIONS = (
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "avif",
)

NPZ_EXTENSIONS = ("npz",)


def is_hidden_file(file: Path) -> bool:
    """Check if a file is hidden, or some of its parent directories are hidden."""
    return any(part.startswith(".") for part in file.parts)


def retrieve_image_paths(dir_path: Path | str, *, ignore_hidden: bool = True, recursive: bool = True) -> Generator[Path, None, None]:
    """Glob image files in a directory."""
    dir_path = Path(dir_path).resolve()
    if recursive:
        for root, _, files in os.walk(dir_path):
            for file in files:
                path = Path(root) / file
                if ignore_hidden and is_hidden_file(path):
                    continue
                if path.suffix[1:].lower() in IMAGE_EXTENSIONS:
                    yield path
    else:
        for file in os.listdir(dir_path):
            path = dir_path / file
            if ignore_hidden and is_hidden_file(path):
                continue
            if path.suffix[1:].lower() in IMAGE_EXTENSIONS:
                yield path


def retrieve_npz_path(dir_path: Path | str, *, ignore_hidden: bool = True, recursive: bool = True) -> Generator[Path, None, None]:
    """Glob npz files in a directory."""
    dir_path = Path(dir_path).resolve()
    if recursive:
        for root, _, files in os.walk(dir_path):
            for file in files:
                path = Path(root) / file
                if ignore_hidden and is_hidden_file(path):
                    continue
                if path.suffix[1:].lower() in NPZ_EXTENSIONS:
                    yield path
    else:
        for file in os.listdir(dir_path):
            path = dir_path / file
            if ignore_hidden and is_hidden_file(path):
                continue
            if path.suffix[1:].lower() in NPZ_EXTENSIONS:
                yield path


def retrieve_text_path(dir_path: Path | str, *, ignore_hidden: bool = True, recursive: bool = True) -> Generator[Path, None, None]:
    """Glob npz files in a directory."""
    dir_path = Path(dir_path).resolve()
    if recursive:
        for root, _, files in os.walk(dir_path):
            for file in files:
                path = Path(root) / file
                if ignore_hidden and is_hidden_file(path):
                    continue
                if path.suffix[1:].lower() in ("txt",):
                    yield path
    else:
        for file in os.listdir(dir_path):
            path = dir_path / file
            if ignore_hidden and is_hidden_file(path):
                continue
            if path.suffix[1:].lower() in ("txt",):
                yield path
