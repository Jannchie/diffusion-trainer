"""Utility functions for dataset module."""

import hashlib
import os
from collections.abc import Generator
from functools import cache
from pathlib import Path


def get_meta_key_from_path(path: Path, base_path: Path) -> str:
    """Get the metadata key from a path."""
    return path.relative_to(base_path.resolve()).with_suffix("").as_posix()


@cache
def calculate_file_sha256(path: Path) -> str:
    """Return the SHA256 hex digest of a file's content.

    Cached so a given file is hashed at most once per run: the preprocessing
    pipeline otherwise reads each (multi-MB) image 2-3 times to recompute the
    same hash. ``functools.cache`` is thread-safe in CPython, which the
    threaded processors rely on.
    """
    hash_sha256 = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(65536):  # 64KB chunks
            hash_sha256.update(chunk)
    return hash_sha256.hexdigest()


def sharded_path(root: Path | str, sha256: str, suffix: str) -> Path:
    """Resolve the canonical sharded storage path ``root/ab/cd/<sha256>.<suffix>``.

    This is the single source of truth for the SHA256-based directory layout
    used across latents, tags and metadata. ``suffix`` may be given with or
    without a leading dot.
    """
    suffix = suffix.lstrip(".")
    return Path(root) / sha256[:2] / sha256[2:4] / f"{sha256}.{suffix}"


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
        for path in dir_path.iterdir():
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
        for path in dir_path.iterdir():
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
        for path in dir_path.iterdir():
            if ignore_hidden and is_hidden_file(path):
                continue
            if path.suffix[1:].lower() in ("txt",):
                yield path
