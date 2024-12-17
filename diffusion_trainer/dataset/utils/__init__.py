"""Utility functions for dataset module."""

import os
import re
from collections.abc import Generator
from pathlib import Path
from re import Pattern


def get_meta_key_from_path(path: Path, ds_path: Path) -> str:
    """Get the metadata key from a path."""
    return path.relative_to(ds_path).with_suffix("").as_posix()


IMAGE_EXTENSIONS = [
    "jpg",
    "jpeg",
    "png",
    "gif",
    "bmp",
    "tiff",
    "webp",
    "avif",
]

NPZ_EXTENSIONS = ["npz"]


def regex_files(
    dir_path: Path | str,
    pattern: Pattern[str],
    *,
    recursive: bool = True,
) -> Generator[Path, None, None]:
    """Regex files in a directory."""
    if recursive:
        return regex_files_recursive(dir_path, pattern)
    return regex_files_flat(dir_path, pattern)


def regex_files_flat(dir_path: Path | str, pattern: Pattern[str]) -> Generator[Path, None, None]:
    for file in os.listdir(dir_path):
        path = Path(file)
        if pattern.match(path.as_posix()):
            abs_path = Path(dir_path) / file
            yield abs_path


def regex_files_recursive(dir_path: Path | str, pattern: Pattern[str]) -> Generator[Path, None, None]:
    for root, _, files in os.walk(dir_path):
        for file in files:
            path = (Path(root) / file).relative_to(dir_path)
            if pattern.match(path.as_posix()):
                abs_path = Path(root) / file
                yield abs_path


def compile_pattern(extensions: list[str], *, ignore_hidden: bool) -> re.Pattern:
    # Create a string that matches any of the given file extensions.
    ext_pattern = "|".join(re.escape(ext) for ext in extensions)
    pattern = f"^(?!\\.).*\\.({ext_pattern})$" if ignore_hidden else f".*\\.({ext_pattern})$"
    return re.compile(pattern, re.IGNORECASE)


def retrieve_image_paths(dir_path: Path | str, *, ignore_hidden: bool = True, recursive: bool = True) -> Generator[Path, None, None]:
    """Glob image files in a directory."""
    pattern = compile_pattern(IMAGE_EXTENSIONS, ignore_hidden=ignore_hidden)
    return regex_files(dir_path, pattern, recursive=recursive)


def retrieve_npz_path(dir_path: Path | str, *, ignore_hidden: bool = True, recursive: bool = True) -> Generator[Path, None, None]:
    """Glob npz files in a directory."""
    pattern = compile_pattern(NPZ_EXTENSIONS, ignore_hidden=ignore_hidden)
    return regex_files(dir_path, pattern, recursive=recursive)
