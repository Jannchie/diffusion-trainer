"""Utility functions for dataset module."""

from pathlib import Path


def get_meta_key_from_path(path: Path, ds_path: Path) -> str:
    """Get the metadata key from a path."""
    return path.relative_to(ds_path).with_suffix("").as_posix()


IMAGE_EXTENSIONS = [
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".PNG",
    ".JPG",
    ".JPEG",
    ".WEBP",
    ".BMP",
]

NPZ_EXTENSIONS = [".npz"]


def glob_files(
    dir_path: str | Path,
    extensions: list[str],
    *,
    recursive: bool = False,
    ignore_hidden: bool = True,
) -> list[Path]:
    """Glob files with given extensions in a directory."""
    dir_path = Path(dir_path)
    paths = set()

    def is_hidden(p: Path) -> bool:
        """Check if a path is hidden."""
        return any(part.startswith(".") for part in p.parts)

    glob_func = dir_path.rglob if recursive else dir_path.glob

    for ext in extensions:
        for path in glob_func(f"*{ext}"):
            if (ignore_hidden and is_hidden(path)) or path.is_dir():
                continue
            paths.add(path)

    return sorted(paths)


def glob_images_pathlib(dir_path: Path, *, recursive: bool = False, ignore_hidden: bool = True) -> list[Path]:
    """Glob image files in a directory."""
    return glob_files(dir_path, IMAGE_EXTENSIONS, recursive=recursive, ignore_hidden=ignore_hidden)


def glob_npz_files(dir_path: Path, *, recursive: bool = False, ignore_hidden: bool = True) -> list:
    """Glob npz files in a directory."""
    return glob_files(dir_path, NPZ_EXTENSIONS, recursive=recursive, ignore_hidden=ignore_hidden)
