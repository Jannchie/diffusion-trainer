"""Single source of truth for dtype string parsing and defaults."""

import torch


def str_to_dtype(dtype: str) -> torch.dtype:
    """Map a dtype name (and its common aliases) to a ``torch.dtype``."""
    if dtype in ("float16", "half", "fp16"):
        return torch.float16
    if dtype in ("float32", "float", "fp32"):
        return torch.float32
    if dtype in ("float64", "double", "fp64"):
        return torch.float64
    if dtype in ("bfloat16", "bf16"):
        return torch.bfloat16
    msg = f"Unknown dtype {dtype}"
    raise ValueError(msg)


def get_default_dtype() -> torch.dtype:
    """Get the best default dtype based on hardware support."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16
