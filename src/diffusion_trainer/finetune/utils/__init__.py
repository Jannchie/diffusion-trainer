"""Utility functions for finetuning."""

import datetime
import hashlib
import os
from dataclasses import asdict
from logging import getLogger
from pathlib import Path
from typing import Literal

import torch
import wandb
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import PrecisionType, ProfileKwargs

from diffusion_trainer.config import SampleOptions

logger = getLogger("diffusion_trainer")


def prepare_logger(log_with: str, logging_dir: Path) -> None:
    """Prepare logger for the training."""
    if log_with == "wandb":
        # logging_dir = logging_dir / f"finetune_{time.strftime('%Y%m%d%H%M%S', time.localtime())}"
        # logging_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = logging_dir.as_posix()
        wandb_api_key = os.getenv("WANDB_API_KEY")
        if wandb_api_key is not None:
            wandb.login(key=wandb_api_key)
    elif log_with == "tensorboard":
        msg = "Tensorboard logging is not implemented yet."
        raise NotImplementedError(msg)
    elif log_with == "none":
        pass


def prepare_deepspeed_plugin() -> None | DeepSpeedPlugin:
    """Prepare DeepSpeed plugin."""
    # TODO: Implement this function
    return None


def prepare_ddp_kwargs(
    *,
    ddp_timeout: int = 100_000,
    ddp_gradient_as_bucket_view: bool = False,
    ddp_static_graph: bool = False,
) -> list:
    """Prepare kwargs for DistributedDataParallel."""
    kwargs_handlers = (
        InitProcessGroupKwargs(timeout=datetime.timedelta(minutes=ddp_timeout)) if ddp_timeout else None,
        (
            DistributedDataParallelKwargs(gradient_as_bucket_view=ddp_gradient_as_bucket_view, static_graph=ddp_static_graph)
            if ddp_gradient_as_bucket_view or ddp_static_graph
            else None
        ),
    )
    return list(filter(lambda x: x is not None, kwargs_handlers))


def prepare_accelerator(
    gradient_accumulation_steps: int,
    dtype: torch.dtype = torch.float16,
    log_with: Literal["wandb", "tensorboard", "none"] = "none",
) -> Accelerator:
    """Prepare training tools, such as logger, accelerator, deepspeed, etc."""
    logging_dir = Path("./.logs")
    prepare_logger(log_with=log_with, logging_dir=logging_dir)

    if dtype == torch.float16:
        mixed_precision = PrecisionType.FP16
    elif dtype == torch.half:
        mixed_precision = PrecisionType.FP8
    elif dtype == torch.float32:
        mixed_precision = PrecisionType.NO
    elif dtype == torch.bfloat16:
        mixed_precision = PrecisionType.BF16
    else:
        msg = f"Unsupported dtype: {dtype}"
        raise ValueError(msg)

    dynamo_backend = None
    torch_compile = False

    if torch_compile:
        dynamo_backend = "inductor"

    deepspeed_plugin = prepare_deepspeed_plugin()
    ddp_kwargs = prepare_ddp_kwargs()

    profile_kwargs = ProfileKwargs(
        activities=["cuda"],
        profile_memory=True,
        record_shapes=True,
    )

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=log_with if log_with != "none" else None,
        project_dir=logging_dir,
        kwargs_handlers=[*ddp_kwargs, profile_kwargs],
        dynamo_backend=dynamo_backend,
        deepspeed_plugin=deepspeed_plugin,
    )
    logger.info("accelerator device: %s", accelerator.device)
    return accelerator


def str_to_dtype(dtype: str) -> torch.dtype:
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


def get_sample_options_hash(sample_options: SampleOptions) -> str:
    d = asdict(sample_options)
    return hashlib.sha256(str(d).encode()).hexdigest()[:8]
