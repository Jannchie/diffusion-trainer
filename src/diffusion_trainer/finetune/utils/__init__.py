"""Utility functions for finetuning."""

import datetime
import hashlib
import os
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Literal, NamedTuple, NotRequired, Protocol, TypedDict, TypeVar

import torch
import wandb
from accelerate import Accelerator, DeepSpeedPlugin, DistributedDataParallelKwargs, InitProcessGroupKwargs
from accelerate.utils import PrecisionType, ProfileKwargs
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.utils.torch_utils import is_compiled_module

from diffusion_trainer.config import SampleOptions

logger = getLogger("diffusion_trainer")


class ParamDict(TypedDict):
    lr: float
    params: list[torch.Tensor]
    eps: NotRequired[tuple[float | None, float]]  # Optional for Adafactor optimizer


class TrainableModel(NamedTuple):
    model: torch.nn.Module
    lr: float


class PipelineProtocol(Protocol):
    @classmethod
    def from_single_file(cls, pretrained_model_link_or_path: str, **kwargs) -> "PipelineProtocol": ...  # noqa: ANN003

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | PathLike | None, **kwargs) -> "PipelineProtocol": ...  # noqa: ANN003

    @property
    def progress_bar(self) -> object: ...


T = TypeVar("T", bound=PipelineProtocol)


def format_size(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f} B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f} M"
    if num >= 1_000:
        return f"{num / 1_000:.1f} K"
    return str(num)


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

    # Enable torch.compile (PyTorch 2.0+)
    dynamo_backend = None
    torch_compile = torch.__version__ >= "2.0.0"

    if torch_compile:
        # Use the inductor backend, which is the fastest backend in PyTorch 2.0
        dynamo_backend = "inductor"
        logger.info("Torch compile enabled with %s backend", dynamo_backend)

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


class DummyProgressBar:
    def __init__(self, total: int) -> None:
        pass

    def __enter__(self) -> "DummyProgressBar":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def update(self) -> None:
        pass


def load_sdxl_pipeline(
    path: PathLike | str,
    dtype: torch.dtype,
    *,
    enable_flash_attention: bool = True,
    flash_attention_unet: bool = True,
) -> StableDiffusionXLPipeline:
    return load_pipeline(
        path,
        dtype,
        StableDiffusionXLPipeline,
        enable_flash_attention=enable_flash_attention,
        flash_attention_unet=flash_attention_unet,
    )


def load_sd15_pipeline(
    path: PathLike | str,
    dtype: torch.dtype,
    *,
    enable_flash_attention: bool = True,
    flash_attention_unet: bool = True,
) -> StableDiffusionPipeline:
    return load_pipeline(
        path,
        dtype,
        StableDiffusionPipeline,
        enable_flash_attention=enable_flash_attention,
        flash_attention_unet=flash_attention_unet,
    )


def load_pipeline[T: PipelineProtocol](
    path: PathLike | str,
    dtype: torch.dtype,
    pipe_type: type[T],
    *,
    enable_flash_attention: bool = True,
    flash_attention_unet: bool = True,
) -> T:
    path = Path(path)

    logger.info('Loading models from "%s" (%s)', path, dtype)
    with Path(os.devnull).open("w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        if path.suffix == ".safetensors":
            pipe = pipe_type.from_single_file(path.as_posix(), torch_dtype=dtype)
        else:
            pipe = pipe_type.from_pretrained(path.as_posix(), torch_dtype=dtype)
    logger.info("Models loaded successfully.")

    # Enable Flash Attention if requested
    if enable_flash_attention:
        from diffusion_trainer.utils.flash_attention import enable_flash_attention_pipeline

        enable_flash_attention_pipeline(
            pipe,
            enable_unet=flash_attention_unet,
            enable_text_encoder=False,
        )

    if isinstance(pipe, pipe_type):
        pipe.progress_bar = DummyProgressBar  # type: ignore
        return pipe
    msg = "Failed to load models."
    raise ValueError(msg)


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model  # type: ignore  # noqa: SLF001


def get_n_params(trainable_parameters: list[ParamDict]) -> int:
    n_params = 0
    for param in trainable_parameters:
        n_params += sum(p.numel() for p in param["params"])
    return n_params


def prepare_params(accelerator: Accelerator, model: torch.nn.Module, lr: float) -> ParamDict:
    params = ParamDict(
        params=list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=lr,
    )
    logger.info("%s learning rate: %s, number of parameters: %s", model.__class__.__name__, lr, format_size(get_n_params([params])))
    accelerator.prepare(model)
    return params


def get_trainable_parameter_dicts(accelerator: Accelerator, trainable_mdels: list[TrainableModel]) -> list[ParamDict]:
    return [prepare_params(accelerator, model.model, model.lr) for model in trainable_mdels]


def initialize_optimizer(optimizer_str: str, trainable_parameters_dicts: list[ParamDict]) -> torch.optim.Optimizer:
    if optimizer_str == "adamW8bit":
        import bitsandbytes as bnb

        optimizer = bnb.optim.AdamW8bit(
            trainable_parameters_dicts,
            # lr=self.unet_lr,
            betas=(0.9, 0.999),
            weight_decay=1e-2,
            eps=1e-6,
        )
    else:
        # Adafactor with Kohya_SS style parameters
        from torch.optim import Adafactor

        # Ensure each parameter group has the correct eps format
        for param_dict in trainable_parameters_dicts:
            param_dict["eps"] = (1e-30, 1e-3)  # Set eps for each param group

        optimizer = Adafactor(
            trainable_parameters_dicts,  # type: ignore
            eps=(1e-30, 1e-3),  # Global default (eps1, eps2) as required by Adafactor
            beta2_decay=-0.8,
            weight_decay=0.0,
        )
    return optimizer


def compute_sqrt_inv_snr_weights(timesteps: torch.Tensor, all_snr: torch.Tensor) -> torch.Tensor:
    """
    Compute 1 / sqrt(SNR) weights.
    """
    snr_t = all_snr[timesteps].to(timesteps.device)  # Ensure device consistency
    snr_t = torch.clamp(snr_t, min=1e-8, max=1000)  # Prevent both division by zero and excessively large values by adding a minimum clamp
    return 1.0 / torch.sqrt(snr_t)
