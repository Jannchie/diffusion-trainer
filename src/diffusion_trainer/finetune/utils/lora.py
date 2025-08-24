from logging import getLogger
from typing import TYPE_CHECKING, Literal

import torch
from lycoris import LycorisNetwork, create_lycoris

from diffusion_trainer.finetune.utils import format_size

if TYPE_CHECKING:
    from diffusion_trainer.config import BaseConfig

logger = getLogger("diffusion_trainer")


def apply_lora_config(mode: Literal["lora", "loha", "lokr", "locon"], model: torch.nn.Module, config: "BaseConfig | None" = None) -> LycorisNetwork:
    # Use config values if provided, otherwise use defaults
    lora_rank = config.lora_rank if config else 16
    lora_alpha = config.lora_alpha if config else 1.0
    lora_dropout = config.lora_dropout if config else 0.0
    lokr_factor = config.lokr_factor if config else 16

    # Advanced configuration parameters
    multiplier = config.lora_multiplier if config else 1.0
    lokr_linear_dim = config.lokr_linear_dim if config else 10000
    lokr_ff_factor_ratio = config.lokr_feedforward_factor_ratio if config else 0.5

    if mode == "lora":
        lycoris_config = {
            "algo": "lora",
            "multiplier": multiplier,
            "linear_dim": lora_rank,
            "linear_alpha": lora_alpha,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
            },
        )
    elif mode == "loha":
        lycoris_config = {
            "algo": "loha",
            "multiplier": multiplier,
            "linear_dim": lora_rank,  # LoHA uses same rank parameter as LoRA
            "linear_alpha": lora_alpha,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
            },
        )
    elif mode == "lokr":
        lycoris_config = {
            "algo": "lokr",
            "multiplier": multiplier,
            "linear_dim": lokr_linear_dim,
            "linear_alpha": 1,  # Ignored when using full dimension
            "factor": lokr_factor,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
                "module_algo_map": {
                    "Attention": {"factor": lokr_factor},
                    "FeedForward": {"factor": max(1, int(lokr_factor * lokr_ff_factor_ratio))},
                },
            },
        )
    elif mode == "locon":
        lycoris_config = {
            "algo": "locon",
            "multiplier": multiplier,
            "linear_dim": lora_rank,
            "linear_alpha": lora_alpha,
            "linear_dropout": lora_dropout,
            "conv_dim": lora_rank,
            "conv_alpha": lora_alpha,
            "conv_dropout": lora_dropout,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
            },
        )
    else:  # type: ignore[misc]  # Defensive programming for runtime safety
        msg = f"Unsupported mode: {mode}"
        raise ValueError(msg)

    lycoris_model = create_lycoris(
        model,
        **lycoris_config,
    )
    lycoris_model.apply_to()
    lycoris_num_params = sum(p.numel() for p in lycoris_model.parameters())
    logger.info(
        "LyCORIS network (%s) has been initialized with %s parameters (%s)",
        mode.upper(),
        f"{lycoris_num_params:,}",
        format_size(lycoris_num_params),
    )
    return lycoris_model
