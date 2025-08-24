from logging import getLogger
from typing import TYPE_CHECKING, Literal

import torch
from lycoris import LycorisNetwork, create_lycoris

from diffusion_trainer.finetune.utils import format_size

if TYPE_CHECKING:
    from diffusion_trainer.config import BaseConfig

logger = getLogger("diffusion_trainer")


def apply_lora_config(mode: Literal["lora", "loha", "lokr"], model: torch.nn.Module, config: "BaseConfig | None" = None) -> LycorisNetwork:
    # Use config values if provided, otherwise use defaults
    lora_rank = config.lora_rank if config else 4
    lora_alpha = config.lora_alpha if config else 1
    lokr_factor = config.lokr_factor if config else 16

    if mode == "lora":
        lycoris_config = {
            "algo": "lora",
            "multiplier": 1.0,
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
            "multiplier": 1.0,
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
            "multiplier": 1.0,
            "linear_dim": 10000,  # Full dimension
            "linear_alpha": 1,  # Ignored in full dimension
            "factor": lokr_factor,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
                "module_algo_map": {
                    "Attention": {"factor": lokr_factor},
                    "FeedForward": {"factor": lokr_factor // 2},
                },
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
