from logging import getLogger
from typing import Literal

import torch
from lycoris import LycorisNetwork, create_lycoris

from diffusion_trainer.finetune.utils import format_size

logger = getLogger("diffusion_trainer")


def apply_lora_config(mode: Literal["lora", "loha", "lokr"], model: torch.nn.Module) -> LycorisNetwork:
    if mode == "lokr":
        lycoris_config = {
            "algo": "lokr",
            "multiplier": 1.0,
            "linear_dim": 10000,  # Full dimension
            "linear_alpha": 1,  # Ignored in full dimension
            "factor": 16,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
                "module_algo_map": {
                    "Attention": {"factor": 16},
                    "FeedForward": {"factor": 8},
                },
            },
        )
        lycoris_model = create_lycoris(
            model,
            **lycoris_config,
        )
        lycoris_model.apply_to()
        lycoris_num_params = sum(p.numel() for p in lycoris_model.parameters())
        logger.info(
            "LyCORIS network has been initialized with %s parameters (%s)",
            f"{lycoris_num_params:,}",
            format_size(lycoris_num_params),
        )
        return lycoris_model
    raise NotImplementedError
