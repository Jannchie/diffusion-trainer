import functools

import numpy as np
import torch

from diffusion_trainer.utils.log_normal_pdf import log_normal_pdf


@functools.lru_cache(maxsize=1)
def uniform_timestep_weights(num_timesteps: int = 1000, *, device: torch.device) -> torch.Tensor:
    weights = np.ones(num_timesteps) / num_timesteps
    return torch.tensor(weights, dtype=torch.float32, device=device)


@functools.lru_cache(maxsize=1)
def logit_timestep_weights(num_timesteps: int = 1000, *, m: float = 0, s: float = 1.0, device: torch.device) -> torch.Tensor:
    t_values = np.linspace(1e-10, 1, num_timesteps, endpoint=False)
    pdf = log_normal_pdf(t_values, m=m, s=s)
    # Ensure pdf is ndarray and compute sum
    pdf_array = np.asarray(pdf)
    weights = pdf_array / pdf_array.sum()
    return torch.tensor(weights, dtype=torch.float32, device=device)
