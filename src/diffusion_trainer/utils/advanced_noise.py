"""Advanced noise generation techniques for diffusion model training."""

from typing import Literal

import torch
import torch.nn.functional as F


def pyramid_noise(
    shape: torch.Size | tuple[int, ...],
    discount_factor: float = 0.8,
    num_levels: int = 5,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate pyramid noise by combining noise at different resolutions.

    Pyramid noise improves training stability and output quality by adding
    structured noise patterns across multiple scales.

    Args:
        shape: Target tensor shape (batch_size, channels, height, width)
        discount_factor: How much each level is discounted (0.0 to 1.0)
        num_levels: Number of pyramid levels to generate
        device: Target device for the tensor
        dtype: Data type for the tensor

    Returns:
        Pyramid noise tensor with the specified shape
    """
    # Ensure we have at least 4 dimensions for BCHW format
    if len(shape) < 4:
        msg = f"Shape must have at least 4 dimensions (BCHW), got {len(shape)}"
        raise ValueError(msg)

    batch_size, channels, height, width = shape[:4]
    noise = torch.zeros(shape, device=device, dtype=dtype)

    for level in range(num_levels):
        # Calculate the scale factor for this level
        scale = 2 ** level
        level_height = max(1, height // scale)
        level_width = max(1, width // scale)

        # Generate noise at this resolution
        level_noise = torch.randn(
            batch_size, channels, level_height, level_width,
            device=device, dtype=dtype,
        )

        # Upscale to target resolution if needed
        if level_height != height or level_width != width:
            level_noise = F.interpolate(
                level_noise,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        # Add to the accumulated noise with discount factor
        weight = discount_factor ** level
        noise += weight * level_noise

    return noise


def multi_resolution_noise(
    shape: torch.Size | tuple[int, ...],
    scales: list[float] | None = None,
    weights: list[float] | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate multi-resolution noise by combining different frequency components.

    This technique helps the model learn both fine details and global structure
    by providing noise at multiple frequency scales.

    Args:
        shape: Target tensor shape (batch_size, channels, height, width)
        scales: List of resolution scales to combine
        weights: Relative weights for each scale (defaults to equal weights)
        device: Target device for the tensor
        dtype: Data type for the tensor

    Returns:
        Multi-resolution noise tensor
    """
    if scales is None:
        scales = [1.0, 0.5, 0.25]

    if weights is None:
        weights = [1.0 / len(scales)] * len(scales)

    if len(scales) != len(weights):
        msg = "scales and weights must have the same length"
        raise ValueError(msg)

    # Ensure we have at least 4 dimensions for BCHW format
    if len(shape) < 4:
        msg = f"Shape must have at least 4 dimensions (BCHW), got {len(shape)}"
        raise ValueError(msg)

    batch_size, channels, height, width = shape[:4]
    combined_noise = torch.zeros(shape, device=device, dtype=dtype)

    for scale, weight in zip(scales, weights, strict=True):
        # Calculate scaled dimensions
        scaled_height = max(1, int(height * scale))
        scaled_width = max(1, int(width * scale))

        # Generate noise at scaled resolution
        scale_noise = torch.randn(
            batch_size, channels, scaled_height, scaled_width,
            device=device, dtype=dtype,
        )

        # Resize to target dimensions
        if scaled_height != height or scaled_width != width:
            scale_noise = F.interpolate(
                scale_noise,
                size=(height, width),
                mode="bilinear",
                align_corners=False,
            )

        # Add weighted contribution
        combined_noise += weight * scale_noise

    return combined_noise


def smooth_min_snr_weights(
    timesteps: torch.Tensor,
    all_snr: torch.Tensor,
    min_snr_gamma: float = 5.0,
    smoothing_factor: float = 0.1,
    mode: Literal["clip", "sigmoid", "tanh"] = "sigmoid",
) -> torch.Tensor:
    """
    Generate smooth Min-SNR weights with configurable smoothing.

    Args:
        timesteps: Current timesteps
        all_snr: Pre-computed SNR values for all timesteps
        min_snr_gamma: Min-SNR gamma value
        smoothing_factor: Smoothing strength (0.0 = no smoothing)
        mode: Smoothing mode ('clip', 'sigmoid', 'tanh')

    Returns:
        Smooth Min-SNR weights
    """
    snr = all_snr[timesteps]

    if mode == "clip":
        # Standard Min-SNR clipping
        weights = torch.clamp(snr, max=min_snr_gamma)
    elif mode == "sigmoid":
        # Smooth transition using sigmoid
        sigmoid_input = (snr - min_snr_gamma) / smoothing_factor
        smooth_factor = torch.sigmoid(-sigmoid_input)
        weights = snr * smooth_factor + min_snr_gamma * (1 - smooth_factor)
    elif mode == "tanh":
        # Smooth transition using tanh
        tanh_input = (snr - min_snr_gamma) / smoothing_factor
        smooth_factor = (1 - torch.tanh(tanh_input)) / 2
        weights = snr * smooth_factor + min_snr_gamma * (1 - smooth_factor)
    else:
        msg = f"Unknown smoothing mode: {mode}"
        raise ValueError(msg)

    return weights



def adaptive_noise_schedule(
    base_noise: torch.Tensor,
    timesteps: torch.Tensor,
    noise_schedule_type: Literal["linear", "cosine", "exponential"] = "cosine",
    strength_factor: float = 1.0,
) -> torch.Tensor:
    """
    Apply adaptive noise scheduling based on timesteps.

    Args:
        base_noise: Base noise tensor
        timesteps: Current timesteps
        noise_schedule_type: Type of noise schedule to apply
        strength_factor: Overall strength multiplier

    Returns:
        Scheduled noise tensor
    """
    # Normalize timesteps to [0, 1]
    t_norm = timesteps.float() / 1000.0  # Assuming max 1000 timesteps

    if noise_schedule_type == "linear":
        schedule = 1.0 - t_norm
    elif noise_schedule_type == "cosine":
        schedule = 0.5 * (1 + torch.cos(torch.pi * t_norm))
    elif noise_schedule_type == "exponential":
        schedule = torch.exp(-2.0 * t_norm)
    else:
        msg = f"Unknown noise schedule type: {noise_schedule_type}"
        raise ValueError(msg)

    # Reshape schedule to match noise dimensions
    while schedule.dim() < base_noise.dim():
        schedule = schedule.unsqueeze(-1)

    return base_noise * schedule * strength_factor
