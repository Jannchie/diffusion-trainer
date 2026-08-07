"""Shared fixtures for the finetune tests."""

from collections.abc import Callable

import pytest
from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel

TinyTransformerFactory = Callable[..., Lumina2Transformer2DModel]


def build_tiny_transformer(
    *,
    hidden_size: int = 64,
    num_layers: int = 2,
    cap_feat_dim: int = 32,
) -> Lumina2Transformer2DModel:
    """A structurally faithful, numerically tiny Lumina 2 DiT.

    Shared because the rope dimensions are constrained — ``sum(axes_dim_rope)``
    must equal ``hidden_size // num_attention_heads`` — so the sizes are not
    freely choosable and getting them wrong fails deep inside diffusers.
    Callers read ``model.config`` for the dimensions rather than importing
    constants, so the model stays the single source of truth for its own shape.
    """
    num_attention_heads = 4
    head_dim = hidden_size // num_attention_heads
    axis = head_dim // 4  # three rope axes (time, height, width) splitting head_dim
    return Lumina2Transformer2DModel(
        sample_size=16,
        patch_size=2,
        in_channels=16,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_refiner_layers=1,
        num_attention_heads=num_attention_heads,
        num_kv_heads=2,
        multiple_of=8,
        axes_dim_rope=(head_dim - 2 * axis, axis, axis),
        axes_lens=(300, 512, 512),
        cap_feat_dim=cap_feat_dim,
    )


@pytest.fixture
def tiny_transformer() -> TinyTransformerFactory:
    """Factory for tiny Lumina 2 DiTs; call it with the sizes a test needs."""
    return build_tiny_transformer
