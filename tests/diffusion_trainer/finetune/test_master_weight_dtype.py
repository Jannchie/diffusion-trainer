"""``weight_dtype`` covers frozen models only; the optimizer's targets stay fp32.

See ``BaseTuner.model_dtype`` for why a bf16 trainable weight both rounds away
small updates and breaks SD 1.5's clip_skip path.
"""

from types import SimpleNamespace

import torch

from diffusion_trainer.finetune.base import BaseTuner


class _DtypeTuner(BaseTuner):
    """Concrete BaseTuner whose only working part is model_dtype."""

    def get_pipeline(self) -> object:
        raise NotImplementedError

    def process_batch(self, batch: object) -> object:
        raise NotImplementedError

    def train_each_batch(self, batch: object) -> None:
        raise NotImplementedError

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def _configure_full_finetune(self) -> None:
        raise NotImplementedError


def make_tuner(mode: str, weight_dtype: torch.dtype) -> _DtypeTuner:
    """Built with ``__new__``: ``__init__`` would spin up an accelerator and load
    a real pipeline, while model_dtype reads nothing but these two attributes."""
    tuner = _DtypeTuner.__new__(_DtypeTuner)
    tuner.config = SimpleNamespace(mode=mode)  # type: ignore[assignment]
    tuner.weight_dtype = weight_dtype
    return tuner


def test_full_finetune_splits_by_learning_rate() -> None:
    tuner = make_tuner("full-finetune", torch.bfloat16)
    assert tuner.model_dtype(1e-5) == torch.float32, "the optimizer updates it, so it needs fp32 master weights"
    assert tuner.model_dtype(0) == torch.bfloat16, "a zero LR means frozen, so it may stay compact"
    assert tuner.model_dtype(None) == torch.bfloat16


def test_lora_keeps_the_frozen_base_model_compact() -> None:
    """The base model is frozen in every LoRA mode; the adapter carries its own dtype."""
    assert make_tuner("lokr", torch.bfloat16).model_dtype(1e-3) == torch.bfloat16


def test_fp32_weight_dtype_is_unchanged_either_way() -> None:
    tuner = make_tuner("full-finetune", torch.float32)
    assert tuner.model_dtype(1e-5) == torch.float32
    assert tuner.model_dtype(0) == torch.float32
