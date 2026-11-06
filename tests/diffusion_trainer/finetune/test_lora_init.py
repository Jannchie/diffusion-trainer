"""Normalized-LoRA down-projection inits (arXiv 2608.31036) on a LyCORIS network.

What must hold: B still starts at zero, the forward pass sees unit-norm
columns of A, and an exported LoRA holds the normalized A -- so ComfyUI and
friends, which know nothing about NoRA, reproduce the trained delta.
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from diffusion_trainer.finetune.utils.lora import MOSS_IMAGE_LORA_TARGETS, apply_lora_config, normalize_columns


class _Block(nn.Module):  # the class name MOSS_IMAGE_LORA_TARGETS keys on
    def __init__(self, dim: int = 48) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({"q": nn.Linear(dim, dim, bias=False), "ffn_down": nn.Linear(2 * dim, dim, bias=False)})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj["q"](x)


class Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _config(lora_init: str, dim: int = 8) -> SimpleNamespace:
    return SimpleNamespace(
        lora_dim=dim,
        lora_alpha=dim,
        lora_dropout=0.0,
        lora_init=lora_init,
        lokr_factor=16,
        conv_dim=None,
        conv_alpha=None,
        lora_multiplier=1.0,
        lokr_linear_dim=10000,
        lokr_feedforward_factor_ratio=0.5,
    )


def _network(lora_init: str, dim: int = 8):  # noqa: ANN202
    torch.manual_seed(0)
    model = Tiny()
    network = apply_lora_config("lora", model, _config(lora_init, dim), targets=MOSS_IMAGE_LORA_TARGETS)  # type: ignore[arg-type]
    return model, network


def _column_norms(a: torch.Tensor) -> torch.Tensor:
    return a.reshape(a.shape[0], -1).norm(dim=0)


@pytest.mark.parametrize("lora_init", ["nora", "nora-init", "bimi"])
def test_inits_give_unit_columns_and_keep_b_zero(lora_init: str) -> None:
    _, network = _network(lora_init)
    for module in network.loras:
        torch.testing.assert_close(_column_norms(module.lora_down.weight), torch.ones(module.lora_down.weight.shape[1]))
        assert torch.count_nonzero(module.lora_up.weight) == 0


def test_bimi_is_block_identity_with_a_partial_tail() -> None:
    _, network = _network("bimi", dim=5)
    a = next(m for m in network.loras if m.lora_name.endswith("_q")).lora_down.weight  # 5 x 48 = 9 full blocks + E_3
    for b in range(9):
        torch.testing.assert_close(a[:, 5 * b : 5 * b + 5], torch.eye(5))
    torch.testing.assert_close(a[:, 45:], torch.eye(5)[:, :3])


def test_default_init_is_untouched() -> None:
    _, network = _network("default")
    norms = _column_norms(network.loras[0].lora_down.weight)
    assert norms.max() < 0.5  # kaiming: ~sqrt(r / 3k), far from 1


def test_nora_stays_normalized_through_training_and_export() -> None:
    model, network = _network("nora")
    # Grads must reach the raw A through the parametrization.
    params = list(network.parameters())
    assert any(p.shape == network.loras[0].lora_down.parametrizations.weight.original.shape for p in params)
    module = network.loras[0]
    raw = module.lora_down.parametrizations.weight.original
    raw_before = raw.detach().clone()
    opt = torch.optim.Adam(params, lr=0.05)
    x = torch.randn(4, 48)
    for _ in range(5):  # B starts at zero, so A only gets a gradient from step 2 on
        opt.zero_grad()
        model(x).pow(2).mean().backward()
        opt.step()

    assert not torch.allclose(raw, raw_before)  # A trained...
    assert not torch.allclose(_column_norms(raw), torch.ones(raw.shape[1]))  # ...its raw columns left the sphere...
    torch.testing.assert_close(_column_norms(module.lora_down.weight), torch.ones(raw.shape[1]))  # ...the used ones did not

    exported = network.state_dict()
    down = exported[f"{module.lora_name}.lora_down.weight"]
    torch.testing.assert_close(down, normalize_columns(raw))
    # An unaware loader computes scale * up @ down; that must equal the trained delta.
    delta = exported[f"{module.lora_name}.lora_up.weight"] @ down * module.scale
    torch.testing.assert_close(delta, module.get_diff_weight()[0])


def test_nora_checkpoint_round_trips_through_load_state_dict() -> None:
    # The accelerate resume path: state_dict() out, load_state_dict() into a fresh network.
    _, network = _network("nora")
    with torch.no_grad():
        for module in network.loras:
            module.lora_up.weight.normal_()
            module.lora_down.parametrizations.weight.original.mul_(3.0).add_(0.1)

    _, restored = _network("nora")
    restored.load_state_dict(network.state_dict())
    for before, after in zip(network.loras, restored.loras, strict=True):
        torch.testing.assert_close(after.get_diff_weight()[0], before.get_diff_weight()[0])
        torch.testing.assert_close(_column_norms(after.lora_down.weight), torch.ones(after.lora_down.weight.shape[1]))


def test_non_lora_modes_reject_custom_init() -> None:
    with pytest.raises(ValueError, match="lora_down"):
        apply_lora_config("lokr", Tiny(), _config("nora"), targets=MOSS_IMAGE_LORA_TARGETS)  # type: ignore[arg-type]
