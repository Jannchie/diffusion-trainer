"""MOSS-Image conventions that don't need the 26 GB checkpoint.

The DiT itself is remote code inside the model repo, so these pin the parts
this repo owns: the tech-report sigma sampler, the time/target conventions, the
FLUX.2 latent patchify, and that the LoRA target really reaches the linears
nested in each block's ModuleDict.
"""

import math

import pytest
import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from torch import nn

from diffusion_trainer.config import MOSSImageConfig
from diffusion_trainer.finetune.moss_image import patchify_latents, unpatchify_latents
from diffusion_trainer.finetune.objective import MOSSFlowMatchObjective
from diffusion_trainer.finetune.utils.lora import MOSS_IMAGE_LORA_TARGETS, apply_lora_config

DEVICE = torch.device("cpu")


def moss_scheduler() -> FlowMatchEulerDiscreteScheduler:
    """Stand-in for the remote ``MOSSImageScheduler``: shift 5 to sample, 1 to train."""
    scheduler = FlowMatchEulerDiscreteScheduler(shift=5.0)
    scheduler.register_to_config(training_shift=1.0)
    return scheduler


def make_objective(**overrides: object) -> MOSSFlowMatchObjective:
    config = MOSSImageConfig(model_path="x", dataset_path="x", **overrides)  # type: ignore[arg-type]
    return MOSSFlowMatchObjective(config, moss_scheduler(), DEVICE)


def report_pmf(steps: int = 1000, mu: float = -1.09, s: float = 0.95, cutoff: float = 0.95, uniform_mix: float = 0.01) -> torch.Tensor:
    """Tech report eq. 4 in its own convention: S = 1 - sigmoid(Z), Z ~ N(mu, s^2)."""

    def cdf(x: float) -> float:
        if x <= 0:
            return 0.0
        if x >= 1:
            return 1.0
        return 1 - 0.5 * math.erfc(-(math.log((1 - x) / x) - mu) / (s * math.sqrt(2)))

    edges = [cdf(cutoff * j / steps) for j in range(steps + 1)]
    mass = torch.tensor([edges[j] - edges[j - 1] for j in range(1, steps + 1)], dtype=torch.float64) / edges[-1]
    return (1 - uniform_mix) * mass + uniform_mix / steps


def test_default_sampler_matches_the_tech_report_pmf() -> None:
    """The sign flip between the report (mu=-1.09) and this repo (+1.09) is the trap here."""
    torch.manual_seed(0)
    objective = make_objective()
    sigmas = objective.sample_timesteps(400_000, global_step=0)

    # Every draw sits on the 1/1000 grid, inside (0, 1].
    grid = sigmas * 1000
    torch.testing.assert_close(grid, grid.round())
    assert sigmas.min() >= 1e-3
    assert sigmas.max() <= 1.0

    # Compare on 20 coarse bins: fine bins are too noisy at this sample size.
    counts = torch.bincount((grid.round().long() - 1), minlength=1000).double()
    observed = counts.view(20, 50).sum(1) / counts.sum()
    expected = report_pmf().view(20, 50).sum(1)
    torch.testing.assert_close(observed, expected, atol=4e-3, rtol=0)
    # The report's "median near 0.78": noise-heavy, not clean-heavy.
    assert 0.76 < sigmas.median().item() < 0.80


def test_unset_shift_follows_training_shift_not_inference_shift() -> None:
    assert make_objective().shift == 1.0
    assert make_objective(flow_match_shift=3.0).shift == 3.0


def test_time_input_and_target_follow_the_pipeline() -> None:
    """MOSSImagePipeline feeds sigma*1000 and Euler-steps the raw output with dx/dsigma."""
    objective = make_objective()
    latents = torch.randn(2, 128, 4, 4)
    noise = torch.randn_like(latents)
    sigmas = torch.tensor([0.25, 0.9])

    torch.testing.assert_close(objective.model_timesteps(sigmas), sigmas * 1000)

    noisy = objective.add_noise(latents, noise, sigmas)
    target, _ = objective.target_and_pred(latents, noise, sigmas, torch.zeros_like(latents))
    # A perfect prediction, Euler-stepped from sigma to 0 the way scheduler.step
    # does (x += (sigma_next - sigma) * v), must land exactly on the clean latents.
    stepped = noisy + (0 - sigmas.view(-1, 1, 1, 1)) * target
    torch.testing.assert_close(stepped, latents, atol=1e-5, rtol=1e-5)


def test_patchify_round_trips_and_matches_the_pipeline_decode() -> None:
    raw = torch.randn(2, 32, 16, 12)
    packed = patchify_latents(raw)
    assert packed.shape == (2, 128, 8, 6)
    torch.testing.assert_close(unpatchify_latents(packed), raw)

    # The exact reshape MOSSImagePipeline.__call__ decodes with.
    b, c, h, w = packed.shape
    pipeline_unpacked = packed.reshape(b, c // 4, 2, 2, h, w).permute(0, 1, 4, 2, 5, 3).reshape(b, c // 4, 2 * h, 2 * w)
    torch.testing.assert_close(pipeline_unpacked, raw)


class _Block(nn.Module):  # must match the remote DiT's class name, which is what LyCORIS keys on
    """The shape of MOSS's joint block: every linear in a ModuleDict, no linear classes of its own."""

    def __init__(self, dim: int = 32, ffn_dim: int = 64) -> None:
        super().__init__()
        self.proj = nn.ModuleDict({name: nn.Linear(dim, dim, bias=False) for name in ("q", "k", "v", "o")})
        self.proj["ffn_up_gate"] = nn.Linear(dim, 2 * ffn_dim, bias=False)
        self.proj["ffn_down"] = nn.Linear(ffn_dim, dim, bias=False)


class _TextBlock(nn.Module):
    def __init__(self, dim: int = 32) -> None:
        super().__init__()
        self.qkv = nn.Linear(dim, 3 * dim)


class TinyMOSS(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.text_embedder = nn.ModuleDict({"blocks": nn.ModuleList([_TextBlock()])})
        self.layers = nn.ModuleList([_Block(), _Block()])


@pytest.mark.parametrize("mode", ["lora", "lokr"])
def test_lora_target_reaches_all_six_block_linears_and_nothing_else(mode: str) -> None:
    network = apply_lora_config(mode, TinyMOSS(), targets=MOSS_IMAGE_LORA_TARGETS)  # type: ignore[arg-type]
    wrapped = sorted(module.lora_name for module in network.loras)

    expected = sorted(f"lycoris_layers_{i}_proj_{name}" for i in range(2) for name in ("q", "k", "v", "o", "ffn_up_gate", "ffn_down"))
    assert wrapped == expected
