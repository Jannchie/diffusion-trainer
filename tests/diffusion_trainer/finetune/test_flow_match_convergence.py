"""End-to-end learnability check for the rectified-flow training chain.

Overfits a tiny DiT onto a single latent through the REAL chain
(``sample_timesteps`` -> ``noisy_latents`` -> ``model_timesteps`` ->
``target_and_pred``), then samples it back with the pipeline's own arithmetic.

This is the test that catches a sign or direction flip end to end: a model
trained on a negated target still shows a falling loss in isolation, but the
sampling half then walks away from the data. Running both halves against each
other is what makes the pair meaningful.
"""

from collections.abc import Callable

import pytest
import torch
from diffusers import FlowMatchEulerDiscreteScheduler
from diffusers.models.transformers.transformer_lumina2 import Lumina2Transformer2DModel

from diffusion_trainer.config import Lumina2Config
from diffusion_trainer.finetune.objective import FlowMatchObjective

# The fixture itself is injected by pytest; only its type needs naming here.
TinyTransformerFactory = Callable[..., Lumina2Transformer2DModel]

CAP_FEAT_DIM = 48
TRAIN_STEPS = 300


@pytest.mark.slow
def test_tiny_dit_overfits_and_samples_back(tiny_transformer: TinyTransformerFactory) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    transformer = tiny_transformer(hidden_size=96, num_layers=3, cap_feat_dim=CAP_FEAT_DIM).to(device).train()

    config = Lumina2Config(model_path="x", dataset_path="x", flow_match_shift=6.0)
    objective = FlowMatchObjective(config, FlowMatchEulerDiscreteScheduler(shift=6.0), device)

    # A single fixed sample, so the optimal velocity field is analytic.
    x0 = torch.randn(1, 16, 8, 8, device=device)
    cond = torch.randn(1, 8, CAP_FEAT_DIM, device=device)
    mask = torch.ones(1, 8, dtype=torch.long, device=device)

    optimizer = torch.optim.AdamW(transformer.parameters(), lr=3e-3)
    losses = []
    for step in range(TRAIN_STEPS):
        noise = torch.randn_like(x0)
        sigmas = objective.sample_timesteps(1, global_step=step)
        noisy = objective.noisy_latents(x0, noise, sigmas, global_step=step)

        pred = transformer(
            hidden_states=noisy,
            timestep=objective.model_timesteps(sigmas),
            encoder_hidden_states=cond,
            encoder_attention_mask=mask,
            return_dict=False,
        )[0]
        target, pred = objective.target_and_pred(x0, noise, sigmas, pred)
        loss = torch.nn.functional.mse_loss(pred.float(), target.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    first = sum(losses[:20]) / 20
    last = sum(losses[-20:]) / 20
    assert last < first * 0.5, f"loss did not halve: {first:.4f} -> {last:.4f}"

    # Sampling half: replicate the pipeline's loop exactly, including its
    # negation of the network output before the Euler step.
    transformer.eval()
    scheduler = FlowMatchEulerDiscreteScheduler(shift=6.0)
    scheduler.set_timesteps(num_inference_steps=30, device=device)

    torch.manual_seed(7)
    latents = torch.randn_like(x0)
    start_distance = (latents - x0).abs().mean().item()

    with torch.no_grad():
        for index, t in enumerate(scheduler.timesteps):
            current_timestep = (1 - t / scheduler.config.num_train_timesteps).expand(1)
            out = transformer(
                hidden_states=latents,
                timestep=current_timestep,
                encoder_hidden_states=cond,
                encoder_attention_mask=mask,
                return_dict=False,
            )[0]
            scheduler._step_index = index  # noqa: SLF001
            latents = scheduler.step(-out, t, latents, return_dict=False)[0]

    end_distance = (latents - x0).abs().mean().item()
    assert end_distance < start_distance * 0.5, f"sampling diverged from the learned latent: {start_distance:.4f} -> {end_distance:.4f}"
