"""Conventions of the rectified-flow objective.

Every assertion here pins a sign or a direction that is easy to get backwards
and impossible to notice from the loss curve alone: a model trained on the
negated target still converges, it just runs the ODE the wrong way and renders
noise. The reference is diffusers' ``Lumina2Pipeline`` denoising loop.
"""

import torch
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from diffusion_trainer.config import Lumina2Config
from diffusion_trainer.finetune.objective import FlowMatchObjective

DEVICE = torch.device("cpu")


def make_objective(**overrides: object) -> FlowMatchObjective:
    config = Lumina2Config(model_path="x", dataset_path="x", **overrides)  # type: ignore[arg-type]
    return FlowMatchObjective(config, FlowMatchEulerDiscreteScheduler(), DEVICE)


def test_add_noise_matches_scheduler_scale_noise() -> None:
    """Training-time noising must equal the scheduler's own interpolation."""
    objective = make_objective()
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(num_inference_steps=10)

    latents = torch.randn(2, 16, 8, 8)
    noise = torch.randn_like(latents)

    # Pick a real schedule point so scale_noise can resolve it to the same sigma.
    timestep = scheduler.timesteps[3]
    sigma = scheduler.sigmas[3]

    ours = objective.add_noise(latents, noise, sigma.repeat(2))
    theirs = scheduler.scale_noise(latents, timestep.repeat(2), noise)

    torch.testing.assert_close(ours, theirs)


def test_sigma_endpoints() -> None:
    """sigma=1 is pure noise, sigma=0 is the clean latent."""
    objective = make_objective()
    latents = torch.randn(1, 16, 4, 4)
    noise = torch.randn_like(latents)

    torch.testing.assert_close(objective.add_noise(latents, noise, torch.ones(1)), noise)
    torch.testing.assert_close(objective.add_noise(latents, noise, torch.zeros(1)), latents)


def test_model_timesteps_match_pipeline_convention() -> None:
    """The DiT's time input is 1 - sigma (Lumina counts t=1 as the image)."""
    objective = make_objective()
    sigmas = torch.tensor([0.0, 0.25, 1.0])
    torch.testing.assert_close(objective.model_timesteps(sigmas), torch.tensor([1.0, 0.75, 0.0]))

    # This is exactly what the pipeline computes: 1 - t / num_train_timesteps,
    # where t is the scheduler timestep and sigma = t / num_train_timesteps.
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(num_inference_steps=10)
    for index in range(len(scheduler.timesteps)):
        pipeline_timestep = 1 - scheduler.timesteps[index] / scheduler.config.num_train_timesteps
        ours = objective.model_timesteps(scheduler.sigmas[index].reshape(1))
        torch.testing.assert_close(ours, pipeline_timestep.reshape(1), rtol=1e-5, atol=1e-5)


def test_target_is_negative_velocity() -> None:
    """The network's output space is x0 - noise, not noise - x0."""
    objective = make_objective()
    latents = torch.randn(2, 16, 4, 4)
    noise = torch.randn_like(latents)
    model_pred = torch.zeros_like(latents)

    target, pred = objective.target_and_pred(latents, noise, torch.rand(2), model_pred)

    torch.testing.assert_close(target, latents - noise)
    assert pred is model_pred


def test_perfect_model_denoises_toward_the_image() -> None:
    """End-to-end sign check across training and sampling.

    A model that emits exactly the training target, negated by the pipeline and
    fed to ``scheduler.step``, must move the sample TOWARD the clean latent. If
    either the target sign or the timestep direction were flipped, this walks
    away from x0 and the assertion fails.
    """
    objective = make_objective()
    scheduler = FlowMatchEulerDiscreteScheduler()
    scheduler.set_timesteps(num_inference_steps=20)

    torch.manual_seed(0)
    x0 = torch.randn(1, 16, 4, 4)
    noise = torch.randn_like(x0)

    index = 5
    sigma = scheduler.sigmas[index]
    sample = objective.add_noise(x0, noise, sigma.reshape(1))

    # The training target this model has learned to emit.
    target, _ = objective.target_and_pred(x0, noise, sigma.reshape(1), torch.empty(0))

    scheduler._step_index = index  # noqa: SLF001  # step() otherwise re-resolves it from the timestep
    # The pipeline negates the network output before stepping.
    stepped = scheduler.step(-target, scheduler.timesteps[index], sample, return_dict=False)[0]

    before = (sample - x0).abs().mean()
    after = (stepped - x0).abs().mean()
    assert after < before, f"denoising moved away from x0: {before:.4f} -> {after:.4f}"


def test_unset_shift_follows_the_sampler() -> None:
    """A config that doesn't pin a shift must inherit the model's own.

    Otherwise training and inference sit on different sigma bands: Lumina 2
    samples with shift=6, so a hardcoded training default of 1 would train a
    distribution the sampler never visits.
    """
    config = Lumina2Config(model_path="x", dataset_path="x")
    assert config.flow_match_shift is None

    following = FlowMatchObjective(config, FlowMatchEulerDiscreteScheduler(shift=6.0), DEVICE)
    assert following.shift == 6.0

    # An explicit value still wins over the sampler's.
    pinned = FlowMatchObjective(
        Lumina2Config(model_path="x", dataset_path="x", flow_match_shift=1.0),
        FlowMatchEulerDiscreteScheduler(shift=6.0),
        DEVICE,
    )
    assert pinned.shift == 1.0


def test_shift_biases_sigma_toward_noise() -> None:
    """flow_match_shift > 1 pushes sampled sigmas up, matching Lumina's sampler."""
    unshifted = make_objective(flow_match_shift=1.0)
    shifted = make_objective(flow_match_shift=6.0)

    sigmas = torch.linspace(0.01, 0.99, 50)
    moved = shifted._apply_shift(sigmas)  # noqa: SLF001

    torch.testing.assert_close(unshifted._apply_shift(sigmas), sigmas)  # noqa: SLF001
    assert (moved > sigmas).all()
    # Endpoints stay pinned so the range is still [0, 1].
    torch.testing.assert_close(shifted._apply_shift(torch.tensor([0.0, 1.0])), torch.tensor([0.0, 1.0]))  # noqa: SLF001


def test_sampled_sigmas_stay_in_range() -> None:
    objective = make_objective(flow_match_shift=6.0)
    sigmas = objective.sample_timesteps(512, global_step=0)
    assert sigmas.shape == (512,)
    assert (sigmas >= 0).all()
    assert (sigmas <= 1).all()


def test_uniform_loss_weighting_is_ones() -> None:
    objective = make_objective(flow_match_loss_weighting="uniform")
    weights = objective.loss_weights(torch.rand(8))
    torch.testing.assert_close(weights, torch.ones(8))


def test_sigma_sqrt_weighting_is_finite_near_zero() -> None:
    """sigma**-2 diverges at sigma=0; the floor must keep it finite."""
    objective = make_objective(flow_match_loss_weighting="sigma_sqrt")
    weights = objective.loss_weights(torch.tensor([0.0, 1e-9, 0.5, 1.0]))
    assert torch.isfinite(weights).all()
