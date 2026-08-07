"""Regression cover for the DDPM objective extracted out of ``BaseTuner``.

The SD 1.5 lineage (the pictoria v0.6/v0.8 configs) depends on these numerics
exactly, so the extraction had to be behavior-preserving. Each test recomputes
the expected value from first principles rather than from the objective's own
helpers, so a refactor that changes the formula cannot make the test agree with
itself.
"""

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import compute_snr

from diffusion_trainer.config import SD15Config
from diffusion_trainer.finetune.objective import DDPMObjective

DEVICE = torch.device("cpu")


def make_objective(prediction_type: str = "epsilon", **overrides: object) -> DDPMObjective:
    config = SD15Config(model_path="x", dataset_path="x", **overrides)  # type: ignore[arg-type]
    scheduler = DDPMScheduler(num_train_timesteps=1000, prediction_type=prediction_type)
    return DDPMObjective(config, scheduler, DEVICE)


def test_epsilon_target_is_the_noise() -> None:
    objective = make_objective("epsilon")
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    pred = torch.randn_like(latents)

    target, out_pred = objective.target_and_pred(latents, noise, torch.tensor([10, 500]), pred)

    assert target is noise
    assert out_pred is pred


def test_v_prediction_target_matches_scheduler() -> None:
    objective = make_objective("v_prediction")
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([10, 500])

    target, _ = objective.target_and_pred(latents, noise, timesteps, torch.zeros_like(latents))

    torch.testing.assert_close(target, objective.scheduler.get_velocity(latents, noise, timesteps))


def test_sample_prediction_subtracts_the_noise_residual() -> None:
    objective = make_objective("sample")
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    pred = torch.randn_like(latents)

    target, out_pred = objective.target_and_pred(latents, noise, torch.tensor([10, 500]), pred)

    assert target is latents
    torch.testing.assert_close(out_pred, pred - noise)


def test_default_loss_weights_are_unit() -> None:
    """No SNR options set: every sample weighs the same."""
    objective = make_objective()
    weights = objective.loss_weights(torch.tensor([0, 250, 999]))
    torch.testing.assert_close(weights, torch.ones(3))


def test_min_snr_weights_match_the_published_formula() -> None:
    """min(SNR, gamma)/SNR for epsilon — the Min-SNR-gamma paper's epsilon form."""
    objective = make_objective("epsilon", snr_gamma=5.0, use_smooth_min_snr=False)
    timesteps = torch.tensor([0, 100, 500, 999])

    weights = objective.loss_weights(timesteps)

    all_snr = compute_snr(objective.scheduler, torch.arange(0, 1000, dtype=torch.long))
    snr = all_snr[timesteps].clamp(min=1e-8)
    expected = torch.minimum(snr, torch.full_like(snr, 5.0)) / snr

    torch.testing.assert_close(weights, expected)


def test_min_snr_weights_v_prediction_divides_by_snr_plus_one() -> None:
    objective = make_objective("v_prediction", snr_gamma=5.0, use_smooth_min_snr=False)
    timesteps = torch.tensor([0, 100, 500, 999])

    weights = objective.loss_weights(timesteps)

    all_snr = compute_snr(objective.scheduler, torch.arange(0, 1000, dtype=torch.long))
    snr = all_snr[timesteps].clamp(min=1e-8)
    expected = torch.minimum(snr, torch.full_like(snr, 5.0)) / (snr + 1)

    torch.testing.assert_close(weights, expected)


def test_debiased_estimation_is_inverse_sqrt_snr() -> None:
    objective = make_objective(use_debiased_estimation=True)
    timesteps = torch.tensor([100, 500, 999])

    weights = objective.loss_weights(timesteps)

    all_snr = compute_snr(objective.scheduler, torch.arange(0, 1000, dtype=torch.long))
    expected = 1.0 / torch.sqrt(all_snr[timesteps].clamp(min=1e-8, max=1000))

    torch.testing.assert_close(weights, expected)


def test_uniform_timesteps_span_the_schedule() -> None:
    objective = make_objective()
    timesteps = objective.sample_timesteps(4096, global_step=0)

    assert timesteps.dtype == torch.long
    assert timesteps.min() >= 0
    assert timesteps.max() < 1000
    assert timesteps.float().mean() > 400  # uniform over [0, 1000)


def test_timestep_bias_curriculum_defers_to_uniform() -> None:
    """A bias strategy stays uniform until timestep_bias_start_step.

    snr-detail concentrates hard on low timesteps, so the curriculum boundary is
    visible in the mean; before it, the distribution must still be flat.
    """
    objective = make_objective(timestep_bias_strategy="snr-detail", timestep_bias_start_step=100)

    before = objective.sample_timesteps(4096, global_step=0).float().mean()
    after = objective.sample_timesteps(4096, global_step=100).float().mean()

    assert before > 400
    assert after < before


def test_input_perturbation_decays_linearly_to_zero() -> None:
    objective = make_objective(input_perturbation=0.1, input_perturbation_steps=100)
    noise = torch.zeros(1, 4, 8, 8)

    torch.manual_seed(0)
    assert objective.perturb_noise(noise, global_step=0).abs().sum() > 0
    # Past the decay window the noise is returned untouched.
    assert objective.perturb_noise(noise, global_step=100).abs().sum() == 0
    assert objective.perturb_noise(noise, global_step=500).abs().sum() == 0


def test_input_perturbation_off_returns_the_same_tensor() -> None:
    objective = make_objective()
    noise = torch.randn(1, 4, 8, 8)
    assert objective.perturb_noise(noise, global_step=0) is noise


def test_add_noise_matches_scheduler() -> None:
    objective = make_objective()
    latents = torch.randn(2, 4, 8, 8)
    noise = torch.randn_like(latents)
    timesteps = torch.tensor([10, 500])

    torch.testing.assert_close(
        objective.add_noise(latents, noise, timesteps),
        objective.scheduler.add_noise(latents, noise, timesteps),
    )


def test_model_timesteps_is_identity() -> None:
    """UNets take the schedule index directly — no remapping, unlike Lumina."""
    objective = make_objective()
    timesteps = torch.tensor([0, 250, 999])
    assert objective.model_timesteps(timesteps) is timesteps
