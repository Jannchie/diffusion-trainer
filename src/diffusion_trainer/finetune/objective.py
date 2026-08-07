"""Training objectives: the noising process, the prediction target and the loss weighting.

``BaseTuner`` used to hardcode one chain — discrete timesteps drawn against
``alphas_cumprod``, an ``epsilon``/``v_prediction`` target, SNR-derived loss
weights. Rectified-flow DiTs (Lumina 2) disagree at *every* link of that chain:
continuous sigmas instead of integer indices, a velocity target instead of a
noise target, and no SNR at all. Rather than branch the tuner on model family,
the chain lives here behind one interface and the tuner only orchestrates it.

What the families genuinely share stays in the tuner: how the noise tensor
itself is drawn (offset / pyramid / brownian), gradient accumulation, EMA,
checkpointing and previews.

``DDPMObjective`` is a verbatim port of the pre-refactor ``BaseTuner`` methods
(``sample_timesteps``, ``get_noisy_latents``, ``get_pred_target`` and the
weighting half of ``get_loss``). The pictoria lineage configs depend on those
numerics exactly, so the port must stay behavior-preserving.
"""

import logging
from abc import ABC, abstractmethod

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, compute_snr

from diffusion_trainer.config import BaseConfig, FlowMatchConfig
from diffusion_trainer.utils.advanced_noise import smooth_min_snr_weights
from diffusion_trainer.utils.timestep_weights import logit_timestep_weights

logger = logging.getLogger("diffusion_trainer")


def compute_sqrt_inv_snr_weights(timesteps: torch.Tensor, all_snr: torch.Tensor) -> torch.Tensor:
    """Compute 1 / sqrt(SNR) weights (debiased estimation)."""
    snr_t = all_snr[timesteps].to(timesteps.device)  # Ensure device consistency
    snr_t = torch.clamp(snr_t, min=1e-8, max=1000)  # Prevent both division by zero and excessively large values by adding a minimum clamp
    return 1.0 / torch.sqrt(snr_t)


class DiffusionObjective(ABC):
    """One diffusion formulation: how to noise, what to predict, how to weight.

    ``timesteps`` flows through every method as an opaque per-sample tensor. Its
    meaning is the objective's business — integer indices into a 1000-step
    schedule for DDPM, continuous sigmas in ``[0, 1]`` for flow matching — so
    callers must never index tables with it or assume a dtype.
    """

    def __init__(self, config: BaseConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    @abstractmethod
    def sample_timesteps(self, batch_size: int, *, global_step: int) -> torch.Tensor:
        """Draw one timestep per sample. ``global_step`` drives curriculum schedules."""

    @abstractmethod
    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Interpolate clean latents toward noise at ``timesteps``."""

    @abstractmethod
    def target_and_pred(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        model_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(target, prediction)`` for the MSE, both in model output space."""

    @abstractmethod
    def loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Per-sample loss weights, shape ``(batch,)``."""

    def model_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Map sampled timesteps into what the network expects as its time input.

        Identity for the DDPM lineage; flow-matching models may parameterize
        time in the opposite direction (see ``FlowMatchObjective``).
        """
        return timesteps

    def perturb_noise(self, noise: torch.Tensor, *, global_step: int) -> torch.Tensor:
        """Apply input perturbation to the noise that gets ADDED to the latents.

        The clean ``noise`` is what the target is built from; only the noising
        path sees the perturbed copy. Optional linear decay over
        ``input_perturbation_steps`` lets a run start perturbed and anneal off.
        """
        strength = self.config.input_perturbation
        if strength <= 0:
            return noise

        decay_steps = self.config.input_perturbation_steps
        if decay_steps > 0:
            strength = strength * (1.0 - global_step / decay_steps) if global_step < decay_steps else 0.0

        if strength <= 0:
            return noise
        return noise + strength * torch.randn_like(noise)

    def noisy_latents(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, *, global_step: int) -> torch.Tensor:
        """Full noising step: perturb the noise, then interpolate."""
        return self.add_noise(latents, self.perturb_noise(noise, global_step=global_step), timesteps)

    @abstractmethod
    def log_summary(self) -> None:
        """Log the objective's effective settings once at startup."""


class DDPMObjective(DiffusionObjective):
    """Discrete DDPM objective — the SD 1.5 / SDXL epsilon and v-prediction lineage.

    Timesteps are integer indices into the scheduler's 1000-step schedule, so
    every SNR table lookup in this class is a direct index.
    """

    def __init__(self, config: BaseConfig, scheduler: DDPMScheduler, device: torch.device) -> None:
        super().__init__(config, device)
        self.scheduler = scheduler
        self.num_train_timesteps: int = scheduler.config.get("num_train_timesteps", 1000)
        self.all_snr = compute_snr(scheduler, torch.arange(0, self.num_train_timesteps, dtype=torch.long)).to(device)  # type: ignore[arg-type]
        self._sigma_for_timesteps: torch.Tensor | None = None
        # Depends only on constants, so build it once rather than per step.
        self._snr_detail_weights = self.all_snr.clamp(min=config.vpred_snr_floor) / (self.all_snr + 1.0)

    @property
    def prediction_type(self) -> str:
        return self.scheduler.config.get("prediction_type", "epsilon")

    def _sample_timesteps_lognormal(self, batch_size: int) -> torch.Tensor:
        """Draw sigma from a lognormal distribution (EDM-style), map to the closest scheduler timestep."""
        if self._sigma_for_timesteps is None:
            alphas_cumprod = self.scheduler.alphas_cumprod  # type: ignore[attr-defined]
            if alphas_cumprod is None:
                msg = "Noise scheduler missing alphas_cumprod for lognormal sampling"
                raise ValueError(msg)
            eps = 1e-12
            alphas_cumprod = alphas_cumprod.to(device=self.device, dtype=torch.float32).clamp(min=eps)
            self._sigma_for_timesteps = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)

        sigma_tensor: torch.Tensor = self._sigma_for_timesteps

        mean = torch.tensor(self.config.timestep_lognormal_mean, device=self.device, dtype=torch.float32)
        std = torch.tensor(self.config.timestep_lognormal_std, device=self.device, dtype=torch.float32)
        lognormal = torch.distributions.LogNormal(mean, std)
        sampled_sigma = lognormal.sample((batch_size,))
        if sampled_sigma is None:
            msg = "Failed to sample sigma from lognormal distribution"
            raise ValueError(msg)
        # Find nearest timestep by sigma distance
        distance = torch.abs(sigma_tensor.view(1, -1) - sampled_sigma.view(-1, 1))
        return distance.argmin(dim=1).to(dtype=torch.long)

    def sample_timesteps(self, batch_size: int, *, global_step: int) -> torch.Tensor:
        # Curriculum: any bias strategy samples uniformly until its start step,
        # so the epsilon->v remap and the ZTSNR terminal regime train at full
        # density before compute is reallocated to the detail regime.
        strategy = self.config.timestep_bias_strategy
        if strategy != "uniform" and global_step < self.config.timestep_bias_start_step:
            strategy = "uniform"

        if strategy == "uniform":
            # Sample a random timestep for each image without bias.
            return torch.randint(0, self.num_train_timesteps, (batch_size,), device=self.device, dtype=torch.long)
        if strategy == "logit":
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = logit_timestep_weights(
                self.num_train_timesteps,
                m=self.config.timestep_bias_m,
                s=self.config.timestep_bias_s,
                device=self.device,
            )
            return torch.multinomial(weights, batch_size, replacement=True)
        if strategy == "lognormal":
            return self._sample_timesteps_lognormal(batch_size)
        if strategy == "snr-detail":
            # Sampler-side epsilon-equivalent allocation: density proportional to the
            # same max(SNR, floor)/(SNR+1) used by the loss weighting, with unit loss
            # weights — identical expected gradient, but no compute spent on
            # near-zero-weight samples and homogeneous per-sample gradient scale.
            # The floor keeps the ZTSNR terminal band at ~floor density.
            return torch.multinomial(self._snr_detail_weights, batch_size, replacement=True).to(self.device)
        msg = f"Unknown timestep bias strategy {strategy}"
        raise ValueError(msg)

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        return self.scheduler.add_noise(latents, noise, timesteps)  # type: ignore[arg-type]

    def target_and_pred(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        model_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        prediction_type = self.prediction_type
        if prediction_type == "epsilon":
            return noise, model_pred
        if prediction_type == "v_prediction":
            return self.scheduler.get_velocity(latents, noise, timesteps), model_pred  # type: ignore[arg-type]
        if prediction_type == "sample":
            # The target is the latents, but the model returns a noise sample
            # prediction, so the noise residual is subtracted from the prediction.
            return latents, model_pred - noise
        msg = f"Unknown prediction type {prediction_type}"
        raise ValueError(msg)

    def loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        weights = torch.ones(timesteps.shape[0], device=timesteps.device, dtype=torch.float32)

        if self.config.use_debiased_estimation:
            weights = weights * compute_sqrt_inv_snr_weights(timesteps, self.all_snr)

        snr_gamma = self.config.snr_gamma
        if snr_gamma is not None and snr_gamma > 0:
            # Clamp away from zero: with rescale_betas_zero_snr the terminal SNR
            # is 0, which would make the divisions below produce inf/nan.
            snr = self.all_snr[timesteps].clamp(min=1e-8)

            if self.config.use_smooth_min_snr:
                mse_loss_weights = smooth_min_snr_weights(
                    timesteps,
                    self.all_snr,
                    min_snr_gamma=snr_gamma,
                    smoothing_factor=self.config.smooth_min_snr_factor,
                    mode=self.config.smooth_min_snr_mode,
                )
            else:
                # Standard Min-SNR clipping (full_like keeps the float dtype of snr)
                mse_loss_weights = torch.stack([snr, torch.full_like(snr, snr_gamma)], dim=1).min(dim=1)[0]

            # Adjust weights according to prediction_type
            if self.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            weights = weights * mse_loss_weights

        return weights

    def log_summary(self) -> None:
        logger.info("Training objective: DDPM (%s), %d timesteps", self.prediction_type, self.num_train_timesteps)


class FlowMatchObjective(DiffusionObjective):
    """Rectified-flow objective for Lumina 2 and other flow-matching DiTs.

    Conventions, all verified against ``diffusers`` 0.39's ``Lumina2Pipeline``:

    * ``sigma`` in ``[0, 1]`` interpolates ``x = (1 - sigma) * x0 + sigma * noise``
      — matching ``FlowMatchEulerDiscreteScheduler.scale_noise``. ``sigma = 1``
      is pure noise, ``sigma = 0`` is the clean image.
    * The network's time input is ``1 - sigma``. Lumina numbers time with t=0 at
      the noise end and t=1 at the image end, the reverse of sigma; the pipeline
      spells this ``current_timestep = 1 - t / num_train_timesteps``.
    * The network predicts ``x0 - noise`` — the NEGATIVE flow velocity. The
      pipeline negates the output (``noise_pred = -noise_pred``) before handing
      it to ``scheduler.step``, whose Euler update ``x += (sigma_next - sigma) * v``
      requires ``v = dx/dsigma = noise - x0``. Training the un-negated target
      would produce a model that runs the ODE backwards.

    Timesteps here are continuous sigmas in ``[0, 1]``, NOT schedule indices, so
    nothing may index an SNR table with them.
    """

    # sigma**-2 diverges as sigma -> 0. Floor it so a sample drawn near the clean
    # end cannot produce an astronomically weighted gradient (sigma_sqrt only).
    _MIN_WEIGHTING_SIGMA = 1e-3

    def __init__(self, config: FlowMatchConfig, scheduler: FlowMatchEulerDiscreteScheduler, device: torch.device) -> None:
        super().__init__(config, device)
        self.config: FlowMatchConfig = config
        # The sampler's shift is the single source of truth for which sigma band
        # this model operates on; an unset config value follows it so training
        # and inference cannot silently disagree.
        configured_shift = config.flow_match_shift
        self.shift: float = float(scheduler.config.get("shift", 1.0)) if configured_shift is None else configured_shift

    def _apply_shift(self, sigmas: torch.Tensor) -> torch.Tensor:
        """Bias sigma toward the noisy end, the training-side mirror of the sampler's shift.

        ``shift = 1`` is the identity — the same formula
        ``FlowMatchEulerDiscreteScheduler`` applies to its own sigma schedule.
        """
        if self.shift == 1.0:
            return sigmas
        return self.shift * sigmas / (1.0 + (self.shift - 1.0) * sigmas)

    def sample_timesteps(self, batch_size: int, *, global_step: int) -> torch.Tensor:
        del global_step  # no curriculum for flow matching (yet)
        sigmas = compute_density_for_timestep_sampling(
            weighting_scheme=self.config.flow_match_timestep_sampling,
            batch_size=batch_size,
            logit_mean=self.config.flow_match_logit_mean,
            logit_std=self.config.flow_match_logit_std,
            mode_scale=self.config.flow_match_mode_scale,
            device=self.device,
        )
        return self._apply_shift(sigmas.to(dtype=torch.float32))

    def add_noise(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        # lerp(a, b, w) == (1 - w) * a + w * b, in one fused kernel.
        return torch.lerp(latents, noise, self._broadcast(timesteps, latents))

    def model_timesteps(self, timesteps: torch.Tensor) -> torch.Tensor:
        # Lumina counts time from noise (0) to image (1); sigma runs the other way.
        return 1.0 - timesteps

    def target_and_pred(
        self,
        latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        model_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del timesteps
        # Negative velocity: the pipeline negates the network output before the
        # Euler step, so the network's own output space is x0 - noise.
        return latents - noise, model_pred

    def loss_weights(self, timesteps: torch.Tensor) -> torch.Tensor:
        scheme = self.config.flow_match_loss_weighting
        if scheme == "uniform":
            return torch.ones_like(timesteps, dtype=torch.float32)
        sigmas = timesteps.to(dtype=torch.float32).clamp(min=self._MIN_WEIGHTING_SIGMA)
        return compute_loss_weighting_for_sd3(weighting_scheme=scheme, sigmas=sigmas).float()

    @staticmethod
    def _broadcast(timesteps: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        """Reshape per-sample sigmas to broadcast against ``(batch, C, H, W)`` latents."""
        return timesteps.to(device=reference.device, dtype=reference.dtype).view(-1, *([1] * (reference.ndim - 1)))

    def log_summary(self) -> None:
        logger.info(
            "Training objective: rectified flow (sampling=%s, shift=%s%s, weighting=%s)",
            self.config.flow_match_timestep_sampling,
            self.shift,
            "" if self.config.flow_match_shift is not None else " from sampler",
            self.config.flow_match_loss_weighting,
        )
