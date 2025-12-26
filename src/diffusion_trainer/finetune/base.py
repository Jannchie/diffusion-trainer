import logging
import math
import random
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Generator, Sequence
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
import torch.nn.functional as F
from accelerate import PartialState
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel, compute_snr, free_memory
from torch.utils.data import DataLoader

from diffusion_trainer.config import BaseConfig, SampleOptions
from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch, DiffusionDataset
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import LatentsGenerateProcessor
from diffusion_trainer.dataset.processors.tagging_processor import TaggingProcessor
from diffusion_trainer.finetune.utils import (
    TrainableModel,
    compute_sqrt_inv_snr_weights,
    get_sample_options_hash,
    get_trainable_parameter_dicts,
    initialize_optimizer,
    prepare_accelerator,
    str_to_dtype,
    unwrap_model,
)
from diffusion_trainer.finetune.utils.lora import apply_lora_config
from diffusion_trainer.shared import get_progress
from diffusion_trainer.utils.advanced_noise import (
    adaptive_noise_schedule,
    brownian_noise,
    multi_resolution_noise,
    pyramid_noise,
    smooth_min_snr_weights,
)
from diffusion_trainer.utils.timestep_weights import logit_timestep_weights

if TYPE_CHECKING:
    from lycoris import LycorisNetwork

logger = logging.getLogger("diffusion_trainer")

ModelPredFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class BaseTuner(ABC):
    @staticmethod
    def from_config(_config: BaseConfig) -> "BaseTuner":
        """Create a Tuner from a config."""
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.save_path = Path(config.save_dir)
        self.mixed_precision = str_to_dtype(config.mixed_precision)
        self.weight_dtype = str_to_dtype(config.weight_dtype)
        self.save_dtype = str_to_dtype(config.save_dtype)
        self.vae_dtype = str_to_dtype(config.vae_dtype)
        self._sigma_for_timesteps: torch.Tensor | None = None
        self.ema_unet: EMAModel | None = None
        self.ema_unet_short: EMAModel | None = None

        # Common initialization steps
        self._initialize_environment()
        self._initialize_accelerator()
        self._initialize_pipeline_and_models()
        self._log_initialization_info()

    def _initialize_environment(self) -> None:
        """Initialize environment settings like seeds and CUDA cache."""
        self.apply_seed_settings(self.config.seed)

        # Clear CUDA cache before loading models to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initialize_accelerator(self) -> None:
        """Initialize accelerator and device settings."""
        self.accelerator = prepare_accelerator(
            self.config.gradient_accumulation_steps,
            self.mixed_precision,
            self.config.log_with,
        )
        self.device = self.accelerator.device

    def _initialize_pipeline_and_models(self) -> None:
        """Initialize pipeline, models, and training-related attributes."""
        self.pipeline = self.get_pipeline()
        self.lycoris_model: LycorisNetwork | None = None
        self.noise_scheduler: DDPMScheduler = self.get_noise_scheduler()
        self.all_snr = compute_snr(self.noise_scheduler, torch.arange(0, self.noise_scheduler.config.num_train_timesteps, dtype=torch.long)).to(self.device)  # type: ignore
        self.train_loss = 0.0
        self.trainable_models_with_lr: list[TrainableModel] = []
        self.training_models: list[torch.nn.Module] = []
        self.global_step = 0  # Track global training step for input perturbation decay

        # Initialize model-specific components (implemented by subclasses)
        self._setup_models()

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _setup_models(self) -> None:
        """Setup model-specific components. Must be implemented by subclasses."""
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for models. Can be overridden by subclasses."""
        models = getattr(self, "models", [])
        if models:
            self.init_gradient_checkpointing(models)

    def _log_initialization_info(self) -> None:
        """Log initialization information."""
        logger.info("Initialized %s with config:", self.__class__.__name__)
        for key, value in self.config.__dict__.items():
            logger.info("  %s: %s", key, value)

    def prepare_training(self, data_loader: torch.utils.data.DataLoader) -> tuple[int, int]:
        """
        Prepares the training process including dataset, trainable models, optimizer and scheduler.
        Returns a tuple containing dataloader, number of update steps per epoch, and total steps.
        """
        self.accelerator.wait_for_everyone()

        # Configure trainable models (subclass-specific implementation)
        self._configure_models()

        # Set models to training mode and freeze non-trainable models
        self.training_models = [model.model for model in self.trainable_models_with_lr]
        for model in self.training_models:
            model.train()
        freeze_models = [model for model in getattr(self, "models", []) if model not in self.training_models]
        self.freeze_model(freeze_models)

        # Initialize EMA models if enabled
        if self.config.use_ema:
            self.ema_unet = self.get_ema(self.pipeline.unet, self.pipeline.unet.config, decay=self.config.ema_decay_long)
            if getattr(self.config, "use_dual_ema", False):
                self.ema_unet_short = self.get_ema(self.pipeline.unet, self.pipeline.unet.config, decay=self.config.ema_decay_short)

        # Prepare optimizer
        self.trainable_parameters_dicts = get_trainable_parameter_dicts(self.accelerator, self.trainable_models_with_lr)
        self.optimizer = initialize_optimizer(
            self.config.optimizer,
            self.trainable_parameters_dicts,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        num_update_steps_per_epoch = math.ceil(len(data_loader) / self.config.gradient_accumulation_steps)
        n_total_steps = self.config.n_epochs * num_update_steps_per_epoch

        # Initialize learning rate scheduler
        # Use different scheduler based on optimizer type
        if self.config.optimizer == "adafactor":
            # For Adafactor, use constant_with_warmup as recommended by SS-Script
            scheduler_type = SchedulerType.CONSTANT_WITH_WARMUP
            self.lr_scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.optimizer_warmup_steps,
                num_training_steps=n_total_steps,
            )
        else:
            # For other optimizers, use cosine with restarts
            scheduler_type = SchedulerType.COSINE_WITH_RESTARTS
            self.lr_scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=self.config.optimizer_warmup_steps,
                num_training_steps=n_total_steps,
                num_cycles=self.config.optimizer_num_cycles,
            )
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        # Initialize trackers
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
        self.accelerator.init_trackers(
            "diffusion-trainer",
            config=self.config.__dict__,
            init_kwargs={
                "wandb": {
                    "name": timestamp,  # Set run name to timestamp
                },
            },
        )

        return num_update_steps_per_epoch, n_total_steps

    def prepare_data_loader(self) -> torch.utils.data.DataLoader:
        dataset = self.prepare_dataset(self.config)
        if self.accelerator.is_main_process:
            dataset.print_bucket_info()
        sampler = BucketBasedBatchSampler(dataset, self.config.batch_size)
        with self.accelerator.main_process_first(): # type: ignore
            data_loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=0,
                collate_fn=dataset.collate_fn,
                pin_memory=True,
            )
        return self.accelerator.prepare(data_loader)

    def train(self) -> None:
        # Prepare data loader
        data_loader = self.prepare_data_loader()

        num_update_steps_per_epoch, n_total_steps = self.prepare_training(data_loader)
        self.execute_training_epoch(
            data_loader,
            self.lr_scheduler,
            self.training_models,
            num_update_steps_per_epoch,
            n_total_steps,
        )

    def _configure_models(self) -> None:
        """Configure which models should be trainable with their learning rates."""
        if self.config.mode == "full-finetune":
            self._configure_full_finetune()
        elif self.config.mode in ("lora", "lokr", "loha"):
            self._configure_lora_finetune()
        else:
            msg = f"Unknown training mode: {self.config.mode}"
            raise ValueError(msg)

    @abstractmethod
    def _configure_full_finetune(self) -> None:
        """Configure models for full fine-tuning. Must be implemented by subclasses."""
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def _configure_lora_finetune(self) -> None:
        """Configure models for LoRA fine-tuning. Uses template method pattern."""
        unet_model = self._get_unet_model()
        # Type check - this should never happen if called correctly, but satisfies type checker
        if self.config.mode == "full-finetune":
            msg = "Cannot configure LoRA for full-finetune mode"
            raise ValueError(msg)

        lycoris_model = apply_lora_config(self.config.mode, unet_model, self.config)

        # Ensure LoRA model dtype matches the weight dtype to prevent dtype mismatch errors
        lycoris_model.to(dtype=self.weight_dtype)
        logger.info("LoRA model dtype set to: %s", self.weight_dtype)

        # Use getattr to safely access unet_lr which may be defined in subclasses
        unet_lr = getattr(self.config, "unet_lr", 1e-5)
        self.trainable_models_with_lr.append(TrainableModel(model=lycoris_model, lr=unet_lr))
        self.lycoris_model = lycoris_model

        # Allow subclasses to perform additional setup
        self._post_lora_setup(lycoris_model)

    @abstractmethod
    def _get_unet_model(self) -> torch.nn.Module:
        """Get the UNet model for LoRA configuration. Must be implemented by subclasses."""
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def _post_lora_setup(self, lycoris_model: "LycorisNetwork") -> None:
        """Perform additional setup after LoRA model creation. Can be overridden by subclasses."""
        # Default implementation does nothing

    def optimizer_step(self, loss: torch.Tensor) -> None:
        """
        Performs backward pass, gradient clipping, optimizer step,
        learning rate scheduling and optimizer zero_grad.
        """
        if torch.isnan(loss):
            logger.info("Loss is NaN.")
            msg = "Loss is NaN."
            raise ValueError(msg)

        avg_loss = self.accelerator.gather(loss.repeat(self.config.batch_size)).mean()  # type: ignore
        self.train_loss += avg_loss.item() / self.config.gradient_accumulation_steps

        # Backpropagate
        self.accelerator.backward(loss)

        # Free memory
        del loss, avg_loss

        # Apply gradient clipping if configured
        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
            params_to_clip = []
            for group in self.optimizer.param_groups:
                group_params = group.get("params", [])
                if isinstance(group_params, torch.Tensor):
                    params_to_clip.append(group_params)
                else:
                    params_to_clip.extend(group_params)
            if params_to_clip:
                self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

        # Optimizer step
        self.optimizer.step()

        # Update EMA model(s) if enabled
        step_for_ema = self.global_step + (1 if self.accelerator.sync_gradients else 0)
        self._step_emas(step_for_ema)

        # LR scheduler step
        self.lr_scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)

        # Clear CUDA cache only when necessary (not after every sync)
        # Removed frequent cache clearing to improve performance

    def _should_step_ema(self, step: int) -> bool:
        return self.config.use_ema and step >= getattr(self.config, "ema_start_step", 0)

    def _step_emas(self, step: int) -> None:
        if not self.accelerator.sync_gradients:
            return
        if not self._should_step_ema(step):
            return
        if self.ema_unet is not None:
            self.ema_unet.step(self.pipeline.unet.parameters())
        if getattr(self.config, "use_dual_ema", False) and self.ema_unet_short is not None:
            self.ema_unet_short.step(self.pipeline.unet.parameters())

    def generate_initial_preview(self) -> None:
        """Generate preview images before training starts if configured to do so."""
        logger.info("Generating preview before training starts")
        self.generate_preview(f"{self.config.model_name}-before-training", 0)

    @abstractmethod
    def get_pipeline(self) -> DiffusionPipeline:
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    @abstractmethod
    def process_batch(self, batch: DiffusionBatch) -> Any:  # noqa: ANN401
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    @abstractmethod
    def train_each_batch(self, batch: Any) -> None:  # noqa: ANN401
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def train_on_latents(
        self,
        img_latents: torch.Tensor,
        model_pred_fn: ModelPredFn,
        *,
        extra_tensors: Sequence[torch.Tensor | None] = (),
    ) -> None:
        """Run a single training step given latents and a model prediction function."""
        # Apply VAE scaling
        img_latents = self._apply_vae_scaling(img_latents)

        # Prepare training tensors
        img_latents, noise, timesteps, img_noisy_latents = self._prepare_training_tensors(img_latents)

        # Predict and calculate loss
        model_pred = model_pred_fn(img_noisy_latents, timesteps)
        target, model_pred = self.get_pred_target(img_latents, noise, timesteps, model_pred)
        loss = self.get_loss(timesteps, model_pred, target)

        # Free memory efficiently
        self._free_tensors(
            noise,
            img_noisy_latents,
            model_pred,
            target,
            img_latents,
            *extra_tensors,
        )

        # Optimizer step
        self.optimizer_step(loss)

    def get_loss(self, timesteps: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # 1. Compute the basic MSE loss (per sample)
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss.mean(dim=list(range(1, len(loss.shape))))

        # Initialize weights to 1
        weights = torch.ones_like(loss_per_sample)

        # 2. If Debiased Estimation is enabled, compute and apply debiasing weights
        if self.config.use_debiased_estimation:
            debias_weights = compute_sqrt_inv_snr_weights(timesteps, self.all_snr)
            weights = weights * debias_weights  # Multiply by debiasing weights

        # 3. If SNR weighting is enabled, compute and apply SNR weights
        if self.config.snr_gamma is not None:
            # Extract SNR values for timesteps once
            snr = self.all_snr[timesteps]

            # Use smooth Min-SNR if configured
            if getattr(self.config, "use_smooth_min_snr", False):
                mse_loss_weights = smooth_min_snr_weights(
                    timesteps,
                    self.all_snr,
                    min_snr_gamma=self.config.snr_gamma,
                    smoothing_factor=getattr(self.config, "smooth_min_snr_factor", 0.1),
                    mode=getattr(self.config, "smooth_min_snr_mode", "sigmoid"),
                )
            else:
                # Standard Min-SNR clipping
                mse_loss_weights = torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]

            # Adjust weights according to prediction_type
            if self.noise_scheduler.config.get("prediction_type") == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif self.noise_scheduler.config.get("prediction_type") == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            weights = weights * mse_loss_weights  # Further multiply by SNR weights

        # 4. Apply the final weights and compute the mean loss
        return (loss_per_sample * weights).mean()

    def get_pred_target(
        self,
        img_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        model_pred: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise_scheduler = self.noise_scheduler
        pred = model_pred
        if noise_scheduler.config.get("prediction_type") == "epsilon":
            target = noise
        elif noise_scheduler.config.get("prediction_type") == "v_prediction":
            target = noise_scheduler.get_velocity(img_latents, noise, timesteps)  # type: ignore
        elif noise_scheduler.config.get("prediction_type") == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = img_latents
            # We will have to subtract the noise residual from the prediction to get the target sample.
            pred = model_pred - noise
        else:
            msg = f"Unknown prediction type {noise_scheduler.config.get('prediction_type')}"
            raise ValueError(msg)
        return target, pred

    def sample_noise(self, latents: torch.Tensor) -> torch.Tensor:
        """Sample noise that will be added to the latents."""
        # Use Brownian noise if requested
        if getattr(self.config, "use_brownian_noise", False):
            noise = brownian_noise(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                scale=getattr(self.config, "brownian_noise_scale", 1.0),
            )
        # Use multi-resolution noise if enabled
        elif getattr(self.config, "use_multires_noise", True):
            # Check if custom scales are provided
            custom_scales = getattr(self.config, "multires_noise_scales", None)
            custom_weights = getattr(self.config, "multires_noise_weights", None)

            if custom_scales is not None:
                # Use custom scales/weights method
                noise = multi_resolution_noise(
                    latents.shape,
                    scales=custom_scales,
                    weights=custom_weights,
                    device=latents.device,
                    dtype=latents.dtype,
                )
            else:
                # Use pyramid method with iterations and discount
                noise = pyramid_noise(
                    latents.shape,
                    discount_factor=getattr(self.config, "multires_noise_discount", 0.8),
                    num_levels=getattr(self.config, "multires_noise_iterations", 6),
                    device=latents.device,
                    dtype=latents.dtype,
                )
        else:
            # Standard random noise
            noise = torch.randn_like(latents)

        # Apply traditional noise offset if configured with probability
        if hasattr(self.config, "noise_offset") and self.config.noise_offset > 0:
            # Apply noise offset with configured probability
            noise_offset_prob = getattr(self.config, "noise_offset_probability", 1.0)
            if noise_offset_prob >= 1.0 or random.random() < noise_offset_prob:
                # Add noise to the image latents
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                )
        return noise

    def get_noisy_latents(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Get noisy latents"""
        if self.config.input_perturbation > 0:
            # Apply input perturbation with optional step-based decay
            input_perturbation = self.config.input_perturbation

            # Apply linear decay if input_perturbation_steps is configured
            if self.config.input_perturbation_steps > 0 and self.global_step < self.config.input_perturbation_steps:
                # Linear decay: starts at full strength, decays to 0 over input_perturbation_steps
                decay_factor = 1.0 - (self.global_step / self.config.input_perturbation_steps)
                input_perturbation *= decay_factor
            elif self.config.input_perturbation_steps > 0 and self.global_step >= self.config.input_perturbation_steps:
                # After decay period, set perturbation to 0
                input_perturbation = 0.0

            if input_perturbation > 0:
                noise = noise + input_perturbation * torch.randn_like(noise)

        # Apply adaptive noise scheduling if configured
        if getattr(self.config, "use_adaptive_noise", False):
            noise = adaptive_noise_schedule(
                noise,
                timesteps,
                noise_schedule_type=getattr(self.config, "adaptive_noise_type", "cosine"),
                strength_factor=getattr(self.config, "adaptive_noise_strength", 1.0),
            )

        return self.noise_scheduler.add_noise(latents, noise, timesteps)  # type: ignore

    def _sample_timesteps_lognormal(self, batch_size: int) -> torch.Tensor:
        """
        Sample timesteps by drawing sigma from a lognormal distribution (EDM-style)
        and mapping to the closest scheduler timestep.
        """
        if self._sigma_for_timesteps is None:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod  # type: ignore[attr-defined]
            if alphas_cumprod is None:
                msg = "Noise scheduler missing alphas_cumprod for lognormal sampling"
                raise ValueError(msg)
            eps = 1e-12
            alphas_cumprod = alphas_cumprod.to(device=self.accelerator.device, dtype=torch.float32).clamp(min=eps)
            sigma_table = torch.sqrt((1 - alphas_cumprod) / alphas_cumprod)
            self._sigma_for_timesteps = sigma_table

        sigma_table = self._sigma_for_timesteps
        if sigma_table is None:
            msg = "Sigma table was not initialized"
            raise ValueError(msg)

        sigma_tensor: torch.Tensor = sigma_table

        mean = torch.tensor(self.config.timestep_lognormal_mean, device=self.accelerator.device, dtype=torch.float32)
        std = torch.tensor(self.config.timestep_lognormal_std, device=self.accelerator.device, dtype=torch.float32)
        lognormal = torch.distributions.LogNormal(mean, std)
        sampled_sigma = lognormal.sample((batch_size,))
        if sampled_sigma is None:
            msg = "Failed to sample sigma from lognormal distribution"
            raise ValueError(msg)
        # Find nearest timestep by sigma distance
        distance = torch.abs(sigma_tensor.view(1, -1) - sampled_sigma.view(-1, 1))
        return distance.argmin(dim=1).to(dtype=torch.long)

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        num_timesteps: int = self.noise_scheduler.config.get("num_train_timesteps", 1000)

        if self.config.timestep_bias_strategy == "uniform":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(
                0,
                num_timesteps,
                (batch_size,),
                device=self.accelerator.device,
                dtype=torch.long,
            )
        elif self.config.timestep_bias_strategy == "logit":
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps

            # Get m and s parameters from config or use defaults
            m = self.config.timestep_bias_m
            s = self.config.timestep_bias_s

            # Use these parameters for the logit distribution
            weights = logit_timestep_weights(
                num_timesteps,
                m=m,
                s=s,
                device=self.accelerator.device,
            )
            timesteps = torch.multinomial(weights, batch_size, replacement=True)
        elif self.config.timestep_bias_strategy == "lognormal":
            timesteps = self._sample_timesteps_lognormal(batch_size)
        else:
            msg = f"Unknown timestep bias strategy {self.config.timestep_bias_strategy}"
            raise ValueError(msg)

        # Use Counter to count sampled timesteps
        if hasattr(self, "timesteps_counter"):
            for t in timesteps.cpu().numpy().tolist():
                self.timesteps_counter[t] += 1

        return timesteps

    def apply_seed_settings(self, seed: int) -> None:
        logger.info("Setting seed to %s", seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _move_tensors_to_device_and_dtype(self, **tensors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Move tensors to the correct device and dtype efficiently."""
        result = {}
        for name, tensor in tensors.items():
            if tensor is not None:
                result[name] = tensor.to(self.device, dtype=self.weight_dtype)
        return result

    def _free_tensors(self, *tensors: torch.Tensor | None) -> None:
        """Free memory by deleting tensors."""
        for tensor in tensors:
            if tensor is not None:
                del tensor
        # Trigger garbage collection for CUDA tensors
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @contextmanager
    def use_ema_weights(self) -> Generator[None, None, None]:
        """Temporarily swap UNet weights to EMA for eval/save, then restore."""
        if not self.config.use_ema or self.ema_unet is None:
            yield
            return
        self.ema_unet.store(self.pipeline.unet.parameters())
        self.ema_unet.copy_to(self.pipeline.unet.parameters())
        try:
            yield
        finally:
            self.ema_unet.restore(self.pipeline.unet.parameters())

    def _sample_condition_dropout_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample a boolean mask for conditional dropout (CFG-style).
        True values indicate samples where text condition should be dropped.
        """
        prob = getattr(self.config, "condition_dropout_prob", 0.0)
        if prob <= 0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < prob

    def apply_condition_dropout_to_prompts(self, prompts_str: list[str]) -> list[str]:
        """Apply conditional dropout by replacing selected prompts with empty strings."""
        mask = self._sample_condition_dropout_mask(len(prompts_str), self.accelerator.device)
        if not mask.any():
            return prompts_str
        dropped = mask.cpu().tolist()
        updated_prompts = list(prompts_str)
        for idx, should_drop in enumerate(dropped):
            if should_drop:
                updated_prompts[idx] = ""
        return updated_prompts

    def _apply_vae_scaling(self, latents: torch.Tensor) -> torch.Tensor:
        """Apply VAE scaling factor to latents."""
        scaling_factor = self.pipeline.vae.config.get("scaling_factor", 1.0)
        return scaling_factor * latents

    def _prepare_training_tensors(self, img_latents: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare tensors for training: apply scaling, sample noise and timesteps."""
        # Sample noise and timesteps
        noise = self.sample_noise(img_latents)
        timesteps = self.sample_timesteps(img_latents.shape[0])

        # Get noisy latents
        img_noisy_latents = self.get_noisy_latents(img_latents, noise, timesteps)

        return img_latents, noise, timesteps, img_noisy_latents

    def init_gradient_checkpointing(self, models: list[torch.nn.Module]) -> None:
        for model in models:
            m: Any = unwrap_model(self.accelerator, model)
            if hasattr(m, "enable_gradient_checkpointing"):
                m.enable_gradient_checkpointing()
            elif hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
            else:
                logger.warning("cannot checkpointing!")

    def prepare_dataset(self, config: BaseConfig) -> DiffusionDataset:
        if config.dataset_path:
            dataset_root = Path(config.dataset_path)
        elif config.image_path:
            dataset_root = Path(config.image_path) / "metadata"
        else:
            msg = "Please specify the meta path in the config file."
            raise ValueError(msg)
        parquet_path = dataset_root / "metadata.parquet"
        latents_dir = dataset_root / "latents"
        tags_dir = dataset_root / "tags"
        with self.accelerator.main_process_first():
            if not self.accelerator.is_main_process:
                # Wait for the main process to finish preparing, then load the dataset.
                return DiffusionDataset.from_parquet(parquet_path)
            if config.image_path and config.skip_prepare_image is False:
                logger.info("Prepare image from %s", config.image_path)
                if not config.vae_path:
                    msg = "Please specify the vae_path in the config file."
                    raise ValueError(msg)

                latents_processor = LatentsGenerateProcessor(
                    vae_path=config.vae_path,
                    img_path=config.image_path,
                    target_path=str(latents_dir),
                    vae_dtype=self.vae_dtype,
                )

                latents_processor()

                tagging_processor = TaggingProcessor(img_path=config.image_path, target_path=str(tags_dir), num_workers=1)
                tagging_processor()

            if not parquet_path.exists():
                logger.info('Creating parquet file at "%s"', parquet_path)
                CreateParquetProcessor(target_dir=dataset_root)(max_workers=8)
            else:
                logger.info('found parquet file at "%s"', parquet_path)
        return DiffusionDataset.from_parquet(parquet_path)

    def freeze_model(self, models: Sequence[torch.nn.Module]) -> None:
        """
        Freeze all models except the ones in training_models.
        """
        for model in models:
            model.requires_grad_(False)
            model.eval()
        self.accelerator.wait_for_everyone()

    def get_noise_scheduler(self) -> DDPMScheduler:
        """Set up the noise scheduler"""
        scheduler_config_updates = {}
        prediction_type = getattr(self.config, "prediction_type", None)
        if prediction_type is not None:
            scheduler_config_updates["prediction_type"] = prediction_type
            scheduler_config_updates["timestep_spacing"] = "trailing"
        if getattr(self.config, "rescale_betas_zero_snr", False):
            scheduler_config_updates["rescale_betas_zero_snr"] = True
            scheduler_config_updates.setdefault("timestep_spacing", "trailing")
        if scheduler_config_updates:
            self.pipeline.scheduler.register_to_config(**scheduler_config_updates)

        # Create the noise scheduler from the pipeline's scheduler configuration
        noise_scheduler = DDPMScheduler.from_config(
            self.pipeline.scheduler.config,
        )
        if type(noise_scheduler) is not DDPMScheduler:
            msg = "Scheduler is not DDPMScheduler"
            raise TypeError(msg)
        logger.info("Noise scheduler config:")
        for key, value in noise_scheduler.config.items():
            logger.info("- %s: %s", key, value)
        return noise_scheduler  # type: ignore

    def execute_training_epoch(  # noqa: C901, PLR0912, PLR0915
        self,
        data_loader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        training_models: Sequence[torch.nn.Module],
        num_update_steps_per_epoch: int,
        n_total_steps: int,
    ) -> None:
        progress = get_progress()

        self.checkpointing_path = self.save_path / "state"
        self.global_steps_file = self.checkpointing_path / "global_steps"

        # Use Counter instead of a list to count timesteps
        self.timesteps_counter = Counter()

        # Attempt to load previous state for resuming training
        global_step = 0
        try:
            logger.info("Attempting to load checkpoint from: %s", self.checkpointing_path.as_posix())
            if not self.checkpointing_path.exists():
                logger.warning("Checkpoint directory does not exist: %s", self.checkpointing_path)
            elif not (self.checkpointing_path / "optimizer.bin").exists():
                logger.warning("Optimizer state file does not exist in checkpoint directory")
            elif not self.global_steps_file.exists():
                logger.warning("Global steps file does not exist: %s", self.global_steps_file)
            else:
                self.accelerator.load_state(self.checkpointing_path.as_posix())
                global_step = int(self.global_steps_file.read_text())
                self.global_step = global_step  # Update instance variable for input perturbation decay
                logger.info("Successfully loaded checkpoint at global step: %d", global_step)
        except Exception:
            logger.exception("Failed to load checkpoint:")
            # Instead of immediately deleting, try to backup and provide recovery options
            import shutil
            from datetime import datetime

            if self.checkpointing_path.exists():
                # Create a backup of the corrupted checkpoint
                timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
                backup_path = self.checkpointing_path.parent / f"{self.checkpointing_path.name}_corrupted_{timestamp}"
                try:
                    shutil.copytree(self.checkpointing_path, backup_path)
                    logger.info("Corrupted checkpoint backed up to: %s", backup_path)
                except Exception as backup_e:
                    logger.warning("Failed to backup corrupted checkpoint: %s", backup_e)

                # Only remove after successful backup
                try:
                    shutil.rmtree(self.checkpointing_path)
                    logger.warning("Removed corrupted checkpoint directory: %s", self.checkpointing_path)
                except Exception:
                    logger.exception("Failed to remove corrupted checkpoint:")

            global_step = 0
            logger.info("Starting training from scratch due to checkpoint corruption")

        # 1. Calculate the number of completed full epochs
        skipped_epoch = global_step // num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0

        # 2. Calculate the number of dataloader batches to skip in the current epoch
        num_batches_to_skip_in_current_epoch = 0
        if global_step > 0 and num_update_steps_per_epoch > 0:
            # Calculate the number of optimizer steps completed in the current epoch
            steps_in_current_epoch = global_step % num_update_steps_per_epoch
            # Calculate the number of dataloader batches corresponding to these optimizer steps
            num_batches_to_skip_in_current_epoch = steps_in_current_epoch * self.config.gradient_accumulation_steps

        if global_step != 0:
            # Adjust log information
            logger.info(
                "Resuming from global step %d (Epoch %d, skipping %d dataloader batches in current epoch)",
                global_step,
                skipped_epoch,
                num_batches_to_skip_in_current_epoch,
            )
            # (Optional) Calculate and record the total number of processed samples
            total_samples_processed = global_step * self.config.gradient_accumulation_steps * self.accelerator.num_processes * self.config.batch_size
            logger.info("Approximately %d total samples processed before resuming.", total_samples_processed)

        # 3. 使用正确的批次数调用 skip_first_batches
        skipped_data_loader = self.accelerator.skip_first_batches(
            data_loader,
            num_batches_to_skip_in_current_epoch,  # Use the correctly calculated value
        )

        logger.info("full_loader_length: %d", len(data_loader))

        if global_step == 0 and self.config.preview_before_training:
            self.generate_initial_preview()

        total_task = progress.add_task(
            "Total Progress",
            total=n_total_steps,
            completed=global_step,
        )

        if self.accelerator.is_main_process:
            progress.start()
        for epoch in range(skipped_epoch, self.config.n_epochs):
            self.train_loss = 0.0
            current_epoch_task = progress.add_task(
                f"Epoch {epoch + 1}",
                total=num_update_steps_per_epoch,
                completed=global_step % num_update_steps_per_epoch,
            )
            dl = skipped_data_loader if epoch == skipped_epoch else data_loader
            for _step, orig_batch in enumerate(dl):
                if not isinstance(orig_batch, DiffusionBatch):
                    msg = f"Expected DiffusionBatch, got something else. Got: {type(orig_batch)}"
                    raise TypeError(msg)
                batch = self.process_batch(orig_batch)

                # Free up memory from original batch if it contains large tensors
                del orig_batch

                with self.accelerator.accumulate(training_models):  # type: ignore
                    self.train_each_batch(batch)

                    # Free up processed batch memory
                    del batch

                    # Only clear CUDA cache occasionally to avoid performance issues
                    if hasattr(torch.cuda, "empty_cache") and self.accelerator.sync_gradients and global_step % 50 == 0:
                        torch.cuda.empty_cache()

                if self.accelerator.sync_gradients:
                    current_lr = lr_scheduler.get_last_lr()[0]

                    global_step += 1
                    self.global_step = global_step  # Update instance variable for input perturbation decay
                    log_data = {"train_loss": self.train_loss, "lr": current_lr}
                    self.accelerator.log(log_data, step=global_step)

                    self.train_loss = 0.0

                    if self.accelerator.is_main_process:
                        current_completed = global_step % num_update_steps_per_epoch
                        progress.update(total_task, completed=global_step, description=f"Epoch: {epoch}")
                        progress.update(current_epoch_task, completed=current_completed, description=f"Lr: {current_lr:.2e}")

                    # Check step-based conditions
                    should_save_step = self.config.save_every_n_steps and global_step % self.config.save_every_n_steps == 0 and global_step != 0
                    should_checkpoint_step = self.config.checkpoint_every_n_steps and global_step % self.config.checkpoint_every_n_steps == 0
                    should_preview_step = self.config.preview_every_n_steps and global_step % self.config.preview_every_n_steps == 0

                    if should_save_step:
                        self.saving_model(f"{self.config.model_name}-step{global_step}")
                    if should_checkpoint_step:
                        # 确保优化器状态能被正确保存
                        if not hasattr(self, "optimizer"):
                            logger.warning("Optimizer not found in self. Using optimizer from SDXL class.")
                            # 在没有 self.optimizer 的情况下，我们假设优化器已经被传递给了 accelerator

                        # 创建 checkpoint 目录
                        self.checkpointing_path.mkdir(parents=True, exist_ok=True)
                        # 保存状态
                        self.accelerator.save_state(self.checkpointing_path.as_posix())
                        # 保存当前步数
                        self.global_steps_file.write_text(str(global_step))
                        # 验证保存结果
                        if not (self.checkpointing_path / "optimizer.bin").exists():
                            logger.warning("Failed to save optimizer state. Please make sure the optimizer is correctly prepared with accelerator.")
                        else:
                            logger.info("Successfully saved checkpoint at global step: %d", global_step)
                    if should_preview_step:
                        self.generate_preview(f"{self.config.model_name}-step{global_step}", global_step)
            # Check epoch-based conditions
            should_save_epoch = self.config.save_every_n_epochs and (epoch % self.config.save_every_n_epochs == 0) and epoch != 0
            should_preview_epoch = self.config.preview_every_n_epochs and (epoch % self.config.preview_every_n_epochs == 0) and epoch != 0

            if should_save_epoch:
                self.saving_model(f"{self.config.model_name}-ep{epoch}")
            if should_preview_epoch:
                self.generate_preview(f"{self.config.model_name}-ep{epoch}", global_step)
            progress.remove_task(current_epoch_task)
        if self.accelerator.is_main_process:
            progress.stop()
        self.saving_model(f"{self.config.model_name}")

    @abstractmethod
    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> tuple[torch.Tensor, torch.Tensor]: ...

    @torch.no_grad()
    def generate_preview(self, filename: str, global_step: int = 0) -> None:  # noqa: C901, PLR0912, PLR0915
        # Release memory to ensure sufficient VRAM for preview generation
        free_memory()

        # Simple callback function to meet the pipeline interface requirements
        def callback_on_step_end(_pipe: StableDiffusionXLPipelineOutput, _step: int, _timestep: int, _kwargs: dict) -> dict:
            return _kwargs

        state = PartialState()
        self.accelerator.wait_for_everyone()

        # Split preview sample options across processes
        with state.split_between_processes(self.config.preview_sample_options) as sample_options:
            for sample_option in sample_options:
                if not isinstance(sample_option, SampleOptions):
                    msg = f"Expected SampleOption, got {type(sample_option)}"
                    raise TypeError(msg)
                hash_hex = get_sample_options_hash(sample_option)
                filename_with_hash = f"{filename}-{hash_hex}"

                # Check if preview file already exists, skip if it does
                path = self.save_path / "previews" / f"{filename_with_hash}.png"
                if path.exists():
                    logger.info("Preview file already exists, skipping: %s", path)
                    continue

                logger.info("Generating preview for %s", filename_with_hash)

                # Save original training-related settings and device information for later restoration
                original_training_mode = {}
                original_device = {}
                for name, model in [("unet", self.pipeline.unet), ("text_encoder", self.pipeline.text_encoder), ("vae", self.pipeline.vae)]:
                    original_training_mode[name] = model.training
                    original_device[name] = next(model.parameters()).device
                    model.eval()  # Switch to evaluation mode for inference

                # Use automatic mixed precision to generate preview images
                autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(self.accelerator.device.type)
                generator = torch.Generator(device=self.accelerator.device).manual_seed(sample_option.seed)

                with self.use_ema_weights(), autocast_ctx:
                    self.pipeline.to(self.accelerator.device)
                    # Use the configured VAE dtype but add NaN checking
                    self.pipeline.vae.to(dtype=self.vae_dtype)
                    try:
                        prompt_embeds, prompt_neg_embeds = self.get_preview_prompt_embeds(
                            sample_option.prompt,
                            sample_option.negative_prompt,
                            getattr(sample_option, "clip_skip", 2),
                        )
                        result = self.pipeline(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=prompt_neg_embeds,
                            num_inference_steps=sample_option.steps,
                            generator=generator,
                            callback_on_step_end=callback_on_step_end,  # type: ignore
                            width=sample_option.width,
                            height=sample_option.height,
                        )
                    except NotImplementedError:
                        logger.info("Using prompts directly for preview generation")
                        result = self.pipeline(
                            prompt=sample_option.prompt,
                            negative_prompt=sample_option.negative_prompt,
                            num_inference_steps=sample_option.steps,
                            generator=generator,
                            callback_on_step_end=callback_on_step_end,  # type: ignore
                            width=sample_option.width,
                            height=sample_option.height,
                        )

                    # Check if the pipeline output contains NaN values and handle them
                    if hasattr(result, "images") and result.images:
                        import numpy as np

                        image_array = np.array(result.images[0])
                        if np.any(np.isnan(image_array)) or np.any(np.isinf(image_array)):
                            logger.warning("Pipeline output contains NaN/Inf values, may cause conversion warnings")

                logger.info("Preview generated for %s", filename_with_hash)

                path.parent.mkdir(parents=True, exist_ok=True)

                # Clean up any potential NaN/Inf values in the image before saving
                image = result.images[0]

                # Convert to numpy array for NaN checking and cleanup
                import numpy as np
                from PIL import Image as PILImage

                image_array = np.array(image)

                # Check for and replace NaN/Inf values
                if np.any(np.isnan(image_array)) or np.any(np.isinf(image_array)):
                    logger.warning("Found NaN/Inf values in preview image, cleaning up...")
                    image_array = np.nan_to_num(image_array, nan=128.0, posinf=255.0, neginf=0.0)

                # Ensure values are in valid range [0, 255]
                image_array = np.clip(image_array, 0, 255).astype(np.uint8)

                # Convert back to PIL Image and save
                clean_image = PILImage.fromarray(image_array)
                clean_image.save(path)

                if self.config.log_with == "wandb":
                    import wandb

                    self.accelerator.log(
                        {
                            f"{hash_hex}": [wandb.Image(clean_image, caption=f"{sample_option.prompt}")],
                        },
                        step=global_step,
                    )

                # Release generated results promptly to reduce memory usage
                del result
                # Only clear cache after preview generation, not after each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Restore the training mode of the models
                for name, model in [("unet", self.pipeline.unet), ("text_encoder", self.pipeline.text_encoder), ("vae", self.pipeline.vae)]:
                    if model.training != original_training_mode[name]:
                        model.train(original_training_mode[name])
                    if next(model.parameters()).device != original_device[name]:
                        model.to(original_device[name])

        # Ensure all processes have completed preview generation
        free_memory()
        self.accelerator.wait_for_everyone()

    def saving_model(self, filename: str) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_context = self.use_ema_weights() if self.config.use_ema else nullcontext()
            with save_context:
                if self.config.mode in ("lora", "lokr", "loha"):
                    self.save_lora_model(filename)
                else:
                    self.save_full_finetune_model(filename)

    def _create_lora_metadata(self) -> dict[str, str]:
        """Create metadata for LoRA models."""
        if self.lycoris_model is None:
            msg = "LyCORIS model is not initialized."
            raise ValueError(msg)

        # Create essential metadata for compatibility
        metadata = {
            # Core LoRA parameters - essential for loading
            "ss_network_dim": str(getattr(self.lycoris_model, "lora_dim", self.config.lora_dim)),
            "ss_network_alpha": str(getattr(self.lycoris_model, "alpha", self.config.lora_alpha)),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": f'{{"algo": "{self.config.mode}"}}',
            # Basic info
            "format": "pt",
            "ss_base_model_version": "sd_v1" if "sd15" in str(self.config.model_path).lower() else "sdxl_v1",
            "ss_training_comment": f"LyCORIS {self.config.mode.upper()} training",
            "ss_resolution": "512,512" if "sd15" in str(self.config.model_path).lower() else "1024,1024",
        }

        # Add algorithm-specific metadata
        if self.config.mode == "lokr":
            metadata["ss_lokr_factor"] = str(self.config.lokr_factor)

        return metadata

    def _save_lycoris_format_lora(self, filename: str, metadata: dict[str, str]) -> None:
        """Save LyCORIS format LoRA."""
        out_path = self.save_path / f"{filename}.safetensors"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.lycoris_model is None:
            msg = "LyCORIS model is not initialized."
            raise ValueError(msg)

        # Save LyCORIS weights with metadata
        self.lycoris_model.save_weights(
            self.save_path / f"{filename}.safetensors",
            dtype=self.save_dtype,
            metadata=metadata,
        )

    def save_lora_model(self, filename: str) -> None:
        """Save LoRA model in both LyCORIS and diffusers formats."""
        # Create metadata first
        metadata = self._create_lora_metadata()

        # Save LyCORIS format
        self._save_lycoris_format_lora(filename, metadata)

        # Also create a diffusers-compatible version if possible
        self._save_diffusers_compatible_lora(filename, metadata)

    def _save_diffusers_compatible_lora(self, filename: str, metadata: dict[str, str]) -> None:
        """Save a diffusers-compatible LoRA that can be loaded with pipeline.load_lora_weights()"""
        try:
            # Only attempt for LoRA mode
            if self.config.mode == "lora":
                from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
                from peft.utils import get_peft_model_state_dict

                diffusers_path = self.save_path / f"{filename}_diffusers.safetensors"

                # Get the current state dict directly from the model
                if hasattr(self, "lycoris_model") and self.lycoris_model is not None:
                    # Method 1: Try to get PEFT state dict if available
                    try:
                        peft_state_dict = get_peft_model_state_dict(self.lycoris_model)
                        diffusers_state_dict = convert_state_dict_to_diffusers(peft_state_dict)

                        # Save using pipeline's method
                        unet_lora_layers = {}
                        text_encoder_lora_layers = {}

                        for key, value in diffusers_state_dict.items():
                            if key.startswith("text_encoder"):
                                text_encoder_lora_layers[key] = value
                            else:
                                unet_lora_layers[key] = value

                        self.pipeline.save_lora_weights(
                            save_directory=str(self.save_path),
                            unet_lora_layers=unet_lora_layers if unet_lora_layers else None,
                            text_encoder_lora_layers=text_encoder_lora_layers if text_encoder_lora_layers else None,
                            weight_name=f"{filename}_diffusers.safetensors",
                        )

                        logger.info("Created diffusers-compatible LoRA using PEFT conversion at %s", diffusers_path)
                    except Exception as e1:
                        logger.debug("PEFT conversion failed: %s", e1)
                    else:
                        return

                # Method 2: Manual key conversion fallback
                logger.info("Attempting manual LyCORIS to diffusers conversion...")
                from safetensors.torch import load_file, save_file

                # Load the just-saved LyCORIS file
                lycoris_file = self.save_path / f"{filename}.safetensors"
                lycoris_state_dict = load_file(lycoris_file)

                # Convert LyCORIS keys to diffusers format
                diffusers_state_dict = {}
                for key, value in lycoris_state_dict.items():
                    if key.startswith("lycoris_"):
                        # Remove lycoris_ prefix and convert format
                        new_key = key.replace("lycoris_", "")
                        # Convert underscore format to dot format for diffusers
                        new_key = new_key.replace("_", ".", 6)  # Convert first 6 underscores to dots
                        diffusers_state_dict[new_key] = value
                    else:
                        diffusers_state_dict[key] = value

                # Add diffusers-specific metadata
                diffusers_metadata = metadata.copy()
                diffusers_metadata["library_name"] = "diffusers"

                # Save the converted file
                save_file(diffusers_state_dict, diffusers_path, metadata=diffusers_metadata)

                logger.info("Created diffusers-compatible LoRA using manual conversion at %s", diffusers_path)

        except Exception as e:
            logger.warning("Could not create diffusers-compatible version: %s", e)
            # This is not a fatal error, continue with the original LyCORIS save

    def save_full_finetune_model(self, filename: str) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.pipeline.to(self.save_dtype)
        self.pipeline.save_pretrained(self.save_path / f"{filename}")
        self.pipeline.to(self.weight_dtype)

    def create_prompts_str(self, batch: DiffusionBatch) -> list[str]:
        prompts = []
        caption_dropout_ratio = self.config.caption_dropout
        all_tags_dropout_ratio = self.config.all_tags_dropout
        single_tag_dropout_ratio = self.config.single_tag_dropout
        shuffle_tags = self.config.shuffle_tags

        for caption, tags in zip(batch.caption, batch.tags, strict=True):
            # Decide if the caption should be dropped
            true_caption = "" if random.random() < caption_dropout_ratio else caption

            # Decide if all tags should be dropped or process individual tag dropout
            if random.random() < all_tags_dropout_ratio:
                true_tags = []
            elif single_tag_dropout_ratio > 0:
                true_tags = [tag for tag in tags if random.random() >= single_tag_dropout_ratio]
            else:
                true_tags = tags

            # Shuffle tags if necessary
            if shuffle_tags:
                random.shuffle(true_tags)

            # Create prompt string
            if true_caption and true_tags:
                prompt = f"{true_caption}, " + ", ".join(true_tags)
            elif true_caption:
                prompt = true_caption
            else:
                prompt = ", ".join(true_tags)

            prompts.append(prompt)

        return prompts

    def get_ema(self, model: torch.nn.Module, config: dict, *, decay: float) -> EMAModel:
        """Initialize Exponential Moving Average for the model if enabled in config."""
        ema_model = EMAModel(
            parameters=model.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=config,
            decay=decay,
        )
        ema_model.to(self.device)
        return ema_model
