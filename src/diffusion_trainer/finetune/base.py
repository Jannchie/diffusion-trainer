import logging
import random
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from accelerate import PartialState
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel, compute_snr, free_memory

from diffusion_trainer.config import BaseConfig, SampleOptions
from diffusion_trainer.dataset.dataset import DiffusionBatch, DiffusionDataset
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import LatentsGenerateProcessor
from diffusion_trainer.dataset.processors.tagging_processor import TaggingProcessor
from diffusion_trainer.finetune.utils import (
    compute_sqrt_inv_snr_weights,
    get_sample_options_hash,
    prepare_accelerator,
    str_to_dtype,
    unwrap_model,
)
from diffusion_trainer.shared import get_progress
from diffusion_trainer.utils.timestep_weights import logit_timestep_weights

if TYPE_CHECKING:
    from lycoris import LycorisNetwork

logger = logging.getLogger("diffusion_trainer")


class BaseTuner:
    @staticmethod
    def from_config(config: BaseConfig) -> "BaseTuner":
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

        self.accelerator = prepare_accelerator(
            config.gradient_accumulation_steps,
            self.mixed_precision,
            config.log_with,
        )
        self.device = self.accelerator.device

        self.pipeline = self.get_pipeline()
        self.lycoris_model: LycorisNetwork | None = None
        self.noise_scheduler: DDPMScheduler = self.get_noise_scheduler()
        self.all_snr = compute_snr(self.noise_scheduler, torch.arange(0, self.noise_scheduler.config.num_train_timesteps, dtype=torch.long)).to(self.device)  # type: ignore
        self.train_loss = 0.0

    def generate_initial_preview(self) -> None:
        """Generate preview images before training starts if configured to do so."""
        logger.info("Generating preview before training starts")
        self.generate_preview(f"{self.config.model_name}-before-training", 0)

    @abstractmethod
    def train(self) -> None:
        """
        Start the training process.
        """
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

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
            snr = self.all_snr[timesteps]
            mse_loss_weights = torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            # Adjust weights according to prediction_type
            if self.noise_scheduler.config.get("prediction_type") == "epsilon":
                epsilon = 1e-8
                mse_loss_weights = mse_loss_weights / (snr + epsilon)
            elif self.noise_scheduler.config.get("prediction_type") == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)
            weights = weights * mse_loss_weights  # Further multiply by SNR weights

        # 4. Apply the final weights and compute the mean loss
        return (loss_per_sample * weights).mean()

    def get_pred_target(
        self,
        img_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
        model_pred: torch.Tensor,
    ) -> torch.Tensor:
        noise_scheduler = self.noise_scheduler
        if noise_scheduler.config.get("prediction_type") == "epsilon":
            target = noise
        elif noise_scheduler.config.get("prediction_type") == "v_prediction":
            target = noise_scheduler.get_velocity(img_latents, noise, timesteps)
        elif noise_scheduler.config.get("prediction_type") == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = img_latents
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            msg = f"Unknown prediction type {noise_scheduler.config.get('prediction_type')}"
            raise ValueError(msg)
        return target

    def sample_noise(self, latents: torch.Tensor) -> torch.Tensor:
        """Sample noise that will be added to the latents."""
        noise = torch.randn_like(latents)
        if hasattr(self.config, "noise_offset") and self.config.noise_offset:
            # Add noise to the image latents
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.config.noise_offset * torch.randn(
                (latents.shape[0], latents.shape[1], 1, 1),
                device=latents.device,
            )
        return noise

    def get_noisy_latents(self, latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.IntTensor) -> torch.Tensor:
        """Get noisy latents"""
        if self.config.input_perturbation > 0:
            # Apply perturbation only if the configuration exists and the value is non-zero
            noise = noise + self.config.input_perturbation * torch.randn_like(noise)
        return self.noise_scheduler.add_noise(latents, noise, timesteps)

    def sample_timesteps(self, batch_size: int) -> torch.IntTensor:
        num_timesteps: int = self.noise_scheduler.config.get("num_train_timesteps", 1000)
        if self.config.timestep_bias_strategy == "uniform":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.accelerator.device)
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
            timesteps = torch.multinomial(weights, batch_size, replacement=True).int()
        else:
            msg = f"Unknown timestep bias strategy {self.config.timestep_bias_strategy}"
            raise ValueError(msg)

        # Use Counter to count sampled timesteps
        if hasattr(self, "timesteps_counter"):
            for t in timesteps.cpu().numpy().tolist():
                self.timesteps_counter[t] += 1

        return timesteps  # type: ignore

    def apply_seed_settings(self, seed: int) -> None:
        logger.info("Setting seed to %s", seed)
        random.seed(seed)
        torch.manual_seed(seed)

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
        if config.meta_path:
            meta_path = Path(config.meta_path)
        elif config.image_path:
            meta_path = Path(config.image_path) / "metadata"
        else:
            msg = "Please specify the meta path in the config file."
            raise ValueError(msg)
        parquet_path = meta_path / "metadata.parquet"
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
                    target_path=config.meta_path,
                    vae_dtype=self.vae_dtype,
                )

                latents_processor()

                tagging_processor = TaggingProcessor(img_path=config.image_path, target_path=config.meta_path, num_workers=1)
                tagging_processor()

            if not parquet_path.exists():
                logger.info('Creating parquet file at "%s"', parquet_path)
                CreateParquetProcessor(target_dir=config.meta_path)(max_workers=8)
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
        # Use the specified prediction type from the configuration if provided
        if hasattr(self.config, "prediction_type") and self.config.prediction_type == "v_prediction":
            self.pipeline.scheduler.register_to_config(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )

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
        from collections import Counter

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
                logger.info("Successfully loaded checkpoint at global step: %d", global_step)
        except Exception as e:
            logger.warning("Failed to load checkpoint: %s", e)
            global_step = 0

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

                with self.accelerator.accumulate(training_models):
                    self.train_each_batch(batch)

                    # Free up processed batch memory
                    del batch

                    # Manually trigger garbage collection after each batch
                    if hasattr(torch.cuda, "empty_cache") and self.accelerator.sync_gradients:
                        torch.cuda.empty_cache()

                if self.accelerator.sync_gradients:
                    current_lr = lr_scheduler.get_last_lr()[0]

                    global_step += 1
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
    def generate_preview(self, filename: str, global_step: int = 0) -> None:
        # Release memory to ensure sufficient VRAM for preview generation
        free_memory()

        # Simple callback function to meet the pipeline interface requirements
        def callback_on_step_end(_pipe: StableDiffusionXLPipelineOutput, _step: int, _timestep: int, _kwargs: dict) -> dict:
            return {}

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

                with autocast_ctx:
                    self.pipeline.to(self.accelerator.device)
                    self.pipeline.vae.to(dtype=self.vae_dtype)
                    inference_steps = min(sample_option.steps, 25)
                    try:
                        prompt_embeds, prompt_neg_embeds = self.get_preview_prompt_embeds(
                            sample_option.prompt,
                            sample_option.negative_prompt,
                            getattr(sample_option, "clip_skip", 2),
                        )
                        result = self.pipeline(
                            prompt_embeds=prompt_embeds,
                            negative_prompt_embeds=prompt_neg_embeds,
                            num_inference_steps=inference_steps,
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
                            num_inference_steps=inference_steps,
                            generator=generator,
                            callback_on_step_end=callback_on_step_end,  # type: ignore
                            width=sample_option.width,
                            height=sample_option.height,
                        )

                logger.info("Preview generated for %s", filename_with_hash)

                path = self.save_path / "previews" / f"{filename_with_hash}.png"
                path.parent.mkdir(parents=True, exist_ok=True)

                result.images[0].save(path)

                if self.config.log_with == "wandb":
                    import wandb

                    self.accelerator.log(
                        {
                            f"{hash_hex}": [wandb.Image(result.images[0], caption=f"{sample_option.prompt}")],
                        },
                        step=global_step,
                    )

                # Release generated results promptly to reduce memory usage
                del result
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
            if self.config.mode in ("lora", "lokr", "loha"):
                self.save_lora_model(filename)
            else:
                self.save_full_finetune_model(filename)

    def save_lora_model(self, filename: str) -> None:
        out_path = self.save_path / f"{filename}.safetensors"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if self.lycoris_model is None:
            msg = "LyCORIS model is not initialized."
            raise ValueError(msg)
        self.lycoris_model.save_weights(self.save_path / f"{filename}.safetensors", dtype=self.save_dtype, metadata={})

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

    def get_ema(self, model: torch.nn.Module, config: dict) -> EMAModel:
        """Initialize Exponential Moving Average for the model if enabled in config."""
        ema_model = EMAModel(
            parameters=model.parameters(),
            model_cls=UNet2DConditionModel,
            model_config=config,
        )
        ema_model.to(self.device)
        return ema_model
