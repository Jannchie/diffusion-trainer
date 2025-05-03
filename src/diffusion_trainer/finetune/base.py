import logging
import math
import random
from abc import abstractmethod
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import compute_snr, free_memory

from diffusion_trainer.config import BaseConfig, SampleOptions
from diffusion_trainer.dataset.dataset import DiffusionBatch, DiffusionDataset
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import LatentsGenerateProcessor
from diffusion_trainer.dataset.processors.tagging_processor import TaggingProcessor
from diffusion_trainer.finetune.utils import (
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


class BaseBatch: ...


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

        self.accelerator = prepare_accelerator(
            config.gradient_accumulation_steps,
            self.mixed_precision,
            config.log_with,
        )
        self.device = self.accelerator.device

        self.pipeline = self.get_pipeline()
        self.lycoris_model: LycorisNetwork | None = None
        self.noise_scheduler: DDPMScheduler = self.get_noise_scheduler()
        self.train_loss = 0.0

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
        if self.config.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean()
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.config.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            if self.noise_scheduler.config.get("prediction_type") == "epsilon":
                epsilon = 1e-8
                snr_safe = snr + epsilon
                mse_loss_weights = mse_loss_weights / (snr_safe + epsilon)
            elif self.noise_scheduler.config.get("prediction_type") == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return loss

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

    def sample_timesteps(self, batch_size: int) -> torch.IntTensor:
        num_timesteps: int = self.noise_scheduler.config.get("num_train_timesteps", 1000)
        if self.config.timestep_bias_strategy == "uniform":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.accelerator.device)
        elif self.config.timestep_bias_strategy == "logit":
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = logit_timestep_weights(num_timesteps, device=self.accelerator.device)
            timesteps = torch.multinomial(weights, batch_size, replacement=True).int()
        else:
            msg = f"Unknown timestep bias strategy {self.config.timestep_bias_strategy}"
            raise ValueError(msg)
        return timesteps  # type: ignore

    def apply_seed_settings(self, seed: int) -> None:
        logger.info("Setting seed to %s", seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def init_gradient_checkpointing(self, accelerator: Accelerator, models: list[torch.nn.Module]) -> None:
        for model in models:
            m: Any = unwrap_model(accelerator, model)
            if hasattr(m, "enable_gradient_checkpointing"):
                m.enable_gradient_checkpointing()
            elif hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
            else:
                logger.warning("cannot checkpointing!")

    def prepare_dataset(self, accelerator: Accelerator, config: BaseConfig) -> DiffusionDataset:
        if config.meta_path:
            meta_path = Path(config.meta_path)
        elif config.image_path:
            meta_path = Path(config.image_path) / "metadata"
        else:
            msg = "Please specify the meta path in the config file."
            raise ValueError(msg)
        parquet_path = meta_path / "metadata.parquet"
        with accelerator.main_process_first():
            if not accelerator.is_main_process:
                # Wait for the main process to finish preparing, then load the dataset.
                return DiffusionDataset.from_parquet(parquet_path)
            if config.image_path and config.skip_prepare_image is False:
                logger.info("Prepare image from %s", config.image_path)
                if not config.vae_path:
                    msg = "Please specify the vae_path in the config file."
                    raise ValueError(msg)

                vae_dtype = str_to_dtype(config.vae_dtype)
                latents_processor = LatentsGenerateProcessor(
                    vae_path=config.vae_path,
                    img_path=config.image_path,
                    target_path=config.meta_path,
                    vae_dtype=vae_dtype,
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

    def freeze_model(self, accelerator: Accelerator, models: Sequence[torch.nn.Module]) -> None:
        """
        Freeze all models except the ones in training_models.
        """
        for model in models:
            model.requires_grad_(False)
            model.eval()
        accelerator.wait_for_everyone()

    def get_noise_scheduler(self) -> DDPMScheduler:
        """设置噪声调度器"""
        # 如果配置中指定了预测类型，则使用该类型
        if hasattr(self.config, "prediction_type") and self.config.prediction_type == "v_prediction":
            self.pipeline.scheduler.register_to_config(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )

        # 从pipeline的调度器配置中创建噪声调度器
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

    def execute_training_epoch(  # noqa: C901, PLR0912, PLR0913, PLR0915
        self,
        accelerator: Accelerator,
        data_loader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        training_models: Sequence[torch.nn.Module],
        num_update_steps_per_epoch: int,
        n_total_steps: int,
    ) -> None:
        progress = get_progress()

        self.checkpointing_path = self.save_path / "state"
        self.global_steps_file = self.checkpointing_path / "global_steps"

        try:
            accelerator.load_state(self.checkpointing_path.as_posix())
            global_step = int(self.global_steps_file.read_text())
        except Exception:
            global_step = 0

        skiped_epoch = math.floor(global_step / num_update_steps_per_epoch)
        # no need to divide by gradient_accumulation_steps
        skiped_batch = global_step * accelerator.num_processes * self.config.batch_size
        if global_step != 0:
            logger.info(
                "skiping %d global steps (%d batches)",
                global_step,
                skiped_batch,
            )
        skiped_data_loader = accelerator.skip_first_batches(data_loader, skiped_batch % len(data_loader))

        logger.info("full_loader_length: %d", len(data_loader))
        logger.info("skiped_loader_length: %d", len(skiped_data_loader))

        total_task = progress.add_task(
            "Total Progress",
            total=n_total_steps,
            completed=global_step,
        )

        if accelerator.is_main_process:
            progress.start()
        for epoch in range(skiped_epoch, self.config.n_epochs):
            self.train_loss = 0.0
            current_epoch_task = progress.add_task(
                f"Epoch {epoch + 1}",
                total=num_update_steps_per_epoch,
                completed=global_step % num_update_steps_per_epoch,
            )
            dl = skiped_data_loader if epoch == skiped_epoch else data_loader
            for _step, orig_batch in enumerate(dl):
                if not isinstance(orig_batch, DiffusionBatch):
                    msg = f"Expected DiffusionBatch, got something else. Got: {type(orig_batch)}"
                    raise TypeError(msg)
                batch = self.process_batch(orig_batch)

                with accelerator.accumulate(training_models):
                    self.train_each_batch(batch)

                if accelerator.sync_gradients:
                    current_lr = lr_scheduler.get_last_lr()[0]
                    # if self.ema_unet:
                    #     self.ema_unet.step(self.pipeline.unet.parameters())

                    global_step += 1
                    accelerator.log({"train_loss": self.train_loss, "lr": current_lr}, step=global_step)
                    self.train_loss = 0.0

                    if accelerator.is_main_process:
                        current_completed = global_step % num_update_steps_per_epoch
                        progress.update(total_task, completed=global_step, description=f"Epoch: {epoch + 1}")
                        progress.update(current_epoch_task, completed=current_completed, description=f"Lr: {current_lr:.2e}")

                    if self.config.save_every_n_steps and global_step % self.config.save_every_n_steps == 0 and global_step != 0:
                        self.saving_model(accelerator, f"{self.config.model_name}-step{global_step}")
                    if self.config.checkpoint_every_n_steps and global_step % self.config.checkpoint_every_n_steps == 0:
                        accelerator.save_state(self.checkpointing_path.as_posix())
                        self.global_steps_file.write_text(str(global_step))
                    if self.config.preview_every_n_steps and global_step % self.config.preview_every_n_steps == 0:
                        self.generate_preview(accelerator, f"{self.config.model_name}-step{global_step}", global_step)
            if self.config.save_every_n_epochs and epoch % self.config.save_every_n_epochs == 0 and epoch != 0:
                self.saving_model(accelerator, f"{self.config.model_name}-ep{epoch + 1}")
            if self.config.preview_every_n_epochs and epoch % self.config.preview_every_n_epochs == 0:
                self.generate_preview(accelerator, f"{self.config.model_name}-ep{epoch + 1}", global_step)
            progress.remove_task(current_epoch_task)
        if accelerator.is_main_process:
            progress.stop()
        self.saving_model(accelerator, f"{self.config.model_name}")

    @torch.no_grad()
    def generate_preview(self, accelerator: Accelerator, filename: str, global_step: int = 0) -> None:
        # 显式释放内存
        free_memory()

        def callback_on_step_end(_pipe: StableDiffusionXLPipelineOutput, _step: int, _timestep: int, _kwargs: dict) -> dict:
            return {}

        state = PartialState()
        accelerator.wait_for_everyone()
        with state.split_between_processes(self.config.preview_sample_options) as sample_options:
            for sample_option in sample_options:
                if not isinstance(sample_option, SampleOptions):
                    msg = f"Expected SampleOption, got {type(sample_option)}"
                    raise TypeError(msg)
                hash_hex = get_sample_options_hash(sample_option)
                filename_with_hash = f"{filename}-{hash_hex}"
                logger.info("Generating preview for %s", filename_with_hash)

                # 切换为eval模式并移动到CPU如果可能，以节省显存
                original_device = {}
                for name, model in [("unet", self.pipeline.unet), ("text_encoder", self.pipeline.text_encoder)]:
                    original_device[name] = model.device
                    model.eval()

                # 使用自动混合精度生成预览图像
                autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(accelerator.device.type)
                generator = torch.Generator(device=accelerator.device).manual_seed(sample_option.seed)

                with autocast_ctx:
                    # 确保移回设备进行推理
                    self.pipeline.to(accelerator.device)
                    # 设置较小的height和width，减少显存使用
                    inference_steps = min(sample_option.steps, 25)  # 减少步数
                    result = self.pipeline(
                        prompt=sample_option.prompt,
                        negative_prompt=sample_option.negative_prompt,
                        num_inference_steps=inference_steps,
                        generator=generator,
                        callback_on_step_end=callback_on_step_end,  # type: ignore
                    )
                logger.info("Preview generated for %s", filename_with_hash)

                path = (Path(self.save_path) / "previews" / filename_with_hash).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                result.images[0].save(path)

                # 记录到wandb（如果启用）
                if self.config.log_with == "wandb":
                    import wandb

                    accelerator.log(
                        {
                            f"{hash_hex}": [wandb.Image(result.images[0], caption=f"{sample_option.prompt}")],
                        },
                        step=global_step,
                    )

                # 清理预览相关的内存
                del result
                torch.cuda.empty_cache()

                # 恢复模型到原始设备
                for name, model in [("unet", self.pipeline.unet), ("text_encoder", self.pipeline.text_encoder)]:
                    if original_device[name] != model.device:
                        model.to(original_device[name])

        # 最终清理
        free_memory()
        accelerator.wait_for_everyone()

    def saving_model(self, accelerator: Accelerator, filename: str) -> None:
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
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
