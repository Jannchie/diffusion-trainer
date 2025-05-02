import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import compute_snr
from torch.utils.data import DataLoader
from transformers.models.clip import CLIPTextModel

from diffusion_trainer.config import BaseConfig, SD15Config
from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch
from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.utils import (
    TrainableModel,
    get_trainable_parameter_dicts,
    initialize_optimizer,
    load_sd15_pipeline,
    prepare_accelerator,
)
from diffusion_trainer.finetune.utils.lora import apply_lora_config
from diffusion_trainer.utils.timestep_weights import logit_timestep_weights

if TYPE_CHECKING:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline
    from lycoris import LycorisNetwork


class SD15Models(NamedTuple):
    unet: UNet2DConditionModel
    text_encoder: CLIPTextModel


@dataclass
class SD15Batch:
    img_latents: torch.Tensor
    prompt_embeds: torch.Tensor


class SD15Tuner(BaseTuner):
    """
    A class to finetune Stable Diffusion 1.5 models.
    """

    @staticmethod
    def from_config(config: BaseConfig) -> "SD15Tuner":
        """Create a new instance from a configuration dictionary."""
        if not isinstance(config, SD15Config):
            msg = f"Expected SD15Config, got {type(config)}"
            raise TypeError(msg)
        return SD15Tuner(config)

    def get_pipeline(self) -> "StableDiffusionPipeline":
        return load_sd15_pipeline(self.config.model_path, self.weight_dtype)

    def __init__(self, config: SD15Config) -> None:
        super().__init__(config)
        self.config = config
        self.logger = logging.getLogger(__name__)

        self.apply_seed_settings(config.seed)

        self.accelerator = prepare_accelerator(self.config.gradient_accumulation_steps, self.mixed_precision, self.config.log_with)

        # 使用 torch.cuda.empty_cache() 确保在加载模型前清理GPU内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.models = SD15Models(
            unet=self.pipeline.unet,
            text_encoder=self.pipeline.text_encoder,
        )
        self.models.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.models.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)
        self.lycoris_model: LycorisNetwork | None = None

        # 确认梯度检查点启用
        if self.config.gradient_checkpointing:
            self.init_gradient_checkpointing(self.accelerator, list(self.models))

        for key, value in config.__dict__.items():
            self.logger.info("%s: %s", key, value)

    def train(self) -> None:
        """
        Start the training process.
        """
        self.accelerator.wait_for_everyone()
        dataset = self.prepare_dataset(self.accelerator, self.config)
        if self.accelerator.is_main_process:
            dataset.print_bucket_info()

        trainable_models_with_lr = self.configure_trainable_models(self.models)
        trainable_models = [model.model for model in trainable_models_with_lr]
        for model in trainable_models:
            model.train()
        freeze_models = [model for model in self.models if model not in trainable_models]
        self.freeze_model(self.accelerator, freeze_models)

        self.trainable_parameters_dicts = get_trainable_parameter_dicts(self.accelerator, trainable_models_with_lr)
        self.optimizer = initialize_optimizer(self.config.optimizer, self.trainable_parameters_dicts)

        if self.config.prediction_type == "v_prediction":
            self.pipeline.scheduler.register_to_config(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )

        # https://huggingface.co/docs/diffusers/api/schedulers/ddim
        noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.pipeline.scheduler.config,
        )  # type: ignore
        self.logger.info("Noise scheduler config:")
        for key, value in noise_scheduler.config.items():
            self.logger.info("- %s: %s", key, value)
        self.noise_scheduler = noise_scheduler

        sampler = BucketBasedBatchSampler(dataset, self.config.batch_size)
        with self.accelerator.main_process_first():
            data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=dataset.collate_fn)
        data_loader = self.accelerator.prepare(data_loader)

        num_update_steps_per_epoch = math.ceil(len(data_loader) / self.config.gradient_accumulation_steps / self.accelerator.num_processes)
        n_total_steps = self.config.n_epochs * num_update_steps_per_epoch
        lr_scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.optimizer_warmup_steps * self.accelerator.num_processes,
            num_training_steps=n_total_steps * self.accelerator.num_processes,
            num_cycles=self.config.optimizer_num_cycles,
        )
        self.lr_scheduler = self.accelerator.prepare(lr_scheduler)
        self.accelerator.init_trackers(f"diffusion-trainer-{self.config.mode}", config=self.config.__dict__)

        self.execute_training_epoch(
            self.accelerator,
            data_loader,
            lr_scheduler,
            training_models=trainable_models,
            num_update_steps_per_epoch=num_update_steps_per_epoch,
            n_total_steps=n_total_steps,
        )

    def configure_trainable_models(self, models: SD15Models) -> list[TrainableModel]:
        trainable_models_with_lr = []
        if self.config.mode == "full-finetune":
            # 如果是全量微调，根据配置文件中的学习率，将模型的参数设置为可训练
            if self.config.unet_lr:
                trainable_models_with_lr.append(TrainableModel(model=models.unet, lr=self.config.unet_lr))
            if self.config.text_encoder_lr:
                trainable_models_with_lr.append(TrainableModel(model=models.text_encoder, lr=self.config.text_encoder_lr))

        elif self.config.mode in ("lora", "lokr", "loha"):
            # 如果是高效微调，则模型本身无需训练，只需训练Lora模型。
            lycoris_model = apply_lora_config(self.config.mode, models.unet)
            trainable_models_with_lr.append(TrainableModel(model=lycoris_model, lr=self.config.unet_lr))
            self.lycoris_model = lycoris_model

        else:
            msg = f"Unknown mode {self.config.mode}"
            raise ValueError(msg)
        return trainable_models_with_lr

    @torch.no_grad()
    def process_batch(self, batch: DiffusionBatch) -> SD15Batch:
        prompts_str = self.create_prompts_str(batch)
        prompt_embeds = self.get_prompt_embeds(prompts_str)

        return SD15Batch(
            img_latents=batch.img_latents,
            prompt_embeds=prompt_embeds,
        )

    def train_each_batch(self, batch: SD15Batch) -> None:
        # 使用统一的精度
        img_latents = self.pipeline.vae.config.get("scaling_factor", 0) * batch.img_latents.to(self.accelerator.device)
        prompt_embeds = batch.prompt_embeds.to(self.accelerator.device)

        # 只保留必须的变量，及时释放不再需要的内存
        noise = self.sample_noise(img_latents)
        timesteps = self.sample_timesteps(img_latents.shape[0])

        # 统一使用指定的dtype，避免精度转换消耗额外内存
        img_noisy_latents = self.noise_scheduler.add_noise(
            img_latents.to(self.weight_dtype),
            noise.to(self.weight_dtype),
            timesteps,
        )

        # 释放不再需要的变量
        del noise

        model_pred = self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds)
        target = self.get_pred_target(img_latents, self.sample_noise(img_latents), timesteps, model_pred)

        # 释放不再需要的变量
        del img_latents, img_noisy_latents

        loss = self.get_loss(timesteps, model_pred, target)

        # 释放不再需要的变量
        del model_pred, target, timesteps

        if torch.isnan(loss):
            self.logger.info("Loss is NaN.")
            msg = "Loss is NaN."
            raise ValueError(msg)

        avg_loss = self.accelerator.gather(loss.repeat(self.config.batch_size)).mean()  # type: ignore
        self.train_loss += avg_loss.item() / self.config.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # 释放不再需要的变量
        del loss, avg_loss

        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
            params_to_clip = self.pipeline.unet.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        # ref: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)

        # 定期清理缓存
        if self.accelerator.sync_gradients and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_pred(
        self,
        img_noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        return self.pipeline.unet(
            img_noisy_latents,
            timesteps,
            prompt_embeds,
            return_dict=False,
        )[0]

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

    def get_prompt_embeds(self, prompts_str: list[str]) -> torch.Tensor:
        text_inputs = self.pipeline.tokenizer(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs["input_ids"].to(self.accelerator.device)
        prompt_embeds_output = self.pipeline.text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )
        # use the second to last hidden state as the prompt embedding
        return prompt_embeds_output.hidden_states[-2]

    def sample_noise(self, img_latents: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img_latents)
        if self.config.noise_offset:
            # Add noise to the image latents
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.config.noise_offset * torch.randn(
                (img_latents.shape[0], img_latents.shape[1], 1, 1),
                device=img_latents.device,
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
