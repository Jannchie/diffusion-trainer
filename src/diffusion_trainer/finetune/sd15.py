import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
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

if TYPE_CHECKING:
    from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


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

        # 设置噪声调度器
        self.get_noise_scheduler()

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
        # Convert all tensors to correct precision at once to reduce memory overhead
        img_latents = self.pipeline.vae.config.get("scaling_factor", 0) * batch.img_latents.to(self.accelerator.device, dtype=self.weight_dtype)
        prompt_embeds = batch.prompt_embeds.to(self.accelerator.device, dtype=self.weight_dtype)

        # Generate noise and timesteps
        noise = self.sample_noise(img_latents)
        timesteps = self.sample_timesteps(img_latents.shape[0])

        # Apply noise directly with the correct dtype
        img_noisy_latents = self.noise_scheduler.add_noise(img_latents, noise, timesteps)

        # Get model prediction
        model_pred = self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds)

        # Calculate target and free memory we no longer need
        target = self.get_pred_target(img_latents, noise, timesteps, model_pred)

        # Free intermediate tensors to save memory
        del noise, img_noisy_latents

        # Calculate loss
        loss = self.get_loss(timesteps, model_pred, target)

        # Free more memory
        del model_pred, target, img_latents

        if torch.isnan(loss):
            self.logger.info("Loss is NaN.")
            msg = "Loss is NaN."
            raise ValueError(msg)

        avg_loss = self.accelerator.gather(loss.repeat(self.config.batch_size)).mean()  # type: ignore
        self.train_loss += avg_loss.item() / self.config.gradient_accumulation_steps

        self.accelerator.backward(loss)

        # Free memory after backward pass
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
