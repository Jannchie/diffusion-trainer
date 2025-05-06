"""Fintunner for Stable Diffusion XL model."""

import math
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
from accelerate.logging import get_logger
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from sd_embed.embedding_funcs import get_weighted_text_embeddings_sdxl
from torch.utils.data import DataLoader
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection

from diffusion_trainer.config import BaseConfig, SDXLConfig
from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch
from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.utils import (
    TrainableModel,
    format_size,
    get_n_params,
    get_trainable_parameter_dicts,
    initialize_optimizer,
    load_sdxl_pipeline,
)
from diffusion_trainer.finetune.utils.lora import apply_lora_config

logger = getLogger("diffusion_trainer.finetune.sdxl")


class SDXLModels(NamedTuple):
    unet: UNet2DConditionModel
    text_encoder_1: CLIPTextModel
    text_encoder_2: CLIPTextModelWithProjection


@dataclass
class SDXLBatch:
    img_latents: torch.Tensor
    prompt_embeds_1: torch.Tensor
    prompt_embeds_2: torch.Tensor
    prompt_embeds_pooled_2: torch.Tensor
    time_ids: torch.Tensor


class SDXLTuner(BaseTuner):
    """Finetune Stable Diffusion XL model."""

    @staticmethod
    def from_config(config: BaseConfig) -> "SDXLTuner":
        """Create a new instance from a configuration dictionary."""
        if not isinstance(config, SDXLConfig):
            msg = f"Expected SDXLConfig, got {type(config)}"
            raise TypeError(msg)
        return SDXLTuner(config)

    def __init__(self, config: SDXLConfig) -> None:
        """Initialize."""
        super().__init__(config)
        self.config = config
        self.apply_seed_settings(self.config.seed)

        # Clear CUDA cache before loading models to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Only convert to Path when needed
        self.save_path = Path(config.save_dir) / config.model_name

        # Keep mode as it needs type conversion
        self.mode: Literal["full-finetune", "lora", "lokr", "loha"] = config.mode

        self.logger = get_logger("diffusion_trainer.finetune.sdxl")

        sdxl_models = SDXLModels(
            unet=self.pipeline.unet.to(self.device, dtype=self.weight_dtype),
            text_encoder_1=self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype),
            text_encoder_2=self.pipeline.text_encoder_2.to(self.device, dtype=self.weight_dtype),
        )

        self.models: list[Any] = list(sdxl_models)

        self.configure_trainable_models(config, sdxl_models)

        self.training_models = [m.model for m in self.trainable_models_with_lr]

        # Enable gradient checkpointing early if configured
        if self.config.gradient_checkpointing:
            self.init_gradient_checkpointing(self.models)

        for key, value in config.__dict__.items():
            self.logger.info("%s: %s", key, value)

    def get_pipeline(self) -> DiffusionPipeline:
        return load_sdxl_pipeline(self.config.model_path, self.weight_dtype)

    def configure_trainable_models(self, config: SDXLConfig, sdxl_models: SDXLModels) -> None:
        self.trainable_models_with_lr: list[TrainableModel] = []
        if config.mode == "full-finetune":
            # 如果是全量微调，根据配置文件中的学习率，将模型的参数设置为可训练
            if config.unet_lr:
                self.trainable_models_with_lr.append(TrainableModel(model=sdxl_models.unet, lr=config.unet_lr))
            if config.text_encoder_1_lr:
                self.trainable_models_with_lr.append(TrainableModel(model=sdxl_models.text_encoder_1, lr=config.text_encoder_1_lr))
            if config.text_encoder_2_lr:
                self.trainable_models_with_lr.append(TrainableModel(model=sdxl_models.text_encoder_2, lr=config.text_encoder_2_lr))

        elif config.mode in ("lora", "lokr", "loha"):
            # 如果是高效微调，则模型本身无需训练，只需训练Lora模型。
            self.lycoris_model = apply_lora_config(config.mode, sdxl_models.unet)
            self.trainable_models_with_lr.append(TrainableModel(model=self.lycoris_model, lr=config.unet_lr))
            self.models.append(self.lycoris_model)
        else:
            msg = f"Unknown mode {config.mode}"
            raise ValueError(msg)

    def train(self) -> None:
        self.accelerator.wait_for_everyone()
        dataset = self.prepare_dataset(self.config)
        if self.accelerator.is_main_process:
            dataset.print_bucket_info()
        freeze_models = [model for model in self.models if model not in self.training_models]
        self.freeze_model(freeze_models)

        self.trainable_parameters_dicts = get_trainable_parameter_dicts(self.accelerator, self.trainable_models_with_lr)
        optimizer = initialize_optimizer(self.config.optimizer, self.trainable_parameters_dicts)

        self.trainable_parameters: list[list[torch.Tensor]] = [param["params"] for param in self.trainable_parameters_dicts]

        sampler = BucketBasedBatchSampler(dataset, self.config.batch_size)
        with self.accelerator.main_process_first():
            # Optimize DataLoader by using appropriate num_workers based on system
            num_workers = 0
            data_loader = DataLoader(
                dataset,
                batch_sampler=sampler,
                num_workers=num_workers,
                collate_fn=dataset.collate_fn,
                pin_memory=True,  # Enable pin_memory for faster data transfer to GPU
            )
        self.data_loader = self.accelerator.prepare(data_loader)

        num_update_steps_per_epoch = math.ceil(len(data_loader) / self.config.gradient_accumulation_steps / self.accelerator.num_processes)
        n_total_steps = self.config.n_epochs * num_update_steps_per_epoch

        lr_scheduler = get_scheduler(
            SchedulerType.COSINE_WITH_RESTARTS,
            optimizer=optimizer,
            num_warmup_steps=self.config.optimizer_warmup_steps * self.accelerator.num_processes,
            num_training_steps=n_total_steps * self.accelerator.num_processes,
            num_cycles=self.config.optimizer_num_cycles,
        )

        self.optimizer = self.accelerator.prepare(optimizer)
        self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        if self.config.gradient_checkpointing:
            self.init_gradient_checkpointing(self.models)

        self.log_training_parameters()

        self.accelerator.init_trackers(f"diffusion-trainer-{self.mode}", config=self.config.__dict__)
        self.execute_training_epoch(
            self.data_loader,
            self.lr_scheduler,
            self.training_models,
            num_update_steps_per_epoch,
            n_total_steps,
        )

    def log_training_parameters(self) -> None:
        if self.accelerator.is_main_process:
            n_params = get_n_params(self.trainable_parameters_dicts)
            self.logger.info("Number of epochs: %s", self.config.n_epochs)
            num_processes = self.accelerator.num_processes
            effective_batch_size = self.config.batch_size * num_processes * self.config.gradient_accumulation_steps
            self.logger.info("Effective batch size: %s", effective_batch_size)
            self.logger.info("Prediction type: %s", self.noise_scheduler.config.get("prediction_type"))
            self.logger.info("Number of trainable parameters: %s (%s)", f"{n_params:,}", format_size(n_params))
            self.logger.info("Hold tight, training is about to start!")

    def train_each_batch(self, batch: SDXLBatch) -> None:
        # Pre-convert all tensors to the correct dtype at once to avoid multiple conversions
        img_latents = self.pipeline.vae.config.get("scaling_factor", 0) * batch.img_latents.to(self.accelerator.device, dtype=self.weight_dtype)
        prompt_embeds_1 = batch.prompt_embeds_1.to(self.accelerator.device, dtype=self.weight_dtype)
        prompt_embeds_2 = batch.prompt_embeds_2.to(self.accelerator.device, dtype=self.weight_dtype)
        prompt_embeds_pooled_2 = batch.prompt_embeds_pooled_2.to(self.accelerator.device, dtype=self.weight_dtype)
        time_ids = batch.time_ids.to(self.accelerator.device)

        # Concatenate prompt embeds once and reuse
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=2)
        unet_added_conditions = {
            "text_embeds": prompt_embeds_pooled_2,
            "time_ids": time_ids,
        }

        # Sample noise and timesteps
        noise = self.sample_noise(img_latents)
        timesteps = self.sample_timesteps(img_latents.shape[0])

        # Add noise once with the correct dtype to avoid conversions
        img_noisy_latents = self.get_noisy_latents(img_latents, noise, timesteps)

        # Predict and calculate loss
        model_pred = self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds, unet_added_conditions)
        target = self.get_pred_target(img_latents, noise, timesteps, model_pred)

        # Free memory proactively
        del noise, img_noisy_latents

        loss = self.get_loss(timesteps, model_pred, target)

        # Free more memory
        del model_pred, target, img_latents
        del prompt_embeds, prompt_embeds_1, prompt_embeds_2, prompt_embeds_pooled_2

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

        # Periodically clear CUDA cache to prevent memory fragmentation
        if self.accelerator.sync_gradients and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_model_pred(
        self,
        img_noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        unet_added_conditions: dict,
    ) -> torch.Tensor:
        return self.pipeline.unet(
            img_noisy_latents,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]

    def process_batch(self, batch: DiffusionBatch) -> SDXLBatch:
        prompts_str = self.create_prompts_str(batch)

        # Process prompt embeddings efficiently
        prompt_embeds_1 = self.get_prompt_embeds_1(prompts_str)
        prompt_embeds_2, prompt_embeds_pooled_2 = self.get_prompt_embeds_2(prompts_str)

        # Create time_ids tensor once and directly with the correct device
        time_ids = torch.stack(
            [
                batch.original_size[:, 1],
                batch.original_size[:, 0],
                batch.crop_ltrb[:, 1],
                batch.crop_ltrb[:, 0],
                batch.train_resolution[:, 1],
                batch.train_resolution[:, 0],
            ],
            dim=1,
        ).to(self.accelerator.device)

        return SDXLBatch(
            img_latents=batch.img_latents,
            prompt_embeds_1=prompt_embeds_1,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_pooled_2=prompt_embeds_pooled_2,
            time_ids=time_ids,
        )

    @torch.no_grad()
    def get_prompt_embeds_1(self, prompts_str: list[str]) -> torch.Tensor:
        text_inputs = self.pipeline.tokenizer(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs["input_ids"].to(self.accelerator.device)

        # Use no_grad for inference to save memory
        prompt_embeds_output = self.pipeline.text_encoder(
            text_input_ids,
            output_hidden_states=True,
        )

        # Extract only what we need and free the rest
        hidden_states = prompt_embeds_output.hidden_states[-2]
        del prompt_embeds_output, text_input_ids, text_inputs

        return hidden_states

    @torch.no_grad()
    def get_prompt_embeds_2(self, prompts_str: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs_2 = self.pipeline.tokenizer_2(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2["input_ids"].to(self.accelerator.device)

        # Use no_grad for inference to save memory
        prompt_embeds_output_2 = self.pipeline.text_encoder_2(
            text_input_ids_2,
            output_hidden_states=True,
        )

        # Extract only what we need and free the rest
        prompt_embeds_pooled_2 = prompt_embeds_output_2[0]
        prompt_embeds = prompt_embeds_output_2.hidden_states[-2]

        # Clean up to save memory
        del prompt_embeds_output_2, text_input_ids_2, text_inputs_2

        return prompt_embeds, prompt_embeds_pooled_2

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str) -> tuple[torch.Tensor, torch.Tensor]:
        return get_weighted_text_embeddings_sdxl(self.pipeline, prompt, neg_prompt, clip_skip=2)  # type: ignore
