"""Fintunner for Stable Diffusion XL model."""

from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, NamedTuple

import torch
from accelerate.logging import get_logger
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from transformers.models.clip import CLIPTextModel, CLIPTextModelWithProjection

from diffusion_trainer.config import BaseConfig, SDXLConfig
from diffusion_trainer.dataset.dataset import DiffusionBatch
from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.utils import (
    TrainableModel,
    format_size,
    get_n_params,
    load_sdxl_pipeline,
)

if TYPE_CHECKING:
    from lycoris import LycorisNetwork

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
        self.config = config
        # Update save path for model-specific naming
        super().__init__(config)
        self.save_path = Path(config.save_dir) / config.model_name

        # Keep mode as it needs type conversion
        self.mode: Literal["full-finetune", "lora", "lokr", "loha"] = config.mode
        self.logger = get_logger("diffusion_trainer.finetune.sdxl")

    def _setup_models(self) -> None:
        """Setup SDXL-specific models."""
        self.sdxl_models = SDXLModels(
            unet=self.pipeline.unet.to(self.device, dtype=self.weight_dtype),
            text_encoder_1=self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype),
            text_encoder_2=self.pipeline.text_encoder_2.to(self.device, dtype=self.weight_dtype),
        )
        self.models: list[Any] = list(self.sdxl_models)

    def get_pipeline(self) -> DiffusionPipeline:
        return load_sdxl_pipeline(self.config.model_path, self.weight_dtype)

    def _configure_full_finetune(self) -> None:
        """Configure models for full fine-tuning."""
        if self.config.unet_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.sdxl_models.unet, lr=self.config.unet_lr))
        if self.config.text_encoder_1_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.sdxl_models.text_encoder_1, lr=self.config.text_encoder_1_lr))
        if self.config.text_encoder_2_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.sdxl_models.text_encoder_2, lr=self.config.text_encoder_2_lr))

    def _post_lora_setup(self, lycoris_model: "LycorisNetwork") -> None:
        """Add lycoris model to the models list for SDXL."""
        self.models.append(lycoris_model)

    def _get_unet_model(self) -> torch.nn.Module:
        """Get the UNet model for LoRA configuration."""
        return self.sdxl_models.unet

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
        # Move all tensors to device and dtype efficiently
        tensors = self._move_tensors_to_device_and_dtype(
            img_latents=batch.img_latents,
            prompt_embeds_1=batch.prompt_embeds_1,
            prompt_embeds_2=batch.prompt_embeds_2,
            prompt_embeds_pooled_2=batch.prompt_embeds_pooled_2,
        )
        time_ids = batch.time_ids.to(self.device)

        # Apply VAE scaling
        img_latents = self._apply_vae_scaling(tensors["img_latents"])

        # Concatenate prompt embeds once and reuse
        prompt_embeds = torch.cat([tensors["prompt_embeds_1"], tensors["prompt_embeds_2"]], dim=2)
        unet_added_conditions = {
            "text_embeds": tensors["prompt_embeds_pooled_2"],
            "time_ids": time_ids,
        }

        # Prepare training tensors
        img_latents, noise, timesteps, img_noisy_latents = self._prepare_training_tensors(img_latents)

        # Predict and calculate loss
        model_pred = self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds, unet_added_conditions)
        target = self.get_pred_target(img_latents, noise, timesteps, model_pred) # type: ignore

        loss = self.get_loss(timesteps, model_pred, target)

        # Free memory efficiently
        self._free_tensors(
            noise, img_noisy_latents, model_pred, target, img_latents,
            prompt_embeds, tensors["prompt_embeds_1"], tensors["prompt_embeds_2"], tensors["prompt_embeds_pooled_2"],
        )

        # Use base class optimizer step method
        self.optimizer_step(loss)

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

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        # return get_weighted_text_embeddings_sdxl(self.pipeline, prompt, neg_prompt, clip_skip=clip_skip)  # type: ignore
        msg = "get_preview_prompt_embeds not yet implemented for SDXL"
        raise NotImplementedError(msg)
