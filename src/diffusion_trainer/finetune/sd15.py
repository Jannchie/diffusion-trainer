import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusion_prompt_embedder import get_embeddings_sd15, get_embeddings_sd15_batch
from transformers.models.clip import CLIPTextModel

from diffusion_trainer.config import BaseConfig, SD15Config
from diffusion_trainer.dataset.dataset import DiffusionBatch
from diffusion_trainer.finetune.base import BaseTuner
from diffusion_trainer.finetune.utils import (
    TrainableModel,
    load_sd15_pipeline,
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

        # Use torch.cuda.empty_cache() to clear GPU memory before loading the model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        self.models = SD15Models(
            unet=self.pipeline.unet,
            text_encoder=self.pipeline.text_encoder,
        )
        self.models.unet.to(self.accelerator.device, dtype=self.weight_dtype)
        self.models.text_encoder.to(self.accelerator.device, dtype=self.weight_dtype)

        # Ensure gradient checkpointing is enabled
        if self.config.gradient_checkpointing:
            self.init_gradient_checkpointing(list(self.models))

        for key, value in config.__dict__.items():
            self.logger.info("%s: %s", key, value)

    def _configure_models(self) -> None:
        """Configure which models should be trainable with their learning rates."""
        if self.config.mode == "full-finetune":
            # For full fine-tuning, set model parameters as trainable according to the learning rates in the config file
            if self.config.unet_lr:
                self.trainable_models_with_lr.append(TrainableModel(model=self.models.unet, lr=self.config.unet_lr))
            if self.config.text_encoder_lr:
                self.trainable_models_with_lr.append(TrainableModel(model=self.models.text_encoder, lr=self.config.text_encoder_lr))

        elif self.config.mode in ("lora", "lokr", "loha"):
            # For efficient fine-tuning, only the Lora model needs to be trained; the base model itself does not require training
            lycoris_model = apply_lora_config(self.config.mode, self.models.unet)
            self.trainable_models_with_lr.append(TrainableModel(model=lycoris_model, lr=self.config.unet_lr))
            self.lycoris_model = lycoris_model
        else:
            msg = f"Unknown mode {self.config.mode}"
            raise ValueError(msg)

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
        img_noisy_latents = self.get_noisy_latents(img_latents, noise, timesteps)

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

        # Use the base class optimizer step method
        self.optimizer_step(loss)

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

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
        return get_embeddings_sd15(
            self.pipeline.tokenizer,
            self.pipeline.text_encoder,
            prompt=prompt,
            neg_prompt=neg_prompt,
            clip_skip=clip_skip,
        )

    def get_prompt_embeds(self, prompts_str: list[str]) -> torch.Tensor:
        if self.config.use_enhanced_embeddings:
            return get_embeddings_sd15_batch(
                self.pipeline.tokenizer,
                self.pipeline.text_encoder,
                prompts=prompts_str,
                pad_last_block=True,
            )

        # 使用原生的 CLIPTextModel
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
        return prompt_embeds_output.last_hidden_state
