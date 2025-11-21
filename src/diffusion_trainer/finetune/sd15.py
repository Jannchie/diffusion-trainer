import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, NamedTuple

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
        return load_sd15_pipeline(
            self.config.model_path,
            self.weight_dtype,
            enable_flash_attention=getattr(self.config, "enable_flash_attention", True),
        )

    def __init__(self, config: SD15Config) -> None:
        self.config = config
        super().__init__(config)
        self.logger = logging.getLogger(__name__)

    def _setup_models(self) -> None:
        """Setup SD1.5-specific models."""
        self.sd15_models = SD15Models(
            unet=self.pipeline.unet,
            text_encoder=self.pipeline.text_encoder,
        )
        self.sd15_models.unet.to(self.device, dtype=self.weight_dtype)
        self.sd15_models.text_encoder.to(self.device, dtype=self.weight_dtype)

        # Create models list for BaseTuner compatibility
        self.models: list[Any] = list(self.sd15_models)

    def _configure_full_finetune(self) -> None:
        """Configure models for full fine-tuning."""
        if self.config.unet_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.sd15_models.unet, lr=self.config.unet_lr))
        if self.config.text_encoder_lr:
            self.trainable_models_with_lr.append(TrainableModel(model=self.sd15_models.text_encoder, lr=self.config.text_encoder_lr))

    def _get_unet_model(self) -> torch.nn.Module:
        """Get the UNet model for LoRA configuration."""
        return self.sd15_models.unet

    def _post_lora_setup(self, lycoris_model: "LycorisNetwork") -> None:
        """Add lycoris model to the models list for SD15."""
        self.models.append(lycoris_model)

    def process_batch(self, batch: DiffusionBatch) -> SD15Batch:
        prompts_str = self.create_prompts_str(batch)
        prompt_embeds = self.get_prompt_embeds(prompts_str)

        return SD15Batch(
            img_latents=batch.img_latents,
            prompt_embeds=prompt_embeds,
        )

    def train_each_batch(self, batch: SD15Batch) -> None:
        # Move tensors to device and dtype efficiently
        tensors = self._move_tensors_to_device_and_dtype(
            img_latents=batch.img_latents,
            prompt_embeds=batch.prompt_embeds,
        )

        # Apply VAE scaling
        img_latents = self._apply_vae_scaling(tensors["img_latents"])

        # Prepare training tensors
        img_latents, noise, timesteps, img_noisy_latents = self._prepare_training_tensors(img_latents)

        # Get model prediction
        model_pred = self.get_model_pred(img_noisy_latents, timesteps, tensors["prompt_embeds"])

        # Calculate target
        target = self.get_pred_target(img_latents, noise, timesteps, model_pred)  # type: ignore

        # Calculate loss
        loss = self.get_loss(timesteps, model_pred, target)

        # Free memory efficiently
        self._free_tensors(
            noise,
            img_noisy_latents,
            model_pred,
            target,
            img_latents,
            tensors["prompt_embeds"],
        )

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
                clip_skip=self.config.clip_skip,
            )

        # Use the native CLIPTextModel path when enhanced embeddings are disabled
        text_inputs = self.pipeline.tokenizer(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs["input_ids"].to(self.accelerator.device)
        attention_mask = text_inputs["attention_mask"].to(self.accelerator.device)
        prompt_embeds_output = self.pipeline.text_encoder(
            text_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = prompt_embeds_output.hidden_states
        clip_skip = max(self.config.clip_skip, 0)
        if clip_skip == 0 or hidden_states is None:
            return prompt_embeds_output.last_hidden_state

        target_index = -(clip_skip + 1)
        # Ensure the index is within hidden_states range
        target_index = max(target_index, -len(hidden_states))

        selected_hidden_state = hidden_states[target_index]
        final_layer_norm = self.pipeline.text_encoder.text_model.final_layer_norm
        return final_layer_norm(selected_hidden_state)
