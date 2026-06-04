import logging
from contextlib import nullcontext
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

    @property
    def training_prompts_use_attention_parser(self) -> bool:
        return self.config.use_enhanced_embeddings

    def process_batch(self, batch: DiffusionBatch) -> SD15Batch:
        prompts_str = self.create_prompts_str(batch)
        prompts_str = self.apply_condition_dropout_to_prompts(prompts_str)
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
        prompt_embeds = tensors["prompt_embeds"]
        img_latents = tensors["img_latents"]

        def model_pred_fn(img_noisy_latents: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
            return self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds)

        self.train_on_latents(
            img_latents,
            model_pred_fn,
            extra_tensors=(prompt_embeds,),
        )

    def get_model_pred(
        self,
        img_noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
    ) -> torch.Tensor:
        runtime_unet = self.get_runtime_model(self.sd15_models.unet)
        return runtime_unet(
            img_noisy_latents,
            timesteps,
            prompt_embeds,
            return_dict=False,
        )[0]

    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> dict[str, torch.Tensor]:
        # pad_last_block=True pads short prompts to the full 77-token block with
        # EOS, which (unmasked) is bit-identical to the training-path conditioning.
        prompt_embeds, neg_prompt_embeds = get_embeddings_sd15(
            self.pipeline.tokenizer,
            self.pipeline.text_encoder,
            prompt=prompt,
            neg_prompt=neg_prompt,
            clip_skip=clip_skip,
            pad_last_block=True,
        )
        return {"prompt_embeds": prompt_embeds, "negative_prompt_embeds": neg_prompt_embeds}

    def get_prompt_embeds(self, prompts_str: list[str]) -> torch.Tensor:
        runtime_text_encoder = self.get_runtime_model(self.sd15_models.text_encoder)
        text_encoder_context = nullcontext() if any(param.requires_grad for param in self.sd15_models.text_encoder.parameters()) else torch.no_grad()
        if self.config.use_enhanced_embeddings:
            with text_encoder_context:
                return get_embeddings_sd15_batch(
                    self.pipeline.tokenizer,
                    runtime_text_encoder,
                    prompts=prompts_str,
                    pad_last_block=True,
                    clip_skip=self.config.clip_skip,
                )

        # Use the native CLIPTextModel path when enhanced embeddings are disabled.
        # No attention_mask: the SD1.x ecosystem (CompVis training, diffusers,
        # sd-scripts, WebUI) encodes EOS-padded sequences unmasked, and masked
        # padding embeddings diverge wildly (~20x norm) from what any inference
        # stack will feed the cross-attention at sampling time.
        text_inputs = self.pipeline.tokenizer(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs["input_ids"].to(self.accelerator.device)
        with text_encoder_context:
            prompt_embeds_output = runtime_text_encoder(
                text_input_ids,
                output_hidden_states=True,
            )
        hidden_states = prompt_embeds_output.hidden_states
        # A1111/WebUI semantics: clip_skip=1 -> last layer, clip_skip=2 ->
        # penultimate layer (NAI convention). hidden_states[0] is the embedding
        # output, so clamp the index to the deepest real layer.
        clip_skip = max(self.config.clip_skip, 1)
        if clip_skip == 1 or hidden_states is None:
            return prompt_embeds_output.last_hidden_state

        target_index = max(-clip_skip, -(len(hidden_states) - 1))
        selected_hidden_state = hidden_states[target_index]
        # transformers 5.x flattened CLIPTextModel (no .text_model wrapper);
        # fall back to the wrapped layout for transformers 4.x.
        text_encoder = self.pipeline.text_encoder
        final_layer_norm = getattr(text_encoder, "text_model", text_encoder).final_layer_norm
        return final_layer_norm(selected_hidden_state)
