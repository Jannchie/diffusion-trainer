"""Fintunner for Stable Diffusion XL model."""

import math
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple

import torch
import torch.nn.functional as F
from accelerate.logging import get_logger
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel, compute_snr
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
    prepare_accelerator,
    str_to_dtype,
)
from diffusion_trainer.finetune.utils.lora import apply_lora_config
from diffusion_trainer.utils.timestep_weights import logit_timestep_weights

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
        self.apply_seed_settings(self.config.seed)

        self.model_name = config.model_name
        self.save_dir = config.save_dir

        self.save_path = Path(self.save_dir) / self.model_name

        self.save_dtype = str_to_dtype(config.save_dtype)
        self.weight_dtype = str_to_dtype(config.weight_dtype)

        self.batch_size = config.batch_size
        self.gradient_accumulation_steps = config.gradient_accumulation_steps

        self.mode: Literal["full-finetune", "lora", "lokr", "loha"] = config.mode

        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha
        self.lokr_factor = config.lokr_factor

        self.timestep_bias_strategy = config.timestep_bias_strategy

        self.use_ema = config.use_ema

        self.save_every_n_epochs = config.save_every_n_epochs
        self.save_every_n_steps = config.save_every_n_steps

        self.mixed_precision = str_to_dtype(config.mixed_precision)

        self.accelerator = prepare_accelerator(self.gradient_accumulation_steps, self.mixed_precision, self.config.log_with)
        self.logger = get_logger("diffusion_trainer.finetune.sdxl")
        self.device = self.accelerator.device

        self.pipeline = load_sdxl_pipeline(self.config.model_path, self.weight_dtype)

        sdxl_models = SDXLModels(
            unet=self.pipeline.unet.to(self.device, dtype=self.weight_dtype),
            text_encoder_1=self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype),
            text_encoder_2=self.pipeline.text_encoder_2.to(self.device, dtype=self.weight_dtype),
        )

        self.models: list[Any] = list(sdxl_models)

        self.configure_trainable_models(config, sdxl_models)

        self.training_models = [m.model for m in self.trainable_models_with_lr]

        for key, value in config.__dict__.items():
            self.logger.info("%s: %s", key, value)

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
            lycoris_model = apply_lora_config(config.mode, sdxl_models.unet)
            self.trainable_models_with_lr.append(TrainableModel(model=lycoris_model, lr=config.unet_lr))
            self.models.append(lycoris_model)
        else:
            msg = f"Unknown mode {config.mode}"
            raise ValueError(msg)

    def init_ema(self, model: torch.nn.Module, config: dict) -> None:
        # Create EMA for the unet.
        self.ema_unet: None | EMAModel = None
        if self.use_ema:
            self.ema_unet = EMAModel(
                parameters=model.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=config,
            ).to(self.device)

    def train(self) -> None:
        self.accelerator.wait_for_everyone()
        dataset = self.prepare_dataset(self.accelerator, self.config)
        if self.accelerator.is_main_process:
            dataset.print_bucket_info()
        freeze_models = [model for model in self.models if model not in self.training_models]
        self.freeze_model(self.accelerator, freeze_models)
        self.init_ema(self.pipeline.unet, self.pipeline.unet.config)

        self.trainable_parameters_dicts = get_trainable_parameter_dicts(self.accelerator, self.trainable_models_with_lr)
        optimizer = initialize_optimizer(self.config.optimizer, self.trainable_parameters_dicts)

        self.trainable_parameters: list[list[torch.Tensor]] = [param["params"] for param in self.trainable_parameters_dicts]

        if self.config.prediction_type == "v_prediction":
            self.pipeline.scheduler.register_to_config(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )

        # https://huggingface.co/docs/diffusers/api/schedulers/ddim
        self.noise_scheduler: DDPMScheduler = DDPMScheduler.from_config(
            self.pipeline.scheduler.config,
        )  # type: ignore

        self.logger.info("Noise scheduler config:")
        for key, value in self.noise_scheduler.config.items():
            self.logger.info("- %s: %s", key, value)

        sampler = BucketBasedBatchSampler(dataset, self.batch_size)
        with self.accelerator.main_process_first():
            data_loader = DataLoader(dataset, batch_sampler=sampler, num_workers=0, collate_fn=dataset.collate_fn)
        self.data_loader = self.accelerator.prepare(data_loader)

        num_update_steps_per_epoch = math.ceil(len(data_loader) / self.gradient_accumulation_steps / self.accelerator.num_processes)
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
            self.init_gradient_checkpointing(self.accelerator, self.models)

        self.log_training_parameters()

        self.accelerator.init_trackers(f"diffusion-trainer-{self.mode}", config=self.config.__dict__)
        self.execute_training_epoch(num_update_steps_per_epoch, n_total_steps)

    def save_full_finetune_model(self, filename: str) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        self.pipeline.to(self.save_dtype)
        self.pipeline.save_pretrained(self.save_path / f"{filename}")
        self.pipeline.to(self.weight_dtype)

    def log_training_parameters(self) -> None:
        if self.accelerator.is_main_process:
            n_params = get_n_params(self.trainable_parameters_dicts)
            self.logger.info("Number of epochs: %s", self.config.n_epochs)
            num_processes = self.accelerator.num_processes
            effective_batch_size = self.batch_size * num_processes * self.gradient_accumulation_steps
            self.logger.info("Effective batch size: %s", effective_batch_size)
            self.logger.info("Prediction type: %s", self.noise_scheduler.config.get("prediction_type"))
            self.logger.info("Number of trainable parameters: %s (%s)", f"{n_params:,}", format_size(n_params))
            self.logger.info("Hold tight, training is about to start!")

    def train_each_batch(self, batch: SDXLBatch) -> None:
        img_latents = self.pipeline.vae.config.get("scaling_factor", 0) * batch.img_latents.to(self.accelerator.device)
        prompt_embeds_1 = batch.prompt_embeds_1.to(self.accelerator.device)
        prompt_embeds_2 = batch.prompt_embeds_2.to(self.accelerator.device)
        prompt_embeds_pooled_2 = batch.prompt_embeds_pooled_2.to(self.accelerator.device)
        time_ids = batch.time_ids.to(self.accelerator.device)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=2)
        unet_added_conditions = {
            "text_embeds": prompt_embeds_pooled_2,
            "time_ids": time_ids,
        }
        noise = self.sample_noise(img_latents)
        timesteps = self.sample_timesteps(img_latents.shape[0])
        img_noisy_latents = self.noise_scheduler.add_noise(img_latents.float(), noise.float(), timesteps).to(self.weight_dtype)

        model_pred = self.get_model_pred(img_noisy_latents, timesteps, prompt_embeds, unet_added_conditions)
        target = self.get_pred_target(img_latents, noise, timesteps, model_pred)
        loss = self.get_loss(timesteps, model_pred, target)

        if torch.isnan(loss):
            self.logger.info("img_latents: %s %s", img_latents.shape, img_latents.mean())
            self.logger.info("prompt_embeds: %s %s", prompt_embeds.shape, prompt_embeds.mean())
            self.logger.info("prompt_embeds_pooled_2: %s %s", prompt_embeds_pooled_2.shape, prompt_embeds_pooled_2.mean())
            self.logger.info("time_ids: %s %s", time_ids.shape, time_ids)
            self.logger.info("noise: %s %s", noise.shape, noise.mean())
            self.logger.info("timesteps: %s %s", timesteps.shape, timesteps)
            self.logger.info("img_noisy_latents: %s %s", img_noisy_latents.shape, img_noisy_latents.mean())
            self.logger.info("model_pred: %s %s", model_pred.shape, model_pred.mean())
            self.logger.info("target: %s %s", target.shape, target.mean())
            msg = "Loss is NaN."
            raise ValueError(msg)

        avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()  # type: ignore
        self.train_loss += avg_loss.item() / self.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
            params_to_clip = self.pipeline.unet.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

        self.optimizer.step()
        self.lr_scheduler.step()
        # ref: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)

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

    @torch.no_grad()
    def process_batch(self, batch: DiffusionBatch) -> SDXLBatch:
        prompts_str = self.create_prompts_str(batch)
        prompt_embeds_1 = self.get_prompt_embeds_1(prompts_str)
        prompt_embeds_2, prompt_embeds_pooled_2 = self.get_prompt_embeds_2(prompts_str)

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

    def get_prompt_embeds_1(self, prompts_str: list[str]) -> torch.Tensor:
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

    def get_prompt_embeds_2(self, prompts_str: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs_2 = self.pipeline.tokenizer_2(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer_2.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2["input_ids"]
        prompt_embeds_output_2 = self.pipeline.text_encoder_2(
            text_input_ids_2.to(self.accelerator.device),
            output_hidden_states=True,
        )

        # We are only interested in the pooled output of the final text encoder
        prompt_embeds_pooled_2 = prompt_embeds_output_2[0]

        # use the second to last hidden state as the prompt embedding
        prompt_embeds = prompt_embeds_output_2.hidden_states[-2]
        return prompt_embeds, prompt_embeds_pooled_2

    def sample_timesteps(self, batch_size: int) -> torch.IntTensor:
        num_timesteps: int = self.noise_scheduler.config.get("num_train_timesteps", 1000)
        if self.timestep_bias_strategy == "uniform":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.accelerator.device)
        elif self.timestep_bias_strategy == "logit":
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = logit_timestep_weights(num_timesteps, device=self.accelerator.device)
            timesteps = torch.multinomial(weights, batch_size, replacement=True).int()
        else:
            msg = f"Unknown timestep bias strategy {self.timestep_bias_strategy}"
            raise ValueError(msg)
        return timesteps  # type: ignore
