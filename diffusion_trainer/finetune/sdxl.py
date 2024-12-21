"""Fintunner for Stable Diffusion XL model."""

import os
import random
import secrets
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel, cast_training_params, compute_snr
from diffusers.utils.state_dict_utils import convert_state_dict_to_diffusers
from peft.tuners.loha.config import LoHaConfig
from peft.tuners.lokr.config import LoKrConfig
from peft.tuners.lora.config import LoraConfig
from peft.utils.save_and_load import get_peft_model_state_dict
from torch.utils.data import DataLoader

from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch, DiffusionDataset
from diffusion_trainer.finetune.utils import prepare_accelerator
from diffusion_trainer.shared import get_progress

logger = getLogger("diffusion_trainer.finetune.sdxl")


@dataclass
class SDXLBatch:
    img_latents: torch.Tensor
    prompt_embeds_1: torch.Tensor
    prompt_embeds_2: torch.Tensor
    prompt_embeds_pooled_2: torch.Tensor
    time_ids: torch.Tensor


def load_pipeline(path: PathLike | str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path = Path(path)

    logger.info('Loading models from "%s" (%s)', path, dtype)
    with Path(os.devnull).open("w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        if path.is_dir():
            pipe = StableDiffusionXLPipeline.from_pretrained(path, torch_dtype=dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(path, torch_dtype=dtype)
    logger.info("Models loaded successfully.")
    if isinstance(pipe, StableDiffusionXLPipeline):
        return pipe
    msg = "Failed to load models."
    raise ValueError(msg)


class SDXLTuner:
    """Finetune Stable Diffusion XL model."""

    def __init__(self, *, model_path: str, dataset: DiffusionDataset, seed: None | int = None) -> None:
        """Initialize."""
        self.model_path = Path(model_path)
        self.seed = seed if seed is not None else secrets.randbelow(1_000_000_000)

        self.accelerator = prepare_accelerator()
        self.device = self.accelerator.device

        self.save_dtype = torch.float16
        self.weight_dtype = torch.float16

        self.pipeline = load_pipeline(self.model_path, self.weight_dtype)

        # TODO: ignore the warning about triton
        self.prediction_type = None

        self.n_epochs = 1
        self.batch_size = 1
        self.gradient_accumulation_steps = 1

        self.unet_lr = 1e-5
        self.text_encoder_1_lr = 1e-6
        self.text_encoder_2_lr = 1e-6

        self.mode: Literal["full-finetune", "lora", "lokr", "loha"] = "lokr"

        self.lora_rank = 4
        self.lora_alpha = 1

        self.lokr_factor = 16  # For LoKr

        self.noise_offset = 0.0

        # reduce memory usage by checkpointing the gradients
        self.gradient_checkpointing = True

        self.init_model_modules()
        self.init_gradient_checkpointing()

        self.dataset = dataset

        # The timestep bias strategy, which may help direct the model toward learning low or high frequency details.
        # The default value is 'none', which means no bias is applied.
        # The value of 'later' will increase the frequency of the model's final training timesteps.
        self.timestep_bias_strategy: Literal["none", "earlier", "later", "range"] = "none"
        # The multiplier for the bias. Defaults to 1.0, which means no bias is applied.
        # A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it.
        self.timestep_bias_multiplier = 1.0
        # The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased.
        # A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines
        # whether the biased portions are in the earlier or later timesteps.
        self.timestep_bias_portion = 0.25
        # When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias.
        # Defaults to zero, which equates to having no specific bias.
        self.timestep_bias_begin = 0
        # When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias.
        # Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on.
        self.timestep_bias_end = 1000

        # SNR weighting gamma to be used if rebalancing the loss. Recommended value is 5.0.
        # More details here: https://arxiv.org/abs/2303.09556.
        self.snr_gamma = 5.0

        # Max gradient norm.
        self.max_grad_norm = 1.0

        self.use_ema = False

        self.init_ema()

    def init_ema(self) -> None:
        # Create EMA for the unet.
        self.ema_unet: None | EMAModel = None
        if self.use_ema:
            unet_copy = self.unet.parameters().copy()
            self.ema_unet = EMAModel(
                unet_copy,
                model_cls=UNet2DConditionModel,
                model_config=self.unet.config,
            ).to(self.device)

    def init_model_modules(self) -> None:
        self.unet = self.pipeline.unet.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_1 = self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_2 = self.pipeline.text_encoder_2.to(self.device, dtype=self.weight_dtype)
        self.vae = self.pipeline.vae

    def init_gradient_checkpointing(self) -> None:
        if self.gradient_checkpointing:
            self.unet.enable_gradient_checkpointing()
            self.text_encoder_1.gradient_checkpointing_enable()
            self.text_encoder_2.gradient_checkpointing_enable()

    def get_n_params(self, trainable_parameters: list[dict]) -> int:
        n_params = 0
        for param in trainable_parameters:
            n_params += sum(p.numel() for p in param["params"])
        return n_params

    def get_trainable_parameters(self) -> list[dict]:
        trainable_parameters = []
        if self.unet_lr != 0:
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.unet.parameters())),
                    "lr": self.unet_lr,
                },
            )
        if self.text_encoder_1_lr != 0:
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.text_encoder_1.parameters())),
                    "lr": self.text_encoder_1_lr,
                },
            )
        if self.text_encoder_2_lr != 0:
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.text_encoder_2.parameters())),
                    "lr": self.text_encoder_2_lr,
                },
            )
        return trainable_parameters

    @property
    def training_models(self) -> list[torch.nn.Module]:
        """Get the training model."""
        model = []
        if self.unet_lr != 0:
            model.append(self.pipeline.unet)
        if self.text_encoder_1_lr != 0:
            model.append(self.pipeline.text_encoder)
        if self.text_encoder_2_lr != 0:
            model.append(self.pipeline.text_encoder_2)
        return model

    def apply_lora_config(self) -> None:
        # Lora config
        if self.mode == "lora":
            if not self.lora_alpha or not self.lora_rank:
                msg = "Lora rank and alpha must be provided for lora mode."
                raise ValueError(msg)
            unet_lora_config = LoraConfig(
                r=self.lora_rank,
                lora_alpha=self.lora_alpha,
                init_lora_weights="gaussian",
                target_modules=["to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_lora_config)

            if self.text_encoder_1_lr != 0:
                text_lora_config = LoraConfig(
                    r=self.lora_rank,
                    lora_alpha=self.lora_alpha,
                    init_lora_weights="gaussian",
                    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
                )
                self.text_encoder_1.add_adapter(text_lora_config)
                self.text_encoder_2.add_adapter(text_lora_config)
        elif self.mode == "lokr":
            if not self.lora_alpha or not self.lora_rank:
                msg = "Lora rank and alpha must be provided for lokr mode."
                raise ValueError(msg)
            unet_lokr_config = LoKrConfig(
                r=self.lora_rank,
                alpha=self.lora_alpha,
                use_effective_conv2d=True,
                rank_dropout=0.5,
                module_dropout=0.5,
                target_modules=[
                    "proj_in",
                    "proj_out",
                    "to_k",
                    "to_q",
                    "to_v",
                    "to_out.0",
                    # "ff.net.0.proj",
                    # "ff.net.2",
                ],
            )
            self.unet.add_adapter(unet_lokr_config)

            if self.text_encoder_1_lr != 0:
                text_lokr_config = LoKrConfig(
                    r=self.lora_rank,
                    alpha=self.lora_alpha,
                    decompose_factor=self.lokr_factor,
                    # target_modules=["k_proj", "q_proj", "v_proj", "out_proj", "fc1", "fc2"],
                    target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
                )
                self.text_encoder_1.add_adapter(text_lokr_config)
                self.text_encoder_2.add_adapter(text_lokr_config)
        elif self.mode == "loha":
            if not self.lora_alpha or not self.lora_rank:
                msg = "Lora rank and alpha must be provided for loha mode."
                raise ValueError(msg)
            unet_loha_config = LoHaConfig(
                r=self.lora_rank,
                alpha=self.lora_alpha,
                use_effective_conv2d=True,
                target_modules=["proj_in", "proj_out", "to_k", "to_q", "to_v", "to_out.0"],
            )
            self.unet.add_adapter(unet_loha_config)
            if self.text_encoder_1_lr != 0:
                text_loha_config = LoHaConfig(
                    r=self.lora_rank,
                    alpha=self.lora_alpha,
                    target_modules=["k_proj", "q_proj", "v_proj", "out_proj"],
                )
                self.text_encoder_1.add_adapter(text_loha_config)
                self.text_encoder_2.add_adapter(text_loha_config)

    def train(self) -> None:
        sampler = BucketBasedBatchSampler(self.dataset, self.batch_size)
        data_loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=self.dataset.collate_fn)
        n_epochs = self.n_epochs
        progress = get_progress()

        self.update_training_flags()
        self.apply_lora_config()

        trainable_parameters = self.get_trainable_parameters()
        cast_training_params([self.unet, self.text_encoder_1, self.text_encoder_2])

        optimizer = torch.optim.Adafactor(trainable_parameters, lr=self.unet_lr)

        self.noise_scheduler = DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule="scaled_linear",
            num_train_timesteps=1000,
            clip_sample=False,
        )

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

        optimizer = self.accelerator.prepare(optimizer)
        lr_scheduler = self.accelerator.prepare(lr_scheduler)
        self.unet = self.accelerator.prepare(self.unet)
        self.text_encoder_1 = self.accelerator.prepare(self.text_encoder_1)
        self.text_encoder_2 = self.accelerator.prepare(self.text_encoder_2)

        self.log_training_parameters(trainable_parameters)

        total_task = progress.add_task("Total Progress", total=n_epochs * len(data_loader))
        global_step = 0
        with progress:
            for epoch in range(n_epochs):
                self.train_loss = 0.0
                for _step, batch in enumerate(progress.track(data_loader, description=f"Epoch {epoch+1}")):
                    with self.accelerator.accumulate(self.pipeline.unet):
                        if not isinstance(batch, DiffusionBatch):
                            msg = f"Expected DiffusionBatch, got something else. Got: {type(batch)}"
                            raise TypeError(msg)

                        loss = self.train_each_batch(batch)

                        optimizer.step()
                        lr_scheduler.step()
                        optimizer.zero_grad()
                        progress.update(total_task, advance=1)

                    if self.accelerator.sync_gradients:
                        if self.ema_unet:
                            self.ema_unet.step(self.unet.parameters())

                        global_step += 1
                        self.accelerator.log({"train_loss": self.train_loss}, step=global_step)
                        self.train_loss = 0.0

                    progress.update(total_task, description=f"LR: {lr_scheduler.get_last_lr()[0]:.2e} - Loss: {loss:.2f}")
            # end of epoch
            self.accelerator.wait_for_everyone()
            if self.accelerator.is_main_process:
                if self.mode in ("lora", "lokr", "loha"):
                    self.save_lora_model()
                else:
                    self.save_full_finetune_model()

    def save_lora_model(self) -> None:
        unet = self.accelerator.unwrap_model(self.unet)
        unet_lora_state_dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))

        if self.text_encoder_1_lr:
            text_encoder_1 = self.accelerator.unwrap_model(self.text_encoder_1)
            text_encoder_2 = self.accelerator.unwrap_model(self.text_encoder_2)

            text_encoder_lora_layers: dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_1))
            text_encoder_2_lora_layers: dict = convert_state_dict_to_diffusers(get_peft_model_state_dict(text_encoder_2))
        else:
            text_encoder_lora_layers = None  # type: ignore
            text_encoder_2_lora_layers = None  # type: ignore

        StableDiffusionXLPipeline.save_lora_weights(
            save_directory="out",
            unet_lora_layers=unet_lora_state_dict,
            text_encoder_lora_layers=text_encoder_lora_layers,
            text_encoder_2_lora_layers=text_encoder_2_lora_layers,
        )

    def save_full_finetune_model(self) -> None:
        msg = "Full finetune model saving is not implemented yet."
        raise NotImplementedError(msg)

    def log_training_parameters(self, trainable_parameters: list[dict]) -> None:
        n_params = self.get_n_params(trainable_parameters)
        logger.info("Number of trainable parameters: %s (%sB)", f"{n_params:,}", f"{n_params * 4 / 1024 / 1024 / 1024:.2f}")
        logger.info("Number of epochs: %s", self.n_epochs)
        logger.info("Batch size: %s", self.batch_size)
        logger.info("Gradient accumulation steps: %s", self.gradient_accumulation_steps)
        logger.info("Gradient checkpointing: %s", self.gradient_checkpointing)
        logger.info("Noise offset: %s", self.noise_offset)
        logger.info("SNR gamma: %s", self.snr_gamma)
        logger.info("Max gradient norm: %s", self.max_grad_norm)
        logger.info("Unet: %s %s", self.unet.device, self.unet.dtype)
        logger.info("Text Encoder 1: %s %s", self.text_encoder_1.device, self.text_encoder_1.dtype)
        logger.info("Text Encoder 2: %s %s", self.text_encoder_2.device, self.text_encoder_2.dtype)
        logger.info("Starting training.")

    def update_training_flags(self) -> None:
        if self.mode in ("lora", "lokr", "loha"):
            logger.info("Training with %s, freezing the models.", self.mode)
            self.unet.requires_grad_(False)
            self.unet.train(False)
            self.text_encoder_1.requires_grad_(False)
            self.text_encoder_1.train(False)
            self.text_encoder_2.requires_grad_(False)
            self.text_encoder_2.train(False)
        elif self.mode == "full-finetune":
            if self.unet_lr:
                logger.info("Training the UNet with learning rate %s", self.unet_lr)
                self.unet.requires_grad_(True)
                self.unet.train(True)
            else:
                self.unet.requires_grad_(False)
                self.unet.train(False)
            if self.text_encoder_1_lr:
                logger.info("Training the text encoder 1 with learning rate %s", self.text_encoder_1_lr)
                self.text_encoder_1.train(True)
                self.text_encoder_1.requires_grad_(True)
            else:
                self.text_encoder_1.train(False)
                self.text_encoder_1.requires_grad_(False)
            if self.text_encoder_2_lr:
                logger.info("Training the text encoder 2 with learning rate %s", self.text_encoder_2_lr)
                self.text_encoder_2.train(True)
                self.text_encoder_2.requires_grad_(True)
            else:
                self.text_encoder_2.train(False)
                self.text_encoder_2.requires_grad_(False)
        else:
            msg = f"Unknown mode {self.mode}"
            raise ValueError(msg)

    def train_each_batch(self, original_batch: DiffusionBatch) -> float:
        batch = self.process_batch(original_batch)

        img_latents = batch.img_latents.to(self.accelerator.device)
        prompt_embeds_1 = batch.prompt_embeds_1.to(self.accelerator.device)
        prompt_embeds_2 = batch.prompt_embeds_2.to(self.accelerator.device)
        prompt_embeds_pooled_2 = batch.prompt_embeds_pooled_2.to(self.accelerator.device)
        time_ids = batch.time_ids.to(self.accelerator.device)
        prompt_embeds = torch.cat([prompt_embeds_1, prompt_embeds_2], dim=2)
        unet_added_conditions = {
            "text_embeds": prompt_embeds_pooled_2,
            "time_ids": time_ids,
        }
        # Sample noise
        noise = self.sample_noise(img_latents)

        batch_size = img_latents.shape[0]
        timesteps = self.sample_timesteps(batch_size)
        img_latents_with_noise = self.noise_scheduler.add_noise(img_latents, noise, timesteps)

        model_pred = self.pipeline.unet(
            img_latents_with_noise,
            timesteps,
            prompt_embeds,
            added_cond_kwargs=unet_added_conditions,
            return_dict=False,
        )[0]

        target = self.get_pred_target(img_latents, noise, timesteps, model_pred)
        loss = self.get_loss(timesteps, model_pred, target)

        avg_loss = self.accelerator.gather(loss.repeat(batch_size)).mean()  # type: ignore
        self.train_loss += avg_loss.item() / self.gradient_accumulation_steps

        self.accelerator.backward(loss)
        if self.accelerator.sync_gradients:
            params_to_clip = self.unet.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
        return loss.detach().item()

    def get_loss(self, timesteps: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            snr = compute_snr(self.noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            if self.noise_scheduler.config.get("prediction_type") == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
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
        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.pipeline.register_to_config(prediction_type=self.prediction_type)
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

    def generate_timestep_weights(
        self,
        num_timesteps: int,
    ) -> torch.Tensor:
        weights = torch.ones(num_timesteps)
        timestep_bias_strategy = self.timestep_bias_strategy
        timestep_bias_multiplier = self.timestep_bias_multiplier
        timestep_bias_portion = self.timestep_bias_portion
        timestep_bias_begin = self.timestep_bias_begin
        timestep_bias_end = self.timestep_bias_end
        # Determine the indices to bias
        num_to_bias = int(timestep_bias_portion * num_timesteps)

        if timestep_bias_strategy == "later":
            bias_indices = slice(-num_to_bias, None)
        elif timestep_bias_strategy == "earlier":
            bias_indices = slice(0, num_to_bias)
        elif timestep_bias_strategy == "range":
            # Out of the possible 1000 timesteps, we might want to focus on eg. 200-500.
            range_begin = timestep_bias_begin
            range_end = timestep_bias_end
            if range_begin < 0:
                msg = "When using the range strategy for timestep bias, you must provide a beginning timestep greater or equal to zero."
                raise ValueError(msg)
            if range_end > num_timesteps:
                msg = (
                    "When using the range strategy for timestep bias, "
                    "you must provide an ending timestep smaller than the number of timesteps."
                )
                raise ValueError(msg)
            bias_indices = slice(range_begin, range_end)
        else:  # 'none' or any other string
            return weights
        if timestep_bias_multiplier <= 0:
            msg = (
                "The parameter --timestep_bias_multiplier is not intended to be used to disable the training of specific timesteps."
                " If it was intended to disable timestep bias, use `--timestep_bias_strategy none` instead."
                " A timestep bias multiplier less than or equal to 0 is not allowed."
            )
            raise ValueError(msg)

        # Apply the bias
        weights[bias_indices] *= timestep_bias_multiplier

        # Normalize
        weights /= weights.sum()

        return weights

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
            img_latents=batch.img_latents.to(self.accelerator.device),
            prompt_embeds_1=prompt_embeds_1,
            prompt_embeds_2=prompt_embeds_2,
            prompt_embeds_pooled_2=prompt_embeds_pooled_2,
            time_ids=time_ids,
        )

    def create_prompts_str(self, batch: DiffusionBatch) -> list[str]:
        shuffle_tags = False
        prompts = []
        for caption, tags in zip(batch.caption, batch.tags, strict=True):
            if not tags:
                prompt = caption
            else:
                if shuffle_tags:
                    random.shuffle(tags)
                prompt = ",".join(tags) if not caption else caption + "," + ",".join(tags)
            prompts.append(prompt)
        return prompts

    def get_prompt_embeds_1(self, prompts_str: list[str]) -> torch.Tensor:
        text_inputs = self.pipeline.tokenizer(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs["input_ids"]
        prompt_embeds_output = self.pipeline.text_encoder(
            text_input_ids.to(self.text_encoder_1.device),
            output_hidden_states=True,
        )

        # use the second to last hidden state as the prompt embedding
        return prompt_embeds_output.hidden_states[-2]

    def get_prompt_embeds_2(self, prompts_str: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        text_inputs_2 = self.pipeline.tokenizer_2(
            prompts_str,
            padding="max_length",
            max_length=self.pipeline.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs_2["input_ids"]
        prompt_embeds_output_2 = self.pipeline.text_encoder_2(
            text_input_ids_2.to(self.text_encoder_2.device),
            output_hidden_states=True,
        )

        # We are only interested in the pooled output of the final text encoder
        prompt_embeds_pooled_2 = prompt_embeds_output_2[0]

        # use the second to last hidden state as the prompt embedding
        prompt_embeds = prompt_embeds_output_2.hidden_states[-2]
        return prompt_embeds, prompt_embeds_pooled_2

    def sample_noise(self, img_latents: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(img_latents)
        if self.noise_offset:
            # Add noise to the image latents
            # https://www.crosslabs.org//blog/diffusion-with-offset-noise
            noise += self.noise_offset * torch.randn(
                (img_latents.shape[0], img_latents.shape[1], 1, 1),
                device=img_latents.device,
            )

        return noise

    def sample_timesteps(self, batch_size: int) -> torch.IntTensor:
        num_timesteps: int = self.noise_scheduler.config.get("num_train_timesteps", 1000)
        if self.timestep_bias_strategy == "none":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.accelerator.device)
        else:
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = self.generate_timestep_weights(num_timesteps).to(self.accelerator.device)
            timesteps = torch.multinomial(weights, batch_size, replacement=True).long()
        return timesteps  # type: ignore

    def __call__(self) -> None:
        """Run the finetuning process."""
        self.train()
