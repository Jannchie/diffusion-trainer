"""Fintunner for Stable Diffusion XL model."""

import random
import secrets
from dataclasses import dataclass
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import Literal

import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.training_utils import EMAModel, compute_snr
from rich.status import Status
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


def load_pipeline(path: PathLike | str, device: torch.device, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path = Path(path)
    with Status(f'Loading models from "{path}" to {device}({dtype})'):
        if path.is_dir():
            pipe = StableDiffusionXLPipeline.from_pretrained(path, dtype=dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_single_file(path, dtype=dtype)
        pipe.to(device)
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

        self.vae_dtype = torch.float16
        self.save_dtype = torch.float16
        self.weight_dtype = torch.float16

        self.pipeline = load_pipeline(self.model_path, self.accelerator.device, self.vae_dtype)
        self.prediction_type = None

        self.n_epochs = 10
        self.batch_size = 1
        self.gradient_accumulation_steps = 1

        self.unet_lr = 1e-5
        self.text_encoder_1_lr = 1e-6
        self.text_encoder_2_lr = 1e-6

        self.noise_offset = 0.0

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

        trainable_parameters = self.get_trainable_parameters()
        n_params = self.get_n_params(trainable_parameters)
        # Create EMA for the unet.
        self.ema_unet: None | EMAModel = None
        if self.use_ema:
            unet_copy = self.pipeline.unet.parameters().copy()
            self.ema_unet = EMAModel(
                unet_copy,
                model_cls=UNet2DConditionModel,
                model_config=self.pipeline.unet.config,
            ).to(self.device)

        logger.info("Number of trainable parameters: %s", f"{n_params:,}")

    def get_n_params(self, trainable_parameters: list[torch.nn.Parameter]) -> int:
        return sum(p.numel() for p in trainable_parameters)

    def get_trainable_parameters(self) -> list[torch.nn.Parameter]:
        trainable_parameters = []
        if self.unet_lr != 0:
            trainable_parameters.extend(self.pipeline.unet.parameters())
        if self.text_encoder_1_lr != 0:
            trainable_parameters.extend(self.pipeline.text_encoder.parameters())
        if self.text_encoder_2_lr != 0:
            trainable_parameters.extend(self.pipeline.text_encoder_2.parameters())
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

    def train(self) -> None:
        sampler = BucketBasedBatchSampler(self.dataset, self.batch_size)
        data_loader = DataLoader(self.dataset, batch_sampler=sampler, num_workers=0, collate_fn=self.dataset.collate_fn)
        n_epochs = self.n_epochs
        progress = get_progress()

        optimizer = torch.optim.AdamW(self.get_trainable_parameters(), lr=self.unet_lr)
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.unet_lr,
            total_steps=n_epochs * len(data_loader),
            pct_start=0.1,
            base_momentum=0.9,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=10000.0,
        )

        total_task = progress.add_task("Total Progress", total=n_epochs * len(data_loader))
        global_step = 0
        for epoch in range(n_epochs):
            self.train_loss = 0.0
            for _step, batch in enumerate(progress.track(data_loader, description=f"Epoch {epoch+1}")):
                with self.accelerator.accumulate(*self.training_models), progress:
                    if not isinstance(batch, DiffusionBatch):
                        msg = f"Expected DiffusionBatch, got something else. Got: {type(batch)}"
                        raise TypeError(msg)

                    loss = self.train_each_batch(batch)

                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                if self.accelerator.sync_gradients:
                    if self.ema_unet:
                        self.ema_unet.step(self.pipeline.unet.parameters())

                    progress.update(total_task, advance=1)
                    global_step += 1
                    self.accelerator.log({"train_loss": self.train_loss}, step=global_step)
                    self.train_loss = 0.0

                logs = {"step_loss": loss, "lr": lr_scheduler.get_last_lr()[0]}
                progress.print(f"Step {global_step}: {logs}")

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
        img_latents_with_noise = self.pipeline.scheduler.add_noise(img_latents, noise, timesteps)

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
            params_to_clip = self.pipeline.unet.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.max_grad_norm)
        return loss.detach().item()

    def get_loss(self, timesteps: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.snr_gamma is None:
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        else:
            # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
            # Since we predict the noise instead of x_0, the original formulation is slightly changed.
            # This is discussed in Section 4.2 of the same paper.
            noise_scheduler = self.pipeline.scheduler
            snr = compute_snr(noise_scheduler, timesteps)
            mse_loss_weights = torch.stack([snr, self.snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0]
            if noise_scheduler.config.prediction_type == "epsilon":
                mse_loss_weights = mse_loss_weights / snr
            elif noise_scheduler.config.prediction_type == "v_prediction":
                mse_loss_weights = mse_loss_weights / (snr + 1)

            loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
            loss = loss.mean()
        return loss

    def get_pred_target(
        self,
        img_latents: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        model_pred: torch.Tensor,
    ) -> torch.Tensor:
        noise_scheduler = self.pipeline.scheduler
        # Get the target for loss depending on the prediction type
        if self.prediction_type is not None:
            # set prediction_type of scheduler if defined
            self.pipeline.register_to_config(prediction_type=self.prediction_type)
        if noise_scheduler.config.prediction_type == "epsilon":
            target = noise
        elif noise_scheduler.config.prediction_type == "v_prediction":
            target = noise_scheduler.get_velocity(img_latents, noise, timesteps)
        elif noise_scheduler.config.prediction_type == "sample":
            # We set the target to latents here, but the model_pred will return the noise sample prediction.
            target = img_latents
            # We will have to subtract the noise residual from the prediction to get the target sample.
            model_pred = model_pred - noise
        else:
            msg = f"Unknown prediction type {noise_scheduler.config.prediction_type}"
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

        # 利用 batch.original_size, batch.crop_ltrb, batch.train_resolution， 获取 item ids
        # 每行分别为 original_size[1], original_size[0], crop_ltrb[0], crop_ltrb[1], crop_ltrb[0], train_resolution[1], train_resolution[0]

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
            text_input_ids.to(self.pipeline.text_encoder.device),
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
            text_input_ids_2.to(self.pipeline.text_encoder_2.device),
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

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        num_timesteps: int = self.pipeline.scheduler.config.num_train_timesteps
        if self.timestep_bias_strategy == "none":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.accelerator.device)
        else:
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = self.generate_timestep_weights(num_timesteps).to(self.accelerator.device)
            timesteps = torch.multinomial(weights, batch_size, replacement=True).long()
        return timesteps

    def __call__(self) -> None:
        """Run the finetuning process."""
        self.train()
