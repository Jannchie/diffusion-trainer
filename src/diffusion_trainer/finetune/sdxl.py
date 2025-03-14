"""Fintunner for Stable Diffusion XL model."""

import math
import random
from contextlib import nullcontext
from dataclasses import dataclass
from logging import getLogger
from pathlib import Path
from typing import Any, Literal, NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from accelerate.logging import get_logger
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel, compute_snr, free_memory
from diffusers.utils.torch_utils import is_compiled_module
from lycoris import LycorisNetwork, create_lycoris
from torch.utils.data import DataLoader
from transformers import CLIPTextModel, CLIPTextModelWithProjection

from diffusion_trainer.config import SampleOptions, SDXLConfig
from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch, DiffusionDataset
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import LatentsGenerateProcessor
from diffusion_trainer.dataset.processors.tagging_processor import TaggingProcessor
from diffusion_trainer.finetune.utils import format_size, get_sample_options_hash, load_pipeline, prepare_accelerator, str_to_dtype
from diffusion_trainer.shared import get_progress
from diffusion_trainer.utils.timestep_weights import logit_timestep_weights

logger = getLogger("diffusion_trainer.finetune.sdxl")


def apply_lora_config(mode: Literal["lora", "loha", "lokr"], model: torch.nn.Module) -> LycorisNetwork:
    if mode == "lokr":
        lycoris_config = {
            "algo": "lokr",
            "multiplier": 1.0,
            "linear_dim": 10000,  # Full dimension
            "linear_alpha": 1,  # Ignored in full dimension
            "factor": 16,
        }
        LycorisNetwork.apply_preset(
            {
                "target_module": ["Attention", "FeedForward"],
                "module_algo_map": {
                    "Attention": {"factor": 16},
                    "FeedForward": {"factor": 8},
                },
            },
        )
        lycoris_model = create_lycoris(
            model,
            **lycoris_config,
        )
        lycoris_model.apply_to()
        lycoris_num_params = sum(p.numel() for p in lycoris_model.parameters())
        logger.info(
            "LyCORIS network has been initialized with %s parameters (%s)",
            f"{lycoris_num_params:,}",
            format_size(lycoris_num_params),
        )
        return lycoris_model
    raise NotImplementedError


class ParamDict(TypedDict):
    lr: float
    params: list[torch.Tensor]


class TrainableModel(NamedTuple):
    model: torch.nn.Module
    lr: float


def get_n_params(trainable_parameters: list[ParamDict]) -> int:
    n_params = 0
    for param in trainable_parameters:
        n_params += sum(p.numel() for p in param["params"])
    return n_params


def prepare_params(accelerator: Accelerator, model: torch.nn.Module, lr: float) -> ParamDict:
    params = ParamDict(
        params=list(filter(lambda p: p.requires_grad, model.parameters())),
        lr=lr,
    )
    logger.info("%s learning rate: %s, number of parameters: %s", model.__class__.__name__, lr, format_size(get_n_params([params])))
    accelerator.prepare(model)
    return params


def get_trainable_parameter_dicts(accelerator: Accelerator, trainable_mdels: list[TrainableModel]) -> list[ParamDict]:
    return [prepare_params(accelerator, model.model, model.lr) for model in trainable_mdels]


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model  # type: ignore # noqa: SLF001


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


class SDXLTuner:
    """Finetune Stable Diffusion XL model."""

    @staticmethod
    def from_config(config: dict) -> "SDXLTuner":
        """Create a new instance from a configuration dictionary."""
        return SDXLTuner(**config)

    def __init__(self, config: SDXLConfig) -> None:
        """Initialize."""
        self.config = config
        self.apply_seed_settings()

        self.model_path = Path(config.model_path)

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

        self.pipeline = load_pipeline(self.model_path, self.weight_dtype)

        sdxl_models = SDXLModels(
            unet=self.pipeline.unet.to(self.device, dtype=self.weight_dtype),
            text_encoder_1=self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype),
            text_encoder_2=self.pipeline.text_encoder_2.to(self.device, dtype=self.weight_dtype),
        )

        self.models: list[Any] = list(sdxl_models)

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

        self.training_models = [m.model for m in self.trainable_models_with_lr]

        for key, value in config.__dict__.items():
            self.logger.info("%s: %s", key, value)

    def apply_seed_settings(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        torch.manual_seed(seed)

    def init_ema(self, model: torch.nn.Module, config: dict) -> None:
        # Create EMA for the unet.
        self.ema_unet: None | EMAModel = None
        if self.use_ema:
            self.ema_unet = EMAModel(
                parameters=model.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=config,
            ).to(self.device)

    def init_gradient_checkpointing(self) -> None:
        if self.config.gradient_checkpointing:
            for model in self.models:
                m: Any = unwrap_model(self.accelerator, model)
                if hasattr(m, "enable_gradient_checkpointing"):
                    m.enable_gradient_checkpointing()
                elif hasattr(m, "gradient_checkpointing_enable"):
                    m.gradient_checkpointing_enable()
                else:
                    logger.warning("cannot checkpointing!")

    def prepare_dataset(self) -> DiffusionDataset:
        if self.config.meta_path:
            meta_path = Path(self.config.meta_path)
        elif self.config.image_path:
            meta_path = Path(self.config.image_path) / "metadata"
        else:
            msg = "Please specify the meta path in the config file."
            raise ValueError(msg)
        parquet_path = meta_path / "metadata.parquet"
        with self.accelerator.main_process_first():
            if not self.accelerator.is_main_process:
                # Wait for the main process to finish preparing, then load the dataset.
                return DiffusionDataset.from_parquet(parquet_path)
            if self.config.image_path and self.config.skip_prepare_image is False:
                self.logger.info("Prepare image from %s", self.config.image_path)
                if not self.config.vae_path:
                    msg = "Please specify the vae_path in the config file."
                    raise ValueError(msg)

                if not self.config.meta_path:
                    self.config.meta_path = (Path(self.config.image_path) / "metadata").as_posix()
                    self.logger.info("Metadata path not set. Using %s as metadata path.", self.config.meta_path)

                vae_dtype = str_to_dtype(self.config.vae_dtype)
                latents_processor = LatentsGenerateProcessor(
                    vae_path=self.config.vae_path,
                    img_path=self.config.image_path,
                    meta_path=self.config.meta_path,
                    vae_dtype=vae_dtype,
                )
                latents_processor()

                tagging_processor = TaggingProcessor(img_path=self.config.image_path, num_workers=1)
                tagging_processor()

            if not self.config.meta_path:
                msg = "Please specify the meta path in the config file."
                raise ValueError(msg)
            if not parquet_path.exists():
                self.logger.info('Creating parquet file at "%s"', parquet_path)
                CreateParquetProcessor(meta_dir=self.config.meta_path)(max_workers=8)
            else:
                self.logger.info('found parquet file at "%s"', parquet_path)
        return DiffusionDataset.from_parquet(parquet_path)

    def train(self) -> None:
        self.accelerator.wait_for_everyone()
        dataset = self.prepare_dataset()
        if self.accelerator.is_main_process:
            dataset.print_bucket_info()

        self.freeze_model()
        self.init_ema(self.pipeline.unet, self.pipeline.unet.config)

        self.trainable_parameters_dicts = get_trainable_parameter_dicts(self.accelerator, self.trainable_models_with_lr)
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

        optimizer = self.initialize_optimizer()

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

        self.init_gradient_checkpointing()

        self.log_training_parameters()

        self.accelerator.init_trackers(f"diffusion-trainer-{self.mode}", config=self.config.__dict__)
        self.execute_training_epoch(num_update_steps_per_epoch, n_total_steps)

    def execute_training_epoch(  # noqa: C901, PLR0912, PLR0915
        self,
        num_update_steps_per_epoch: int,
        n_total_steps: int,
    ) -> None:
        progress = get_progress()

        self.checkpointing_path = self.save_path / "state"
        self.global_steps_file = self.checkpointing_path / "global_steps"

        try:
            self.accelerator.load_state(self.checkpointing_path.as_posix())
            global_step = int(self.global_steps_file.read_text())
        except Exception:
            global_step = 0

        skiped_epoch = math.floor(global_step / num_update_steps_per_epoch)
        # no need to divide by gradient_accumulation_steps
        skiped_batch = global_step * self.accelerator.num_processes * self.config.batch_size
        if global_step != 0:
            self.logger.info(
                "skiping %d global steps (%d batches)",
                global_step,
                skiped_batch,
            )
        skiped_data_loader = self.accelerator.skip_first_batches(self.data_loader, skiped_batch % len(self.data_loader))

        self.logger.info("full_loader_length: %d", len(self.data_loader))
        self.logger.info("skiped_loader_length: %d", len(skiped_data_loader))

        total_task = progress.add_task(
            "Total Progress",
            total=n_total_steps,
            completed=global_step,
        )

        if self.accelerator.is_main_process:
            progress.start()
        for epoch in range(skiped_epoch, self.config.n_epochs):
            self.train_loss = 0.0
            current_epoch_task = progress.add_task(
                f"Epoch {epoch+1}",
                total=num_update_steps_per_epoch,
                completed=global_step % num_update_steps_per_epoch,
            )
            dl = skiped_data_loader if epoch == skiped_epoch else self.data_loader
            for _step, orig_batch in enumerate(dl):
                if not isinstance(orig_batch, DiffusionBatch):
                    msg = f"Expected DiffusionBatch, got something else. Got: {type(orig_batch)}"
                    raise TypeError(msg)
                batch = self.process_batch(orig_batch)

                with self.accelerator.accumulate(self.training_models):
                    self.train_each_batch(batch)

                if self.accelerator.sync_gradients:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    if self.ema_unet:
                        self.ema_unet.step(self.pipeline.unet.parameters())

                    global_step += 1
                    self.accelerator.log({"train_loss": self.train_loss, "lr": current_lr}, step=global_step)
                    self.train_loss = 0.0

                    if self.accelerator.is_main_process:
                        current_completed = global_step % num_update_steps_per_epoch
                        progress.update(total_task, completed=global_step, description=f"Epoch: {epoch+1}")
                        progress.update(current_epoch_task, completed=current_completed, description=f"Lr: {current_lr:.2e}")

                    if self.save_every_n_steps and global_step % self.save_every_n_steps == 0 and global_step != 0:
                        self.saving_model(f"{self.model_name}-step{global_step}")
                    if self.config.checkpoint_every_n_steps and global_step % self.config.checkpoint_every_n_steps == 0:
                        self.accelerator.save_state(self.checkpointing_path.as_posix())
                        self.global_steps_file.write_text(str(global_step))
                    if self.config.preview_every_n_steps and global_step % self.config.preview_every_n_steps == 0:
                        self.generate_preview(f"{self.model_name}-step{global_step}")
            if self.save_every_n_epochs and epoch % self.save_every_n_epochs == 0:
                self.saving_model(f"{self.model_name}-ep{epoch+1}")
            if self.config.preview_every_n_epochs and epoch % self.config.preview_every_n_epochs == 0:
                self.generate_preview(f"{self.model_name}-ep{epoch+1}")
            progress.remove_task(current_epoch_task)
        if self.accelerator.is_main_process:
            progress.stop()
        self.saving_model(f"{self.model_name}")

    @torch.no_grad()
    def generate_preview(self, filename: str) -> None:
        free_memory()

        def callback_on_step_end(_pipe: StableDiffusionXLPipelineOutput, _step: int, _timestep: int, _kwargs: dict) -> dict:
            return {}

        state = PartialState()
        self.accelerator.wait_for_everyone()
        with state.split_between_processes(self.config.preview_sample_options) as sample_options:
            for sample_option in sample_options:
                if not isinstance(sample_option, SampleOptions):
                    msg = f"Expected SampleOption, got {type(sample_option)}"
                    raise TypeError(msg)
                hash_hex = get_sample_options_hash(sample_option)
                filename_with_hash = f"{filename}-{hash_hex}"
                self.logger.info("Generating preview for %s", filename_with_hash)
                autocast_ctx = nullcontext() if torch.backends.mps.is_available() else torch.autocast(self.accelerator.device.type)
                generator = torch.Generator(device=self.accelerator.device).manual_seed(sample_option.seed)
                with autocast_ctx:
                    self.pipeline.to(self.accelerator.device)
                    result = self.pipeline(
                        prompt=sample_option.prompt,
                        negative_prompt=sample_option.negative_prompt,
                        num_inference_steps=sample_option.steps,
                        generator=generator,
                        callback_on_step_end=callback_on_step_end,  # type: ignore
                    )
                self.logger.info("Preview generated for %s", filename_with_hash)

                path = (Path(self.save_path) / "previews" / filename_with_hash).with_suffix(".png")
                path.parent.mkdir(parents=True, exist_ok=True)
                if isinstance(result, StableDiffusionXLPipelineOutput):
                    result.images[0].save(path)
                    if self.config.log_with == "wandb":
                        import wandb

                        self.accelerator.log(
                            {
                                f"{hash_hex}": [wandb.Image(result.images[0], caption=f"{sample_option.prompt}")],
                            },
                        )
                else:
                    msg = f"Expected StableDiffusionXLPipelineOutput, got {type(result)}"
                    raise TypeError(msg)
        free_memory()
        self.accelerator.wait_for_everyone()

    def initialize_optimizer(self) -> torch.optim.Optimizer:
        if self.config.optimizer == "adamW8bit":
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(
                self.trainable_parameters_dicts,
                # lr=self.unet_lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-6,
            )
        else:
            optimizer = torch.optim.Adafactor(
                self.trainable_parameters_dicts,  # type: ignore
                # lr=self.unet_lr,
            )
        return optimizer

    def saving_model(self, filename: str) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            if self.mode in ("lora", "lokr", "loha"):
                self.save_lora_model(filename)
            else:
                self.save_full_finetune_model(filename)

    def save_lora_model(self, filename: str) -> None:
        out_path = self.save_path / f"{filename}.safetensors"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        lycoris_network = next(m for m in self.models if isinstance(m, LycorisNetwork))
        lycoris_network.save_weights(self.save_path / f"{filename}.safetensors", dtype=self.save_dtype, metadata={})

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

    def freeze_model(self) -> None:
        for model in self.models:
            if model in self.training_models:
                model.requires_grad_(True)
            else:
                model.requires_grad_(False)
        self.accelerator.wait_for_everyone()

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

    def create_prompts_str(self, batch: DiffusionBatch) -> list[str]:
        prompts = []
        caption_dropout_ratio = self.config.caption_dropout
        all_tags_dropout_ratio = self.config.all_tags_dropout
        single_tag_dropout_ratio = self.config.single_tag_dropout
        shuffle_tags = self.config.shuffle_tags

        for caption, tags in zip(batch.caption, batch.tags, strict=True):
            # Decide if the caption should be dropped
            true_caption = "" if random.random() < caption_dropout_ratio else caption

            # Decide if all tags should be dropped or process individual tag dropout
            if random.random() < all_tags_dropout_ratio:
                true_tags = []
            elif single_tag_dropout_ratio > 0:
                true_tags = [tag for tag in tags if random.random() >= single_tag_dropout_ratio]
            else:
                true_tags = tags

            # Shuffle tags if necessary
            if shuffle_tags:
                random.shuffle(true_tags)

            # Create prompt string
            if true_caption and true_tags:
                prompt = f"{true_caption}, " + ", ".join(true_tags)
            elif true_caption:
                prompt = true_caption
            else:
                prompt = ", ".join(true_tags)

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
