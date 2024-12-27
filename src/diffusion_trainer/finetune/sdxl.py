"""Fintunner for Stable Diffusion XL model."""

import math
import os
import random
from contextlib import nullcontext, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from logging import getLogger
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypedDict

import torch
import torch.nn.functional as F
from accelerate import Accelerator, PartialState
from diffusers.models.unets.unet_2d_condition import UNet2DConditionModel
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.pipelines.stable_diffusion_xl.pipeline_stable_diffusion_xl import StableDiffusionXLPipeline
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel, compute_snr, free_memory
from diffusers.utils.torch_utils import is_compiled_module
from lycoris import LycorisNetwork, create_lycoris
from peft.tuners.loha.config import LoHaConfig
from peft.tuners.lora.config import LoraConfig
from torch.utils.data import DataLoader

from diffusion_trainer.config import SampleOptions, SDXLConfig
from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch, DiffusionDataset
from diffusion_trainer.finetune.utils import prepare_accelerator, str_to_dtype
from diffusion_trainer.shared import get_progress

if TYPE_CHECKING:
    from transformers import CLIPTextModel, CLIPTextModelWithProjection

logger = getLogger("diffusion_trainer.finetune.sdxl")


def format_size(num: int) -> str:
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.1f} B"
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f} M"
    if num >= 1_000:
        return f"{num / 1_000:.1f} K"
    return str(num)


def unwrap_model(accelerator: Accelerator, model: torch.nn.Module) -> torch.nn.Module:
    model = accelerator.unwrap_model(model)
    return model._orig_mod if is_compiled_module(model) else model  # noqa: SLF001


@dataclass
class SDXLBatch:
    img_latents: torch.Tensor
    prompt_embeds_1: torch.Tensor
    prompt_embeds_2: torch.Tensor
    prompt_embeds_pooled_2: torch.Tensor
    time_ids: torch.Tensor


class DummyProgressBar:
    def __init__(self, total: int) -> None:
        pass

    def __enter__(self) -> "DummyProgressBar":
        return self

    def __exit__(self, *_args: object) -> None:
        pass

    def update(self) -> None:
        pass


def load_pipeline(path: PathLike | str, dtype: torch.dtype) -> StableDiffusionXLPipeline:
    path = Path(path)

    logger.info('Loading models from "%s" (%s)', path, dtype)
    with Path(os.devnull).open("w") as fnull, redirect_stdout(fnull), redirect_stderr(fnull):
        if path.suffix == ".safetensors":
            pipe = StableDiffusionXLPipeline.from_single_file(path.as_posix(), torch_dtype=dtype)
        else:
            pipe = StableDiffusionXLPipeline.from_pretrained(path.as_posix(), torch_dtype=dtype)
    logger.info("Models loaded successfully.")
    if isinstance(pipe, StableDiffusionXLPipeline):
        pipe.progress_bar = DummyProgressBar  # type: ignore
        return pipe
    msg = "Failed to load models."
    raise ValueError(msg)


class ParamDict(TypedDict):
    lr: float
    params: list[torch.Tensor]


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

        self.unet_lr = config.unet_lr
        self.text_encoder_1_lr = config.text_encoder_1_lr
        self.text_encoder_2_lr = config.text_encoder_2_lr

        self.mode: Literal["full-finetune", "lora", "lokr", "loha"] = config.mode

        self.lora_rank = config.lora_rank
        self.lora_alpha = config.lora_alpha

        self.lokr_factor = config.lokr_factor

        # reduce memory usage by checkpointing the gradients
        self.gradient_checkpointing = config.gradient_checkpointing

        # The timestep bias strategy, which may help direct the model toward learning low or high frequency details.
        # The default value is 'none', which means no bias is applied.
        # The value of 'later' will increase the frequency of the model's final training timesteps.
        self.timestep_bias_strategy = config.timestep_bias_strategy
        # The multiplier for the bias. Defaults to 1.0, which means no bias is applied.
        # A value of 2.0 will double the weight of the bias, and a value of 0.5 will halve it.
        self.timestep_bias_multiplier = config.timestep_bias_multiplier
        # The portion of timesteps to bias. Defaults to 0.25, which 25% of timesteps will be biased.
        # A value of 0.5 will bias one half of the timesteps. The value provided for `--timestep_bias_strategy` determines
        # whether the biased portions are in the earlier or later timesteps.
        self.timestep_bias_portion = config.timestep_bias_portion
        # When using `--timestep_bias_strategy=range`, the beginning (inclusive) timestep to bias.
        # Defaults to zero, which equates to having no specific bias.
        self.timestep_bias_begin = config.timestep_bias_begin
        # When using `--timestep_bias_strategy=range`, the final timestep (inclusive) to bias.
        # Defaults to 1000, which is the number of timesteps that Stable Diffusion is trained on.
        self.timestep_bias_end = config.timestep_bias_end

        self.use_ema = config.use_ema

        self.save_every_n_epochs = config.save_every_n_epochs
        self.save_every_n_steps = config.save_every_n_steps

        self.mixed_precision = str_to_dtype(config.mixed_precision)

        self.pipeline = load_pipeline(self.model_path, self.weight_dtype)
        self.accelerator = prepare_accelerator(self.gradient_accumulation_steps, self.mixed_precision, self.config.log_with)
        self.device = self.accelerator.device
        self.init_model_modules()
        self.init_ema()

        for key, value in config.__dict__.items():
            logger.info("%s: %s", key, value)

    def apply_seed_settings(self) -> None:
        seed = self.config.seed
        random.seed(seed)
        torch.manual_seed(seed)

    def init_ema(self) -> None:
        # Create EMA for the unet.
        self.ema_unet: None | EMAModel = None
        if self.use_ema:
            self.ema_unet = EMAModel(
                parameters=self.unet.parameters(),
                model_cls=UNet2DConditionModel,
                model_config=self.unet.config,
            ).to(self.device)

    def init_model_modules(self) -> None:
        self.unet: UNet2DConditionModel = self.pipeline.unet.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_1: CLIPTextModel = self.pipeline.text_encoder.to(self.device, dtype=self.weight_dtype)
        self.text_encoder_2: CLIPTextModelWithProjection = self.pipeline.text_encoder_2.to(self.device, dtype=self.weight_dtype)
        self.vae = self.pipeline.vae

    def init_gradient_checkpointing(self) -> None:
        if self.gradient_checkpointing:
            unwrap_model(self.accelerator, self.unet).enable_gradient_checkpointing()
            unwrap_model(self.accelerator, self.text_encoder_1).gradient_checkpointing_enable()
            unwrap_model(self.accelerator, self.text_encoder_2).gradient_checkpointing_enable()
            unwrap_model(self.accelerator, self.lycoris_model).enable_gradient_checkpointing()

    def get_n_params(self, trainable_parameters: list[ParamDict]) -> int:
        n_params = 0
        for param in trainable_parameters:
            n_params += sum(p.numel() for p in param["params"])
        return n_params

    def get_trainable_parameter_dicts(self) -> list[ParamDict]:
        trainable_parameters = []
        if self.unet_lr != 0 and self.mode == "full-finetune":
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.unet.parameters())),
                    "lr": self.unet_lr,
                },
            )
            logger.info(
                "UNet learning rate: %s, number of parameters: %s",
                self.unet_lr,
                format_size(self.get_n_params([trainable_parameters[-1]])),
            )
            self.accelerator.prepare(self.unet)
        if self.text_encoder_1_lr != 0:
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.text_encoder_1.parameters())),
                    "lr": self.text_encoder_1_lr,
                },
            )
            logger.info(
                "Text Encoder 1 learning rate: %s, number of parameters: %s",
                self.text_encoder_1_lr,
                format_size(self.get_n_params([trainable_parameters[-1]])),
            )
            self.accelerator.prepare(self.text_encoder_1)
        if self.text_encoder_2_lr != 0:
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.text_encoder_2.parameters())),
                    "lr": self.text_encoder_2_lr,
                },
            )
            logger.info(
                "Text Encoder 2 learning rate: %s, number of parameters: %s",
                self.text_encoder_2_lr,
                format_size(self.get_n_params([trainable_parameters[-1]])),
            )
            self.accelerator.prepare(self.text_encoder_2)
        if self.mode in ("lokr", "loha") and self.lycoris_model:
            trainable_parameters.append(
                {
                    "params": list(filter(lambda p: p.requires_grad, self.lycoris_model.parameters())),
                    "lr": self.unet_lr,
                },
            )
            logger.info(
                "LyCORIS network learning rate: %s, number of parameters: %s",
                self.unet_lr,
                format_size(self.get_n_params([trainable_parameters[-1]])),
            )
            self.accelerator.prepare(self.unet)
            self.unet.train()
        return trainable_parameters

    @property
    def training_models(self) -> list[torch.nn.Module]:
        """Get the training model."""
        model = []
        if self.unet_lr != 0 and self.mode == "full-finetune":
            model.append(self.pipeline.unet)
        if self.text_encoder_1_lr != 0:
            model.append(self.pipeline.text_encoder)
        if self.text_encoder_2_lr != 0:
            model.append(self.pipeline.text_encoder_2)
        if self.lycoris_model:
            model.append(self.lycoris_model)
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

            lycoris_config = {
                "algo": "lokr",
                "multiplier": 1.0,
                "linear_dim": 10000,  # Full dimension
                "linear_alpha": 1,  # Ignored in full dimension
                "factor": 16,
                "apply_preset": {
                    "target_module": ["Attention", "FeedForward"],
                    "module_algo_map": {
                        "Attention": {"factor": 16},
                        "FeedForward": {"factor": 8},
                    },
                },
            }
            LycorisNetwork.apply_preset(lycoris_config.get("apply_preset"))

            self.lycoris_model = create_lycoris(
                self.unet,
                **lycoris_config,
            )
            self.lycoris_model.apply_to()
            lycoris_num_params = sum(p.numel() for p in self.lycoris_model.parameters())
            self.lycoris_model = self.lycoris_model.to(self.accelerator.device, dtype=self.weight_dtype)
            logger.info(
                "LyCORIS network has been initialized with %s parameters (%s)",
                f"{lycoris_num_params:,}",
                format_size(lycoris_num_params),
            )
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

    def train(self, dataset: DiffusionDataset) -> None:
        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            dataset.print_bucket_info()

        self.freeze_model()
        self.apply_lora_config()

        self.trainable_parameters_dicts = self.get_trainable_parameter_dicts()
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

        logger.info("Noise scheduler config:")
        for key, value in self.noise_scheduler.config.items():
            logger.info("- %s: %s", key, value)

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
            num_warmup_steps=self.config.optimizer_warmup_steps,
            num_training_steps=n_total_steps,
            num_cycles=self.config.optimizer_num_cycles,
        )

        self.optimizer = self.accelerator.prepare(optimizer)
        self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

        self.init_gradient_checkpointing()

        self.log_training_parameters()

        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(f"diffusion-trainer-{self.mode}", config=self.config.__dict__)
        self.execute_training_epoch(num_update_steps_per_epoch, n_total_steps)

    def execute_training_epoch(  # noqa: C901, PLR0912
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
        if global_step != 0:
            logger.info("skiping %d steps", global_step)
        skiped_data_loader = self.accelerator.skip_first_batches(self.data_loader, global_step % num_update_steps_per_epoch)
        total_task = progress.add_task("Total Progress", total=n_total_steps, completed=global_step)

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

                with self.accelerator.accumulate([self.unet]):
                    loss = self.train_each_batch(batch)

                if self.accelerator.sync_gradients:
                    current_lr = self.lr_scheduler.get_last_lr()[0]
                    if self.ema_unet:
                        self.ema_unet.step(self.unet.parameters())

                    global_step += 1
                    self.accelerator.log({"train_loss": self.train_loss, "lr": current_lr}, step=global_step)
                    self.train_loss = 0.0

                    if self.accelerator.is_main_process:
                        progress.update(total_task, advance=1, description=f"Global Step: {global_step} - Epoch: {epoch+1}")
                        progress.update(current_epoch_task, advance=1, description=f"LR: {current_lr:.2e} - Loss: {loss:.2f}")

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
                filename_hash = f"{filename}-{sample_option.seed}"
                logger.info("Generating preview for %s", filename_hash)
                hash_value = hash(sample_option)
                hash_hex = f"{hash_value:x}"
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
                logger.info("Preview generated for %s", filename_hash)

                path = (Path(self.save_path) / "previews" / f"{filename}-{hash_hex}").with_suffix(".png")
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
                lr=self.unet_lr,
                betas=(0.9, 0.999),
                weight_decay=1e-2,
                eps=1e-6,
            )
        else:
            optimizer = torch.optim.Adafactor(self.trainable_parameters_dicts, lr=self.unet_lr)  # type: ignore
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
        self.lycoris_model.save_weights(self.save_path / f"{filename}.safetensors", dtype=self.save_dtype, metadata={})

    def save_full_finetune_model(self, filename: str) -> None:
        msg = "Full finetune model saving is not implemented yet."
        raise NotImplementedError(msg)

    def log_training_parameters(self) -> None:
        if self.accelerator.is_main_process:
            n_params = self.get_n_params(self.trainable_parameters_dicts)
            logger.info("Number of epochs: %s", self.config.n_epochs)
            num_processes = self.accelerator.num_processes
            logger.info("Unet: %s %s", self.unet.device, self.unet.dtype)
            logger.info("Text Encoder 1: %s %s", self.text_encoder_1.device, self.text_encoder_1.dtype)
            logger.info("Text Encoder 2: %s %s", self.text_encoder_2.device, self.text_encoder_2.dtype)
            effective_batch_size = self.batch_size * num_processes * self.gradient_accumulation_steps
            logger.info("Effective batch size: %s", effective_batch_size)
            logger.info("Prediction type: %s", self.noise_scheduler.config.get("prediction_type"))
            logger.info("Number of trainable parameters: %s (%s)", f"{n_params:,}", format_size(n_params))
            logger.info("Hold tight, training is about to start!")

    def freeze_model(self) -> None:
        if self.mode in ("lora", "lokr", "loha"):
            logger.info("Training with %s, freezing the models.", self.mode)
            self.unet.requires_grad_(False)
            self.text_encoder_1.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
        elif self.mode == "full-finetune":
            if self.unet_lr:
                logger.info("Training the UNet with learning rate %s", self.unet_lr)
                self.unet.requires_grad_(True)
            else:
                self.unet.requires_grad_(False)
            if self.text_encoder_1_lr:
                logger.info("Training the text encoder 1 with learning rate %s", self.text_encoder_1_lr)
                self.text_encoder_1.requires_grad_(True)
            else:
                self.text_encoder_1.requires_grad_(False)
            if self.text_encoder_2_lr:
                logger.info("Training the text encoder 2 with learning rate %s", self.text_encoder_2_lr)
                self.text_encoder_2.requires_grad_(True)
            else:
                self.text_encoder_2.requires_grad_(False)
        else:
            msg = f"Unknown mode {self.mode}"
            raise ValueError(msg)
        self.accelerator.wait_for_everyone()

    def train_each_batch(self, batch: SDXLBatch) -> float:
        img_latents = self.vae.config.get("scaling_factor", 0) * batch.img_latents.to(self.accelerator.device)
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
            logger.info("img_latents: %s %s", img_latents.shape, img_latents.mean())
            logger.info("prompt_embeds: %s %s", prompt_embeds.shape, prompt_embeds.mean())
            logger.info("prompt_embeds_pooled_2: %s %s", prompt_embeds_pooled_2.shape, prompt_embeds_pooled_2.mean())
            logger.info("time_ids: %s %s", time_ids.shape, time_ids)
            logger.info("noise: %s %s", noise.shape, noise.mean())
            logger.info("timesteps: %s %s", timesteps.shape, timesteps)
            logger.info("img_noisy_latents: %s %s", img_noisy_latents.shape, img_noisy_latents.mean())
            logger.info("model_pred: %s %s", model_pred.shape, model_pred.mean())
            logger.info("target: %s %s", target.shape, target.mean())
            msg = "Loss is NaN."
            raise ValueError(msg)

        avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()  # type: ignore
        self.train_loss += avg_loss.item() / self.gradient_accumulation_steps

        self.accelerator.backward(loss)

        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
            params_to_clip = self.unet.parameters()
            self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

        self.optimizer.step()
        # ref: https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)

        self.lr_scheduler.step()

        return loss.detach().item()

    def get_model_pred(
        self,
        img_noisy_latents: torch.Tensor,
        timesteps: torch.Tensor,
        prompt_embeds: torch.Tensor,
        unet_added_conditions: dict,
    ) -> torch.Tensor:
        return self.unet(
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
            img_latents=batch.img_latents,
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
        text_input_ids = text_inputs["input_ids"].to(self.accelerator.device)
        prompt_embeds_output = self.text_encoder_1(
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
        prompt_embeds_output_2 = self.text_encoder_2(
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
        if self.timestep_bias_strategy == "none":
            # Sample a random timestep for each image without bias.
            timesteps = torch.randint(0, num_timesteps, (batch_size,), device=self.accelerator.device)
        else:
            # Sample a random timestep for each image, potentially biased by the timestep weights.
            # Biasing the timestep weights allows us to spend less time training irrelevant timesteps.
            weights = self.generate_timestep_weights(num_timesteps).to(self.accelerator.device)
            timesteps = torch.multinomial(weights, batch_size, replacement=True).long()
        return timesteps  # type: ignore

    def __call__(self, *, dataset: DiffusionDataset) -> None:
        """Run the finetuning process."""
        self.train(dataset)
