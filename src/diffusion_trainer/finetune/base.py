import json
import logging
import math
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Generator, Sequence
from contextlib import AbstractContextManager, contextmanager, nullcontext
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from accelerate import PartialState
from diffusers.optimization import SchedulerType, get_scheduler
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.stable_diffusion_xl.pipeline_output import StableDiffusionXLPipelineOutput
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.schedulers.scheduling_utils import SchedulerMixin
from diffusers.training_utils import EMAModel, free_memory
from PIL import Image
from torch.utils.data import DataLoader

from diffusion_trainer.config import BaseConfig, SampleOptions
from diffusion_trainer.dataset.dataset import BucketBasedBatchSampler, DiffusionBatch, DiffusionDataset, TagFilters
from diffusion_trainer.dataset.processors.create_parquet_processor import CreateParquetProcessor
from diffusion_trainer.dataset.processors.latents_generate_processor import LatentsGenerateProcessor
from diffusion_trainer.dataset.processors.tagging_processor import TaggingProcessor
from diffusion_trainer.dataset.streaming import StreamingDiffusionDataset
from diffusion_trainer.finetune.objective import DDPMObjective, DiffusionObjective
from diffusion_trainer.finetune.utils import (
    DummyProgressBar,
    TrainableModel,
    get_sample_options_hash,
    get_trainable_parameter_dicts,
    initialize_optimizer,
    prepare_accelerator,
    str_to_dtype,
    unwrap_model,
)
from diffusion_trainer.finetune.utils.lora import UNET_LORA_TARGETS, LoraTargets, apply_lora_config
from diffusion_trainer.shared import get_progress
from diffusion_trainer.utils.advanced_noise import (
    brownian_noise,
    multi_resolution_noise,
    pyramid_noise,
)

if TYPE_CHECKING:
    from lycoris import LycorisNetwork

logger = logging.getLogger("diffusion_trainer")

ModelPredFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_ATTENTION_SYNTAX_RE = re.compile(r"([\\()\[\]])")


def escape_attention_syntax(text: str) -> str:
    """Escape A1111 attention syntax so booru tags like ``ask_(askzy)`` tokenize literally.

    The enhanced embedder (``diffusion_prompt_embedder``) parses ``(...)``/``[...]``
    as attention weights, which would strip the parentheses from booru tags and
    scale their embeddings by 1.1. Escaping keeps training text identical to the
    plain-tokenizer path and to WebUI inference with ``\\(...\\)`` escapes.
    """
    return _ATTENTION_SYNTAX_RE.sub(r"\\\1", text)


@dataclass(frozen=True)
class TagCompositionRules:
    """Category-aware prompt assembly rules derived from the training config."""

    category_order: tuple[str, ...]
    shuffled_categories: frozenset[str]
    droppable_categories: frozenset[str]
    single_tag_dropout: float

    @classmethod
    def from_config(cls, config: BaseConfig) -> "TagCompositionRules":
        return cls(
            category_order=tuple(config.tag_category_order),
            # Folding the global switches in here keeps compose_prompt_tags branch-free.
            shuffled_categories=frozenset(config.shuffled_tag_categories) if config.shuffle_tags else frozenset(),
            droppable_categories=frozenset(config.droppable_tag_categories) if config.single_tag_dropout > 0 else frozenset(),
            single_tag_dropout=config.single_tag_dropout,
        )


def compose_prompt_tags(tags: Sequence[str], categories: Sequence[str], rules: TagCompositionRules) -> list[str]:
    """Assemble one sample's tags in category order with per-category augmentation.

    Categories absent from ``rules.category_order`` (e.g. booru ``meta`` noise)
    never enter the prompt. Shuffle and single-tag dropout apply only inside
    the listed categories, so quality/artist/copyright/character conditioning
    stays pinned at the front while the free-form tail randomizes.

    Datasets without category info (``categories`` empty or misaligned) treat
    every tag as ``general`` — identical to the historical flat behavior.
    """
    if len(categories) != len(tags):
        categories = ["general"] * len(tags)
    grouped: dict[str, list[str]] = {}
    for tag, category in zip(tags, categories, strict=True):
        grouped.setdefault(category, []).append(tag)
    result: list[str] = []
    for category in rules.category_order:
        category_tags = grouped.get(category, [])
        if category in rules.droppable_categories:
            category_tags = [tag for tag in category_tags if random.random() >= rules.single_tag_dropout]
        if category in rules.shuffled_categories:
            random.shuffle(category_tags)  # per-sample lists built above; mutation is local
        result.extend(category_tags)
    return result


class BaseTuner(ABC):
    @staticmethod
    def from_config(_config: BaseConfig) -> "BaseTuner":
        """Create a Tuner from a config."""
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def __init__(self, config: BaseConfig) -> None:
        self.config = config
        self.save_path = Path(config.save_dir)
        self.mixed_precision = str_to_dtype(config.mixed_precision)
        self.weight_dtype = str_to_dtype(config.weight_dtype)
        self.save_dtype = str_to_dtype(config.save_dtype)
        self.vae_dtype = str_to_dtype(config.vae_dtype)
        self.ema_denoiser: EMAModel | None = None
        self.ema_denoiser_short: EMAModel | None = None
        self.prepared_model_map: dict[int, torch.nn.Module] = {}

        # Common initialization steps
        self._initialize_environment()
        self._initialize_accelerator()
        self._initialize_pipeline_and_models()
        self._log_initialization_info()

    def _initialize_environment(self) -> None:
        """Initialize environment settings like CUDA cache.

        Seeding is deferred to ``_initialize_accelerator`` so it can be offset
        per rank once the distributed context exists.
        """
        # Clear CUDA cache before loading models to ensure maximum available memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _initialize_accelerator(self) -> None:
        """Initialize accelerator and device settings."""
        self.accelerator = prepare_accelerator(
            self.config.gradient_accumulation_steps,
            self.mixed_precision,
            self.config.log_with,
            torch_compile=self.config.torch_compile,
        )
        self.device = self.accelerator.device

        # Offset the seed per rank so each process draws independent timesteps,
        # noise and tag shuffle/dropout sequences (kohya convention: seed + rank).
        # No-op for single-process runs where process_index == 0.
        self.apply_seed_settings(self.config.seed + self.accelerator.process_index)

    def _initialize_pipeline_and_models(self) -> None:
        """Initialize pipeline, models, and training-related attributes."""
        self.pipeline = self.get_pipeline()
        self.lycoris_model: LycorisNetwork | None = None
        # pandm run handle; set on the main process when log_with == "pandm".
        self.pandm_run: Any | None = None
        self.noise_scheduler: SchedulerMixin = self.get_noise_scheduler()
        self.objective: DiffusionObjective = self.create_objective()
        self.objective.log_summary()
        self.train_loss = 0.0
        # On-device accumulator for the per-window loss; gathered+synced once per
        # optimizer step instead of once per micro-step (see optimizer_step).
        self._loss_accum: torch.Tensor | None = None
        self._micro_step_count = 0
        self.trainable_models_with_lr: list[TrainableModel] = []
        self.training_models: list[torch.nn.Module] = []
        self.global_step = 0  # Track global training step for input perturbation decay

        # Initialize model-specific components (implemented by subclasses)
        self._setup_models()

        # Enable gradient checkpointing if configured
        if self.config.gradient_checkpointing:
            self._enable_gradient_checkpointing()

    def _setup_models(self) -> None:
        """Setup model-specific components. Must be implemented by subclasses."""
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def create_objective(self) -> DiffusionObjective:
        """Build the training objective (noising, target, loss weighting).

        Defaults to the discrete DDPM formulation the SD 1.5 / SDXL lineage
        uses; flow-matching families override this.
        """
        if not isinstance(self.noise_scheduler, DDPMScheduler):
            msg = f"The default DDPM objective needs a DDPMScheduler, got {type(self.noise_scheduler).__name__}. Override create_objective()."
            raise TypeError(msg)
        return DDPMObjective(self.config, self.noise_scheduler, self.device)

    @property
    def denoiser(self) -> torch.nn.Module:
        """The network that gets trained to denoise: a UNet here, a DiT for Lumina.

        EMA, gradient checkpointing, preview eval-mode handling and LoRA
        targeting all route through this so architectures that don't ship a
        ``pipeline.unet`` stay first-class.
        """
        return self.pipeline.unet

    def _enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for models. Can be overridden by subclasses."""
        models = getattr(self, "models", [])
        if models:
            self.init_gradient_checkpointing(models)

    def _log_initialization_info(self) -> None:
        """Log initialization information."""
        logger.info("Initialized %s with config:", self.__class__.__name__)
        for key, value in self.config.__dict__.items():
            logger.info("  %s: %s", key, value)

    def prepare_training(self, data_loader: torch.utils.data.DataLoader) -> tuple[int, int]:
        """
        Prepares the training process including dataset, trainable models, optimizer and scheduler.
        Returns a tuple containing dataloader, number of update steps per epoch, and total steps.
        """
        self.accelerator.wait_for_everyone()

        # Configure trainable models (subclass-specific implementation)
        self._configure_models()

        trainable_models = [model.model for model in self.trainable_models_with_lr]
        freeze_models = [model for model in getattr(self, "models", []) if model not in trainable_models]
        self.freeze_model(freeze_models)
        self._prepare_trainable_models()
        self.training_models = [model.model for model in self.trainable_models_with_lr]
        for model in self.training_models:
            model.train()

        # Initialize EMA models if enabled. EMA tracks the denoiser's parameters,
        # which only train in full-finetune; in LoRA/LyCORIS modes the base
        # denoiser is frozen, so an EMA over it would shadow constant weights —
        # wasted memory and a no-op. Skip it (and warn) outside full-finetune.
        if self.config.use_ema and self.config.mode != "full-finetune":
            logger.warning("use_ema is ignored in %s mode: the base denoiser is frozen, so EMA over it is a no-op.", self.config.mode)
        elif self.config.use_ema:
            denoiser = self.denoiser
            self.ema_denoiser = self.get_ema(denoiser, denoiser.config, decay=self.config.ema_decay_long)  # type: ignore[attr-defined]
            if self.config.use_dual_ema:
                self.ema_denoiser_short = self.get_ema(denoiser, denoiser.config, decay=self.config.ema_decay_short)  # type: ignore[attr-defined]

        # Prepare optimizer
        self.trainable_parameters_dicts = get_trainable_parameter_dicts(self.trainable_models_with_lr)
        self.optimizer = initialize_optimizer(
            self.config.optimizer,
            self.trainable_parameters_dicts,
            weight_decay=self.config.weight_decay,
        )
        self.optimizer = self.accelerator.prepare(self.optimizer)

        num_update_steps_per_epoch = math.ceil(len(data_loader) / self.config.gradient_accumulation_steps)
        n_total_steps = self.config.n_epochs * num_update_steps_per_epoch

        # accelerate's AcceleratedScheduler advances the underlying scheduler once
        # per process per optimizer step, so the schedule must be expressed in
        # post-prepare units (× num_processes) or an N-GPU run finishes its LR
        # curve N× early. No-op at num_processes == 1. (diffusers examples do the
        # same: num_training_steps = max_train_steps * num_processes.)
        sched_scale = self.accelerator.num_processes
        sched_warmup_steps = self.config.optimizer_warmup_steps * sched_scale
        sched_total_steps = n_total_steps * sched_scale

        # Initialize learning rate scheduler
        # Use different scheduler based on optimizer type
        if self.config.optimizer == "adafactor":
            # For Adafactor, use constant_with_warmup as recommended by SS-Script
            scheduler_type = SchedulerType.CONSTANT_WITH_WARMUP
            self.lr_scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=sched_warmup_steps,
                num_training_steps=sched_total_steps,
            )
        elif self.config.optimizer_restart_decay != 1.0 and self.config.optimizer_num_cycles > 1:
            # SGDR-style restarts with decaying peaks: cycle k restarts at
            # peak * decay^k. diffusers' cosine_with_restarts always returns to
            # the full peak, which periodically re-enters the LR regime that
            # grinds away pretrained high-frequency detail; decaying peaks
            # shrink that damage every cycle while each valley consolidates.
            warmup = sched_warmup_steps
            cycles = self.config.optimizer_num_cycles
            decay = self.config.optimizer_restart_decay

            def sgdr_lambda(step: int) -> float:
                if step < warmup:
                    return step / max(1, warmup)
                progress = (step - warmup) / max(1, sched_total_steps - warmup)
                progress = min(progress, 1.0 - 1e-8)
                cycle = int(progress * cycles)
                cycle_progress = progress * cycles - cycle
                return (decay**cycle) * 0.5 * (1.0 + math.cos(math.pi * cycle_progress))

            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, sgdr_lambda)
        else:
            # For other optimizers, use cosine with restarts
            scheduler_type = SchedulerType.COSINE_WITH_RESTARTS
            self.lr_scheduler = get_scheduler(
                scheduler_type,
                optimizer=self.optimizer,
                num_warmup_steps=sched_warmup_steps,
                num_training_steps=sched_total_steps,
                num_cycles=self.config.optimizer_num_cycles,
            )
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        # Initialize trackers
        self._initialize_trackers(n_total_steps)

        return num_update_steps_per_epoch, n_total_steps

    def _initialize_trackers(self, n_total_steps: int) -> None:
        """Initialize the experiment tracker (pandm run or accelerate trackers)."""
        timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
        if self.config.log_with != "pandm":
            self.accelerator.init_trackers(
                "diffusion-trainer",
                config=self.config.__dict__,
                init_kwargs={"wandb": {"name": timestamp}},
            )
            return

        if not self.accelerator.is_main_process:
            return

        import pandm

        # Round-trip through JSON so non-scalar config entries (paths, nested
        # sample options, ...) serialize instead of erroring.
        config_snapshot = json.loads(json.dumps(self.config.__dict__, default=str))

        # Resume continuity: when a usable accelerate checkpoint exists we reopen
        # the SAME pandm run (id persisted alongside the state) so a restart
        # appends to one timeline instead of spawning a fresh run each time.
        # total_steps declares the training length so the dashboard shows an ETA
        # — it then tracks the per-step log(step=global_step) automatically.
        # (Literal state filenames mirror the resume path in execute_training;
        # they are stable accelerate conventions.)
        state_dir = self.save_path / "state"
        run_id_file = state_dir / "pandm_run_id"
        resuming = (state_dir / "optimizer.bin").exists() and (state_dir / "global_steps").exists()
        saved_run_id = run_id_file.read_text().strip() if resuming and run_id_file.exists() else None

        self.pandm_run = pandm.init(
            project="diffusion-trainer",
            name=f"{self.config.model_name}-{timestamp}",
            config=config_snapshot,
            total_steps=n_total_steps,
            id=saved_run_id,
            resume="allow" if saved_run_id else False,
        )

        # Persist the run id so a future resume reattaches to this run.
        if saved_run_id is None:
            state_dir.mkdir(parents=True, exist_ok=True)
            run_id_file.write_text(self.pandm_run.id)

    def _prepare_trainable_models(self) -> None:
        """Wrap trainable models with accelerator and keep runtime references aligned."""
        if not self.trainable_models_with_lr:
            msg = "No trainable models configured."
            raise ValueError(msg)

        original_models = [item.model for item in self.trainable_models_with_lr]
        prepared_models = self.accelerator.prepare(*original_models)
        prepared_model_list = [prepared_models] if isinstance(prepared_models, torch.nn.Module) else list(prepared_models)

        self.prepared_model_map = {
            id(original): prepared
            for original, prepared in zip(original_models, prepared_model_list, strict=True)
        }
        self.trainable_models_with_lr = [
            TrainableModel(model=prepared, lr=item.lr)
            for item, prepared in zip(self.trainable_models_with_lr, prepared_model_list, strict=True)
        ]

    def get_runtime_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Return the accelerator-wrapped model when available."""
        return self.prepared_model_map.get(id(model), model)

    @staticmethod
    def text_encoder_grad_context(text_encoder: torch.nn.Module) -> AbstractContextManager[Any]:
        """Build the encoding context: track gradients only if the encoder is being trained.

        Frozen text encoders are the common case, and running them under
        ``no_grad`` keeps their activations out of the autograd graph.
        """
        if any(param.requires_grad for param in text_encoder.parameters()):
            return nullcontext()
        return torch.no_grad()

    def prepare_data_loader(self) -> torch.utils.data.DataLoader:
        dataset = self.prepare_dataset(self.config)
        if self.accelerator.is_main_process:
            dataset.print_bucket_info()
        num_workers = self.config.dataloader_num_workers
        with self.accelerator.main_process_first(): # type: ignore
            if isinstance(dataset, StreamingDiffusionDataset):
                if self.accelerator.num_processes > 1:
                    msg = "Streaming datasets (hf://) are not supported with multi-GPU training yet; import the dataset locally instead."
                    raise NotImplementedError(msg)
                # The dataset yields ready-made bucket-consistent batches, so automatic
                # batching is disabled. Workers must be re-created each epoch (no
                # persistent_workers) so set_epoch() reaches them via re-pickling.
                # spawn (not fork): the parent is thread-heavy (CUDA, wandb, rich)
                # by the time workers start, and forked workers have been observed
                # deadlocking on inherited locks before yielding their first batch.
                data_loader = DataLoader(
                    dataset,
                    batch_size=None,
                    num_workers=num_workers,
                    collate_fn=DiffusionDataset.collate_fn,
                    pin_memory=True,
                    prefetch_factor=2 if num_workers > 0 else None,
                    multiprocessing_context="spawn" if num_workers > 0 else None,
                )
            else:
                sampler = BucketBasedBatchSampler(dataset, self.config.batch_size)
                data_loader = DataLoader(
                    dataset,
                    batch_sampler=sampler,
                    num_workers=num_workers,
                    collate_fn=dataset.collate_fn,
                    pin_memory=True,
                    persistent_workers=num_workers > 0,
                    prefetch_factor=2 if num_workers > 0 else None,
                )
        return self.accelerator.prepare(data_loader)

    def train(self) -> None:
        # Prepare data loader
        data_loader = self.prepare_data_loader()

        num_update_steps_per_epoch, n_total_steps = self.prepare_training(data_loader)
        self.execute_training_epoch(
            data_loader,
            self.lr_scheduler,
            self.training_models,
            num_update_steps_per_epoch,
            n_total_steps,
        )
        # On exceptions we fall through WITHOUT finishing: pandm's excepthook /
        # heartbeat then marks the run crashed instead of falsely "finished".
        if self.pandm_run is not None:
            self.pandm_run.finish()

    def _configure_models(self) -> None:
        """Configure which models should be trainable with their learning rates."""
        if self.config.mode == "full-finetune":
            self._configure_full_finetune()
        elif self.config.mode in ("lora", "lokr", "loha", "locon"):
            self._configure_lora_finetune()
        else:
            msg = f"Unknown training mode: {self.config.mode}"
            raise ValueError(msg)

    @abstractmethod
    def _configure_full_finetune(self) -> None:
        """Configure models for full fine-tuning. Must be implemented by subclasses."""
        msg = "This method must be implemented by subclasses."
        raise NotImplementedError(msg)

    def _configure_lora_finetune(self) -> None:
        """Configure models for LoRA fine-tuning. Uses template method pattern."""
        # Type check - this should never happen if called correctly, but satisfies type checker
        if self.config.mode == "full-finetune":
            msg = "Cannot configure LoRA for full-finetune mode"
            raise ValueError(msg)

        # Deliberately the same `denoiser` that EMA, gradient checkpointing and
        # previews use: a second "which model is the denoiser" hook would let
        # LoRA wrap one module while EMA shadows another, silently.
        lycoris_model = apply_lora_config(self.config.mode, self.denoiser, self.config, targets=self.lora_targets)

        # Ensure LoRA model dtype matches the weight dtype to prevent dtype mismatch errors
        lycoris_model.to(dtype=self.weight_dtype)
        logger.info("LoRA model dtype set to: %s", self.weight_dtype)

        self.trainable_models_with_lr.append(TrainableModel(model=lycoris_model, lr=self.config.unet_lr))
        self.lycoris_model = lycoris_model

        # Allow subclasses to perform additional setup
        self._post_lora_setup(lycoris_model)

    @property
    def lora_targets(self) -> LoraTargets:
        """Which module classes LyCORIS should wrap. Defaults to the UNet layout."""
        return UNET_LORA_TARGETS

    def _post_lora_setup(self, lycoris_model: "LycorisNetwork") -> None:
        """Register the LyCORIS network alongside the base models.

        Every family needs this — ``self.models`` drives gradient checkpointing
        and the preview eval-mode dance — so it is the default rather than an
        override each tuner has to remember.
        """
        self.models.append(lycoris_model)

    def optimizer_step(self, loss: torch.Tensor) -> None:
        """
        Performs backward pass, gradient clipping, optimizer step,
        learning rate scheduling and optimizer zero_grad.
        """
        # NaN is detected at the optimizer-step boundary below via the loss value
        # that is already synced for logging — a per-micro-step torch.isnan(loss)
        # check would force its own GPU->CPU sync every micro-step. A NaN micro-loss
        # propagates into _loss_accum (NaN-poisoning the mean), so the boundary
        # check still catches it; training aborts before the corrupted step matters.
        #
        # Accumulate the local loss on-device. The cross-process gather and the
        # GPU->CPU sync are deferred to the optimizer-step boundary below, so they
        # run once per optimizer step rather than once per micro-step.
        if self._loss_accum is None:
            self._loss_accum = torch.zeros((), device=loss.device, dtype=torch.float32)
        self._loss_accum += loss.detach().float()
        self._micro_step_count += 1

        # Backpropagate
        self.accelerator.backward(loss)

        # Free memory
        del loss

        # Apply gradient clipping if configured
        if self.accelerator.sync_gradients and self.config.max_grad_norm > 0:
            params_to_clip = []
            for group in self.optimizer.param_groups:
                group_params = group.get("params", [])
                if isinstance(group_params, torch.Tensor):
                    params_to_clip.append(group_params)
                else:
                    params_to_clip.extend(group_params)
            if params_to_clip:
                self.accelerator.clip_grad_norm_(params_to_clip, self.config.max_grad_norm)

        # Optimizer step
        self.optimizer.step()

        # Update EMA model(s) if enabled
        step_for_ema = self.global_step + (1 if self.accelerator.sync_gradients else 0)
        self._step_emas(step_for_ema)

        # LR scheduler step
        self.lr_scheduler.step()

        # Zero gradients
        self.optimizer.zero_grad(set_to_none=self.config.zero_grad_set_to_none)

        # Once per optimizer step: gather the window-mean loss across processes
        # for logging. This is the only GPU->CPU sync on the loss path.
        if self.accelerator.sync_gradients and self._loss_accum is not None:
            mean_local = self._loss_accum / max(self._micro_step_count, 1)
            avg_loss = self.accelerator.gather(mean_local.repeat(self.config.batch_size)).mean()  # type: ignore
            self.train_loss = avg_loss.item()
            self._loss_accum.zero_()
            self._micro_step_count = 0
            if math.isnan(self.train_loss):
                logger.info("Loss is NaN.")
                msg = "Loss is NaN."
                raise ValueError(msg)

    def _should_step_ema(self, step: int) -> bool:
        return self.config.use_ema and step >= self.config.ema_start_step

    def _step_emas(self, step: int) -> None:
        if not self.accelerator.sync_gradients:
            return
        if not self._should_step_ema(step):
            return
        if self.ema_denoiser is not None:
            self.ema_denoiser.step(self.denoiser.parameters())
        if self.config.use_dual_ema and self.ema_denoiser_short is not None:
            self.ema_denoiser_short.step(self.denoiser.parameters())

    def generate_initial_preview(self) -> None:
        """Generate preview images before training starts if configured to do so."""
        logger.info("Generating preview before training starts")
        self.generate_preview(f"{self.config.model_name}-before-training", 0)

    @abstractmethod
    def get_pipeline(self) -> DiffusionPipeline:
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    @abstractmethod
    def process_batch(self, batch: DiffusionBatch) -> Any:  # noqa: ANN401
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    @abstractmethod
    def train_each_batch(self, batch: Any) -> None:  # noqa: ANN401
        msg = "This method should be implemented in subclasses."
        raise NotImplementedError(msg)

    def train_on_latents(
        self,
        img_latents: torch.Tensor,
        model_pred_fn: ModelPredFn,
        *,
        extra_tensors: Sequence[torch.Tensor | None] = (),
    ) -> None:
        """Run a single training step given latents and a model prediction function."""
        # Apply VAE scaling
        img_latents = self._apply_vae_scaling(img_latents)

        # Prepare training tensors. What a "timestep" means here is the
        # objective's business (schedule index vs continuous sigma), so it is
        # only ever passed back to the objective — never indexed into locally.
        noise = self.sample_noise(img_latents)
        timesteps = self.objective.sample_timesteps(img_latents.shape[0], global_step=self.global_step)
        img_noisy_latents = self.objective.noisy_latents(img_latents, noise, timesteps, global_step=self.global_step)

        # Predict and calculate loss. model_timesteps translates into whatever
        # time parameterization the network itself expects.
        model_pred = model_pred_fn(img_noisy_latents, self.objective.model_timesteps(timesteps))
        target, model_pred = self.objective.target_and_pred(img_latents, noise, timesteps, model_pred)
        loss = self.get_loss(timesteps, model_pred, target)

        # Free memory efficiently
        self._free_tensors(
            noise,
            img_noisy_latents,
            model_pred,
            target,
            img_latents,
            *extra_tensors,
        )

        # Optimizer step
        self.optimizer_step(loss)

    def get_loss(self, timesteps: torch.Tensor, model_pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-sample MSE, reweighted by the objective (SNR terms, flow-matching schemes, ...)."""
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
        loss_per_sample = loss.mean(dim=list(range(1, len(loss.shape))))
        weights = self.objective.loss_weights(timesteps).to(loss_per_sample.device)
        return (loss_per_sample * weights).mean()

    def sample_noise(self, latents: torch.Tensor) -> torch.Tensor:
        """Sample noise that will be added to the latents."""
        # Use Brownian noise if requested
        if self.config.use_brownian_noise:
            noise = brownian_noise(
                latents.shape,
                device=latents.device,
                dtype=latents.dtype,
                scale=self.config.brownian_noise_scale,
            )
        # Use multi-resolution noise if enabled
        elif self.config.use_multires_noise:
            # Check if custom scales are provided
            custom_scales = self.config.multires_noise_scales
            custom_weights = self.config.multires_noise_weights

            if custom_scales is not None:
                # Use custom scales/weights method
                noise = multi_resolution_noise(
                    latents.shape,
                    scales=custom_scales,
                    weights=custom_weights,
                    device=latents.device,
                    dtype=latents.dtype,
                )
            else:
                # Use pyramid method with iterations and discount
                noise = pyramid_noise(
                    latents.shape,
                    discount_factor=self.config.multires_noise_discount,
                    num_levels=self.config.multires_noise_iterations,
                    device=latents.device,
                    dtype=latents.dtype,
                )
        else:
            # Standard random noise
            noise = torch.randn_like(latents)

        # Apply traditional noise offset if configured with probability
        if self.config.noise_offset > 0:
            # Apply noise offset with configured probability
            noise_offset_prob = self.config.noise_offset_probability
            if noise_offset_prob >= 1.0 or random.random() < noise_offset_prob:
                # Add noise to the image latents
                # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                noise += self.config.noise_offset * torch.randn(
                    (latents.shape[0], latents.shape[1], 1, 1),
                    device=latents.device,
                    dtype=noise.dtype,
                )
        return noise

    def apply_seed_settings(self, seed: int) -> None:
        logger.info("Setting seed to %s", seed)
        random.seed(seed)
        torch.manual_seed(seed)

    def _move_tensors_to_device_and_dtype(self, **tensors: torch.Tensor) -> dict[str, torch.Tensor]:
        """Move tensors to the correct device and dtype efficiently."""
        result = {}
        for name, tensor in tensors.items():
            if tensor is not None:
                result[name] = tensor.to(self.device, dtype=self.weight_dtype)
        return result

    def _free_tensors(self, *tensors: torch.Tensor | None) -> None:
        """Drop local references to step tensors so Python can reclaim them.

        Deliberately does NOT call ``torch.cuda.empty_cache()``: doing so on every
        micro-step forces a CUDA sync and returns cached blocks to the driver,
        which severely hurts throughput. Periodic cache cleanup is handled in
        ``execute_training_epoch`` instead.
        """
        for tensor in tensors:
            if tensor is not None:
                del tensor

    @contextmanager
    def use_ema_weights(self) -> Generator[None, None, None]:
        """Temporarily swap UNet weights to EMA for eval/save, then restore."""
        # Before ema_start_step the shadow still holds the construction-time
        # (initial) weights — swapping it in would export a barely-trained model.
        # _should_step_ema gates on the same boundary the EMA updates begin at.
        if self.ema_denoiser is None or not self._should_step_ema(self.global_step):
            yield
            return
        self.ema_denoiser.store(self.denoiser.parameters())
        self.ema_denoiser.copy_to(self.denoiser.parameters())
        try:
            yield
        finally:
            self.ema_denoiser.restore(self.denoiser.parameters())

    def _sample_condition_dropout_mask(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample a boolean mask for conditional dropout (CFG-style).
        True values indicate samples where text condition should be dropped.
        """
        prob = self.config.condition_dropout_prob
        if prob <= 0:
            return torch.zeros(batch_size, device=device, dtype=torch.bool)
        return torch.rand(batch_size, device=device) < prob

    def apply_condition_dropout_to_prompts(self, prompts_str: list[str]) -> list[str]:
        """Apply conditional dropout by replacing selected prompts with empty strings."""
        mask = self._sample_condition_dropout_mask(len(prompts_str), self.accelerator.device)
        if not mask.any():
            return prompts_str
        dropped = mask.cpu().tolist()
        updated_prompts = list(prompts_str)
        for idx, should_drop in enumerate(dropped):
            if should_drop:
                updated_prompts[idx] = ""
        return updated_prompts

    def _apply_vae_scaling(self, latents: torch.Tensor) -> torch.Tensor:
        """Normalize raw VAE encoder output into the denoiser's latent space.

        The SD VAEs only scale. The 16-channel VAEs used by the flow-matching
        families (Lumina 2, FLUX) also carry a ``shift_factor`` that must be
        subtracted first — skipping it leaves the latents off-center by ~0.12
        and the model trains against a distribution no sampler will reproduce.
        Inverse of the decode path in ``_decode_preview_latents``.
        """
        vae_config = self.pipeline.vae.config
        scaling_factor = vae_config.get("scaling_factor", 1.0)
        shift_factor = vae_config.get("shift_factor") or 0.0
        return (latents - shift_factor) * scaling_factor

    def init_gradient_checkpointing(self, models: list[torch.nn.Module]) -> None:
        for model in models:
            m: Any = unwrap_model(self.accelerator, model)
            if hasattr(m, "enable_gradient_checkpointing"):
                m.enable_gradient_checkpointing()
            elif hasattr(m, "gradient_checkpointing_enable"):
                m.gradient_checkpointing_enable()
            else:
                logger.warning("cannot checkpointing!")

    def prepare_dataset(self, config: BaseConfig) -> DiffusionDataset | StreamingDiffusionDataset:
        # Declarative subsetting: one exported dataset serves many runs (e.g.
        # ["best quality"] trains the top-tier subset from the full manifest).
        tag_filters = TagFilters(include_any=tuple(config.dataset_include_any_tags), exclude=tuple(config.dataset_exclude_tags))
        if config.dataset_path and config.dataset_path.startswith("hf://"):
            # Stream an exported dataset straight from a HuggingFace dataset repo
            # ("hf://user/repo" or "hf://user/repo@revision"). Shards download
            # lazily on first epoch and hit the local HF cache afterwards.
            repo_id, _, revision = config.dataset_path.removeprefix("hf://").partition("@")
            logger.info("Streaming dataset from HuggingFace Hub repo %s", repo_id)
            return StreamingDiffusionDataset.from_hub(
                repo_id,
                config.batch_size,
                revision=revision or None,
                seed=config.seed,
            ).apply_tag_filters(tag_filters)
        if config.dataset_path:
            dataset_root = Path(config.dataset_path)
        elif config.image_path:
            dataset_root = Path(config.image_path) / "metadata"
        else:
            msg = "Please specify the meta path in the config file."
            raise ValueError(msg)
        parquet_path = dataset_root / "metadata.parquet"
        latents_dir = dataset_root / "latents"
        tags_dir = dataset_root / "tags"
        with self.accelerator.main_process_first():
            if not self.accelerator.is_main_process:
                # Wait for the main process to finish preparing, then load the dataset.
                return DiffusionDataset.from_parquet(parquet_path, tag_filters=tag_filters)
            if config.image_path and config.skip_prepare_image is False:
                logger.info("Prepare image from %s", config.image_path)
                if not config.vae_path:
                    msg = "Please specify the vae_path in the config file."
                    raise ValueError(msg)

                latents_processor = LatentsGenerateProcessor(
                    vae_path=config.vae_path,
                    img_path=config.image_path,
                    target_path=str(latents_dir),
                    vae_dtype=self.vae_dtype,
                )

                latents_processor()

                tagging_processor = TaggingProcessor(img_path=config.image_path, target_path=str(tags_dir), num_workers=1)
                tagging_processor()

            if not parquet_path.exists():
                logger.info('Creating parquet file at "%s"', parquet_path)
                CreateParquetProcessor(target_dir=dataset_root)(max_workers=8)
            else:
                logger.info('found parquet file at "%s"', parquet_path)
        return DiffusionDataset.from_parquet(parquet_path, tag_filters=tag_filters)

    def freeze_model(self, models: Sequence[torch.nn.Module]) -> None:
        """
        Freeze all models except the ones in training_models.
        """
        for model in models:
            model.requires_grad_(False)
            model.eval()
        self.accelerator.wait_for_everyone()

    def _scheduler_config_dict(self) -> dict:
        """Pipeline scheduler config as a plain dict, without private keys.

        ``from_config`` silently re-applies class defaults for every key listed
        in ``_use_default_values`` — which ``register_to_config`` does NOT clear.
        A base model whose scheduler JSON omits e.g. ``rescale_betas_zero_snr``
        would therefore drop our registered updates. Stripping private keys
        makes every explicit config value authoritative.
        """
        return {k: v for k, v in self.pipeline.scheduler.config.items() if not k.startswith("_")}

    def get_noise_scheduler(self) -> SchedulerMixin:
        """Set up the training-side noise scheduler.

        Defaults to the DDPM lineage: rebuild a ``DDPMScheduler`` from the
        pipeline's config, applying the prediction-type and ZTSNR overrides.
        Flow-matching families override this — none of those knobs apply to a
        rectified-flow sampler.
        """
        scheduler_config_updates = {}
        prediction_type = self.config.prediction_type
        if prediction_type is not None:
            scheduler_config_updates["prediction_type"] = prediction_type
            scheduler_config_updates["timestep_spacing"] = "trailing"
        if self.config.rescale_betas_zero_snr:
            effective_prediction = prediction_type or self.pipeline.scheduler.config.get("prediction_type", "epsilon")
            if effective_prediction != "v_prediction":
                logger.warning(
                    "rescale_betas_zero_snr with %s prediction is unsound: x0 cannot be recovered at SNR=0, "
                    "and previews may intermittently render black (NaN at the terminal step). Use v_prediction or disable it.",
                    effective_prediction,
                )
            if self.config.use_debiased_estimation:
                logger.warning(
                    "use_debiased_estimation with rescale_betas_zero_snr is unstable: the 1/sqrt(SNR) weight "
                    "hits ~1e4 at the ZTSNR terminal band (SNR clamped to 1e-8), so terminal samples dominate "
                    "the gradient and loss can explode. Disable one of them.",
                )
            scheduler_config_updates["rescale_betas_zero_snr"] = True
            scheduler_config_updates.setdefault("timestep_spacing", "trailing")
        if scheduler_config_updates:
            self.pipeline.scheduler.register_to_config(**scheduler_config_updates)
            # register_to_config only mutates the config dict; betas/alphas_cumprod
            # are computed in __init__. Rebuild the live instance so previews sample
            # with the same schedule the training targets use (e.g. the
            # ZTSNR-rescaled alphas — a stale instance keeps ~6.8% terminal signal
            # and renders every v-pred preview washed out).
            self.pipeline.scheduler = type(self.pipeline.scheduler).from_config(
                self._scheduler_config_dict(),
            )

        # Create the noise scheduler from the pipeline's scheduler configuration
        noise_scheduler = DDPMScheduler.from_config(
            self._scheduler_config_dict(),
        )
        if type(noise_scheduler) is not DDPMScheduler:
            msg = "Scheduler is not DDPMScheduler"
            raise TypeError(msg)
        logger.info("Noise scheduler config:")
        for key, value in noise_scheduler.config.items():
            logger.info("- %s: %s", key, value)
        return noise_scheduler  # type: ignore

    def execute_training_epoch(  # noqa: C901, PLR0912, PLR0915
        self,
        data_loader: torch.utils.data.DataLoader,
        lr_scheduler: torch.optim.lr_scheduler.LambdaLR,
        training_models: Sequence[torch.nn.Module],
        num_update_steps_per_epoch: int,
        n_total_steps: int,
    ) -> None:
        progress = get_progress()

        self.checkpointing_path = self.save_path / "state"
        self.global_steps_file = self.checkpointing_path / "global_steps"

        # Attempt to load previous state for resuming training
        global_step = 0
        try:
            logger.info("Attempting to load checkpoint from: %s", self.checkpointing_path.as_posix())
            if not self.checkpointing_path.exists():
                logger.warning("Checkpoint directory does not exist: %s", self.checkpointing_path)
            elif not (self.checkpointing_path / "optimizer.bin").exists():
                logger.warning("Optimizer state file does not exist in checkpoint directory")
            elif not self.global_steps_file.exists():
                logger.warning("Global steps file does not exist: %s", self.global_steps_file)
            else:
                self.accelerator.load_state(self.checkpointing_path.as_posix())
                global_step = int(self.global_steps_file.read_text())
                self.global_step = global_step  # Update instance variable for input perturbation decay
                logger.info("Successfully loaded checkpoint at global step: %d", global_step)
        except Exception:
            logger.exception("Failed to load checkpoint:")
            # Instead of immediately deleting, try to backup and provide recovery options
            import shutil
            from datetime import datetime

            if self.accelerator.is_main_process and self.checkpointing_path.exists():
                # Create a backup of the corrupted checkpoint
                timestamp = datetime.now(tz=UTC).strftime("%Y%m%d_%H%M%S")
                backup_path = self.checkpointing_path.parent / f"{self.checkpointing_path.name}_corrupted_{timestamp}"
                try:
                    shutil.copytree(self.checkpointing_path, backup_path)
                    logger.info("Corrupted checkpoint backed up to: %s", backup_path)
                except Exception as backup_e:
                    logger.warning("Failed to backup corrupted checkpoint: %s", backup_e)

                try:
                    shutil.rmtree(self.checkpointing_path)
                    logger.warning("Removed corrupted checkpoint directory: %s", self.checkpointing_path)
                except Exception:
                    logger.exception("Failed to remove corrupted checkpoint:")

            self.accelerator.wait_for_everyone()

            global_step = 0
            logger.info("Starting training from scratch due to checkpoint corruption")

        # 1. Calculate the number of completed full epochs
        skipped_epoch = global_step // num_update_steps_per_epoch if num_update_steps_per_epoch > 0 else 0

        # 2. Calculate the number of dataloader batches to skip in the current epoch
        num_batches_to_skip_in_current_epoch = 0
        if global_step > 0 and num_update_steps_per_epoch > 0:
            # Calculate the number of optimizer steps completed in the current epoch
            steps_in_current_epoch = global_step % num_update_steps_per_epoch
            # Calculate the number of dataloader batches corresponding to these optimizer steps
            num_batches_to_skip_in_current_epoch = steps_in_current_epoch * self.config.gradient_accumulation_steps

        if global_step != 0:
            # Adjust log information
            logger.info(
                "Resuming from global step %d (Epoch %d, skipping %d dataloader batches in current epoch)",
                global_step,
                skipped_epoch,
                num_batches_to_skip_in_current_epoch,
            )
            # (Optional) Calculate and record the total number of processed samples
            total_samples_processed = global_step * self.config.gradient_accumulation_steps * self.accelerator.num_processes * self.config.batch_size
            logger.info("Approximately %d total samples processed before resuming.", total_samples_processed)

        self._set_data_loader_epoch(data_loader, skipped_epoch)
        skipped_data_loader = self.accelerator.skip_first_batches(
            data_loader,
            num_batches_to_skip_in_current_epoch,
        )

        logger.info("full_loader_length: %d", len(data_loader))

        if global_step == 0 and self.config.preview_before_training:
            self.generate_initial_preview()

        total_task = progress.add_task(
            "Total Progress",
            total=n_total_steps,
            completed=global_step,
        )

        if self.accelerator.is_main_process:
            progress.start()
        for epoch in range(skipped_epoch, self.config.n_epochs):
            self._set_data_loader_epoch(data_loader, epoch)
            if epoch == skipped_epoch:
                self._set_data_loader_epoch(skipped_data_loader, epoch)
            self.train_loss = 0.0
            current_epoch_task = progress.add_task(
                f"Epoch {epoch + 1}",
                total=num_update_steps_per_epoch,
                completed=global_step % num_update_steps_per_epoch,
            )
            dl = skipped_data_loader if epoch == skipped_epoch else data_loader
            for _step, orig_batch in enumerate(dl):
                if not isinstance(orig_batch, DiffusionBatch):
                    msg = f"Expected DiffusionBatch, got something else. Got: {type(orig_batch)}"
                    raise TypeError(msg)
                batch = self.process_batch(orig_batch)

                # Free up memory from original batch if it contains large tensors
                del orig_batch

                with self.accelerator.accumulate(*training_models):
                    self.train_each_batch(batch)

                    # Free up processed batch memory
                    del batch

                    # Only clear CUDA cache occasionally to avoid performance issues
                    if hasattr(torch.cuda, "empty_cache") and self.accelerator.sync_gradients and global_step % 50 == 0:
                        torch.cuda.empty_cache()

                if self.accelerator.sync_gradients:
                    current_lr = lr_scheduler.get_last_lr()[0]

                    global_step += 1
                    self.global_step = global_step  # Update instance variable for input perturbation decay
                    log_data = {"train_loss": self.train_loss, "lr": current_lr}
                    if self.pandm_run is not None:
                        self.pandm_run.log(log_data, step=global_step)
                    else:
                        self.accelerator.log(log_data, step=global_step)

                    self.train_loss = 0.0

                    if self.accelerator.is_main_process:
                        current_completed = global_step % num_update_steps_per_epoch
                        progress.update(total_task, completed=global_step, description=f"Epoch: {epoch}")
                        progress.update(current_epoch_task, completed=current_completed, description=f"Lr: {current_lr:.2e}")

                    # Check step-based conditions
                    should_save_step = self.config.save_every_n_steps and global_step % self.config.save_every_n_steps == 0 and global_step != 0
                    should_checkpoint_step = self.config.checkpoint_every_n_steps and global_step % self.config.checkpoint_every_n_steps == 0
                    should_preview_step = self.config.preview_every_n_steps and global_step % self.config.preview_every_n_steps == 0

                    if should_save_step:
                        self.saving_model(f"{self.config.model_name}-step{global_step}")
                    if should_checkpoint_step:
                        # 确保优化器状态能被正确保存
                        if not hasattr(self, "optimizer"):
                            logger.warning("Optimizer not found in self. Using optimizer from SDXL class.")
                            # 在没有 self.optimizer 的情况下，我们假设优化器已经被传递给了 accelerator

                        # 创建 checkpoint 目录
                        self.checkpointing_path.mkdir(parents=True, exist_ok=True)
                        # 保存状态
                        self.accelerator.save_state(self.checkpointing_path.as_posix())
                        # 保存当前步数
                        self.global_steps_file.write_text(str(global_step))
                        # 验证保存结果
                        if not (self.checkpointing_path / "optimizer.bin").exists():
                            logger.warning("Failed to save optimizer state. Please make sure the optimizer is correctly prepared with accelerator.")
                        else:
                            logger.info("Successfully saved checkpoint at global step: %d", global_step)
                    if should_preview_step:
                        self.generate_preview(f"{self.config.model_name}-step{global_step}", global_step)
            epoch_number = epoch + 1
            should_save_epoch = self.config.save_every_n_epochs and epoch_number % self.config.save_every_n_epochs == 0
            should_preview_epoch = self.config.preview_every_n_epochs and epoch_number % self.config.preview_every_n_epochs == 0

            if should_save_epoch:
                self.saving_model(f"{self.config.model_name}-ep{epoch_number}")
            if should_preview_epoch:
                self.generate_preview(f"{self.config.model_name}-ep{epoch_number}", global_step)
            progress.remove_task(current_epoch_task)
        if self.accelerator.is_main_process:
            progress.stop()
        self.saving_model(f"{self.config.model_name}")

    @abstractmethod
    def get_preview_prompt_embeds(self, prompt: str, neg_prompt: str, clip_skip: int = 2) -> dict[str, torch.Tensor]:
        """Return weighted prompt embeddings as pipeline kwargs.

        SD1.5 returns ``prompt_embeds``/``negative_prompt_embeds``; SDXL also
        returns the pooled pair. The dict is splatted directly into the pipeline.
        """

    def preview_pipeline_kwargs(self, sample_option: SampleOptions) -> dict[str, Any]:
        """Sampling arguments for the preview pipeline call, minus conditioning.

        Split out because pipelines disagree on what they accept: the SD
        pipelines take ``guidance_rescale``, ``Lumina2Pipeline`` does not (it
        has ``cfg_normalization``/``cfg_trunc_ratio`` instead), and passing an
        unknown keyword is a TypeError mid-run.
        """
        return {
            "num_inference_steps": sample_option.steps,
            "width": sample_option.width,
            "height": sample_option.height,
            "guidance_scale": sample_option.guidance_scale,
            "guidance_rescale": sample_option.guidance_rescale,
        }

    @property
    def supports_hires_preview(self) -> bool:
        """Whether the hires-fix img2img pass is available for this architecture."""
        return True

    @torch.no_grad()
    def _decode_preview_latents(self, latents: torch.Tensor) -> "Image.Image":
        """Decode pipeline latents to PIL in fp32, outside any autocast.

        Half-precision VAE decode intermittently overflows to NaN (pure-black
        frames) at 640+ resolutions, so previews decode through a temporary
        fp32 cast and restore the configured VAE dtype afterwards.
        """
        vae = self.pipeline.vae
        original_dtype = next(vae.parameters()).dtype
        vae.to(dtype=torch.float32)
        try:
            # Exact inverse of _apply_vae_scaling: unscale, then undo the shift
            # (0 for the SD VAEs, ~0.1159 for the 16-channel FLUX/Lumina one).
            shift_factor = vae.config.get("shift_factor") or 0.0
            latents = latents.to(torch.float32) / vae.config.scaling_factor + shift_factor
            image = vae.decode(latents).sample
            return self.pipeline.image_processor.postprocess(image, output_type="pil")[0]  # type: ignore[attr-defined]
        finally:
            vae.to(dtype=original_dtype)

    @torch.no_grad()
    def _hires_preview(
        self,
        image: "Image.Image",
        prompt_kwargs: dict[str, Any],
        sample_option: SampleOptions,
        generator: torch.Generator,
        make_autocast: Callable[[], AbstractContextManager[Any]],
    ) -> "Image.Image":
        """A1111-style hires fix for previews: lanczos-upscale, then img2img."""
        from diffusers import AutoPipelineForImage2Image

        width = int(image.width * sample_option.hires_scale) // 8 * 8
        height = int(image.height * sample_option.hires_scale) // 8 * 8
        upscaled = image.resize((width, height), Image.LANCZOS)
        img2img = AutoPipelineForImage2Image.from_pipe(self.pipeline)
        img2img.progress_bar = DummyProgressBar  # type: ignore[method-assign]
        # Same seam as the base pass, so a family only has to declare its
        # accepted kwargs once. img2img derives its size from the input image
        # and its step count from hires_steps, so those two are dropped.
        pipeline_kwargs = self.preview_pipeline_kwargs(sample_option)
        for key in ("width", "height", "num_inference_steps"):
            pipeline_kwargs.pop(key, None)
        with make_autocast():
            result = img2img(
                **prompt_kwargs,
                **pipeline_kwargs,
                image=upscaled,
                strength=sample_option.hires_strength,
                num_inference_steps=sample_option.hires_steps,
                generator=generator,
                output_type="latent",
            )
        return self._decode_preview_latents(result.images)

    @torch.no_grad()
    def generate_preview(self, filename: str, global_step: int = 0) -> None:  # noqa: C901, PLR0912, PLR0915
        # Release memory to ensure sufficient VRAM for preview generation
        free_memory()

        # Simple callback function to meet the pipeline interface requirements
        def callback_on_step_end(_pipe: StableDiffusionXLPipelineOutput, _step: int, _timestep: int, _kwargs: dict) -> dict:
            return _kwargs

        state = PartialState()
        self.accelerator.wait_for_everyone()

        # Split preview sample options across processes
        with state.split_between_processes(self.config.preview_sample_options) as sample_options:
            for sample_option in sample_options:
                if not isinstance(sample_option, SampleOptions):
                    msg = f"Expected SampleOption, got {type(sample_option)}"
                    raise TypeError(msg)
                hash_hex = get_sample_options_hash(sample_option)
                filename_with_hash = f"{filename}-{hash_hex}"

                # Check if preview file already exists, skip if it does
                path = self.save_path / "previews" / f"{filename_with_hash}.png"
                if path.exists():
                    logger.info("Preview file already exists, skipping: %s", path)
                    continue

                logger.info("Generating preview for %s", filename_with_hash)

                # Save original training-related settings and device information for later restoration
                original_training_mode = {}
                original_device = {}
                models_for_preview = [
                    ("denoiser", self.denoiser),
                    ("text_encoder", self.pipeline.text_encoder),
                    ("vae", self.pipeline.vae),
                ]
                text_encoder_2 = getattr(self.pipeline, "text_encoder_2", None)
                if text_encoder_2 is not None:
                    models_for_preview.append(("text_encoder_2", text_encoder_2))
                for name, model in models_for_preview:
                    original_training_mode[name] = model.training
                    original_device[name] = next(model.parameters()).device
                    model.eval()  # Switch to evaluation mode for inference

                # Use automatic mixed precision to generate preview images
                generator = torch.Generator(device=self.accelerator.device).manual_seed(sample_option.seed)

                def make_autocast() -> AbstractContextManager[Any]:
                    return nullcontext() if torch.backends.mps.is_available() else torch.autocast(self.accelerator.device.type)

                with self.use_ema_weights():
                    self.pipeline.to(self.accelerator.device)
                    self.pipeline.vae.to(dtype=self.vae_dtype)
                    # Resolve conditioning once; the same kwargs feed the base
                    # pass and the optional hires img2img pass.
                    try:
                        with make_autocast():
                            prompt_kwargs: dict[str, Any] = dict(
                                self.get_preview_prompt_embeds(
                                    sample_option.prompt,
                                    sample_option.negative_prompt,
                                    getattr(sample_option, "clip_skip", 2),
                                ),
                            )
                    except NotImplementedError:
                        logger.info("Using prompts directly for preview generation")
                        prompt_kwargs = {"prompt": sample_option.prompt, "negative_prompt": sample_option.negative_prompt}
                    with make_autocast():
                        result = self.pipeline(
                            **prompt_kwargs,
                            **self.preview_pipeline_kwargs(sample_option),
                            generator=generator,
                            callback_on_step_end=callback_on_step_end,  # type: ignore
                            output_type="latent",
                        )
                    # Decode in fp32 OUTSIDE autocast: half-precision VAE decode
                    # intermittently overflows to NaN (pure-black previews) at
                    # 640+ resolutions; the UNet latents themselves are clean.
                    image = self._decode_preview_latents(result.images)
                    if sample_option.hires_scale > 1.0:
                        if not self.supports_hires_preview:
                            logger.warning("hires_scale is set but %s has no img2img pipeline; keeping the base image.", type(self).__name__)
                        else:
                            try:
                                image = self._hires_preview(image, prompt_kwargs, sample_option, generator, make_autocast)
                            except Exception:
                                logger.exception("Hires preview pass failed; keeping the base image")

                logger.info("Preview generated for %s", filename_with_hash)

                path.parent.mkdir(parents=True, exist_ok=True)

                image.save(path)

                if self.pandm_run is not None:
                    self.pandm_run.log_image(hash_hex, image, step=global_step, caption=sample_option.prompt)
                elif self.config.log_with == "wandb":
                    import wandb

                    self.accelerator.log(
                        {
                            f"{hash_hex}": [wandb.Image(image, caption=f"{sample_option.prompt}")],
                        },
                        step=global_step,
                    )

                # Release generated results promptly to reduce memory usage
                del result
                # Only clear cache after preview generation, not after each image
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Restore the training mode of the models
                for name, model in models_for_preview:
                    if model.training != original_training_mode[name]:
                        model.train(original_training_mode[name])
                    if next(model.parameters()).device != original_device[name]:
                        model.to(original_device[name])

        # Ensure all processes have completed preview generation
        free_memory()
        self.accelerator.wait_for_everyone()

    def saving_model(self, filename: str) -> None:
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            save_context = self.use_ema_weights() if self.config.use_ema else nullcontext()
            with save_context:
                if self.config.mode in ("lora", "lokr", "loha", "locon"):
                    self.save_lora_model(filename)
                else:
                    self.save_full_finetune_model(filename)

    def _create_lora_metadata(self) -> dict[str, str]:
        """Create metadata for LoRA models."""
        if self.lycoris_model is None:
            msg = "LyCORIS model is not initialized."
            raise ValueError(msg)

        # Create essential metadata for compatibility
        metadata = {
            # Core LoRA parameters - essential for loading
            "ss_network_dim": str(getattr(self.lycoris_model, "lora_dim", self.config.lora_dim)),
            "ss_network_alpha": str(getattr(self.lycoris_model, "alpha", self.config.lora_alpha)),
            "ss_network_module": "lycoris.kohya",
            "ss_network_args": f'{{"algo": "{self.config.mode}"}}',
            # Basic info
            "format": "pt",
            "ss_base_model_version": self.lora_base_model_version,
            "ss_training_comment": f"LyCORIS {self.config.mode.upper()} training",
            "ss_resolution": self.lora_metadata_resolution,
        }

        # Add algorithm-specific metadata
        if self.config.mode == "lokr":
            metadata["ss_lokr_factor"] = str(self.config.lokr_factor)

        return metadata

    def _save_lycoris_format_lora(self, filename: str, metadata: dict[str, str]) -> None:
        """Save LyCORIS format LoRA."""
        out_path = self.save_path / f"{filename}.safetensors"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if self.lycoris_model is None:
            msg = "LyCORIS model is not initialized."
            raise ValueError(msg)

        # Save LyCORIS weights with metadata
        self.lycoris_model.save_weights(
            self.save_path / f"{filename}.safetensors",
            dtype=self.save_dtype,
            metadata=metadata,
        )

    def save_lora_model(self, filename: str) -> None:
        """Save LoRA model in LyCORIS native format (loadable by kohya/WebUI/LyCORIS)."""
        metadata = self._create_lora_metadata()
        self._save_lycoris_format_lora(filename, metadata)

    def save_full_finetune_model(self, filename: str) -> None:
        self.save_path.mkdir(parents=True, exist_ok=True)
        # Fast path: when saving in the same dtype as training, save the live
        # weights directly. Casting here would be a no-op that still rewrites
        # every parameter tensor.
        if self.save_dtype == self.weight_dtype:
            self.pipeline.save_pretrained(self.save_path / f"{filename}")
            return
        # Different save dtype: diffusers' save_pretrained has no dtype argument,
        # so we round-trip the live pipeline. This is an in-place cast, so for an
        # intermediate checkpoint the training weights take a precision hit
        # (notably fp32 -> fp16 -> fp32). Acceptable for the documented use of
        # saving a lower-precision export; a lossless path would need a full
        # CPU state_dict copy.
        with torch.no_grad():
            self.pipeline.to(self.save_dtype)
            self.pipeline.save_pretrained(self.save_path / f"{filename}")
            self.pipeline.to(self.weight_dtype)

    @property
    def lora_base_model_version(self) -> str:
        """kohya/WebUI ``ss_base_model_version`` tag. Overridden per architecture."""
        return "sdxl_v1"

    @property
    def lora_metadata_resolution(self) -> str:
        """kohya/WebUI ``ss_resolution`` tag. Overridden per architecture."""
        return "1024,1024"

    @property
    def training_prompts_use_attention_parser(self) -> bool:
        """Whether training prompts go through the A1111 attention-syntax parser.

        Trainers whose encoding path parses ``(word:weight)`` syntax must
        override this so dataset text gets escaped (booru parens stay literal).
        """
        return False

    def create_prompts_str(self, batch: DiffusionBatch) -> list[str]:
        prompts = []
        caption_dropout_ratio = self.config.caption_dropout
        all_tags_dropout_ratio = self.config.all_tags_dropout
        rules = TagCompositionRules.from_config(self.config)

        for caption, tags, tag_categories in zip(batch.caption, batch.tags, batch.tag_categories, strict=True):
            # Decide if the caption should be dropped
            true_caption = "" if random.random() < caption_dropout_ratio else caption

            # Decide if all tags should be dropped, otherwise assemble them in
            # category order with per-category shuffle/dropout.
            true_tags = [] if random.random() < all_tags_dropout_ratio else compose_prompt_tags(tags, tag_categories, rules)

            # Create prompt string
            if true_caption and true_tags:
                prompt = f"{true_caption}, " + ", ".join(true_tags)
            elif true_caption:
                prompt = true_caption
            else:
                prompt = ", ".join(true_tags)

            # The enhanced embedder parses A1111 attention syntax; keep dataset
            # text literal (the plain tokenizer path must NOT see backslashes).
            if self.training_prompts_use_attention_parser:
                prompt = escape_attention_syntax(prompt)

            prompts.append(prompt)

        return prompts

    def _set_data_loader_epoch(self, data_loader: torch.utils.data.DataLoader, epoch: int) -> None:
        """Propagate epoch information to custom samplers/datasets used for deterministic resume."""
        batch_sampler = getattr(data_loader, "batch_sampler", None)
        if batch_sampler is not None and hasattr(batch_sampler, "set_epoch"):
            batch_sampler.set_epoch(epoch)

        # Streaming datasets reshuffle per epoch through the dataset itself.
        dataset = getattr(data_loader, "dataset", None)
        if dataset is not None and hasattr(dataset, "set_epoch"):
            dataset.set_epoch(epoch)

        inner_data_loader = getattr(data_loader, "base_dataloader", None)
        if inner_data_loader is not None and inner_data_loader is not data_loader:
            self._set_data_loader_epoch(inner_data_loader, epoch)

    def get_ema(self, model: torch.nn.Module, config: dict, *, decay: float) -> EMAModel:
        """Initialize Exponential Moving Average for the model if enabled in config."""
        ema_model = EMAModel(
            parameters=model.parameters(),
            # model_cls only matters for EMAModel.save_pretrained/from_pretrained,
            # which this trainer never calls (weights are swapped in place via
            # store/copy_to). Reporting the real class keeps it honest anyway.
            model_cls=type(model),
            model_config=config,
            decay=decay,
        )
        ema_model.to(self.device)
        return ema_model
