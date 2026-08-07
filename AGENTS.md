# Repository Guidelines

## Project Overview

A diffusion model trainer framework covering UNet families (SD 1.5, SDXL) and flow-matching DiTs (Lumina 2 / Neta Lumina), supporting full fine-tuning and parameter-efficient methods (LoRA, LoKr). Built with PyTorch, Diffusers, and Accelerate for distributed training. Uses a custom PyTorch index for CUDA support (defaults to CUDA 12.6, adaptable to the system CUDA version).

## Project Structure & Module Organization

- Core code lives in `src/diffusion_trainer`: `config/` (typed TOML-backed dataclasses), `dataset/` (processors, bucketing, tagging), `finetune/` (trainers for SD 1.5/SDXL, utilities), `shared/` (losses, schedulers), `utils/` and `lib/` (helpers and third-party glue).
- Entry scripts: `run_prepare.py` (latents, tagging, parquet metadata), `run_train.py` (training), `run_load_dataset.py` (metadata inspection). Example configs in `configs/`.
- Data and artifacts are expected under `datasets/` (cached latents and metadata), `models/` (pre-trained weights), `playground/`, and `out/` (checkpoints, previews); keep large assets out of version control.
- Docs live in `docs/` (Chinese VitePress site); tests mirror module paths under `tests/`.

## Build, Test, and Development Commands

- Uses `uv` package manager (not pip). Install: `uv install`. Run anything: `uv run <command>`.
- Lint/format: `uv run ruff check --fix .`
- Type check: `pyright` (configured in pyproject.toml).
- Tests: `uv run pytest tests/` (single test file: `uv run pytest tests/diffusion_trainer/finetune/utils/test_compute_all_snr.py`).
- Data prep: `uv run python run_prepare.py --image_path <input> --target_path <output> --vae_path <vae_model>` (`--base_resolution 512/768/1024` selects the bucket table; default 1024 for SDXL).
- Train SDXL: `uv run python run_train.py --config configs/sdxl.toml --model_family sdxl`
- Train SD1.5: `uv run python run_train.py --config configs/sd15.toml --model_family sd15`
- Train Lumina 2: `uv run python run_train.py --config configs/lumina2_lora.toml --model_family lumina2`
- Slow tests (a few hundred training steps) are marked `slow`: `uv run pytest -m 'not slow'` skips them.
- This environment lacks `python3.12-dev`, so torch 2.13's `_native` triton path cannot JIT-compile and Gemma-2's rope crashes. Until the headers are installed, prefix Lumina runs with `TORCH_DISABLE_NATIVE_JIT=1` (SD 1.5 / SDXL are unaffected — CLIP does not hit that op).
- Load dataset from external source: `uv run python scripts/load_dataset_from_pictoria.py` (live API) or `uv run python scripts/load_dataset_from_pictoria_db.py` (SQLite snapshot; filters by star score / content rating / short edge, canonical posts only by default; emits category-ordered tags — one quality tag per post from manual score with SILVA aesthetic fallback and auto-calibrated thresholds, one era tag bucketed from `published_at`, then artist/copyright/character/general/meta from the DB tag groups — plus a `tag_categories` parquet column); convert legacy sd-scripts metadata: `uv run python scripts/convert_ss_meta_to_trainer_meta.py`.
- Share datasets: `uv run python scripts/export_dataset.py --dataset-dir <prepared> --output-dir <export> --vae-name <vae>` packs latents into bucket-grouped tar shards (uploadable to a HF dataset repo as-is); `uv run python scripts/import_dataset.py --source <dir-or-hf-repo-id> --target-dir <dataset_path>` restores the local layout.

## Architecture Overview

### Training System

- `BaseTuner` (`finetune/base.py`): abstract base class for all trainers. It orchestrates the loop (data, accumulation, EMA, checkpointing, previews) but owns no diffusion math.
- `SDXLTuner` (`finetune/sdxl.py`), `SD15Tuner` (`finetune/sd15.py`) and `Lumina2Tuner` (`finetune/lumina2.py`): model-family-specific implementations.
- `DiffusionObjective` (`finetune/objective.py`): the noising process, prediction target and loss weighting, factored out so one loop drives two incompatible formulations.
  - `DDPMObjective` — discrete timesteps, epsilon/v-prediction targets, SNR weighting (Min-SNR, debiased estimation, ZTSNR). The SD 1.5 / SDXL lineage.
  - `FlowMatchObjective` — continuous sigmas, rectified-flow target, SD3-style sigma sampling. Lumina 2. Note its two easily-inverted conventions: the network's time input is `1 - sigma`, and its output space is `x0 - noise` (the pipeline negates before stepping). `tests/diffusion_trainer/finetune/test_flow_match_objective.py` pins both.
- Architecture seams a new family overrides: `denoiser` (UNet vs DiT — the single "which model is trained" hook, used by LoRA, EMA, gradient checkpointing and previews alike), `create_objective()`, `get_noise_scheduler()`, `lora_targets` (a `LoraTargets` declaring attention/feedforward/conv class names by role — LyCORIS matches by class name, and Lumina's FFN is `LuminaFeedForward`, not `FeedForward`), `preview_pipeline_kwargs()` and `supports_hires_preview`.
- Family-specific config knobs live on family configs, never `BaseConfig`: `FlowMatchSettings` (the `flow_match_*` block) is mixed into `Lumina2Config` so writing one in an SD 1.5 config is a construction-time `TypeError` rather than a silently ignored value. Dataclass inheritance keeps the TOML key space flat, so config files are unaffected.
- `BaseTuner._apply_vae_scaling` handles both the SD VAEs (scale only) and the 16-channel FLUX/Lumina one (scale **and** `shift_factor`); `_decode_preview_latents` is its exact inverse.
- Supported techniques: full fine-tuning of UNet/DiT and text encoders, LoRA, LoKr, noise offset, input perturbation, SNR gamma weighting, EMA, gradient checkpointing, mixed precision (fp16/bf16), multiple optimizers (AdamW, Adafactor, Prodigy, Lion, ...), timestep bias strategies (uniform, logit, range), LR schedulers with warmup.
- Training utilities (`finetune/utils/`): optimizer/scheduler setup, LoRA/LoKr network creation, pipeline loading, sample generation during training.

### Lumina 2 specifics

- Train with `--model_family lumina2` and a `Lumina2Config`; `configs/lumina2_lora.toml` documents which SD-lineage options go dead (`prediction_type`, `snr_gamma`, `rescale_betas_zero_snr`, `use_debiased_estimation`, `timestep_bias_strategy`, `clip_skip`) and their `flow_match_*` replacements. The tuner warns at startup if any are set.
- `model_path` must be a **diffusers-format** repo (e.g. `VirtualAddressExtension/Neta-Lumina-v1.0-diffusers`). The official all-in-one ComfyUI safetensors has no diffusers config and is rejected with a pointer.
- Conditioning is Gemma-2 hidden states behind the `<Prompt Start>` preamble, with an attention mask that must reach the DiT (`gemma_skip_layers = 2` is the CLIP-skip analogue). Previews go through the same encoder path so they cannot drift from training.
- Datasets must be re-encoded: the 16-channel VAE makes SD-lineage latents unusable. `run_prepare.py --vae_path <pipeline repo>` now falls back to the repo's `vae/` subfolder automatically.

### Data Pipeline

- Preprocessing runs as three stages (`run_prepare.py`): `LatentsGenerateProcessor` (pre-computes VAE latents) → `TaggingProcessor` (WD Tagger; also imports same-name sidecar `.txt` tags) → `CreateParquetProcessor` (aggregates metadata).
- Storage layout is SHA256-sharded via `dataset/utils/sharded_path()`: `root/ab/cd/<sha256>.<ext>` for both `latents/*.npz` and `tags/*.txt`.
- NPZ files store **latents only**. Per-image metadata (`train_resolution`, `original_size`, `crop_ltrb`) lives in `latents/latents_meta.parquet`, and `CreateParquetProcessor` joins it with tags into the final `metadata.parquet` — the single source of truth consumed by training. Legacy NPZ files with embedded metadata are still readable: the processors harvest their metadata without re-encoding, and `DiffusionDataset` falls back to embedded fields when manifest columns are absent.
- `DiffusionDataset` loads image-latent/text pairs; `BucketBasedBatchSampler` groups images by aspect-ratio bucket (resolutions from 640×1536 to 1536×640).
- Train-time subsetting: `dataset_include_any_tags` / `dataset_exclude_tags` filter manifest rows (map-style and streaming) so one exported dataset serves many runs — e.g. `["best quality"]` trains only the top quality tier.
- Category-aware prompts: `metadata.parquet` may carry a `tag_categories` column parallel to `tags` (written by the pictoria DB loader). Training assembles prompts per `tag_category_order` (default quality → artist → copyright → character → general; unlisted categories like `meta` are dropped) and applies shuffle/single-tag-dropout only inside `shuffled_tag_categories`/`droppable_tag_categories` (default `general`), so quality/style/character conditioning stays pinned at the prompt front. Datasets without the column treat every tag as `general` — identical to the old flat behavior (`compose_prompt_tags` in `finetune/base.py`, quality tiers in `dataset/quality_tags.py`).
- Long-running processors extend `ThreadedPipelineProcessor` (`dataset/processors/base.py`), a reader→processor→writer threaded pipeline base class.
- Dataset sharing (`dataset/sharing.py`): `export_dataset()` packs latent NPZ files into bucket-grouped tar shards (`shards/<W>x<H>-<idx>.tar`, deterministic tar metadata) and adds `shard`/`npz_sha256` columns to `metadata.parquet`; `import_dataset()` restores the local prepared layout (npz tree, sidecar tags, `latents_meta.parquet`) with checksum verification and idempotent re-runs.
- Streaming training (`dataset/streaming.py`): `StreamingDiffusionDataset` (IterableDataset) trains directly from exported shards — set `dataset_path = "hf://user/repo"` (optionally `@revision`) in the config. Shards download lazily in the workers (epoch 1 overlaps training with the transfer, later epochs hit the HF cache); `from_hub` pins the commit sha so cached shards resolve without network, and `prefetch_shards()` warms the cache up front when wanted. Workers use `multiprocessing_context="spawn"` (forked workers deadlock on inherited locks from the thread-heavy parent) and no `persistent_workers` (so `set_epoch` reaches workers). Bucket-pure shards yield resolution-consistent pre-assembled batches (`DataLoader` with `batch_size=None`); `len()` is exact (partial batches flush per shard). Single-GPU only for now; requires `dispatch_batches=False` (set globally in `prepare_accelerator`).

### Configuration System

- TOML-based configs in `configs/`, parsed into type-safe dataclasses in `src/diffusion_trainer/config/`. Separate configs for SD15 and SDXL.

## Coding Style & Naming Conventions

- Python 3.12+ with strict type hints on all functions (modern syntax); prefer `pathlib.Path`, f-strings, and small, composable functions.
- Ruff enforces `select = ["ALL"]` with limited ignores (line length 160). Do not add new ignores without justification; avoid `noqa` except for clear false positives.
- Conditional imports for optional dependencies are allowed only in `finetune/base.py` and `finetune/utils/__init__.py` (per-file ruff ignores).
- Modules/functions use `snake_case`; classes use `PascalCase`; configs and datasets use lowercase with hyphens/underscores (e.g., `sdxl.toml`, `metadata.parquet`).
- Rich logging for training progress; keep logging informative but concise and avoid heavyweight prints in hot paths. WandB integration for experiment tracking.

## Testing Guidelines

- Add tests in `tests/` mirroring module paths; use pytest style and `@pytest.mark` for GPU or slow cases.
- Keep fixtures lightweight and deterministic; stub I/O with temp dirs and small sample tensors.
- For new trainers or processors, cover one happy path and one failure/edge path; ensure tests pass with `uv run pytest`.

## Commit & Pull Request Guidelines

- Follow Conventional Commits with scopes seen in history (`feat(config)`, `fix(dataset)`, `docs(readme)`, `chore(...)`).
- Before sending a PR, run `uv run ruff check --fix .` and `uv run pytest`; include results in the description.
- PRs should state the intent, configs touched, expected resource needs (GPU/VRAM), and any new CLI flags. Add screenshots or sample metrics when UI or logging output changes.

## Security & Configuration Tips

- Do not commit model weights, dataset shards, API keys, or WandB tokens; load them via environment variables or `.env` entries ignored by git.
- Keep `scripts/from_diffusers/` in sync when upstream changes; verify licensing before importing new assets.
- Validate paths passed to scripts to avoid overwriting `out/` runs; prefer writing to new timestamped subfolders.
