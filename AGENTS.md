# Repository Guidelines

## Project Overview

A diffusion model trainer framework for Stable Diffusion models (SD 1.5 and SDXL), supporting full fine-tuning and parameter-efficient methods (LoRA, LoKr). Built with PyTorch, Diffusers, and Accelerate for distributed training. Uses a custom PyTorch index for CUDA support (defaults to CUDA 12.6, adaptable to the system CUDA version).

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
- Load dataset from external source: `uv run python scripts/load_dataset_from_pictoria.py` (live API) or `uv run python scripts/load_dataset_from_pictoria_db.py` (SQLite snapshot; filters by star score / short edge, tags from the DB); convert legacy sd-scripts metadata: `uv run python scripts/convert_ss_meta_to_trainer_meta.py`.
- Share datasets: `uv run python scripts/export_dataset.py --dataset-dir <prepared> --output-dir <export> --vae-name <vae>` packs latents into bucket-grouped tar shards (uploadable to a HF dataset repo as-is); `uv run python scripts/import_dataset.py --source <dir-or-hf-repo-id> --target-dir <dataset_path>` restores the local layout.

## Architecture Overview

### Training System

- `BaseTuner` (`finetune/base.py`): abstract base class for all trainers.
- `SDXLTuner` (`finetune/sdxl.py`) and `SD15Tuner` (`finetune/sd15.py`): model-family-specific implementations.
- Supported techniques: full fine-tuning of UNet and text encoders, LoRA, LoKr, noise offset, input perturbation, SNR gamma weighting, EMA, gradient checkpointing, mixed precision (fp16/bf16), multiple optimizers (AdamW, Adafactor, Prodigy, Lion, ...), timestep bias strategies (uniform, logit, range), LR schedulers with warmup.
- Training utilities (`finetune/utils/`): optimizer/scheduler setup, LoRA/LoKr network creation, sample generation during training, SNR weighting and timestep sampling strategies.

### Data Pipeline

- Preprocessing runs as three stages (`run_prepare.py`): `LatentsGenerateProcessor` (pre-computes VAE latents) → `TaggingProcessor` (WD Tagger; also imports same-name sidecar `.txt` tags) → `CreateParquetProcessor` (aggregates metadata).
- Storage layout is SHA256-sharded via `dataset/utils/sharded_path()`: `root/ab/cd/<sha256>.<ext>` for both `latents/*.npz` and `tags/*.txt`.
- NPZ files store **latents only**. Per-image metadata (`train_resolution`, `original_size`, `crop_ltrb`) lives in `latents/latents_meta.parquet`, and `CreateParquetProcessor` joins it with tags into the final `metadata.parquet` — the single source of truth consumed by training. Legacy NPZ files with embedded metadata are still readable: the processors harvest their metadata without re-encoding, and `DiffusionDataset` falls back to embedded fields when manifest columns are absent.
- `DiffusionDataset` loads image-latent/text pairs; `BucketBasedBatchSampler` groups images by aspect-ratio bucket (resolutions from 640×1536 to 1536×640).
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
