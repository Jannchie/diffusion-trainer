# Repository Guidelines

## Project Structure & Module Organization
- Core code lives in `src/diffusion_trainer`: `config/` (typed configs), `dataset/` (processors, bucketing, tagging), `finetune/` (trainers for SD 1.5/SDXL, utilities), `shared/` (losses, schedulers), `utils/` and `lib/` (helpers and third-party glue).
- Entry scripts: `run_prepare.py` (latents, tagging, parquet metadata), `run_train.py` (training), `run_load_dataset.py` (metadata inspection). Example configs in `configs/`.
- Data and artifacts are expected under `datasets/`, `models/`, `playground/`, and `out/`; keep large assets out of version control.
- Docs live in `docs/`; tests belong in `tests/` (currently minimal).

## Build, Test, and Development Commands
- Install: `uv install`.
- Lint/format: `uv run ruff check --fix .`.
- Type check: `pyright` (ensure it is available in your environment).
- Tests: `uv run pytest tests/`.
- Data prep example: `uv run python run_prepare.py --image_path /data/raw --target_path datasets/sample --vae_path models/sdxl-vae`.
- Train example: `uv run python run_train.py --config configs/sdxl.toml --model_family sdxl`.

## Coding Style & Naming Conventions
- Python 3.12+ with strict type hints on all functions; prefer `pathlib.Path`, f-strings, and small, composable functions.
- Ruff enforces `select = ["ALL"]` with limited ignores (line length 160). Do not add new ignores without justification; avoid `noqa` except for clear false positives.
- Modules/functions use `snake_case`; classes use `PascalCase`; configs and datasets use lowercase with hyphens/underscores (e.g., `sdxl.toml`, `bucketed-metadata.parquet`).
- Keep logging informative but concise; avoid heavyweight prints in hot paths.

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
