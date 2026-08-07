# Diffusion Trainer

PyTorch-based training framework for diffusion models. The repository covers SD 1.5, SDXL and Lumina 2 (NextDiT) workflows, including full fine-tuning, LoRA, LoKr, and dataset preparation.

## Features

- Support for SD 1.5, SDXL and Lumina 2 / Neta Lumina
- Both the DDPM (epsilon / v-prediction) and rectified-flow objectives
- Full fine-tuning and parameter-efficient training
- Dataset preparation pipeline for latents, tags, and parquet metadata
- Bucket-based batching for mixed aspect ratios
- Mixed precision training with fp16 and bf16
- Gradient checkpointing and EMA support
- WandB integration for experiment tracking

## Installation

Python 3.12+ is required. This project uses `uv`.

```bash
git clone <repository-url>
cd diffusion-trainer
uv sync
```

## Quick Start

### Prepare the dataset

```bash
uv run python run_prepare.py \
  --image_path /path/to/images \
  --target_path datasets/sample \
  --vae_path /path/to/vae
```

### Start training

```bash
# SDXL
uv run python run_train.py --config configs/sdxl.toml --model_family sdxl

# SD 1.5
uv run python run_train.py --config configs/sd15.toml --model_family sd15

# Lumina 2 / Neta Lumina
uv run python run_train.py --config configs/lumina2_lora.toml --model_family lumina2
```

Lumina 2 uses a 16-channel VAE and a rectified-flow objective, so it needs its
own prepared dataset and its own config block — see
[`configs/lumina2_lora.toml`](configs/lumina2_lora.toml) for the details and for
which SD-lineage options stop applying.

### Inspect a dataset

```bash
uv run python run_load_dataset.py
```

## Main Files

```text
.
|- configs/
|  |- sd15.toml
|  |- sd15_lora.toml
|  |- sdxl.toml
|  `- sdxl_lokr.toml
|- docs/
|  `- zh/
|- src/diffusion_trainer/
|  |- config/
|  |- dataset/
|  |  |- processors/
|  |  `- utils/
|  |- finetune/
|  |  |- base.py
|  |  |- sd15.py
|  |  |- sdxl.py
|  |  `- utils/
|  |- lib/
|  |- shared/
|  `- utils/
|- run_load_dataset.py
|- run_prepare.py
`- run_train.py
```

## Data Preparation Pipeline

`run_prepare.py` combines three steps:

1. Generate VAE latents.
2. Create image tags with WD Tagger.
3. Build `metadata.parquet` for training.

The generated dataset is typically written under a target directory such as `datasets/sample/`.

## Development

```bash
uv run ruff check --fix .
uv run pytest tests/
pyright
```

## Documentation

Detailed documentation is available in [`docs/zh`](docs/zh/index.md), including installation, configuration, data preparation, training, and FAQ pages.
