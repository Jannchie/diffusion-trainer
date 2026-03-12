# 快速开始

本页给出一条最短可运行路径：准备数据、挑选配置、启动训练。

## 1. 准备图像目录

假设你的原始图像位于：

```text
/data/raw-images
```

建议单独准备一个数据缓存目录，例如：

```text
datasets/sample
```

这个目录会被 `run_prepare.py` 写入：

- `latents/`
- `tags/`
- `metadata.parquet`

## 2. 运行数据预处理

```bash
uv run python run_prepare.py \
  --image_path /data/raw-images \
  --target_path datasets/sample \
  --vae_path madebyollin/sdxl-vae-fp16-fix
```

如果你训练的是 `SD 1.5`，也可以替换成对应 VAE 路径。

## 3. 选择配置文件

仓库已经提供了 4 个配置样例：

- `configs/sd15.toml`
- `configs/sd15_lora.toml`
- `configs/sdxl.toml`
- `configs/sdxl_lokr.toml`

推荐起点：

- 训练 `SDXL` 全量微调：`configs/sdxl.toml`
- 训练 `SD 1.5` 全量微调：`configs/sd15.toml`
- 训练 `SD 1.5 LoRA`：`configs/sd15_lora.toml`
- 训练 `SDXL LoKr`：`configs/sdxl_lokr.toml`

## 4. 修改最关键的配置项

至少检查下面几个字段：

```toml
model_path = "models/your-base-model"
dataset_path = "datasets/sample"
vae_path = "madebyollin/sdxl-vae-fp16-fix"
model_name = "your-run-name"
save_dir = "./out"
```

还要确认：

- `model_family` 与配置文件一致
- `mode` 与你的训练目标一致
- `batch_size` 和 `gradient_accumulation_steps` 适配显存
- `mixed_precision` 与硬件兼容

## 5. 启动训练

训练 `SDXL`：

```bash
uv run python run_train.py --config configs/sdxl.toml --model_family sdxl
```

训练 `SD 1.5`：

```bash
uv run python run_train.py --config configs/sd15.toml --model_family sd15
```

## 6. 查看输出目录

训练输出默认写入：

```text
out/<model_name>/
```

`save_dir` 会在运行时自动拼接 `model_name`，也就是说：

```toml
save_dir = "./out"
model_name = "my-model"
```

最终输出目录实际是：

```text
./out/my-model
```

## 7. 下一步

如果你想调优配置，请继续阅读：

- [数据预处理](/data-preparation)
- [训练流程](/training)
- [配置说明](/configuration)
