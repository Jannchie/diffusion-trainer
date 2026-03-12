# 训练流程

`run_train.py` 是当前训练入口：

```bash
uv run python run_train.py --config configs/sdxl.toml --model_family sdxl
```

## 命令行参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--config` | `configs/sd15.toml` | TOML 配置文件路径 |
| `--model_family` | `sd15` | 可选 `sd15` 或 `sdxl` |

## 训练前会发生什么

训练器启动后会按以下顺序准备运行环境：

1. 应用随机种子
2. 初始化 `accelerate`
3. 加载 pipeline、UNet、Text Encoder、VAE 等模型
4. 创建噪声调度器
5. 准备数据集与 DataLoader
6. 按 `mode` 决定可训练模块
7. 创建优化器和学习率调度器
8. 初始化日志与预览采样

## 数据集加载策略

训练阶段会优先读取：

```text
<dataset_path>/metadata.parquet
```

如果配置里同时给了 `image_path`，并把：

```toml
skip_prepare_image = false
```

设为 `false`，主进程会在训练开始前自动执行：

1. 生成 latent
2. 生成 tag
3. 生成 `metadata.parquet`

这适合小规模实验，但对于正式训练，更推荐提前执行 [数据预处理](/data-preparation)。

## 训练模式

### full-finetune

```toml
mode = "full-finetune"
```

适合需要最大表达能力的训练场景，显存和训练成本也最高。

### lora

```toml
mode = "lora"
```

适合多数角色、风格、小数据集微调，是最常见选择。

### lokr

```toml
mode = "lokr"
```

适合进一步控制可训练参数量的场景，当前项目样例也给出了 SDXL LoKr 配置。

### loha

```toml
mode = "loha"
```

也是 LyCORIS 的一种参数高效微调方式，可用于尝试不同适配器特性。

## 训练输出

输出目录规则如下：

```toml
save_dir = "./out"
model_name = "example-run"
```

最终保存目录会被拼接为：

```text
./out/example-run
```

该目录一般会包含：

- 模型权重或适配器权重
- checkpoint
- 预览图
- 训练日志

实际输出数量取决于：

- `save_every_n_steps`
- `save_every_n_epochs`
- `checkpoint_every_n_steps`
- `checkpoint_epochs`
- `preview_every_n_steps`
- `preview_every_n_epochs`

## 日志与实验跟踪

`log_with` 支持：

- `wandb`
- `tensorboard`
- `none`

如果你不想接入外部日志系统，建议设为：

```toml
log_with = "none"
```

## 学习率调度

项目会根据优化器自动选择调度器：

- `adafactor` 使用 `constant_with_warmup`
- 其他优化器使用 `cosine_with_restarts`

因此：

- `optimizer_warmup_steps` 直接影响 warmup
- `optimizer_num_cycles` 主要作用于非 `adafactor` 场景

## 预览采样

你可以在配置中定义多个 `[[preview_sample_options]]` 条目，训练过程中会定期生成预览图。

示例：

```toml
[[preview_sample_options]]
prompt = "1girl, solo, smile"
negative_prompt = "worst quality, blurry"
seed = 47
steps = 25
width = 768
height = 768
clip_skip = 2
```

这对于观察训练是否跑偏非常有用。
