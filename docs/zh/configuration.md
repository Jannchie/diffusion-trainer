# 配置说明

训练配置使用 TOML 文件定义，并在运行时映射到 dataclass。当前仓库的核心配置类型是：

- `BaseConfig`
- `SDXLConfig`
- `SD15Config`

## 最小可用配置

下面是一份适合作为起点的最小配置：

```toml
model_path = "models/base-model"
dataset_path = "datasets/sample"
vae_path = "madebyollin/sdxl-vae-fp16-fix"

model_name = "demo-run"
save_dir = "./out"

mode = "lora"
optimizer = "adamW8bit"

n_epochs = 10
batch_size = 1
gradient_accumulation_steps = 1

mixed_precision = "bf16"
weight_dtype = "fp32"
save_dtype = "fp16"

unet_lr = 1e-4
text_encoder_1_lr = 0
text_encoder_2_lr = 0
```

## 配置分组

## 数据与输入

| 字段 | 类型 / 可选值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `model_path` | `str` | 无 | 基础模型路径或模型标识 |
| `dataset_path` | `str` | 无 | 训练缓存目录，通常包含 `metadata.parquet`、`latents/`、`tags/` |
| `image_path` | `str \| None` | `None` | 原始图像目录，用于训练前自动预处理 |
| `skip_prepare_image` | `bool` | `true` | 是否跳过训练前自动预处理 |
| `vae_path` | `str \| None` | `None` | VAE 路径 |
| `vae_dtype` | `str` | `fp32` | VAE 推理精度 |
| `ss_latent_path` | `str \| None` | `None` | 兼容其他数据源的 latent 路径，当前主流程通常不需要 |
| `ss_meta_path` | `str \| None` | `None` | 兼容其他数据源的元数据路径，当前主流程通常不需要 |

## 文本条件与 dropout

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `shuffle_tags` | `false` | 是否打乱标签顺序 |
| `single_tag_dropout` | `0.0` | 单个标签 dropout 概率 |
| `all_tags_dropout` | `0.0` | 丢弃全部标签的概率 |
| `caption_dropout` | `0.0` | 丢弃 caption 的概率 |
| `use_enhanced_embeddings` | `false` | 是否启用增强文本嵌入 |
| `condition_dropout_prob` | `0.0` | 条件文本整体 dropout 概率，类似 CFG dropout |

## 基础训练参数

| 字段 | 类型 / 可选值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `seed` | `int` | 随机值 | 随机种子 |
| `model_name` | `str` | `my_model` | 训练任务名称 |
| `save_dir` | `str` | `out` | 输出根目录，运行时会自动拼接 `model_name` |
| `save_dtype` | `str` | `fp32` | 保存权重精度 |
| `weight_dtype` | `str` | `fp32` | 训练权重 dtype |
| `mixed_precision` | `float16` / `bfloat16` / `fp16` / `bf16` | `bf16` | 混合精度模式 |
| `prediction_type` | `epsilon` / `v_prediction` / `sample` / `None` | `None` | 噪声预测目标类型 |
| `n_epochs` | `int` | `10` | 训练 epoch 数 |
| `batch_size` | `int` | `8` | batch 大小 |
| `gradient_accumulation_steps` | `int` | `4` | 梯度累积步数 |
| `weight_decay` | `float` | `1e-2` | 权重衰减 |
| `max_grad_norm` | `float` | `1.0` | 梯度裁剪上限 |

## 训练模式与 LyCORIS 参数

| 字段 | 类型 / 可选值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `mode` | `full-finetune` / `lora` / `lokr` / `loha` | `lokr` | 训练模式 |
| `lora_dim` | `int` | `16` | LoRA / LoHA 公共 rank |
| `lora_rank` | `int \| None` | `None` | `lora_dim` 的兼容别名 |
| `lora_alpha` | `float` | `1.0` | LoRA 缩放参数 |
| `lora_dropout` | `float` | `0.0` | 适配器 dropout |
| `conv_dim` | `int \| None` | `None` | 卷积层 rank，未设置时退回 `lora_dim` |
| `conv_alpha` | `float \| None` | `None` | 卷积层 alpha，未设置时退回 `lora_alpha` |
| `lokr_factor` | `int` | `16` | LoKr 分解因子，`-1` 可用于自适应 |
| `lora_multiplier` | `float` | `1.0` | 适配器全局倍率 |
| `lokr_linear_dim` | `int` | `10000` | LoKr 线性层维度 |
| `lokr_feedforward_factor_ratio` | `float` | `0.5` | LoKr 中 FFN 因子相对 Attention 的比例 |

## 噪声、SNR 与采样权重

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `noise_offset` | `0.0` | 噪声偏移强度（epsilon 模型常用 0.02–0.1；ZTSNR 下保持 0，二者修正同一问题） |
| `noise_offset_probability` | `1.0` | 应用噪声偏移的概率 |
| `input_perturbation` | `0.01` | 输入扰动强度 |
| `input_perturbation_steps` | `0` | 输入扰动线性衰减步数 |
| `use_multires_noise` | `false` | 是否启用多分辨率（金字塔）噪声；空间相关噪声与推理时的白噪声不一致，按需开启 |
| `multires_noise_iterations` | `6` | 多分辨率噪声层数 |
| `multires_noise_discount` | `0.8` | 多层权重折扣 |
| `multires_noise_scales` | `None` | 自定义多分辨率缩放列表 |
| `multires_noise_weights` | `None` | 自定义多分辨率权重列表 |
| `use_brownian_noise` | `false` | 是否使用 Brownian noise |
| `brownian_noise_scale` | `1.0` | Brownian noise 强度 |
| `use_smooth_min_snr` | `true` | 是否启用平滑版 Min-SNR |
| `smooth_min_snr_mode` | `sigmoid` | `clip` / `sigmoid` / `tanh` |
| `smooth_min_snr_factor` | `0.15` | Min-SNR 平滑因子 |
| `use_adaptive_noise` | `false` | 是否使用自适应噪声调度 |
| `adaptive_noise_type` | `cosine` | `linear` / `cosine` / `exponential` |
| `adaptive_noise_strength` | `1.0` | 自适应噪声强度 |
| `timestep_bias_strategy` | `uniform` | `uniform` / `logit` / `lognormal`；ZTSNR 需要 `uniform` 保证高噪声尾部被采样 |
| `timestep_lognormal_mean` | `-1.2` | lognormal timestep 采样均值 |
| `timestep_lognormal_std` | `1.2` | lognormal timestep 采样标准差 |
| `timestep_bias_m` | `0.0` | logit 策略参数 m |
| `timestep_bias_s` | `1.0` | logit 策略参数 s |
| `snr_gamma` | `None` | Min-SNR gamma（epsilon 模型常用 5.0；v-pred + ZTSNR 下会把终端步权重压成 0，勿开启） |
| `use_debiased_estimation` | `false` | 是否启用 debiased estimation |
| `rescale_betas_zero_snr` | `false` | 是否启用 zero terminal SNR（仅与 v_prediction 搭配才是健全的） |

## 显存与性能

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `enable_flash_attention` | `true` | 是否启用 Flash Attention / xformers |
| `flash_attention_unet` | `true` | 是否对 UNet 启用 Flash Attention |
| `gradient_checkpointing` | `true` | 是否启用 gradient checkpointing |

## EMA、保存与预览

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `use_ema` | `false` | 是否启用 EMA |
| `ema_start_step` | `0` | 从第几步开始更新 EMA |
| `use_dual_ema` | `false` | 是否维护双 EMA |
| `ema_decay_long` | `0.9999` | 长 EMA 衰减 |
| `ema_decay_short` | `0.9` | 短 EMA 衰减 |
| `save_every_n_steps` | `0` | 每多少步保存一次 |
| `save_every_n_epochs` | `1` | 每多少个 epoch 保存一次 |
| `preview_every_n_steps` | `0` | 每多少步生成预览图 |
| `preview_every_n_epochs` | `1` | 每多少个 epoch 生成预览图 |
| `preview_before_training` | `true` | 训练开始前是否先生成预览 |
| `checkpoint_every_n_steps` | `1000` | checkpoint 步数间隔 |
| `checkpoint_epochs` | `0` | checkpoint 的 epoch 间隔 |

## 日志与优化器

| 字段 | 类型 / 可选值 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `log_with` | `wandb` / `tensorboard` / `none` | `wandb` | 日志后端 |
| `optimizer` | `adamW8bit` / `adafactor` / `prodigy` / `lion` / `lion8bit` | `adamW8bit` | 优化器 |
| `optimizer_warmup_steps` | `int` | `0` | warmup 步数 |
| `optimizer_num_cycles` | `int` | `1` | cosine restart 周期数 |
| `zero_grad_set_to_none` | `bool` | `true` | 是否使用 `set_to_none=True` 清梯度 |

## 预览采样配置

`preview_sample_options` 是一个列表，每项都会被解析为 `SampleOptions`。

单项字段如下：

| 字段 | 必填 | 说明 |
| --- | --- | --- |
| `prompt` | 是 | 正向提示词 |
| `negative_prompt` | 是 | 负向提示词 |
| `steps` | 是 | 采样步数 |
| `seed` | 是 | 采样随机种子 |
| `width` | 否 | 采样宽度 |
| `height` | 否 | 采样高度 |
| `clip_skip` | 否 | `SD 1.5` 相关的 CLIP skip，默认 `2` |

## 模型族专属字段

### SDXLConfig

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `unet_lr` | `1e-5` | UNet 学习率 |
| `text_encoder_1_lr` | `1e-6` | 第一文本编码器学习率 |
| `text_encoder_2_lr` | `1e-6` | 第二文本编码器学习率 |

### SD15Config

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `unet_lr` | `1e-5` | UNet 学习率 |
| `text_encoder_lr` | `1e-6` | 文本编码器学习率 |
| `clip_skip` | `2` | 跳过最后若干个 CLIP block |

## 配置实践建议

### 显存较小

优先调整：

- `batch_size`
- `gradient_accumulation_steps`
- `gradient_checkpointing`
- `mixed_precision`
- `mode`

### 角色或风格微调

建议优先从 `lora` 或 `lokr` 开始，而不是直接 `full-finetune`。

### 想提高训练稳定性

优先关注：

- `noise_offset`
- `input_perturbation`
- `snr_gamma`
- `use_smooth_min_snr`

### 想减少日志依赖

可以直接设：

```toml
log_with = "none"
```
