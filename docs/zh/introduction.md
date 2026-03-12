# 项目简介

## 项目定位

`Diffusion Trainer` 用于训练 Stable Diffusion 系列模型，当前代码库重点覆盖以下场景：

- `SD 1.5` 与 `SDXL` 两个模型族
- `full-finetune` 全量微调
- `lora`、`lokr`、`loha` 参数高效微调
- 基于 `VAE latent` 的训练数据缓存
- 基于 `WD Tagger` 的自动打标
- 基于分桶分辨率的 batch 采样

## 仓库结构

```text
.
├─ configs/                      # 训练配置示例
├─ docs/                         # VitePress 文档站点
│  ├─ .vitepress/               # VitePress 配置
│  └─ zh/                       # 中文文档内容
├─ src/diffusion_trainer/
│  ├─ config/                   # dataclass 配置定义
│  ├─ dataset/                  # 数据集与预处理逻辑
│  ├─ finetune/                 # 训练器实现
│  ├─ shared/                   # 共享工具
│  └─ utils/                    # 训练辅助函数
├─ run_prepare.py               # 数据预处理入口
├─ run_train.py                 # 训练入口
└─ run_load_dataset.py          # 数据集读取示例
```

## 核心能力

### 1. 数据预处理流水线

`run_prepare.py` 会按顺序执行三件事：

1. 生成 `latents`
2. 生成 `tags`
3. 生成 `metadata.parquet`

这三步都可以通过命令行参数单独跳过。

### 2. 训练模式

当前配置字段 `mode` 支持以下值：

- `full-finetune`
- `lora`
- `lokr`
- `loha`

其中：

- `full-finetune` 适合完整训练 UNet / Text Encoder
- `lora` 是最常用的轻量微调方式
- `lokr` 更适合希望进一步压缩参数量的场景
- `loha` 也是 LyCORIS 变体之一

### 3. 训练增强能力

项目已经实现多项训练技巧，包括：

- `noise_offset`
- `input_perturbation`
- `SNR gamma`
- `smooth Min-SNR`
- `multires noise`
- `adaptive noise`
- `EMA`
- `gradient checkpointing`
- `Flash Attention / xformers`

## 推荐阅读路径

如果你已经有图像数据，建议直接进入 [快速开始](/quick-start)。

如果你需要理解配置项含义，直接看 [配置说明](/configuration)。
