# 数据预处理

`run_prepare.py` 是项目的数据预处理入口。它负责把原始图像整理成训练直接可读的数据缓存。

## 预处理输出内容

运行完成后，目标目录通常包含：

```text
datasets/sample/
├─ latents/
├─ tags/
└─ metadata.parquet
```

其中：

- `latents/` 存放 `.npz` 文件
- `tags/` 存放与图像一一对应的标签 `.txt`
- `metadata.parquet` 存放训练时需要的索引信息

## 基本命令

```bash
uv run python run_prepare.py \
  --image_path /path/to/images \
  --target_path datasets/sample \
  --vae_path /path/to/vae
```

## 命令行参数

### 必填参数

| 参数 | 说明 |
| --- | --- |
| `--image_path` | 原始图像目录 |
| `--target_path` | 输出缓存目录 |
| `--vae_path` | VAE 模型路径，支持本地路径、目录或远程模型标识 |

### latent 生成参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--vae_dtype` | 自动选择 | `fp16`、`fp32`、`bf16` 等 |
| `--num_reader` | `4` | 读取线程数 |
| `--num_writer` | `4` | 写入线程数 |

### 标注参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--num_workers` | `4` | 打标工作线程数 |
| `--general_threshold` | `0.35` | 普通标签阈值 |
| `--character_threshold` | `0.9` | 角色标签阈值 |
| `--tag_source` | `wd_tagger` | 标签来源，可选 `wd_tagger` 或 `sidecar_txt` |

### Parquet 参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--max_workers` | `8` | 生成 `metadata.parquet` 的线程数 |

### 流程控制参数

| 参数 | 说明 |
| --- | --- |
| `--skip_latents` | 跳过 latent 生成 |
| `--skip_tagging` | 跳过自动打标 |
| `--skip_parquet` | 跳过 parquet 生成 |
| `--no_skip_existing` | 不跳过已有文件，强制重跑 |

## 处理逻辑

### 1. 生成 latent

项目会读取原图，按预设分辨率桶缩放与裁切，然后使用 VAE 编码成 latent，并保存为压缩的 `.npz` 文件。

每个样本会保存以下字段：

- `latents`
- `crop_ltrb`
- `original_size`
- `train_resolution`

### 2. 自动打标

项目使用 `WD Tagger` 生成标签文本，并保存为逗号分隔的 `.txt` 文件。

如果你已经有与图片同名的 `.txt` 标签文件，可以切换为导入模式：

```bash
uv run python run_prepare.py \
  --image_path /path/to/images \
  --target_path datasets/sample \
  --vae_path /path/to/vae \
  --tag_source sidecar_txt
```

此时程序会读取如下结构中的 sidecar 标签文件：

```text
images/
├─ 001.png
├─ 001.txt
├─ 002.jpg
└─ 002.txt
```

并把这些标签转换写入目标目录下的哈希结构：

```text
datasets/sample/tags/ab/cd/<sha256>.txt
```

### 3. 生成 parquet

当前 `CreateParquetProcessor` 会把如下信息写入 `metadata.parquet`：

- `key`
- `tags`
- `train_resolution`
- `original_size`
- `crop_ltrb`

注意：

- 当前默认预处理流程不会自动写入 `caption`
- 训练数据加载器支持可选的 `caption` 文件和 parquet 列，但需要你自行补充

## 输出目录建议

建议让一个数据集对应一个独立目录，例如：

```text
datasets/
├─ character-a/
├─ style-b/
└─ project-c/
```

这样训练配置里的 `dataset_path` 就能直接指向对应缓存目录。

## 何时需要重跑预处理

以下情况建议重跑：

- 原始图像新增或删除
- 你更换了 `vae_path`
- 你调整了标签阈值
- 你要重新生成 `metadata.parquet`

如果只是训练超参数变化，一般不需要重新预处理。
