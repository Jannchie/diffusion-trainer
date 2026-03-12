# 数据集结构

项目当前的数据缓存格式以 `SHA256` 为键组织文件。

## 标准目录布局

推荐的数据集目录如下：

```text
datasets/sample/
├─ latents/
│  └─ ab/
│     └─ cd/
│        └─ abcdef...1234.npz
├─ tags/
│  └─ ab/
│     └─ cd/
│        └─ abcdef...1234.txt
├─ caption/
│  └─ ab/
│     └─ cd/
│        └─ abcdef...1234.txt
└─ metadata.parquet
```

其中：

- `ab` 是哈希前 2 位
- `cd` 是哈希第 3 到第 4 位
- 文件名本体是完整 `SHA256`

## latent 文件格式

每个 `.npz` 至少包含以下字段：

| 字段 | 说明 |
| --- | --- |
| `latents` | VAE 编码后的 latent 张量 |
| `crop_ltrb` | 裁切区域 |
| `original_size` | 原图尺寸 |
| `train_resolution` | 当前样本进入的训练分辨率桶 |

## tag 文件格式

每个 `.txt` 文件保存逗号分隔的标签，例如：

```text
1girl, solo, smile, long hair
```

训练时会把这些标签解析成字符串列表。

## caption 文件格式

`caption/` 目录不是 `run_prepare.py` 默认生成的一部分，但数据集加载器已经支持读取它。

如果你想混合使用自然语言描述与 tag，可以按相同哈希路径手动放置：

```text
caption/ab/cd/<sha256>.txt
```

## parquet 文件角色

`metadata.parquet` 是训练最直接依赖的索引文件。训练器默认优先读取它，再去定位 `latents/` 中的 `.npz` 文件。

当前实现会从 parquet 中读取或推导：

- 样本 key
- `train_resolution`
- `tags`
- `caption`（如果 parquet 里存在该列）

## dataset_path 与 image_path 的区别

### dataset_path

应该指向缓存后的数据集根目录，例如：

```toml
dataset_path = "datasets/sample"
```

### image_path

应该指向原始图像目录，例如：

```toml
image_path = "/data/raw-images"
```

只有当你希望训练前自动执行预处理时，`image_path` 才是必须的。

## 训练时的读取方式

训练器会优先定位：

```text
<dataset_path>/metadata.parquet
```

然后根据 `key` 反推出：

```text
<dataset_path>/latents/<dir1>/<dir2>/<key>.npz
```

因此：

- 不要随意修改哈希目录层级
- 不要手动更改 `.npz` 文件名
- 重新打包数据时要保留目录结构
