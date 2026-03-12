# 常见问题

## `dataset_path` 应该填什么

应该填预处理后的数据集根目录，而不是原图目录。例如：

```toml
dataset_path = "datasets/sample"
```

这个目录里通常需要有：

- `metadata.parquet`
- `latents/`
- `tags/`

## `image_path` 和 `dataset_path` 可以同时存在吗

可以。

常见用法是：

- `dataset_path` 指向训练缓存
- `image_path` 指向原图目录

只有在你把 `skip_prepare_image = false` 时，训练器才会在启动时使用 `image_path` 自动准备数据。

## 没有 `metadata.parquet` 会怎样

训练器会在主进程尝试自动创建它。

但前提是你的 `dataset_path` 目录里已经有：

- `latents/`
- `tags/`

如果这些目录都还没有，应该先运行 `run_prepare.py`。

## 预处理后为什么没有 `caption`

因为当前默认的 `run_prepare.py` 只会生成：

- latent
- tag
- parquet

其中 parquet 默认不写入 caption 列。训练加载器支持 caption，但需要你自行扩展生成流程或手动补充。

## 训练输出目录为什么多了一层 `model_name`

因为 `save_dir` 会在配置对象初始化后自动拼接 `model_name`。例如：

```toml
save_dir = "./out"
model_name = "demo"
```

最终目录是：

```text
./out/demo
```

## 应该优先用 `full-finetune` 还是 `lora`

一般建议：

- 数据量较小或资源有限时优先 `lora`
- 想保留最小训练开销时优先 `lora` 或 `lokr`
- 明确需要完整参数更新时再用 `full-finetune`

## `adamW8bit` 和 `adafactor` 怎么选

可以按下面理解：

- `adamW8bit` 更常见，适合作为默认起点
- `adafactor` 占用更省，项目里还会自动切换成 `constant_with_warmup`

如果你没有明确偏好，先用样例配置即可。

## 什么时候需要重新生成 latent

通常在这些情况下需要：

- 更换了 `vae_path`
- 更换了训练数据
- 数据缓存损坏

如果只是改学习率或训练模式，通常不需要重跑预处理。
