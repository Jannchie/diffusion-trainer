# 安装

## 环境要求

- Python `3.12+`
- `uv`
- Node.js `22+`
- `pnpm`
- NVIDIA GPU 与对应 CUDA 环境

如果你只想阅读文档，Python 环境不是必须的；如果你要实际训练模型，建议先完成 Python 依赖安装。

## 克隆仓库

```bash
git clone <repository-url>
cd diffusion-trainer
```

## 安装 Python 依赖

项目使用 `uv` 管理 Python 依赖：

```bash
uv install
```

安装完成后，你可以运行以下命令确认环境正常：

```bash
uv run python run_train.py --help
uv run python run_prepare.py --help
```

## 安装文档依赖

文档站点使用 `VitePress`，依赖独立放在 `docs/` 目录：

```bash
cd docs
pnpm install
```

## 启动文档站点

```bash
cd docs
pnpm dev
```

构建静态文档：

```bash
cd docs
pnpm build
```

预览构建结果：

```bash
cd docs
pnpm preview
```

## 可选的质量检查

```bash
uv run ruff check --fix .
uv run pytest tests/
pyright
```

如果你只维护文档，至少应确保 `cd docs && pnpm build` 可以通过。
