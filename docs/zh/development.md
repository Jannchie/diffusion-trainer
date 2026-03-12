# 开发说明

本页面向需要维护代码或文档的开发者。

## 常用命令

### Python 侧

```bash
uv install
uv run ruff check --fix .
uv run pytest tests/
pyright
```

### 文档侧

```bash
cd docs
pnpm install
pnpm dev
pnpm build
pnpm preview
```

## 代码风格

项目当前开发约束可以概括为：

- Python `3.12+`
- 函数参数与返回值显式标注类型
- 使用 `ruff` 做严格检查
- 使用 `uv` 管理 Python 依赖

如果你要增加 Python 依赖，应优先使用：

```bash
uv add <package>
```

如果你要增加文档或前端依赖，应优先使用：

```bash
pnpm add -D <package>
```

## 文档维护约定

当前文档站点约定如下：

- VitePress 配置位于 `docs/.vitepress/`
- 中文内容位于 `docs/zh/`
- Node 依赖位于 `docs/package.json`

## 新增文档页面

新增页面时通常需要做三件事：

1. 在 `docs/zh/` 下新增 Markdown 文件
2. 在 `docs/.vitepress/config.mts` 的 `sidebar` 中加入导航
3. 运行 `cd docs && pnpm build` 验证无死链与构建错误

## 配置变更同步

当你修改以下位置时，应该同步更新文档：

- `src/diffusion_trainer/config/__init__.py`
- `run_prepare.py`
- `run_train.py`
- `configs/*.toml`

尤其是：

- 新增配置字段
- 更改默认值
- 更改训练模式
- 更改输出目录结构

## 提交前建议

至少执行以下检查：

```bash
uv run ruff check --fix .
cd docs && pnpm build
```

如果改动涉及训练逻辑，再补上：

```bash
uv run pytest tests/
```
