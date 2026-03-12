---
layout: home

hero:
  name: Diffusion Trainer
  text: 稳定扩散训练框架
  tagline: 覆盖 SD 1.5、SDXL、全量微调、LoRA、LoKr 以及数据预处理工作流。
  actions:
    - theme: brand
      text: 快速开始
      link: /quick-start
    - theme: alt
      text: 配置参考
      link: /configuration

features:
  - title: 面向实际训练流程
    details: 文档按安装、数据预处理、配置、训练、排错组织，直接对应仓库当前脚本和目录结构。
  - title: 覆盖主要训练模式
    details: 包含 full-finetune、LoRA、LoKr、LoHA 的配置说明，以及 SD 1.5 / SDXL 的差异。
  - title: 对齐代码实现
    details: 文档内容来自当前入口脚本、配置 dataclass 与数据集实现，不依赖过时 README 假设。
---

`Diffusion Trainer` 是一个基于 PyTorch 的 Stable Diffusion 训练框架，支持 `SD 1.5` 与 `SDXL` 两个模型族，提供全量微调与多种参数高效微调方式，并内置数据预处理流水线。

如果你准备第一次接入这个项目，建议按下面顺序阅读：

1. [安装](/installation)
2. [快速开始](/quick-start)
3. [数据预处理](/data-preparation)
4. [配置说明](/configuration)
5. [训练流程](/training)
