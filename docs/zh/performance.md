# 训练性能

这一页记录训练路径上已经做过的性能取舍，以及经过评估后**故意不做**的那些。目的不是给通用调优建议，而是把「为什么代码长这样」和「哪些开关还留在桌上」写清楚，免得下次有人重新推导一遍。

## 已落地的优化

### 条件 dropout 不再每步同步 GPU

`base.py` 的 `apply_condition_dropout_to_prompts` 只决定哪些 prompt 字符串被清空，抽签因此留在 CPU 上（`random.random()`）。

之前的实现在 GPU 上生成 mask 再用 `mask.any()` 读回，**每个 micro-step 强制一次设备同步**，而且 `condition_dropout_prob <= 0` 时也照样发生。同步点落在 `process_batch` 开头，意味着 CPU 必须等上一个 micro-step 的 backward 跑完，才能开始下发分词与文本编码的 kernel——梯度累积窗口内的 CPU run-ahead 被完全打断。这与 `optimizer_step` 刻意把 loss 的 gather 推迟到每个 optimizer step 一次的努力正好相反。

代价：随机序列改变，训练不再逐位复现历史 run；训练分布上等价。

### EMA 走 foreach 融合 kernel

`get_ema` 传 `foreach=True`。diffusers 至今默认 `False`，意味着每个 optimizer step 对整个 denoiser 逐张量做 Python 循环——SDXL UNet 有 1500+ 个张量，`use_dual_ema` 再翻一倍。数值上是同一个 lerp 公式（已实测两种模式结果逐位相同）。

顺带记一笔显存：EMA shadow 是 fp32 且常驻 GPU，SDXL 单 EMA 约 10 GB，dual EMA 约 20 GB。

### latent 拷贝用 non_blocking

DataLoader 一直开着 `pin_memory=True`，但 `_move_tensors_to_device_and_dtype` 是同步拷贝，pinned memory 的意义只剩「拷贝本身略快」。加上 `non_blocking=True` 后拷贝才能与流上已排队的计算重叠。

### 预览解码启用 VAE tiling

`_decode_preview_latents` 把 VAE 临时 cast 到 fp32 再整图 decode。这个峰值往往是**整个 run 的显存上限来源**：它发生在权重、优化器状态、EMA shadow 全都驻留显存的时刻，hires 预览还会在更大分辨率上再来一次。改为 `enable_tiling()` 后峰值被压到单块 tile 的激活量级，接缝在预览画质下不可见。

### 用 expandable_segments 取代周期性 empty_cache

`run_train.py` 在 import torch 之前 `setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")`（必须早于 CUDA 初始化）。分桶多分辨率训练会让 caching allocator 产生大量不可用空洞，这正是 expandable segments 的设计场景。

原先的兜底手段是每 50 个 optimizer step 调一次 `torch.cuda.empty_cache()`，但它自带同步，且把缓存块还给驱动后接下来几步要走慢速 `cudaMalloc` 重新扩池。现在放宽到每 1000 步，仅作为兜底保留。

如果需要复现旧行为，设置自己的 `PYTORCH_CUDA_ALLOC_CONF` 即可覆盖（用的是 `setdefault`）。

### 数据集清单用 to_pylist 而不是 iterrows

`DiffusionDataset.from_parquet` 改用 `table.to_pylist()`。`to_pandas().iterrows()` 会为每一行构造一个 Series，实测在 10 万行的合成 manifest 上是 3.3 s 对 2.0 s（约 1.7×），而这条路径每个进程启动时都要走一遍，多卡场景成本 ×N。流式加载器（`streaming.py`）本来就是这么读的，两条路径现在一致。

### latent 不再压缩存储

`np.savez_compressed` 换成 `np.savez`。latent 近似高斯噪声，本来就压不动：实测一张 SDXL 1024px latent 压缩后 122 KB、不压缩 132 KB（**只省 8%**），而加载耗时 0.97 ms 对 0.41 ms（**慢 2.4×**），这个解压成本由每个 dataloader worker 在每个 batch 上重复支付。

`np.load` 对两种格式透明，**已有的压缩数据集无需重新生成**。只有 dataloader-bound 的场景（小模型 + 大 batch + 网络存储）能把它转化成吞吐。

### 入口层的 allocator 配置

`src/diffusion_trainer/__init__.py` 在包导入时 `setdefault` 而不是写在 `run_train.py` 顶部——`run_prepare.py` 和 `scripts/` 下那批同样跑多分辨率 GPU 编码的工具否则都拿不到。allocator 只在**首次分配**时读取这个变量，所以包导入时设置来得及；`setdefault` 保证调用方显式给的值仍然优先。

## 评估过但没有做的

### latent 改存 fp16

目前 `latents_to_numpy` 把 bf16 widen 成 fp32 再落盘，而所有配置的 `vae_dtype` 都是 bf16——也就是按 fp32 存了只有 bf16 有效精度的数据。改存 fp16 能直接砍掉 50% 的文件体积、50% 的 npz 解析与 H2D 流量，**收益比取消压缩大一个量级**。

没做的原因：这是存储精度变更，要动读侧、且新旧数据集会混着两种 dtype，属于独立决策而不是顺手清理。`save_latents_npz`（`latents_generate_processor.py`）现在是唯一的落盘点，真要改只需改那一处。

### manifest 分块读取与裁列

`from_parquet` 的 `to_pylist()` 会一次性物化整份 manifest（实测约 2.07 KB/行，百万行约 2 GB，DDP 下每个 rank 各一份）。`to_pandas().iterrows()` 同样不惰性，所以这轮改动没有变差，只是没做彻底：`table.to_batches(max_chunksize=8192)` 可把峰值降到 O(chunk)，`pq.read_table(columns=...)` 只取实际用到的列还能再省解析时间（从 sharing 导出回来的 manifest 会多带 `shard` / `npz_sha256`）。当前数据集规模下不构成问题。

### 分享包体积

npz 不压缩后，`sharing.py` 的 shard tar（`tarfile.open(..., "w")`，本身无压缩）体积约 +9%，且重新生成 latents 会让所有 `npz_sha256` 变化、远端 dedup 失效需整包重传。读侧完全不受影响：导出只做 `read_bytes` + sha256，`streaming.py` 的 `np.load(BytesIO(...))` 两种格式通吃。

注意别顺手把 shard 改成压缩 tar——`streaming.py` 用的是 `mode="r|"` 而非 `"r|*"`，不做透明解压。


### `dataloader_num_workers` 默认值

默认 2，在 NFS 之类的网络存储上偏保守，建议按机器调到 4–8。这是配置决策，不改默认值。

### 关闭 gradient_checkpointing

默认 `True`。关掉是**单项吞吐收益最大的开关**（+30% 量级），前提是显存放得下。SDXL LoRA 在 80 GB 卡上通常放得下，但默认桶表含 1536×640，dual EMA 与 full-finetune 场景显存确实紧，所以默认值不动——显存富余时自己关。

### UNet channels_last

A100 + bf16 下卷积走 NHWC 通常快 5–15%，但与 xformers attention processor、gradient checkpointing、LyCORIS 包装的兼容性需要实测，个别 op 回退 NCHW 反而会插入 permute。对 Lumina2 的 DiT（无卷积主干）无意义。**尚未 A/B，先不动。**

### 文本编码缓存

latent 已经预缓存，但 prompt 每个 micro-step 都要重新组装、分词、过文本编码器。缓存它是个自然的想法，结论分两半。

**SD 1.5 / SDXL：不做。** 实测 SD1.5、batch 4、768px、85 token caption（enhanced embeddings 切成 2 个 chunk）：

| | 耗时 |
| --- | --- |
| 文本编码 | 20.31 ms |
| UNet fwd+bwd（gradient checkpointing 开） | 1030.87 ms |
| 文本编码占比 | **1.9%** |

即便缓存 100% 命中，整步提速的上限也就是这个数（测量时同卡有其他进程争抢 SM，UNet 的绝对值偏慢，真实占比会高一些，但仍是个位数）。

命中率还有个更硬的问题：训练每个 epoch 完整遍历数据集，所以**带容量上限的 LRU 命中率是 0**，除非容量能装下全部样本。SD1.5 每样本约 236 KB、SDXL 约 318 KB，10 万样本就是 23–31 GB 内存。小数据集能全装，但那正是训练本来就快、最不需要优化的场景。

为个位数的收益引入一个能**静默训错**的缓存层（缓存 key 漏掉任何一个影响 embedding 的因素，训练照跑，只是学错东西），性价比不成立。

**Lumina2：TODO，值得做。** 收益的大头不是时间而是显存——Gemma-2 是 2B、256 token 序列，编码估计占步时 10–25%（未实测），且**常驻约 5 GB 显存**。预编码之后整个 text encoder 可以从显存卸载，这 5 GB 能直接换成更大的 batch，或者换成关掉 gradient checkpointing（+30% 量级的单项开关）。

计划形态：

- 训练前按样本 key 把 Gemma-2 的 hidden states 与 attention mask 写进 `latents/` 旁边的缓存目录
- 配置项显式开启，**不做自动降级**：开启时若检测到下面任一前提不满足，直接报错拒绝启动，而不是静默退回实时编码
- 缓存 key 用 **prompt 字符串本身**，而不是样本 key 加一堆配置项。prompt 已经是 tag 组装、category order、trigger 前缀等全部配置作用后的产物，用它做 key 就不存在「改了 tag 配置而缓存没失效」这类静默错误；额外只需把 `clip_skip` 纳入 key
- 可缓存的前提：`shuffle_tags = false`、`single_tag_dropout` / `all_tags_dropout` / `caption_dropout` 均为 0、且 text encoder 不参与训练。`condition_dropout_prob` **不必关**——它只是在确定性 prompt 和空串之间二选一，两者都是可缓存的常量
- 不把 SD 1.5 / SDXL 纳入同一机制

### TF32

当前是关闭状态，且**不建议打开**：训练前向都在 bf16 autocast 下，TF32 只影响残余的 fp32 matmul（例如 fp32 预览解码），收益近零。这是一个「看似该开其实无用」的开关。

### torch.compile

保持 opt-in。分桶多分辨率等于动态形状，`finetune/utils/__init__.py` 里记着真实的翻车记录（inductor `CantSplit`）。真要上，方向是 `torch.compile(dynamic=True)` 加按桶预热，而不是改默认值。

### 流式训练每 epoch 重建 worker

`spawn` 启动等于新解释器加 import torch，每 epoch 付 5–20 s。可以改成 persistent workers 加共享内存传 epoch，但只有流式用户受益，改动中等，暂未做。
