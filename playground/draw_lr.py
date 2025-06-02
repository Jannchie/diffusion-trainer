# CosineAnnealingLR

# 绘制 CosineAnnealingLR 的学习率曲线
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# 假设一个虚拟的 optimizer
optimizer = torch.optim.SGD([torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))], lr=0.1)

# 参数设置
T_0 = 50  # 初始周期
T_mult = 2  # 每次周期长度倍增
num_cycles = 3  # 总轮数
max_lr_start = 0.1  # 初始最高学习率
max_lr_decay = 0.5  # 每轮最高学习率衰减系数
eta_min = 0.05  # 最低学习率保持不变
warmup_steps = 5  # 预热步数

lrs = []
total_steps = 0
max_lr = max_lr_start

for _ in range(num_cycles):
    for param_group in optimizer.param_groups:
        param_group["lr"] = max_lr
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=1, eta_min=eta_min)
    for step in range(T_0):
        if step < warmup_steps:
            # 线性预热
            warmup_lr = (max_lr - eta_min) * (step + 1) / warmup_steps + eta_min
            for param_group in optimizer.param_groups:
                param_group["lr"] = warmup_lr
        else:
            scheduler.step(total_steps + step)
        optimizer.step()
        lrs.append(optimizer.param_groups[0]["lr"])
    total_steps += T_0
    max_lr *= max_lr_decay  # 衰减最高学习率

plt.plot(range(len(lrs)), lrs)
plt.xlabel("Step")
plt.ylabel("Learning Rate")
plt.title("CosineAnnealingWarmRestarts with Decaying max_lr")
plt.grid(True)
plt.savefig("cosine_lr.png")  # 保存图片
# plt.show()  # 非交互环境下注释掉
