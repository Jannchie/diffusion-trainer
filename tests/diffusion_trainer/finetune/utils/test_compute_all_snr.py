from typing import Any

import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import compute_snr


def test_compute_all_snr() -> None:
    """测试计算所有时间步信噪比(SNR)的函数。"""
    # 创建一个简单的 DDPMScheduler 实例
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="linear",
        variance_type="fixed_small",
    )

    config: Any = scheduler.config

    # 计算 SNR 值
    all_snr = compute_snr(scheduler, torch.arange(0, config.num_train_timesteps, dtype=torch.long))

    # 验证返回值是否是张量
    assert isinstance(all_snr, torch.Tensor), "all_snr should be a torch.Tensor"

    # 验证张量长度是否等于时间步数
    assert len(all_snr) == config.num_train_timesteps, "Length of all_snr should match num_train_timesteps"

    # 验证 SNR 值是否正数
    assert torch.all(all_snr > 0)

    # 验证 SNR 值是否随着时间步增加而减小（早期时间步 SNR 更高）
    assert torch.all(all_snr[:-1] >= all_snr[1:])

    # 手动计算一些时间步的 SNR 值进行验证
    alphas_cumprod = scheduler.alphas_cumprod
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    alpha = sqrt_alphas_cumprod
    sigma = sqrt_one_minus_alphas_cumprod
    expected_snr = (alpha / sigma) ** 2

    # 验证计算结果是否与预期相符
    torch.testing.assert_close(all_snr, expected_snr, rtol=1e-5, atol=1e-5)

    # 检查边界条件
    # SNR 在 t=0 时应该最大
    assert torch.argmax(all_snr).item() == 0
    # SNR 在 t=999 (最后一步) 时应该最小
    assert torch.argmin(all_snr).item() == config.num_train_timesteps - 1
