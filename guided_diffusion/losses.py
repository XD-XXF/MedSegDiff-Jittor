"""
Helpers for various likelihood-based losses. These are ported from the original
Ho et al. diffusion models codebase:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/utils.py
"""

import numpy as np
import jittor as jt


def normal_kl(mean1, logvar1, mean2, logvar2):
    """
    计算两个高斯分布之间的KL散度 (Jittor版本)
    """
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, jt.Var):
            tensor = obj
            break
    assert tensor is not None, "at least one argument must be a Tensor"

    # 确保方差为Jittor张量（支持标量自动转换）
    logvar1, logvar2 = [
        x if isinstance(x, jt.Var) else jt.array(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # 计算KL散度：KL(N(μ₁,σ₁²) || N(μ₂,σ₂²))
    return 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + jt.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * jt.exp(-logvar2)
    )


def approx_standard_normal_cdf(x):
    """
    标准正态分布累积分布函数的快速近似 (Jittor版本)
    """
    return 0.5 * (1.0 + jt.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * jt.pow(x, 3))))


def discretized_gaussian_log_likelihood(x, *, means, log_scales):
    """
    计算离散化高斯分布的对数似然 (Jittor版本)
    """
    assert x.shape == means.shape == log_scales.shape

    # 计算标准化后的上下界
    centered_x = x - means
    inv_stdv = jt.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
    min_in = inv_stdv * (centered_x - 1.0 / 255.0)

    # 使用近似CDF计算概率
    cdf_plus = approx_standard_normal_cdf(plus_in)
    cdf_min = approx_standard_normal_cdf(min_in)

    # 计算对数概率，避免数值不稳定
    log_cdf_plus = jt.log(cdf_plus.clamp(min=1e-12))
    log_one_minus_cdf_min = jt.log((1.0 - cdf_min).clamp(min=1e-12))
    cdf_delta = cdf_plus - cdf_min

    # 根据x的值选择合适的对数概率计算方式
    log_probs = jt.where(
        x < -0.999,
        log_cdf_plus,
        jt.where(x > 0.999, log_one_minus_cdf_min, jt.log(cdf_delta.clamp(min=1e-12))),
    )

    assert log_probs.shape == x.shape
    return log_probs
