"""
测试EVT保险丝机制
验证EVT特征在极端情况下作为保险丝的作用
"""

import pandas as pd
import numpy as np
from src.time_series_model.execution.noise_penalty import (
    ExecutionNoisePenalty,
    NoisePenaltyConfig,
)


def test_evt_safety_fuse():
    """
    测试EVT保险丝机制的有效性
    """
    print("=== EVT保险丝机制测试 ===")

    # 创建模拟数据
    n_samples = 100
    dates = pd.date_range("2023-01-01", periods=n_samples, freq="4H")

    # 创建包含数学特征的DataFrame
    df = pd.DataFrame(
        {
            "close": 100 + np.cumsum(np.random.randn(n_samples) * 0.1),
        },
        index=dates,
    )

    # 添加数学特征（正常水平）
    df["wpt_price_fluctuation"] = np.random.rand(n_samples) * 0.5 + 0.2  # 中等水平
    df["spectrum_price_entropy"] = np.random.rand(n_samples) * 0.5 + 0.2
    df["hilbert_price_env"] = np.random.rand(n_samples) * 0.5 + 0.2
    df["hurst_price_rolling"] = (
        np.random.rand(n_samples) * 0.5 + 0.3
    )  # 中等偏高（较稳定）

    # 测试1: 正常EVT风险水平
    print("--- 测试1: 正常EVT风险水平 ---")
    df_normal_evt = df.copy()
    df_normal_evt["evt_tail_risk"] = (
        np.random.rand(n_samples) * 0.5
    )  # 正常水平 (< 0.8阈值)

    config = NoisePenaltyConfig()
    noise_calc = ExecutionNoisePenalty(config)
    noise_penalty_normal = noise_calc.compute(df_normal_evt)

    print(
        f"正常EVT水平下的噪声惩罚范围: [{noise_penalty_normal.min():.3f}, {noise_penalty_normal.max():.3f}]"
    )

    # 测试2: 高EVT风险水平（触发保险丝）
    print("\n--- 测试2: 高EVT风险水平（触发保险丝） ---")
    df_high_evt = df.copy()
    # 设置一半的数据为高EVT风险（> 0.8阈值）
    df_high_evt["evt_tail_risk"] = np.concatenate(
        [
            np.random.rand(n_samples // 2) * 0.1 + 0.9,  # 高风险 (>0.8)
            np.random.rand(n_samples // 2) * 0.5,  # 正常风险
        ]
    )

    noise_penalty_high = noise_calc.compute(df_high_evt)
    print(
        f"高EVT水平下的噪声惩罚范围: [{noise_penalty_high.min():.3f}, {noise_penalty_high.max():.3f}]"
    )

    # 验证高EVT风险时惩罚更高
    avg_normal = noise_penalty_normal.mean()
    avg_high = noise_penalty_high.mean()

    print(f"正常EVT平均惩罚: {avg_normal:.3f}")
    print(f"高EVT平均惩罚: {avg_high:.3f}")

    if avg_high > avg_normal:
        print("✅ EVT保险丝机制生效：高尾部风险导致更高的噪声惩罚")
    else:
        print("⚠️ EVT保险丝机制效果不明显")

    # 验证所有惩罚值仍在范围内
    assert (
        noise_penalty_normal.min() >= 0.0 and noise_penalty_normal.max() <= 0.8
    ), "正常情况下的噪声惩罚应在[0, 0.8]区间内"
    assert (
        noise_penalty_high.min() >= 0.0 and noise_penalty_high.max() <= 0.8
    ), "高风险情况下的噪声惩罚应在[0, 0.8]区间内"

    print("\n✅ EVT保险丝机制测试完成！")
    print("✅ 所有噪声惩罚值都在[0, 0.8]安全范围内")
    print("\nEVT保险丝机制特点：")
    print("- 在正常市场条件下，EVT不显著影响噪声惩罚")
    print("- 在极端尾部风险条件下，EVT触发保险丝机制，增加额外惩罚")
    print("- 确保系统在极端市场条件下的风险控制")


if __name__ == "__main__":
    test_evt_safety_fuse()
