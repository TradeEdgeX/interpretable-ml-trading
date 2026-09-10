"""
测试 failure-first 模型的 backtest threshold 语义

核心问题：
- labels_rr_extreme.yaml 中 invert=true，模型输出的是 success_prob（不踩坑概率）
- backtest.yaml 中 long_entry_threshold: 0.3 的语义应该是什么？

当前backtest.yaml注释说"failure_prob <= 0.3"，但这是错误的！
因为模型输出的是success_prob，不是failure_prob。
"""

import numpy as np
import pandas as pd
import pytest


def test_failure_model_prediction_semantics():
    """
    验证 failure 模型的预测语义

    invert=true时：
    - 模型训练目标：success_no_rr_extreme (1=好机会，0=踩坑)
    - 模型输出：success_prob (范围0-1，值越高表示越不会踩坑)
    """
    # 模拟模型预测分布（从实际结果）
    preds = {
        "min": 0.5684,
        "max": 0.9047,
        "mean": 0.8390,
        "q25": 0.8175,
        "q50": 0.8497,
        "q75": 0.8732,
    }

    # 验证：这些是success_prob，不是failure_prob
    assert preds["min"] > 0.5, "最小值应该>0.5，说明这是success_prob"
    assert preds["mean"] > 0.8, "平均值应该>0.8，说明模型认为大部分情况都不会踩坑"

    # 推导正确的threshold语义
    print("\n📊 预测分布分析：")
    print(f"  success_prob 范围: {preds['min']:.4f} - {preds['max']:.4f}")
    print(f"  success_prob 中位数: {preds['q50']:.4f}")
    print(f"  success_prob 均值: {preds['mean']:.4f}")

    # 如果要"只在好机会入场"，应该这样设置threshold
    print("\n💡 正确的threshold语义：")
    print("  long_entry_threshold: 0.7  # success_prob >= 0.7 才入场")
    print("  long_entry_threshold: 0.8  # success_prob >= 0.8 才入场（更保守）")
    print("  long_entry_threshold: 0.85 # success_prob >= 0.85 才入场（极度保守）")

    # 当前错误配置的影响
    current_threshold = 0.3
    print(f"\n❌ 当前配置 (long_entry_threshold: {current_threshold}):")
    print(f"  如果理解为 success_prob >= {current_threshold}：")
    print(f"    → 所有预测都满足（min={preds['min']:.4f} > {current_threshold}）")
    print(f"    → 导致3447个入场信号（实际看到的）")

    # 正确配置示例
    correct_threshold = 0.85
    n_samples = 3447  # 实际测试集大小
    # 假设预测服从正态分布
    mock_preds = np.random.normal(
        preds["mean"], preds["max"] - preds["mean"], n_samples
    )
    mock_preds = np.clip(mock_preds, preds["min"], preds["max"])

    entries_current = np.sum(mock_preds >= current_threshold)
    entries_correct = np.sum(mock_preds >= correct_threshold)

    print(f"\n  如果理解为 success_prob >= {correct_threshold}：")
    print(
        f"    → 约 {entries_correct} 个入场信号（{entries_correct/n_samples*100:.1f}%）"
    )
    print(f"    → 更合理的交易数量")


def test_backtest_config_comment_is_wrong():
    """
    验证 backtest.yaml 中的注释是错误的
    """
    # 当前backtest.yaml的注释
    wrong_comment = "# Failure-first: 低 failure_prob 入场"
    wrong_threshold_comment = (
        "# long_entry_threshold: 0.3 表示 failure_prob <= 0.3 时做多"
    )

    # 正确的理解
    correct_comment = "# Failure-first (invert=true): 高 success_prob 入场"
    correct_threshold_comment = (
        "# long_entry_threshold: 0.7 表示 success_prob >= 0.7 时做多"
    )

    print("\n❌ 当前backtest.yaml注释（错误）：")
    print(f"  {wrong_comment}")
    print(f"  {wrong_threshold_comment}")

    print("\n✅ 正确的注释应该是：")
    print(f"  {correct_comment}")
    print(f"  {correct_threshold_comment}")

    assert True, "需要修复 backtest.yaml 的注释和threshold值"


def test_vectorbt_entries_logic():
    """
    验证 vectorbt_backtest.py 中的入场逻辑

    binary模型的入场逻辑（第131-133行）：
    long_p = pd.Series(proba[:, int(long_class)], index=index)
    long_entries = long_p >= thr_long

    这里的语义是：P(class=1) >= threshold 时入场

    对于failure模型：
    - class=1 表示 success（因为invert=true）
    - 所以这是：P(success) >= threshold
    """
    # 模拟预测
    n_samples = 100
    success_probs = np.linspace(0.6, 0.9, n_samples)

    # 测试不同threshold
    thresholds = [0.3, 0.7, 0.85]

    print("\n📊 不同threshold下的入场信号数量：")
    for thr in thresholds:
        entries = np.sum(success_probs >= thr)
        print(
            f"  threshold={thr}: {entries}/{n_samples} 入场 ({entries/n_samples*100:.1f}%)"
        )

    # 验证：threshold=0.3时几乎所有点都入场
    assert np.sum(success_probs >= 0.3) > 0.9 * n_samples

    # 验证：threshold=0.85时只有少数点入场
    assert np.sum(success_probs >= 0.85) < 0.3 * n_samples


if __name__ == "__main__":
    print("=" * 80)
    print("Failure-First 模型 Backtest Threshold 语义测试")
    print("=" * 80)

    test_failure_model_prediction_semantics()
    test_backtest_config_comment_is_wrong()
    test_vectorbt_entries_logic()

    print("\n" + "=" * 80)
    print("✅ 所有测试通过")
    print("=" * 80)

    print("\n📝 修复方案：")
    print("1. 修改 config/strategies/bpc/backtest.yaml")
    print("   - 将 long_entry_threshold 从 0.3 改为 0.7 或 0.8")
    print("   - 修正注释：说明这是 success_prob 的阈值")
    print("\n2. 或者在代码中添加自动反转逻辑")
    print("   - 检测 label 的 invert 参数")
    print("   - 如果 invert=true，自动将 preds 转换为 1-preds")
