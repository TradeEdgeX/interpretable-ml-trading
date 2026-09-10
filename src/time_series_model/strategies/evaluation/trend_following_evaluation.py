"""
趋势跟踪策略专属评估方法

评估指标：
- Rank IC：预测 Rank 与真实 Rank 的 Spearman 相关系数
- Top-Bottom Spread：Top 10% 与 Bottom 10% 的平均收益差
- 分位数收益：按预测 Rank 分组的平均收益
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.stats import spearmanr


def evaluate_trend_following(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    backtest_results: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    评估趋势跟踪策略模型性能（Rank 回归任务）

    Args:
        y_true: 真实收益（用于计算 Rank）
        y_pred: 预测收益（用于计算 Rank）
        backtest_results: 回测结果字典，可选

    Returns:
        评估指标字典
    """
    metrics = {}

    # 过滤 NaN 值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(valid_mask) == 0:
        return {"error": "No valid samples for evaluation"}

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # 1. Rank IC（Spearman 相关系数）
    if len(y_true_valid) > 1:
        spearman_corr, spearman_p = spearmanr(y_pred_valid, y_true_valid)
        metrics["rank_ic"] = spearman_corr
        metrics["rank_ic_pvalue"] = spearman_p

        # Pearson 相关系数
        pearson_corr = np.corrcoef(y_pred_valid, y_true_valid)[0, 1]
        metrics["pearson_corr"] = pearson_corr

    # 2. Top-Bottom Spread
    if len(y_pred_valid) >= 10:
        # 按预测值排序，取 Top 10% 和 Bottom 10%
        top_decile_idx = np.argsort(y_pred_valid)[-len(y_pred_valid) // 10 :]
        bottom_decile_idx = np.argsort(y_pred_valid)[: len(y_pred_valid) // 10]

        top_decile_returns = y_true_valid[top_decile_idx]
        bottom_decile_returns = y_true_valid[bottom_decile_idx]

        metrics["top_decile_avg_return"] = np.mean(top_decile_returns)
        metrics["bottom_decile_avg_return"] = np.mean(bottom_decile_returns)
        metrics["top_bottom_spread"] = (
            metrics["top_decile_avg_return"] - metrics["bottom_decile_avg_return"]
        )

        # Top 10% 的胜率
        top_decile_win_rate = np.mean(top_decile_returns > 0) * 100.0
        metrics["top_decile_win_rate"] = top_decile_win_rate

        # Bottom 10% 的胜率
        bottom_decile_win_rate = np.mean(bottom_decile_returns > 0) * 100.0
        metrics["bottom_decile_win_rate"] = bottom_decile_win_rate

    # 3. 分位数收益（按预测值分成 5 个分位数）
    if len(y_pred_valid) >= 5:
        n_quantiles = 5
        quantile_labels = pd.qcut(
            y_pred_valid, q=n_quantiles, labels=False, duplicates="drop"
        )

        quantile_returns = []
        for q in range(n_quantiles):
            q_mask = quantile_labels == q
            if np.sum(q_mask) > 0:
                q_avg_return = np.mean(y_true_valid[q_mask])
                quantile_returns.append(q_avg_return)
                metrics[f"quantile_{q+1}_avg_return"] = q_avg_return

        # 分位数收益单调性检查（理想情况下，高预测值应该对应高收益）
        if len(quantile_returns) == n_quantiles:
            # 计算分位数收益的 Spearman 相关系数（应该接近 1.0）
            quantile_ranks = np.arange(1, n_quantiles + 1)
            quantile_monotonicity, _ = spearmanr(quantile_ranks, quantile_returns)
            metrics["quantile_monotonicity"] = quantile_monotonicity

    # 4. 从回测结果获取其他指标
    if backtest_results:
        if "total_return" in backtest_results:
            metrics["total_return"] = backtest_results["total_return"]
        if "sharpe_ratio" in backtest_results:
            metrics["sharpe_ratio"] = backtest_results["sharpe_ratio"]
        if "max_drawdown" in backtest_results:
            metrics["max_drawdown"] = backtest_results["max_drawdown"]

    return metrics
