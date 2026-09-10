"""
SR 突破策略专属评估方法

评估指标：
- MSE：预测 R/R 与真实 R/R 的均方误差
- Top Decile R/R：预测值最高的 10% 样本的平均 R/R
- 相关性：预测值与真实值的 Spearman 相关系数
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional
from scipy.stats import spearmanr


def evaluate_sr_breakout(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    backtest_results: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    评估 SR 突破策略模型性能（回归任务）

    Args:
        y_true: 真实 R/R 值（连续值）
        y_pred: 预测 R/R 值（连续值）
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

    # 1. 回归指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    mse = mean_squared_error(y_true_valid, y_pred_valid)
    metrics["mse"] = mse
    metrics["rmse"] = np.sqrt(mse)
    metrics["mae"] = mean_absolute_error(y_true_valid, y_pred_valid)
    metrics["r2"] = r2_score(y_true_valid, y_pred_valid)

    # 2. 相关性指标
    if len(y_true_valid) > 1:
        # Spearman 秩相关系数（Rank IC）
        spearman_corr, spearman_p = spearmanr(y_pred_valid, y_true_valid)
        metrics["spearman_corr"] = spearman_corr
        metrics["spearman_pvalue"] = spearman_p

        # Pearson 相关系数
        pearson_corr = np.corrcoef(y_pred_valid, y_true_valid)[0, 1]
        metrics["pearson_corr"] = pearson_corr

    # 3. Top Decile R/R（预测值最高的 10% 样本的平均真实 R/R）
    if len(y_pred_valid) >= 10:
        top_decile_idx = np.argsort(y_pred_valid)[-len(y_pred_valid) // 10 :]
        top_decile_true_rr = y_true_valid[top_decile_idx]
        metrics["top_decile_avg_rr"] = np.mean(top_decile_true_rr)
        metrics["top_decile_median_rr"] = np.median(top_decile_true_rr)

        # Bottom Decile R/R（预测值最低的 10% 样本的平均真实 R/R）
        bottom_decile_idx = np.argsort(y_pred_valid)[: len(y_pred_valid) // 10]
        bottom_decile_true_rr = y_true_valid[bottom_decile_idx]
        metrics["bottom_decile_avg_rr"] = np.mean(bottom_decile_true_rr)

        # Top-Bottom Spread
        metrics["top_bottom_spread"] = (
            metrics["top_decile_avg_rr"] - metrics["bottom_decile_avg_rr"]
        )

    # 4. 从回测结果获取其他指标
    if backtest_results:
        if "avg_rr" in backtest_results:
            metrics["backtest_avg_rr"] = backtest_results["avg_rr"]
        if "total_return" in backtest_results:
            metrics["total_return"] = backtest_results["total_return"]
        if "sharpe_ratio" in backtest_results:
            metrics["sharpe_ratio"] = backtest_results["sharpe_ratio"]
        if "max_drawdown" in backtest_results:
            metrics["max_drawdown"] = backtest_results["max_drawdown"]

    return metrics
