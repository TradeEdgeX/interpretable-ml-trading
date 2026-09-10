"""
SR 反转策略专属评估方法

评估指标：
- 胜率（Win Rate）：P(label=1) 的准确率
- 平均 R/R：成功交易的 R/R 平均值
- 失败交易的 R/R 平均值
- Profit Factor：总盈利 / 总亏损
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional


def evaluate_sr_reversal(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predictions_proba: Optional[np.ndarray] = None,
    backtest_results: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    评估 SR 反转策略模型性能

    Args:
        y_true: 真实标签（0=失败, 1=成功）
        y_pred: 预测标签（0=失败, 1=成功）
        predictions_proba: 预测概率（P(success)），可选
        backtest_results: 回测结果字典（包含 trades 列表），可选

    Returns:
        评估指标字典
    """
    metrics = {}

    # 1. 基础分类指标
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    # 过滤 NaN 值
    valid_mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if np.sum(valid_mask) == 0:
        return {"error": "No valid samples for evaluation"}

    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]

    # 准确率（胜率）
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    metrics["win_rate"] = accuracy * 100.0  # 转换为百分比

    # 精确率（Precision）
    precision = precision_score(y_true_valid, y_pred_valid, zero_division=0.0)
    metrics["precision"] = precision

    # 召回率（Recall）
    recall = recall_score(y_true_valid, y_pred_valid, zero_division=0.0)
    metrics["recall"] = recall

    # F1-score
    f1 = f1_score(y_true_valid, y_pred_valid, zero_division=0.0)
    metrics["f1_score"] = f1

    # ROC-AUC（如果有概率预测）
    if predictions_proba is not None:
        proba_valid = predictions_proba[valid_mask]
        if len(np.unique(y_true_valid)) > 1:  # 需要至少两个类别
            try:
                auc = roc_auc_score(y_true_valid, proba_valid)
                metrics["roc_auc"] = auc
            except ValueError:
                metrics["roc_auc"] = 0.0
        else:
            metrics["roc_auc"] = 0.0

    # 2. R/R 相关指标（从回测结果获取）
    if backtest_results and "trades" in backtest_results:
        trades = backtest_results["trades"]
        if len(trades) > 0:
            # 提取 R/R 值
            rr_values = [
                t.get("rr_achieved", 0.0)
                for t in trades
                if t.get("rr_achieved") is not None
            ]

            if rr_values:
                metrics["avg_rr"] = np.mean(rr_values)
                metrics["median_rr"] = np.median(rr_values)
                metrics["std_rr"] = np.std(rr_values)

                # 成功交易的 R/R
                winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
                if winning_trades:
                    win_rr = [
                        t.get("rr_achieved", 0.0)
                        for t in winning_trades
                        if t.get("rr_achieved") is not None
                    ]
                    if win_rr:
                        metrics["avg_win_rr"] = np.mean(win_rr)
                        metrics["max_win_rr"] = np.max(win_rr)

                # 失败交易的 R/R
                losing_trades = [t for t in trades if t.get("pnl", 0) <= 0]
                if losing_trades:
                    loss_rr = [
                        t.get("rr_achieved", 0.0)
                        for t in losing_trades
                        if t.get("rr_achieved") is not None
                    ]
                    if loss_rr:
                        metrics["avg_loss_rr"] = np.mean(loss_rr)
                        metrics["min_loss_rr"] = np.min(loss_rr)

                # Profit Factor
                total_win = sum(
                    t.get("pnl", 0) for t in winning_trades if t.get("pnl", 0) > 0
                )
                total_loss = abs(
                    sum(t.get("pnl", 0) for t in losing_trades if t.get("pnl", 0) < 0)
                )
                if total_loss > 0:
                    metrics["profit_factor"] = total_win / total_loss
                else:
                    metrics["profit_factor"] = float("inf") if total_win > 0 else 0.0

    # 3. 从回测结果获取其他指标
    if backtest_results:
        if "win_rate" in backtest_results:
            metrics["backtest_win_rate"] = backtest_results["win_rate"]
        if "total_return" in backtest_results:
            metrics["total_return"] = backtest_results["total_return"]
        if "sharpe_ratio" in backtest_results:
            metrics["sharpe_ratio"] = backtest_results["sharpe_ratio"]
        if "max_drawdown" in backtest_results:
            metrics["max_drawdown"] = backtest_results["max_drawdown"]

    return metrics
