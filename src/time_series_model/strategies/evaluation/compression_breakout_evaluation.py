"""
压缩区突破策略专属评估方法

评估指标：
- 准确率：多分类准确率
- F1-score：每个类别的 F1-score（macro, micro, weighted）
- 混淆矩阵：3x3 混淆矩阵（-1=向下突破, 0=无突破, +1=向上突破）
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)


def evaluate_compression_breakout(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    predictions_proba: Optional[np.ndarray] = None,
    backtest_results: Optional[Dict] = None,
) -> Dict[str, float]:
    """
    评估压缩区突破策略模型性能（多分类任务）

    Args:
        y_true: 真实标签（-1=向下突破, 0=无突破, +1=向上突破）
        y_pred: 预测标签（-1, 0, +1）
        predictions_proba: 预测概率（shape: [n_samples, 3]），可选
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

    # 1. 准确率
    accuracy = accuracy_score(y_true_valid, y_pred_valid)
    metrics["accuracy"] = accuracy * 100.0  # 转换为百分比

    # 2. F1-score（macro, micro, weighted）
    f1_macro = f1_score(y_true_valid, y_pred_valid, average="macro", zero_division=0.0)
    f1_micro = f1_score(y_true_valid, y_pred_valid, average="micro", zero_division=0.0)
    f1_weighted = f1_score(
        y_true_valid, y_pred_valid, average="weighted", zero_division=0.0
    )
    metrics["f1_macro"] = f1_macro
    metrics["f1_micro"] = f1_micro
    metrics["f1_weighted"] = f1_weighted

    # 3. 每个类别的精确率、召回率、F1
    classes = [-1, 0, 1]
    for cls in classes:
        cls_mask_true = y_true_valid == cls
        cls_mask_pred = y_pred_valid == cls

        if np.sum(cls_mask_true) > 0:
            precision = precision_score(
                y_true_valid == cls,
                y_pred_valid == cls,
                zero_division=0.0,
            )
            recall = recall_score(
                y_true_valid == cls,
                y_pred_valid == cls,
                zero_division=0.0,
            )
            f1 = f1_score(
                y_true_valid == cls,
                y_pred_valid == cls,
                zero_division=0.0,
            )

            cls_name = "down" if cls == -1 else ("none" if cls == 0 else "up")
            metrics[f"precision_{cls_name}"] = precision
            metrics[f"recall_{cls_name}"] = recall
            metrics[f"f1_{cls_name}"] = f1

    # 4. 混淆矩阵
    cm = confusion_matrix(y_true_valid, y_pred_valid, labels=classes)
    metrics["confusion_matrix"] = cm.tolist()

    # 混淆矩阵统计
    metrics["true_positive_up"] = int(cm[2, 2]) if cm.shape[0] > 2 else 0
    metrics["true_positive_down"] = int(cm[0, 0]) if cm.shape[0] > 0 else 0
    metrics["true_positive_none"] = int(cm[1, 1]) if cm.shape[0] > 1 else 0

    # 5. 从回测结果获取其他指标
    if backtest_results:
        if "total_return" in backtest_results:
            metrics["total_return"] = backtest_results["total_return"]
        if "sharpe_ratio" in backtest_results:
            metrics["sharpe_ratio"] = backtest_results["sharpe_ratio"]
        if "max_drawdown" in backtest_results:
            metrics["max_drawdown"] = backtest_results["max_drawdown"]

    return metrics
