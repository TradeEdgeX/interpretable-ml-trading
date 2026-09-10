"""
自定义分位数损失函数，实现Q50约束

Q50约束：Q50的损失应该 <= Q10和Q90的损失
在训练Q10/Q90时，如果它们的损失小于Q50，就增加惩罚项
"""

import numpy as np
import lightgbm as lgb
from typing import Callable, Tuple


def quantile_loss_with_q50_constraint(
    q50_predictions: np.ndarray,
    q50_loss: float,
    constraint_weight: float = 1.0,
) -> Callable:
    """
    创建带Q50约束的分位数损失函数

    Args:
        q50_predictions: Q50模型的预测值（用于计算约束）
        q50_loss: Q50模型的损失值（用于约束）
        constraint_weight: 约束权重（默认1.0）

    Returns:
        LightGBM自定义objective函数
    """

    def quantile_objective_with_constraint(
        y_pred: np.ndarray,
        y_true: lgb.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        自定义分位数损失函数（带Q50约束）

        Args:
            y_pred: 当前模型的预测值
            y_true: LightGBM Dataset对象，包含真实值

        Returns:
            (gradient, hessian) tuple
        """
        # 获取真实值
        y = y_true.get_label()

        # 获取quantile_alpha（从y_true的metadata中获取，如果没有则使用默认值）
        # 注意：LightGBM的Dataset不支持metadata，我们需要通过闭包传递
        # 这里假设alpha已经在外部设置，我们通过闭包变量获取

        # 计算标准quantile loss的gradient和hessian
        # 标准quantile loss: L = alpha * max(0, y - y_pred) + (1 - alpha) * max(0, y_pred - y)
        # Gradient: dL/dy_pred = -alpha if y >= y_pred, else (1 - alpha)
        # Hessian: d²L/dy_pred² = 0 (quantile loss是分段线性的)

        # 由于我们不知道alpha，我们需要从外部传入
        # 这里我们使用一个技巧：通过比较y和y_pred来推断alpha
        # 但实际上，alpha应该在创建objective时传入

        # 临时方案：使用闭包变量存储alpha
        # 更好的方案：在创建objective时传入alpha

        # 计算误差
        error = y - y_pred

        # 标准quantile loss的gradient和hessian
        # 注意：这里需要知道alpha，但LightGBM的objective函数无法直接获取
        # 我们需要通过闭包传递alpha

        # 由于LightGBM的限制，我们使用一个变通方法：
        # 在创建objective时，通过闭包传递alpha和q50信息

        # 这里先返回标准quantile loss的gradient和hessian
        # 实际的约束会在外部通过调整权重实现

        # 标准quantile loss的gradient
        # gradient = -alpha if error >= 0 else (1 - alpha)
        # 但我们需要alpha，所以这里先返回一个占位符

        # 实际上，LightGBM的quantile objective已经内置了alpha参数
        # 我们需要创建一个包装函数，在调用时传入alpha

        raise NotImplementedError(
            "This function should be created with create_quantile_objective_with_q50_constraint"
        )

    return quantile_objective_with_constraint


def create_quantile_objective_with_q50_constraint(
    alpha: float,
    q50_predictions: np.ndarray,
    q50_loss: float,
    constraint_weight: float = 1.0,
) -> Callable:
    """
    创建带Q50约束的分位数损失函数

    Args:
        alpha: 分位数alpha值（0.1 for Q10, 0.9 for Q90）
        q50_predictions: Q50模型的预测值（与y_true对齐）
        q50_loss: Q50模型的平均损失值
        constraint_weight: 约束权重（默认1.0，越大约束越强）

    Returns:
        LightGBM自定义objective函数 (gradient, hessian)
    """

    def quantile_objective(
        y_pred: np.ndarray,
        y_true: lgb.Dataset,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        自定义分位数损失函数（带Q50约束）

        标准quantile loss:
        L = alpha * max(0, y - y_pred) + (1 - alpha) * max(0, y_pred - y)

        带Q50约束的损失:
        L_constrained = L_quantile + constraint_weight * penalty

        其中penalty = max(0, q50_loss - current_loss) 如果current_loss < q50_loss

        Args:
            y_pred: 当前模型的预测值
            y_true: LightGBM Dataset对象，包含真实值

        Returns:
            (gradient, hessian) tuple
        """
        # 获取真实值
        y = y_true.get_label()

        # 确保y_pred和y长度一致
        if len(y_pred) != len(y):
            raise ValueError(f"Length mismatch: y_pred={len(y_pred)}, y={len(y)}")

        # 确保q50_predictions长度一致
        q50_preds = q50_predictions
        if len(q50_preds) != len(y):
            # 如果长度不匹配，尝试对齐（使用前len(y)个值）
            if len(q50_preds) > len(y):
                q50_preds = q50_preds[: len(y)]
            else:
                # 如果q50_predictions太短，使用最后一个值填充
                q50_preds = np.pad(
                    q50_preds, (0, len(y) - len(q50_predictions)), mode="edge"
                )

        # 计算误差
        error = y - y_pred

        # 标准quantile loss的gradient
        # dL/dy_pred = -alpha if error >= 0, else (1 - alpha)
        gradient_base = np.where(error >= 0, -alpha, 1.0 - alpha)

        # 计算当前样本的quantile loss
        current_loss_per_sample = np.where(
            error >= 0, alpha * error, (1.0 - alpha) * (-error)
        )

        # 计算约束惩罚项
        # 如果当前损失 < q50_loss，增加惩罚
        # penalty = max(0, q50_loss - current_loss) * constraint_weight
        loss_diff = q50_loss - current_loss_per_sample
        penalty_mask = loss_diff > 0  # 当前损失小于Q50损失

        # 惩罚项的gradient
        # d(penalty)/dy_pred = -constraint_weight * sign(error) if penalty > 0
        # 如果error >= 0: d(penalty)/dy_pred = constraint_weight
        # 如果error < 0: d(penalty)/dy_pred = -constraint_weight
        penalty_gradient = np.where(
            penalty_mask,
            np.where(error >= 0, constraint_weight, -constraint_weight),
            0.0,
        )

        # 总gradient = 标准gradient + 惩罚gradient
        gradient = gradient_base + penalty_gradient

        # Hessian（quantile loss的hessian为0，但约束项也是线性的，所以hessian也是0）
        # 但为了数值稳定性，我们使用一个小的正数
        hessian = np.full_like(gradient, 1e-6)

        return gradient, hessian

    return quantile_objective


def create_quantile_metric_with_q50_constraint(
    alpha: float,
    q50_predictions: np.ndarray,
    q50_loss: float,
) -> Callable:
    """
    创建带Q50约束的分位数评估函数

    Args:
        alpha: 分位数alpha值
        q50_predictions: Q50模型的预测值
        q50_loss: Q50模型的平均损失值

    Returns:
        LightGBM自定义evaluation函数
    """

    def quantile_metric(
        y_pred: np.ndarray,
        y_true: lgb.Dataset,
    ) -> Tuple[str, float, bool]:
        """
        计算带Q50约束的分位数损失

        Returns:
            (name, value, is_higher_better) tuple
        """
        y = y_true.get_label()
        error = y - y_pred

        # 标准quantile loss
        quantile_loss = np.mean(
            np.where(error >= 0, alpha * error, (1.0 - alpha) * (-error))
        )

        # 约束惩罚
        current_loss_per_sample = np.where(
            error >= 0, alpha * error, (1.0 - alpha) * (-error)
        )
        loss_diff = q50_loss - current_loss_per_sample
        penalty = np.mean(np.maximum(0, loss_diff))

        # 总损失
        total_loss = quantile_loss + penalty

        return ("quantile_loss_with_constraint", float(total_loss), False)

    return quantile_metric
