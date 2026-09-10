"""
Outcome-Based 审计标签模块

🟥 套路 B：archetype 审计（必须无信仰）
- 全样本都有 forward_rr
- 目标：发现"在哪些情况下 forward_rr 系统性为负"

核心范式：
- 这不是监督学习，而是「反例空间枚举」
- 树模型不是模型，是高维条件空间的可读切片器
- 产出不是规则，是"死亡证据"

工作流：
1. compute_path_extreme_forward_rr() → 全样本 forward_rr (raw + clipped)
2. 训练浅树模型（max_depth=3-5，单树或少量树）
3. extract_negative_leaves() → 筛选 effect_size < -0.5 sigma 的叶节点
4. validate_rule_stability() → 时间覆盖 + 规则条件稳定性
5. classify_failures() → 分类为 structural / regime-conditional / noise
6. imodels (scripts/export_tree_rules_imodels.py) → 规则文本导出
7. 人工标注语义标签 → archetype 死亡清单 → gate.yaml
"""

from __future__ import annotations

from typing import Literal, Optional
from datetime import datetime
from dataclasses import dataclass

import numpy as np
import pandas as pd


EPS = 1e-8  # 防止除零


# ============================================================
# 语义标签常量（仅限四类原子标签，不可自定义）
# ============================================================

SEMANTIC_LABELS = [
    "volatility_spike",  # 波动率尖峰
    "range_compression",  # 区间压缩后的假突破
    "trend_exhaustion",  # 趋势衰竭
    "news_like_bar",  # 新闻式 K 线
]


# ============================================================
# 失败分类（Class A/B/C）
# ============================================================


@dataclass
class FailureClassification:
    """
    负规则失败分类

    Class A: structural_failure
        - 多月覆盖
        - effect_size_global << 0
        - effect_size_leaf << 0
        → 直接写入 gate.yaml (deny)

    Class B: regime_conditional_failure
        - time stable
        - global bad
        - leaf std 大（不稳定但均值差）
        → 作为 regime guardrail

    Class C: noise
        - 时间覆盖不足或 effect_size 不显著
        → 丢弃
    """

    class_name: str  # "structural_failure" | "regime_conditional_failure" | "noise"
    confidence: str  # "high" | "medium" | "low"
    action: str  # "deny_gate" | "regime_guardrail" | "discard"


def compute_path_extreme_forward_rr(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    clip_range: tuple = (-5, 5),
) -> tuple[pd.Series, pd.Series]:
    """
    计算 path-extreme forward RR（审计用）。

    公式：forward_rr = (mfe - mae) / atr
    - mfe: Maximum Favorable Excursion（最大有利偏移）
    - mae: Maximum Adverse Excursion（最大不利偏移）

    ⚠️ 关键设计决策：返回 raw 和 clipped 两个版本
    - raw: 用于 leaf 内 mean/effect_size，保留极端失败
    - clipped: 仅用于 global baseline/std 估计，防止污染

    ⚠️ Long/Short 必须分开审计，不要混！

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向，"long" 或 "short"
        horizon: 持仓窗口（bars）
        clip_range: clipped 版本的裁剪范围

    Returns:
        tuple[pd.Series, pd.Series]: (rr_raw, rr_clipped)
        - rr_raw: 原始 RR，用于 leaf 内统计
        - rr_clipped: 裁剪后 RR，仅用于 global baseline

    Raises:
        KeyError: 如果缺少必需的列（包括 ATR）
    """
    required_cols = [price_col, high_col, low_col, atr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    forward_rr = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= EPS:
            continue

        future_high = np.nanmax(high[i + 1 : i + horizon + 1])
        future_low = np.nanmin(low[i + 1 : i + horizon + 1])

        if direction == "long":
            mfe = future_high - entry_price
            mae = entry_price - future_low
        else:
            mfe = entry_price - future_low
            mae = future_high - entry_price

        forward_rr[i] = (mfe - mae) / max(current_atr, EPS)

    rr_raw = pd.Series(forward_rr, index=df.index, name="forward_rr_raw")

    # ⚠️ clip 只用于 baseline 估计，不用于 leaf 内统计
    # 这样可以保留"极端失败"的信号，不会"抹平失败形态"
    rr_clipped = pd.Series(
        np.clip(forward_rr, clip_range[0], clip_range[1]),
        index=df.index,
        name="forward_rr_clipped",
    )

    return rr_raw, rr_clipped


def compute_global_baseline(
    rr_clipped: pd.Series, method: str = "trimmed_mean"
) -> tuple[float, float]:
    """
    计算全局基线和标准差（用于 effect size）。

    ⚠️ 只用 clipped 版本，防止极端 regime 污染 baseline
    ⚠️ 不要直接用 mean()，会被极端 regime 污染

    Args:
        rr_clipped: clipped 版本的 RR 序列
        method: "trimmed_mean" | "median" | "clipped_mean"

    Returns:
        tuple[float, float]: (baseline, std)
    """
    valid = rr_clipped.dropna()
    if len(valid) == 0:
        return np.nan, np.nan

    if method == "median":
        baseline = valid.median()
        # MAD-based robust std
        std = (valid - baseline).abs().median() * 1.4826
    elif method == "clipped_mean":
        baseline = valid.mean()
        std = valid.std()
    else:  # trimmed_mean
        q10, q90 = valid.quantile([0.1, 0.9])
        trimmed = valid[(valid >= q10) & (valid <= q90)]
        if len(trimmed) > 0:
            baseline = trimmed.mean()
            std = trimmed.std()
        else:
            baseline = valid.median()
            std = valid.std()

    return baseline, max(std, EPS)


def get_leaf_signature(model, X: pd.DataFrame, n_trees: int = 3) -> np.ndarray:
    """
    获取 composite leaf signature（前 n 棵树的叶节点组合）。

    ⚠️ Boosting 的"坏区域"往往不是一棵树，而是多棵树组合形成的稳定子空间
    这是 archetype 审计的必需设计，不是优化

    Args:
        model: 树模型
        X: 特征数据
        n_trees: 使用前 n 棵树

    Returns:
        np.ndarray: shape (n_samples,) 的 signature 数组
    """
    leaf_ids = model.apply(X)
    if hasattr(leaf_ids, "values"):
        leaf_ids = leaf_ids.values

    if leaf_ids.ndim == 1:
        return leaf_ids

    n_trees = min(n_trees, leaf_ids.shape[1])

    signatures = []
    for i in range(len(X)):
        sig = tuple(leaf_ids[i, :n_trees])
        signatures.append(sig)

    unique_sigs = list(set(signatures))
    sig_to_id = {sig: idx for idx, sig in enumerate(unique_sigs)}

    return np.array([sig_to_id[sig] for sig in signatures])


def compute_time_coverage(
    mask: np.ndarray,
    timestamps: pd.Series,
    min_months: int = 3,
) -> dict:
    """
    计算时间覆盖（而非样本覆盖）。

    ⚠️ 失败不是频率问题，而是"跨世界是否成立"
    高频 regime 会"假装很稳定"，应该看覆盖了多少不同月份
    这是 PM 级别设计，防止 microstructure 假失败

    Args:
        mask: 样本掩码
        timestamps: 时间戳列
        min_months: 最少覆盖月份

    Returns:
        dict: {n_months, month_list, is_stable}
    """
    if not isinstance(timestamps.iloc[0], (datetime, pd.Timestamp)):
        try:
            timestamps = pd.to_datetime(timestamps)
        except Exception:
            return {"n_months": 0, "month_list": [], "is_stable": False}

    selected_times = timestamps[mask]
    if len(selected_times) == 0:
        return {"n_months": 0, "month_list": [], "is_stable": False}

    months = selected_times.dt.to_period("M").unique()
    month_list = [str(m) for m in months]

    return {
        "n_months": len(months),
        "month_list": month_list,
        "is_stable": len(months) >= min_months,
    }


def compute_effect_sizes(
    leaf_rr_raw: pd.Series,
    global_baseline: float,
    global_std: float,
) -> dict:
    """
    计算双重 effect size（global + leaf）。

    ⚠️ 关键设计：不能只用 global std
    global_std = 所有 regime 的混合波动，会稀释稳定失败的 leaf

    双重检验：
    - effect_size_global: delta / global_std （相对于整体市场）
    - effect_size_leaf: delta / leaf_std （leaf 内部的稳定性）

    判断标准：
    - veto_hard: 两者都很差（稳定且显著的失败）
    - veto_soft: 只 global 很差（可能是 regime-conditional）

    Args:
        leaf_rr_raw: leaf 内的 raw RR 序列
        global_baseline: 全局基线
        global_std: 全局标准差

    Returns:
        dict: {mean_rr, delta_rr, effect_size_global, effect_size_leaf, leaf_std}
    """
    mean_rr = leaf_rr_raw.mean()
    delta_rr = mean_rr - global_baseline
    leaf_std = leaf_rr_raw.std()

    effect_size_global = delta_rr / global_std

    # ⚠️ leaf_std 很小说明 leaf 内部很稳定
    # 如果 leaf_std 接近 0，effect_size_leaf 会很极端
    # 这正是我们想要的：稳定的失败应该被高权重
    effect_size_leaf = (
        delta_rr / max(leaf_std, EPS) if leaf_std > EPS else delta_rr / EPS
    )

    return {
        "mean_rr": float(mean_rr),
        "delta_rr": float(delta_rr),
        "effect_size_global": float(effect_size_global),
        "effect_size_leaf": float(effect_size_leaf),
        "leaf_std": float(leaf_std),
    }


def classify_failure(
    effect_size_global: float,
    effect_size_leaf: float,
    time_coverage_months: int,
    time_stable: bool,
    global_threshold: float = -0.5,
    leaf_threshold: float = -1.0,
    min_months_for_structural: int = 6,
) -> FailureClassification:
    """
    将负规则分类为 Class A/B/C。

    ⚠️ 这是将"统计发现"转化为"可执行决策"的关键步骤

    Class A: structural_failure (结构性失败)
        条件：
        - 时间覆盖 >= 6 个月
        - effect_size_global < -0.5 sigma
        - effect_size_leaf < -1.0 sigma（leaf 内部也很稳定差）
        动作：直接写入 gate.yaml (deny)

    Class B: regime_conditional_failure (regime 条件性失败)
        条件：
        - 时间稳定
        - effect_size_global < -0.5 sigma
        - 但 leaf_std 大（不稳定但均值差）
        动作：作为 regime guardrail

    Class C: noise (噪声)
        条件：以上都不满足
        动作：丢弃

    Args:
        effect_size_global: 相对于全局的 effect size
        effect_size_leaf: 相对于 leaf 内部的 effect size
        time_coverage_months: 覆盖月份数
        time_stable: 时间切片是否稳定

    Returns:
        FailureClassification: 分类结果
    """
    # Class A: 结构性失败 - 最高置信度
    if (
        time_coverage_months >= min_months_for_structural
        and effect_size_global < global_threshold
        and effect_size_leaf < leaf_threshold
    ):
        return FailureClassification(
            class_name="structural_failure", confidence="high", action="deny_gate"
        )

    # Class B: regime 条件性失败
    if time_stable and effect_size_global < global_threshold:
        return FailureClassification(
            class_name="regime_conditional_failure",
            confidence="medium",
            action="regime_guardrail",
        )

    # Class C: 噪声
    return FailureClassification(class_name="noise", confidence="low", action="discard")


def extract_negative_leaves(
    model,
    X: pd.DataFrame,
    rr_raw: pd.Series,
    rr_clipped: pd.Series,
    effect_size_threshold: float = -0.5,
    min_time_coverage_months: int = 3,
    min_sample_coverage: float = 0.01,
    baseline_method: str = "trimmed_mean",
    time_col: str = "timestamp",
    n_trees_for_signature: int = 3,
) -> list[dict]:
    """
    从树模型中提取负规则叶节点。

    ⚠️ 关键：使用 rr_raw 计算 leaf 统计，使用 rr_clipped 计算 baseline
    这样可以保留极端失败的信号，同时防止 baseline 被污染

    筛选条件：
    1. mean_rr < 0（叶节点平均收益为负）
    2. effect_size_global < threshold（比正常波动差多少 sigma）
    3. 时间覆盖 >= min_months（不是样本覆盖）

    Args:
        model: 训练好的树模型
        X: 特征数据
        rr_raw: 原始 RR（用于 leaf 内统计）
        rr_clipped: 裁剪 RR（用于 global baseline）

    Returns:
        list[dict]: 负规则列表，包含分类信息
    """
    leaf_signatures = get_leaf_signature(model, X, n_trees=n_trees_for_signature)

    unique_leaves = np.unique(leaf_signatures)
    n_total = len(X)

    # ⚠️ 用 clipped 版本计算 global baseline，防止极端 regime 污染
    global_baseline, global_std = compute_global_baseline(
        rr_clipped, method=baseline_method
    )

    timestamps = X[time_col] if time_col in X.columns else None

    negative_leaves = []

    for leaf_id in unique_leaves:
        leaf_mask = leaf_signatures == leaf_id
        n_leaf = leaf_mask.sum()
        sample_coverage = n_leaf / n_total

        if sample_coverage < min_sample_coverage:
            continue

        # ⚠️ 用 raw 版本计算 leaf 内统计，保留极端失败
        leaf_rr_raw = rr_raw[leaf_mask]

        effect_sizes = compute_effect_sizes(leaf_rr_raw, global_baseline, global_std)

        # 时间覆盖检查
        time_coverage = {"n_months": 0, "is_stable": True}
        if timestamps is not None:
            time_coverage = compute_time_coverage(
                leaf_mask, timestamps, min_months=min_time_coverage_months
            )

        # 筛选条件：mean_rr < 0 且 effect_size_global 显著
        if (
            effect_sizes["mean_rr"] < 0
            and effect_sizes["effect_size_global"] < effect_size_threshold
            and time_coverage["is_stable"]
        ):

            negative_leaves.append(
                {
                    "leaf_id": int(leaf_id),
                    **effect_sizes,
                    "global_baseline": float(global_baseline),
                    "global_std": float(global_std),
                    "sample_coverage": float(sample_coverage),
                    "n_samples": int(n_leaf),
                    "time_coverage_months": time_coverage["n_months"],
                    "semantic_label": None,  # 预留：人工标注语义（仅限四类原子标签）
                }
            )

    # 按 effect_size_global 排序（最差的在前）
    negative_leaves.sort(key=lambda x: x["effect_size_global"])

    return negative_leaves


def validate_rule_stability(
    model,
    X: pd.DataFrame,
    rr_raw: pd.Series,
    rr_clipped: pd.Series,
    leaf_id: int,
    time_col: str = "timestamp",
    n_time_splits: int = 5,
    threshold_perturbation: float = 0.05,
    n_perturbations: int = 10,
    effect_size_threshold: float = -0.5,
    baseline_method: str = "trimmed_mean",
    n_trees_for_signature: int = 3,
) -> dict:
    """
    验证规则稳定性。

    两种检验：
    1. 时间切片稳定性：在不同时间段是否一致为负
    2. 阈值扰动稳定性：对特征值加噪声后重新判断

    ⚠️ 重要限制：perturbation 稳定性只能作为 soft evidence

    原因：
    - tree leaf ≠ rule
    - boosting leaf ≠ decision boundary
    - perturb 后落回同一 leaf，不代表语义条件不变

    因此：
    - perturb_stable 只能升级 veto_soft → veto_hard
    - 真正的 veto_hard 必须来自 imodels rule-level 再筛选

    veto 分级：
    - veto_hard: 时间+扰动双通过（但扰动只是 soft evidence）
    - veto_soft: 仅时间通过
    - discard: 都不通过
    """
    leaf_signatures = get_leaf_signature(model, X, n_trees=n_trees_for_signature)
    leaf_mask = leaf_signatures == leaf_id

    global_baseline, global_std = compute_global_baseline(
        rr_clipped, method=baseline_method
    )

    # ========== 1. 时间切片稳定性 ==========
    time_stable = False
    if time_col in X.columns:
        time_values = X[time_col]
        try:
            time_splits = pd.qcut(
                time_values, n_time_splits, labels=False, duplicates="drop"
            )
            actual_splits = time_splits.nunique()

            time_stable_count = 0
            for split_id in range(actual_splits):
                split_mask = (time_splits == split_id) & leaf_mask
                if split_mask.sum() < 10:
                    continue

                # ⚠️ 用 raw 版本计算 split 内统计
                split_mean = rr_raw[split_mask].mean()
                split_effect = (split_mean - global_baseline) / global_std
                if split_effect < effect_size_threshold:
                    time_stable_count += 1

            time_stable = time_stable_count >= actual_splits * 0.6
        except Exception:
            time_stable = False

    # ========== 2. 阈值扰动稳定性 ==========
    # ⚠️ 注意：这只是 soft evidence，不能单独作为 veto_hard 依据
    # 原因：leaf id 邻域稳定性 ≠ 语义条件稳定性
    # 真正的验证需要在 imodels 导出规则后，用规则条件重新筛选
    perturb_stable_count = 0

    for _ in range(n_perturbations):
        X_perturbed = X.copy()
        for col in X_perturbed.select_dtypes(include=[np.number]).columns:
            if col == time_col:
                continue
            noise = np.random.normal(0, threshold_perturbation, len(X_perturbed))
            X_perturbed[col] = X_perturbed[col] * (1 + noise)

        perturbed_signatures = get_leaf_signature(
            model, X_perturbed, n_trees=n_trees_for_signature
        )

        perturbed_mask = perturbed_signatures == leaf_id
        combined_mask = leaf_mask | perturbed_mask

        if combined_mask.sum() >= 10:
            perturbed_mean = rr_raw[combined_mask].mean()
            perturbed_effect = (perturbed_mean - global_baseline) / global_std
            if perturbed_effect < effect_size_threshold:
                perturb_stable_count += 1

    perturb_stable = perturb_stable_count >= n_perturbations * 0.8

    # ========== 3. 确定 veto level ==========
    # ⚠️ perturb_stable 只能 *升级* veto_soft → veto_hard
    # 不能单独作为 veto 依据
    if time_stable and perturb_stable:
        veto_level = "veto_hard"
    elif time_stable:
        veto_level = "veto_soft"
    else:
        veto_level = "discard"

    return {
        "leaf_id": leaf_id,
        "time_stable": time_stable,
        "perturb_stable": perturb_stable,
        "veto_level": veto_level,
    }


def audit_archetype(
    model,
    X: pd.DataFrame,
    rr_raw: pd.Series,
    rr_clipped: pd.Series,
    time_col: str = "timestamp",
    effect_size_threshold: float = -0.5,
    min_time_coverage_months: int = 3,
    n_trees_for_signature: int = 3,
) -> list[dict]:
    """
    完整的 archetype 审计流程。

    这不是监督学习，而是「反例空间枚举」：
    - 不需要 train/test split（我们在问"历史上被市场反驳过吗"）
    - 树模型不调参、不 ensemble（它只是高维条件空间的切片器）
    - 产出是"死亡证据"，不是 alpha

    工作流：
    1. 提取负规则叶节点（基于双重 effect size + 时间覆盖）
    2. 对每个负规则做稳定性验证
    3. 分类为 structural / regime-conditional / noise
    4. 返回带 veto_level 和分类的负规则列表

    下一步：
    - 用 imodels 导出规则文本
    - 人工标注 semantic_label（仅限四类：volatility_spike/range_compression/trend_exhaustion/news_like_bar）
    - Class A → gate.yaml (deny)
    - Class B → regime guardrail

    Args:
        rr_raw: 原始 RR（用于 leaf 内统计）
        rr_clipped: 裁剪 RR（用于 global baseline）

    Returns:
        list[dict]: 带稳定性标注和分类的负规则（archetype 死亡清单）
    """
    negative_leaves = extract_negative_leaves(
        model,
        X,
        rr_raw,
        rr_clipped,
        effect_size_threshold=effect_size_threshold,
        min_time_coverage_months=min_time_coverage_months,
        time_col=time_col,
        n_trees_for_signature=n_trees_for_signature,
    )

    results = []
    for leaf_info in negative_leaves:
        stability = validate_rule_stability(
            model,
            X,
            rr_raw,
            rr_clipped,
            leaf_id=leaf_info["leaf_id"],
            time_col=time_col,
            effect_size_threshold=effect_size_threshold,
            n_trees_for_signature=n_trees_for_signature,
        )

        # 分类失败类型
        classification = classify_failure(
            effect_size_global=leaf_info["effect_size_global"],
            effect_size_leaf=leaf_info["effect_size_leaf"],
            time_coverage_months=leaf_info["time_coverage_months"],
            time_stable=stability["time_stable"],
        )

        results.append(
            {
                **leaf_info,
                **stability,
                "failure_class": classification.class_name,
                "failure_confidence": classification.confidence,
                "failure_action": classification.action,
            }
        )

    return results


# ============================================================
# Training Pipeline Wrapper
# ============================================================


def compute_generic_outcome_audit_label(
    df: pd.DataFrame,
    archetype: str = "generic",
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    return_features: bool = False,
    **kwargs,
) -> pd.Series:
    """
    训练流水线适配函数：计算 outcome audit 标签。

    为训练流水线提供统一接口，返回 forward_rr Series。

    Args:
        df: 输入数据，必须包含 OHLC 和 ATR
        archetype: archetype 名称（当前未使用，保留用于未来扩展）
        direction: 交易方向 "long" 或 "short"
        horizon: 持仓窗口（bars）
        return_features: 是否返回额外特征（当前未实现）
        **kwargs: 其他参数（向后兼容）

    Returns:
        pd.Series: forward_rr 标签序列

    Raises:
        KeyError: 如果缺少必需列（OHLC + ATR）
    """
    # 处理空 DataFrame（例如 test 集为空）
    if df.empty:
        return pd.Series([], dtype=float, name="forward_rr")

    # 计算 forward_rr
    rr_raw, rr_clipped = compute_path_extreme_forward_rr(
        df=df,
        direction=direction,
        horizon=horizon,
        price_col=kwargs.get("price_col", "close"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        atr_col=kwargs.get("atr_col", "atr"),
        clip_range=kwargs.get("clip_range", (-5, 5)),
    )

    # 返回 raw 版本（保留极端值用于树模型学习）
    return rr_raw
