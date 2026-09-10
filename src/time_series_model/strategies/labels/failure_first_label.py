"""
Failure-first Binary Label 模块

🟥 Failure-first 的核心思想：
- 不是问"能不能赚钱"，而是问"在哪些结构条件下会系统性失败"
- 树学习的目标是找到 failure probability 显著升高的区域
- 产出是"禁区地图"，不是交易信号

⚠️ 关键设计：
- failure 是 boolean，不是连续值
- failure 定义是"不可接受的失败"，不是简单的"亏钱"
- 包含 MAE（最大不利偏移）、recovery_time 等结构性判断

参考文档：docs/architecture/OUTCOME_BASED_TREE_LABELING.md
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Literal, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yaml

from src.time_series_model.live.direction_rule_ops import (
    dual_position_agree_deadband_series,
    is_direction_rule_enabled,
    parse_dual_rule,
    parse_single_position_band_rule,
    single_position_band_series,
)


EPS = 1e-8


@dataclass
class FailureDefinition:
    """
    失败定义配置。

    ⚠️ 这是整个 Failure-first 系统的"分水岭"。
    失败不是"亏钱"，而是"不可接受的失败"。

    Attributes:
        rr_threshold: forward_rr 阈值（低于此值视为失败）
        mae_mult: MAE 乘数（相对于 expected_stop）
        mfe_mult: MFE 乘数（相对于 expected_target）
        recovery_time_mult: 恢复时间乘数（相对于 expected_holding）
        require_all: 是否要求所有条件同时满足（False = 任一满足即为失败）
    """

    rr_threshold: float = -0.8  # forward_rr < -0.8R 视为失败
    mae_mult: float = 1.2  # MAE > 1.2 * expected_stop
    mfe_mult: float = 0.3  # MFE < 0.3 * expected_target
    recovery_time_mult: float = 2.0  # recovery_time > 2 * expected
    require_all: bool = False  # 任一条件满足即为失败


def compute_failure_first_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    failure_def: Optional[FailureDefinition] = None,
    expected_stop_atr: float = 1.0,  # 预期止损 = 1 ATR
    expected_target_atr: float = 2.0,  # 预期目标 = 2 ATR
    **kwargs,
) -> pd.Series:
    """
    计算 Failure-first 二值标签。

    🟥 核心语义：
    - 1 = 不可接受的失败（unacceptable failure）
    - 0 = 可接受或正常

    ⚠️ 失败定义（默认，可通过 failure_def 自定义）：
    failure = (
        forward_rr < -0.8R
        OR (
            mae > 1.2 * expected_stop
            AND mfe < 0.3 * expected_target
        )
    )

    语义解释：
    - forward_rr < -0.8R: 路径极端不利，亏损严重
    - mae > 1.2 * expected_stop: 被打穿止损 1.2 倍
    - mfe < 0.3 * expected_target: 几乎没有达到目标（没给你赚钱的机会）

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向，"long" 或 "short"
        horizon: 持仓窗口（bars）
        failure_def: 自定义失败定义
        expected_stop_atr: 预期止损（ATR 倍数）
        expected_target_atr: 预期目标（ATR 倍数）

    Returns:
        pd.Series: 二值失败标签（1=失败，0=正常）
    """
    if failure_def is None:
        failure_def = FailureDefinition()

    required_cols = [price_col, high_col, low_col, atr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    # 输出数组
    failure_label = np.full(n, np.nan)

    # 用于诊断的中间变量
    forward_rr = np.full(n, np.nan)
    mae_atr = np.full(n, np.nan)
    mfe_atr = np.full(n, np.nan)

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        if np.isnan(current_atr) or current_atr <= EPS:
            continue

        # 计算持仓窗口内的 MFE 和 MAE
        future_high = np.nanmax(high[i + 1 : i + horizon + 1])
        future_low = np.nanmin(low[i + 1 : i + horizon + 1])

        if direction == "long":
            mfe = future_high - entry_price  # 最大有利偏移
            mae = entry_price - future_low  # 最大不利偏移
        else:
            mfe = entry_price - future_low
            mae = future_high - entry_price

        # 归一化为 ATR 倍数
        mfe_norm = mfe / max(current_atr, EPS)
        mae_norm = mae / max(current_atr, EPS)
        rr = (mfe - mae) / max(current_atr, EPS)

        forward_rr[i] = rr
        mfe_atr[i] = mfe_norm
        mae_atr[i] = mae_norm

        # ========== 失败判断 ==========
        # 条件 1: forward_rr 极端负
        cond_rr_bad = rr < failure_def.rr_threshold

        # 条件 2: MAE 过大 且 MFE 过小
        cond_mae_bad = mae_norm > failure_def.mae_mult * expected_stop_atr
        cond_mfe_bad = mfe_norm < failure_def.mfe_mult * expected_target_atr
        cond_structural_bad = cond_mae_bad and cond_mfe_bad

        # 综合判断
        if failure_def.require_all:
            # 严格模式：所有条件都满足才算失败
            is_failure = cond_rr_bad and cond_structural_bad
        else:
            # 宽松模式：任一条件满足即为失败（推荐）
            is_failure = cond_rr_bad or cond_structural_bad

        failure_label[i] = 1.0 if is_failure else 0.0

    return pd.Series(failure_label, index=df.index, name="failure_label")


def compute_failure_components(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    expected_stop_atr: float = 1.0,
    expected_target_atr: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """
    计算失败标签的各个组成部分（用于诊断和理解）。

    返回的 DataFrame 包含：
    - forward_rr: 路径极端收益风险比
    - mfe_atr: 最大有利偏移（ATR 倍数）
    - mae_atr: 最大不利偏移（ATR 倍数）
    - is_rr_failure: forward_rr < threshold
    - is_structural_failure: MAE 大 且 MFE 小
    - failure_label: 最终失败标签

    用于：
    - 理解失败的具体原因
    - 调整失败定义参数
    - 诊断 failure_rate 分布
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

    results = {
        "forward_rr": np.full(n, np.nan),
        "mfe_atr": np.full(n, np.nan),
        "mae_atr": np.full(n, np.nan),
        "is_rr_failure": np.full(n, np.nan),
        "is_structural_failure": np.full(n, np.nan),
        "failure_label": np.full(n, np.nan),
    }

    failure_def = FailureDefinition()

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

        mfe_norm = mfe / max(current_atr, EPS)
        mae_norm = mae / max(current_atr, EPS)
        rr = (mfe - mae) / max(current_atr, EPS)

        results["forward_rr"][i] = rr
        results["mfe_atr"][i] = mfe_norm
        results["mae_atr"][i] = mae_norm

        # 失败条件
        cond_rr_bad = rr < failure_def.rr_threshold
        cond_mae_bad = mae_norm > failure_def.mae_mult * expected_stop_atr
        cond_mfe_bad = mfe_norm < failure_def.mfe_mult * expected_target_atr
        cond_structural_bad = cond_mae_bad and cond_mfe_bad

        results["is_rr_failure"][i] = 1.0 if cond_rr_bad else 0.0
        results["is_structural_failure"][i] = 1.0 if cond_structural_bad else 0.0
        results["failure_label"][i] = (
            1.0 if (cond_rr_bad or cond_structural_bad) else 0.0
        )

    return pd.DataFrame(results, index=df.index)


def get_failure_stats(
    df: pd.DataFrame,
    failure_col: str = "failure_label",
    feature_cols: Optional[list] = None,
    n_quantiles: int = 5,
) -> dict:
    """
    计算失败统计信息。

    用于理解：
    - 全局 failure_rate（baseline）
    - 各特征分位数的 failure_rate
    - 哪些特征区域 failure_rate 显著升高

    Returns:
        dict: {
            "baseline_failure_rate": float,
            "total_samples": int,
            "failure_samples": int,
            "feature_failure_rates": {feature: {quantile: failure_rate, lift}}
        }
    """
    valid_mask = df[failure_col].notna()
    df_valid = df[valid_mask]

    total = len(df_valid)
    failures = (df_valid[failure_col] == 1).sum()
    baseline = failures / total if total > 0 else 0

    stats = {
        "baseline_failure_rate": float(baseline),
        "total_samples": int(total),
        "failure_samples": int(failures),
        "feature_failure_rates": {},
    }

    if feature_cols is None:
        return stats

    for col in feature_cols:
        if col not in df_valid.columns:
            continue

        try:
            quantiles = pd.qcut(
                df_valid[col], n_quantiles, labels=False, duplicates="drop"
            )
            col_stats = {}

            for q in range(quantiles.nunique()):
                q_mask = quantiles == q
                q_total = q_mask.sum()
                q_failures = (df_valid.loc[q_mask, failure_col] == 1).sum()
                q_rate = q_failures / q_total if q_total > 0 else 0
                lift = q_rate / baseline if baseline > 0 else 0

                col_stats[f"q{q}"] = {
                    "failure_rate": float(q_rate),
                    "lift": float(lift),
                    "samples": int(q_total),
                }

            stats["feature_failure_rates"][col] = col_stats
        except Exception:
            continue

    return stats


# ============================================================
# Failure 子标签（用于分析，不进模型）
# ============================================================


def compute_failure_subtypes(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    price_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    atr_col: str = "atr",
    expected_stop_atr: float = 1.0,
    expected_target_atr: float = 2.0,
    **kwargs,
) -> pd.DataFrame:
    """
    计算 Failure 子标签（只用于分析，不进模型）。

    🟥 两种 failure 子类型：

    1. failure_rr_extreme: forward_rr < -0.8R
       - 路径极端不利，亏损严重
       - 通常意味着"踩了大坑"

    2. failure_no_opportunity: MAE > 1.2*stop AND MFE < 0.3*target
       - 被打穿止损，且没给赚钱机会
       - 意味着"入场后立刻反向，没有任何转戴空间"

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向
        horizon: 持仓窗口（bars）

    Returns:
        pd.DataFrame: 包含以下列
        - failure_rr_extreme: 1=极端RR失败, 0=否
        - failure_no_opportunity: 1=无机会失败, 0=否
        - failure_any: 1=任一失败, 0=否
        - forward_rr: 路径RR
        - mfe_atr: 最大有利偏移
        - mae_atr: 最大不利偏移
    """
    # Handle empty DataFrame - return empty result with correct structure
    if len(df) == 0:
        return pd.DataFrame(
            {
                "failure_rr_extreme": pd.Series(dtype=float),
                "failure_no_opportunity": pd.Series(dtype=float),
                "failure_any": pd.Series(dtype=float),
                "forward_rr": pd.Series(dtype=float),
                "mfe_atr": pd.Series(dtype=float),
                "mae_atr": pd.Series(dtype=float),
            },
            index=df.index,
        )

    required_cols = [price_col, high_col, low_col, atr_col]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise KeyError(f"缺少必需列: {missing}")

    # 🔍 DEBUG: 检查数据完整性（多币种场景）
    if "_symbol" in df.columns:
        symbols = df["_symbol"].unique()
        if len(symbols) > 1:
            # 多币种场景：不应该混在一起计算
            raise ValueError(
                f"compute_failure_subtypes 不支持多币种混合计算！"
                f"发现 {len(symbols)} 个币种: {symbols.tolist()}。"
                f"请按币种分别调用。"
            )

    # 重置索引，确保连续（避免多币种合并后索引不连续的问题）
    df = df.reset_index(drop=True)

    close = df[price_col].values
    high = df[high_col].values
    low = df[low_col].values
    atr = df[atr_col].values
    n = len(df)

    failure_def = FailureDefinition()

    results = {
        "failure_rr_extreme": np.full(n, np.nan),
        "failure_no_opportunity": np.full(n, np.nan),
        "failure_any": np.full(n, np.nan),
        "forward_rr": np.full(n, np.nan),
        "mfe_atr": np.full(n, np.nan),
        "mae_atr": np.full(n, np.nan),
    }

    for i in range(n - horizon):
        entry_price = close[i]
        current_atr = atr[i]

        # 处理 current_atr 可能是数组的情况
        if isinstance(current_atr, (np.ndarray, pd.Series)):
            if len(current_atr) > 0:
                current_atr = float(
                    current_atr.iloc[0]
                    if isinstance(current_atr, pd.Series)
                    else current_atr[0]
                )
            else:
                continue

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

        mfe_norm = mfe / max(current_atr, EPS)
        mae_norm = mae / max(current_atr, EPS)
        rr = (mfe - mae) / max(current_atr, EPS)

        results["forward_rr"][i] = rr
        results["mfe_atr"][i] = mfe_norm
        results["mae_atr"][i] = mae_norm

        # 失败子类型 1: RR 极端失败
        cond_rr_bad = rr < failure_def.rr_threshold
        results["failure_rr_extreme"][i] = 1.0 if cond_rr_bad else 0.0

        # 失败子类型 2: 无机会失败
        cond_mae_bad = mae_norm > failure_def.mae_mult * expected_stop_atr
        cond_mfe_bad = mfe_norm < failure_def.mfe_mult * expected_target_atr
        cond_no_opp = cond_mae_bad and cond_mfe_bad
        results["failure_no_opportunity"][i] = 1.0 if cond_no_opp else 0.0

        # 任一失败
        results["failure_any"][i] = 1.0 if (cond_rr_bad or cond_no_opp) else 0.0

    return pd.DataFrame(results, index=df.index)


# ============================================================
# 拆分 Failure 子标签训练函数
# ============================================================


# ---- Direction-aware helpers ----


def _load_direction_config_for_label(strategy: str) -> dict:
    """Load direction.yaml for a given strategy.

    Searches:
      1. config/strategies/{strategy}/archetypes/direction.yaml
      2. config/strategies/{strategy}/direction.yaml
    """
    project_root = Path(__file__).resolve().parents[4]
    candidates = [
        project_root
        / "config"
        / "strategies"
        / strategy
        / "archetypes"
        / "direction.yaml",
        project_root / "config" / "strategies" / strategy / "direction.yaml",
    ]
    for path in candidates:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    raise FileNotFoundError(
        f"direction.yaml not found for strategy '{strategy}'. "
        f"Searched: {[str(p) for p in candidates]}"
    )


def _compute_direction_from_rules(direction_cfg: dict, df: pd.DataFrame) -> np.ndarray:
    """Compute per-bar direction from direction.yaml rules.

    Returns:
        np.ndarray of +1.0 (long), -1.0 (short), or 0.0 (unknown).
    """
    rules = direction_cfg.get("direction_rules", [])
    n = len(df)
    direction = np.zeros(n, dtype=float)
    assigned = np.zeros(n, dtype=bool)

    for rule in rules:
        if not is_direction_rule_enabled(rule):
            continue
        dual = parse_dual_rule(rule)
        band = parse_single_position_band_rule(rule)
        if dual is not None:
            col_a, col_b, eps = dual
            if col_a not in df.columns or col_b not in df.columns:
                continue
            vals = dual_position_agree_deadband_series(df, col_a, col_b, eps).values
        elif band is not None:
            fcol, inner_a, outer_a = band
            if fcol not in df.columns:
                continue
            vals = single_position_band_series(df, fcol, inner_a, outer_a).values
        elif str(rule.get("method", "")).strip().lower() == "threshold_compare":
            feat_name = str(rule.get("feature", "")).strip()
            if feat_name not in df.columns:
                continue
            thr = rule.get("thresholds") or {}
            col_data = df[feat_name]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            s = pd.to_numeric(col_data, errors="coerce")
            vals = np.zeros(len(df), dtype=float)
            try:
                lb = thr.get("long_if_below")
                sa = thr.get("short_if_above")
                if lb is not None:
                    vals = np.where(s.values <= float(lb), 1.0, vals)
                if sa is not None:
                    vals = np.where(s.values >= float(sa), -1.0, vals)
            except (TypeError, ValueError):
                vals = np.zeros(len(df), dtype=float)
        else:
            feature = rule.get("feature", "")
            transform = rule.get("transform", "raw")
            if feature not in df.columns:
                continue

            col_data = df[feature]
            if isinstance(col_data, pd.DataFrame):
                col_data = col_data.iloc[:, 0]
            series = pd.to_numeric(col_data, errors="coerce").fillna(0.0).values

            if transform == "sign":
                vals = np.sign(series)
            elif transform == "negate_sign":
                vals = -np.sign(series)
            elif transform == "center_sign":
                vals = np.sign(series - 0.5)
            elif transform == "negate":
                vals = -series
            else:
                vals = series

        unassigned = ~assigned
        direction[unassigned] = vals[unassigned]
        newly = unassigned & (direction != 0)
        assigned = assigned | newly
        if assigned.all():
            break

    return direction


def _direction_aware_subtypes_single_symbol(
    df: pd.DataFrame,
    horizon: int,
    direction_cfg: dict,
    **kwargs,
) -> pd.DataFrame:
    """单币种方向感知 failure subtypes 计算。

    返回与 compute_failure_subtypes 相同 schema 的 DataFrame，
    但 forward_rr / mfe / mae / failure 标签均按 per-bar 方向翻转。

    数学基础:
      forward_rr_short = -forward_rr_long
      MFE_short = MAE_long,  MAE_short = MFE_long
      direction=-1 → forward_rr, MFE, MAE 全部翻转
      direction= 0 → 保守处理，假定 long
    """
    failure_def = FailureDefinition()
    expected_stop_atr = kwargs.get("expected_stop_atr", 1.0)
    expected_target_atr = kwargs.get("expected_target_atr", 2.0)

    # Step 1: 始终以 direction="long" 计算基础 subtypes
    subtypes = compute_failure_subtypes(
        df=df,
        direction="long",
        horizon=horizon,
        price_col=kwargs.get("price_col", "close"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        atr_col=kwargs.get("atr_col", "atr"),
        expected_stop_atr=expected_stop_atr,
        expected_target_atr=expected_target_atr,
    )

    # Step 2: 计算 per-bar direction
    df_reset = df.reset_index(drop=True)
    direction_arr = _compute_direction_from_rules(direction_cfg, df_reset)
    short_mask = direction_arr == -1

    # Step 3: 翻转 forward_rr
    forward_rr = subtypes["forward_rr"].values.copy()
    forward_rr[short_mask] = -forward_rr[short_mask]

    # Step 4: 交换 MFE/MAE (short 时 MFE_long 变 MAE_short，反之亦然)
    mfe_atr = subtypes["mfe_atr"].values.copy()
    mae_atr = subtypes["mae_atr"].values.copy()
    mfe_short_tmp = mae_atr[short_mask].copy()
    mae_short_tmp = mfe_atr[short_mask].copy()
    mfe_atr[short_mask] = mfe_short_tmp
    mae_atr[short_mask] = mae_short_tmp

    # Step 5: 重新计算 failure 标签
    valid = ~np.isnan(forward_rr)
    failure_rr = np.full(len(forward_rr), np.nan)
    failure_rr[valid] = (forward_rr[valid] < failure_def.rr_threshold).astype(float)

    valid_opp = ~np.isnan(mfe_atr) & ~np.isnan(mae_atr)
    failure_no_opp = np.full(len(forward_rr), np.nan)
    cond_mae = mae_atr > failure_def.mae_mult * expected_stop_atr
    cond_mfe = mfe_atr < failure_def.mfe_mult * expected_target_atr
    failure_no_opp[valid_opp] = (cond_mae[valid_opp] & cond_mfe[valid_opp]).astype(
        float
    )

    failure_any = np.full(len(forward_rr), np.nan)
    valid_any = valid & valid_opp
    failure_any[valid_any] = (
        (failure_rr[valid_any] == 1) | (failure_no_opp[valid_any] == 1)
    ).astype(float)

    # 打印方向统计
    n_total = len(direction_arr)
    n_long = int((direction_arr == 1).sum())
    n_short = int((direction_arr == -1).sum())
    n_zero = int((direction_arr == 0).sum())
    print(
        f"   \U0001f4d0 Direction-aware labels: "
        f"long={n_long}({n_long/max(n_total,1)*100:.1f}%), "
        f"short={n_short}({n_short/max(n_total,1)*100:.1f}%), "
        f"zero={n_zero}({n_zero/max(n_total,1)*100:.1f}%)"
    )
    if n_zero > 0:
        warnings.warn(
            f"Direction coverage not 100%: {n_zero} bars have direction=0 "
            f"(treated as long). Check direction.yaml feature availability.",
            stacklevel=2,
        )

    return pd.DataFrame(
        {
            "forward_rr": forward_rr,
            "mfe_atr": mfe_atr,
            "mae_atr": mae_atr,
            "failure_rr_extreme": failure_rr,
            "failure_no_opportunity": failure_no_opp,
            "failure_any": failure_any,
        },
        index=subtypes.index,
    )


def _direction_aware_subtypes(
    df: pd.DataFrame,
    horizon: int,
    strategy: str,
    **kwargs,
) -> pd.DataFrame:
    """方向感知 failure subtypes（多币种调度）。"""
    direction_cfg = _load_direction_config_for_label(strategy)

    if "_symbol" in df.columns and df["_symbol"].nunique() > 1:
        results = []
        for symbol in df["_symbol"].unique():
            sym_mask = df["_symbol"] == symbol
            sym_df = df[sym_mask].copy()
            sym_result = _direction_aware_subtypes_single_symbol(
                sym_df, horizon, direction_cfg, **kwargs
            )
            sym_result.index = df[sym_mask].index
            results.append(sym_result)
        return pd.concat(results, sort=False).sort_index()
    else:
        result = _direction_aware_subtypes_single_symbol(
            df, horizon, direction_cfg, **kwargs
        )
        result.index = df.index
        return result


def _compute_direction_aware_rr_extreme(
    df: pd.DataFrame,
    horizon: int = 50,
    invert: bool = True,
    strategy: str = "",
    **kwargs,
) -> pd.Series:
    """方向感知的 rr_extreme 标签计算。"""
    subtypes = _direction_aware_subtypes(df, horizon, strategy, **kwargs)
    failure_label = subtypes["failure_rr_extreme"]

    # 回传 forward_rr 到输入 df（与 sample_weight 同模式，供 --prepare-only 导出）
    if "forward_rr" in subtypes.columns:
        df["forward_rr"] = subtypes["forward_rr"].values

    if invert:
        success_label = 1.0 - failure_label
        success_label.name = "success_no_rr_extreme"
        return success_label
    failure_label.name = "failure_rr_extreme"
    return failure_label


def _compute_direction_aware_no_opportunity(
    df: pd.DataFrame,
    horizon: int = 50,
    invert: bool = True,
    strategy: str = "",
    **kwargs,
) -> pd.Series:
    """方向感知的 no_opportunity 标签计算。

    Short 时 MFE/MAE 互换:
      MAE_short = MFE_long,  MFE_short = MAE_long
      failure_no_opp = (MAE_actual > 1.2*stop) AND (MFE_actual < 0.3*target)
    """
    subtypes = _direction_aware_subtypes(df, horizon, strategy, **kwargs)
    failure_label = subtypes["failure_no_opportunity"]

    if invert:
        success_label = 1.0 - failure_label
        success_label.name = "success_has_opportunity"
        return success_label
    failure_label.name = "failure_no_opportunity"
    return failure_label


def _compute_direction_aware_return_tree(
    df: pd.DataFrame,
    horizon: int = 50,
    filter_good_only: bool = True,
    strategy: str = "",
    **kwargs,
) -> pd.Series:
    """方向感知的 return_tree 标签计算。

    forward_rr 按方向翻转，GOOD 样本筛选也使用方向感知的 failure_any。
    """
    subtypes = _direction_aware_subtypes(df, horizon, strategy, **kwargs)
    forward_rr = subtypes["forward_rr"].copy()
    forward_rr.name = "forward_rr"

    if filter_good_only:
        failure_any = subtypes["failure_any"]
        forward_rr = forward_rr.where(failure_any == 0)

    return forward_rr


# ---- End direction-aware helpers ----


def compute_bpc_failure_rr_extreme_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    invert: bool = True,
    direction_aware: bool = False,
    strategy: str = "",
    **kwargs,
) -> pd.Series:
    """
    只预测 "failure_rr_extreme"（踩大坑）的二值标签。

    🟥 核心语义：
    - failure_rr_extreme = forward_rr < -0.8R
    - 路径极端不利，亏损严重，通常意味着"踩了大坑"

    ❗ invert=True 时（默认）：
    - 输出: 1=好机会（不会踩坑），0=踩坑
    - 适配回测代码的 `preds >= threshold` 逻辑

    📐 direction_aware=True 时（方向感知模式）：
    - 先以 direction="long" 计算 forward_rr_long
    - 再从 direction.yaml 规则计算 per-bar 方向
    - 翻转: forward_rr = forward_rr_long × direction
    - 需要 strategy 参数指定策略名（加载 direction.yaml）

    Args:
        df: 价格数据
        direction: 基础交易方向 (direction_aware=False 时使用)
        horizon: 持仓窗口
        invert: 是否反转标签
        direction_aware: 是否启用方向感知模式
        strategy: 策略名（direction_aware=True 时必须提供）

    Returns:
        pd.Series: 二值标签
    """
    # 方向感知模式：per-bar direction 翻转 forward_rr
    if direction_aware:
        if not strategy:
            raise ValueError(
                "direction_aware=True requires 'strategy' parameter "
                "(e.g., strategy='me') to load direction.yaml"
            )
        try:
            return _compute_direction_aware_rr_extreme(
                df=df, horizon=horizon, invert=invert, strategy=strategy, **kwargs
            )
        except FileNotFoundError:
            warnings.warn(
                f"⚠️  direction.yaml not found for strategy '{strategy}'. "
                f"Falling back to direction='{direction}' (all-long). "
                f"This is expected during --prepare-only analysis phase. "
                f"For training (Step 5+), create direction.yaml first (Step 4).",
                stacklevel=2,
            )
            # Fall through to non-direction-aware computation below

    # 🔍 多币种支持：按 symbol 分别计算
    if "_symbol" in df.columns and df["_symbol"].nunique() > 1:
        results = []
        rr_parts = []
        for symbol in df["_symbol"].unique():
            sym_mask = df["_symbol"] == symbol
            sym_df = df[sym_mask].copy()
            sym_subtypes = compute_failure_subtypes(
                df=sym_df,
                direction=direction,
                horizon=horizon,
                price_col=kwargs.get("price_col", "close"),
                high_col=kwargs.get("high_col", "high"),
                low_col=kwargs.get("low_col", "low"),
                atr_col=kwargs.get("atr_col", "atr"),
                expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
                expected_target_atr=kwargs.get("expected_target_atr", 2.0),
            )
            sym_failure = sym_subtypes["failure_rr_extreme"].copy()
            sym_failure.index = df[sym_mask].index
            results.append(sym_failure)
            # 回传 forward_rr
            if "forward_rr" in sym_subtypes.columns:
                sym_rr = sym_subtypes["forward_rr"].copy()
                sym_rr.index = df[sym_mask].index
                rr_parts.append(sym_rr)
        failure_label = pd.concat(results, sort=False).sort_index()
        if rr_parts:
            df["forward_rr"] = pd.concat(rr_parts, sort=False).sort_index()
    else:
        # 单币种场景
        subtypes = compute_failure_subtypes(
            df=df,
            direction=direction,
            horizon=horizon,
            price_col=kwargs.get("price_col", "close"),
            high_col=kwargs.get("high_col", "high"),
            low_col=kwargs.get("low_col", "low"),
            atr_col=kwargs.get("atr_col", "atr"),
            expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
            expected_target_atr=kwargs.get("expected_target_atr", 2.0),
        )
        failure_label = subtypes["failure_rr_extreme"]
        # 修复索引：compute_failure_subtypes 内部 reset_index 会导致返回 RangeIndex
        failure_label.index = df.index
        # 回传 forward_rr
        if "forward_rr" in subtypes.columns:
            df["forward_rr"] = subtypes["forward_rr"].values

    if invert:
        success_label = 1.0 - failure_label
        success_label.name = "success_no_rr_extreme"
        return success_label

    failure_label.name = "failure_rr_extreme"
    return failure_label


def compute_bpc_failure_no_opportunity_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    invert: bool = True,
    direction_aware: bool = False,
    strategy: str = "",
    **kwargs,
) -> pd.Series:
    """
    只预测 "failure_no_opportunity"（入场即反）的二值标签。

    🟥 核心语义：
    - failure_no_opportunity = (MAE > 1.2*stop) AND (MFE < 0.3*target)
    - 被打穿止损，且没给赚钱机会
    - 意味着"入场后立刻反向，没有任何转载空间"

    📐 direction_aware=True 时：
    - Short 时 MFE/MAE 互换: MAE_short=MFE_long, MFE_short=MAE_long
    - 再用互换后的值判断 no_opportunity

    Args:
        df: 价格数据
        direction: 基础交易方向 (direction_aware=False 时使用)
        horizon: 持仓窗口
        invert: 是否反转标签
        direction_aware: 是否启用方向感知模式
        strategy: 策略名 (direction_aware=True 时必须提供)

    Returns:
        pd.Series: 二值标签
    """
    if direction_aware:
        if not strategy:
            raise ValueError("direction_aware=True requires 'strategy' parameter")
        try:
            return _compute_direction_aware_no_opportunity(
                df=df, horizon=horizon, invert=invert, strategy=strategy, **kwargs
            )
        except FileNotFoundError:
            warnings.warn(
                f"⚠️  direction.yaml not found for strategy '{strategy}'. "
                f"Falling back to direction='{direction}' (all-long). "
                f"For training, create direction.yaml first.",
                stacklevel=2,
            )
            # Fall through to non-direction-aware computation below
    # 🔍 多币种支持：按 symbol 分别计算
    if "_symbol" in df.columns and df["_symbol"].nunique() > 1:
        results = []
        for symbol in df["_symbol"].unique():
            sym_mask = df["_symbol"] == symbol
            sym_df = df[sym_mask].copy()
            sym_subtypes = compute_failure_subtypes(
                df=sym_df,
                direction=direction,
                horizon=horizon,
                price_col=kwargs.get("price_col", "close"),
                high_col=kwargs.get("high_col", "high"),
                low_col=kwargs.get("low_col", "low"),
                atr_col=kwargs.get("atr_col", "atr"),
                expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
                expected_target_atr=kwargs.get("expected_target_atr", 2.0),
            )
            sym_failure = sym_subtypes["failure_no_opportunity"].copy()
            sym_failure.index = df[sym_mask].index
            results.append(sym_failure)
        failure_label = pd.concat(results, sort=False).sort_index()
    else:
        # 单币种场景
        subtypes = compute_failure_subtypes(
            df=df,
            direction=direction,
            horizon=horizon,
            price_col=kwargs.get("price_col", "close"),
            high_col=kwargs.get("high_col", "high"),
            low_col=kwargs.get("low_col", "low"),
            atr_col=kwargs.get("atr_col", "atr"),
            expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
            expected_target_atr=kwargs.get("expected_target_atr", 2.0),
        )
        failure_label = subtypes["failure_no_opportunity"]
        # 修复索引：compute_failure_subtypes 内部 reset_index 会导致返回 RangeIndex
        failure_label.index = df.index

    if invert:
        success_label = 1.0 - failure_label
        success_label.name = "success_has_opportunity"
        return success_label

    failure_label.name = "failure_no_opportunity"
    return failure_label


# ============================================================
# 训练流水线适配函数
# ============================================================


def compute_bpc_failure_label(
    df: pd.DataFrame,
    archetype: str = "bpc",
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    invert: bool = True,  # 默认输出 success_label，方便回测
    **kwargs,
) -> pd.Series:
    """
    BPC 策略的 Failure-first 标签（训练流水线适配）。

    这是 compute_failure_first_label 的包装函数，
    为 BPC 策略提供合适的默认参数。

    ❗ invert=True 时（默认）：
    - 输出 success_label: 1=好机会，0=失败
    - 适配回测代码的 `preds >= threshold` 逻辑

    ❗ invert=False 时：
    - 输出 failure_label: 1=失败，0=正常
    - 适合用于分析“哪里会失败”

    Example labels.yaml:
    ```yaml
    name: bpc_failure_first
    target_column: success_label  # invert=True 时

    label_generator:
      module: src.time_series_model.strategies.labels.failure_first_label
      function: compute_bpc_failure_label
      params:
        direction: long
        horizon: 50
        invert: true  # 1=好机会，0=失败
    ```
    """
    failure_label = compute_failure_first_label(
        df=df,
        direction=direction,
        horizon=horizon,
        price_col=kwargs.get("price_col", "close"),
        high_col=kwargs.get("high_col", "high"),
        low_col=kwargs.get("low_col", "low"),
        atr_col=kwargs.get("atr_col", "atr"),
        expected_stop_atr=kwargs.get("expected_stop_atr", 1.0),
        expected_target_atr=kwargs.get("expected_target_atr", 2.0),
    )

    if invert:
        # 反转为 success_label: 1=好机会，0=失败
        success_label = 1.0 - failure_label
        success_label.name = "success_label"
        return success_label
    return failure_label


# ============================================================
# Return Tree 标签（用于 GOOD 样本空间）
# ============================================================


def compute_return_tree_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    filter_good_only: bool = True,
    **kwargs,
) -> pd.Series:
    """
    计算 Return Tree 标签：GOOD 样本的 forward_rr。

    🟢 用途：在已经排除 failure 的样本中，学习“如何让 RR 更大”

    Args:
        df: 价格数据，必须包含 OHLC 和 ATR
        direction: 交易方向
        horizon: 持仓窗口（bars）
        filter_good_only: 是否只返回 GOOD 样本（默认 True）

    Returns:
        pd.Series: forward_rr 值，失败样本为 NaN
    """
    # 🔍 多币种支持：按 _symbol 分组计算
    if "_symbol" in df.columns and df["_symbol"].nunique() > 1:
        results = []
        for symbol in df["_symbol"].unique():
            sym_mask = df["_symbol"] == symbol
            sym_df = df[sym_mask].copy()
            sym_labels = _compute_return_tree_single_symbol(
                sym_df,
                direction=direction,
                horizon=horizon,
                filter_good_only=filter_good_only,
                **kwargs,
            )
            # 重要：使用原始 DataFrame 的索引，而不是 sym_df 的索引
            # 这样可以确保合并后的 Series 索引与原始 df 对齐
            sym_labels.index = df[sym_mask].index
            results.append(sym_labels)

        # 合并结果，使用 sort=False 保持原始顺序
        return pd.concat(results, sort=False).sort_index()
    else:
        # 单币种场景
        result = _compute_return_tree_single_symbol(
            df,
            direction=direction,
            horizon=horizon,
            filter_good_only=filter_good_only,
            **kwargs,
        )
        # 修复索引：_compute_return_tree_single_symbol 调用 compute_failure_subtypes
        # 内部 reset_index 会导致返回 RangeIndex
        result.index = df.index
        return result


def _compute_return_tree_single_symbol(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    filter_good_only: bool = True,
    **kwargs,
) -> pd.Series:
    """
    单币种 Return Tree label 计算（内部函数）。
    """
    # 计算 failure 子标签
    failure_df = compute_failure_subtypes(
        df=df,
        direction=direction,
        horizon=horizon,
        **kwargs,
    )

    # 提取 forward_rr
    forward_rr = failure_df["forward_rr"].copy()
    forward_rr.name = "forward_rr"

    if filter_good_only:
        # GOOD 样本 = ~failure_any
        failure_any = failure_df["failure_any"]
        # 将 failure 样本的 forward_rr 设为 NaN（会被过滤掉）
        forward_rr = forward_rr.where(failure_any == 0)

    return forward_rr


def compute_bpc_return_tree_label(
    df: pd.DataFrame,
    direction: Literal["long", "short"] = "long",
    horizon: int = 50,
    filter_good_only: bool = True,
    direction_aware: bool = False,
    strategy: str = "",
    **kwargs,
) -> pd.Series:
    """
    BPC 策略的 Return Tree 标签（训练流水线适配）。

    🟢 Phase 2 核心任务：在 GOOD 样本中学习"如何让 RR 更大"

    📐 direction_aware=True 时：
    - forward_rr 按 per-bar 方向翻转
    - GOOD 样本筛选也使用方向感知的 failure_any

    输出：
    - forward_rr: 路径 RR 值
    - 失败样本自动过滤（failure_any = 1 的样本 forward_rr 为 NaN）

    GOOD 样本空间：
    - ~failure_rr_extreme: forward_rr >= -0.8R
    - ~failure_no_opportunity: 有机会的交易
    """
    if direction_aware:
        if not strategy:
            raise ValueError("direction_aware=True requires 'strategy' parameter")
        try:
            return _compute_direction_aware_return_tree(
                df=df,
                horizon=horizon,
                filter_good_only=filter_good_only,
                strategy=strategy,
                **kwargs,
            )
        except FileNotFoundError:
            warnings.warn(
                f"⚠️  direction.yaml not found for strategy '{strategy}'. "
                f"Falling back to direction='{direction}' (all-long). "
                f"For training, create direction.yaml first.",
                stacklevel=2,
            )
    return compute_return_tree_label(
        df=df,
        direction=direction,
        horizon=horizon,
        filter_good_only=filter_good_only,
        **kwargs,
    )
