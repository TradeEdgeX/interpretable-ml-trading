"""
Session & Microstructure Features — 时间结构 + 流动性微观特征

来源: token交易里execution往往比signal更重要.md
- Session Liquidity Map: Asia 堆流动性 → EU breakout → US 趋势
- Microstructure Trigger: sweep + reclaim 结构

特征列表:
  1. session_id             — Asia=0, EU=1, US=2 (categorical int)
  2. hour_sin / hour_cos    — 24h 周期编码 (连续, 对树模型友好)
  3. is_session_overlap     — EU/US 重叠时段 (高流动性)
  4. bars_since_local_high  — 距离最近局部高点的 bar 数 (归一化)
  5. bars_since_local_low   — 距离最近局部低点的 bar 数 (归一化)

Note:
  sweep/wick 相关特征已有:
  - wick_ratios_f (wick_upper_ratio, wick_lower_ratio)
  - wick_scene_semantic_scores_f (exhaustion/absorption)
  - liquidity_sweep_features (lsr_sweep_*, lsr_wick_rejection)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.features.registry import register_feature

logger = logging.getLogger(__name__)


# ── Session 时区定义 (UTC) ──
# Asia:  00:00-08:00 UTC  (08:00-16:00 Beijing)
# EU:    08:00-14:00 UTC  (09:00-15:00 London)
# US:    14:00-21:00 UTC  (09:00-16:00 New York)
# Late:  21:00-00:00 UTC  (off-hours, low liquidity)
SESSION_BOUNDARIES = {
    "asia": (0, 8),
    "eu": (8, 14),
    "us": (14, 21),
    "late": (21, 24),
}
# EU-US overlap: 14:00-16:00 UTC
OVERLAP_START = 14
OVERLAP_END = 16


def _extract_hour(index: pd.Index) -> pd.Series:
    """从 DatetimeIndex 提取 UTC 小时, 处理 RangeIndex 的情况."""
    if hasattr(index, "hour"):
        return pd.Series(index.hour, index=index, dtype=float)
    # 非 DatetimeIndex (例如 RangeIndex), 返回 NaN
    return pd.Series(np.nan, index=index, dtype=float)


# ═══════════════════════════════════════════════════════════════════════
# 1. Session Features (时段分类 + 周期编码)
# ═══════════════════════════════════════════════════════════════════════


@register_feature(
    "compute_session_features_from_series",
    category="session",
    description=(
        "Session liquidity map features: session_id (Asia/EU/US/Late), "
        "hour cyclic encoding (sin/cos), session overlap indicator"
    ),
    outputs=[
        "session_id",
        "hour_sin",
        "hour_cos",
        "is_session_overlap",
    ],
)
def compute_session_features_from_series(
    *,
    close: pd.Series,
) -> pd.DataFrame:
    """
    时段流动性特征.

    输入: close (只用于获取 index 的时间信息)
    输出:
      - session_id: 0=Asia, 1=EU, 2=US, 3=Late (categorical int)
      - hour_sin: sin(2π * hour/24)  — 周期编码
      - hour_cos: cos(2π * hour/24)  — 周期编码
      - is_session_overlap: EU-US 重叠时段 = 1.0, 否则 = 0.0

    Note:
      树模型可以直接 split session_id;
      hour_sin/cos 给线性组合一个连续的时间距离度量.
    """
    hour = _extract_hour(close.index)

    # Session ID
    session_id = pd.Series(3.0, index=close.index)  # default = late
    for name, (start, end) in SESSION_BOUNDARIES.items():
        mask = (hour >= start) & (hour < end)
        if name == "asia":
            session_id[mask] = 0.0
        elif name == "eu":
            session_id[mask] = 1.0
        elif name == "us":
            session_id[mask] = 2.0
        # late stays 3.0

    # Cyclic encoding
    hour_rad = 2 * np.pi * hour / 24.0
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)

    # EU-US overlap
    is_overlap = ((hour >= OVERLAP_START) & (hour < OVERLAP_END)).astype(float)

    result = pd.DataFrame(
        {
            "session_id": session_id,
            "hour_sin": hour_sin,
            "hour_cos": hour_cos,
            "is_session_overlap": is_overlap,
        },
        index=close.index,
    )

    # 如果 index 没有时间信息, 全部 fill 0.5 (neutral)
    if hour.isna().all():
        logger.warning("session_features: index has no datetime info, filling neutral")
        result["session_id"] = 0.0
        result["hour_sin"] = 0.0
        result["hour_cos"] = 0.0
        result["is_session_overlap"] = 0.0

    return result


# ═══════════════════════════════════════════════════════════════════════
# 2. Bars Since Local Extreme (距离局部极值的 bar 数)
# ═══════════════════════════════════════════════════════════════════════


@register_feature(
    "compute_bars_since_extreme_from_series",
    category="session",
    description=(
        "Bars since local high/low — measures how far we are from recent "
        "extremes, useful for entry timing (avoid chasing)"
    ),
    outputs=[
        "bars_since_local_high",
        "bars_since_local_low",
    ],
)
def compute_bars_since_extreme_from_series(
    *,
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20,
    normalize_window: int = 50,
) -> pd.DataFrame:
    """
    距离局部极值的 bar 数.

    输入: high, low
    输出:
      - bars_since_local_high: 距离 lookback 窗口内最高点的 bar 数 / normalize_window
      - bars_since_local_low:  距离 lookback 窗口内最低点的 bar 数 / normalize_window

    语义:
      - 值小 (0.0~0.1) = 刚创新高/新低, 可能追单风险
      - 值大 (0.5+) = 离极值远, 可能在回踩中
      - Entry Filter 用途: "bars_since_local_high > 0.1" 避免追高入场
    """
    high = pd.to_numeric(high, errors="coerce").astype(float)
    low = pd.to_numeric(low, errors="coerce").astype(float)

    n = len(high)
    bars_high = np.full(n, np.nan)
    bars_low = np.full(n, np.nan)

    high_arr = high.values
    low_arr = low.values

    for i in range(n):
        start = max(0, i - lookback + 1)
        window_h = high_arr[start : i + 1]
        window_l = low_arr[start : i + 1]

        if len(window_h) > 0:
            idx_max = np.nanargmax(window_h)
            bars_high[i] = len(window_h) - 1 - idx_max

            idx_min = np.nanargmin(window_l)
            bars_low[i] = len(window_l) - 1 - idx_min

    # 归一化到 [0, 1] 范围
    bars_high_norm = np.clip(bars_high / normalize_window, 0, 1)
    bars_low_norm = np.clip(bars_low / normalize_window, 0, 1)

    return pd.DataFrame(
        {
            "bars_since_local_high": pd.Series(bars_high_norm, index=high.index).fillna(
                0.0
            ),
            "bars_since_local_low": pd.Series(bars_low_norm, index=low.index).fillna(
                0.0
            ),
        }
    )
