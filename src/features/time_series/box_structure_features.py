"""
Box Structure Features - causal consolidation-box detection.

设计理念（与既往 archetype features 风格对齐，不看未来）：

- 在固定特征档位上维护滚动 hi/lo（60/120/240/480/1200 根）。
- 输出以下原子信号，供多个策略按需消费（srb/bpc/me prefilter 升级、CRF 策略入场）：
    * ``box_hi_{N}``, ``box_lo_{N}`` - 过去 N 根（含当前）最高 high / 最低 low。
    * ``box_width_pct_{N}`` - ``(hi - lo) / mid``。
    * ``box_pos_{N}`` - ``(close - lo) / (hi - lo) ∈ [0, 1]`` 位置指示。
    * ``box_stability_{N}`` - 过去 N 根里 close 保持在 (lo+tol, hi-tol) 内的比例，
      其中 ``tol = max(1 × atr, 0.015 × mid)``。stability → 1 表示真盘整。
    * ``box_touches_hi_{N}`` / ``box_touches_lo_{N}`` - 过去 N 根触及上下沿的次数。
    * ``box_compression_score`` - ``box_width_pct_60 / box_width_pct_240``，<1 表短期压缩。
    * ``box_regime_label`` - 字符串分类：``small/mid/big/none``（由 stability × width 决定）。
    * ``box_breakout_up`` / ``box_breakout_down`` - 上一根刚从 mid 档 box 上/下破（±1/0）。
    * ``box_prior_trend_sign`` - box 形成前 60 根的 trend_r2 × sign(close change)。

规范：
- 仅 causal rolling；``box_hi/lo`` 用 ``high.rolling(N).max().shift(0)`` 即可；
  触发器（breakout）使用 ``.shift(1)`` 保证决策只能基于上一根信息。
- 所有输出 NaN-safe；warm-up 期填默认值（width=NaN, pos=0.5, stability=0.0, label='none'）。
- 无外部 CVD/OI 依赖；只需 OHLC + ATR。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.registry import register_feature

FEATURE_VERSION = "1.0"

# 档位窗口。策略通过 features.yaml 显式请求 box_*_{N} 列来锁定周期。
BOX_WINDOWS = (60, 120, 240, 480, 1200)

# tol 计算参数：tol = max(TOL_ATR_MULT * atr, TOL_PCT * mid)
TOL_ATR_MULT = 1.0
TOL_PCT = 0.015

# regime 判定阈值
REGIME_STABILITY_MIN = 0.70
REGIME_WIDTH_MAX_SMALL = 0.04  # <=4% 宽度 -> small
REGIME_WIDTH_MAX_MID = 0.08  # <=8% -> mid；>8% 且稳定 -> big

# 突破判定：close 越过上一根的 hi/lo（含 tol 松弛）
BREAKOUT_TOL_FRAC = 0.0  # 0 = 严格越过


def _nan_safe_series(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype(float)


def _rolling_atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    """Simple rolling ATR; lighter than baseline.compute_atr but identical semantics.

    Used as a fallback when caller does not pass ``atr``.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def _trend_r2_signed(close: pd.Series, n: int = 60) -> pd.Series:
    """Signed trend strength: R^2 of linear fit × sign of net change over N bars.

    Causal: uses only past N closes inclusive of the current bar.
    """
    x = np.arange(n, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()

    def _one(y: np.ndarray) -> float:
        if np.isnan(y).any():
            return 0.0
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        y_var = ((y - y_mean) ** 2).sum()
        if y_var <= 0 or x_var <= 0:
            return 0.0
        r2 = (cov * cov) / (x_var * y_var)
        sgn = 1.0 if y[-1] >= y[0] else -1.0
        return r2 * sgn

    return close.rolling(n, min_periods=n).apply(_one, raw=True)


def _classify_regime(stability: pd.Series, width: pd.Series) -> pd.Series:
    """Return a string Series in {small, mid, big, none}.

    - stability < REGIME_STABILITY_MIN -> 'none'
    - width <= REGIME_WIDTH_MAX_SMALL -> 'small'
    - width <= REGIME_WIDTH_MAX_MID -> 'mid'
    - else -> 'big'
    """
    label = pd.Series("none", index=stability.index, dtype=object)
    stable = stability.fillna(0.0) >= REGIME_STABILITY_MIN
    w = width.fillna(np.inf)
    label[stable & (w <= REGIME_WIDTH_MAX_SMALL)] = "small"
    label[stable & (w > REGIME_WIDTH_MAX_SMALL) & (w <= REGIME_WIDTH_MAX_MID)] = "mid"
    label[stable & (w > REGIME_WIDTH_MAX_MID)] = "big"
    return label


def _compute_one_window(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    atr: pd.Series,
    n: int,
) -> pd.DataFrame:
    """Compute box_* columns for a single window size n (causal)."""
    idx = close.index

    # past-N high/low (inclusive of current bar: window ends at t)
    box_hi = high.rolling(n, min_periods=n).max()
    box_lo = low.rolling(n, min_periods=n).min()
    mid = (box_hi + box_lo) * 0.5
    width = (box_hi - box_lo).clip(lower=1e-12)
    width_pct = (width / mid.replace(0, np.nan)).clip(lower=0.0)

    pos = ((close - box_lo) / width).clip(0.0, 1.0)

    # tol in price units (atr or pct of mid, whichever is larger)
    tol = np.maximum(TOL_ATR_MULT * atr.fillna(0.0), TOL_PCT * mid.abs())

    # Stability: how range-bound vs trending the past N bars are.
    #
    # Old definition (low>=box_lo-tol & high<=box_hi+tol) was TRIVIALLY TRUE
    # by construction (box_lo/hi are rolling min/max), so stability ≡ 1.0 and
    # prefilter never rejected anything.
    #
    # New causal definition: stability = 1 - R² of linear fit of close[-N:].
    #   • Pure trend:     R² ≈ 1  → stability ≈ 0
    #   • Noisy chop:     R² ≈ 0  → stability ≈ 1
    # Uses the same helper as trend_r2 to stay causal.
    stability = (1.0 - _trend_r2_signed(close, n).abs()).clip(0.0, 1.0)

    # touches within tol of the edge (again past-N window)
    near_hi = (high >= (box_hi - tol)).astype(float).fillna(0.0)
    near_lo = (low <= (box_lo + tol)).astype(float).fillna(0.0)
    touches_hi = near_hi.rolling(n, min_periods=n).sum()
    touches_lo = near_lo.rolling(n, min_periods=n).sum()

    out = pd.DataFrame(
        {
            f"box_hi_{n}": box_hi,
            f"box_lo_{n}": box_lo,
            f"box_width_pct_{n}": width_pct,
            f"box_pos_{n}": pos,
            f"box_stability_{n}": stability,
            f"box_touches_hi_{n}": touches_hi,
            f"box_touches_lo_{n}": touches_lo,
        },
        index=idx,
    )
    return out


@register_feature(
    "compute_box_structure_from_series",
    category="box_structure",
    description=(
        "Causal consolidation-box detector: rolling hi/lo + stability/touches on fixed "
        "scales (60/120/240/480/1200 2H bars), plus compression score, regime label, and "
        "breakout / prior-trend-direction signals."
    ),
    outputs=[
        # 60
        "box_hi_60",
        "box_lo_60",
        "box_width_pct_60",
        "box_pos_60",
        "box_stability_60",
        "box_touches_hi_60",
        "box_touches_lo_60",
        # 120
        "box_hi_120",
        "box_lo_120",
        "box_width_pct_120",
        "box_pos_120",
        "box_stability_120",
        "box_touches_hi_120",
        "box_touches_lo_120",
        # 240
        "box_hi_240",
        "box_lo_240",
        "box_width_pct_240",
        "box_pos_240",
        "box_stability_240",
        "box_touches_hi_240",
        "box_touches_lo_240",
        # 480
        "box_hi_480",
        "box_lo_480",
        "box_width_pct_480",
        "box_pos_480",
        "box_stability_480",
        "box_touches_hi_480",
        "box_touches_lo_480",
        # 1200
        "box_hi_1200",
        "box_lo_1200",
        "box_width_pct_1200",
        "box_pos_1200",
        "box_stability_1200",
        "box_touches_hi_1200",
        "box_touches_lo_1200",
        # derived
        "box_compression_score",
        "box_regime_label",
        "box_breakout_up",
        "box_breakout_down",
        "box_prior_trend_sign",
    ],
)
def compute_box_structure_from_series(
    *,
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series = None,
    atr_period: int = 14,
    trend_window: int = 60,
) -> pd.DataFrame:
    """Compute causal box-structure features on (close, high, low, [atr]).

    Args:
        close, high, low: OHLC series (DatetimeIndex aligned).
        atr: optional precomputed ATR in price units; if omitted, a rolling
            14-bar ATR is computed in-line.
        atr_period: fallback ATR period when ``atr`` is not provided.
        trend_window: window for ``box_prior_trend_sign``.

    Returns:
        DataFrame indexed by ``close.index`` with all registered ``box_*`` columns.
    """
    from src.features.live_tail_context import get_live_feature_tail, live_tail_slice_len

    _full_index = close.index
    _tail = get_live_feature_tail()
    # Live only needs short windows (e.g. chop_grid box_pos_60). Skip 480/1200
    # under bounded-tail so we don't force a ~1200-bar warmup that cancels the win.
    if _tail is not None:
        _windows = tuple(n for n in BOX_WINDOWS if n <= int(_tail))
        if not _windows:
            _windows = (min(BOX_WINDOWS),)
    else:
        _windows = BOX_WINDOWS
    _max_window = max(_windows) + int(trend_window) + int(atr_period) + 1
    _slice_len = live_tail_slice_len(len(close), warmup=_max_window)
    if _slice_len is not None:
        close = close.iloc[-_slice_len:]
        high = high.iloc[-_slice_len:]
        low = low.iloc[-_slice_len:]
        if atr is not None:
            atr = atr.iloc[-_slice_len:]

    close = _nan_safe_series(close)
    high = _nan_safe_series(high)
    low = _nan_safe_series(low)

    if atr is None:
        atr = _rolling_atr(high, low, close, n=atr_period)
    else:
        atr = _nan_safe_series(atr)

    frames = [_compute_one_window(high, low, close, atr, n) for n in _windows]
    out = pd.concat(frames, axis=1)
    # Ensure full column contract even when live skips long windows.
    for n in BOX_WINDOWS:
        if f"box_hi_{n}" not in out.columns:
            out[f"box_hi_{n}"] = np.nan
            out[f"box_lo_{n}"] = np.nan
            out[f"box_width_pct_{n}"] = np.nan
            out[f"box_pos_{n}"] = 0.5
            out[f"box_stability_{n}"] = 0.0
            out[f"box_touches_hi_{n}"] = 0.0
            out[f"box_touches_lo_{n}"] = 0.0

    # ── compression score: short-term width / long-term width ──
    wp60 = out["box_width_pct_60"]
    wp240 = out["box_width_pct_240"]
    compression = (wp60 / wp240.replace(0, np.nan)).clip(lower=0.0, upper=10.0)
    out["box_compression_score"] = compression

    # ── regime label (based on 120 window as the "default" timeframe) ──
    out["box_regime_label"] = _classify_regime(
        out["box_stability_120"], out["box_width_pct_120"]
    )

    # ── breakout triggers (use shift(1) so we compare current close to the PREVIOUS
    #    bar's box boundaries; this keeps the trigger strictly causal w.r.t. the
    #    decision at time t). ──
    prev_hi = out["box_hi_120"].shift(1)
    prev_lo = out["box_lo_120"].shift(1)
    breakout_up = (close > prev_hi * (1.0 + BREAKOUT_TOL_FRAC)).astype(int)
    breakout_down = (close < prev_lo * (1.0 - BREAKOUT_TOL_FRAC)).astype(int)
    # only count breakouts that emerge from a real box
    prev_regime = out["box_regime_label"].shift(1)
    in_box = prev_regime.isin(["small", "mid", "big"]).astype(int)
    out["box_breakout_up"] = (breakout_up * in_box).fillna(0).astype(int)
    out["box_breakout_down"] = (breakout_down * in_box).fillna(0).astype(int)

    # ── prior trend sign (signed R^2 over trend_window bars) ──
    out["box_prior_trend_sign"] = _trend_r2_signed(close, n=trend_window)

    # default-fill for downstream consumers
    for n in BOX_WINDOWS:
        out[f"box_pos_{n}"] = out[f"box_pos_{n}"].fillna(0.5)
        out[f"box_stability_{n}"] = out[f"box_stability_{n}"].fillna(0.0)
        out[f"box_touches_hi_{n}"] = out[f"box_touches_hi_{n}"].fillna(0.0)
        out[f"box_touches_lo_{n}"] = out[f"box_touches_lo_{n}"].fillna(0.0)
    out["box_compression_score"] = out["box_compression_score"].fillna(1.0)
    out["box_prior_trend_sign"] = out["box_prior_trend_sign"].fillna(0.0)

    if _slice_len is not None:
        # Older rows are unused live; fill numeric defaults / regime 'none'.
        out = out.reindex(_full_index)
        for col in out.columns:
            if col == "box_regime_label":
                out[col] = out[col].fillna("none")
            else:
                fill = 0.5 if "box_pos_" in str(col) else 0.0
                if col == "box_compression_score":
                    fill = 1.0
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(fill)
    return out


@register_feature(
    "compute_bpt_macro_box_direction_from_series",
    category="box_structure",
    description=(
        "Box Pullback Trend direction feature: local EMA1200 macro trend aligned "
        "with 120-bar box edge/chop pullback window."
    ),
    outputs=[
        "bpt_macro_box_direction",
        "bpt_box_pullback_window",
        "bpt_semantic_chop",
    ],
)
def compute_bpt_macro_box_direction_from_series(
    *,
    close: pd.Series,
    box_pos_120: pd.Series,
    box_stability_120: pd.Series,
    box_width_pct_120: pd.Series,
    box_touches_hi_120: pd.Series,
    box_touches_lo_120: pd.Series,
    ema_1200_position: pd.Series,
    ema_1200_slope_10: pd.Series,
    edge_frac: float = 0.15,
    stability_min: float = 0.85,
    width_min: float = 0.04,
    width_max: float = 0.30,
    touches_min: float = 5.0,
    chop_min: float = 0.40,
    ema_position_min_abs: float = 0.03,
) -> pd.DataFrame:
    """Emit +1/-1/0 for macro-aligned box-edge pullbacks.

    This is the pipeline version of the research diagnostic. It currently uses
    each symbol's own EMA1200 state as macro direction; the BTC-anchored variant
    remains in the diagnostic script until a generic cross-symbol EMA overlay is
    added to the feature pipeline.
    """
    idx = box_pos_120.index
    close = _nan_safe_series(close).reindex(idx)
    pos = _nan_safe_series(box_pos_120).reindex(idx).fillna(0.5)
    stability = _nan_safe_series(box_stability_120).reindex(idx).fillna(0.0)
    width = _nan_safe_series(box_width_pct_120).reindex(idx).fillna(np.inf)
    touches_hi = _nan_safe_series(box_touches_hi_120).reindex(idx).fillna(0.0)
    touches_lo = _nan_safe_series(box_touches_lo_120).reindex(idx).fillna(0.0)
    ema_pos = _nan_safe_series(ema_1200_position).reindex(idx).fillna(0.0)
    ema_slope = _nan_safe_series(ema_1200_slope_10).reindex(idx).fillna(0.0)

    ma = close.rolling(20, min_periods=20).mean()
    std = close.rolling(20, min_periods=20).std()
    bb_width = (4.0 * std / ma.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
    bb_width_pctile = (
        bb_width.rolling(240, min_periods=60).rank(pct=True).fillna(0.5)
    )
    signs = pd.concat(
        [
            np.sign(close.pct_change(3)),
            np.sign(close.pct_change(5)),
            np.sign(close.pct_change(10)),
        ],
        axis=1,
    ).fillna(0.0)
    direction_confidence = signs.abs().mean(axis=1) * signs.mean(axis=1).abs()
    bb_compression = (1.0 - bb_width_pctile.clip(0.0, 1.0)).clip(0.0, 1.0)
    chop = (bb_compression * (1.0 - direction_confidence.clip(0.0, 1.0)) * 2.0).clip(
        0.0, 1.0
    )

    box_ok = (
        (stability >= float(stability_min))
        & (width >= float(width_min))
        & (width <= float(width_max))
        & (chop >= float(chop_min))
    )
    lower_edge = (pos <= float(edge_frac)) & (touches_lo >= float(touches_min))
    upper_edge = (pos >= 1.0 - float(edge_frac)) & (touches_hi >= float(touches_min))
    macro_up = (ema_pos >= float(ema_position_min_abs)) & (ema_slope > 0.0)
    macro_down = (ema_pos <= -float(ema_position_min_abs)) & (ema_slope < 0.0)

    direction = pd.Series(0, index=idx, dtype="int8")
    direction.loc[box_ok & lower_edge & macro_up] = 1
    direction.loc[box_ok & upper_edge & macro_down] = -1

    return pd.DataFrame(
        {
            "bpt_macro_box_direction": direction.astype("int8"),
            "bpt_box_pullback_window": (direction != 0).astype("int8"),
            "bpt_semantic_chop": chop.fillna(0.0).astype(float),
        },
        index=idx,
    )
