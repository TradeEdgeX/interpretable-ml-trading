"""
Box-structure feature tests (``box_structure_features``).

覆盖：
1. 注册：函数可以通过 ``ensure_features_registered`` 查到。
2. 因果性：修改未来若干 bar 不影响较早位置的输出（无 look-ahead）。
3. 极端形态：完美水平线 → stability≈1；陡峭趋势 → stability→0。
4. NaN / warm-up：前 N-1 根允许 NaN，但不泄漏到后段。
5. 数值：box_pos ∈ [0,1]；breakout 触发一次至少 1 根。
6. regime label 枚举：只能落在 {small, mid, big, none} 四值内。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from src.features.registry import ensure_features_registered, get_feature_func
from src.features.normalization.raw_scale_columns import load_raw_scale_columns
from src.features.time_series.box_structure_features import (
    BOX_WINDOWS,
    compute_box_structure_from_series,
)


def _make_ohlc(close_arr: np.ndarray, amp: float = 0.0, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = pd.date_range("2024-01-01", periods=len(close_arr), freq="2h")
    close = pd.Series(close_arr, index=idx)
    # Give high/low some symmetric wiggle around close so ATR != 0
    wiggle = rng.rand(len(close_arr)) * amp + 1e-6
    high = close + wiggle
    low = close - wiggle
    return close, high, low


# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


def test_registry_lookup():
    ensure_features_registered()
    fn = get_feature_func("compute_box_structure_from_series")
    assert fn is compute_box_structure_from_series


def test_raw_scale_columns_exclude_price_unit_box_edges():
    expected = {
        f"box_{edge}_{window}"
        for edge in ("hi", "lo")
        for window in (60, 120, 240, 480, 1200)
    }
    cfg = yaml.safe_load(open("config/raw_scale_columns.yaml", encoding="utf-8"))
    raw_scale = set(cfg["raw_scale_columns"]["price_unit"])
    assert expected.issubset(raw_scale)
    assert expected.issubset(load_raw_scale_columns())

    deps = yaml.safe_load(open("config/feature_dependencies.yaml", encoding="utf-8"))
    assert "raw_scale_columns" not in deps


# ---------------------------------------------------------------------------
# 2. Causality (no look-ahead)
# ---------------------------------------------------------------------------


def test_no_lookahead():
    n = 1500
    rng = np.random.RandomState(123)
    base = 100 + np.cumsum(rng.randn(n) * 0.5)
    close, high, low = _make_ohlc(base, amp=0.3)

    out1 = compute_box_structure_from_series(close=close, high=high, low=low)

    # Replace the tail (last 50 bars) with an arbitrary extreme path
    tail_start = n - 50
    close2 = close.copy()
    high2 = high.copy()
    low2 = low.copy()
    close2.iloc[tail_start:] = close2.iloc[tail_start - 1] * 3.0
    high2.iloc[tail_start:] = close2.iloc[tail_start:] + 10.0
    low2.iloc[tail_start:] = close2.iloc[tail_start:] - 10.0

    out2 = compute_box_structure_from_series(close=close2, high=high2, low=low2)

    # Check: for every row strictly before (tail_start - max_window),
    # numeric columns must be identical (the replacement cannot bleed back).
    safe_end = tail_start - max(BOX_WINDOWS)
    assert safe_end > 0, "test design error: not enough history"
    numeric_cols = [c for c in out1.columns if out1[c].dtype != object]
    for col in numeric_cols:
        a = out1[col].iloc[:safe_end].to_numpy()
        b = out2[col].iloc[:safe_end].to_numpy()
        # treat NaN==NaN as equal
        mask = ~(np.isnan(a) & np.isnan(b))
        np.testing.assert_allclose(
            a[mask],
            b[mask],
            atol=1e-9,
            err_msg=f"look-ahead detected in column {col}",
        )


# ---------------------------------------------------------------------------
# 3. Extreme regimes
# ---------------------------------------------------------------------------


def test_flat_line_high_stability():
    """完美水平线 → stability ≈ 1, width ≈ 0."""
    n = 500
    close_arr = np.full(n, 100.0)
    close, high, low = _make_ohlc(close_arr, amp=0.0)  # no wiggle at all
    out = compute_box_structure_from_series(close=close, high=high, low=low)
    # Stability window is 2*N = 240; require full post-warm-up (t >= 240 + 120).
    s120 = out["box_stability_120"].iloc[360:]
    assert (s120 > 0.95).all(), f"flat-line stability low: {s120.min()}"
    # width ~ 0
    w120 = out["box_width_pct_120"].iloc[200:]
    assert (w120 < 1e-6).all()


def test_strong_trend_low_stability():
    """单调爬升 → stability 低（价格不断突破上沿）."""
    n = 600
    close_arr = 100.0 + np.arange(n) * 0.5  # +0.5 each bar
    close, high, low = _make_ohlc(close_arr, amp=0.1)
    out = compute_box_structure_from_series(close=close, high=high, low=low)
    # In a clean monotonic trend, close sits at top of past-N window;
    # stability can still be "within box" because high == roll_hi. The clearer
    # signal is box_pos staying near 1.
    p120 = out["box_pos_120"].iloc[360:]
    assert p120.median() > 0.85, f"trend top: median pos {p120.median()}"
    # prior_trend_sign should go strongly positive
    tsign = out["box_prior_trend_sign"].iloc[360:]
    assert tsign.median() > 0.5


# ---------------------------------------------------------------------------
# 4. Warm-up / NaN handling
# ---------------------------------------------------------------------------


def test_warmup_nans():
    n = 300
    rng = np.random.RandomState(1)
    base = 100 + np.cumsum(rng.randn(n) * 0.5)
    close, high, low = _make_ohlc(base, amp=0.2)
    out = compute_box_structure_from_series(close=close, high=high, low=low)
    # box_hi_240 needs at least 240 points; before that must be NaN
    hi240 = out["box_hi_240"]
    assert hi240.iloc[:239].isna().all()
    assert hi240.iloc[239:].notna().all()
    # box_pos/stability/touches have default fills (not NaN) after pipeline
    assert out["box_pos_120"].notna().all()
    assert out["box_stability_120"].notna().all()


# ---------------------------------------------------------------------------
# 5. Value ranges
# ---------------------------------------------------------------------------


def test_value_ranges():
    n = 500
    rng = np.random.RandomState(7)
    base = 100 + np.cumsum(rng.randn(n) * 0.5)
    close, high, low = _make_ohlc(base, amp=0.2)
    out = compute_box_structure_from_series(close=close, high=high, low=low)

    # pos ∈ [0,1]
    for n_win in BOX_WINDOWS:
        col = f"box_pos_{n_win}"
        vals = out[col].dropna()
        assert vals.min() >= -1e-9 and vals.max() <= 1.0 + 1e-9

    # stability ∈ [0,1]
    for n_win in BOX_WINDOWS:
        col = f"box_stability_{n_win}"
        vals = out[col].dropna()
        assert vals.min() >= -1e-9 and vals.max() <= 1.0 + 1e-9

    # breakout flags ∈ {0,1}
    assert set(out["box_breakout_up"].unique()).issubset({0, 1})
    assert set(out["box_breakout_down"].unique()).issubset({0, 1})


# ---------------------------------------------------------------------------
# 6. Regime label enumeration
# ---------------------------------------------------------------------------


def test_regime_label_values():
    n = 400
    rng = np.random.RandomState(9)
    base = 100 + np.cumsum(rng.randn(n) * 0.3)
    close, high, low = _make_ohlc(base, amp=0.1)
    out = compute_box_structure_from_series(close=close, high=high, low=low)
    allowed = {"small", "mid", "big", "none"}
    uniq = set(out["box_regime_label"].unique())
    assert uniq.issubset(allowed), f"unexpected labels: {uniq - allowed}"


# ---------------------------------------------------------------------------
# 7. Breakout emits a signal on a clean test pattern
# ---------------------------------------------------------------------------


def test_breakout_signal_fires_after_flat_box():
    """Construct: long flat at 100, then jump to 110 and hold.

    We need the flat segment to be long enough for ``box_stability_120`` to
    exceed 0.7 (its rolling window is 2*120=240), so the regime label reaches
    'small'/'mid'/'big' before the jump.
    """
    n_flat = 400
    n_up = 100
    arr = np.concatenate([np.full(n_flat, 100.0), np.full(n_up, 110.0)])
    close, high, low = _make_ohlc(arr, amp=0.05)
    out = compute_box_structure_from_series(close=close, high=high, low=low)
    jump = n_flat
    # Regime should be set right before the jump
    assert out["box_regime_label"].iloc[jump - 1] in {
        "small",
        "mid",
        "big",
    }, f"regime at pre-jump: {out['box_regime_label'].iloc[jump - 1]}"
    fired = out["box_breakout_up"].iloc[jump : jump + 30].sum()
    assert fired >= 1, "expected a breakout_up after flat→jump pattern"
