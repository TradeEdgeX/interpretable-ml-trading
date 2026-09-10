"""Unit tests for crypto zigzag / VPVR strong 2nd-buy helpers in ashare_utils."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.ashare_utils import (
    BUY2_POOL_REGIMES,
    REGIME_INSUFFICIENT,
    REGIME_MID_2BUY,
    REGIME_STRONG_2BUY,
    REGIME_THIRD_BUY,
    REGIME_WEAK_LOWER_LOW,
    classify_buy2_regime,
    compute_vpvr_snapshot,
    load_daily,
    pivots_from_crypto_zigzag,
    strong_buy2_alpha,
    v3_poc_launch_skip,
    vp_buy2_adjustment,
    vp_buy2_bonus,
)


def _make_daily(closes: list[float] | np.ndarray, spread: float = 0.02) -> pd.DataFrame:
    close = pd.Series(closes, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * (1 + spread),
            "low": close * (1 - spread),
            "close": close,
            "volume": 1_000_000.0,
        }
    )


def _hlhl_pivots(
    *,
    pivot_top: float = 100.0,
    one_buy: float = 85.0,
    rally: float = 105.0,
    pull: float = 102.0,
) -> list[tuple[int, float, str]]:
    return [
        (10, pivot_top, "H"),
        (20, one_buy, "L"),
        (30, rally, "H"),
        (40, pull, "L"),
    ]


def test_pivots_from_crypto_zigzag_merges_same_leg_updates():
    zz = pd.Series([100.0, 100.0, 90.0, 90.0, 110.0, 110.0, 105.0, 105.0])
    high = pd.Series([100.0] * len(zz))
    low = pd.Series([100.0] * len(zz))

    with patch(
        "src.features.time_series.baseline_features.compute_zigzag",
        return_value=zz,
    ):
        pivots = pivots_from_crypto_zigzag(high, low, threshold=0.08)

    assert pivots == [(2, 90.0, "L"), (4, 110.0, "H"), (6, 105.0, "L")]


def test_pivots_from_crypto_zigzag_returns_empty_when_flat():
    zz = pd.Series([50.0, 50.0])
    high = pd.Series([50.0, 50.0])
    low = pd.Series([50.0, 50.0])

    with patch(
        "src.features.time_series.baseline_features.compute_zigzag",
        return_value=zz,
    ):
        assert pivots_from_crypto_zigzag(high, low) == []


def test_classify_buy2_regime_strong_2buy():
    daily = _make_daily([100.0] * 80)
    pivots = _hlhl_pivots(pivot_top=100, one_buy=85, rally=105, pull=102)

    with patch("src.ashare_utils.pivots_from_crypto_zigzag", return_value=pivots):
        info = classify_buy2_regime(daily)

    assert info["regime"] == REGIME_STRONG_2BUY
    assert info["pattern"] == "HLHL"
    assert info["hold_above"] is True
    assert info["reclaimed"] is True


def test_classify_buy2_regime_mid_2buy_when_pull_below_pivot_top():
    daily = _make_daily([100.0] * 80)
    pivots = _hlhl_pivots(pivot_top=100, one_buy=85, rally=105, pull=98)

    with patch("src.ashare_utils.pivots_from_crypto_zigzag", return_value=pivots):
        info = classify_buy2_regime(daily)

    assert info["regime"] == REGIME_MID_2BUY
    assert info["reclaimed"] is True
    assert info["hold_above"] is False


def test_classify_buy2_regime_third_buy_when_extended():
    daily = _make_daily([100.0] * 79 + [108.0])
    pivots = _hlhl_pivots(pivot_top=100, one_buy=85, rally=105, pull=102)

    with patch("src.ashare_utils.pivots_from_crypto_zigzag", return_value=pivots):
        info = classify_buy2_regime(daily)

    assert info["regime"] == REGIME_THIRD_BUY
    assert info["close"] > info["rally"] * 1.02


def test_classify_buy2_regime_third_buy_on_high_ret20d():
    closes = [100.0] * 60 + list(np.linspace(80.0, 105.0, 21))
    daily = _make_daily(closes)
    pivots = _hlhl_pivots(pivot_top=100, one_buy=85, rally=105, pull=102)

    with patch("src.ashare_utils.pivots_from_crypto_zigzag", return_value=pivots):
        info = classify_buy2_regime(daily)

    assert info["ret_20d"] > 25.0
    assert info["close"] <= info["rally"] * 1.02
    assert info["regime"] == REGIME_THIRD_BUY


def test_classify_buy2_regime_weak_lower_low():
    daily = _make_daily([100.0] * 79 + [95.0])
    pivots = _hlhl_pivots(pivot_top=100, one_buy=85, rally=98, pull=84)

    with patch("src.ashare_utils.pivots_from_crypto_zigzag", return_value=pivots):
        info = classify_buy2_regime(daily)

    assert info["regime"] == REGIME_WEAK_LOWER_LOW
    assert info["higher_low"] is False


def test_classify_buy2_regime_insufficient_short_history():
    daily = _make_daily([100.0] * 30)
    info = classify_buy2_regime(daily)
    assert info["regime"] == REGIME_INSUFFICIENT


def test_classify_buy2_regime_insufficient_few_pivots():
    daily = _make_daily([100.0] * 80)
    with patch(
        "src.ashare_utils.pivots_from_crypto_zigzag",
        return_value=[(1, 100.0, "H"), (2, 90.0, "L")],
    ):
        info = classify_buy2_regime(daily)

    assert info["regime"] == REGIME_INSUFFICIENT
    assert info["pivot_count"] == 2


def test_strong_buy2_alpha_only_scores_pool_regimes():
    strong = {
        "regime": REGIME_STRONG_2BUY,
        "one_buy": 85,
        "rally": 105,
        "pull": 102,
        "ret_20d": 5.0,
        "dist_ma200": 10.0,
    }
    mid = {**strong, "regime": REGIME_MID_2BUY}
    third = {**strong, "regime": REGIME_THIRD_BUY}

    assert strong_buy2_alpha(third) == 0.0
    assert strong_buy2_alpha(strong) > strong_buy2_alpha(mid)
    assert 0.0 < strong_buy2_alpha(mid) <= 1.0


def test_strong_buy2_alpha_includes_buy2_progress():
    regime_info = {
        "regime": REGIME_STRONG_2BUY,
        "one_buy": 85,
        "rally": 105,
        "pull": 95,
        "ret_20d": 5.0,
        "dist_ma200": 10.0,
    }
    progress = {
        "progress": {
            "deep_pullback": True,
            "stopped_falling": True,
            "vol_confirm": True,
            "not_surged": True,
            "has_amp": True,
        }
    }
    base = strong_buy2_alpha(regime_info)
    boosted = strong_buy2_alpha(regime_info, progress)
    assert boosted > base


def test_vp_buy2_bonus_applies_only_in_pool():
    vp = {
        "valid": True,
        "near_poc": True,
        "on_hvn": True,
        "vpvr_price_in_lvn": 0.0,
    }
    assert vp_buy2_bonus(vp, REGIME_STRONG_2BUY) == pytest.approx(0.07)
    assert vp_buy2_bonus(vp, REGIME_THIRD_BUY) == 0.0
    assert vp_buy2_bonus({"valid": False}, REGIME_STRONG_2BUY) == 0.0


def test_compute_vpvr_snapshot_too_short():
    daily = _make_daily([10.0] * 50)
    assert compute_vpvr_snapshot(daily, window=100)["valid"] is False


def test_compute_vpvr_snapshot_returns_vpvr_fields():
    rng = np.random.default_rng(42)
    n = 150
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    close = np.maximum(close, 1.0)
    daily = _make_daily(close)
    daily["volume"] = rng.uniform(5e5, 2e6, n)

    snap = compute_vpvr_snapshot(daily, window=100)
    assert snap["valid"] is True
    assert snap["poc_price"] > 0
    assert "dist_to_poc_pct" in snap
    assert "vpvr_volume_density" in snap
    assert "vpvr_lvn_distance" in snap
    assert isinstance(snap["near_poc"], bool)
    assert isinstance(snap["on_hvn"], bool)


def test_integration_crypto_zigzag_hlhl_strong_2buy():
    """End-to-end: synthetic swings → crypto compute_zigzag → STRONG_2buy."""
    segments = [(88, 40), (120, 30), (88, 30), (132, 30), (125, 25)]
    closes: list[float] = []
    for price, n in segments:
        closes.extend([float(price)] * n)
    daily = _make_daily(closes)

    info = classify_buy2_regime(
        daily,
        threshold=0.08,
        extended_ret60_pct=999.0,
        extended_ret20_pct=999.0,
        extended_ma200_pct=999.0,
    )
    assert info["regime"] == REGIME_STRONG_2BUY
    assert info["pattern"] == "HLHL"
    assert info["regime"] in BUY2_POOL_REGIMES


def test_integration_600160_classified_third_buy_after_tightening():
    daily = load_daily("600160")
    if daily is None:
        pytest.skip("no daily data for 600160")
    info = classify_buy2_regime(daily, threshold=0.08)
    assert info["regime"] == REGIME_THIRD_BUY


def test_v3_poc_launch_skip_far_from_chip_peak():
    vp = {"valid": True, "dist_to_poc_pct": 20.8}
    assert v3_poc_launch_skip(vp, 12.0) is True
    assert v3_poc_launch_skip(vp, 999.0) is False


def test_vp_buy2_adjustment_penalizes_far_from_poc():
    vp = {
        "valid": True,
        "near_poc": False,
        "on_hvn": False,
        "dist_to_poc_pct": 20.8,
        "vpvr_price_in_lvn": 1.0,
    }
    assert vp_buy2_adjustment(vp, REGIME_STRONG_2BUY) < 0


def test_integration_crypto_zigzag_mid_2buy_when_pull_below_first_high():
    """End-to-end: shallow pullback leaves price below first zigzag high → mid_2buy."""
    segments = [(88, 40), (120, 30), (88, 30), (132, 30), (122, 25)]
    closes: list[float] = []
    for price, n in segments:
        closes.extend([float(price)] * n)
    daily = _make_daily(closes)

    info = classify_buy2_regime(
        daily,
        threshold=0.08,
        extended_ret60_pct=999.0,
        extended_ret20_pct=999.0,
        extended_ma200_pct=999.0,
    )
    assert info["regime"] == REGIME_MID_2BUY
    assert info["pattern"] == "HLHL"
    assert info["hold_above"] is False
