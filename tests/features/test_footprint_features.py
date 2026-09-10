import pandas as pd
import numpy as np

from src.features.time_series.utils_footprint import (
    FootprintConfig,
    compute_kline_footprint_features,
)


def _make_ticks():
    idx = pd.date_range("2024-01-01 00:00:00", periods=8, freq="15min")
    prices = [100, 100.1, 100.2, 100.2, 100.3, 100.4, 100.4, 100.4]
    volumes = [1, 2, 3, 4, 3, 5, 6, 2]
    sides = [1, 1, -1, 1, -1, 1, 1, -1]
    return pd.DataFrame(
        {"price": prices, "volume": volumes, "side": sides},
        index=idx,
    )


def _make_klines():
    opens = pd.date_range("2024-01-01 00:00:00", periods=2, freq="1h")
    closes = opens + pd.Timedelta("1h")
    # add dummy open/close price for delta_divergence test (first bar up, second down)
    open_prices = [100.0, 101.0]
    close_prices = [101.0, 100.5]
    return pd.DataFrame(
        {
            "open_time": opens,
            "close_time": closes,
            "open": open_prices,
            "close": close_prices,
        },
        index=opens,
    )


def test_footprint_poc_and_value_area():
    ticks = _make_ticks()
    klines = _make_klines()
    cfg = FootprintConfig(
        price_bin_size=0.1,  # explicit bin for determinism
        value_area_pct=0.7,
    )
    res = compute_kline_footprint_features(ticks, klines, cfg=cfg)

    # first bar covers 4 ticks (0:4), second bar covers 4 ticks (4:8)
    first_poc = res.loc[klines.index[0], "fp_poc"]
    second_poc = res.loc[klines.index[1], "fp_poc"]

    assert not np.isnan(first_poc)
    assert not np.isnan(second_poc)
    # first bar highest volume around 100.2 (bin centered at 100.2)
    assert abs(first_poc - 100.25) < 1e-6
    # second bar highest volume around 100.4
    assert abs(second_poc - 100.45) < 1e-6

    # value area should be finite when volume exists
    assert np.isfinite(res["fp_vah"].iloc[0])
    assert np.isfinite(res["fp_val"].iloc[0])
    # exhaustion zscore computed
    assert "fp_exhaustion_zscore" in res.columns
    assert res["fp_exhaustion_zscore"].notna().iloc[0]
    # delta divergence for first bar: open->close up, delta_poc >0 -> aligned -> 0
    assert res["fp_delta_divergence"].iloc[0] == 0.0


def test_empty_bar_returns_nan():
    ticks = _make_ticks()
    klines = _make_klines()
    # shift ticks so they fall outside second bar
    ticks_shifted = ticks.copy()
    ticks_shifted.index = ticks_shifted.index - pd.Timedelta("2h")

    cfg = FootprintConfig(price_bin_size=0.1)
    res = compute_kline_footprint_features(ticks_shifted, klines, cfg=cfg)
    assert np.isnan(res["fp_poc"].iloc[1])
    assert np.isnan(res["fp_vah"].iloc[1])
    assert np.isnan(res["fp_delta_divergence"].iloc[1])


def test_exhaustion_and_divergence_signals():
    # Bar 1: strong negative delta at price 100.5, price closes up -> divergence = 1
    idx = pd.date_range("2024-01-01 02:00:00", periods=6, freq="10min")
    prices = [100.4, 100.5, 100.5, 100.5, 100.4, 100.3]
    volumes = [1, 5, 5, 5, 1, 1]
    sides = [-1, -1, -1, -1, 1, 1]  # heavy sells at 100.5
    ticks = pd.DataFrame({"price": prices, "volume": volumes, "side": sides}, index=idx)

    opens = pd.to_datetime(["2024-01-01 02:00:00"])
    closes = opens + pd.Timedelta("1h")
    klines = pd.DataFrame(
        {
            "open_time": opens,
            "close_time": closes,
            "open": [100.4],
            "close": [100.8],  # price up, delta negative -> divergence
        },
        index=opens,
    )

    cfg = FootprintConfig(price_bin_size=0.1, value_area_pct=0.7)
    res = compute_kline_footprint_features(ticks, klines, cfg=cfg)

    # Exhaustion should point near 100.5 where delta spike occurred
    assert abs(res["fp_exhaustion_price"].iloc[0] - 100.5) < 0.06
    assert res["fp_exhaustion_zscore"].iloc[0] > 1.0  # noticeable spike
    # Divergence flagged (price up, delta_poc negative)
    assert res["fp_delta_divergence"].iloc[0] == 1.0


def test_value_area_bounds_fixed_logic():
    """
    测试修复后的 Value Area 计算逻辑

    修复前的问题：cum <= value_area_pct 会漏掉刚好超过阈值的 bin
    修复后：使用 np.searchsorted 正确包含达到阈值的 bin
    """
    from src.features.time_series.utils_footprint import _value_area_bounds

    # 创建测试数据：3个 bins，成交量分别为 5, 3, 2（总计10）
    # 70% 应该是 7，应该包含前两个 bins（5+3=8 >= 7）
    volume_by_bin = pd.Series({0: 5, 1: 3, 2: 2})
    bin_edges = np.array([100.0, 100.1, 100.2, 100.3])

    vah, val = _value_area_bounds(
        volume_by_bin, value_area_pct=0.7, bin_edges=bin_edges
    )

    # 应该包含前两个 bins（索引 0 和 1）
    # bin 0: [100.0, 100.1), bin 1: [100.1, 100.2)
    # VAH 应该是 bin_edges[0] = 100.0
    # VAL 应该是 bin_edges[2] = 100.2（bin 1 的结束边界）
    assert vah == 100.0, f"VAH should be 100.0, got {vah}"
    assert val == 100.2, f"VAL should be 100.2, got {val}"

    # 测试边界情况：70% 应该刚好包含第一个 bin（5 < 7，但包含它以达到至少 POC）
    vah2, val2 = _value_area_bounds(
        volume_by_bin, value_area_pct=0.5, bin_edges=bin_edges
    )
    # 5 < 5.0 (50% of 10)，但至少应该包含 POC（第一个 bin）
    assert vah2 == 100.0
    assert val2 >= 100.1  # 至少包含一个 bin


def test_value_area_bounds_edge_cases():
    """测试 Value Area 边界情况"""
    from src.features.time_series.utils_footprint import _value_area_bounds

    # 测试空数据
    empty_vol = pd.Series(dtype=float)
    bin_edges = np.array([100.0, 100.1, 100.2])
    vah, val = _value_area_bounds(empty_vol, value_area_pct=0.7, bin_edges=bin_edges)
    assert np.isnan(vah) and np.isnan(val)

    # 测试单个 bin
    single_bin = pd.Series({0: 10})
    bin_edges = np.array([100.0, 100.1])
    vah, val = _value_area_bounds(single_bin, value_area_pct=0.7, bin_edges=bin_edges)
    assert vah == 100.0
    assert val == 100.1
