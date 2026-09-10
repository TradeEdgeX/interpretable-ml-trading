import numpy as np
import pandas as pd

from src.time_series_model.pipeline.training.label_utils import (
    compute_rr_label,
    simulate_rr_exits,
)


def _build_simple_path(
    direction: int,
    hit_tp_first: bool,
    atr_value: float = 10.0,
    max_holding_bars: int = 10,
) -> pd.DataFrame:
    """
    构造一段极简价格路径：
    - t=0 有信号（direction=+1 多头 / -1 空头）
    - t=1 以 open[1] 入场
    - t=2 根据 hit_tp_first 决定先触达 TP (±2R) 或 SL (∓1R)
    - 之后价格保持平稳
    """
    n = max_holding_bars + 5
    index = pd.RangeIndex(n)

    base_price = 100.0
    open_ = np.full(n, base_price, dtype=float)
    high = np.full(n, base_price, dtype=float)
    low = np.full(n, base_price, dtype=float)
    close = np.full(n, base_price, dtype=float)

    signal = np.zeros(n, dtype=float)
    signal[0] = float(direction)

    entry_price = base_price
    open_[1] = entry_price
    high[1] = entry_price
    low[1] = entry_price
    close[1] = entry_price

    R = atr_value

    if direction > 0:
        tp_level = entry_price + 2 * R
        sl_level = entry_price - 1 * R
        if hit_tp_first:
            low[2] = entry_price
            high[2] = tp_level + 1e-6
        else:
            low[2] = sl_level - 1e-6
            high[2] = entry_price
    else:
        tp_level = entry_price - 2 * R
        sl_level = entry_price + 1 * R
        if hit_tp_first:
            high[2] = entry_price
            low[2] = tp_level - 1e-6
        else:
            high[2] = sl_level + 1e-6
            low[2] = entry_price

    for t in range(3, n):
        open_[t] = base_price
        high[t] = base_price
        low[t] = base_price
        close[t] = base_price

    df = pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "atr": np.full(n, R, dtype=float),
            "signal": signal,
        },
        index=index,
    )
    return df


def test_simulate_rr_exits_matches_rr_label_long_tp_first():
    """多头：先触达 +2R，simulate_rr_exits 的平仓点应与 RR 标签的“成功”逻辑一致。"""
    df = _build_simple_path(direction=1, hit_tp_first=True)

    labels = compute_rr_label(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )
    long_exits, short_exits = simulate_rr_exits(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        entry_price_col="open",
        entry_offset=1,
    )

    # 标签在 t=0 应该是成功 (1.0)
    assert labels.iloc[0] == 1.0
    # 平仓应发生在 t=2
    assert long_exits.sum() == 1
    assert long_exits.index[long_exits.to_numpy()][0] == 2
    assert not short_exits.any()


def test_simulate_rr_exits_matches_rr_label_long_sl_first():
    """多头：先触达 -1R，simulate_rr_exits 应在 SL 处平仓，标签为 0。"""
    df = _build_simple_path(direction=1, hit_tp_first=False)

    labels = compute_rr_label(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )
    long_exits, short_exits = simulate_rr_exits(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        entry_price_col="open",
        entry_offset=1,
    )

    assert labels.iloc[0] == 0.0
    assert long_exits.sum() == 1
    assert long_exits.index[long_exits.to_numpy()][0] == 2
    assert not short_exits.any()


def test_simulate_rr_exits_matches_rr_label_short_tp_first():
    """空头：先触达 +2R（价格向下 2R），simulate_rr_exits 应在 TP 处平仓，标签为 1。"""
    df = _build_simple_path(direction=-1, hit_tp_first=True)

    labels = compute_rr_label(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )
    long_exits, short_exits = simulate_rr_exits(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        entry_price_col="open",
        entry_offset=1,
    )

    assert labels.iloc[0] == 1.0
    assert short_exits.sum() == 1
    assert short_exits.index[short_exits.to_numpy()][0] == 2
    assert not long_exits.any()


def test_simulate_rr_exits_matches_rr_label_short_sl_first():
    """空头：先触达 -1R（价格向上 1R），simulate_rr_exits 应在 SL 处平仓，标签为 0。"""
    df = _build_simple_path(direction=-1, hit_tp_first=False)

    labels = compute_rr_label(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        use_continuous_label=False,
        entry_price_col="open",
        entry_offset=1,
    )
    long_exits, short_exits = simulate_rr_exits(
        df,
        signal_col="signal",
        price_col="close",
        atr_col="atr",
        atr_window=14,
        max_holding_bars=10,
        stop_loss_r=1.0,
        take_profit_r=2.0,
        entry_price_col="open",
        entry_offset=1,
    )

    assert labels.iloc[0] == 0.0
    assert short_exits.sum() == 1
    assert short_exits.index[short_exits.to_numpy()][0] == 2
    assert not long_exits.any()
