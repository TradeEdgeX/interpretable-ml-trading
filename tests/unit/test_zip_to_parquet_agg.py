import pandas as pd
import numpy as np
import pytest

from src.data_tools.zip_to_parquet import DataConverter


def test_preprocess_tick_data_aggregate_1s():
    """
    确认 100ms 级别的原始 tick 被聚合为 1s 粒度：
    - 价格使用全秒 VWAP（price*volume / sum(volume)）
    - 买卖方向分别累加 volume，输出独立两行
    """
    dc = DataConverter(input_dir=".", output_dir=".")

    # 构造同一秒内的 buy/sell ticks（100ms 间隔）
    ts_base = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
    df = pd.DataFrame(
        {
            "agg_trade_id": [1, 2, 3],
            "price": [100.0, 110.0, 120.0],
            "quantity": [2.0, 3.0, 5.0],
            "first_trade_id": [1, 2, 3],
            "last_trade_id": [1, 2, 3],
            "transact_time": [
                int((ts_base + pd.Timedelta(milliseconds=0)).timestamp() * 1000),
                int((ts_base + pd.Timedelta(milliseconds=100)).timestamp() * 1000),
                int((ts_base + pd.Timedelta(milliseconds=200)).timestamp() * 1000),
            ],
            # 前两条视为买单，最后一条视为卖单
            "is_buyer_maker": [False, False, True],
        }
    )

    agg_df = dc._preprocess_tick_data(df)  # type: ignore

    # 预期：同一秒聚合，输出两行（买/卖各一条）
    assert len(agg_df) == 2

    # 总成交量与 VWAP（全秒）检验
    total_volume = df["quantity"].sum()  # 2 + 3 + 5 = 10
    total_pv = (df["price"] * df["quantity"]).sum()  # 100*2 + 110*3 + 120*5 = 1140
    expected_vwap = total_pv / total_volume  # 114

    # 买方聚合
    buy_row = agg_df[agg_df["side"] == 1].iloc[0]
    assert buy_row["volume"] == pytest.approx(5.0)  # 2 + 3
    assert buy_row["price"] == pytest.approx(expected_vwap)

    # 卖方聚合
    sell_row = agg_df[agg_df["side"] == -1].iloc[0]
    assert sell_row["volume"] == pytest.approx(5.0)
    assert sell_row["price"] == pytest.approx(expected_vwap)

    # 时间戳应为该秒的起始（00:00:00，UTC tz-aware）
    assert buy_row["timestamp"] == ts_base
    assert sell_row["timestamp"] == ts_base
