"""
集成测试：验证 7 月份 Trade Clustering 数据计算
模拟真实场景，检查为什么测试集 Trade Clustering 特征全是 NaN
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from src.features.time_series.utils_order_flow_features import (
    extract_trade_clustering_features,
)
from src.data_tools.tick_loader import (
    serialize_tick_loader_params,
    load_tick_data,
)


@pytest.fixture
def temp_cache_dir():
    """临时缓存目录"""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir)


def create_realistic_july_data(
    symbol: str = "BTCUSDT",
    ticks_dir: str = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """创建真实的 7 月份数据（模拟训练场景）"""
    # 创建 7 月份的 K 线数据（测试集时间范围：2025-06-29 08:00:00 to 2025-07-31 00:00:00）
    # 但实际测试集是 7 月份的数据
    test_start = pd.Timestamp("2025-07-01 00:00:00")
    test_end = pd.Timestamp("2025-07-31 00:00:00")

    # 生成 4H K 线
    kline_times = pd.date_range(start=test_start, end=test_end, freq="4H")
    n_bars = len(kline_times)

    klines = pd.DataFrame(
        {
            "open": 50000.0 + np.random.randn(n_bars) * 10,
            "high": 50000.0 + np.abs(np.random.randn(n_bars)) * 20,
            "low": 50000.0 - np.abs(np.random.randn(n_bars)) * 20,
            "close": 50000.0 + np.random.randn(n_bars) * 10,
            "volume": np.random.uniform(100, 1000, size=n_bars),
        },
        index=kline_times,
    )

    # 生成 7 月份的 tick 数据
    tick_times = pd.date_range(
        start=test_start,
        end=test_end,
        freq="1S",  # 每秒一个 tick
    )
    n_ticks = len(tick_times)

    ticks = pd.DataFrame(
        {
            "price": 50000.0 + np.cumsum(np.random.randn(n_ticks) * 0.1),
            "volume": np.random.uniform(0.1, 10.0, size=n_ticks),
            "side": np.random.choice([1, -1], size=n_ticks, p=[0.52, 0.48]),
        },
        index=tick_times,
    )

    # 如果指定了 ticks_dir，保存 tick 数据
    if ticks_dir:
        tick_file = Path(ticks_dir) / f"{symbol}_2025-07.parquet"
        tick_file.parent.mkdir(parents=True, exist_ok=True)
        ticks_save = ticks.reset_index()
        ticks_save.rename(columns={"index": "timestamp"}, inplace=True)
        ticks_save.to_parquet(tick_file, index=False)
        print(f"   💾 Saved tick data to {tick_file}")

    return klines, ticks


def test_july_trade_clustering_direct_ticks():
    """测试 7 月份数据：直接使用 ticks 参数"""
    klines, ticks = create_realistic_july_data()

    print(f"\n📊 July data (direct ticks):")
    print(f"   Klines: {len(klines)} ({klines.index.min()} to {klines.index.max()})")
    print(f"   Ticks: {len(ticks)} ({ticks.index.min()} to {ticks.index.max()})")

    # 计算 Trade Clustering
    result_df = extract_trade_clustering_features(
        df=klines,
        ticks=ticks,
        window_size=100,
        freq="4H",
        monthly_cache_dir=None,
    )

    # 验证结果
    trade_cluster_cols = [
        col for col in result_df.columns if col.startswith("trade_cluster_")
    ]
    print(f"\n✅ Trade Clustering features: {len(trade_cluster_cols)}")

    # 检查是否有非 NaN 值
    for col in trade_cluster_cols[:10]:
        non_nan_count = result_df[col].notna().sum()
        non_zero_count = (result_df[col] != 0.0).sum()
        print(f"   {col}: {non_nan_count} non-NaN, {non_zero_count} non-zero")

        # 7 月份应该有数据
        assert non_zero_count > 0, f"{col} should have non-zero values for July data"

    # 检查时间对齐
    assert len(result_df) == len(klines), "Result should have same length as input"
    assert result_df.index.equals(
        klines.index
    ), "Result should have same index as input"


def test_july_trade_clustering_with_loader(temp_cache_dir):
    """测试 7 月份数据：使用 ticks_loader_json（模拟真实训练场景）"""
    symbol = "BTCUSDT"
    ticks_dir = temp_cache_dir

    klines, ticks = create_realistic_july_data(symbol=symbol, ticks_dir=ticks_dir)

    print(f"\n📊 July data (with loader):")
    print(f"   Klines: {len(klines)} ({klines.index.min()} to {klines.index.max()})")
    print(f"   Ticks: {len(ticks)} ({ticks.index.min()} to {ticks.index.max()})")

    # 创建 ticks_loader_json
    start_ts = klines.index.min().strftime("%Y-%m-%d %H:%M:%S")
    end_ts = klines.index.max().strftime("%Y-%m-%d %H:%M:%S")

    tick_params = {
        "symbol": symbol,
        "tick_files": [str(Path(ticks_dir) / f"{symbol}_2025-07.parquet")],
        "start_ts": start_ts,
        "end_ts": end_ts,
        "lookback_minutes": 0,
    }
    ticks_loader_json = serialize_tick_loader_params(tick_params)

    # 使用 ticks_loader_json 计算
    result_df = extract_trade_clustering_features(
        df=klines,
        ticks=None,
        ticks_loader_json=ticks_loader_json,
        window_size=100,
        freq="4H",
        monthly_cache_dir=temp_cache_dir,
        merge_batch_size=1,
        persist_monthly=False,
    )

    # 验证结果
    trade_cluster_cols = [
        col for col in result_df.columns if col.startswith("trade_cluster_")
    ]
    print(f"\n✅ Trade Clustering features: {len(trade_cluster_cols)}")

    if len(trade_cluster_cols) == 0:
        print("   ⚠️  No trade clustering features computed!")
        print("   This might indicate a problem with tick data loading or alignment")
        pytest.fail("No trade clustering features computed")

    # 检查是否有非 NaN 值
    for col in trade_cluster_cols[:10]:
        non_nan_count = result_df[col].notna().sum()
        non_zero_count = (result_df[col] != 0.0).sum()
        print(f"   {col}: {non_nan_count} non-NaN, {non_zero_count} non-zero")

        # 7 月份应该有数据
        assert non_zero_count > 0, f"{col} should have non-zero values for July data"


def test_july_trade_clustering_time_range_issue():
    """测试时间范围问题：模拟测试集场景（7 月份，但时间范围可能不匹配）"""
    # 测试集时间范围：2025-06-29 08:00:00 to 2025-07-31 00:00:00
    # 但实际测试集 K 线可能是从 7 月 1 日开始
    test_start = pd.Timestamp("2025-07-01 00:00:00")
    test_end = pd.Timestamp("2025-07-31 00:00:00")

    # 生成测试集 K 线（只包含 7 月份）
    kline_times = pd.date_range(start=test_start, end=test_end, freq="4H")
    klines = pd.DataFrame(
        {
            "open": 50000.0 + np.random.randn(len(kline_times)) * 10,
            "high": 50000.0 + np.abs(np.random.randn(len(kline_times))) * 20,
            "low": 50000.0 - np.abs(np.random.randn(len(kline_times))) * 20,
            "close": 50000.0 + np.random.randn(len(kline_times)) * 10,
            "volume": np.random.uniform(100, 1000, size=len(kline_times)),
        },
        index=kline_times,
    )

    # 生成 7 月份的 tick 数据（完整月份）
    tick_times = pd.date_range(
        start=pd.Timestamp("2025-07-01 00:00:00"),
        end=pd.Timestamp("2025-07-31 23:59:59"),
        freq="1S",
    )
    ticks = pd.DataFrame(
        {
            "price": 50000.0 + np.cumsum(np.random.randn(len(tick_times)) * 0.1),
            "volume": np.random.uniform(0.1, 10.0, size=len(tick_times)),
            "side": np.random.choice([1, -1], size=len(tick_times)),
        },
        index=tick_times,
    )

    print(f"\n📊 Test set scenario:")
    print(f"   Klines: {len(klines)} ({klines.index.min()} to {klines.index.max()})")
    print(f"   Ticks: {len(ticks)} ({ticks.index.min()} to {ticks.index.max()})")
    print(
        f"   Tick covers kline range: {ticks.index.min() <= klines.index.min() and ticks.index.max() >= klines.index.max()}"
    )

    # 计算 Trade Clustering
    result_df = extract_trade_clustering_features(
        df=klines,
        ticks=ticks,
        window_size=100,
        freq="4H",
        monthly_cache_dir=None,
    )

    # 验证结果
    trade_cluster_cols = [
        col for col in result_df.columns if col.startswith("trade_cluster_")
    ]

    print(f"\n📊 Results:")
    for col in trade_cluster_cols[:10]:
        non_nan_count = result_df[col].notna().sum()
        non_zero_count = (result_df[col] != 0.0).sum()
        print(f"   {col}: {non_nan_count} non-NaN, {non_zero_count} non-zero")

        # 如果 tick 数据覆盖了 K 线时间范围，应该有对齐的值
        if (
            ticks.index.min() <= klines.index.min()
            and ticks.index.max() >= klines.index.max()
        ):
            assert (
                non_zero_count > 0
            ), f"{col} should have aligned values when tick data covers kline range"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
