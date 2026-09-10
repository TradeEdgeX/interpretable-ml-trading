"""
集成测试：验证 Trade Clustering 特征计算和对齐
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from src.features.time_series.utils_order_flow_features import (
    extract_trade_clustering_features,
    compute_trade_clustering_from_ticks,
)
from src.data_tools.tick_loader import (
    serialize_tick_loader_params,
    list_tick_files,
)


@pytest.fixture
def temp_cache_dir():
    """临时缓存目录"""
    cache_dir = tempfile.mkdtemp()
    yield cache_dir
    shutil.rmtree(cache_dir)


def create_synthetic_ticks_for_month(
    symbol: str,
    year: int,
    month: int,
    n_ticks_per_day: int = 1000,
    base_price: float = 50000.0,
) -> pd.DataFrame:
    """为指定月份创建合成 tick 数据"""
    start_date = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end_date = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end_date = pd.Timestamp(year=year, month=month + 1, day=1)

    # 生成每天的 tick 数据
    all_ticks = []
    current_date = start_date
    while current_date < end_date:
        # 每天生成 n_ticks_per_day 个 tick
        tick_times = pd.date_range(
            start=current_date,
            end=current_date + pd.Timedelta(days=1),
            periods=n_ticks_per_day + 1,
        )[
            :-1
        ]  # 排除最后一个（下一天的开始）

        # 生成价格（随机游走）
        price_changes = np.random.randn(len(tick_times)) * 10
        prices = base_price + np.cumsum(price_changes)

        # 生成 side（1 或 -1）
        sides = np.random.choice([1, -1], size=len(tick_times), p=[0.52, 0.48])

        # 生成 volume
        volumes = np.random.uniform(0.1, 10.0, size=len(tick_times))

        ticks = pd.DataFrame(
            {
                "price": prices,
                "volume": volumes,
                "side": sides,
            },
            index=tick_times,
        )

        all_ticks.append(ticks)
        current_date += pd.Timedelta(days=1)

    result = pd.concat(all_ticks).sort_index()
    return result


def create_synthetic_klines_for_month(
    year: int,
    month: int,
    timeframe: str = "4H",
) -> pd.DataFrame:
    """为指定月份创建合成 K 线数据"""
    start_date = pd.Timestamp(year=year, month=month, day=1)
    if month == 12:
        end_date = pd.Timestamp(year=year + 1, month=1, day=1)
    else:
        end_date = pd.Timestamp(year=year, month=month + 1, day=1)

    # 生成 K 线时间索引
    kline_times = pd.date_range(start=start_date, end=end_date, freq=timeframe)

    # 生成 OHLCV 数据
    base_price = 50000.0
    n_bars = len(kline_times)
    returns = np.random.randn(n_bars) * 0.01
    prices = base_price * (1 + np.cumsum(returns))

    df = pd.DataFrame(
        {
            "open": prices * (1 + np.random.randn(n_bars) * 0.001),
            "high": prices * (1 + np.abs(np.random.randn(n_bars)) * 0.002),
            "low": prices * (1 - np.abs(np.random.randn(n_bars)) * 0.002),
            "close": prices,
            "volume": np.random.uniform(100, 1000, size=n_bars),
        },
        index=kline_times,
    )

    return df


def test_trade_clustering_single_month(temp_cache_dir):
    """测试单个月份的 Trade Clustering 计算"""
    year, month = 2025, 7

    # 创建合成数据
    ticks = create_synthetic_ticks_for_month(
        "BTCUSDT", year, month, n_ticks_per_day=500
    )
    klines = create_synthetic_klines_for_month(year, month, timeframe="4H")

    print(f"\n📊 Test data:")
    print(f"   Ticks: {len(ticks)} ({ticks.index.min()} to {ticks.index.max()})")
    print(f"   Klines: {len(klines)} ({klines.index.min()} to {klines.index.max()})")

    # 直接使用 ticks 计算
    result_df = extract_trade_clustering_features(
        df=klines,
        ticks=ticks,
        window_size=100,
        freq="4H",
        monthly_cache_dir=None,  # 不使用缓存
    )

    # 验证结果
    trade_cluster_cols = [
        col for col in result_df.columns if col.startswith("trade_cluster_")
    ]
    print(f"\n✅ Trade Clustering features: {len(trade_cluster_cols)}")
    print(f"   Columns: {trade_cluster_cols[:5]}...")

    # 检查是否有非 NaN 值
    for col in trade_cluster_cols[:5]:  # 检查前 5 个特征
        non_nan_count = result_df[col].notna().sum()
        print(f"   {col}: {non_nan_count}/{len(result_df)} non-NaN values")
        assert non_nan_count > 0, f"{col} should have non-NaN values"

    # 检查时间对齐
    assert len(result_df) == len(klines), "Result should have same length as input"
    assert result_df.index.equals(
        klines.index
    ), "Result should have same index as input"


def test_trade_clustering_with_ticks_loader_json(temp_cache_dir):
    """测试使用 ticks_loader_json 的 Trade Clustering 计算（模拟真实场景）"""
    year, month = 2025, 7

    # 创建合成数据
    ticks = create_synthetic_ticks_for_month(
        "BTCUSDT", year, month, n_ticks_per_day=500
    )
    klines = create_synthetic_klines_for_month(year, month, timeframe="4H")

    # 保存 tick 数据到临时文件（模拟真实场景）
    # 注意：load_tick_data 期望 parquet 文件有 timestamp 列（不是索引）
    tick_file = Path(temp_cache_dir) / f"BTCUSDT_{year}-{month:02d}.parquet"
    tick_file.parent.mkdir(parents=True, exist_ok=True)
    # 重置索引，将 timestamp 作为列保存
    ticks_save = ticks.reset_index()
    ticks_save.to_parquet(tick_file, index=False)

    print(f"\n📊 Test data:")
    print(f"   Tick file: {tick_file}")
    print(f"   Ticks: {len(ticks)} ({ticks.index.min()} to {ticks.index.max()})")
    print(f"   Klines: {len(klines)} ({klines.index.min()} to {klines.index.max()})")

    # 创建 ticks_loader_json
    start_ts = klines.index.min().strftime("%Y-%m-%d %H:%M:%S")
    end_ts = klines.index.max().strftime("%Y-%m-%d %H:%M:%S")

    tick_params = {
        "symbol": "BTCUSDT",
        "tick_files": [str(tick_file)],
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

    # 检查是否有非 NaN 值
    for col in trade_cluster_cols[:5]:
        non_nan_count = result_df[col].notna().sum()
        print(f"   {col}: {non_nan_count}/{len(result_df)} non-NaN values")
        assert non_nan_count > 0, f"{col} should have non-NaN values"

    # 检查时间对齐
    assert len(result_df) == len(klines), "Result should have same length as input"
    assert result_df.index.equals(
        klines.index
    ), "Result should have same index as input"


def test_trade_clustering_time_alignment():
    """测试 Trade Clustering 的时间对齐逻辑"""
    # 创建时间范围不匹配的数据（模拟测试集场景）
    year, month = 2025, 7

    # K 线数据（测试集时间范围）
    klines = create_synthetic_klines_for_month(year, month, timeframe="4H")
    # 只保留后 15 天（模拟测试集）
    test_start = klines.index.min() + pd.Timedelta(days=15)
    klines_test = klines.loc[test_start:].copy()

    # Tick 数据（完整月份）
    ticks = create_synthetic_ticks_for_month(
        "BTCUSDT", year, month, n_ticks_per_day=500
    )

    print(f"\n📊 Test data (time alignment):")
    print(
        f"   Klines (test set): {len(klines_test)} ({klines_test.index.min()} to {klines_test.index.max()})"
    )
    print(
        f"   Ticks (full month): {len(ticks)} ({ticks.index.min()} to {ticks.index.max()})"
    )

    # 计算 Trade Clustering
    result_df = extract_trade_clustering_features(
        df=klines_test,
        ticks=ticks,
        window_size=100,
        freq="4H",
        monthly_cache_dir=None,
    )

    # 验证结果
    trade_cluster_cols = [
        col for col in result_df.columns if col.startswith("trade_cluster_")
    ]

    # 检查对齐结果
    print(f"\n📊 Alignment results:")
    for col in trade_cluster_cols[:5]:
        non_nan_count = result_df[col].notna().sum()
        non_zero_count = (result_df[col] != 0.0).sum()
        print(f"   {col}: {non_nan_count} non-NaN, {non_zero_count} non-zero")

        # 即使时间范围不匹配，也应该有一些对齐的值（如果 tick 数据覆盖了 K 线时间范围）
        if (
            ticks.index.min() <= klines_test.index.min()
            and ticks.index.max() >= klines_test.index.max()
        ):
            assert (
                non_zero_count > 0
            ), f"{col} should have aligned values when tick data covers kline range"


def test_trade_clustering_cross_month_continuity():
    """测试跨月连续性（模拟训练集和测试集）"""
    # 训练集：6 月
    ticks_train = create_synthetic_ticks_for_month(
        "BTCUSDT", 2025, 6, n_ticks_per_day=500
    )
    klines_train = create_synthetic_klines_for_month(2025, 6, timeframe="4H")

    # 测试集：7 月
    ticks_test = create_synthetic_ticks_for_month(
        "BTCUSDT", 2025, 7, n_ticks_per_day=500
    )
    klines_test = create_synthetic_klines_for_month(2025, 7, timeframe="4H")

    print(f"\n📊 Test data (cross-month):")
    print(f"   Train: {len(klines_train)} bars, {len(ticks_train)} ticks")
    print(f"   Test: {len(klines_test)} bars, {len(ticks_test)} ticks")

    # 计算训练集
    result_train = extract_trade_clustering_features(
        df=klines_train,
        ticks=ticks_train,
        window_size=100,
        freq="4H",
        monthly_cache_dir=None,
    )

    # 计算测试集
    result_test = extract_trade_clustering_features(
        df=klines_test,
        ticks=ticks_test,
        window_size=100,
        freq="4H",
        monthly_cache_dir=None,
    )

    # 验证两个集合都有特征值
    trade_cluster_cols = [
        col for col in result_train.columns if col.startswith("trade_cluster_")
    ]

    print(f"\n📊 Results:")
    for col in trade_cluster_cols[:5]:
        train_non_zero = (result_train[col] != 0.0).sum()
        test_non_zero = (result_test[col] != 0.0).sum()
        print(
            f"   {col}: Train {train_non_zero}/{len(result_train)}, Test {test_non_zero}/{len(result_test)}"
        )

        assert train_non_zero > 0, f"Train set {col} should have values"
        assert test_non_zero > 0, f"Test set {col} should have values"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
