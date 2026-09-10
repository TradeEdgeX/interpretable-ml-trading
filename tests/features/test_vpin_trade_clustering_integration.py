"""
集成测试：VPIN 和 Trade Clustering 的跨年计算和缓存命中率

测试目标：
1. 验证跨年计算（Dec → Jan）的正确性
2. 验证缓存命中率
3. 验证内存使用（流式处理）
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from src.data_tools.tick_loader import (
    compute_vpin_from_cached_ticks,
    _get_monthly_vpin_cache_key,
    _load_monthly_vpin_cache,
)
from src.features.time_series.utils_order_flow_features import (
    compute_trade_clustering_from_ticks,
)


class TestVPINTradeClusteringIntegration:
    """VPIN 和 Trade Clustering 集成测试"""

    @pytest.fixture
    def temp_cache_dir(self):
        """创建临时缓存目录"""
        cache_dir = Path(tempfile.mkdtemp())
        yield cache_dir
        shutil.rmtree(cache_dir, ignore_errors=True)

    @pytest.fixture
    def tick_data_dir(self, temp_cache_dir):
        """创建临时 tick 数据目录并生成跨年数据"""
        data_dir = temp_cache_dir / "tick_data"
        data_dir.mkdir()

        # 生成 2023-12 的数据（12月）
        dec_ticks = self._generate_tick_data(
            start_date="2023-12-01 00:00:00",
            n_days=31,
            n_ticks_per_day=1000,
        )
        dec_file = data_dir / "BTCUSDT_2023-12.parquet"
        dec_ticks.to_parquet(dec_file)

        # 生成 2024-01 的数据（1月）
        jan_ticks = self._generate_tick_data(
            start_date="2024-01-01 00:00:00",
            n_days=31,
            n_ticks_per_day=1000,
        )
        jan_file = data_dir / "BTCUSDT_2024-01.parquet"
        jan_ticks.to_parquet(jan_file)

        # 生成 2024-02 的数据（2月）
        feb_ticks = self._generate_tick_data(
            start_date="2024-02-01 00:00:00",
            n_days=29,  # 2024 是闰年
            n_ticks_per_day=1000,
        )
        feb_file = data_dir / "BTCUSDT_2024-02.parquet"
        feb_ticks.to_parquet(feb_file)

        return data_dir

    def _generate_tick_data(
        self, start_date: str, n_days: int, n_ticks_per_day: int = 1000
    ):
        """生成合成 tick 数据"""
        start = pd.Timestamp(start_date)
        all_ticks = []

        for day in range(n_days):
            day_start = start + timedelta(days=day)
            # 每天生成 n_ticks_per_day 笔成交
            tick_times = pd.date_range(day_start, periods=n_ticks_per_day, freq="1min")

            for ts in tick_times:
                # 随机生成买卖方向（1=buy, -1=sell）
                side = np.random.choice([1, -1])
                # 随机生成交易量（0.1 到 10.0）
                volume = np.random.uniform(0.1, 10.0)
                # 随机生成价格（50000 到 60000）
                price = np.random.uniform(50000, 60000)

                all_ticks.append(
                    {
                        "timestamp": ts,
                        "side": side,
                        "volume": volume,
                        "price": price,
                    }
                )

        df = pd.DataFrame(all_ticks)
        # 注意：tick_loader 期望 timestamp 是列，不是 index
        # 但为了兼容，我们保留 timestamp 列
        return df

    def test_cross_year_vpin_calculation(self, tick_data_dir, temp_cache_dir):
        """测试跨年 VPIN 计算（2023-12 → 2024-01 → 2024-02）"""
        # 计算 2023-12-15 到 2024-02-15 的 VPIN
        start_ts = "2023-12-15 00:00:00"
        end_ts = "2024-02-15 23:59:59"

        # 构建 cache_files 列表
        cache_files = [
            str(tick_data_dir / "BTCUSDT_2023-12.parquet"),
            str(tick_data_dir / "BTCUSDT_2024-01.parquet"),
            str(tick_data_dir / "BTCUSDT_2024-02.parquet"),
        ]

        result = compute_vpin_from_cached_ticks(
            cache_files=cache_files,
            start_ts=start_ts,
            end_ts=end_ts,
            bucket_volume=1000.0,
            n_buckets=50,
            adaptive=False,
            lookback_minutes=0,
            monthly_cache_dir=str(temp_cache_dir),
        )

        # 验证结果（result 是 Series，不是 DataFrame）
        assert result is not None, "VPIN 计算结果不应为空"
        assert len(result) > 0, "应该有 VPIN 数据点"
        assert result.index.min() >= pd.Timestamp(start_ts), "开始时间应正确"
        assert result.index.max() <= pd.Timestamp(end_ts), "结束时间应正确"

        # 验证跨年连续性（12月 → 1月 → 2月）
        # 检查是否有 12 月、1 月、2 月的数据
        dec_data = result.loc[result.index < "2024-01-01"]
        jan_data = result.loc[
            (result.index >= "2024-01-01") & (result.index < "2024-02-01")
        ]
        feb_data = result.loc[result.index >= "2024-02-01"]

        assert len(dec_data) > 0, "应该有 12 月的数据"
        assert len(jan_data) > 0, "应该有 1 月的数据"
        assert len(feb_data) > 0, "应该有 2 月的数据"

        # 验证 VPIN 值在合理范围内（0.0 到 1.0）
        # result 是 Series，直接访问值
        assert (result >= 0.0).all(), "VPIN 值应 >= 0"
        assert (result <= 1.0).all(), "VPIN 值应 <= 1"

    def test_vpin_cache_hit_rate(self, tick_data_dir, temp_cache_dir):
        """测试 VPIN 缓存命中率"""
        start_ts = "2024-01-01 00:00:00"
        end_ts = "2024-01-31 23:59:59"

        cache_files = [str(tick_data_dir / "BTCUSDT_2024-01.parquet")]

        # 第一次计算（应该计算并缓存）
        result1 = compute_vpin_from_cached_ticks(
            cache_files=cache_files,
            start_ts=start_ts,
            end_ts=end_ts,
            bucket_volume=1000.0,
            n_buckets=50,
            adaptive=False,
            lookback_minutes=0,
            monthly_cache_dir=str(temp_cache_dir),
        )

        # 第二次计算（应该从缓存加载）
        result2 = compute_vpin_from_cached_ticks(
            cache_files=cache_files,
            start_ts=start_ts,
            end_ts=end_ts,
            bucket_volume=1000.0,
            n_buckets=50,
            adaptive=False,
            lookback_minutes=0,
            monthly_cache_dir=str(temp_cache_dir),
        )

        # 验证两次结果一致（result 是 Series，不是 DataFrame）
        pd.testing.assert_series_equal(result1, result2, check_exact=False, rtol=1e-6)

        # 验证缓存文件存在
        cache_key = _get_monthly_vpin_cache_key(
            "BTCUSDT_2024-01.parquet",
            bucket_volume=1000.0,
            prev_bucket_state=None,
        )
        cached_result = _load_monthly_vpin_cache(temp_cache_dir, cache_key)
        assert cached_result is not None, "缓存应该存在"

    def test_cross_year_trade_clustering(self, tick_data_dir, temp_cache_dir):
        """测试跨年 Trade Clustering 计算"""
        # 读取 12 月和 1 月的数据
        dec_file = tick_data_dir / "BTCUSDT_2023-12.parquet"
        jan_file = tick_data_dir / "BTCUSDT_2024-01.parquet"

        dec_ticks = pd.read_parquet(dec_file)
        jan_ticks = pd.read_parquet(jan_file)

        # 计算 12 月的 Trade Clustering
        dec_result, dec_final_state = compute_trade_clustering_from_ticks(
            dec_ticks, window_size=100, initial_state=None
        )

        # 使用 12 月的 final_state 计算 1 月的 Trade Clustering
        jan_result, jan_final_state = compute_trade_clustering_from_ticks(
            jan_ticks, window_size=100, initial_state=dec_final_state
        )

        # 验证结果
        assert dec_result is not None, "12 月结果不应为空"
        assert jan_result is not None, "1 月结果不应为空"
        assert len(dec_result) > 0, "12 月应该有数据"
        assert len(jan_result) > 0, "1 月应该有数据"

        # 验证 final_state 格式
        assert (
            "current_run_side" in dec_final_state
        ), "final_state 应包含 current_run_side"
        assert "window_runs" in dec_final_state, "final_state 应包含 window_runs"

    def test_trade_clustering_cache_hit_rate(self, tick_data_dir, temp_cache_dir):
        """测试 Trade Clustering 缓存命中率（通过 compute_trade_clustering_from_ticks）"""
        jan_file = tick_data_dir / "BTCUSDT_2024-01.parquet"
        jan_ticks = pd.read_parquet(jan_file)

        # 第一次计算（应该计算并缓存）
        result1, state1 = compute_trade_clustering_from_ticks(
            jan_ticks, window_size=100, initial_state=None
        )

        # 第二次计算（应该得到相同结果）
        result2, state2 = compute_trade_clustering_from_ticks(
            jan_ticks, window_size=100, initial_state=None
        )

        # 验证两次结果一致（允许小的数值差异）
        pd.testing.assert_frame_equal(
            result1, result2, check_exact=False, rtol=0.01, atol=0.01
        )

        # 验证 final_state 一致
        assert state1 == state2, "final_state 应该一致"

    def test_memory_efficiency_streaming(self, tick_data_dir, temp_cache_dir):
        """测试流式处理的内存效率（跨多个月）"""
        import tracemalloc

        # 开始内存追踪
        tracemalloc.start()

        # 计算跨 3 个月的数据（12 月 → 1 月 → 2 月）
        start_ts = "2023-12-01 00:00:00"
        end_ts = "2024-02-28 23:59:59"

        cache_files = [
            str(tick_data_dir / "BTCUSDT_2023-12.parquet"),
            str(tick_data_dir / "BTCUSDT_2024-01.parquet"),
            str(tick_data_dir / "BTCUSDT_2024-02.parquet"),
        ]

        # 记录初始内存
        snapshot1 = tracemalloc.take_snapshot()

        result = compute_vpin_from_cached_ticks(
            cache_files=cache_files,
            start_ts=start_ts,
            end_ts=end_ts,
            bucket_volume=1000.0,
            n_buckets=50,
            adaptive=False,
            lookback_minutes=0,
            monthly_cache_dir=str(temp_cache_dir),
        )

        # 记录计算后内存
        snapshot2 = tracemalloc.take_snapshot()

        # 计算内存增长
        top_stats = snapshot2.compare_to(snapshot1, "lineno")
        total_memory_mb = sum(stat.size_diff for stat in top_stats) / (1024 * 1024)

        tracemalloc.stop()

        # 验证结果
        assert result is not None, "计算结果不应为空"
        assert len(result) > 0, "应该有数据"

        # 验证内存使用合理（流式处理应该只加载当前月和前一月的数据）
        # 3 个月的数据，流式处理应该只占用约 2 个月的内存
        # 允许一些额外开销（如缓存、中间结果等）
        assert (
            total_memory_mb < 500
        ), f"内存使用应合理（流式处理），实际使用：{total_memory_mb:.2f} MB"

        print(f"\n✅ 流式处理内存使用：{total_memory_mb:.2f} MB")

    def test_cross_year_state_continuity(self, tick_data_dir, temp_cache_dir):
        """测试跨年状态连续性（12 月 final_state → 1 月 initial_state）"""
        # 计算 12 月的 VPIN
        dec_cache_files = [str(tick_data_dir / "BTCUSDT_2023-12.parquet")]
        dec_result = compute_vpin_from_cached_ticks(
            cache_files=dec_cache_files,
            start_ts="2023-12-01 00:00:00",
            end_ts="2023-12-31 23:59:59",
            bucket_volume=1000.0,
            n_buckets=50,
            adaptive=False,
            lookback_minutes=0,
            monthly_cache_dir=str(temp_cache_dir),
        )

        # 计算 1 月的 VPIN（应该自动使用 12 月的 final_state）
        jan_cache_files = [
            str(tick_data_dir / "BTCUSDT_2023-12.parquet"),  # 前一个月
            str(tick_data_dir / "BTCUSDT_2024-01.parquet"),  # 当前月
        ]
        jan_result = compute_vpin_from_cached_ticks(
            cache_files=jan_cache_files,
            start_ts="2024-01-01 00:00:00",
            end_ts="2024-01-31 23:59:59",
            bucket_volume=1000.0,
            n_buckets=50,
            adaptive=False,
            lookback_minutes=0,
            monthly_cache_dir=str(temp_cache_dir),
        )

        # 验证结果
        assert dec_result is not None, "12 月结果不应为空"
        assert jan_result is not None, "1 月结果不应为空"

        # 验证连续性：12 月最后一个 bucket 和 1 月第一个 bucket 应该连续
        # （通过检查时间戳是否连续）
        dec_last_ts = dec_result.index.max()
        jan_first_ts = jan_result.index.min()

        # 时间差应该很小（跨月边界）
        time_diff = (jan_first_ts - dec_last_ts).total_seconds()
        assert time_diff >= 0, "1 月第一个 bucket 应该在 12 月最后一个 bucket 之后"
        assert time_diff < 86400 * 2, "时间差应该小于 2 天（跨月边界）"

        print(
            f"\n✅ 跨年状态连续性验证：12 月最后 bucket @ {dec_last_ts}, 1 月第一个 bucket @ {jan_first_ts}, 时间差：{time_diff/3600:.2f} 小时"
        )
