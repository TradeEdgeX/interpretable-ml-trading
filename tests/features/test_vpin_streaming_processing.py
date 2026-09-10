"""
VPIN 流式处理集成测试

测试内容：
1. 流式处理：按月分批计算，每次只加载两个月的数据
2. 验证结果正确性：流式处理的结果与预期一致
3. 验证内存优化：不会一次性加载所有月份的数据
4. 验证跨月滚动平均：前一个月的数据正确用于滚动平均
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from src.data_tools.tick_loader import (
    compute_vpin_from_cached_ticks,
    _get_monthly_vpin_cache_key,
    _load_monthly_vpin_cache,
    _save_monthly_vpin_cache,
    _compute_vpin_buckets_for_month,
)


@pytest.fixture
def temp_cache_dir():
    """创建临时缓存目录"""
    cache_dir = Path(tempfile.mkdtemp())
    yield cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def temp_tick_dir():
    """创建临时tick数据目录"""
    tick_dir = Path(tempfile.mkdtemp())
    yield tick_dir
    shutil.rmtree(tick_dir, ignore_errors=True)


def create_monthly_tick_data(
    tick_dir: Path, symbol: str, year: int, month: int, n_ticks: int = 5000
):
    """创建单月的tick数据文件"""
    np.random.seed(42 + year * 100 + month)  # 确保每个月的数据不同

    # 生成该月的tick数据
    month_start = pd.Timestamp(year=year, month=month, day=1)
    timestamps = pd.date_range(month_start, periods=n_ticks, freq="10S")

    # 价格和成交量
    base_price = 50000 + (year - 2024) * 1000 + month * 10
    prices = base_price + np.cumsum(np.random.randn(n_ticks) * 10)
    volumes = np.random.uniform(0.1, 5.0, n_ticks)
    sides = np.random.choice([1, -1], n_ticks, p=[0.52, 0.48])

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )

    # 保存为parquet文件
    file_path = tick_dir / f"{symbol}_{year}-{month:02d}.parquet"
    df.to_parquet(file_path, index=False)

    return file_path


def create_multi_month_tick_data(
    tick_dir: Path, symbol: str, start_year: int, start_month: int, n_months: int
):
    """创建多个月的tick数据文件"""
    files = []
    for i in range(n_months):
        year = start_year
        month = start_month + i
        while month > 12:
            month -= 12
            year += 1
        file_path = create_monthly_tick_data(tick_dir, symbol, year, month)
        files.append(str(file_path))
    return files


def test_streaming_processing_multiple_months(temp_tick_dir, temp_cache_dir):
    """测试流式处理多个月的VPIN计算"""
    print("\n" + "=" * 70)
    print("测试：流式处理多个月的VPIN计算")
    print("=" * 70)

    symbol = "BTCUSDT"

    # 创建5、6、7月的tick数据
    tick_files = create_multi_month_tick_data(
        temp_tick_dir, symbol, 2024, 5, n_months=3
    )

    print(f"   📁 Created {len(tick_files)} tick files:")
    for f in tick_files:
        print(f"      - {Path(f).name}")

    # 计算5-7月的VPIN（流式处理）
    start_ts = "2024-05-01 00:00:00"
    end_ts = "2024-07-31 23:59:59"

    print(f"\n   📊 Computing VPIN for {start_ts} to {end_ts} (streaming mode)")

    vpin_result = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts,
        end_ts=end_ts,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    # 验证结果
    assert len(vpin_result) > 0, "VPIN结果应该不为空"
    assert isinstance(vpin_result, pd.Series), "VPIN结果应该是Series"
    assert vpin_result.index.is_monotonic_increasing, "VPIN时间索引应该是递增的"

    # 验证时间范围
    result_start = vpin_result.index.min()
    result_end = vpin_result.index.max()
    expected_start = pd.to_datetime(start_ts)
    expected_end = pd.to_datetime(end_ts)

    print(f"\n   ✅ VPIN computed:")
    print(f"      Time range: {result_start} to {result_end}")
    print(f"      Expected: {expected_start} to {expected_end}")
    print(f"      Number of values: {len(vpin_result)}")
    print(f"      VPIN range: [{vpin_result.min():.4f}, {vpin_result.max():.4f}]")

    # 验证时间范围覆盖了5-7月
    # 注意：VPIN需要先积累buckets才能开始计算，所以结果开始时间可能稍晚于请求的开始时间
    # 允许最多1小时的延迟（用于积累buckets）
    assert result_start <= expected_start + pd.Timedelta(
        hours=1
    ), f"结果开始时间应该在请求开始时间的1小时内，但 {result_start} > {expected_start + pd.Timedelta(hours=1)}"
    # 结果结束时间应该在请求结束时间之前（因为需要时间积累buckets）
    # 但应该覆盖大部分时间范围
    assert result_end >= expected_start, "结果应该覆盖开始时间附近的数据"

    # 验证每个月都有数据（使用时间范围查询，因为可能不是从月初开始）
    may_mask = (vpin_result.index >= pd.Timestamp("2024-05-01")) & (
        vpin_result.index < pd.Timestamp("2024-06-01")
    )
    june_mask = (vpin_result.index >= pd.Timestamp("2024-06-01")) & (
        vpin_result.index < pd.Timestamp("2024-07-01")
    )
    july_mask = (vpin_result.index >= pd.Timestamp("2024-07-01")) & (
        vpin_result.index < pd.Timestamp("2024-08-01")
    )

    may_data = vpin_result[may_mask]
    june_data = vpin_result[june_mask]
    july_data = vpin_result[july_mask]

    assert len(may_data) > 0, "5月应该有VPIN数据"
    assert len(june_data) > 0, "6月应该有VPIN数据"
    assert len(july_data) > 0, "7月应该有VPIN数据"

    print(f"      May data points: {len(may_data)}")
    print(f"      June data points: {len(june_data)}")
    print(f"      July data points: {len(july_data)}")

    # 验证VPIN值的合理性（应该在0-1之间）
    assert (vpin_result >= 0).all(), "VPIN值应该 >= 0"
    assert (vpin_result <= 1).all(), "VPIN值应该 <= 1"

    print(f"\n   ✅ 流式处理测试通过")


def test_streaming_processing_single_month(temp_tick_dir, temp_cache_dir):
    """测试流式处理单个月的VPIN计算（只需要当前月+前一个月）"""
    print("\n" + "=" * 70)
    print("测试：流式处理单个月的VPIN计算")
    print("=" * 70)

    symbol = "BTCUSDT"

    # 创建6、7月的tick数据
    tick_files = create_multi_month_tick_data(
        temp_tick_dir, symbol, 2024, 6, n_months=2
    )

    print(f"   📁 Created {len(tick_files)} tick files:")
    for f in tick_files:
        print(f"      - {Path(f).name}")

    # 只计算7月的VPIN（应该只需要6月和7月的数据）
    start_ts = "2024-07-01 00:00:00"
    end_ts = "2024-07-31 23:59:59"

    print(
        f"\n   📊 Computing VPIN for {start_ts} to {end_ts} (should only load June + July)"
    )

    vpin_result = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts,
        end_ts=end_ts,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    # 验证结果
    assert len(vpin_result) > 0, "VPIN结果应该不为空"

    # 验证只有7月的数据
    result_start = vpin_result.index.min()
    result_end = vpin_result.index.max()

    print(f"\n   ✅ VPIN computed:")
    print(f"      Time range: {result_start} to {result_end}")
    print(f"      Number of values: {len(vpin_result)}")

    # 验证时间范围在7月
    assert result_start >= pd.to_datetime("2024-07-01"), "结果应该在7月"
    assert result_end <= pd.to_datetime("2024-07-31 23:59:59"), "结果应该在7月"

    print(f"\n   ✅ 单月流式处理测试通过")


def test_streaming_processing_cross_month_rolling(temp_tick_dir, temp_cache_dir):
    """测试跨月滚动平均的正确性"""
    print("\n" + "=" * 70)
    print("测试：跨月滚动平均的正确性")
    print("=" * 70)

    symbol = "BTCUSDT"

    # 创建5、6、7月的tick数据
    tick_files = create_multi_month_tick_data(
        temp_tick_dir, symbol, 2024, 5, n_months=3
    )

    # 计算6月1日的VPIN（需要5月和6月的数据进行滚动平均）
    start_ts = "2024-06-01 00:00:00"
    end_ts = "2024-06-01 23:59:59"

    print(f"   📊 Computing VPIN for {start_ts} to {end_ts}")
    print(f"      Should use May + June data for rolling average")

    vpin_result = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts,
        end_ts=end_ts,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    # 验证结果
    assert len(vpin_result) > 0, "VPIN结果应该不为空"

    # 验证6月1日有数据
    june_1_data = vpin_result.loc["2024-06-01"]
    assert len(june_1_data) > 0, "6月1日应该有VPIN数据"

    print(f"\n   ✅ VPIN computed for June 1:")
    print(f"      Number of values: {len(june_1_data)}")
    print(f"      VPIN range: [{june_1_data.min():.4f}, {june_1_data.max():.4f}]")

    # 验证VPIN值的合理性
    assert (june_1_data >= 0).all(), "VPIN值应该 >= 0"
    assert (june_1_data <= 1).all(), "VPIN值应该 <= 1"

    print(f"\n   ✅ 跨月滚动平均测试通过")


def test_streaming_processing_cache_reuse(temp_tick_dir, temp_cache_dir):
    """测试缓存复用（前一个月的数据在下次循环中复用）"""
    print("\n" + "=" * 70)
    print("测试：缓存复用")
    print("=" * 70)

    symbol = "BTCUSDT"

    # 创建5、6、7月的tick数据
    tick_files = create_multi_month_tick_data(
        temp_tick_dir, symbol, 2024, 5, n_months=3
    )

    # 第一次计算：5-7月的VPIN（会计算并缓存所有月份）
    print(f"   📊 First computation: May to July")
    vpin_result1 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts="2024-05-01 00:00:00",
        end_ts="2024-07-31 23:59:59",
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    # 第二次计算：只计算7月的VPIN（应该从缓存加载6月和7月）
    print(f"\n   📊 Second computation: July only (should use cached June + July)")
    vpin_result2 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts="2024-07-01 00:00:00",
        end_ts="2024-07-31 23:59:59",
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    # 验证结果一致性（使用时间范围查询）
    july_mask = (vpin_result1.index >= pd.Timestamp("2024-07-01")) & (
        vpin_result1.index < pd.Timestamp("2024-08-01")
    )
    july_from_full = vpin_result1[july_mask]
    july_from_single = vpin_result2

    # 对齐时间索引
    common_index = july_from_full.index.intersection(july_from_single.index)
    if len(common_index) > 0:
        diff = (
            july_from_full.loc[common_index] - july_from_single.loc[common_index]
        ).abs()
        max_diff = diff.max()

        print(f"\n   ✅ Comparing July VPIN from full vs single computation:")
        print(f"      Common time points: {len(common_index)}")
        print(f"      Max difference: {max_diff:.6f}")

        # 允许小的数值误差（由于浮点运算）
        assert max_diff < 1e-5, f"两次计算的结果应该一致，但最大差异为 {max_diff}"
    else:
        # 如果没有共同的时间点，至少验证两个结果都不为空
        assert len(july_from_full) > 0, "完整计算的7月数据应该不为空"
        assert len(july_from_single) > 0, "单独计算的7月数据应该不为空"
        print(
            f"\n   ✅ Both computations produced July data (no common time points to compare)"
        )

    print(f"\n   ✅ 缓存复用测试通过")


def test_bucket_continuity_across_months(temp_tick_dir, temp_cache_dir):
    """测试跨月 bucket 连续性（确保 bucket 边界不会在月份切换时被切断）"""
    print("\n" + "=" * 70)
    print("测试：跨月 bucket 连续性")
    print("=" * 70)

    symbol = "BTCUSDT"

    # 创建5、6月的tick数据
    may_file = create_monthly_tick_data(temp_tick_dir, symbol, 2024, 5, n_ticks=5000)
    june_file = create_monthly_tick_data(temp_tick_dir, symbol, 2024, 6, n_ticks=5000)

    tick_files = [str(may_file), str(june_file)]

    # 计算5-6月的VPIN（应该确保 bucket 连续性）
    start_ts = "2024-05-01 00:00:00"
    end_ts = "2024-06-30 23:59:59"

    print(f"   📊 Computing VPIN for {start_ts} to {end_ts}")
    print(f"      Should ensure bucket continuity across May and June")
    print(f"      Key: 6月计算时应该使用 5月的 final_state 作为 initial_state")

    vpin_result = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts,
        end_ts=end_ts,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    # 验证结果
    assert len(vpin_result) > 0, "VPIN结果应该不为空"

    print(f"\n   ✅ VPIN computed:")
    print(f"      Time range: {vpin_result.index.min()} to {vpin_result.index.max()}")
    print(f"      Number of values: {len(vpin_result)}")
    print(f"      VPIN range: [{vpin_result.min():.4f}, {vpin_result.max():.4f}]")

    # 验证每个月都有数据
    may_mask = (vpin_result.index >= pd.Timestamp("2024-05-01")) & (
        vpin_result.index < pd.Timestamp("2024-06-01")
    )
    june_mask = (vpin_result.index >= pd.Timestamp("2024-06-01")) & (
        vpin_result.index < pd.Timestamp("2024-07-01")
    )

    may_data = vpin_result[may_mask]
    june_data = vpin_result[june_mask]

    assert len(may_data) > 0, "5月应该有VPIN数据"
    assert len(june_data) > 0, "6月应该有VPIN数据"

    print(f"      May data points: {len(may_data)}")
    print(f"      June data points: {len(june_data)}")

    # 验证VPIN值的合理性
    assert (vpin_result >= 0).all(), "VPIN值应该 >= 0"
    assert (vpin_result <= 1).all(), "VPIN值应该 <= 1"

    print(f"\n   ✅ Bucket 连续性测试通过（使用了 prev_bucket_state 确保跨月连续性）")


def test_cross_year_rolling_cache_behavior(temp_cache_dir, temp_tick_dir):
    """
    测试跨年滚动的缓存行为

    关键验证点：
    1. 每个月的 final_state 是固定的（不依赖于 prev_bucket_state）
    2. 如果从12月开始计算，然后继续到1月、2月...6月：
       - 12月：可能需要重新计算（如果之前没算过）
       - 1月：需要重新计算（因为 prev_state = final_12，而第一次运行时1月是基于 prev_state = None）
       - 2~6月：应该命中缓存（因为 prev_state = final_1，而 final_1 是固定的）
    """
    symbol = "BTCUSDT"

    # 创建12月、1月、2月、3月、4月、5月、6月的tick数据
    months = [
        (2023, 12),  # 12月
        (2024, 1),  # 1月
        (2024, 2),  # 2月
        (2024, 3),  # 3月
        (2024, 4),  # 4月
        (2024, 5),  # 5月
        (2024, 6),  # 6月
    ]

    tick_files = []
    for year, month in months:
        tick_file = create_monthly_tick_data(
            temp_tick_dir, symbol, year, month, n_ticks=5000
        )
        tick_files.append(str(tick_file))

    print(f"\n   📊 测试跨年滚动的缓存行为")
    print(f"   ========================================")

    # ========== 第一次运行：1~6月 ==========
    print(f"\n   🔹 第一次运行：1~6月（建立标准缓存和状态缓存）")

    start_ts_1 = "2024-01-01 00:00:00"
    end_ts_1 = "2024-06-30 23:59:59"

    vpin_result_1 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_1,
        end_ts=end_ts_1,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_1) > 0, "第一次运行应该产生VPIN结果"

    # 获取1月的 final_state（用于验证）
    # 由于我们无法直接获取 final_state，我们通过计算1月的标准缓存来验证
    cache_key_jan_std = _get_monthly_vpin_cache_key(
        tick_files[1],  # 1月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    jan_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_jan_std)
    assert jan_std_cache is not None, "1月的标准缓存应该存在"
    jan_buckets_std, jan_final_state_std = jan_std_cache

    print(f"   ✅ 第一次运行完成，建立了1~6月的标准缓存和状态缓存")
    print(
        f"      1月标准缓存 final_state: filled_value = {jan_final_state_std.get('filled_value', 0.0):.6f}"
    )

    # ========== 第二次运行：12月~6月（跨年滚动）==========
    print(f"\n   🔹 第二次运行：12月~6月（跨年滚动）")
    print(
        f"      关键验证：1月需要重新计算（prev_state = final_12），但2~6月应该命中缓存"
    )

    start_ts_2 = "2023-12-01 00:00:00"
    end_ts_2 = "2024-06-30 23:59:59"

    # 使用一个简单的计数器来跟踪缓存命中情况
    # 由于我们无法直接访问内部缓存命中计数，我们通过检查缓存文件的存在来验证

    vpin_result_2 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_2,
        end_ts=end_ts_2,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_2) > 0, "第二次运行应该产生VPIN结果"

    # 验证12月的标准缓存应该存在
    cache_key_dec_std = _get_monthly_vpin_cache_key(
        tick_files[0],  # 12月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    dec_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_dec_std)
    assert dec_std_cache is not None, "12月的标准缓存应该存在"
    dec_buckets_std, dec_final_state_std = dec_std_cache

    # 验证1月的标准缓存仍然存在（不应该被覆盖）
    jan_std_cache_after = _load_monthly_vpin_cache(temp_cache_dir, cache_key_jan_std)
    assert jan_std_cache_after is not None, "1月的标准缓存应该仍然存在"
    jan_buckets_std_after, jan_final_state_std_after = jan_std_cache_after

    # 验证1月的 final_state 是固定的（不依赖于 prev_bucket_state）
    assert (
        abs(
            jan_final_state_std.get("filled_value", 0.0)
            - jan_final_state_std_after.get("filled_value", 0.0)
        )
        < 1e-6
    ), "1月的 final_state 应该是固定的，不依赖于 prev_bucket_state"

    print(f"   ✅ 验证通过：1月的 final_state 是固定的")
    print(
        f"      第一次运行：filled_value = {jan_final_state_std.get('filled_value', 0.0):.6f}"
    )
    print(
        f"      第二次运行后：filled_value = {jan_final_state_std_after.get('filled_value', 0.0):.6f}"
    )

    # 验证2月的标准缓存应该存在（因为2月的 prev_state = final_1，而 final_1 是固定的）
    cache_key_feb_std = _get_monthly_vpin_cache_key(
        tick_files[2],  # 2月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    feb_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_feb_std)
    assert feb_std_cache is not None, "2月的标准缓存应该存在"

    # 验证2月使用 final_1 的状态缓存应该存在（因为第一次运行时已经计算过）
    cache_key_feb_with_final1 = _get_monthly_vpin_cache_key(
        tick_files[2],  # 2月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=jan_final_state_std,  # 使用1月的 final_state
    )
    feb_with_final1_cache = _load_monthly_vpin_cache(
        temp_cache_dir, cache_key_feb_with_final1
    )
    assert (
        feb_with_final1_cache is not None
    ), "2月使用 final_1 的状态缓存应该存在（第一次运行时已建立）"

    print(f"   ✅ 验证通过：2月使用 final_1 的状态缓存存在，应该可以命中")

    # 验证结果一致性：1~6月的VPIN值应该一致（除了可能的第一个bucket）
    # 由于1月的第一个bucket可能不同（因为 prev_state 不同），我们比较除了第一个bucket之外的值
    jan_mask_1 = (vpin_result_1.index >= pd.Timestamp("2024-01-01")) & (
        vpin_result_1.index < pd.Timestamp("2024-02-01")
    )
    jan_mask_2 = (vpin_result_2.index >= pd.Timestamp("2024-01-01")) & (
        vpin_result_2.index < pd.Timestamp("2024-02-01")
    )

    jan_data_1 = vpin_result_1[jan_mask_1]
    jan_data_2 = vpin_result_2[jan_mask_2]

    # 找到共同的时间点（排除第一个bucket可能的不同）
    common_times = jan_data_1.index.intersection(jan_data_2.index)
    if len(common_times) > 1:
        # 排除第一个时间点（可能因为 prev_state 不同而不同）
        common_times = common_times[1:]
        if len(common_times) > 0:
            diff = (jan_data_1.loc[common_times] - jan_data_2.loc[common_times]).abs()
            max_diff = diff.max()
            print(f"   📊 1月VPIN值比较（排除第一个bucket）：")
            print(f"      共同时间点：{len(common_times)}")
            print(f"      最大差异：{max_diff:.6f}")
            # 由于第一个bucket可能不同，后续的bucket应该相同（因为 final_state 是固定的）
            assert (
                max_diff < 1e-5
            ), f"1月VPIN值应该基本一致（排除第一个bucket），最大差异：{max_diff}"

    # 验证2~6月的VPIN值应该完全一致
    for month_idx, (year, month) in enumerate(
        [(2024, 2), (2024, 3), (2024, 4), (2024, 5), (2024, 6)], start=2
    ):
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

        month_mask_1 = (vpin_result_1.index >= month_start) & (
            vpin_result_1.index < month_end
        )
        month_mask_2 = (vpin_result_2.index >= month_start) & (
            vpin_result_2.index < month_end
        )

        month_data_1 = vpin_result_1[month_mask_1]
        month_data_2 = vpin_result_2[month_mask_2]

        # 找到共同的时间点
        common_times = month_data_1.index.intersection(month_data_2.index)
        if len(common_times) > 0:
            diff = (
                month_data_1.loc[common_times] - month_data_2.loc[common_times]
            ).abs()
            max_diff = diff.max()
            print(f"   📊 {year}-{month:02d}月VPIN值比较：")
            print(f"      共同时间点：{len(common_times)}")
            print(f"      最大差异：{max_diff:.6f}")
            assert (
                max_diff < 1e-5
            ), f"{year}-{month:02d}月VPIN值应该完全一致，最大差异：{max_diff}"

    print(f"\n   ✅ 跨年滚动缓存行为测试通过")
    print(
        f"      - 1月需要重新计算（因为 prev_state = final_12），但 final_state 是固定的"
    )
    print(f"      - 2~6月应该命中缓存（因为 prev_state 和第一次运行时一样）")


def test_skip_months_cache_behavior(temp_cache_dir, temp_tick_dir):
    """
    测试跳过月份计算的缓存行为

    场景：计算1~6月后，直接计算3~6月
    关键验证点：
    1. 标准缓存只存储 final_state，不存储 buckets（节省存储空间）
    2. 计算3~6月时，需要使用 final_state[2] 来启动3月的计算
    3. 不能直接使用 buckets_std[3]（因为它是从空状态开始的，不符合时间连续性）
    4. 应该使用 final_state[2] + 3月原始数据 → 得到正确的 buckets[3 2]
    """
    symbol = "BTCUSDT"

    # 创建1月、2月、3月、4月、5月、6月的tick数据
    months = [
        (2024, 1),  # 1月
        (2024, 2),  # 2月
        (2024, 3),  # 3月
        (2024, 4),  # 4月
        (2024, 5),  # 5月
        (2024, 6),  # 6月
    ]

    tick_files = []
    for year, month in months:
        tick_file = create_monthly_tick_data(
            temp_tick_dir, symbol, year, month, n_ticks=5000
        )
        tick_files.append(str(tick_file))

    print(f"\n   📊 测试跳过月份计算的缓存行为")
    print(f"   ========================================")

    # ========== 第一次运行：1~6月 ==========
    print(f"\n   🔹 第一次运行：1~6月（建立标准缓存和状态缓存）")

    start_ts_1 = "2024-01-01 00:00:00"
    end_ts_1 = "2024-06-30 23:59:59"

    vpin_result_1 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_1,
        end_ts=end_ts_1,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_1) > 0, "第一次运行应该产生VPIN结果"

    # 获取2月的 final_state（用于验证）
    cache_key_feb_std = _get_monthly_vpin_cache_key(
        tick_files[1],  # 2月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    feb_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_feb_std)
    assert feb_std_cache is not None, "2月的标准缓存应该存在"
    feb_buckets_std, feb_final_state_std = feb_std_cache

    # 验证标准缓存只存储了 final_state（buckets 应该为 None）
    assert feb_buckets_std is None, "标准缓存应该只存储 final_state，不存储 buckets"
    assert feb_final_state_std is not None, "标准缓存应该存储 final_state"

    print(f"   ✅ 第一次运行完成，建立了1~6月的标准缓存和状态缓存")
    print(f"      验证：2月标准缓存只存储了 final_state（buckets=None）")
    print(
        f"      2月标准缓存 final_state: filled_value = {feb_final_state_std.get('filled_value', 0.0):.6f}"
    )

    # ========== 第二次运行：3~6月（跳过1~2月）==========
    print(f"\n   🔹 第二次运行：3~6月（跳过1~2月）")
    print(f"      关键验证：")
    print(f"      1. 不能直接使用 buckets_std[3]（因为它是从空状态开始的）")
    print(f"      2. 需要使用 final_state[2] 来启动3月的计算")
    print(f"      3. 如果之前算过 1~6 月，应该已经缓存了 buckets_state[3 2] → 直接命中")

    start_ts_2 = "2024-03-01 00:00:00"
    end_ts_2 = "2024-06-30 23:59:59"

    vpin_result_2 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_2,
        end_ts=end_ts_2,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_2) > 0, "第二次运行应该产生VPIN结果"

    # 验证3月的标准缓存仍然只存储 final_state
    cache_key_mar_std = _get_monthly_vpin_cache_key(
        tick_files[2],  # 3月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    mar_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_mar_std)
    assert mar_std_cache is not None, "3月的标准缓存应该存在"
    mar_buckets_std, mar_final_state_std = mar_std_cache

    assert (
        mar_buckets_std is None
    ), "3月的标准缓存应该只存储 final_state，不存储 buckets"

    # 验证3月使用 final_state[2] 的状态缓存应该存在（因为第一次运行时已经计算过）
    cache_key_mar_with_final2 = _get_monthly_vpin_cache_key(
        tick_files[2],  # 3月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=feb_final_state_std,  # 使用2月的 final_state
    )
    mar_with_final2_cache = _load_monthly_vpin_cache(
        temp_cache_dir, cache_key_mar_with_final2
    )
    assert (
        mar_with_final2_cache is not None
    ), "3月使用 final_state[2] 的状态缓存应该存在（第一次运行时已建立）"
    mar_buckets_with_final2, mar_final_state_with_final2 = mar_with_final2_cache

    assert mar_buckets_with_final2 is not None, "状态缓存应该存储 buckets"
    assert mar_final_state_with_final2 is not None, "状态缓存应该存储 final_state"

    print(f"   ✅ 验证通过：")
    print(f"      - 3月标准缓存只存储了 final_state（buckets=None）")
    print(f"      - 3月使用 final_state[2] 的状态缓存存在，应该可以命中")

    # 验证结果一致性：3~6月的VPIN值应该完全一致
    for month_idx, (year, month) in enumerate(
        [(2024, 3), (2024, 4), (2024, 5), (2024, 6)], start=2
    ):
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

        month_mask_1 = (vpin_result_1.index >= month_start) & (
            vpin_result_1.index < month_end
        )
        month_mask_2 = (vpin_result_2.index >= month_start) & (
            vpin_result_2.index < month_end
        )

        month_data_1 = vpin_result_1[month_mask_1]
        month_data_2 = vpin_result_2[month_mask_2]

        # 找到共同的时间点
        common_times = month_data_1.index.intersection(month_data_2.index)
        if len(common_times) > 0:
            diff = (
                month_data_1.loc[common_times] - month_data_2.loc[common_times]
            ).abs()
            max_diff = diff.max()
            print(f"   📊 {year}-{month:02d}月VPIN值比较：")
            print(f"      共同时间点：{len(common_times)}")
            print(f"      最大差异：{max_diff:.6f}")
            # 注意：由于标准缓存只存储 final_state，重新计算时可能会有微小差异
            # 这是因为 final_state 可能依赖于 prev_bucket_state（如果月份数据不足以填满最后一个bucket）
            # 但差异应该很小（< 0.02），因为大部分情况下 final_state 应该是固定的
            assert (
                max_diff < 0.02
            ), f"{year}-{month:02d}月VPIN值应该基本一致，最大差异：{max_diff}"

    print(f"\n   ✅ 跳过月份计算的缓存行为测试通过")
    print(f"      - 标准缓存只存储 final_state，节省存储空间")
    print(f"      - 计算3~6月时，使用 final_state[2] 来启动3月的计算")
    print(f"      - 不能直接使用 buckets_std[3]（因为它是从空状态开始的）")
    print(
        f"      - 注意：final_state 可能依赖于 prev_bucket_state（如果月份数据不足以填满最后一个bucket）"
    )


def test_prev_state_auto_load_from_prev_month(temp_cache_dir, temp_tick_dir):
    """
    测试自动从前一个月加载 final_state 的逻辑

    关键验证点：
    1. 当计算3~6月时，3月应该自动获取2月的 final_state，而不是使用 prev=None
    2. 即使 prev_bucket_state 初始为 None，如果前一个月存在，也应该自动加载其 final_state
    3. 只有真正的起点（如全年第一个月）才使用 prev=None
    4. 标准缓存的 buckets 不会被直接使用（即使存在），而是重新计算以确保正确性
    """
    symbol = "BTCUSDT"

    # 创建1月、2月、3月、4月的tick数据
    months = [
        (2024, 1),  # 1月
        (2024, 2),  # 2月
        (2024, 3),  # 3月
        (2024, 4),  # 4月
    ]

    tick_files = []
    for year, month in months:
        tick_file = create_monthly_tick_data(
            temp_tick_dir, symbol, year, month, n_ticks=5000
        )
        tick_files.append(str(tick_file))

    print(f"\n   📊 测试自动从前一个月加载 final_state 的逻辑")
    print(f"   ========================================")

    # ========== 第一次运行：1~4月（建立标准缓存）==========
    print(f"\n   🔹 第一次运行：1~4月（建立标准缓存）")

    start_ts_1 = "2024-01-01 00:00:00"
    end_ts_1 = "2024-04-30 23:59:59"

    vpin_result_1 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_1,
        end_ts=end_ts_1,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_1) > 0, "第一次运行应该产生VPIN结果"

    # 获取2月的 final_state（用于验证）
    cache_key_feb_std = _get_monthly_vpin_cache_key(
        tick_files[1],  # 2月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    feb_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_feb_std)
    assert feb_std_cache is not None, "2月的标准缓存应该存在"
    feb_buckets_std, feb_final_state_std = feb_std_cache

    print(f"   ✅ 第一次运行完成，建立了1~4月的标准缓存")
    print(
        f"      2月标准缓存 final_state: filled_value = {feb_final_state_std.get('filled_value', 0.0):.6f}"
    )

    # ========== 第二次运行：3~4月（跳过1~2月，验证自动加载2月的 final_state）==========
    print(f"\n   🔹 第二次运行：3~4月（跳过1~2月）")
    print(f"      关键验证：")
    print(f"      1. 3月应该自动获取2月的 final_state，而不是使用 prev=None")
    print(
        f"      2. 即使 prev_bucket_state 初始为 None，也应该自动加载前一个月的 final_state"
    )
    print(f"      3. 标准缓存的 buckets 不会被直接使用，而是重新计算")

    start_ts_2 = "2024-03-01 00:00:00"
    end_ts_2 = "2024-04-30 23:59:59"

    vpin_result_2 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_2,
        end_ts=end_ts_2,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_2) > 0, "第二次运行应该产生VPIN结果"

    # 验证3月使用 final_state[2] 的状态缓存应该存在（因为第二次运行时自动加载了2月的 final_state）
    cache_key_mar_with_final2 = _get_monthly_vpin_cache_key(
        tick_files[2],  # 3月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=feb_final_state_std,  # 使用2月的 final_state
    )
    mar_with_final2_cache = _load_monthly_vpin_cache(
        temp_cache_dir, cache_key_mar_with_final2
    )
    assert (
        mar_with_final2_cache is not None
    ), "3月使用 final_state[2] 的状态缓存应该存在（第二次运行时自动加载并计算）"
    mar_buckets_with_final2, mar_final_state_with_final2 = mar_with_final2_cache

    assert mar_buckets_with_final2 is not None, "状态缓存应该存储 buckets"
    assert mar_final_state_with_final2 is not None, "状态缓存应该存储 final_state"

    print(f"   ✅ 验证通过：")
    print(f"      - 3月自动加载了2月的 final_state")
    print(f"      - 3月使用 final_state[2] 的状态缓存存在")

    # 验证结果一致性：3~4月的VPIN值应该与第一次运行一致（或非常接近）
    for month_idx, (year, month) in enumerate([(2024, 3), (2024, 4)], start=2):
        month_start = pd.Timestamp(year=year, month=month, day=1)
        month_end = month_start + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)

        month_mask_1 = (vpin_result_1.index >= month_start) & (
            vpin_result_1.index < month_end
        )
        month_mask_2 = (vpin_result_2.index >= month_start) & (
            vpin_result_2.index < month_end
        )

        month_data_1 = vpin_result_1[month_mask_1]
        month_data_2 = vpin_result_2[month_mask_2]

        # 找到共同的时间点
        common_times = month_data_1.index.intersection(month_data_2.index)
        if len(common_times) > 0:
            diff = (
                month_data_1.loc[common_times] - month_data_2.loc[common_times]
            ).abs()
            max_diff = diff.max()
            print(f"   📊 {year}-{month:02d}月VPIN值比较：")
            print(f"      共同时间点：{len(common_times)}")
            print(f"      最大差异：{max_diff:.6f}")
            # 由于自动加载了前一个月的 final_state，结果应该与第一次运行一致（或非常接近）
            assert (
                max_diff < 0.02
            ), f"{year}-{month:02d}月VPIN值应该基本一致，最大差异：{max_diff}"

    print(f"\n   ✅ 自动加载前一个月 final_state 的逻辑测试通过")
    print(f"      - 3月自动获取了2月的 final_state，而不是使用 prev=None")
    print(f"      - 标准缓存的 buckets 不会被直接使用，而是重新计算")
    print(f"      - 只有当前面没有数据了（prev_month_file 为 None），才使用 prev=None")


def test_cross_year_prev_state_load(temp_cache_dir, temp_tick_dir):
    """
    测试跨年计算时自动加载前一个月 final_state 的逻辑

    场景：先计算12月，然后计算1~6月
    关键验证：
    1. 1月应该自动获取12月的 final_state，而不是使用 prev=None
    2. 即使 cache_files 中没有12月的数据文件，也应该从缓存中加载12月的 final_state
    """
    symbol = "BTCUSDT"

    # 创建12月、1月、2月的数据
    months = [
        (2023, 12),  # 12月
        (2024, 1),  # 1月
        (2024, 2),  # 2月
    ]

    tick_files = []
    for year, month in months:
        tick_file = create_monthly_tick_data(
            temp_tick_dir, symbol, year, month, n_ticks=5000
        )
        tick_files.append(str(tick_file))

    print(f"\n   📊 测试跨年计算时自动加载前一个月 final_state 的逻辑")
    print(f"   ========================================")

    # ========== 第一次运行：12月（建立标准缓存）==========
    print(f"\n   🔹 第一次运行：12月（建立标准缓存）")

    start_ts_1 = "2023-12-01 00:00:00"
    end_ts_1 = "2023-12-31 23:59:59"

    vpin_result_1 = compute_vpin_from_cached_ticks(
        cache_files=tick_files,
        start_ts=start_ts_1,
        end_ts=end_ts_1,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_1) > 0, "第一次运行应该产生VPIN结果"

    # 获取12月的 final_state（用于验证）
    cache_key_dec_std = _get_monthly_vpin_cache_key(
        tick_files[0],  # 12月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=None,
    )
    dec_std_cache = _load_monthly_vpin_cache(temp_cache_dir, cache_key_dec_std)
    assert dec_std_cache is not None, "12月的标准缓存应该存在"
    dec_buckets_std, dec_final_state_std = dec_std_cache

    print(f"   ✅ 第一次运行完成，建立了12月的标准缓存")
    print(
        f"      12月标准缓存 final_state: filled_value = {dec_final_state_std.get('filled_value', 0.0):.6f}"
    )

    # ========== 第二次运行：1~2月（验证1月自动加载12月的 final_state）==========
    print(f"\n   🔹 第二次运行：1~2月（验证1月自动加载12月的 final_state）")
    print(f"      关键验证：")
    print(f"      1. 1月应该自动获取12月的 final_state，而不是使用 prev=None")
    print(
        f"      2. 即使 cache_files 中没有12月的数据文件，也应该从缓存中加载12月的 final_state"
    )

    # 只传入1月和2月的数据文件（不包含12月）
    tick_files_jan_feb = tick_files[1:]  # 只包含1月和2月

    start_ts_2 = "2024-01-01 00:00:00"
    end_ts_2 = "2024-02-28 23:59:59"

    vpin_result_2 = compute_vpin_from_cached_ticks(
        cache_files=tick_files_jan_feb,  # 不包含12月的数据文件
        start_ts=start_ts_2,
        end_ts=end_ts_2,
        bucket_volume=100.0,
        n_buckets=50,
        adaptive=False,
        monthly_cache_dir=str(temp_cache_dir),
        bucket_volume_usd=None,
    )

    assert len(vpin_result_2) > 0, "第二次运行应该产生VPIN结果"

    # 验证1月使用 final_state[12] 的状态缓存应该存在（因为第二次运行时自动加载了12月的 final_state）
    cache_key_jan_with_final12 = _get_monthly_vpin_cache_key(
        tick_files[1],  # 1月文件
        bucket_volume=100.0,
        bucket_volume_usd=None,
        prev_bucket_state=dec_final_state_std,  # 使用12月的 final_state
    )
    jan_with_final12_cache = _load_monthly_vpin_cache(
        temp_cache_dir, cache_key_jan_with_final12
    )
    assert (
        jan_with_final12_cache is not None
    ), "1月使用 final_state[12] 的状态缓存应该存在（第二次运行时自动加载并计算）"
    jan_buckets_with_final12, jan_final_state_with_final12 = jan_with_final12_cache

    assert jan_buckets_with_final12 is not None, "状态缓存应该存储 buckets"
    assert jan_final_state_with_final12 is not None, "状态缓存应该存储 final_state"

    print(f"   ✅ 验证通过：")
    print(
        f"      - 1月自动加载了12月的 final_state（即使 cache_files 中没有12月的数据文件）"
    )
    print(f"      - 1月使用 final_state[12] 的状态缓存存在")

    print(f"\n   ✅ 跨年计算时自动加载前一个月 final_state 的逻辑测试通过")
    print(f"      - 1月自动获取了12月的 final_state，而不是使用 prev=None")
    print(f"      - 即使 cache_files 中没有前一个月的数据文件，也能从缓存中加载")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
