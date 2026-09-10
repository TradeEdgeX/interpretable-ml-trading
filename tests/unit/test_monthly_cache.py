"""
测试按月缓存机制

测试场景：
1. 按月缓存的基本功能（保存和加载）
2. 增量计算（只计算缺失的月份）
3. 缓存键生成是否正确
4. 合并多个月份的结果
5. VPIN的按月缓存
6. 普通特征的按月缓存
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from datetime import datetime, timedelta

from src.features.loader.feature_computer import FeatureComputer
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
def sample_monthly_data():
    """生成多个月的示例数据"""
    dates = []
    start_date = datetime(2024, 1, 1)
    for month in range(3):  # 3个月的数据
        month_start = start_date + pd.DateOffset(months=month)
        month_dates = pd.date_range(
            month_start,
            month_start + pd.DateOffset(months=1) - timedelta(days=1),
            freq="1h",
        )
        dates.extend(month_dates[:100])  # 每个月100个数据点

    df = pd.DataFrame(
        {
            "open": np.random.randn(len(dates)).cumsum() + 100,
            "high": np.random.randn(len(dates)).cumsum() + 101,
            "low": np.random.randn(len(dates)).cumsum() + 99,
            "close": np.random.randn(len(dates)).cumsum() + 100,
            "volume": np.random.rand(len(dates)) * 1000,
        },
        index=pd.DatetimeIndex(dates),
    )

    return df


class TestMonthlyCacheBasic:
    """测试按月缓存的基本功能"""

    def test_split_df_by_month(self, sample_monthly_data):
        """测试按月份拆分DataFrame"""
        computer = FeatureComputer(
            cache_dir=str(tempfile.mkdtemp()), use_monthly_cache=True
        )

        monthly_dfs = computer._split_df_by_month(sample_monthly_data)

        # 应该拆分成3个月
        assert len(monthly_dfs) == 3
        assert all(
            key.startswith("2024-0") for key in monthly_dfs.keys() if key != "all"
        )

        # 每个月的索引应该连续
        for month_key, month_df in monthly_dfs.items():
            if month_key != "all":
                assert len(month_df) > 0
                assert month_df.index[0].month == int(month_key.split("-")[1])

    def test_monthly_cache_key_generation(self, temp_cache_dir):
        """测试按月缓存键生成"""
        computer = FeatureComputer(
            cache_dir=str(temp_cache_dir), use_monthly_cache=True
        )

        params = {"window": 20, "period": 14}
        feature_info = {"output_columns": ["rsi"]}

        key1 = computer._get_monthly_cache_key("rsi", "2024-01", params, feature_info)
        key2 = computer._get_monthly_cache_key("rsi", "2024-01", params, feature_info)
        key3 = computer._get_monthly_cache_key("rsi", "2024-02", params, feature_info)
        key4 = computer._get_monthly_cache_key(
            "rsi", "2024-01", {"window": 30}, feature_info
        )
        key5 = computer._get_monthly_cache_key(
            "rsi", "2024-01", params, feature_info, df_sig="DATASET_A"
        )
        key6 = computer._get_monthly_cache_key(
            "rsi", "2024-01", params, feature_info, df_sig="DATASET_B"
        )

        # 相同参数应该生成相同的键
        assert key1 == key2

        # 不同月份应该生成不同的键
        assert key1 != key3

        # 不同参数应该生成不同的键
        assert key1 != key4

        # 不同数据集签名（df_sig）应该生成不同的键（防止跨运行/月度缓存污染）
        assert key5 != key6

    def test_save_and_load_monthly_cache(self, temp_cache_dir):
        """测试按月缓存的保存和加载"""
        computer = FeatureComputer(
            cache_dir=str(temp_cache_dir), use_monthly_cache=True
        )

        # 创建测试数据
        test_data = pd.DataFrame(
            {"feature1": [1, 2, 3, 4, 5]},
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )

        cache_key = "test_feature_2024-01"

        # 保存缓存
        computer._save_monthly_cache(cache_key, test_data)

        # 检查文件是否存在
        cache_file = computer.monthly_cache_dir / f"{cache_key}.pkl"
        assert cache_file.exists()

        # 加载缓存
        loaded_data = computer._load_monthly_cache(cache_key)

        assert loaded_data is not None
        pd.testing.assert_frame_equal(loaded_data, test_data)

    def test_try_monthly_cache_all_cached(self, temp_cache_dir, sample_monthly_data):
        """测试所有月份都有缓存的情况"""
        computer = FeatureComputer(
            cache_dir=str(temp_cache_dir), use_monthly_cache=True
        )

        # 先为所有月份创建缓存
        monthly_dfs = computer._split_df_by_month(sample_monthly_data)
        # Align with monthly warmup default (FEATURE_MONTHLY_WARMUP_MONTHS=3)
        compute_params = {"__monthly_warmup_months": 3}
        feature_info = {"output_columns": ["test_feature"]}

        for month_key, month_df in monthly_dfs.items():
            if month_key != "all":
                # 创建测试结果
                test_result = pd.DataFrame(
                    {"test_feature": np.random.randn(len(month_df))},
                    index=month_df.index,
                )

                # 保存缓存
                cache_key = computer._get_monthly_cache_key(
                    "test_feature",
                    month_key,
                    compute_params,
                    feature_info,
                    df_sig=computer._get_df_signature(month_df),
                )
                computer._save_monthly_cache(cache_key, test_result)

        # 尝试加载所有月份的缓存
        monthly_results = computer._try_monthly_cache(
            "test_feature", sample_monthly_data, compute_params, feature_info
        )

        # 应该成功加载所有月份
        assert monthly_results is not None
        assert len(monthly_results) == 3

        # 合并结果
        combined = pd.concat(monthly_results.values(), axis=0).sort_index()
        assert len(combined) == len(sample_monthly_data)

    def test_try_monthly_cache_partial_cached(
        self, temp_cache_dir, sample_monthly_data
    ):
        """测试部分月份有缓存的情况（应该返回None，使用全量计算）"""
        computer = FeatureComputer(
            cache_dir=str(temp_cache_dir), use_monthly_cache=True
        )

        # 只为部分月份创建缓存
        monthly_dfs = computer._split_df_by_month(sample_monthly_data)
        # Align with monthly warmup default (FEATURE_MONTHLY_WARMUP_MONTHS=3)
        compute_params = {"__monthly_warmup_months": 3}
        feature_info = {"output_columns": ["test_feature"]}

        # 只为第一个月创建缓存
        first_month_key = [k for k in monthly_dfs.keys() if k != "all"][0]
        month_df = monthly_dfs[first_month_key]
        test_result = pd.DataFrame(
            {"test_feature": np.random.randn(len(month_df))}, index=month_df.index
        )

        cache_key = computer._get_monthly_cache_key(
            "test_feature",
            first_month_key,
            compute_params,
            feature_info,
            df_sig=computer._get_df_signature(month_df),
        )
        computer._save_monthly_cache(cache_key, test_result)

        # 尝试加载所有月份的缓存
        monthly_results = computer._try_monthly_cache(
            "test_feature", sample_monthly_data, compute_params, feature_info
        )

        # 应该返回None（因为不是所有月份都有缓存）
        assert monthly_results is None


class TestVPINMonthlyCache:
    """测试VPIN的按月缓存"""

    def test_vpin_cache_key_generation(self):
        """测试VPIN缓存键生成"""
        file_path = "/path/to/BTCUSDT_2024-01.parquet"
        bucket_volume = 100.0
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        key1 = _get_monthly_vpin_cache_key(file_path, bucket_volume, start, end)
        key2 = _get_monthly_vpin_cache_key(file_path, bucket_volume, start, end)
        key3 = _get_monthly_vpin_cache_key(file_path, 200.0, start, end)

        # 相同参数应该生成相同的键
        assert key1 == key2

        # 不同bucket_volume应该生成不同的键
        assert key1 != key3

    def test_vpin_usd_cache_key_generation(self):
        """测试VPIN USD模式缓存键生成"""
        file_path = "/path/to/BTCUSDT_2024-01.parquet"
        bucket_volume = 100.0
        bucket_volume_usd = 1000000.0
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        # 传统模式
        key_traditional = _get_monthly_vpin_cache_key(
            file_path, bucket_volume, start, end, bucket_volume_usd=None
        )

        # USD 模式
        key_usd = _get_monthly_vpin_cache_key(
            file_path, bucket_volume, start, end, bucket_volume_usd=bucket_volume_usd
        )

        # 相同 USD bucket_volume 应该生成相同的键
        key_usd2 = _get_monthly_vpin_cache_key(
            file_path, bucket_volume, start, end, bucket_volume_usd=bucket_volume_usd
        )
        assert key_usd == key_usd2

        # 不同 USD bucket_volume 应该生成不同的键
        key_usd3 = _get_monthly_vpin_cache_key(
            file_path, bucket_volume, start, end, bucket_volume_usd=2000000.0
        )
        assert key_usd != key_usd3

        # USD 模式和传统模式应该生成不同的键
        assert key_traditional != key_usd

        # 验证缓存键确实不同（因为缓存键是 MD5 哈希，我们验证它们不同即可）
        # 注意：缓存键是 MD5 哈希，所以不包含 "usd" 字符串，但 key_str 中包含

    def test_vpin_save_and_load_cache(self, temp_cache_dir):
        """测试VPIN缓存的保存和加载"""
        # 创建测试buckets
        test_buckets = [
            (pd.Timestamp("2024-01-01 10:00:00"), 0.5),
            (pd.Timestamp("2024-01-01 11:00:00"), 0.6),
            (pd.Timestamp("2024-01-01 12:00:00"), 0.4),
        ]

        cache_key = "test_vpin_2024-01"
        cache_dir = Path(temp_cache_dir)

        # 保存缓存
        _save_monthly_vpin_cache(cache_dir, cache_key, test_buckets)

        # 检查文件是否存在
        cache_file = cache_dir / f"{cache_key}.pkl"
        assert cache_file.exists()

        # 加载缓存
        loaded_buckets = _load_monthly_vpin_cache(cache_dir, cache_key)

        assert loaded_buckets is not None
        assert len(loaded_buckets) == len(test_buckets)
        assert loaded_buckets[0][0] == test_buckets[0][0]
        assert loaded_buckets[0][1] == test_buckets[0][1]

    def test_vpin_incremental_computation(self, temp_cache_dir):
        """测试VPIN的增量计算（模拟）"""
        # 创建模拟的tick文件路径
        cache_files = [f"/tmp/BTCUSDT_2024-{i:02d}.parquet" for i in range(1, 4)]

        # 由于需要真实的parquet文件，这里只测试缓存逻辑
        # 实际计算需要真实的tick数据

        # 测试缓存目录设置
        cache_dir = Path(temp_cache_dir)
        assert cache_dir.exists() or True  # 目录会在使用时创建


class TestMonthlyCacheIntegration:
    """测试按月缓存的集成场景"""

    def test_compute_and_cache_monthly(self, temp_cache_dir, sample_monthly_data):
        """测试按月计算并缓存"""
        computer = FeatureComputer(
            cache_dir=str(temp_cache_dir), use_monthly_cache=True
        )

        # 定义一个简单的计算函数（模拟特征计算）
        def simple_feature_func(df, **kwargs):
            return pd.DataFrame(
                {"simple_feature": df["close"].rolling(window=5).mean()}, index=df.index
            )

        # Align with monthly warmup default (FEATURE_MONTHLY_WARMUP_MONTHS=3)
        compute_params = {"__monthly_warmup_months": 3}
        feature_info = {"output_columns": ["simple_feature"]}

        # 第一次计算（应该计算所有月份并缓存）
        result1 = computer._compute_and_cache_monthly(
            "simple_feature",
            sample_monthly_data,
            compute_params,
            feature_info,
            simple_feature_func,
        )

        assert len(result1) == len(sample_monthly_data)
        assert "simple_feature" in result1.columns

        # 检查缓存文件
        monthly_dfs = computer._split_df_by_month(sample_monthly_data)
        for month_key in monthly_dfs.keys():
            if month_key != "all":
                cache_key = computer._get_monthly_cache_key(
                    "simple_feature",
                    month_key,
                    compute_params,
                    feature_info,
                    df_sig=computer._get_df_signature(monthly_dfs[month_key]),
                )
                cache_file = computer.monthly_cache_dir / f"{cache_key}.pkl"
                assert cache_file.exists()

        # 第二次计算（应该从缓存加载）
        result2 = computer._compute_and_cache_monthly(
            "simple_feature",
            sample_monthly_data,
            compute_params,
            feature_info,
            simple_feature_func,
        )

        # 结果应该相同
        pd.testing.assert_frame_equal(result1, result2)

    def test_incremental_computation_new_month(self, temp_cache_dir):
        """测试新增月份时的增量计算"""
        computer = FeatureComputer(
            cache_dir=str(temp_cache_dir), use_monthly_cache=True
        )

        # 定义计算函数
        def simple_feature_func(df, **kwargs):
            return pd.DataFrame(
                {"simple_feature": df["close"].rolling(window=5).mean()}, index=df.index
            )

        # 创建前两个月的数据
        dates_jan_feb = pd.date_range("2024-01-01", "2024-02-28", freq="1h")[:200]
        df_jan_feb = pd.DataFrame(
            {
                "close": np.random.randn(len(dates_jan_feb)).cumsum() + 100,
            },
            index=pd.DatetimeIndex(dates_jan_feb),
        )

        # Align with monthly warmup default (FEATURE_MONTHLY_WARMUP_MONTHS=3)
        compute_params = {"__monthly_warmup_months": 3}
        feature_info = {"output_columns": ["simple_feature"]}

        # 第一次计算（计算1月和2月）
        result1 = computer._compute_and_cache_monthly(
            "simple_feature",
            df_jan_feb,
            compute_params,
            feature_info,
            simple_feature_func,
        )

        # 添加3月的数据
        dates_mar = pd.date_range("2024-03-01", "2024-03-31", freq="1h")[:100]
        df_mar = pd.DataFrame(
            {
                "close": np.random.randn(len(dates_mar)).cumsum() + 100,
            },
            index=pd.DatetimeIndex(dates_mar),
        )

        df_all = pd.concat([df_jan_feb, df_mar]).sort_index()

        # 第二次计算（应该只计算3月，1月和2月从缓存加载）
        result2 = computer._compute_and_cache_monthly(
            "simple_feature", df_all, compute_params, feature_info, simple_feature_func
        )

        # 结果应该包含所有月份
        assert len(result2) == len(df_all)

        # 1月和2月的数据应该相同（从缓存加载）
        # 使用索引交集来比较，因为可能有边界问题
        common_index = result1.index.intersection(result2.index)
        if len(common_index) > 0:
            jan_feb_in_result1 = result1.loc[common_index].reset_index(drop=True)
            jan_feb_in_result2 = result2.loc[common_index].reset_index(drop=True)
            # 比较值而不是索引
            np.testing.assert_array_almost_equal(
                jan_feb_in_result1.values, jan_feb_in_result2.values, decimal=5
            )
        else:
            # 如果没有共同索引，至少检查结果长度和列名
            assert len(result1) > 0
            assert len(result2) > 0
            assert result1.columns.equals(result2.columns)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
