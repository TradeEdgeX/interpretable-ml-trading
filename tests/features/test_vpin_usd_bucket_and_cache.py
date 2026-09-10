"""
VPIN USD bucket_volume 和缓存功能测试

测试内容：
1. USD bucket_volume 计算正确性
2. 多品种兼容性（BTC, ETH, ADA）
3. 缓存保存和加载
4. USD 模式 vs 传统模式对比
5. 集成测试（完整流程）
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
from src.features.time_series.utils_order_flow_features import (
    compute_vpin_from_ticks,
    extract_order_flow_features,
)


@pytest.fixture
def temp_cache_dir():
    """创建临时缓存目录"""
    cache_dir = Path(tempfile.mkdtemp())
    yield cache_dir
    shutil.rmtree(cache_dir, ignore_errors=True)


@pytest.fixture
def sample_ticks_btc():
    """创建 BTC 样本 tick 数据（高价格）"""
    np.random.seed(42)
    n = 2000

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1S")
    prices = 50000 + np.cumsum(np.random.randn(n) * 10)  # BTC 价格 ~50,000
    volumes = np.random.uniform(0.1, 5.0, n)  # BTC 数量
    sides = np.random.choice([1, -1], n, p=[0.52, 0.48])

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


@pytest.fixture
def sample_ticks_eth():
    """创建 ETH 样本 tick 数据（中等价格）"""
    np.random.seed(42)
    n = 2000

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1S")
    prices = 3000 + np.cumsum(np.random.randn(n) * 5)  # ETH 价格 ~3,000
    volumes = np.random.uniform(1.0, 50.0, n)  # ETH 数量
    sides = np.random.choice([1, -1], n, p=[0.52, 0.48])

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


@pytest.fixture
def sample_ticks_ada():
    """创建 ADA 样本 tick 数据（低价格）"""
    np.random.seed(42)
    n = 2000

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1S")
    prices = 1.0 + np.cumsum(np.random.randn(n) * 0.01)  # ADA 价格 ~1.0
    volumes = np.random.uniform(1000.0, 10000.0, n)  # ADA 数量
    sides = np.random.choice([1, -1], n, p=[0.52, 0.48])

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )


@pytest.fixture
def sample_tick_file(temp_cache_dir):
    """创建临时 tick 文件（Parquet 格式）"""
    np.random.seed(42)
    n = 2000

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n, freq="1S")
    prices = 50000 + np.cumsum(np.random.randn(n) * 10)
    volumes = np.random.uniform(0.1, 5.0, n)
    sides = np.random.choice([1, -1], n, p=[0.52, 0.48])

    ticks = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,
            "side": sides,
        }
    )

    tick_file = temp_cache_dir / "BTCUSDT_2024-01.parquet"
    ticks.to_parquet(tick_file, index=False)
    return str(tick_file)


class TestVPINUSDBucketVolume:
    """USD bucket_volume 计算测试"""

    def test_usd_bucket_volume_basic(self, sample_ticks_btc):
        """测试 USD bucket_volume 基本计算"""
        print("\n测试 USD bucket_volume 基本计算...")

        # 使用 USD bucket_volume
        bucket_volume_usd = 100000.0  # 10万 USD

        result = compute_vpin_from_ticks(
            sample_ticks_btc.set_index("timestamp"),
            bucket_volume_usd=bucket_volume_usd,
            n_buckets=10,
            adaptive=False,
        )

        # 验证返回类型
        assert isinstance(result, pd.DataFrame), "应返回 DataFrame"
        assert "vpin" in result.columns, "应包含 vpin 列"
        assert "signed_imbalance" in result.columns, "应包含 signed_imbalance 列"

        # 验证 VPIN 值范围 [0, 1]
        vpin_values = result["vpin"].dropna()
        if len(vpin_values) > 0:
            assert (vpin_values >= 0).all(), "VPIN 应 >= 0"
            assert (vpin_values <= 1).all(), "VPIN 应 <= 1"

        print(f"   ✅ USD bucket_volume 计算成功，生成 {len(result)} 个桶")

    def test_usd_bucket_volume_multi_asset_comparison(
        self, sample_ticks_btc, sample_ticks_eth, sample_ticks_ada
    ):
        """测试多品种使用相同 USD bucket_volume 的可比性"""
        print("\n测试多品种 USD bucket_volume 可比性...")

        bucket_volume_usd = 100000.0  # 10万 USD

        # 计算三个品种的 VPIN
        btc_result = compute_vpin_from_ticks(
            sample_ticks_btc.set_index("timestamp"),
            bucket_volume_usd=bucket_volume_usd,
            n_buckets=10,
            adaptive=False,
        )

        eth_result = compute_vpin_from_ticks(
            sample_ticks_eth.set_index("timestamp"),
            bucket_volume_usd=bucket_volume_usd,
            n_buckets=10,
            adaptive=False,
        )

        ada_result = compute_vpin_from_ticks(
            sample_ticks_ada.set_index("timestamp"),
            bucket_volume_usd=bucket_volume_usd,
            n_buckets=10,
            adaptive=False,
        )

        # 验证所有品种都生成了桶
        assert len(btc_result) > 0, "BTC 应生成桶"
        assert len(eth_result) > 0, "ETH 应生成桶"
        assert len(ada_result) > 0, "ADA 应生成桶"

        # 验证 VPIN 值范围一致（都基于相同的 USD bucket_volume）
        btc_vpin = btc_result["vpin"].dropna()
        eth_vpin = eth_result["vpin"].dropna()
        ada_vpin = ada_result["vpin"].dropna()

        if len(btc_vpin) > 0 and len(eth_vpin) > 0 and len(ada_vpin) > 0:
            # VPIN 值应该在相似范围内（都归一化到 [0, 1]）
            assert (btc_vpin >= 0).all() and (btc_vpin <= 1).all()
            assert (eth_vpin >= 0).all() and (eth_vpin <= 1).all()
            assert (ada_vpin >= 0).all() and (ada_vpin <= 1).all()

            # 均值应该在合理范围内（不应该因为价格差异而有巨大差异）
            btc_mean = btc_vpin.mean()
            eth_mean = eth_vpin.mean()
            ada_mean = ada_vpin.mean()

            print(f"   BTC VPIN mean: {btc_mean:.4f}")
            print(f"   ETH VPIN mean: {eth_mean:.4f}")
            print(f"   ADA VPIN mean: {ada_mean:.4f}")

            # 验证均值在合理范围内（不应该有数量级的差异）
            assert 0.1 < btc_mean < 0.9, "BTC VPIN 均值应在合理范围"
            assert 0.1 < eth_mean < 0.9, "ETH VPIN 均值应在合理范围"
            assert 0.1 < ada_mean < 0.9, "ADA VPIN 均值应在合理范围"

        print("   ✅ 多品种 USD bucket_volume 可比性测试通过")

    def test_usd_vs_traditional_bucket_volume(self, sample_ticks_btc):
        """测试 USD bucket_volume vs 传统 bucket_volume"""
        print("\n测试 USD vs 传统 bucket_volume...")

        # 传统方式：固定数量
        bucket_volume_traditional = 20.0  # 20 BTC

        # USD 方式：固定 USD 价值
        # 假设 BTC 价格 ~50,000，20 BTC = 1,000,000 USD
        bucket_volume_usd = 1000000.0  # 100万 USD

        result_traditional = compute_vpin_from_ticks(
            sample_ticks_btc.set_index("timestamp"),
            bucket_volume=bucket_volume_traditional,
            bucket_volume_usd=None,
            n_buckets=10,
            adaptive=False,
        )

        result_usd = compute_vpin_from_ticks(
            sample_ticks_btc.set_index("timestamp"),
            bucket_volume=None,
            bucket_volume_usd=bucket_volume_usd,
            n_buckets=10,
            adaptive=False,
        )

        # 验证两种方式都生成了结果
        assert len(result_traditional) > 0, "传统方式应生成桶"
        assert len(result_usd) > 0, "USD 方式应生成桶"

        # 验证 VPIN 值范围
        traditional_vpin = result_traditional["vpin"].dropna()
        usd_vpin = result_usd["vpin"].dropna()

        if len(traditional_vpin) > 0 and len(usd_vpin) > 0:
            assert (traditional_vpin >= 0).all() and (traditional_vpin <= 1).all()
            assert (usd_vpin >= 0).all() and (usd_vpin <= 1).all()

        print(f"   传统方式生成 {len(result_traditional)} 个桶")
        print(f"   USD 方式生成 {len(result_usd)} 个桶")
        print("   ✅ USD vs 传统 bucket_volume 测试通过")


class TestVPINUSDCache:
    """USD bucket_volume 缓存测试"""

    def test_usd_cache_key_generation(self, temp_cache_dir):
        """测试 USD 模式缓存键生成"""
        print("\n测试 USD 模式缓存键生成...")

        file_path = "/path/to/BTCUSDT_2024-01.parquet"
        bucket_volume = 100.0
        bucket_volume_usd = 1000000.0
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        # 传统模式缓存键（无 USD）
        key_traditional = _get_monthly_vpin_cache_key(
            file_path, bucket_volume, bucket_volume_usd=None
        )

        # USD 模式缓存键
        key_usd = _get_monthly_vpin_cache_key(
            file_path, bucket_volume, bucket_volume_usd=bucket_volume_usd
        )

        # 验证缓存键不同
        assert key_traditional != key_usd, "USD 模式和传统模式的缓存键应不同"

        # 验证 USD 模式缓存键不同（因为缓存键是 MD5 哈希，我们验证它们不同即可）
        # 注意：缓存键是 MD5 哈希，所以不包含 "usd" 字符串，但 key_str 中包含
        # 我们验证缓存键确实不同即可

        print(f"   传统模式缓存键: {key_traditional[:20]}...")
        print(f"   USD 模式缓存键: {key_usd[:20]}...")
        print("   ✅ USD 模式缓存键生成测试通过")

    def test_usd_cache_save_and_load(self, temp_cache_dir, sample_tick_file):
        """测试 USD 模式缓存保存和加载"""
        print("\n测试 USD 模式缓存保存和加载...")

        bucket_volume_usd = 100000.0
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        # 计算并保存缓存
        cache_key = _get_monthly_vpin_cache_key(
            sample_tick_file, 100.0, bucket_volume_usd=bucket_volume_usd
        )

        # 计算 buckets（不传 start/end，函数会计算整个月份）
        buckets, final_state = _compute_vpin_buckets_for_month(
            Path(sample_tick_file),
            bucket_volume=100.0,
            bucket_volume_usd=bucket_volume_usd,
        )

        # 保存缓存
        _save_monthly_vpin_cache(temp_cache_dir, cache_key, buckets)

        # 验证缓存文件存在
        cache_file = temp_cache_dir / f"{cache_key}.pkl"
        assert cache_file.exists(), "缓存文件应存在"

        # 加载缓存（返回 (buckets, final_state) tuple）
        loaded_result = _load_monthly_vpin_cache(temp_cache_dir, cache_key)

        # 验证加载的数据正确
        assert loaded_result is not None, "应能加载缓存"
        loaded_buckets, loaded_final_state = loaded_result
        assert len(loaded_buckets) == len(buckets), "缓存数据长度应一致"

        # 验证数据内容一致
        for i, (ts1, vpin1) in enumerate(buckets):
            ts2, vpin2 = loaded_buckets[i]
            assert ts1 == ts2, f"时间戳应一致: {i}"
            assert abs(vpin1 - vpin2) < 1e-9, f"VPIN 值应一致: {i}"

        print(f"   保存了 {len(buckets)} 个 buckets")
        print(f"   加载了 {len(loaded_buckets)} 个 buckets")
        print("   ✅ USD 模式缓存保存和加载测试通过")

    def test_usd_cache_incremental_computation(self, temp_cache_dir, sample_tick_file):
        """测试 USD 模式增量计算（缓存命中）"""
        print("\n测试 USD 模式增量计算...")

        bucket_volume_usd = 100000.0
        start = pd.Timestamp("2024-01-01")
        end = pd.Timestamp("2024-01-31")

        # 第一次计算（应该计算）
        cache_files = [sample_tick_file]
        result1 = compute_vpin_from_cached_ticks(
            cache_files=cache_files,
            start_ts=start.isoformat(),
            end_ts=end.isoformat(),
            bucket_volume=100.0,
            n_buckets=10,
            adaptive=False,
            monthly_cache_dir=str(temp_cache_dir),
            bucket_volume_usd=bucket_volume_usd,
        )

        # 第二次计算（应该使用缓存）
        result2 = compute_vpin_from_cached_ticks(
            cache_files=cache_files,
            start_ts=start.isoformat(),
            end_ts=end.isoformat(),
            bucket_volume=100.0,
            n_buckets=10,
            adaptive=False,
            monthly_cache_dir=str(temp_cache_dir),
            bucket_volume_usd=bucket_volume_usd,
        )

        # 验证结果一致
        assert len(result1) == len(result2), "结果长度应一致"
        if len(result1) > 0:
            pd.testing.assert_series_equal(
                result1, result2, check_names=False, rtol=1e-9
            )

        print(f"   第一次计算生成 {len(result1)} 个桶")
        print(f"   第二次计算（使用缓存）生成 {len(result2)} 个桶")
        print("   ✅ USD 模式增量计算测试通过")


class TestVPINUSDIntegration:
    """USD bucket_volume 集成测试"""

    def test_extract_order_flow_features_with_usd_bucket(self, sample_ticks_btc):
        """测试 extract_order_flow_features 使用 USD bucket_volume"""
        print("\n测试 extract_order_flow_features USD bucket_volume...")

        # 创建 OHLCV 数据
        timestamps = pd.date_range("2024-01-01 00:00:00", periods=100, freq="1min")
        df = pd.DataFrame(
            {
                "open": 50000 + np.random.randn(100) * 50,
                "high": 50000 + np.abs(np.random.randn(100) * 50),
                "low": 50000 - np.abs(np.random.randn(100) * 50),
                "close": 50000 + np.random.randn(100) * 50,
                "volume": np.random.uniform(100, 1000, 100),
            },
            index=timestamps,
        )

        # 准备 tick 数据
        ticks = sample_ticks_btc.set_index("timestamp")

        # 测试使用 USD bucket_volume
        result_usd = extract_order_flow_features(
            df,
            ticks=ticks,
            vpin_bucket_volume=None,
            vpin_bucket_volume_usd=100000.0,  # 10万 USD
            vpin_n_buckets=10,
            vpin_adaptive=False,
            monthly_cache_dir=None,  # 不使用缓存（内存计算）
            freq="1min",  # 明确指定频率
        )

        # 验证结果
        assert "vpin" in result_usd.columns, "应包含 vpin 列"

        vpin_values = result_usd["vpin"].dropna()
        if len(vpin_values) > 0:
            assert (vpin_values >= 0).all() and (vpin_values <= 1).all()

        print(f"   生成了 {len(vpin_values)} 个 VPIN 值（USD 模式）")
        print("   ✅ extract_order_flow_features USD bucket_volume 测试通过")


def run_all_tests():
    """运行所有测试"""
    print("=" * 70)
    print("VPIN USD bucket_volume 和缓存功能测试")
    print("=" * 70)

    # 创建测试实例
    test_usd = TestVPINUSDBucketVolume()
    test_cache = TestVPINUSDCache()
    test_integration = TestVPINUSDIntegration()

    # 创建 fixtures
    temp_cache_dir = Path(tempfile.mkdtemp())
    try:
        sample_ticks_btc = test_usd.sample_ticks_btc()
        sample_ticks_eth = test_usd.sample_ticks_eth()
        sample_ticks_ada = test_usd.sample_ticks_ada()
        sample_tick_file = str(temp_cache_dir / "BTCUSDT_2024-01.parquet")
        sample_ticks_btc.to_parquet(sample_tick_file, index=False)

        # 运行测试
        tests = [
            (
                "USD bucket_volume 基本计算",
                test_usd.test_usd_bucket_volume_basic,
                [sample_ticks_btc],
            ),
            (
                "多品种可比性",
                test_usd.test_usd_bucket_volume_multi_asset_comparison,
                [sample_ticks_btc, sample_ticks_eth, sample_ticks_ada],
            ),
            (
                "USD vs 传统",
                test_usd.test_usd_vs_traditional_bucket_volume,
                [sample_ticks_btc],
            ),
            (
                "USD 缓存键生成",
                test_cache.test_usd_cache_key_generation,
                [temp_cache_dir],
            ),
            (
                "USD 缓存保存加载",
                test_cache.test_usd_cache_save_and_load,
                [temp_cache_dir, sample_tick_file],
            ),
            (
                "USD 增量计算",
                test_cache.test_usd_cache_incremental_computation,
                [temp_cache_dir, sample_tick_file],
            ),
            (
                "集成测试",
                test_integration.test_extract_order_flow_features_with_usd_bucket,
                [sample_ticks_btc],
            ),
        ]

        passed = 0
        failed = 0

        for name, test_func, args in tests:
            try:
                print(f"\n{'='*70}")
                print(f"运行测试: {name}")
                print(f"{'='*70}")
                test_func(*args)
                passed += 1
                print(f"✅ {name} 通过")
            except Exception as e:
                failed += 1
                print(f"❌ {name} 失败: {e}")
                import traceback

                traceback.print_exc()

        print(f"\n{'='*70}")
        print(f"测试结果: {passed} 通过, {failed} 失败")
        print(f"{'='*70}")

    finally:
        shutil.rmtree(temp_cache_dir, ignore_errors=True)


if __name__ == "__main__":
    run_all_tests()
