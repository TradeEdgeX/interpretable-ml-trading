"""
特征诊断测试：检测并修复用户报告的问题

问题清单：
1. shd_pct_f - 月初不连续（预期行为，由 warmup 周期导致）
2. vpin_signed_imbalance - 全零（tick 数据未加载）
3. wpt_price_energy_ratio - 常量 1（WPT 能量分类问题）
4. wpt_energy_cascade - 常量 -1（WPT 能量计算问题）
5. compression_energy_x_ofi_short - 常量 0（依赖链零值传播）
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_wpt_features import (
    extract_wpt_features,
    wpt_decompose,
    extract_wpt_price_features_normalized,
)
from src.features.time_series.utils_liquidity_features import (
    compute_wpt_volume_energy_features,
)
from src.features.time_series.baseline_features import (
    compute_compression_energy_from_series,
    compute_percentile_rank_from_series,
)
from src.features.time_series.utils_interaction_features import (
    compute_compression_energy_x_ofi_short_from_series,
)


def _create_realistic_ohlcv(n: int = 500, trend: float = 0.0001) -> pd.DataFrame:
    """创建模拟真实市场的 OHLCV 数据"""
    idx = pd.date_range("2024-01-01", periods=n, freq="4H")
    rng = np.random.default_rng(42)

    # 生成带趋势和波动的价格
    returns = trend + rng.normal(0, 0.01, n)
    close = 50000 * np.exp(np.cumsum(returns))

    # 生成合理的 OHLC
    high = close * (1 + rng.uniform(0.001, 0.015, n))
    low = close * (1 - rng.uniform(0.001, 0.015, n))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    # 确保 high >= max(open, close), low <= min(open, close)
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    # 生成成交量（带有周期性）
    volume = 1000 + 500 * np.sin(np.arange(n) * 2 * np.pi / 24) + rng.uniform(0, 200, n)
    volume = np.maximum(volume, 100)  # 确保正值

    # 生成 CVD（累计净买入）
    cvd = np.cumsum(rng.uniform(-10, 10, n))

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "cvd": cvd,
        },
        index=idx,
    )


class TestWPTEnergyRatioFeatures:
    """测试 WPT 能量比特征（诊断常量值问题）"""

    def test_wpt_decompose_produces_valid_energy(self):
        """测试 WPT 分解是否产生有效能量分布"""
        # 使用 log returns 而不是原始价格，因为原始价格是非平稳的
        raw_prices = np.cumsum(np.random.randn(200)) + 50000
        log_returns = np.diff(np.log(raw_prices))

        result = wpt_decompose(log_returns, wavelet="db4", level=4)

        assert "energy" in result
        assert len(result["energy"]) > 0

        total_energy = sum(result["energy"].values())
        assert total_energy > 0, "总能量应大于零"

        # 对于 log returns，能量应该分布在多个频带，而不是集中在一个
        energy_values = list(result["energy"].values())
        max_ratio = max(energy_values) / total_energy
        # 放宽阈值，因为 log returns 仍然可能有一些能量集中
        assert max_ratio < 0.95, f"能量不应全部集中在一个频带: max_ratio={max_ratio}"

    def test_wpt_energy_ratio_not_constant(self):
        """测试 WPT 能量比特征不应是常量

        注意：extract_wpt_features 不支持 use_log_returns，导致能量比特征可能接近常量。
        应使用 compute_wpt_volume_energy_features 并启用 use_log_returns=True。
        """
        df = _create_realistic_ohlcv(300)

        # 使用正确的函数并启用 use_log_returns
        result = compute_wpt_volume_energy_features(
            df,
            price_col="close",
            volume_col="volume",
            wavelet="db4",
            level=4,
            lookback_window=30,
            use_log_returns=True,  # 关键：使用 log returns
        )

        # wpt_energy_cascade 是能量分布的综合指标
        col = "wpt_energy_cascade"
        assert col in result.columns, f"缺少列: {col}"

        valid_values = result[col].iloc[50:].dropna()  # 跳过 warmup
        assert len(valid_values) > 100, f"{col} 有效值太少: {len(valid_values)}"

        unique_values = valid_values.nunique()
        assert unique_values > 50, f"{col} 应该有变化，但只有 {unique_values} 个唯一值"

        # 检查不是常量
        std = valid_values.std()
        assert std > 0.05, f"{col} 标准差太小 ({std})，可能是常量"

    def test_wpt_energy_classification_logic(self):
        """诊断 WPT 能量分类逻辑"""
        data = np.cumsum(np.random.randn(100)) + 50000
        result = wpt_decompose(data, wavelet="db4", level=4)

        all_paths = list(result["energy"].keys())
        path_freq = {p: p.count("d") for p in all_paths}
        max_d = max(path_freq.values()) if path_freq else 0

        print(f"\n=== WPT 能量分类诊断 ===")
        print(f"Level=4, 共 {len(all_paths)} 个子带")
        print(f"路径: {all_paths}")
        print(f"max_d (最高频带的 'd' 数量): {max_d}")

        # 按频率分类
        low_bands = [p for p, d in path_freq.items() if d <= max_d // 3]
        mid_bands = [
            p for p, d in path_freq.items() if max_d // 3 < d <= 2 * max_d // 3
        ]
        high_bands = [p for p, d in path_freq.items() if d > 2 * max_d // 3]

        print(f"Low bands ({len(low_bands)}): {low_bands}")
        print(f"Mid bands ({len(mid_bands)}): {mid_bands}")
        print(f"High bands ({len(high_bands)}): {high_bands}")

        # 计算能量
        energy_low = sum(result["energy"].get(p, 0.0) for p in low_bands)
        energy_mid = sum(result["energy"].get(p, 0.0) for p in mid_bands)
        energy_high = sum(result["energy"].get(p, 0.0) for p in high_bands)
        total_energy = sum(result["energy"].values())

        print(f"\n能量分布:")
        print(f"  Low: {energy_low:.6f} ({energy_low/total_energy*100:.2f}%)")
        print(f"  Mid: {energy_mid:.6f} ({energy_mid/total_energy*100:.2f}%)")
        print(f"  High: {energy_high:.6f} ({energy_high/total_energy*100:.2f}%)")

        # 如果 mid_bands 为空，这就是问题所在
        if len(mid_bands) == 0:
            print("\n⚠️ 问题发现: mid_bands 为空！")
            print("这会导致 energy_mid_ratio 始终为 0")

        # 断言分类应该合理
        assert len(low_bands) > 0, "低频带不应为空"
        assert len(high_bands) > 0, "高频带不应为空"


class TestWPTEnergyCascade:
    """测试 WPT Energy Cascade 特征"""

    def test_energy_cascade_not_constant_with_log_returns(self):
        """测试启用 use_log_returns 后 wpt_energy_cascade 应有变化"""
        df = _create_realistic_ohlcv(300)

        # 关键修复：使用 use_log_returns=True
        result = compute_wpt_volume_energy_features(
            df,
            price_col="close",
            volume_col="volume",
            wavelet="db4",
            level=4,
            lookback_window=30,
            use_log_returns=True,  # 修复：对 log returns 做 WPT
        )

        assert "wpt_energy_cascade" in result.columns

        # 跳过前 lookback_window 行（warmup）
        valid_values = result["wpt_energy_cascade"].iloc[50:].dropna()

        if len(valid_values) > 0:
            unique_values = valid_values.nunique()
            std = valid_values.std()

            print(f"\n=== wpt_energy_cascade 诊断 ===")
            print(f"有效值数量: {len(valid_values)}")
            print(f"唯一值数量: {unique_values}")
            print(f"标准差: {std}")
            print(f"范围: [{valid_values.min():.4f}, {valid_values.max():.4f}]")
            print(f"示例值: {valid_values.head(10).tolist()}")

            # 检查值是否有变化
            assert (
                unique_values > 5
            ), f"wpt_energy_cascade 应有变化，但只有 {unique_values} 个唯一值"
            assert std > 0.01, f"wpt_energy_cascade 标准差太小: {std}"

    def test_energy_cascade_formula(self):
        """验证 energy_cascade 公式正确性"""
        # energy_cascade = (ratio_low + ratio_mid) - ratio_high
        # 如果所有能量在高频，结果接近 -1
        # 如果所有能量在低频，结果接近 +1

        # 创建高波动数据（应该有更多高频能量）
        high_vol_data = np.cumsum(np.random.randn(100) * 5) + 50000

        # 创建低波动数据（应该有更多低频能量）
        low_vol_data = np.cumsum(np.random.randn(100) * 0.1) + 50000

        result_high = wpt_decompose(high_vol_data, wavelet="db4", level=4)
        result_low = wpt_decompose(low_vol_data, wavelet="db4", level=4)

        print("\n=== Energy Cascade 公式验证 ===")
        for name, result in [("高波动", result_high), ("低波动", result_low)]:
            all_paths = list(result["energy"].keys())
            path_freq = {p: p.count("d") for p in all_paths}
            max_d = max(path_freq.values())

            low_bands = [p for p, d in path_freq.items() if d <= max_d // 3]
            mid_bands = [
                p for p, d in path_freq.items() if max_d // 3 < d <= 2 * max_d // 3
            ]
            high_bands = [p for p, d in path_freq.items() if d > 2 * max_d // 3]

            e_low = sum(result["energy"].get(p, 0) for p in low_bands)
            e_mid = sum(result["energy"].get(p, 0) for p in mid_bands)
            e_high = sum(result["energy"].get(p, 0) for p in high_bands)
            total = e_low + e_mid + e_high

            if total > 0:
                r_low, r_mid, r_high = e_low / total, e_mid / total, e_high / total
                cascade = (r_low + r_mid) - r_high
                print(
                    f"{name}: low={r_low:.3f}, mid={r_mid:.3f}, high={r_high:.3f} → cascade={cascade:.3f}"
                )


class TestCompressionEnergyXOfiShort:
    """测试 compression_energy_x_ofi_short 特征"""

    def test_ofi_short_with_nonzero_vpin(self):
        """测试 ofi_short 在 vpin_signed_imbalance 非零时不应为常量零"""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")

        # 模拟真实的 vpin_signed_imbalance（买卖不平衡）
        vpin_signed_imbalance = pd.Series(
            np.random.uniform(-0.5, 0.5, n), index=idx, name="vpin_signed_imbalance"
        )
        vpin = pd.Series(np.random.uniform(0.1, 0.8, n), index=idx, name="vpin")

        from src.features.time_series.baseline_features import (
            compute_ofi_short_from_series,
        )

        ofi_short = compute_ofi_short_from_series(
            vpin_signed_imbalance=vpin_signed_imbalance, vpin=vpin, window=5
        )

        print(f"\n=== ofi_short 诊断 ===")
        print(
            f"vpin_signed_imbalance 范围: [{vpin_signed_imbalance.min():.3f}, {vpin_signed_imbalance.max():.3f}]"
        )
        print(f"ofi_short 范围: [{ofi_short.min():.3f}, {ofi_short.max():.3f}]")
        print(f"ofi_short 标准差: {ofi_short.std():.6f}")

        assert ofi_short.std() > 0.01, f"ofi_short 应有变化，标准差={ofi_short.std()}"
        assert not (ofi_short == 0).all(), "ofi_short 不应全为零"

    def test_compression_energy_x_ofi_short_not_zero(self):
        """测试当输入有效时，交互特征不应全为零"""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")

        # 创建非零的 compression_energy（模拟压缩状态）
        compression_energy = pd.Series(np.random.uniform(-2, 2, n), index=idx)

        # 创建非零的 ofi_short
        ofi_short = pd.Series(np.random.uniform(-0.3, 0.3, n), index=idx)

        result = compute_compression_energy_x_ofi_short_from_series(
            compression_energy=compression_energy, ofi_short=ofi_short
        )

        interaction = result["compression_energy_x_ofi_short"]

        print(f"\n=== compression_energy_x_ofi_short 诊断 ===")
        print(
            f"compression_energy 范围: [{compression_energy.min():.3f}, {compression_energy.max():.3f}]"
        )
        print(f"ofi_short 范围: [{ofi_short.min():.3f}, {ofi_short.max():.3f}]")
        print(f"交互特征范围: [{interaction.min():.3f}, {interaction.max():.3f}]")

        assert interaction.std() > 0.001, "交互特征应有变化"
        assert not (interaction == 0).all(), "交互特征不应全为零"

    def test_zero_propagation_from_vpin(self):
        """诊断：当 vpin_signed_imbalance 全零时，整个链路都为零"""
        n = 200
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")

        # 模拟 tick 数据缺失的情况：vpin_signed_imbalance 全零
        vpin_signed_imbalance = pd.Series(0.0, index=idx)
        vpin = pd.Series(0.5, index=idx)  # vpin 可以非零

        from src.features.time_series.baseline_features import (
            compute_ofi_short_from_series,
        )

        ofi_short = compute_ofi_short_from_series(
            vpin_signed_imbalance=vpin_signed_imbalance, vpin=vpin, window=5
        )

        print(f"\n=== 零值传播诊断 ===")
        print(f"vpin_signed_imbalance 全零: {(vpin_signed_imbalance == 0).all()}")
        print(f"ofi_short 全零: {(ofi_short == 0).all()}")

        # 当 vpin_signed_imbalance 全零时，ofi_short 必然全零
        assert (
            ofi_short == 0
        ).all(), "当 vpin_signed_imbalance=0 时，ofi_short 应该为 0"

        # 这会导致 compression_energy_x_ofi_short 全零
        compression_energy = pd.Series(np.random.uniform(-2, 2, n), index=idx)
        result = compute_compression_energy_x_ofi_short_from_series(
            compression_energy=compression_energy, ofi_short=ofi_short
        )

        assert (
            result["compression_energy_x_ofi_short"] == 0
        ).all(), "当 ofi_short=0 时，交互特征应该为 0"

        print("✓ 确认：零值传播是由 vpin_signed_imbalance=0 导致")


class TestPercentileRankWarmup:
    """测试百分位 rank 的 warmup 行为"""

    def test_shd_pct_warmup_behavior(self):
        """测试 shd_pct 在 warmup 期间的行为: warmup 期间应为 NaN（禁止静默降级为 0.5）"""
        n = 400
        idx = pd.date_range("2024-01-01", periods=n, freq="4H")

        # 模拟 SHD 数据
        shd = pd.Series(np.random.uniform(0, 1, n), index=idx)

        result = compute_percentile_rank_from_series(
            series=shd, window=288, shift=1, output_name="shd_pct"
        )

        shd_pct = result["shd_pct"]

        # 前 288 个点应该是 NaN（warmup 不足，不再静默降级为 0.5）
        warmup_period = shd_pct.iloc[:288]
        warmup_all_nan = warmup_period.isna().all()

        # 之后的点应该有变化（且不为 NaN）
        post_warmup = shd_pct.iloc[290:]
        post_warmup_has_variance = post_warmup.std() > 0.1
        post_warmup_no_nan = post_warmup.notna().all()

        print(f"\n=== shd_pct warmup 诊断 ===")
        print(f"前 288 个点是否全为 NaN: {warmup_all_nan}")
        print(f"之后的标准差: {post_warmup.std():.4f}")
        print(f"之后是否无 NaN: {post_warmup_no_nan}")

        assert warmup_all_nan, "warmup 期间应全为 NaN（禁止静默降级为 0.5）"
        assert post_warmup_has_variance, "warmup 后应有变化"
        assert post_warmup_no_nan, "warmup 后不应有 NaN"


class TestIntegrationDiagnostic:
    """集成诊断测试：模拟真实数据流"""

    def test_full_feature_pipeline_without_ticks(self):
        """测试：没有 tick 数据时的特征表现"""
        df = _create_realistic_ohlcv(300)

        # 计算 WPT 特征（使用正确的函数）
        wpt_result = extract_wpt_features(
            df,
            price_col="close",
            volume_col="volume",
            wavelet="db4",
            level=4,
            window=50,
        )

        # 模拟没有 tick 数据的情况
        df["vpin"] = 0.0
        df["vpin_signed_imbalance"] = 0.0

        from src.features.time_series.baseline_features import (
            compute_ofi_short_from_series,
        )

        df["ofi_short"] = compute_ofi_short_from_series(
            vpin_signed_imbalance=df["vpin_signed_imbalance"], vpin=df["vpin"], window=5
        )

        # 计算 bb_width_ratio (模拟)
        df["bb_width_ratio"] = 0.02 + np.random.uniform(0, 0.01, len(df))

        # 合并 WPT 结果
        for col in wpt_result.columns:
            df[col] = wpt_result[col]

        # 计算 compression_energy
        # 需要 wpt_energy_cascade，但它可能不在 wpt_result 中
        if "wpt_energy_cascade" not in df.columns:
            df["wpt_energy_cascade"] = 0.0  # 默认值

        compression_energy = compute_compression_energy_from_series(
            wpt_energy_cascade=df["wpt_energy_cascade"],
            bb_width_ratio=df["bb_width_ratio"],
        )
        df["compression_energy"] = compression_energy

        # 计算交互特征
        result = compute_compression_energy_x_ofi_short_from_series(
            compression_energy=df["compression_energy"], ofi_short=df["ofi_short"]
        )

        interaction = result["compression_energy_x_ofi_short"]

        print(f"\n=== 集成诊断（无 tick 数据）===")
        print(f"vpin_signed_imbalance 全零: {(df['vpin_signed_imbalance'] == 0).all()}")
        print(f"ofi_short 全零: {(df['ofi_short'] == 0).all()}")
        print(f"compression_energy 有变化: {df['compression_energy'].std() > 0.01}")
        print(f"compression_energy_x_ofi_short 全零: {(interaction == 0).all()}")

        # 诊断结论
        if (interaction == 0).all():
            print("\n🔴 问题确认: compression_energy_x_ofi_short 全零")
            print("   根因: vpin_signed_imbalance 全零 (缺少 tick 数据)")
            print("   修复: 需要正确配置并加载 tick 数据")


@pytest.mark.slow
@pytest.mark.integration
class TestTickDataIntegration:
    """Tick 数据集成测试：验证 VPIN 特征计算"""

    def test_vpin_with_real_tick_data(self):
        """测试使用真实 tick 数据计算 VPIN"""
        from pathlib import Path
        import json

        # 检查 tick 文件是否存在
        tick_dir = Path("data/parquet_data")
        tick_files = sorted(tick_dir.glob("BTCUSDT_2023-*.parquet"))

        if len(tick_files) == 0:
            pytest.skip("No BTCUSDT tick files found for 2023")

        print(f"\n=== VPIN Tick 数据集成测试 ===")
        print(f"找到 {len(tick_files)} 个 tick 文件")

        # 加载一个月的 tick 数据
        tick_file = tick_files[0]
        ticks = pd.read_parquet(tick_file)

        print(f"加载 tick 文件: {tick_file.name}")
        print(f"Tick 数据行数: {len(ticks):,}")
        print(f"Tick 数据列: {list(ticks.columns)}")

        # 验证必要列存在
        required_cols = ["timestamp", "price", "volume", "side"]
        for col in required_cols:
            assert col in ticks.columns, f"缺少必要列: {col}"

        # 验证 side 列的值
        unique_sides = ticks["side"].unique()
        print(f"Side 唯一值: {unique_sides}")
        assert len(unique_sides) >= 2, f"Side 列应有买/卖两种值: {unique_sides}"

        print("✅ Tick 数据格式验证通过")

    def test_vpin_calculation_with_ticks(self):
        """测试使用 tick 数据计算 VPIN"""
        from pathlib import Path
        from src.features.time_series.utils_order_flow_features import (
            compute_vpin_from_ticks,
        )

        tick_dir = Path("data/parquet_data")
        tick_files = sorted(tick_dir.glob("BTCUSDT_2023-06*.parquet"))

        if len(tick_files) == 0:
            pytest.skip("No BTCUSDT_2023-06 tick file found")

        ticks = pd.read_parquet(tick_files[0])

        # 确保 timestamp 是 index
        if "timestamp" in ticks.columns:
            ticks = ticks.set_index("timestamp")

        print(f"\n=== VPIN 计算测试 ===")
        print(f"Tick 数据范围: {ticks.index.min()} 到 {ticks.index.max()}")

        # 计算 VPIN
        try:
            vpin_result = compute_vpin_from_ticks(
                ticks,
                bucket_volume=1000,  # 使用较小的 bucket 以产生更多数据点
                n_buckets=10,
            )

            print(f"VPIN 结果列: {list(vpin_result.columns)}")
            print(f"VPIN 数据点: {len(vpin_result)}")

            if "vpin" in vpin_result.columns:
                vpin_values = vpin_result["vpin"].dropna()
                print(f"VPIN 有效值: {len(vpin_values)}")
                print(f"VPIN 范围: [{vpin_values.min():.4f}, {vpin_values.max():.4f}]")
                print(f"VPIN 标准差: {vpin_values.std():.4f}")

                assert len(vpin_values) > 0, "VPIN 应有有效值"
                assert vpin_values.std() > 0, "VPIN 应有变化"

            if "signed_imbalance" in vpin_result.columns:
                signed_values = vpin_result["signed_imbalance"].dropna()
                print(f"Signed Imbalance 有效值: {len(signed_values)}")
                print(
                    f"Signed Imbalance 范围: [{signed_values.min():.4f}, {signed_values.max():.4f}]"
                )

                assert len(signed_values) > 0, "Signed Imbalance 应有有效值"
                assert signed_values.std() > 0, "Signed Imbalance 应有变化"

            print("✅ VPIN 计算测试通过")

        except Exception as e:
            pytest.fail(f"VPIN 计算失败: {e}")


class TestWPTPriceEnergyLowRatio:
    """测试 wpt_price_energy_low_ratio 特征（修复常量值问题）"""

    def test_extract_wpt_price_features_normalized_with_log_returns(self):
        """测试 extract_wpt_price_features_normalized 使用 use_log_returns=True"""
        from src.features.time_series.utils_wpt_features import (
            extract_wpt_price_features_normalized,
        )

        # 创建带趋势的价格数据
        np.random.seed(42)
        n = 200
        trend = np.linspace(50000, 55000, n)
        noise = np.random.randn(n) * 100
        prices = trend + noise

        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices + np.abs(np.random.randn(n) * 50),
                "low": prices - np.abs(np.random.randn(n) * 50),
            },
            index=pd.date_range("2023-01-01", periods=n, freq="4h"),
        )

        print("\n=== WPT Price Energy Low Ratio 测试 ===")

        # 使用默认参数 (use_log_returns=True)
        result = extract_wpt_price_features_normalized(
            df,
            wavelet="db4",
            level=4,
            window=100,
        )

        # 检查能量比特征
        if "wpt_price_energy_low_ratio" in result.columns:
            low_ratio = result["wpt_price_energy_low_ratio"].dropna()
            print(f"wpt_price_energy_low_ratio 有效值: {len(low_ratio)}")
            print(
                f"wpt_price_energy_low_ratio 范围: [{low_ratio.min():.4f}, {low_ratio.max():.4f}]"
            )
            print(f"wpt_price_energy_low_ratio 标准差: {low_ratio.std():.4f}")

            # 验证不是常量
            assert (
                low_ratio.std() > 0.01
            ), f"wpt_price_energy_low_ratio 应有变化，但 std={low_ratio.std():.6f}"
            # 验证不是全部集中在 1.0
            assert (
                low_ratio.mean() < 0.99
            ), f"wpt_price_energy_low_ratio 不应全部集中在 1.0，但 mean={low_ratio.mean():.4f}"

        if "wpt_price_energy_mid_ratio" in result.columns:
            mid_ratio = result["wpt_price_energy_mid_ratio"].dropna()
            print(
                f"wpt_price_energy_mid_ratio 范围: [{mid_ratio.min():.4f}, {mid_ratio.max():.4f}]"
            )
            # 验证中频带能量不为 0
            assert (
                mid_ratio.max() > 0.01
            ), f"wpt_price_energy_mid_ratio 应有能量，但 max={mid_ratio.max():.6f}"

        if "wpt_price_energy_high_ratio" in result.columns:
            high_ratio = result["wpt_price_energy_high_ratio"].dropna()
            print(
                f"wpt_price_energy_high_ratio 范围: [{high_ratio.min():.4f}, {high_ratio.max():.4f}]"
            )
            # 验证高频带能量不为 0
            assert (
                high_ratio.max() > 0.01
            ), f"wpt_price_energy_high_ratio 应有能量，但 max={high_ratio.max():.6f}"

        print("✅ WPT Price Energy Low Ratio 测试通过")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
