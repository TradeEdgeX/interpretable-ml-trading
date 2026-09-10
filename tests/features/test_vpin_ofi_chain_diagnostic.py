"""
诊断测试：vpin_signed_imbalance 和 compression_energy_x_ofi_short 计算链路

问题：
1. vpin_signed_imbalance 全为零
2. compression_energy_x_ofi_short 全为零

依赖链：
tick_data → vpin_signed_imbalance → ofi_short → compression_energy_x_ofi_short

诊断结论：
==========
问题根因：tick 数据是分钟级聚合数据（每分钟只有 2 条记录：买和卖），不是真正的逐笔成交数据。
这导致：
1. 每个 VPIN bucket 内的买卖量会相互抵消
2. signed_imbalance 的均值趋近于 0
3. ofi_short 也趋近于 0
4. compression_energy_x_ofi_short = compression_energy × ofi_short ≈ 0

解决方案：
1. 使用真正的逐笔成交数据（而不是分钟级聚合数据）
2. 或者修改 VPIN 计算逻辑，使其能够处理聚合数据
3. 或者使用 CVD（累计量差）作为 signed_imbalance 的替代
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path


def _create_mock_tick_data(
    n_ticks: int = 10000,
    start_time: str = "2023-06-01",
    price_base: float = 30000,
    seed: int = 42,
) -> pd.DataFrame:
    """创建模拟 tick 数据"""
    rng = np.random.default_rng(seed)

    # 生成时间戳（每秒多个 tick）
    timestamps = pd.date_range(start_time, periods=n_ticks, freq="100ms")

    # 生成价格（带随机波动）
    price_changes = rng.normal(0, 5, n_ticks)
    prices = price_base + np.cumsum(price_changes)

    # 生成成交量（BTC 数量）
    # 每个 tick 交易量在 0.001 到 0.5 BTC 之间
    volumes = rng.uniform(0.001, 0.5, n_ticks)

    # 生成买卖方向（1=买, -1=卖）
    # 模拟真实情况：有一定的持续性
    sides = np.zeros(n_ticks)
    current_side = 1
    for i in range(n_ticks):
        if rng.random() < 0.3:  # 30% 概率切换方向
            current_side = -current_side
        sides[i] = current_side

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": prices,
            "volume": volumes,  # 数量（BTC）
            "side": sides.astype(int),
        }
    )


def _create_mock_kline_data(
    n_bars: int = 200,
    start_time: str = "2023-06-01",
    timeframe: str = "4H",
    price_base: float = 30000,
    seed: int = 42,
) -> pd.DataFrame:
    """创建模拟 K 线数据"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start_time, periods=n_bars, freq=timeframe)

    # 生成收盘价
    returns = rng.normal(0, 0.01, n_bars)
    close = price_base * np.exp(np.cumsum(returns))

    # 生成 OHLCV
    high = close * (1 + rng.uniform(0.001, 0.02, n_bars))
    low = close * (1 - rng.uniform(0.001, 0.02, n_bars))
    open_ = np.roll(close, 1)
    open_[0] = close[0]

    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.uniform(500, 2000, n_bars)

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=idx,
    )


class TestVPINSignedImbalanceChain:
    """测试 vpin_signed_imbalance 计算链路"""

    def test_vpin_from_mock_ticks(self):
        """测试：使用模拟 tick 数据计算 VPIN"""
        from src.features.time_series.utils_order_flow_features import (
            compute_vpin_from_ticks,
        )

        # 创建模拟 tick 数据
        ticks = _create_mock_tick_data(n_ticks=50000, seed=42)
        print("\n=== 1. 模拟 Tick 数据 ===")
        print(f"Tick 数量: {len(ticks):,}")
        print(f"时间范围: {ticks['timestamp'].min()} - {ticks['timestamp'].max()}")
        print(f"价格范围: [{ticks['price'].min():.2f}, {ticks['price'].max():.2f}]")
        print(
            f"成交量(BTC)范围: [{ticks['volume'].min():.4f}, {ticks['volume'].max():.4f}]"
        )
        print(f"买单占比: {(ticks['side'] == 1).mean():.2%}")

        # 设置 timestamp 为 index
        ticks_indexed = ticks.set_index("timestamp")

        # 计算 VPIN
        print("\n=== 2. 计算 VPIN ===")
        # 使用 USD bucket：bucket_volume_usd=100000 表示每个桶价值 10万 USD
        vpin_result = compute_vpin_from_ticks(
            ticks_indexed,
            bucket_volume_usd=100000,  # USD bucket size
            n_buckets=20,
        )

        print(f"VPIN 结果列: {list(vpin_result.columns)}")
        print(f"VPIN 数据点: {len(vpin_result)}")

        # 验证 vpin 列
        if "vpin" in vpin_result.columns:
            vpin_values = vpin_result["vpin"].dropna()
            print(f"\n=== VPIN 统计 ===")
            print(f"有效值数量: {len(vpin_values)}")
            print(f"范围: [{vpin_values.min():.4f}, {vpin_values.max():.4f}]")
            print(f"均值: {vpin_values.mean():.4f}")
            print(f"标准差: {vpin_values.std():.4f}")

            assert len(vpin_values) > 0, "VPIN 应有有效值"
            assert (
                vpin_values.std() > 0.01
            ), f"VPIN 应有变化, 但 std={vpin_values.std()}"
        else:
            pytest.fail("缺少 vpin 列")

        # 验证 signed_imbalance 列
        if "signed_imbalance" in vpin_result.columns:
            signed_values = vpin_result["signed_imbalance"].dropna()
            print(f"\n=== Signed Imbalance 统计 ===")
            print(f"有效值数量: {len(signed_values)}")
            print(f"范围: [{signed_values.min():.4f}, {signed_values.max():.4f}]")
            print(f"均值: {signed_values.mean():.4f}")
            print(f"标准差: {signed_values.std():.4f}")

            assert len(signed_values) > 0, "Signed Imbalance 应有有效值"
            assert (
                signed_values.std() > 0.01
            ), f"Signed Imbalance 应有变化, 但 std={signed_values.std()}"
        else:
            pytest.fail("缺少 signed_imbalance 列")

        print("\n✅ VPIN 从 tick 数据计算成功")

    def test_vpin_alignment_to_klines(self):
        """测试：VPIN 对齐到 K 线"""
        from src.features.time_series.utils_order_flow_features import (
            compute_vpin_from_ticks,
        )

        # 创建同时间范围的 tick 和 K 线数据
        # 关键：tick 数据时间范围必须覆盖多根 K 线
        ticks = _create_mock_tick_data(
            n_ticks=500000,  # 足够多的 tick 以覆盖多个 4H K 线
            start_time="2023-06-01",
            seed=42,
        )
        klines = _create_mock_kline_data(
            n_bars=10, start_time="2023-06-01", timeframe="4H", seed=42
        )

        print("\n=== 1. 数据准备 ===")
        print(f"Tick 范围: {ticks['timestamp'].min()} - {ticks['timestamp'].max()}")
        print(f"K 线范围: {klines.index.min()} - {klines.index.max()}")

        # 计算 VPIN
        ticks_indexed = ticks.set_index("timestamp")
        vpin_result = compute_vpin_from_ticks(
            ticks_indexed,
            bucket_volume_usd=50000,  # USD bucket size
            n_buckets=20,
        )

        print(f"\n=== 2. VPIN 计算结果 ===")
        print(f"VPIN 数据点: {len(vpin_result)}")
        if len(vpin_result) > 0:
            print(
                f"VPIN 时间范围: {vpin_result.index.min()} - {vpin_result.index.max()}"
            )

        # 简单对齐：使用 resample 或 asof
        # 方法1: resample 到 4H
        vpin_4h = vpin_result.resample("4H").agg(
            {
                "vpin": "mean",
                "signed_imbalance": "mean",
            }
        )

        print(f"\n=== 3. 对齐到 4H K 线 ===")
        print(f"对齐后数据点: {len(vpin_4h.dropna())}")

        vpin_aligned = vpin_4h.dropna()
        if len(vpin_aligned) > 1:
            print(
                f"vpin 范围: [{vpin_aligned['vpin'].min():.4f}, {vpin_aligned['vpin'].max():.4f}]"
            )
            print(
                f"signed_imbalance 范围: [{vpin_aligned['signed_imbalance'].min():.4f}, {vpin_aligned['signed_imbalance'].max():.4f}]"
            )

            assert vpin_aligned["vpin"].std() > 0, "对齐后 vpin 应有变化"
            assert (
                vpin_aligned["signed_imbalance"].std() > 0
            ), "对齐后 signed_imbalance 应有变化"
        elif len(vpin_aligned) == 1:
            print("⚠️ 只有 1 个有效数据点（tick 时间范围过短）")
            print("跳过方差检查（需要多个数据点）")
        else:
            print("⚠️ 警告: K线中没有有效的 vpin 值（时间范围不匹配）")

        print("\n✅ VPIN 对齐测试完成")


class TestOFIShortChain:
    """测试 ofi_short 计算链路"""

    def test_ofi_short_with_valid_inputs(self):
        """测试：ofi_short 在有效输入下的计算"""
        from src.features.time_series.baseline_features import (
            compute_ofi_short_from_series,
        )

        n = 200
        idx = pd.date_range("2023-06-01", periods=n, freq="4H")
        rng = np.random.default_rng(42)

        # 模拟真实的 vpin_signed_imbalance（有正有负）
        vpin_signed_imbalance = pd.Series(
            rng.uniform(-0.6, 0.6, n), index=idx, name="vpin_signed_imbalance"
        )
        # 模拟 vpin（0-1 范围）
        vpin = pd.Series(rng.uniform(0.1, 0.9, n), index=idx, name="vpin")

        print("\n=== 1. 输入数据 ===")
        print(
            f"vpin_signed_imbalance 范围: [{vpin_signed_imbalance.min():.4f}, {vpin_signed_imbalance.max():.4f}]"
        )
        print(f"vpin_signed_imbalance 标准差: {vpin_signed_imbalance.std():.4f}")
        print(f"vpin 范围: [{vpin.min():.4f}, {vpin.max():.4f}]")

        # 计算 ofi_short
        ofi_short = compute_ofi_short_from_series(
            vpin_signed_imbalance=vpin_signed_imbalance,
            vpin=vpin,
            window=5,
        )

        print(f"\n=== 2. ofi_short 结果 ===")
        print(f"ofi_short 范围: [{ofi_short.min():.4f}, {ofi_short.max():.4f}]")
        print(f"ofi_short 标准差: {ofi_short.std():.4f}")
        print(f"ofi_short 全零: {(ofi_short == 0).all()}")

        assert ofi_short.std() > 0.01, f"ofi_short 应有变化, 但 std={ofi_short.std()}"
        assert not (ofi_short == 0).all(), "ofi_short 不应全为零"

        print("\n✅ ofi_short 计算成功")

    def test_ofi_short_with_zero_vpin_signed_imbalance(self):
        """测试：当 vpin_signed_imbalance 全零时的行为"""
        from src.features.time_series.baseline_features import (
            compute_ofi_short_from_series,
        )

        n = 200
        idx = pd.date_range("2023-06-01", periods=n, freq="4H")

        # 模拟问题情况：vpin_signed_imbalance 全零
        vpin_signed_imbalance = pd.Series(0.0, index=idx)
        vpin = pd.Series(0.5, index=idx)  # vpin 非零

        ofi_short = compute_ofi_short_from_series(
            vpin_signed_imbalance=vpin_signed_imbalance,
            vpin=vpin,
            window=5,
        )

        print("\n=== 问题场景：vpin_signed_imbalance 全零 ===")
        print(f"vpin_signed_imbalance 全零: {(vpin_signed_imbalance == 0).all()}")
        print(f"ofi_short 全零: {(ofi_short == 0).all()}")

        # 验证：当输入全零时，输出也应全零
        assert (
            ofi_short == 0
        ).all(), "当 vpin_signed_imbalance=0 时，ofi_short 应该为 0"

        print("\n🔴 确认：vpin_signed_imbalance=0 会导致 ofi_short=0")
        print("   根因：tick 数据未加载或对齐失败")


class TestCompressionEnergyXOFIShortChain:
    """测试 compression_energy_x_ofi_short 计算链路"""

    def test_compression_energy_x_ofi_short_with_valid_inputs(self):
        """测试：交互特征在有效输入下的计算"""
        from src.features.time_series.utils_interaction_features import (
            compute_compression_energy_x_ofi_short_from_series,
        )

        n = 200
        idx = pd.date_range("2023-06-01", periods=n, freq="4H")
        rng = np.random.default_rng(42)

        # 模拟有效的 compression_energy
        compression_energy = pd.Series(
            rng.uniform(-2, 2, n), index=idx, name="compression_energy"
        )
        # 模拟有效的 ofi_short
        ofi_short = pd.Series(rng.uniform(-0.5, 0.5, n), index=idx, name="ofi_short")

        print("\n=== 1. 输入数据 ===")
        print(
            f"compression_energy 范围: [{compression_energy.min():.4f}, {compression_energy.max():.4f}]"
        )
        print(f"compression_energy 标准差: {compression_energy.std():.4f}")
        print(f"ofi_short 范围: [{ofi_short.min():.4f}, {ofi_short.max():.4f}]")
        print(f"ofi_short 标准差: {ofi_short.std():.4f}")

        # 计算交互特征
        result = compute_compression_energy_x_ofi_short_from_series(
            compression_energy=compression_energy,
            ofi_short=ofi_short,
        )

        interaction = result["compression_energy_x_ofi_short"]

        print(f"\n=== 2. 交互特征结果 ===")
        print(
            f"compression_energy_x_ofi_short 范围: [{interaction.min():.4f}, {interaction.max():.4f}]"
        )
        print(f"compression_energy_x_ofi_short 标准差: {interaction.std():.4f}")
        print(f"compression_energy_x_ofi_short 全零: {(interaction == 0).all()}")

        assert interaction.std() > 0.01, f"交互特征应有变化, 但 std={interaction.std()}"
        assert not (interaction == 0).all(), "交互特征不应全为零"

        print("\n✅ compression_energy_x_ofi_short 计算成功")

    def test_compression_energy_x_ofi_short_with_zero_ofi(self):
        """测试：当 ofi_short 全零时的行为"""
        from src.features.time_series.utils_interaction_features import (
            compute_compression_energy_x_ofi_short_from_series,
        )

        n = 200
        idx = pd.date_range("2023-06-01", periods=n, freq="4H")
        rng = np.random.default_rng(42)

        # 模拟问题情况：ofi_short 全零
        compression_energy = pd.Series(rng.uniform(-2, 2, n), index=idx)
        ofi_short = pd.Series(0.0, index=idx)  # 全零

        result = compute_compression_energy_x_ofi_short_from_series(
            compression_energy=compression_energy,
            ofi_short=ofi_short,
        )

        interaction = result["compression_energy_x_ofi_short"]

        print("\n=== 问题场景：ofi_short 全零 ===")
        print(f"compression_energy 有变化: {compression_energy.std() > 0}")
        print(f"ofi_short 全零: {(ofi_short == 0).all()}")
        print(f"compression_energy_x_ofi_short 全零: {(interaction == 0).all()}")

        assert (interaction == 0).all(), "当 ofi_short=0 时，交互特征应该为 0"

        print("\n🔴 确认：ofi_short=0 会导致 compression_energy_x_ofi_short=0")


class TestFullChainDiagnostic:
    """完整链路诊断测试"""

    def test_full_chain_with_mock_data(self):
        """测试：完整链路（tick → vpin → ofi_short → interaction）"""
        from src.features.time_series.utils_order_flow_features import (
            compute_vpin_from_ticks,
        )
        from src.features.time_series.baseline_features import (
            compute_ofi_short_from_series,
            compute_compression_energy_from_series,
        )
        from src.features.time_series.utils_interaction_features import (
            compute_compression_energy_x_ofi_short_from_series,
        )

        print("\n" + "=" * 60)
        print("完整链路诊断测试")
        print("=" * 60)

        # Step 1: 创建模拟数据
        # 关键：tick 数据时间范围必须覆盖多根 K 线
        print("\n[Step 1] 创建模拟数据")
        ticks = _create_mock_tick_data(n_ticks=500000, start_time="2023-06-01", seed=42)
        klines = _create_mock_kline_data(
            n_bars=10, start_time="2023-06-01", timeframe="4H", seed=42
        )

        print(f"  Tick 数量: {len(ticks):,}")
        print(
            f"  Tick 时间范围: {ticks['timestamp'].min()} - {ticks['timestamp'].max()}"
        )
        print(f"  K 线数量: {len(klines)}")
        print(f"  K 线时间范围: {klines.index.min()} - {klines.index.max()}")

        # Step 2: 计算 VPIN
        print("\n[Step 2] 计算 VPIN")
        ticks_indexed = ticks.set_index("timestamp")
        vpin_result = compute_vpin_from_ticks(
            ticks_indexed,
            bucket_volume_usd=50000,  # USD bucket size
            n_buckets=20,
        )
        print(f"  VPIN 数据点: {len(vpin_result)}")
        if len(vpin_result) > 0:
            print(
                f"  vpin 范围: [{vpin_result['vpin'].min():.4f}, {vpin_result['vpin'].max():.4f}]"
            )
            print(
                f"  signed_imbalance 范围: [{vpin_result['signed_imbalance'].min():.4f}, {vpin_result['signed_imbalance'].max():.4f}]"
            )
        else:
            print("  ⚠️ VPIN 数据为空（tick 数据不足以形成完整的 bucket）")

        # Step 3: 对齐到 K 线
        print("\n[Step 3] 对齐到 K 线")
        vpin_4h = vpin_result.resample("4H").agg(
            {
                "vpin": "mean",
                "signed_imbalance": "mean",
            }
        )
        klines["vpin"] = vpin_4h["vpin"]
        klines["vpin_signed_imbalance"] = vpin_4h["signed_imbalance"]

        # 填充 NaN
        klines["vpin"] = klines["vpin"].fillna(0.0)
        klines["vpin_signed_imbalance"] = klines["vpin_signed_imbalance"].fillna(0.0)

        vpin_nonzero = (klines["vpin_signed_imbalance"] != 0).sum()
        print(f"  K 线中 vpin_signed_imbalance 非零行: {vpin_nonzero}/{len(klines)}")
        if vpin_nonzero > 0:
            print(
                f"  vpin_signed_imbalance 范围: [{klines['vpin_signed_imbalance'].min():.4f}, {klines['vpin_signed_imbalance'].max():.4f}]"
            )

        # Step 4: 计算 ofi_short
        print("\n[Step 4] 计算 ofi_short")
        klines["ofi_short"] = compute_ofi_short_from_series(
            vpin_signed_imbalance=klines["vpin_signed_imbalance"],
            vpin=klines["vpin"],
            window=5,
        )

        ofi_nonzero = (klines["ofi_short"] != 0).sum()
        print(f"  ofi_short 非零行: {ofi_nonzero}/{len(klines)}")
        if ofi_nonzero > 0:
            print(
                f"  ofi_short 范围: [{klines['ofi_short'].min():.4f}, {klines['ofi_short'].max():.4f}]"
            )

        # Step 5: 计算 compression_energy（模拟）
        print("\n[Step 5] 计算 compression_energy")
        rng = np.random.default_rng(42)
        klines["wpt_energy_cascade"] = rng.uniform(-1, 1, len(klines))
        klines["bb_width_ratio"] = 0.02 + rng.uniform(0, 0.01, len(klines))

        klines["compression_energy"] = compute_compression_energy_from_series(
            wpt_energy_cascade=klines["wpt_energy_cascade"],
            bb_width_ratio=klines["bb_width_ratio"],
        )
        print(
            f"  compression_energy 范围: [{klines['compression_energy'].min():.4f}, {klines['compression_energy'].max():.4f}]"
        )

        # Step 6: 计算交互特征
        print("\n[Step 6] 计算 compression_energy_x_ofi_short")
        result = compute_compression_energy_x_ofi_short_from_series(
            compression_energy=klines["compression_energy"],
            ofi_short=klines["ofi_short"],
        )
        klines["compression_energy_x_ofi_short"] = result[
            "compression_energy_x_ofi_short"
        ]

        interaction_nonzero = (klines["compression_energy_x_ofi_short"] != 0).sum()
        print(
            f"  compression_energy_x_ofi_short 非零行: {interaction_nonzero}/{len(klines)}"
        )
        if interaction_nonzero > 0:
            print(
                f"  compression_energy_x_ofi_short 范围: [{klines['compression_energy_x_ofi_short'].min():.4f}, {klines['compression_energy_x_ofi_short'].max():.4f}]"
            )

        # 诊断总结
        print("\n" + "=" * 60)
        print("诊断总结")
        print("=" * 60)

        issues = []
        if vpin_nonzero == 0:
            issues.append(
                "❌ vpin_signed_imbalance 全为零（tick 数据未正确加载或对齐）"
            )
        else:
            print(f"✅ vpin_signed_imbalance 有 {vpin_nonzero} 个非零值")

        if ofi_nonzero == 0:
            issues.append("❌ ofi_short 全为零（由 vpin_signed_imbalance=0 导致）")
        else:
            print(f"✅ ofi_short 有 {ofi_nonzero} 个非零值")

        if interaction_nonzero == 0:
            issues.append(
                "❌ compression_energy_x_ofi_short 全为零（由 ofi_short=0 导致）"
            )
        else:
            print(
                f"✅ compression_energy_x_ofi_short 有 {interaction_nonzero} 个非零值"
            )

        if issues:
            print("\n发现问题:")
            for issue in issues:
                print(f"  {issue}")
            print("\n可能原因:")
            print("  1. tick 数据文件未找到或路径配置错误")
            print("  2. tick 数据时间范围与 K 线不匹配")
            print("  3. tick 数据格式不正确（缺少必要列）")
            print("  4. VPIN bucket 设置过大导致数据点太少")
        else:
            print("\n✅ 所有检查通过！链路计算正常。")

        # 断言确保链路正常
        assert vpin_nonzero > 0, "vpin_signed_imbalance 不应全为零"
        assert ofi_nonzero > 0, "ofi_short 不应全为零"
        assert interaction_nonzero > 0, "compression_energy_x_ofi_short 不应全为零"


@pytest.mark.slow
@pytest.mark.integration
class TestRealDataDiagnostic:
    """真实数据诊断测试（需要实际 tick 数据）"""

    @pytest.mark.skipif(
        not Path("data/parquet_data").exists(), reason="No tick data directory found"
    )
    def test_diagnose_real_tick_data(self):
        """诊断：检查真实 tick 数据"""
        tick_dir = Path("data/parquet_data")
        tick_files = sorted(tick_dir.glob("BTCUSDT_2023-*.parquet"))

        print("\n=== 真实 Tick 数据诊断 ===")
        print(f"Tick 数据目录: {tick_dir}")
        print(f"找到文件数: {len(tick_files)}")

        if len(tick_files) == 0:
            print("❌ 未找到 tick 数据文件")
            pytest.skip("No tick files found")

        # 检查第一个文件
        tick_file = tick_files[0]
        print(f"\n检查文件: {tick_file.name}")

        ticks = pd.read_parquet(tick_file)
        print(f"行数: {len(ticks):,}")
        print(f"列: {list(ticks.columns)}")

        # 检查必要列
        required_cols = ["price", "volume", "side"]
        for col in required_cols:
            if col in ticks.columns:
                print(f"✅ 存在列: {col}")
            else:
                # 尝试查找类似列名
                similar = [c for c in ticks.columns if col.lower() in c.lower()]
                if similar:
                    print(f"⚠️ 缺少 '{col}'，但找到类似列: {similar}")
                else:
                    print(f"❌ 缺少列: {col}")

        # 检查时间戳
        if "timestamp" in ticks.columns:
            print(
                f"\n时间范围: {ticks['timestamp'].min()} - {ticks['timestamp'].max()}"
            )

        # 检查 side 列的值
        if "side" in ticks.columns:
            unique_sides = ticks["side"].unique()
            print(f"Side 唯一值: {unique_sides}")
            if len(unique_sides) < 2:
                print("⚠️ Side 列应有买/卖两种值")

    @pytest.mark.skipif(
        not Path("data/parquet_data").exists(), reason="No tick data directory found"
    )
    def test_diagnose_vpin_from_real_tick(self):
        """诊断：从真实 tick 数据计算 VPIN 并对齐到 K 线"""
        from src.features.time_series.utils_order_flow_features import (
            compute_vpin_from_ticks,
        )

        tick_dir = Path("data/parquet_data")
        tick_files = sorted(tick_dir.glob("BTCUSDT_2023-*.parquet"))

        if len(tick_files) == 0:
            pytest.skip("No tick files found")

        print("\n=== VPIN 真实数据诊断 ===")

        # 加载第一个月的数据
        tick_file = tick_files[0]
        print(f"加载文件: {tick_file.name}")

        ticks = pd.read_parquet(tick_file)
        print(f"Tick 数量: {len(ticks):,}")

        # 设置 timestamp 为 index
        if "timestamp" in ticks.columns:
            ticks = ticks.set_index("timestamp")

        print(f"\n时间范围: {ticks.index.min()} - {ticks.index.max()}")

        # 计算 VPIN
        print("\n计算 VPIN...")
        vpin_result = compute_vpin_from_ticks(
            ticks,
            bucket_volume_usd=100000,  # 10万 USD per bucket
            n_buckets=50,
        )

        print(f"VPIN 数据点: {len(vpin_result)}")
        if len(vpin_result) > 0:
            print(
                f"vpin 范围: [{vpin_result['vpin'].min():.4f}, {vpin_result['vpin'].max():.4f}]"
            )
            print(f"vpin 均值: {vpin_result['vpin'].mean():.4f}")
            print(
                f"signed_imbalance 范围: [{vpin_result['signed_imbalance'].min():.4f}, {vpin_result['signed_imbalance'].max():.4f}]"
            )
            print(
                f"signed_imbalance 均值: {vpin_result['signed_imbalance'].mean():.4f}"
            )
            print(
                f"signed_imbalance 标准差: {vpin_result['signed_imbalance'].std():.4f}"
            )

            # 对齐到 4H K 线
            print("\n对齐到 4H K 线...")
            vpin_4h = vpin_result.resample("4H").agg(
                {
                    "vpin": "mean",
                    "signed_imbalance": "mean",
                }
            )

            vpin_4h_valid = vpin_4h.dropna()
            print(f"对齐后 K 线数: {len(vpin_4h_valid)}")
            if len(vpin_4h_valid) > 0:
                print(
                    f"4H vpin 范围: [{vpin_4h_valid['vpin'].min():.4f}, {vpin_4h_valid['vpin'].max():.4f}]"
                )
                print(
                    f"4H signed_imbalance 范围: [{vpin_4h_valid['signed_imbalance'].min():.4f}, {vpin_4h_valid['signed_imbalance'].max():.4f}]"
                )
                print(
                    f"4H signed_imbalance 均值: {vpin_4h_valid['signed_imbalance'].mean():.4f}"
                )
                print(
                    f"4H signed_imbalance 标准差: {vpin_4h_valid['signed_imbalance'].std():.4f}"
                )

                # 检查是否全为 0 或接近 0
                if vpin_4h_valid["signed_imbalance"].std() < 0.001:
                    print(
                        "\n🟡 警告: signed_imbalance 均值后标准差很小，可能正负相抵消了"
                    )
                    print("   这可能是预期行为，但会导致 ofi_short 近似为 0")
        else:
            print("❌ VPIN 数据为空")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
