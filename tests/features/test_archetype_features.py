"""
Archetype 特征测试（HTF/LTF, ME, FBF, LSR, AER）

测试内容：
1. 未来数据泄露验证（确保特征不使用未来信息）⭐⭐⭐⭐⭐
2. 特征值范围和正确性测试 ⭐⭐⭐
3. 基本功能测试 ⭐⭐⭐

参考规范：
- docs/tests/FEATURE_TEST_DESIGN_AND_COVERAGE_CN.md
- Archetype 语义化特征建模规范
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.htf_ltf_features import (
    compute_htf_ltf_soft_phase_from_series,
    compute_htf_ltf_context_from_series,
    compute_htf_ltf_failure_signals_from_series,
)
from src.features.time_series.momentum_expansion_features import (
    compute_momentum_expansion_soft_phase_from_series,
    compute_momentum_expansion_failure_from_series,
    compute_momentum_expansion_context_from_series,
)
from src.features.time_series.failed_breakout_features import (
    compute_failed_breakout_fade_soft_phase_from_series,
    compute_failed_breakout_fade_failure_from_series,
)
from src.features.time_series.liquidity_sweep_features import (
    compute_liquidity_sweep_rejection_soft_phase_from_series,
    compute_liquidity_sweep_rejection_failure_from_series,
)
from src.features.time_series.auction_exhaustion_features import (
    compute_auction_exhaustion_reversal_soft_phase_from_series,
    compute_auction_exhaustion_reversal_failure_from_series,
    compute_auction_exhaustion_reversal_context_from_series,
)

# =============================================================================
# 📊 测试数据生成器
# =============================================================================


def create_ohlcv_data(
    n_samples: int = 500,
    base_price: float = 50000.0,
    volatility: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """创建模拟 OHLCV 数据"""
    np.random.seed(seed)

    timestamps = pd.date_range("2024-01-01 00:00:00", periods=n_samples, freq="5min")

    # 生成价格序列（随机游走 + 趋势）
    returns = np.random.randn(n_samples) * volatility
    log_prices = np.log(base_price) + np.cumsum(returns)
    close = np.exp(log_prices)

    # 生成 OHLC
    high_spread = np.abs(np.random.randn(n_samples)) * volatility * base_price
    low_spread = np.abs(np.random.randn(n_samples)) * volatility * base_price
    open_offset = np.random.randn(n_samples) * volatility * base_price * 0.5

    high = close + high_spread
    low = close - low_spread
    open_price = close + open_offset

    # 确保 OHLC 关系正确
    high = np.maximum(high, np.maximum(close, open_price))
    low = np.minimum(low, np.minimum(close, open_price))

    # 生成成交量
    volume = np.random.uniform(100, 1000, n_samples)

    # 计算 ATR
    tr = np.maximum(
        high - low,
        np.maximum(np.abs(high - np.roll(close, 1)), np.abs(low - np.roll(close, 1))),
    )
    tr[0] = high[0] - low[0]
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    df = pd.DataFrame(
        {
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "atr": atr,
        },
        index=timestamps,
    )

    return df


def create_orderflow_data(df: pd.DataFrame, seed: int = 42) -> dict:
    """创建模拟订单流数据"""
    np.random.seed(seed)
    n = len(df)

    price_dir = np.sign(df["close"].diff().fillna(0).values)
    cvd_change_5 = (
        price_dir * np.abs(np.random.randn(n)) * 100 + np.random.randn(n) * 30
    )
    vpin = np.clip(np.random.uniform(0.3, 0.8, n), 0, 1)
    bb_width_normalized = np.clip(np.random.uniform(0.2, 0.6, n), 0, 1)

    return {
        "cvd_change_5": pd.Series(cvd_change_5, index=df.index),
        "vpin": pd.Series(vpin, index=df.index),
        "bb_width_normalized": pd.Series(bb_width_normalized, index=df.index),
    }


# =============================================================================
# 📋 测试类：HTFBiasLTFEntry
# =============================================================================


class TestHTFLTFFeatures:
    """HTFBiasLTFEntry 特征测试"""

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_htf_ltf_soft_phase_basic(self, sample_data):
        """测试 HTF/LTF 软阶段基本功能"""
        df, orderflow = sample_data

        result = compute_htf_ltf_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )

        # 检查输出列存在
        expected_cols = [
            "htf_trend_strength",
            "htf_path_efficiency",
            "htf_score_bias",
            "ltf_score_entry",
            "htf_ltf_score_alignment",
            "htf_direction",
            "ltf_direction",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in ["htf_score_bias", "ltf_score_entry", "htf_ltf_score_alignment"]:
            vals = result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ HTF/LTF 基本功能测试通过")

    def test_htf_ltf_no_future_leak(self, sample_data):
        """测试 HTF/LTF 无未来数据泄露"""
        df, orderflow = sample_data

        # 计算第一次特征
        result1 = compute_htf_ltf_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )

        # 修改未来数据
        df_future = df.copy()
        future_idx = df_future.index[300:]
        df_future.loc[future_idx, "close"] *= 2.0

        result2 = compute_htf_ltf_soft_phase_from_series(
            close=df_future["close"],
            high=df_future["high"],
            low=df_future["low"],
            volume=df_future["volume"],
            atr=df_future["atr"],
        )

        # 验证历史特征值相同
        check_idx = df.index[:200]
        for col in ["htf_score_bias", "ltf_score_entry"]:
            vals1 = result1.loc[check_idx, col].dropna()
            vals2 = result2.loc[check_idx, col].dropna()
            common_idx = vals1.index.intersection(vals2.index)
            if len(common_idx) > 0:
                diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs().max()
                assert diff < 1e-6, f"{col} 存在未来数据泄露，差异: {diff}"

        print("✅ HTF/LTF 无未来数据泄露测试通过")


# =============================================================================
# 📋 测试类：MomentumExpansion
# =============================================================================


class TestMomentumExpansionFeatures:
    """MomentumExpansion 特征测试"""

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_me_soft_phase_basic(self, sample_data):
        """测试 MomentumExpansion 基本功能"""
        df, orderflow = sample_data

        result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )

        # 检查输出列存在
        expected_cols = [
            "me_atr_expansion",
            "me_bb_width_expansion",
            "me_score_expansion",
            "me_score_acceleration",
            "me_score_orderflow",
            "me_score_total",
            "me_direction",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in ["me_score_expansion", "me_score_acceleration", "me_score_total"]:
            vals = result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ MomentumExpansion 基本功能测试通过")

    def test_me_no_future_leak(self, sample_data):
        """测试 MomentumExpansion 无未来数据泄露"""
        df, orderflow = sample_data

        result1 = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )

        df_future = df.copy()
        df_future.loc[df_future.index[300:], "close"] *= 2.0

        result2 = compute_momentum_expansion_soft_phase_from_series(
            close=df_future["close"],
            high=df_future["high"],
            low=df_future["low"],
            volume=df_future["volume"],
            atr=df_future["atr"],
        )

        check_idx = df.index[:200]
        for col in ["me_score_expansion", "me_score_total"]:
            vals1 = result1.loc[check_idx, col].dropna()
            vals2 = result2.loc[check_idx, col].dropna()
            common_idx = vals1.index.intersection(vals2.index)
            if len(common_idx) > 0:
                diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs().max()
                assert diff < 1e-6, f"{col} 存在未来数据泄露，差异: {diff}"

        print("✅ MomentumExpansion 无未来数据泄露测试通过")


# =============================================================================
# 📋 测试类：FailedBreakoutFade
# =============================================================================


class TestFailedBreakoutFadeFeatures:
    """FailedBreakoutFade 特征测试"""

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_fbf_soft_phase_basic(self, sample_data):
        """测试 FailedBreakoutFade 基本功能"""
        df, orderflow = sample_data

        result = compute_failed_breakout_fade_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open_=df["open"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=orderflow["cvd_change_5"],
        )

        # 检查输出列存在
        expected_cols = [
            "fbf_breakout_attempt",
            "fbf_breakout_failed",
            "fbf_score_false_breakout",
            "fbf_score_rejection",
            "fbf_score_fade",
            "fbf_score_total",
            "fbf_direction",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in [
            "fbf_score_false_breakout",
            "fbf_score_rejection",
            "fbf_score_total",
        ]:
            vals = result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ FailedBreakoutFade 基本功能测试通过")

    def test_fbf_no_future_leak(self, sample_data):
        """测试 FailedBreakoutFade 无未来数据泄露"""
        df, orderflow = sample_data

        result1 = compute_failed_breakout_fade_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open_=df["open"],
            volume=df["volume"],
            atr=df["atr"],
        )

        df_future = df.copy()
        df_future.loc[df_future.index[300:], "close"] *= 2.0

        result2 = compute_failed_breakout_fade_soft_phase_from_series(
            close=df_future["close"],
            high=df_future["high"],
            low=df_future["low"],
            open_=df_future["open"],
            volume=df_future["volume"],
            atr=df_future["atr"],
        )

        check_idx = df.index[:200]
        for col in ["fbf_score_false_breakout", "fbf_score_total"]:
            vals1 = result1.loc[check_idx, col].dropna()
            vals2 = result2.loc[check_idx, col].dropna()
            common_idx = vals1.index.intersection(vals2.index)
            if len(common_idx) > 0:
                diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs().max()
                assert diff < 1e-6, f"{col} 存在未来数据泄露，差异: {diff}"

        print("✅ FailedBreakoutFade 无未来数据泄露测试通过")


# =============================================================================
# 📋 测试类：LiquiditySweepRejection
# =============================================================================


class TestLiquiditySweepRejectionFeatures:
    """LiquiditySweepRejection 特征测试"""

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_lsr_soft_phase_basic(self, sample_data):
        """测试 LiquiditySweepRejection 基本功能"""
        df, orderflow = sample_data

        result = compute_liquidity_sweep_rejection_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open_=df["open"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=orderflow["cvd_change_5"],
        )

        # 检查输出列存在
        expected_cols = [
            "lsr_sweep_up_detected",
            "lsr_sweep_down_detected",
            "lsr_score_sweep",
            "lsr_score_rejection",
            "lsr_score_reversal",
            "lsr_score_total",
            "lsr_sweep_side",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in ["lsr_score_sweep", "lsr_score_rejection", "lsr_score_total"]:
            vals = result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ LiquiditySweepRejection 基本功能测试通过")

    def test_lsr_no_future_leak(self, sample_data):
        """测试 LiquiditySweepRejection 无未来数据泄露"""
        df, orderflow = sample_data

        result1 = compute_liquidity_sweep_rejection_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open_=df["open"],
            volume=df["volume"],
            atr=df["atr"],
        )

        df_future = df.copy()
        df_future.loc[df_future.index[300:], "close"] *= 2.0

        result2 = compute_liquidity_sweep_rejection_soft_phase_from_series(
            close=df_future["close"],
            high=df_future["high"],
            low=df_future["low"],
            open_=df_future["open"],
            volume=df_future["volume"],
            atr=df_future["atr"],
        )

        check_idx = df.index[:200]
        for col in ["lsr_score_sweep", "lsr_score_total"]:
            vals1 = result1.loc[check_idx, col].dropna()
            vals2 = result2.loc[check_idx, col].dropna()
            common_idx = vals1.index.intersection(vals2.index)
            if len(common_idx) > 0:
                diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs().max()
                assert diff < 1e-6, f"{col} 存在未来数据泄露，差异: {diff}"

        print("✅ LiquiditySweepRejection 无未来数据泄露测试通过")


# =============================================================================
# 📋 测试类：AuctionExhaustionReversal
# =============================================================================


class TestAuctionExhaustionReversalFeatures:
    """AuctionExhaustionReversal 特征测试"""

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_aer_soft_phase_basic(self, sample_data):
        """测试 AuctionExhaustionReversal 基本功能"""
        df, orderflow = sample_data

        result = compute_auction_exhaustion_reversal_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )

        # 检查输出列存在
        expected_cols = [
            "aer_vol_climax",
            "aer_atr_climax",
            "aer_score_climax",
            "aer_score_exhaustion",
            "aer_score_reversal",
            "aer_score_total",
            "aer_trend_direction",
        ]
        for col in expected_cols:
            assert col in result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in ["aer_score_climax", "aer_score_exhaustion", "aer_score_total"]:
            vals = result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ AuctionExhaustionReversal 基本功能测试通过")

    def test_aer_no_future_leak(self, sample_data):
        """测试 AuctionExhaustionReversal 无未来数据泄露"""
        df, orderflow = sample_data

        result1 = compute_auction_exhaustion_reversal_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )

        df_future = df.copy()
        df_future.loc[df_future.index[300:], "close"] *= 2.0

        result2 = compute_auction_exhaustion_reversal_soft_phase_from_series(
            close=df_future["close"],
            high=df_future["high"],
            low=df_future["low"],
            volume=df_future["volume"],
            atr=df_future["atr"],
        )

        check_idx = df.index[:200]
        for col in ["aer_score_exhaustion", "aer_score_total"]:
            vals1 = result1.loc[check_idx, col].dropna()
            vals2 = result2.loc[check_idx, col].dropna()
            common_idx = vals1.index.intersection(vals2.index)
            if len(common_idx) > 0:
                diff = (vals1.loc[common_idx] - vals2.loc[common_idx]).abs().max()
                assert diff < 1e-6, f"{col} 存在未来数据泄露，差异: {diff}"

        print("✅ AuctionExhaustionReversal 无未来数据泄露测试通过")


# =============================================================================
# 📋 测试类：Failure 信号
# =============================================================================


class TestArchetypeFailureFeatures:
    """Archetype Failure 特征测试"""

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def test_htf_ltf_failure_basic(self, sample_data):
        """测试 HTF/LTF Failure 特征"""
        df, orderflow = sample_data

        # 先计算主特征
        main_result = compute_htf_ltf_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )

        # 计算 failure 特征
        failure_result = compute_htf_ltf_failure_signals_from_series(
            close=df["close"],
            htf_score_bias=main_result["htf_score_bias"],
            ltf_score_entry=main_result["ltf_score_entry"],
            htf_ltf_score_alignment=main_result["htf_ltf_score_alignment"],
        )

        # 检查输出列
        expected_cols = [
            "htf_trend_exhaustion",
            "ltf_false_entry",
            "htf_ltf_divergence",
            "htf_ltf_failure_score",
        ]
        for col in expected_cols:
            assert col in failure_result.columns, f"缺少列: {col}"

        # 检查值范围
        for col in expected_cols:
            vals = failure_result[col].dropna()
            assert vals.min() >= 0, f"{col} 最小值小于 0"
            assert vals.max() <= 1, f"{col} 最大值大于 1"

        print("✅ HTF/LTF Failure 特征测试通过")

    def test_me_failure_basic(self, sample_data):
        """测试 MomentumExpansion Failure 特征"""
        df, orderflow = sample_data

        main_result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )

        failure_result = compute_momentum_expansion_failure_from_series(
            close=df["close"],
            me_score_expansion=main_result["me_score_expansion"],
            me_score_acceleration=main_result["me_score_acceleration"],
            me_score_orderflow=main_result["me_score_orderflow"],
            volume=df["volume"],
        )

        expected_cols = [
            "me_false_expansion",
            "me_vol_divergence",
            "me_orderflow_exhaustion",
            "me_failure_score",
        ]
        for col in expected_cols:
            assert col in failure_result.columns, f"缺少列: {col}"

        print("✅ MomentumExpansion Failure 特征测试通过")

    def test_aer_context_basic(self, sample_data):
        """测试 AuctionExhaustionReversal Context 特征"""
        df, _ = sample_data

        # 模拟 jump_risk_pct 和 shd_pct
        jump_risk = pd.Series(np.random.uniform(0.2, 0.8, len(df)), index=df.index)
        shd_pct = pd.Series(np.random.uniform(0.3, 0.7, len(df)), index=df.index)

        context_result = compute_auction_exhaustion_reversal_context_from_series(
            close=df["close"],
            jump_risk_pct=jump_risk,
            shd_pct=shd_pct,
        )

        expected_cols = [
            "aer_jump_risk_suitable",
            "aer_reflex_risk",
            "aer_regime_suitable",
        ]
        for col in expected_cols:
            assert col in context_result.columns, f"缺少列: {col}"

        print("✅ AuctionExhaustionReversal Context 特征测试通过")


# =============================================================================
# 📋 测试类：流式 vs 批量一致性
# =============================================================================


class TestArchetypeStreamingVsBatch:
    """
    Archetype 特征流式 vs 批量一致性测试 ⭐⭐⭐⭐

    对生产部署至关重要：生产环境往往是流式推理，而训练是批量计算
    """

    @pytest.fixture
    def sample_data(self):
        # 使用更长的数据，确保有足够的 warmup 和验证样本
        df = create_ohlcv_data(n_samples=800, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def _test_streaming_consistency(
        self, compute_func, df, orderflow, key_features, warmup_size=120
    ):
        """通用流式 vs 批量一致性测试

        注意：warmup_size 需要大于所有滚动窗口的最大值（包括 pct_window=100）
        """
        # 1. 批量计算
        batch_result = compute_func(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
            cvd_change_5=orderflow["cvd_change_5"],
            vpin=orderflow["vpin"],
        )

        # 2. 流式计算（分块处理，带 warmup）
        chunk_size = 100
        streaming_results = []

        for i in range(warmup_size, len(df), chunk_size):
            start_idx = max(0, i - warmup_size)
            end_idx = min(i + chunk_size, len(df))

            chunk_df = df.iloc[start_idx:end_idx]
            chunk_orderflow = {
                k: v.iloc[start_idx:end_idx] for k, v in orderflow.items()
            }

            chunk_result = compute_func(
                close=chunk_df["close"],
                high=chunk_df["high"],
                low=chunk_df["low"],
                volume=chunk_df["volume"],
                atr=chunk_df["atr"],
                cvd_change_5=chunk_orderflow["cvd_change_5"],
                vpin=chunk_orderflow["vpin"],
            )

            actual_start = i - start_idx
            actual_result = chunk_result.iloc[actual_start:]
            streaming_results.append(actual_result)

        # 3. 合并流式结果
        if streaming_results:
            streaming_combined = pd.concat(streaming_results, axis=0)
            streaming_combined = streaming_combined[
                ~streaming_combined.index.duplicated(keep="first")
            ]
        else:
            streaming_combined = pd.DataFrame()

        # 4. 比较关键特征
        for feat in key_features:
            if feat in batch_result.columns and feat in streaming_combined.columns:
                batch_vals = batch_result[feat].dropna()
                stream_vals = streaming_combined[feat].dropna()
                common_idx = batch_vals.index.intersection(stream_vals.index)
                if len(common_idx) > 10:
                    diff = (
                        batch_vals.loc[common_idx] - stream_vals.loc[common_idx]
                    ).abs()
                    max_diff = diff.max()
                    assert (
                        max_diff < 1e-5
                    ), f"流式与批量计算 {feat} 不一致，最大差异: {max_diff:.8f}"

    def test_htf_ltf_streaming_consistency(self, sample_data):
        """测试 HTF/LTF 流式 vs 批量一致性"""
        df, orderflow = sample_data
        self._test_streaming_consistency(
            compute_htf_ltf_soft_phase_from_series,
            df,
            orderflow,
            ["htf_score_bias", "ltf_score_entry", "htf_ltf_score_alignment"],
        )
        print("✅ HTF/LTF 流式 vs 批量一致性测试通过")

    def test_me_streaming_consistency(self, sample_data):
        """测试 MomentumExpansion 流式 vs 批量一致性"""
        df, orderflow = sample_data
        self._test_streaming_consistency(
            compute_momentum_expansion_soft_phase_from_series,
            df,
            orderflow,
            ["me_score_expansion", "me_score_acceleration", "me_score_total"],
        )
        print("✅ MomentumExpansion 流式 vs 批量一致性测试通过")

    def test_aer_streaming_consistency(self, sample_data):
        """测试 AuctionExhaustionReversal 流式 vs 批量一致性"""
        df, orderflow = sample_data
        self._test_streaming_consistency(
            compute_auction_exhaustion_reversal_soft_phase_from_series,
            df,
            orderflow,
            ["aer_score_climax", "aer_score_exhaustion", "aer_score_total"],
        )
        print("✅ AuctionExhaustionReversal 流式 vs 批量一致性测试通过")


# =============================================================================
# 📋 测试类：功能正确性
# =============================================================================


class TestArchetypeCorrectness:
    """
    Archetype 特征功能正确性测试 ⭐⭐⭐

    验证：
    - 特征值范围 [0, 1]
    - 无 NaN/Inf 溢出
    - 方向特征在预期范围
    """

    @pytest.fixture
    def sample_data(self):
        df = create_ohlcv_data(n_samples=500, seed=42)
        orderflow = create_orderflow_data(df, seed=42)
        return df, orderflow

    def _test_value_ranges(self, result, bounded_features, name):
        """测试特征值范围"""
        for feat in bounded_features:
            if feat in result.columns:
                vals = result[feat].dropna()
                if len(vals) > 0:
                    min_val = vals.min()
                    max_val = vals.max()
                    assert min_val >= -0.01, f"{name}.{feat} 最小值 {min_val:.4f} < 0"
                    assert max_val <= 1.01, f"{name}.{feat} 最大值 {max_val:.4f} > 1"

        # 检查无 Inf
        for col in result.columns:
            assert not np.isinf(result[col]).any(), f"{name}.{col} 包含 Inf 值"

    def test_htf_ltf_value_ranges(self, sample_data):
        """测试 HTF/LTF 特征值范围"""
        df, orderflow = sample_data
        result = compute_htf_ltf_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )
        self._test_value_ranges(
            result,
            [
                "htf_score_bias",
                "ltf_score_entry",
                "htf_ltf_score_alignment",
                "htf_trend_strength",
                "htf_path_efficiency",
                "ltf_pullback_quality",
            ],
            "htf_ltf",
        )
        print("✅ HTF/LTF 特征值范围测试通过")

    def test_me_value_ranges(self, sample_data):
        """测试 MomentumExpansion 特征值范围"""
        df, orderflow = sample_data
        result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )
        self._test_value_ranges(
            result,
            [
                "me_score_expansion",
                "me_score_acceleration",
                "me_score_orderflow",
                "me_score_total",
                "me_atr_expansion",
                "me_bb_width_expansion",
            ],
            "me",
        )
        print("✅ MomentumExpansion 特征值范围测试通过")

    def test_fbf_value_ranges(self, sample_data):
        """测试 FailedBreakoutFade 特征值范围"""
        df, orderflow = sample_data
        result = compute_failed_breakout_fade_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open_=df["open"],
            volume=df["volume"],
            atr=df["atr"],
        )
        self._test_value_ranges(
            result,
            [
                "fbf_score_false_breakout",
                "fbf_score_rejection",
                "fbf_score_fade",
                "fbf_score_total",
                "fbf_wick_exhaustion",
                "fbf_reversal_momentum",
            ],
            "fbf",
        )
        print("✅ FailedBreakoutFade 特征值范围测试通过")

    def test_lsr_value_ranges(self, sample_data):
        """测试 LiquiditySweepRejection 特征值范围"""
        df, orderflow = sample_data
        result = compute_liquidity_sweep_rejection_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            open_=df["open"],
            volume=df["volume"],
            atr=df["atr"],
        )
        self._test_value_ranges(
            result,
            [
                "lsr_score_sweep",
                "lsr_score_rejection",
                "lsr_score_reversal",
                "lsr_score_total",
                "lsr_sweep_depth",
                "lsr_wick_rejection",
            ],
            "lsr",
        )
        print("✅ LiquiditySweepRejection 特征值范围测试通过")

    def test_aer_value_ranges(self, sample_data):
        """测试 AuctionExhaustionReversal 特征值范围"""
        df, orderflow = sample_data
        result = compute_auction_exhaustion_reversal_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )
        self._test_value_ranges(
            result,
            [
                "aer_score_climax",
                "aer_score_exhaustion",
                "aer_score_reversal",
                "aer_score_total",
                "aer_vol_climax",
                "aer_atr_climax",
            ],
            "aer",
        )
        print("✅ AuctionExhaustionReversal 特征值范围测试通过")

    def test_direction_features(self, sample_data):
        """测试方向特征在预期范围"""
        df, _ = sample_data

        # HTF/LTF 方向
        htf_result = compute_htf_ltf_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )
        if "htf_direction" in htf_result.columns:
            direction_vals = htf_result["htf_direction"].dropna().unique()
            assert all(
                d in [-1, 0, 1] for d in direction_vals
            ), f"HTF 方向值超出范围: {direction_vals}"

        # ME 方向
        me_result = compute_momentum_expansion_soft_phase_from_series(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            volume=df["volume"],
            atr=df["atr"],
        )
        if "me_direction" in me_result.columns:
            direction_vals = me_result["me_direction"].dropna().unique()
            assert all(
                d in [-1, 0, 1] for d in direction_vals
            ), f"ME 方向值超出范围: {direction_vals}"

        print("✅ 方向特征范围测试通过")


# =============================================================================
# 运行测试
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
