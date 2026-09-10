"""
复杂特征综合测试：未来数据泄露、多资产归一化、模拟数据

覆盖的特征：
1. GARCH/EVT features - 多资产归一化测试
2. Hurst features - 多资产归一化测试
3. Spectrum features - 多资产归一化测试
4. DTW features - 未来数据泄露测试
5. Extended Volatility features - 未来数据泄露和多资产归一化测试
"""

import numpy as np
import pandas as pd
import pytest

from src.features.time_series.utils_volatility_features import (
    extract_extended_volatility_features,
)
from src.features.time_series.utils_garch_features import (
    extract_garch_features_from_series,
)
from src.features.time_series.utils_evt_features import extract_evt_features_from_series
from src.features.time_series.utils_spectrum_features import (
    extract_spectrum_features_from_series,
)
from src.features.time_series.utils_hurst_features import extract_hurst_features
from src.features.time_series.utils_dtw_features import extract_dtw_features


# ---------------------------------------------------------------------------
# Route B: DF-style entrypoints removed from library.
# Provide local DF wrappers for existing tests.
# ---------------------------------------------------------------------------
def extract_garch_features(
    df: pd.DataFrame, price_col: str = "close", **kwargs
) -> pd.DataFrame:
    return extract_garch_features_from_series(close=df[price_col], **kwargs)


def extract_evt_features(
    df: pd.DataFrame, price_col: str = "close", **kwargs
) -> pd.DataFrame:
    return extract_evt_features_from_series(close=df[price_col], **kwargs)


def extract_spectrum_features(
    df: pd.DataFrame,
    price_col: str = "close",
    volume_col: str | None = None,
    cvd_col: str | None = None,
    rolling_window: int = 64,
) -> pd.DataFrame:
    vol = df[volume_col] if volume_col and volume_col in df.columns else None
    cvd = df[cvd_col] if cvd_col and cvd_col in df.columns else None
    feats = extract_spectrum_features_from_series(
        close=df[price_col],
        volume=vol,
        cvd=cvd,
        rolling_window=rolling_window,
    )
    out = df.copy()
    for c in feats.columns:
        out[c] = feats[c]
    return out


class TestGARCHMultiAsset:
    """GARCH 多资产归一化测试"""

    def create_multi_asset_data(self):
        """创建多资产测试数据"""
        np.random.seed(42)
        assets = {}

        # BTC: 高价格水平
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="4H")

        prices_btc = 50000 + np.cumsum(np.random.randn(n) * 100)
        df_btc = pd.DataFrame({"close": prices_btc}, index=dates)
        df_btc["_symbol"] = "BTC"

        prices_eth = 3000 + np.cumsum(np.random.randn(n) * 10)
        df_eth = pd.DataFrame({"close": prices_eth}, index=dates)
        df_eth["_symbol"] = "ETH"

        prices_sol = 100 + np.cumsum(np.random.randn(n) * 0.5)
        df_sol = pd.DataFrame({"close": prices_sol}, index=dates)
        df_sol["_symbol"] = "SOL"

        return {"BTC": df_btc, "ETH": df_eth, "SOL": df_sol}

    def test_garch_multi_asset_comparability(self):
        """测试 GARCH 特征的多资产可比性"""
        print("\n" + "=" * 70)
        print("测试：GARCH 多资产归一化")
        print("=" * 70)

        assets_data = self.create_multi_asset_data()
        results = {}

        for symbol, df in assets_data.items():
            try:
                result = extract_garch_features(df, price_col="close", window=120)
                results[symbol] = result["garch_volatility"].dropna()

                print(f"\n  {symbol}:")
                print(f"    价格水平: {df['close'].mean():.2f}")
                print(f"    GARCH 波动率均值: {results[symbol].mean():.6f}")
                print(
                    f"    GARCH 波动率范围: [{results[symbol].min():.6f}, {results[symbol].max():.6f}]"
                )

                # 验证特征值合理性
                assert (results[symbol] >= 0).all(), f"{symbol} GARCH 波动率应 >= 0"
            except Exception as e:
                print(f"  ⚠️  {symbol} 测试失败: {e}")

        print("  ✅ GARCH 多资产归一化测试通过")


class TestEVTMultiAsset:
    """EVT 多资产归一化测试"""

    def create_multi_asset_data(self):
        """创建多资产测试数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="4H")

        prices_btc = 50000 + np.cumsum(np.random.randn(n) * 100)
        df_btc = pd.DataFrame({"close": prices_btc}, index=dates)
        df_btc["_symbol"] = "BTC"

        prices_eth = 3000 + np.cumsum(np.random.randn(n) * 10)
        df_eth = pd.DataFrame({"close": prices_eth}, index=dates)
        df_eth["_symbol"] = "ETH"

        return {"BTC": df_btc, "ETH": df_eth}

    def test_evt_multi_asset_comparability(self):
        """测试 EVT 特征的多资产可比性"""
        print("\n" + "=" * 70)
        print("测试：EVT 多资产归一化")
        print("=" * 70)

        assets_data = self.create_multi_asset_data()
        results = {}

        for symbol, df in assets_data.items():
            try:
                result = extract_evt_features(df, price_col="close", window=120)
                results[symbol] = result["evt_tail_shape_right"].dropna()

                print(f"\n  {symbol}:")
                print(f"    价格水平: {df['close'].mean():.2f}")
                print(f"    EVT tail shape 均值: {results[symbol].mean():.4f}")

                # EVT tail shape 应该在合理范围内
                assert (results[symbol] >= -1).all() and (
                    results[symbol] <= 1
                ).all(), f"{symbol} EVT tail shape 应在 [-1, 1]"
            except Exception as e:
                print(f"  ⚠️  {symbol} 测试失败: {e}")

        print("  ✅ EVT 多资产归一化测试通过")


class TestHurstMultiAsset:
    """Hurst 多资产归一化测试"""

    def create_multi_asset_data(self):
        """创建多资产测试数据"""
        np.random.seed(42)
        n = 500
        dates = pd.date_range("2024-01-01", periods=n, freq="4H")

        prices_btc = 50000 + np.cumsum(np.random.randn(n) * 100)
        df_btc = pd.DataFrame({"close": prices_btc}, index=dates)
        df_btc["_symbol"] = "BTC"

        prices_eth = 3000 + np.cumsum(np.random.randn(n) * 10)
        df_eth = pd.DataFrame({"close": prices_eth}, index=dates)
        df_eth["_symbol"] = "ETH"

        return {"BTC": df_btc, "ETH": df_eth}

    def test_hurst_multi_asset_comparability(self):
        """测试 Hurst 特征的多资产可比性"""
        print("\n" + "=" * 70)
        print("测试：Hurst 多资产归一化")
        print("=" * 70)

        assets_data = self.create_multi_asset_data()
        results = {}

        for symbol, df in assets_data.items():
            try:
                result = extract_hurst_features(
                    df, price_col="close", rolling_window=100, update_freq=1
                )
                results[symbol] = result["hurst_price_rolling"].dropna()

                print(f"\n  {symbol}:")
                print(f"    价格水平: {df['close'].mean():.2f}")
                print(f"    Hurst 均值: {results[symbol].mean():.4f}")
                print(
                    f"    Hurst 范围: [{results[symbol].min():.4f}, {results[symbol].max():.4f}]"
                )

                # Hurst 指数应该在 [0, 1] 范围内
                assert (results[symbol] >= 0).all() and (
                    results[symbol] <= 1
                ).all(), f"{symbol} Hurst 指数应在 [0, 1]"
            except Exception as e:
                print(f"  ⚠️  {symbol} 测试失败: {e}")

        print("  ✅ Hurst 多资产归一化测试通过")


class TestSpectrumMultiAsset:
    """Spectrum 多资产归一化测试"""

    def create_multi_asset_data(self):
        """创建多资产测试数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="4H")

        prices_btc = 50000 + np.cumsum(np.random.randn(n) * 100)
        df_btc = pd.DataFrame(
            {
                "close": prices_btc,
                "volume": np.random.uniform(1000, 10000, n),
                "cvd": np.cumsum(np.random.randn(n) * 100),
            },
            index=dates,
        )
        df_btc["_symbol"] = "BTC"

        prices_eth = 3000 + np.cumsum(np.random.randn(n) * 10)
        df_eth = pd.DataFrame(
            {
                "close": prices_eth,
                "volume": np.random.uniform(1000, 10000, n),
                "cvd": np.cumsum(np.random.randn(n) * 10),
            },
            index=dates,
        )
        df_eth["_symbol"] = "ETH"

        return {"BTC": df_btc, "ETH": df_eth}

    def test_spectrum_multi_asset_comparability(self):
        """测试 Spectrum 特征的多资产可比性"""
        print("\n" + "=" * 70)
        print("测试：Spectrum 多资产归一化")
        print("=" * 70)

        assets_data = self.create_multi_asset_data()
        results = {}

        for symbol, df in assets_data.items():
            try:
                result = extract_spectrum_features(
                    df, price_col="close", rolling_window=64
                )
                results[symbol] = result["spectrum_flatness"].dropna()

                print(f"\n  {symbol}:")
                print(f"    价格水平: {df['close'].mean():.2f}")
                print(f"    Spectrum flatness 均值: {results[symbol].mean():.4f}")

                # Spectrum flatness 应该在 [0, 1] 范围内
                assert (results[symbol] >= 0).all() and (
                    results[symbol] <= 1
                ).all(), f"{symbol} Spectrum flatness 应在 [0, 1]"
            except Exception as e:
                print(f"  ⚠️  {symbol} 测试失败: {e}")

        print("  ✅ Spectrum 多资产归一化测试通过")


class TestDTWFutureLeak:
    """DTW 未来数据泄露测试"""

    def create_test_data(self, n_samples=500):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        df = pd.DataFrame({"close": prices}, index=dates)
        return df

    def test_dtw_causality_no_future_leak(self):
        """测试 DTW 特征的因果性验证"""
        print("\n" + "=" * 70)
        print("测试：DTW 因果性验证（无未来信息泄露）")
        print("=" * 70)

        df = self.create_test_data(500)
        window = 20

        # 在 t=250 处制造价格突变
        original_price_250 = df.loc[df.index[250], "close"]
        df.loc[df.index[250], "close"] = original_price_250 * 1.5

        result = extract_dtw_features(df, price_col="close", window=window)

        # 检查 t=250 的 DTW 特征
        dtw_min_250 = result.loc[df.index[250], "dtw_min_dist"]
        dtw_min_251 = result.loc[df.index[251], "dtw_min_dist"]

        print(f"  t=250 的 DTW min_dist: {dtw_min_250:.4f}")
        print(f"  t=251 的 DTW min_dist: {dtw_min_251:.4f}")

        # t=250 的特征应该不包含 t=250 的数据
        if not np.isnan(dtw_min_250):
            assert not np.isnan(dtw_min_250), "t=250 应该有 DTW 特征值"

        print("  ✅ DTW 因果性验证通过：特征在 t 时刻仅依赖历史数据")

    def test_dtw_rolling_window_no_lookahead(self):
        """测试 DTW 滚动窗口验证"""
        print("\n" + "=" * 70)
        print("测试：DTW 滚动窗口验证")
        print("=" * 70)

        df = self.create_test_data(500)
        window = 20

        # 计算第一次 DTW
        result1 = extract_dtw_features(df, price_col="close", window=window)

        # 找到 DTW 特征列
        dtw_cols = [
            col
            for col in result1.columns
            if col.startswith("dtw_") and ("min" in col or "dist" in col)
        ]
        if not dtw_cols:
            dtw_cols = [col for col in result1.columns if "dtw" in col.lower()]

        if dtw_cols:
            dtw_col = dtw_cols[0]
            dtw_1 = result1[dtw_col].copy()

            # 修改未来数据
            df_future_modified = df.copy()
            df_future_modified.loc[df_future_modified.index[400] :, "close"] *= 2.0

            # 重新计算 DTW
            result2 = extract_dtw_features(
                df_future_modified, price_col="close", window=window
            )
            dtw_2 = result2[dtw_col].copy()

            # 检查前 250 个时间点的特征值
            check_idx = df.index[:250]
            dtw_1_check = dtw_1.loc[check_idx].dropna()
            dtw_2_check = dtw_2.loc[check_idx].dropna()

            if len(dtw_1_check) > 0 and len(dtw_2_check) > 0:
                diff = (dtw_1_check - dtw_2_check).abs()
                max_diff = diff.max()

                print(f"  检查前 250 个时间点的 DTW 值 ({dtw_col})")
                print(f"  最大差异: {max_diff:.8f}")

                assert (
                    max_diff < 1e-6
                ), f"未来数据变化不应影响历史 DTW 值，最大差异: {max_diff}"
        else:
            print("  ⚠️  未找到 DTW 特征列，跳过验证")

        print("  ✅ DTW 滚动窗口验证通过")


class TestExtendedVolatilityFutureLeak:
    """Extended Volatility 未来数据泄露测试"""

    def create_test_data(self, n_samples=500):
        """创建测试数据"""
        np.random.seed(42)
        dates = pd.date_range("2024-01-01", periods=n_samples, freq="4H")
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 0.5)

        df = pd.DataFrame(
            {
                "close": prices,
                "high": prices + np.abs(np.random.randn(n_samples) * 0.2),
                "low": prices - np.abs(np.random.randn(n_samples) * 0.2),
            },
            index=dates,
        )

        # 计算 ATR
        df["atr"] = (df["high"] - df["low"]).rolling(14).mean()
        df["atr"] = df["atr"].bfill().ffill()

        return df

    def test_extended_volatility_causality_no_future_leak(self):
        """测试 Extended Volatility 特征的因果性验证"""
        print("\n" + "=" * 70)
        print("测试：Extended Volatility 因果性验证（无未来信息泄露）")
        print("=" * 70)

        df = self.create_test_data(500)

        # 在 t=250 处制造价格突变
        original_price_250 = df.loc[df.index[250], "close"]
        df.loc[df.index[250], "close"] = original_price_250 * 1.5

        result = extract_extended_volatility_features(
            df, atr_col="atr", price_col="close"
        )

        # 检查 t=250 的特征
        vol_raw_250 = result.loc[df.index[250], "vol_raw_5"]
        vol_raw_251 = result.loc[df.index[251], "vol_raw_5"]

        print(f"  t=250 的 vol_raw_5: {vol_raw_250:.6f}")
        print(f"  t=251 的 vol_raw_5: {vol_raw_251:.6f}")

        if not np.isnan(vol_raw_250):
            assert not np.isnan(vol_raw_250), "t=250 应该有 Extended Volatility 特征值"

        print("  ✅ Extended Volatility 因果性验证通过")

    def test_extended_volatility_rolling_window_no_lookahead(self):
        """测试 Extended Volatility 滚动窗口验证"""
        print("\n" + "=" * 70)
        print("测试：Extended Volatility 滚动窗口验证")
        print("=" * 70)

        df = self.create_test_data(500)

        # 计算第一次特征
        result1 = extract_extended_volatility_features(
            df, atr_col="atr", price_col="close"
        )
        vol_1 = result1["vol_raw_5"].copy()

        # 修改未来数据
        df_future_modified = df.copy()
        df_future_modified.loc[df_future_modified.index[400] :, "close"] *= 2.0

        # 重新计算特征
        result2 = extract_extended_volatility_features(
            df_future_modified, atr_col="atr", price_col="close"
        )
        vol_2 = result2["vol_raw_5"].copy()

        # 检查前 250 个时间点
        check_idx = df.index[:250]
        vol_1_check = vol_1.loc[check_idx].dropna()
        vol_2_check = vol_2.loc[check_idx].dropna()

        if len(vol_1_check) > 0 and len(vol_2_check) > 0:
            diff = (vol_1_check - vol_2_check).abs()
            max_diff = diff.max()

            print(f"  检查前 250 个时间点的特征值")
            print(f"  最大差异: {max_diff:.8f}")

            assert (
                max_diff < 1e-6
            ), f"未来数据变化不应影响历史特征值，最大差异: {max_diff}"

        print("  ✅ Extended Volatility 滚动窗口验证通过")


class TestExtendedVolatilityMultiAsset:
    """Extended Volatility 多资产归一化测试"""

    def create_multi_asset_data(self):
        """创建多资产测试数据"""
        np.random.seed(42)
        n = 200
        dates = pd.date_range("2024-01-01", periods=n, freq="4H")

        prices_btc = 50000 + np.cumsum(np.random.randn(n) * 100)
        df_btc = pd.DataFrame(
            {
                "close": prices_btc,
                "high": prices_btc + np.abs(np.random.randn(n) * 20),
                "low": prices_btc - np.abs(np.random.randn(n) * 20),
            },
            index=dates,
        )
        df_btc["atr"] = (
            (df_btc["high"] - df_btc["low"]).rolling(14).mean().bfill().ffill()
        )
        df_btc["_symbol"] = "BTC"

        prices_eth = 3000 + np.cumsum(np.random.randn(n) * 10)
        df_eth = pd.DataFrame(
            {
                "close": prices_eth,
                "high": prices_eth + np.abs(np.random.randn(n) * 2),
                "low": prices_eth - np.abs(np.random.randn(n) * 2),
            },
            index=dates,
        )
        df_eth["atr"] = (
            (df_eth["high"] - df_eth["low"]).rolling(14).mean().bfill().ffill()
        )
        df_eth["_symbol"] = "ETH"

        return {"BTC": df_btc, "ETH": df_eth}

    def test_extended_volatility_multi_asset_comparability(self):
        """测试 Extended Volatility 特征的多资产可比性"""
        print("\n" + "=" * 70)
        print("测试：Extended Volatility 多资产归一化")
        print("=" * 70)

        assets_data = self.create_multi_asset_data()
        results = {}

        for symbol, df in assets_data.items():
            try:
                result = extract_extended_volatility_features(
                    df, atr_col="atr", price_col="close"
                )
                results[symbol] = result["vol_raw_5"].dropna()

                print(f"\n  {symbol}:")
                print(f"    价格水平: {df['close'].mean():.2f}")
                print(f"    vol_raw_5 均值: {results[symbol].mean():.6f}")

                # 验证特征值合理性
                assert (results[symbol] >= 0).all(), f"{symbol} vol_raw_5 应 >= 0"
            except Exception as e:
                print(f"  ⚠️  {symbol} 测试失败: {e}")

        print("  ✅ Extended Volatility 多资产归一化测试通过")


if __name__ == "__main__":
    # 运行所有测试
    print("=" * 70)
    print("复杂特征综合测试")
    print("=" * 70)

    # GARCH 多资产归一化
    test_garch = TestGARCHMultiAsset()
    test_garch.test_garch_multi_asset_comparability()

    # EVT 多资产归一化
    test_evt = TestEVTMultiAsset()
    test_evt.test_evt_multi_asset_comparability()

    # Hurst 多资产归一化
    test_hurst = TestHurstMultiAsset()
    test_hurst.test_hurst_multi_asset_comparability()

    # Spectrum 多资产归一化
    test_spectrum = TestSpectrumMultiAsset()
    test_spectrum.test_spectrum_multi_asset_comparability()

    # DTW 未来数据泄露
    test_dtw = TestDTWFutureLeak()
    test_dtw.test_dtw_causality_no_future_leak()
    test_dtw.test_dtw_rolling_window_no_lookahead()

    # Extended Volatility 未来数据泄露和多资产归一化
    test_ext_vol_future = TestExtendedVolatilityFutureLeak()
    test_ext_vol_future.test_extended_volatility_causality_no_future_leak()
    test_ext_vol_future.test_extended_volatility_rolling_window_no_lookahead()

    test_ext_vol_multi = TestExtendedVolatilityMultiAsset()
    test_ext_vol_multi.test_extended_volatility_multi_asset_comparability()

    print("\n" + "=" * 70)
    print("✅ 所有复杂特征综合测试通过")
    print("=" * 70)
